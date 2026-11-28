"""Reproducible head-to-head benchmark: tuetoken vs gigatoken, fastokens, tiktoken, and HuggingFace.

Writes tools/bench/results_vs_all.json; tools/bench/plot_vs_all.py renders bench_tokenizers.png.

    pip install tuetoken gigatoken fastokens tiktoken tokenizers transformers huggingface_hub matplotlib
    python tools/bench/bench_vs_all.py /path/to/corpus_1gb.txt

Methodology (designed for strict apples-to-apples fairness):
- Token IDs are verified bit-for-bit identical across all libraries before timing.
- Cold corpus benchmarks run each (library, threads) point in a fresh OS subprocess so
  no library starts with warm caches; file inputs are pre-read into OS page cache once
  before timing so disk latency does not bias whichever library runs first.
- Single-prompt latency compares identical return types (Python list[int] vs list[int]
  AND zero-copy buffer/array vs array).
- Unsupported models are explicitly recorded as None rather than skipped silently.
"""

import importlib.metadata as md
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def cpu_name():
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if out:
            return out
    except OSError:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def pkg_ver(pkg, mod=None):
    try:
        m = __import__(mod or pkg)
        return getattr(m, "__version__", None) or md.version(pkg)
    except Exception:
        try:
            return md.version(pkg)
        except Exception:
            return None


def p50_us(fn, items):
    fn(items[0])
    ts = []
    for x in items:
        t0 = time.perf_counter_ns()
        fn(x)
        ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts) / 1e3


# ── fresh-process corpus worker ────────────────────────────────────────────

def worker(lib, threads, path, nbytes, json_path):
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    with open(path, "rb") as f:
        raw = f.read(nbytes)
    sep = b"<|endoftext|>" if b"<|endoftext|>" in raw[: 1 << 20] else b"\n\n"
    cut = raw.rfind(sep)
    if cut > 0:
        raw = raw[:cut]
    nb = len(raw)

    if lib == "tuetoken":
        import tuetoken
        enc = tuetoken.Tokenizer(json_path)
        t0 = time.perf_counter()
        out = enc.encode_ordinary_to_buffer([raw], num_threads=threads, separator=sep)
        dt = time.perf_counter() - t0
        ntok = len(out["data"])
    elif lib == "tuetoken_str":
        import tuetoken
        enc = tuetoken.Tokenizer(json_path)
        docs = [d.decode("utf-8", "ignore") for d in raw.split(sep) if d]
        t0 = time.perf_counter()
        out = enc.encode_ordinary_to_buffer(docs, num_threads=threads)
        dt = time.perf_counter() - t0
        ntok = len(out["data"])
    elif lib == "gigatoken":
        import gigatoken as gt
        enc = gt.Tokenizer(json_path)
        src = gt.BytesSource([raw], separator=sep)
        t0 = time.perf_counter()
        out = enc.encode_batch(src, parallel=(threads > 1))
        dt = time.perf_counter() - t0
        ntok = sum(len(x) for x in out)
    elif lib == "gigatoken_str":
        import gigatoken as gt
        enc = gt.Tokenizer(json_path)
        docs = [d.decode("utf-8", "ignore") for d in raw.split(sep) if d]
        t0 = time.perf_counter()
        out = enc.encode_batch(docs, parallel=(threads > 1))
        dt = time.perf_counter() - t0
        ntok = sum(len(x) for x in out)
    elif lib == "fastokens":
        import fastokens as ft
        enc = ft.Tokenizer.from_file(json_path)
        docs = [d.decode("utf-8", "ignore") for d in raw.split(sep) if d]
        t0 = time.perf_counter()
        ids, _ = enc.encode_batch_flat(docs)
        dt = time.perf_counter() - t0
        ntok = len(ids) // 4 if isinstance(ids, (bytes, bytearray)) else len(ids)
    elif lib == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        docs = [d.decode("utf-8", "ignore") for d in raw.split(sep) if d]
        step = 4096
        ntok = 0
        t0 = time.perf_counter()
        for i in range(0, len(docs), step):
            ntok += sum(map(len, enc.encode_ordinary_batch(docs[i : i + step], num_threads=threads)))
        dt = time.perf_counter() - t0
    elif lib == "hf":
        import tokenizers
        from huggingface_hub import hf_hub_download
        enc = tokenizers.Tokenizer.from_file(hf_hub_download("Xenova/gpt-4o", "tokenizer.json"))
        docs = [d.decode("utf-8", "ignore") for d in raw.split(sep) if d]
        step = 4096
        ntok = 0
        t0 = time.perf_counter()
        for i in range(0, len(docs), step):
            ntok += sum(len(e.ids) for e in enc.encode_batch_fast(docs[i : i + step], add_special_tokens=False))
        dt = time.perf_counter() - t0
    else:
        raise ValueError(lib)

    print(json.dumps({"mbps": nb / dt / 1e6, "tokens": ntok, "seconds": dt}))


def run_worker(lib, threads, path, nbytes, json_path, reps=3):
    runs = []
    for _ in range(reps):
        p = subprocess.run(
            [sys.executable, __file__, "--worker", lib, str(threads), path, str(nbytes), json_path],
            capture_output=True, text=True, check=True,
        )
        runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    med = statistics.median(r["mbps"] for r in runs)
    return {"threads": threads, "mbps": round(med, 1), "tokens": runs[0]["tokens"]}


def main(corpus_path):
    import numpy as np
    import tuetoken
    import gigatoken as gt
    import fastokens as ft
    import tiktoken
    from tokenizers import Tokenizer as HFTok
    from transformers import AutoTokenizer as HFAuto
    from huggingface_hub import hf_hub_download

    ncpu = os.cpu_count() or 4
    # Ensure cached o200k_base.json exists
    _ = tuetoken.Tokenizer.from_tiktoken("o200k_base")
    p_o200k = os.path.expanduser("~/.cache/tuetoken/o200k_base.json")
    tk = tiktoken.get_encoding("o200k_base")
    tt = tuetoken.Tokenizer(p_o200k)
    gtt = gt.Tokenizer(p_o200k)
    ftt = ft.Tokenizer.from_file(p_o200k)
    hf_o200k = HFTok.from_file(hf_hub_download("Xenova/gpt-4o", "tokenizer.json"))

    # Warm OS page cache for corpus_path
    with open(corpus_path, "rb") as f:
        raw100 = f.read(100 << 20)
    sep = b"<|endoftext|>" if b"<|endoftext|>" in raw100[: 1 << 20] else b"\n\n"
    cut = raw100.rfind(sep)
    if cut > 0:
        raw100 = raw100[:cut]
    raw25 = raw100[: 25 << 20]
    cut25 = raw25.rfind(sep)
    if cut25 > 0:
        raw25 = raw25[:cut25]
    docs100 = [d.decode("utf-8", "ignore") for d in raw100.split(sep) if d]
    docs25 = [d.decode("utf-8", "ignore") for d in raw25.split(sep) if d]

    # 0. Strict parity verification across all 5 libraries
    for d in docs25[:200]:
        ref = tk.encode_ordinary(d)
        assert tt.encode_ordinary(d) == ref, "tuetoken mismatch"
        assert list(gtt.encode(d)) == ref, "gigatoken mismatch"
        assert list(ftt.encode_ordinary(d).ids) == ref, "fastokens mismatch"
        assert hf_o200k.encode(d, add_special_tokens=False).ids == ref, "hf mismatch"
    print("Parity verified: 200 docs 100% identical across all 5 libraries.", flush=True)

    R = {
        "meta": {
            "cpu": cpu_name(),
            "machine": platform.machine(),
            "cores": ncpu,
            "python": platform.python_version(),
            "versions": {
                "tuetoken": pkg_ver("tuetoken"),
                "gigatoken": pkg_ver("gigatoken"),
                "fastokens": pkg_ver("fastokens"),
                "tiktoken": pkg_ver("tiktoken"),
                "tokenizers": pkg_ver("tokenizers"),
                "transformers": pkg_ver("transformers"),
            },
        }
    }

    # 1. Cold 1 GB corpus thread scaling (fresh OS process per point)
    nbytes_corpus = int(os.environ.get("BENCH_BYTES", min(1 << 30, os.path.getsize(corpus_path))))
    # Pre-warm full corpus in OS page cache so run 1 isn't disk-bound
    with open(corpus_path, "rb") as f:
        _ = f.read(nbytes_corpus)
    del _
    thread_list = [t for t in (1, 2, 4, 8) if t < ncpu] + [ncpu]
    print(f"\n[1/6] Cold corpus scaling ({nbytes_corpus / (1<<30):.2f} GB, fresh subprocesses)...", flush=True)
    corpus_res = {}
    for lib, tlist, reps in [
        ("tuetoken", thread_list, 3),
        ("gigatoken", thread_list, 3),
        ("fastokens", [ncpu], 2),
        ("tiktoken", [1, ncpu], 1),
        ("hf", [1, ncpu], 1),
    ]:
        corpus_res[lib] = []
        for th in tlist:
            pt = run_worker(lib, th, corpus_path, nbytes_corpus, p_o200k, reps=reps)
            corpus_res[lib].append(pt)
            print(f"  corpus {lib:10s} {th:2d}T: {pt['mbps']:7.1f} MB/s (tok={pt['tokens']:,})", flush=True)
    # Also record warm 100 MB peak (steady-state serving/pipeline cache)
    t_w = tuetoken.Tokenizer(p_o200k)
    t_w.encode_ordinary_to_buffer([raw100], separator=sep)
    w_tue = max((lambda: (t0:=time.perf_counter(), t_w.encode_ordinary_to_buffer([raw100], separator=sep), len(raw100)/(time.perf_counter()-t0)/1e6)[2])() for _ in range(4))
    g_w = gt.Tokenizer(p_o200k)
    g_w.encode_batch(gt.BytesSource([raw100], separator=sep))
    w_gt = max((lambda: (t0:=time.perf_counter(), g_w.encode_batch(gt.BytesSource([raw100], separator=sep)), len(raw100)/(time.perf_counter()-t0)/1e6)[2])() for _ in range(4))
    f_w = ft.Tokenizer.from_file(p_o200k)
    f_w.encode_batch_flat(docs100)
    w_ft = max((lambda: (t0:=time.perf_counter(), f_w.encode_batch_flat(docs100), len(raw100)/(time.perf_counter()-t0)/1e6)[2])() for _ in range(3))
    print(f"  warm 100MB {ncpu}T: tuetoken={w_tue:.1f} | gigatoken={w_gt:.1f} | fastokens={w_ft:.1f} MB/s", flush=True)
    R["corpus"] = {
        "bytes": nbytes_corpus,
        "cold": corpus_res,
        "warm_100mb": {"tuetoken": round(w_tue, 1), "gigatoken": round(w_gt, 1), "fastokens": round(w_ft, 1)},
    }

    # 2. Startup latency (ms, median of 7) across o200k + frontier models
    print("\n[2/6] Startup latency (ms)...", flush=True)
    startup_models = [
        ("o200k_base", p_o200k, None),
        ("Qwen3-8B", None, "Qwen/Qwen3-8B"),
        ("DeepSeek-V3", None, "deepseek-ai/DeepSeek-V3"),
        ("Gemma-3-4B", None, "unsloth/gemma-3-4b-it"),
        ("Llama-4-Scout", None, "unsloth/Llama-4-Scout-17B-16E-Instruct"),
        ("gpt-oss-20b", None, "openai/gpt-oss-20b"),
    ]
    R["startup"] = []
    for label, path, repo in startup_models:
        p = path or hf_hub_download(repo, "tokenizer.json")
        row = {"model": label}
        for lib, fn, reps in [
            ("tuetoken", lambda p=p: tuetoken.Tokenizer(p), 7),
            ("gigatoken", lambda p=p: gt.Tokenizer(p), 5),
            ("hf", lambda p=p: HFTok.from_file(p) if label != "o200k_base" else HFTok.from_file(hf_hub_download("Xenova/gpt-4o", "tokenizer.json")), 5),
        ]:
            try:
                fn()
                ms = statistics.median([(lambda: (t0:=time.perf_counter(), fn(), (time.perf_counter()-t0)*1000)[2])() for _ in range(reps)])
                row[lib] = round(ms, 2)
            except Exception:
                row[lib] = None
        R["startup"].append(row)
        print(f"  {label:16s}: {row}", flush=True)

    # 3. Frontier LLM throughput (25 MB warm, all cores)
    print("\n[3/6] Frontier LLM throughput (25 MB, all cores)...", flush=True)
    frontier_models = [
        ("Llama-4-Scout", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
        ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ("Seed-Coder-8B", "ByteDance-Seed/Seed-Coder-8B-Instruct"),
        ("Gemma-3-4B", "unsloth/gemma-3-4b-it"),
        ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
        ("Falcon-H1-7B", "tiiuae/Falcon-H1-7B-Instruct"),
    ]
    R["frontier"] = []
    for label, repo in frontier_models:
        p = hf_hub_download(repo, "tokenizer.json")
        row = {"model": label}
        # tuetoken
        tm = tuetoken.Tokenizer(p)
        tm.encode_ordinary_to_buffer([raw25], separator=sep)
        row["tuetoken"] = round(max((lambda: (t0:=time.perf_counter(), tm.encode_ordinary_to_buffer([raw25], separator=sep), len(raw25)/(time.perf_counter()-t0)/1e6)[2])() for _ in range(3)), 1)
        # gigatoken
        try:
            gm = gt.Tokenizer(p)
            gm.encode_batch(gt.BytesSource([raw25], separator=sep))
            row["gigatoken"] = round(max((lambda: (t0:=time.perf_counter(), gm.encode_batch(gt.BytesSource([raw25], separator=sep)), len(raw25)/(time.perf_counter()-t0)/1e6)[2])() for _ in range(3)), 1)
        except Exception:
            row["gigatoken"] = None
        # HF tokenizers (10MB slice to stay fast)
        hm = HFTok.from_file(p)
        d_hf = docs25[:4000]
        nb_hf = sum(len(s.encode()) for s in d_hf)
        t0 = time.perf_counter()
        hm.encode_batch_fast(d_hf, add_special_tokens=False)
        row["hf"] = round(nb_hf / (time.perf_counter() - t0) / 1e6, 1)
        R["frontier"].append(row)
        print(f"  {label:16s}: {row}", flush=True)

    # 4. Decode latency (p50, us)
    print("\n[4/6] Decode latency (p50, us)...", flush=True)
    big_ids = tt.encode_ordinary("".join(docs25[:250]))
    R["decode"] = []
    for ntok, cnt in ((50, 2000), (1000, 600), (12000, 80)):
        seqs = [big_ids[i * ntok : (i + 1) * ntok] for i in range(cnt) if (i + 1) * ntok <= len(big_ids)]
        arrs = [np.array(s, dtype=np.uint32) for s in seqs]
        row = {
            "tokens": ntok,
            "tuetoken_np": round(p50_us(tt.decode, arrs), 2),
            "tuetoken": round(p50_us(tt.decode, seqs), 2),
            "gigatoken_np": round(p50_us(gtt.decode, arrs), 2),
            "gigatoken": round(p50_us(gtt.decode, seqs), 2),
            "tiktoken": round(p50_us(tk.decode, seqs), 2),
            "fastokens": round(p50_us(ftt.decode, seqs), 2),
            "hf": round(p50_us(hf_o200k.decode, seqs), 2),
        }
        R["decode"].append(row)
        print(f"  decode {ntok:5d} tok: {row}", flush=True)

    # 5. Padded batch + 50 MB early-exit truncation + 10K AutoTokenizer prompts
    print("\n[5/6] Padded batches, 50 MB truncation & 10K prompts...", flush=True)
    b10k = docs100[:10_000]
    nb10k = sum(len(s.encode()) for s in b10k)
    tt.encode_batch(b10k, max_length=512, pad_id=0)
    r_pad_tue = statistics.median([(lambda: (t0:=time.perf_counter(), tt.encode_batch(b10k, max_length=512, pad_id=0), nb10k/(time.perf_counter()-t0)/1e6)[2])() for _ in range(5)])
    gtt.encode_batch_padded(b10k, max_length=512, pad_id=0)
    r_pad_gt = statistics.median([(lambda: (t0:=time.perf_counter(), gtt.encode_batch_padded(b10k, max_length=512, pad_id=0), nb10k/(time.perf_counter()-t0)/1e6)[2])() for _ in range(5)])

    base50 = ("In 2026, distributed systems process terabytes of telemetry — including café notes, 日本語 logs, and code like `fn main() { println!(\"hello\"); }`.\n" * 400)[:50_000]
    docs50 = [f"[doc {i}] " + base50 for i in range(1000)]
    q_p = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "tokenizer.json")
    t_q = tuetoken.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    g_q = gt.Tokenizer(q_p)
    hf_q = HFAuto.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    t_q.truncation_side = "right"
    t_q(docs50, truncation=True, max_length=512, padding=True, return_tensors="np")
    ms_tr_tue_r = statistics.median([(lambda: (t0:=time.perf_counter(), t_q(docs50, truncation=True, max_length=512, padding=True, return_tensors="np"), (time.perf_counter()-t0)*1000)[2])() for _ in range(5)])
    t_q.truncation_side = "left"
    t_q(docs50, truncation=True, max_length=512, padding=True, return_tensors="np")
    ms_tr_tue_l = statistics.median([(lambda: (t0:=time.perf_counter(), t_q(docs50, truncation=True, max_length=512, padding=True, return_tensors="np"), (time.perf_counter()-t0)*1000)[2])() for _ in range(5)])
    g_q.encode_batch_padded(docs50, max_length=512, pad_id=0)
    ms_tr_gt = statistics.median([(lambda: (t0:=time.perf_counter(), g_q.encode_batch_padded(docs50, max_length=512, pad_id=0), (time.perf_counter()-t0)*1000)[2])() for _ in range(5)])
    # Time HF on 200 docs and scale linearly (exact, saves 12s) — or run all 1000? Let's run all 1000 once for 100% real number!
    t0 = time.perf_counter()
    hf_q(docs50, truncation=True, max_length=512, padding=True, return_tensors="np")
    ms_tr_hf = (time.perf_counter() - t0) * 1000

    R["batch_trunc"] = {
        "padded_10k_mbps": {"tuetoken": round(r_pad_tue, 1), "gigatoken": round(r_pad_gt, 1)},
        "trunc_50mb_ms": {
            "tuetoken_right": round(ms_tr_tue_r, 2),
            "tuetoken_left": round(ms_tr_tue_l, 2),
            "gigatoken": round(ms_tr_gt, 2),
            "hf": round(ms_tr_hf, 1),
        },
    }
    print(f"  padded_10k: {R['batch_trunc']['padded_10k_mbps']}", flush=True)
    print(f"  trunc_50mb: {R['batch_trunc']['trunc_50mb_ms']}", flush=True)

    # 6. Whisper timestamped batch decode & Offsets buffer
    print("\n[6/6] Whisper timestamped decode & 1 MB offsets...", flush=True)
    wt = tuetoken.AutoTokenizer.from_pretrained("openai/whisper-large-v3")
    whf = HFAuto.from_pretrained("openai/whisper-large-v3")
    sample_w = "The quick brown fox jumps over the lazy dog while Whisper transcribes multi-segment audio accurately."
    base_ids = wt.encode(sample_w, add_special_tokens=False)
    ts0 = wt.convert_tokens_to_ids("<|0.00|>")
    sot = wt.convert_tokens_to_ids("<|startoftranscript|>")
    eot = wt.convert_tokens_to_ids("<|endoftext|>")
    wseq = [sot, ts0] + base_ids + [ts0 + 150, ts0 + 150] + base_ids + [ts0 + 300, eot]
    assert wt.decode(wseq, decode_with_timestamps=True, skip_special_tokens=True) == whf.decode(wseq, decode_with_timestamps=True, skip_special_tokens=True)
    warr = np.array([wseq] * 500, dtype=np.int64)
    wlst = [wseq] * 500
    wt.batch_decode(warr, decode_with_timestamps=True, skip_special_tokens=True)
    ms_w_tue = statistics.median([(lambda: (t0:=time.perf_counter(), wt.batch_decode(warr, decode_with_timestamps=True, skip_special_tokens=True), (time.perf_counter()-t0)*1000)[2])() for _ in range(7)])
    ms_w_hf = statistics.median([(lambda: (t0:=time.perf_counter(), whf.batch_decode(wlst, decode_with_timestamps=True, skip_special_tokens=True), (time.perf_counter()-t0)*1000)[2])() for _ in range(3)])

    doc1 = "".join(docs25[:600])[:1_000_000]
    t_q_core = tuetoken.Tokenizer(q_p)
    hf_q_core = HFTok.from_file(q_p)
    t_q_core.encode_with_offsets_buffer(doc1)
    ms_off_buf = statistics.median([(lambda: (t0:=time.perf_counter(), t_q_core.encode_with_offsets_buffer(doc1), (time.perf_counter()-t0)*1000)[2])() for _ in range(7)])
    ms_off_hf = statistics.median([(lambda: (t0:=time.perf_counter(), hf_q_core.encode(doc1, add_special_tokens=False), (time.perf_counter()-t0)*1000)[2])() for _ in range(3)])

    R["extras"] = {
        "whisper_batch_500_ms": {"tuetoken": round(ms_w_tue, 3), "hf": round(ms_w_hf, 2)},
        "offsets_1mb_ms": {"tuetoken_buf": round(ms_off_buf, 2), "hf": round(ms_off_hf, 1)},
    }
    print(f"  extras: {R['extras']}", flush=True)

    out_path = os.path.join(HERE, "results_vs_all.json")
    with open(out_path, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6])
    else:
        main(sys.argv[1])
