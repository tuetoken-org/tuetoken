"""Benchmark tuetoken against tiktoken and Hugging Face tokenizers.

Writes tools/bench/results.json; tools/bench/plot.py turns it into
bench_tokenizers.png.

    pip install tuetoken tiktoken tokenizers transformers jinja2 matplotlib
    python tools/bench/bench.py CORPUS.txt

CORPUS.txt is plain text with documents separated by <|endoftext|> (e.g. an
OpenWebText dump). All encode benchmarks use o200k_base; ids are checked to be
identical across the three libraries before anything is timed.

Every measurement draws distinct text from the corpus (no prompt is encoded
twice), and corpus throughput runs each (library, threads) point in a fresh
process, so no library starts with warm caches.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time

SEP = b"<|endoftext|>"
HERE = os.path.dirname(os.path.abspath(__file__))
HF_O200K = ("Xenova/gpt-4o", "tokenizer.json")   # o200k_base in tokenizer.json form
CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def hf_o200k():
    import tokenizers
    from huggingface_hub import hf_hub_download
    return tokenizers.Tokenizer.from_file(hf_hub_download(*HF_O200K))


def load_docs(path, nbytes, skip=0):
    """Whole documents from the corpus, as str, about `nbytes` in total."""
    with open(path, "rb") as f:
        f.seek(skip)
        raw = f.read(nbytes)
    raw = raw[: raw.rfind(SEP)]
    return [d.decode("utf-8", "ignore") for d in raw.split(SEP) if d]


def text_pieces(path, size, count, skip):
    """`count` distinct pieces of about `size` bytes, cut at whitespace."""
    with open(path, "rb") as f:
        f.seek(skip)
        raw = f.read(size * count * 2 + 4096).replace(SEP, b"\n")
    out, pos = [], 0
    while len(out) < count:
        end = raw.find(b" ", pos + size)
        out.append(raw[pos:end].decode("utf-8", "ignore"))
        pos = end + 1
    return out


def p50_ms(fn, items):
    ts = []
    for x in items:
        t = time.perf_counter_ns()
        fn(x)
        ts.append(time.perf_counter_ns() - t)
    return statistics.median(ts) / 1e6


# ── corpus throughput: one fresh process per point ─────────────────────────

def worker(lib, threads, path, nbytes):
    """Encode the first `nbytes` of the corpus; print bytes/s as JSON."""
    if lib == "hf":
        os.environ["RAYON_NUM_THREADS"] = str(threads)
    with open(path, "rb") as f:
        raw = f.read(nbytes)
    raw = raw[: raw.rfind(SEP)]
    if lib == "tuetoken":
        import tuetoken
        enc = tuetoken.Tokenizer.from_tiktoken("o200k_base")
        t = time.perf_counter()
        out = enc.encode_ordinary_to_buffer([raw], num_threads=threads, separator=SEP)
        dt = time.perf_counter() - t
        ntok = len(out["data"])
    else:
        # The others take a list of str; splitting and decoding are not timed.
        docs = [d.decode("utf-8") for d in raw.split(SEP)]
        step = 4096
        ntok = 0
        if lib == "tiktoken":
            import tiktoken
            enc = tiktoken.get_encoding("o200k_base")
            t = time.perf_counter()
            for i in range(0, len(docs), step):
                ntok += sum(map(len, enc.encode_ordinary_batch(docs[i:i + step], num_threads=threads)))
            dt = time.perf_counter() - t
        else:
            enc = hf_o200k()
            t = time.perf_counter()
            for i in range(0, len(docs), step):
                ntok += sum(len(e.ids) for e in enc.encode_batch_fast(docs[i:i + step], add_special_tokens=False))
            dt = time.perf_counter() - t
    print(json.dumps({"bps": len(raw) / dt, "tokens": ntok, "seconds": dt}))


def corpus_scaling(path, nbytes, thread_list, repeats):
    res = {}
    for lib in ("tuetoken", "tiktoken", "hf"):
        res[lib] = []
        for n in thread_list:
            runs = []
            for _ in range(repeats if lib == "tuetoken" else 1):
                p = subprocess.run([sys.executable, __file__, "--worker", lib, str(n), path, str(nbytes)],
                                   capture_output=True, text=True, check=True)
                runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
            best = statistics.median(r["bps"] for r in runs)
            res[lib].append({"threads": n, "mbps": best / 1e6, "tokens": runs[0]["tokens"]})
            print(f"  corpus {lib:9s} {n:2d} threads: {best / 1e6:8.1f} MB/s", flush=True)
    toks = {res[l][0]["tokens"] for l in res}
    assert len(toks) == 1, f"token counts differ: {toks}"
    return res


# ── in-process benchmarks ──────────────────────────────────────────────────

def main(path):
    import tiktoken
    import tuetoken

    ncpu = os.cpu_count()
    tk = tiktoken.get_encoding("o200k_base")
    tt = tuetoken.Tokenizer.from_tiktoken("o200k_base")
    hf = hf_o200k()

    # parity first
    docs = load_docs(path, 32 << 20)[:5000]
    for d in docs:
        a = tk.encode_ordinary(d)
        assert a == tt.encode_ordinary(d) == hf.encode(d, add_special_tokens=False).ids
    print(f"parity: {len(docs)} documents identical across the three libraries", flush=True)

    R = {"meta": {
        "machine": platform.machine(), "cpu": cpu_name(), "cores": ncpu,
        "python": platform.python_version(), "encoding": "o200k_base",
        "versions": {"tuetoken": tuetoken.__version__, "tiktoken": tiktoken.__version__,
                     "tokenizers": __import__("tokenizers").__version__,
                     "transformers": __import__("transformers").__version__},
    }}

    # (a) corpus throughput vs threads
    nbytes = int(os.environ.get("BENCH_BYTES", 1 << 30))
    threads = [n for n in (1, 2, 4, 8, 10, 14, 16, 32, 64, 96, 128, 192) if n < ncpu] + [ncpu]
    cache = os.path.join(HERE, "corpus_cache.json")   # reuse with BENCH_REUSE_CORPUS=1
    if os.environ.get("BENCH_REUSE_CORPUS") and os.path.exists(cache):
        R["corpus"] = json.load(open(cache))
    else:
        R["corpus"] = {"bytes": nbytes, "results": corpus_scaling(path, nbytes, threads, 3)}
        json.dump(R["corpus"], open(cache, "w"), indent=1)

    skip = 1 << 30   # in-process benchmarks read text after the corpus slice

    # (b) single-call encode latency, distinct pieces of text
    R["encode_latency"] = []
    for size, n in ((128, 4000), (4096, 2000), (32768, 500), (131072, 200)):
        pieces = text_pieces(path, size, n, skip)
        skip += size * n * 2 + 4096
        row = {"size": size,
               "tuetoken": p50_ms(tt.encode_ordinary, pieces),
               "tiktoken": p50_ms(tk.encode_ordinary, pieces),
               "hf": p50_ms(lambda s: hf.encode(s, add_special_tokens=False), pieces)}
        R["encode_latency"].append(row)
        print(f"  encode {size:6d} B: {row}", flush=True)

    # (c) batch encode, all cores; every call gets documents its library has not seen
    R["batch"] = []
    sizes, reps = (100, 1000, 10000), 4      # first rep is warm-up
    pool = load_docs(path, 600 << 20, skip)
    skip += 600 << 20
    need = sum(sizes) * reps
    assert len(pool) >= need + 3000, (len(pool), need)
    batches, pos = {}, 0
    for bs in sizes:
        batches[bs] = [pool[pos + r * bs: pos + (r + 1) * bs] for r in range(reps)]
        pos += bs * reps
    for bs in sizes:
        row = {"batch": bs}
        for lib, fn in (("tuetoken", lambda b: tt.encode_ordinary_batch(b)),
                        ("tiktoken", lambda b: tk.encode_ordinary_batch(b, num_threads=ncpu)),
                        ("hf", lambda b: hf.encode_batch_fast(b, add_special_tokens=False))):
            rates = []
            for rep, batch in enumerate(batches[bs]):
                nb = sum(len(s.encode()) for s in batch)
                t = time.perf_counter()
                fn(batch)
                dt = time.perf_counter() - t
                if rep:
                    rates.append(nb / dt / 1e6)
            row[lib] = statistics.median(rates)
        R["batch"].append(row)
        print(f"  batch {bs:6d}: {row}", flush=True)

    # (d) decode latency
    R["decode_latency"] = []
    ids = tk.encode_ordinary("".join(pool[pos:pos + 3000]))
    for ntok, n in ((50, 4000), (1000, 2000), (12000, 300), (64000, 60)):
        seqs = [ids[i * ntok:(i + 1) * ntok] for i in range(n) if (i + 1) * ntok <= len(ids)]
        assert len(seqs) >= min(n, 10), (ntok, len(seqs), len(ids))
        row = {"tokens": ntok,
               "tuetoken": p50_ms(tt.decode, seqs),
               "tiktoken": p50_ms(tk.decode, seqs),
               "hf": p50_ms(hf.decode, seqs)}
        R["decode_latency"].append(row)
        print(f"  decode {ntok:6d} tok: {row}", flush=True)

    # (e) short chats through the AutoTokenizer API (vs transformers)
    from transformers import AutoTokenizer as HFAuto
    from tuetoken import AutoTokenizer as TTAuto
    hfa, tta = HFAuto.from_pretrained(CHAT_MODEL), TTAuto.from_pretrained(CHAT_MODEL)
    q = text_pieces(path, 200, 3000, skip)
    a = text_pieces(path, 400, 3000, skip + (4 << 20))
    chats = [[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": q[i]},
              {"role": "assistant", "content": a[i]}] for i in range(3000)]
    for c in chats[:300]:
        assert (hfa.apply_chat_template(c, tokenize=True, return_dict=False)
                == tta.apply_chat_template(c, return_dict=False))
    R["chat"] = {"model": CHAT_MODEL,
                 "tuetoken": p50_ms(lambda c: tta.apply_chat_template(c, return_dict=False), chats[300:]),
                 "transformers": p50_ms(lambda c: hfa.apply_chat_template(c, tokenize=True, return_dict=False),
                                        chats[300:])}
    # (f) batch encode through the AutoTokenizer call, tok(texts), vs transformers
    docs = load_docs(path, 64 << 20, skip + (16 << 20))
    sets = [docs[i * 1000:(i + 1) * 1000] for i in range(4)]
    assert hfa(sets[0][:200])["input_ids"] == tta(sets[0][:200])["input_ids"]
    for name, tok in (("tuetoken", tta), ("transformers", hfa)):
        rates = []
        for s_ in sets[1:]:
            nb = sum(len(x.encode()) for x in s_)
            t = time.perf_counter()
            tok(s_)
            rates.append(nb / (time.perf_counter() - t) / 1e6)
        R["chat"][name + "_batch_mbps"] = statistics.median(rates)
    print(f"  chat: {R['chat']}", flush=True)

    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1)
    print("wrote", out)


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        return platform.processor()


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
    else:
        main(sys.argv[1])
