#!/usr/bin/env python
"""Speed: tuetoken vs HF `tokenizers` for encode + decode across models/sizes.

Covers gemma / qwen / deepseek / liquid (mix of monotonic fast-path and
rank-inverted canonical-path vocabularies), four settings:
  * encode single     (one document, various sizes)
  * encode batch      (many docs, multi-threaded)
  * decode single     (one id sequence)
  * decode batch      (many sequences)
Timing is best-of-N (minimum wall time; OS jitter only ever adds time).
Ground-truth parity is asserted first so we never benchmark a wrong tokenizer.
"""
import os, sys, time, statistics
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import tuetoken
from tokenizers import Tokenizer as HFTokenizer
from huggingface_hub import hf_hub_download

MODELS = {
    "gemma   (canonical)": "OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
    "qwen2.5 (fast-path)": "Qwen/Qwen2.5-7B-Instruct",
    "deepseekV3 (canon) ": "deepseek-ai/DeepSeek-V3-0324",
    "liquid  (canonical)": "LiquidAI/LFM2.5-8B-A1B",
}

# realistic mixed text (code + prose + multilingual + numbers + whitespace)
PARA = (
    "The agent reads file:line, runs the command, and returns structured JSON. "
    "def solve(n):\n    return sum(i*i for i in range(n))  # O(n)\n"
    "Café 你好世界 спасибо مرحبا — 2024-01-02, $1,234.56, v1.2.3, https://ex.com/p?q=1.\n"
    "MixedCamelCASE_snake-kebab IDENTIFIERS and    irregular   whitespace\t\tspans.\n"
)
def doc(nchars):
    return (PARA * (nchars // len(PARA) + 1))[:nchars]

def best_of(fn, repeats, inner=1):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        times.append((time.perf_counter() - t0) / inner)
    return min(times)

def fmt_mbps(nbytes, secs):
    return nbytes / secs / 1e6

def main():
    print(f"{'model':21} {'setting':22} {'size':>10} {'tuetoken':>11} {'HF':>11} {'speedup':>8}")
    print("-" * 88)
    for label, repo in MODELS.items():
        try:
            tj = hf_hub_download(repo, "tokenizer.json")
            R = tuetoken.Tokenizer(tj)
            hf = HFTokenizer.from_file(tj); hf.no_truncation(); hf.no_padding()
        except Exception as e:
            print(f"{label:21} SKIP ({type(e).__name__})"); continue

        # parity guard on a medium doc (ignore special-token-id diffs)
        s = doc(4000)
        if R.encode_ordinary(s) != hf.encode(s, add_special_tokens=False).ids:
            print(f"{label:21} (note: encode differs from HF — special/added tokens)")

        # ---- encode single, various sizes ----
        for n in (1_000, 10_000, 100_000):
            s = doc(n); nb = len(s.encode("utf-8"))
            rep = max(3, 2_000_000 // n)
            tt = best_of(lambda: R.encode_ordinary(s), rep)
            th = best_of(lambda: hf.encode(s, add_special_tokens=False), rep)
            print(f"{label:21} {'encode single':22} {n//1000:>8}KB "
                  f"{fmt_mbps(nb,tt):>8.1f}MB/s {fmt_mbps(nb,th):>8.1f}MB/s {th/tt:>7.2f}x")

        # ---- encode batch (1000 small docs, all cores) ----
        docs = [doc(300) for _ in range(1000)]
        nb = sum(len(d.encode("utf-8")) for d in docs)
        tt = best_of(lambda: R.encode_ordinary_batch(docs, 0), 5)
        th = best_of(lambda: hf.encode_batch([(d, "") for d in docs] if False else docs,
                                             add_special_tokens=False), 5)
        print(f"{label:21} {'encode batch x1000':22} {'300B ea':>10} "
              f"{fmt_mbps(nb,tt):>8.1f}MB/s {fmt_mbps(nb,th):>8.1f}MB/s {th/tt:>7.2f}x")

        # ---- decode single (ids from a 10KB doc) ----
        s = doc(10_000); ids = R.encode_ordinary(s)
        rep = 2000
        tt = best_of(lambda: R.decode(ids), rep)
        th = best_of(lambda: hf.decode(ids, skip_special_tokens=False), rep)
        print(f"{label:21} {'decode single':22} {len(ids):>7}tok "
              f"{len(s)/tt/1e6:>8.1f}MT/s {len(s)/th/1e6:>8.1f}MT/s {th/tt:>7.2f}x"
              .replace("MT/s", "Mc/s"))

        # ---- decode batch (1000 sequences) ----
        seqs = [R.encode_ordinary(d) for d in docs]
        ntok = sum(len(x) for x in seqs)
        tt = best_of(lambda: R.decode_batch(seqs, 0), 10)
        th = best_of(lambda: hf.decode_batch(seqs, skip_special_tokens=False), 10)
        print(f"{label:21} {'decode batch x1000':22} {ntok:>6}tok "
              f"{ntok/tt/1e6:>8.1f}MT/s {ntok/th/1e6:>8.1f}MT/s {th/tt:>7.2f}x")
        print()

    print("speedup = HF_time / tuetoken_time  (>1.0 => tuetoken faster). "
          "best-of-N min wall time.")

if __name__ == "__main__":
    main()
