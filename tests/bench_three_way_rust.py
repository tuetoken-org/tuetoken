"""Three-Way Tokenizer Benchmark: tuetoken vs tiktoken vs HuggingFace tokenizers.

Generates a single unified plot with 6 panels (2x3) comparing all three:
  (a) Thread Scaling (line)  (b) Encode Latency (bar)   (c) Batch Throughput (bar)
  (d) Decode Latency (bar)   (e) Token Counting (bar)   (f) Speedup Summary (hbar)

Usage:
    python bench_three_way.py
    python bench_three_way.py --encoding o200k_base
    python bench_three_way.py --plot-only
"""

import argparse
import gc
import json
import os
import random
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # vendored tiktoken->tokenizer.json helper

import tiktoken
import tuetoken
from _tiktoken_json import get_cached_tokenizer_path  # vendored, no C dependency
from tokenizers import Tokenizer as HFTokenizer

# ── Configuration ──
# Corpus: optional tests/data/corpus.txt; generated in-memory if absent.
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "corpus.txt")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "bench_three_way_rust_results.json")
DEFAULT_ENCODING = "o200k_base"
NUM_THREADS = os.cpu_count() or 1


def load_hf_tokenizer(tokenizer_json_path: str) -> HFTokenizer:
    """Load HF tokenizer, patching minimal decoder configs if needed."""
    try:
        return HFTokenizer.from_file(tokenizer_json_path)
    except Exception:
        pass
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    decoder = cfg.get("decoder") or {}
    if decoder.get("type") == "ByteLevel":
        cfg["decoder"] = {
            "type": "ByteLevel",
            "add_prefix_space": decoder.get("add_prefix_space", True),
            "trim_offsets": decoder.get("trim_offsets", True),
            "use_regex": decoder.get("use_regex", True),
        }
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump(cfg, tmp)
    tmp.close()
    hf = HFTokenizer.from_file(tmp.name)
    os.unlink(tmp.name)
    return hf


def load_text():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    # Self-contained fallback: build a ~1 MB corpus from a realistic paragraph so
    # the bench runs anywhere (no external corpus needed). Same text feeds all
    # three tokenizers, so the comparison stays fair.
    para = (
        "The quick brown fox jumps over the lazy dog. In 2024, researchers at the "
        "University published a paper (DOI:10.1/abc) on tokenization: encoding text "
        "into subword units like 'tokenization' -> ['token', 'ization']. "
        "Performance matters — e.g. cl100k_base handles ~10MB/s. café, naïve, 你好, "
        "emoji 🚀, and code:\n    def f(x):\n        return x ** 2 + 1  # squared\n"
        "Numbers: 3.14159, 1,000,000, 0xFF; URLs: https://example.com/path?q=1#frag.\n\n"
    )
    return (para * (1_000_000 // len(para) + 1))[:1_000_000]


def percentile(data, p):
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def generate_documents(text, num_docs=10000, target_chars=4000):
    """Generate documents of approximately target_chars length."""
    docs, text_len = [], len(text)
    random.seed(42)
    for _ in range(num_docs):
        start = random.randint(0, max(0, text_len - target_chars - 1))
        docs.append(text[start:start + target_chars])
    return docs


def generate_short_documents(text, num_docs=10000, target_chars=120):
    """Generate SHORT documents (~30 tokens) - where tuetoken shines most."""
    docs, text_len = [], len(text)
    random.seed(42)
    for _ in range(num_docs):
        start = random.randint(0, max(0, text_len - target_chars - 1))
        docs.append(text[start:start + target_chars])
    return docs


# =============================================================================
# 1. THROUGHPUT — Thread Scaling
# =============================================================================

def bench_throughput(tk, tt, hf, docs, max_threads):
    print("\n" + "=" * 70)
    print("1. ENCODE: Thread Scaling")
    print("=" * 70)
    num_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"   {len(docs):,} docs, {num_bytes / 1e6:.1f} MB\n")

    thread_counts = [t for t in [1, 2, 4, 8, 16, 32, 64, 96] if t <= max_threads]
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)
    thread_counts = sorted(set(thread_counts))

    results = []
    for nt in thread_counts:
        # warmup
        tk.encode_ordinary_batch(docs[:100], num_threads=nt)
        tt.encode_ordinary_batch(docs[:100], num_threads=nt)
        (hf.encode_batch(docs[:100], add_special_tokens=False) if nt > 1
         else [hf.encode(d, add_special_tokens=False) for d in docs[:100]])
        gc.collect()

        t0 = time.perf_counter()
        tk.encode_ordinary_batch(docs, num_threads=nt)
        tk_tp = num_bytes / (time.perf_counter() - t0) / 1e6
        gc.collect()

        t0 = time.perf_counter()
        tt.encode_ordinary_batch(docs, num_threads=nt)
        tt_tp = num_bytes / (time.perf_counter() - t0) / 1e6
        gc.collect()

        t0 = time.perf_counter()
        if nt == 1:
            for d in docs:
                hf.encode(d, add_special_tokens=False)
        else:
            hf.encode_batch(docs, add_special_tokens=False)
        hf_tp = num_bytes / (time.perf_counter() - t0) / 1e6

        print(f"   {nt:>2}T  tk:{tk_tp:>7.1f}  HF:{hf_tp:>7.1f}  tt:{tt_tp:>7.1f} MB/s  "
              f"({tt_tp/tk_tp:.1f}x tk, {tt_tp/hf_tp:.1f}x HF)")

        results.append({"threads": nt, "tk": tk_tp, "hf": hf_tp, "tt": tt_tp,
                         "tt_vs_tk": tt_tp / tk_tp, "tt_vs_hf": tt_tp / hf_tp})

    return {"thread_results": results, "num_bytes": num_bytes,
            "max_tt_vs_tk": max(r["tt_vs_tk"] for r in results)}


# =============================================================================
# 2. ENCODE: Single-Call Latency (p50)
# =============================================================================

def bench_encode_latency(tk, tt, hf, text, iterations=200):
    print("\n" + "=" * 70)
    print("2. ENCODE: Single-Call Latency (p50)")
    print("=" * 70)

    sizes = [("128 B", 128), ("4 KB", 4096), ("32 KB", 32768), ("128 KB", 131072)]
    results = []

    for label, sz in sizes:
        prompt = (text * ((sz // len(text)) + 1))[:sz]
        # warmup
        tk.encode_ordinary(prompt); tt.encode_ordinary(prompt)
        hf.encode(prompt, add_special_tokens=False)

        tk_t = [_time_ms(lambda: tk.encode_ordinary(prompt)) for _ in range(iterations)]
        hf_t = [_time_ms(lambda: hf.encode(prompt, add_special_tokens=False)) for _ in range(iterations)]
        tt_t = [_time_ms(lambda: tt.encode_ordinary(prompt)) for _ in range(iterations)]
        tk_p, hf_p, tt_p = percentile(tk_t, 50), percentile(hf_t, 50), percentile(tt_t, 50)

        print(f"   {label:<8}  tk:{tk_p:>7.3f}  HF:{hf_p:>7.3f}  tt:{tt_p:>7.3f} ms  "
              f"({tk_p/tt_p:.1f}x tk, {hf_p/tt_p:.1f}x HF)")
        results.append({"size": label, "tk": tk_p, "hf": hf_p, "tt": tt_p,
                         "tt_vs_tk": tk_p / tt_p, "tt_vs_hf": hf_p / tt_p})
    return results


# =============================================================================
# 3. ENCODE: Batch Throughput
# =============================================================================

def bench_batch_throughput(tk, tt, hf, text, num_threads):
    print("\n" + "=" * 70)
    print(f"3. ENCODE: Batch Throughput ({num_threads} threads)")
    print("=" * 70)

    text_ext = text * 50
    chunks = [text_ext[i:i+1024] for i in range(0, len(text_ext), 1024)][:50000]
    batch_sizes = [100, 1000, 5000, 10000]
    results = []

    for bs in batch_sizes:
        batch = chunks[:bs]
        nb = sum(len(c.encode("utf-8")) for c in batch)
        # warmup
        tk.encode_ordinary_batch(batch[:50], num_threads=num_threads)
        tt.encode_ordinary_batch(batch[:50], num_threads=num_threads)
        hf.encode_batch(batch[:50], add_special_tokens=False)
        gc.collect()

        t0 = time.perf_counter()
        tk.encode_ordinary_batch(batch, num_threads=num_threads)
        tk_tp = nb / (time.perf_counter() - t0) / 1e6; gc.collect()

        t0 = time.perf_counter()
        hf.encode_batch(batch, add_special_tokens=False)
        hf_tp = nb / (time.perf_counter() - t0) / 1e6; gc.collect()

        t0 = time.perf_counter()
        tt.encode_ordinary_batch(batch, num_threads=num_threads)
        tt_tp = nb / (time.perf_counter() - t0) / 1e6

        print(f"   {bs:>5}  tk:{tk_tp:>7.1f}  HF:{hf_tp:>7.1f}  tt:{tt_tp:>7.1f} MB/s  "
              f"({tt_tp/tk_tp:.1f}x tk, {tt_tp/hf_tp:.1f}x HF)")
        results.append({"batch_size": bs, "tk": tk_tp, "hf": hf_tp, "tt": tt_tp,
                         "tt_vs_tk": tt_tp / tk_tp, "tt_vs_hf": tt_tp / hf_tp})
    return results


# =============================================================================
# 4. DECODE: Tokens → Text (p50 latency)
# =============================================================================

def bench_decode_latency(tk, tt, hf, text, iterations=100):
    print("\n" + "=" * 70)
    print("4. DECODE: Tokens -> Text (p50)")
    print("=" * 70)

    sizes = [("50 tok", 200), ("1K tok", 4096), ("10K tok", 40960), ("50K tok", 204800)]
    results = []

    for label, char_sz in sizes:
        prompt = (text * ((char_sz // len(text)) + 1))[:char_sz]
        tk_ids = tk.encode_ordinary(prompt)
        tt_ids = tt.encode_ordinary(prompt)
        hf_ids = list(hf.encode(prompt, add_special_tokens=False).ids)
        n_tok = len(tt_ids)
        # warmup
        tk.decode(tk_ids); tt.decode(tt_ids); hf.decode(hf_ids)

        tk_t = [_time_ms(lambda: tk.decode(tk_ids)) for _ in range(iterations)]
        hf_t = [_time_ms(lambda: hf.decode(hf_ids)) for _ in range(iterations)]
        tt_t = [_time_ms(lambda: tt.decode(tt_ids)) for _ in range(iterations)]
        tk_p, hf_p, tt_p = percentile(tk_t, 50), percentile(hf_t, 50), percentile(tt_t, 50)

        print(f"   {label:<8} ({n_tok:>5} tok)  tk:{tk_p:>7.3f}  HF:{hf_p:>7.3f}  tt:{tt_p:>7.3f} ms  "
              f"({tk_p/tt_p:.1f}x tk, {hf_p/tt_p:.1f}x HF)")
        results.append({"size": label, "num_tokens": n_tok,
                         "tk": tk_p, "hf": hf_p, "tt": tt_p,
                         "tt_vs_tk": tk_p / tt_p, "tt_vs_hf": hf_p / tt_p})
    return results


# =============================================================================
# 5. SHORT DOC SCALING (where tuetoken shines most - 40x+ speedup)
# =============================================================================

def bench_short_doc_scaling(tk, tt, hf, short_docs, max_threads):
    """Benchmark with SHORT documents (~120 chars) - tiktoken struggles here."""
    print("\n" + "=" * 70)
    print("5. SHORT DOC SCALING (tiktoken's weak spot)")
    print("=" * 70)
    num_bytes = sum(len(d.encode("utf-8")) for d in short_docs)
    avg_len = sum(len(d) for d in short_docs) / len(short_docs)
    print(f"   {len(short_docs):,} docs, avg {avg_len:.0f} chars, {num_bytes / 1e6:.1f} MB\n")

    thread_counts = [t for t in [1, 2, 4, 8, 16, 32, 64, 96] if t <= max_threads]
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)
    thread_counts = sorted(set(thread_counts))

    results = []
    for nt in thread_counts:
        # warmup
        tk.encode_ordinary_batch(short_docs[:100], num_threads=nt)
        tt.encode_ordinary_batch(short_docs[:100], num_threads=nt)
        (hf.encode_batch(short_docs[:100], add_special_tokens=False) if nt > 1
         else [hf.encode(d, add_special_tokens=False) for d in short_docs[:100]])
        gc.collect()

        t0 = time.perf_counter()
        tk.encode_ordinary_batch(short_docs, num_threads=nt)
        tk_tp = num_bytes / (time.perf_counter() - t0) / 1e6
        gc.collect()

        t0 = time.perf_counter()
        tt.encode_ordinary_batch(short_docs, num_threads=nt)
        tt_tp = num_bytes / (time.perf_counter() - t0) / 1e6
        gc.collect()

        t0 = time.perf_counter()
        if nt == 1:
            for d in short_docs:
                hf.encode(d, add_special_tokens=False)
        else:
            hf.encode_batch(short_docs, add_special_tokens=False)
        hf_tp = num_bytes / (time.perf_counter() - t0) / 1e6

        print(f"   {nt:>2}T  tk:{tk_tp:>7.1f}  HF:{hf_tp:>7.1f}  tt:{tt_tp:>7.1f} MB/s  "
              f"({tt_tp/tk_tp:.1f}x tk, {tt_tp/hf_tp:.1f}x HF)")

        results.append({"threads": nt, "tk": tk_tp, "hf": hf_tp, "tt": tt_tp,
                        "tt_vs_tk": tt_tp / tk_tp, "tt_vs_hf": tt_tp / hf_tp})

    return {"thread_results": results, "num_bytes": num_bytes,
            "max_tt_vs_tk": max(r["tt_vs_tk"] for r in results),
            "avg_doc_length": avg_len}


# ── helper ──
def _time_ms(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(results_file) as f:
        R = json.load(f)

    # ── Style ──
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "text.color": "#333333",
        "grid.color": "#e8e8e8", "grid.linewidth": 0.7,
    })

    C = {"tk": "#f97316", "hf": "#3b82f6", "tt": "#22c55e"}
    w = 0.25

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.30,
                        top=0.88, bottom=0.07, left=0.06, right=0.97)

    # ── (a) Thread Scaling — LINE plot ──────────────────────────────────────
    ax = axes[0, 0]
    tr = R["throughput"]["thread_results"]
    threads = [r["threads"] for r in tr]
    ax.plot(threads, [r["tt"] for r in tr], "o-",  color=C["tt"], lw=2.2, ms=7, label="tuetoken (Rust)", zorder=3)
    ax.plot(threads, [r["tk"] for r in tr], "s--", color=C["tk"], lw=1.8, ms=6, label="tiktoken", zorder=2)
    ax.plot(threads, [r["hf"] for r in tr], "^:",  color=C["hf"], lw=1.5, ms=5, label="HuggingFace", zorder=1)
    ax.set_xlabel("Threads"); ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Encode: Thread Scaling", weight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads], fontsize=9, rotation=45, ha="right")
    ax.margins(y=0.1)
    _style(ax)
    _label(ax, "(a)")

    # ── (b) Encode Latency — BAR ───────────────────────────────────────────
    ax = axes[0, 1]
    lat = R["encode_latency"]
    labels = [r["size"] for r in lat]
    x = np.arange(len(labels))
    ax.bar(x - w, [r["tk"] for r in lat], w, color=C["tk"], label="tiktoken")
    ax.bar(x,     [r["hf"] for r in lat], w, color=C["hf"], label="HF tokenizers")
    b = ax.bar(x + w, [r["tt"] for r in lat], w, color=C["tt"], label="tuetoken (Rust)")
    for i, bar in enumerate(b):
        sp = lat[i]["tt_vs_tk"]
        ypos = bar.get_height()
        ax.annotate(f'{sp:.1f}x', xy=(bar.get_x() + bar.get_width()/2, ypos),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C["tt"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (ms)"); ax.set_xlabel("Prompt Size")
    ax.set_title("Encode: Single-Call Latency (p50)", weight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.margins(y=0.15)  # Add headroom for annotations
    _style(ax); _label(ax, "(b)")

    # ── (c) Batch Throughput — BAR ─────────────────────────────────────────
    ax = axes[0, 2]
    bat = R["batch_throughput"]
    labels = [f'{r["batch_size"]:,}' for r in bat]
    x = np.arange(len(labels))
    ax.bar(x - w, [r["tk"] for r in bat], w, color=C["tk"], label="tiktoken")
    ax.bar(x,     [r["hf"] for r in bat], w, color=C["hf"], label="HF tokenizers")
    b = ax.bar(x + w, [r["tt"] for r in bat], w, color=C["tt"], label="tuetoken (Rust)")
    for i, bar in enumerate(b):
        sp = bat[i]["tt_vs_tk"]
        ypos = bar.get_height()
        ax.annotate(f'{sp:.1f}x', xy=(bar.get_x() + bar.get_width()/2, ypos),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C["tt"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Throughput (MB/s)"); ax.set_xlabel("Batch Size (documents)")
    ax.set_title(f"Encode: Batch Throughput ({R['num_threads']} threads)", weight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.margins(y=0.15)  # Add headroom for annotations
    _style(ax); _label(ax, "(c)")

    # ── (d) Decode Latency — BAR ──────────────────────────────────────────
    ax = axes[1, 0]
    dec = R["decode_latency"]
    labels = [r["size"] for r in dec]
    x = np.arange(len(labels))
    ax.bar(x - w, [r["tk"] for r in dec], w, color=C["tk"], label="tiktoken")
    ax.bar(x,     [r["hf"] for r in dec], w, color=C["hf"], label="HF tokenizers")
    b = ax.bar(x + w, [r["tt"] for r in dec], w, color=C["tt"], label="tuetoken (Rust)")
    for i, bar in enumerate(b):
        sp = dec[i]["tt_vs_tk"]
        ypos = bar.get_height()
        ax.annotate(f'{sp:.1f}x', xy=(bar.get_x() + bar.get_width()/2, ypos),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C["tt"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)"); ax.set_xlabel("Token Count")
    ax.set_title("Decode: Tokens \u2192 Text (p50)", weight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.margins(y=0.15)  # Add headroom for annotations
    _style(ax); _label(ax, "(d)")

    # ── (e) Short Doc Scaling — LINE plot (shows 40x+ speedup) ─────────────
    ax = axes[1, 1]
    sd = R["short_doc_scaling"]
    tr = sd["thread_results"]
    threads = [r["threads"] for r in tr]
    ax.plot(threads, [r["tt"] for r in tr], "o-", color=C["tt"], lw=2.2, ms=7, label="tuetoken (Rust)", zorder=3)
    ax.plot(threads, [r["tk"] for r in tr], "s--", color=C["tk"], lw=1.8, ms=6, label="tiktoken", zorder=2)
    ax.plot(threads, [r["hf"] for r in tr], "^:", color=C["hf"], lw=1.5, ms=5, label="HuggingFace", zorder=1)
    ax.set_xlabel("Threads"); ax.set_ylabel("Throughput (MB/s)")
    ax.set_title(f"Short Docs (~{sd['avg_doc_length']:.0f} chars): Thread Scaling", weight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads], fontsize=9, rotation=45, ha="right")
    # Add peak speedup annotation
    peak_idx = max(range(len(tr)), key=lambda i: tr[i]["tt_vs_tk"])
    peak_sp = tr[peak_idx]["tt_vs_tk"]
    ax.annotate(f'{peak_sp:.0f}x faster!',
                xy=(threads[peak_idx], tr[peak_idx]["tt"]),
                xytext=(10, -20), textcoords="offset points",
                fontsize=11, fontweight="bold", color=C["tt"],
                arrowprops=dict(arrowstyle="->", color=C["tt"], lw=1.5))
    ax.margins(y=0.1)
    _style(ax); _label(ax, "(e)")

    # ── (f) Speedup Summary — horizontal bar ──────────────────────────────
    ax = axes[1, 2]
    # Collect speedups vs tiktoken
    sd = R["short_doc_scaling"]
    summary = [
        ("Short docs\n(peak)", sd["max_tt_vs_tk"]),
        ("Long docs\n(peak)", R["throughput"]["max_tt_vs_tk"]),
        ("Batch encode\n(peak)", max(r["tt_vs_tk"] for r in R["batch_throughput"])),
        ("Encode latency\n(avg p50)", sum(r["tt_vs_tk"] for r in R["encode_latency"]) / len(R["encode_latency"])),
        ("Decode latency\n(avg p50)", sum(r["tt_vs_tk"] for r in R["decode_latency"]) / len(R["decode_latency"])),
    ]
    summary.sort(key=lambda x: x[1])
    names = [s[0] for s in summary]
    vals  = [s[1] for s in summary]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=C["tt"], height=0.6, edgecolor="white", linewidth=0.5)
    max_val = max(vals)
    for bar, v in zip(bars, vals):
        offset = max_val * 0.02 + 0.3
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}x', va="center", ha="left",
                fontsize=11, fontweight="bold", color="#333333")
    ax.set_xlim(0, max_val * 1.18)  # Room for labels
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Speedup vs tiktoken")
    ax.set_title("Speedup Summary", weight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0, pad=6)
    _label(ax, "(f)")

    # ── Title ──
    fig.suptitle("tuetoken (Rust) \u2014 Comprehensive Benchmark Summary",
                 fontsize=19, weight="bold", color="#1a1a1a", y=0.97)
    fig.text(0.5, 0.935,
             f"BPE tokenizer implemented in Rust (PyO3 extension)  |  "
             f"{R['encoding']}  |  {R['num_threads']} cores",
             ha="center", fontsize=11, color="#888888")

    out = os.path.join(SCRIPT_DIR, "bench_three_way_rust.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\nPlot saved to: {out}")


def _style(ax):
    ax.yaxis.grid(True, linestyle="-", alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0, pad=6)


def _label(ax, text):
    ax.text(-0.08, 1.08, text, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Three-way tokenizer benchmark")
    parser.add_argument("--encoding", "-e", default=DEFAULT_ENCODING)
    parser.add_argument("--iterations", "-n", type=int, default=100)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"Error: {RESULTS_FILE} not found. Run benchmark first.")
            return
        print("Generating plot from saved results...")
        plot_results(RESULTS_FILE)
        return

    print("=" * 70)
    print("  TUETOKEN vs TIKTOKEN vs HF TOKENIZERS")
    print("=" * 70)
    print(f"  Encoding: {args.encoding}  |  CPU cores: {NUM_THREADS}\n")

    print("Loading tokenizers...")
    tk = tiktoken.get_encoding(args.encoding)
    tt_path = get_cached_tokenizer_path(args.encoding)
    tt = tuetoken.Tokenizer(tt_path)
    hf = load_hf_tokenizer(tt_path)
    print("  tiktoken: Rust  |  HF tokenizers: Rust  |  tuetoken: Rust/PyO3")

    text = load_text()
    docs = generate_documents(text, 10000, 4000)  # Long docs (~1000 tokens)
    short_docs = generate_short_documents(text, 10000, 120)  # Short docs (~30 tokens)
    print(f"  Seed: {len(text):,} chars  |  Long docs: {len(docs):,}  |  Short docs: {len(short_docs):,}\n")

    throughput = bench_throughput(tk, tt, hf, docs, NUM_THREADS)
    enc_lat    = bench_encode_latency(tk, tt, hf, text, args.iterations)
    batch_tp   = bench_batch_throughput(tk, tt, hf, text, NUM_THREADS)
    dec_lat    = bench_decode_latency(tk, tt, hf, text, args.iterations)
    short_doc  = bench_short_doc_scaling(tk, tt, hf, short_docs, NUM_THREADS)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"                            vs tiktoken    vs HF")
    print(f"  Long doc encode (peak):   {throughput['max_tt_vs_tk']:>5.1f}x          "
          f"{max(r['tt_vs_hf'] for r in throughput['thread_results']):>5.1f}x")
    print(f"  Short doc encode (peak):  {short_doc['max_tt_vs_tk']:>5.1f}x          "
          f"{max(r['tt_vs_hf'] for r in short_doc['thread_results']):>5.1f}x")
    print(f"  Encode latency (avg):     {sum(r['tt_vs_tk'] for r in enc_lat)/len(enc_lat):>5.2f}x          "
          f"{sum(r['tt_vs_hf'] for r in enc_lat)/len(enc_lat):>5.2f}x")
    print(f"  Decode latency (avg):     {sum(r['tt_vs_tk'] for r in dec_lat)/len(dec_lat):>5.2f}x          "
          f"{sum(r['tt_vs_hf'] for r in dec_lat)/len(dec_lat):>5.2f}x")
    print(f"  Batch encode (peak):      {max(r['tt_vs_tk'] for r in batch_tp):>5.1f}x          "
          f"{max(r['tt_vs_hf'] for r in batch_tp):>5.1f}x")

    all_results = {
        "encoding": args.encoding, "num_threads": NUM_THREADS,
        "throughput": throughput,
        "encode_latency": enc_lat,
        "batch_throughput": batch_tp,
        "decode_latency": dec_lat,
        "short_doc_scaling": short_doc,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")

    print("\nGenerating plot...")
    plot_results(RESULTS_FILE)
    print("Done!")


if __name__ == "__main__":
    main()
