"""Draw bench_tokenizers.png from tools/bench/results.json (see bench.py).

    python tools/bench/plot.py
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(HERE, "results.json")))
META = R["meta"]

BG, INK, MUTED, GRID = "#f8fafc", "#0f172a", "#64748b", "#e2e8f0"
C = {"tuetoken": "#635bff", "tiktoken": "#10b981", "hf": "#f59e0b", "transformers": "#f59e0b"}
NAME = {"tuetoken": "tuetoken", "tiktoken": "tiktoken (OpenAI)", "hf": "HuggingFace"}
LIBS = ("tuetoken", "tiktoken", "hf")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.facecolor": BG, "figure.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
})


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=14)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=6)


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def fmt_x(v):
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"


def fmt_rate(mbps):
    return f"{mbps / 1000:.1f} GB/s" if mbps >= 1000 else f"{mbps:.0f} MB/s"


def grouped(ax, labels, rows, keys, higher_better, ylabel, xlabel):
    """Bars per library on a log axis, tuetoken's speedup vs the fastest other written above."""
    x = np.arange(len(labels))
    w = 0.26
    for j, k in enumerate(keys):
        ax.bar(x + (j - 1) * w, [r[k] for r in rows], w, color=C[k], zorder=2)
    for i, r in enumerate(rows):
        others = [r[k] for k in keys[1:]]
        sp = (r["tuetoken"] / max(others)) if higher_better else (min(others) / r["tuetoken"])
        top = max(r[k] for k in keys)
        ax.text(x[i], top * 1.25, fmt_x(sp), ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=C["tuetoken"])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    lo = min(r[k] for r in rows for k in keys)
    hi = max(r[k] for r in rows for k in keys)
    ax.set_ylim(lo / 2.5, hi * 4)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def size_label(b):
    return f"{b} B" if b < 1024 else f"{b // 1024} KB"


def tok_label(n):
    return f"{n}" if n < 1000 else f"{n // 1000}K"


fig = plt.figure(figsize=(25.6, 14.4))
gs = fig.add_gridspec(2, 3, left=0.05, right=0.985, top=0.83, bottom=0.07, hspace=0.42, wspace=0.27)
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

# ── header ─────────────────────────────────────────────────────────────────
fig.text(0.02, 0.955, "Tokenizer Benchmark", fontsize=40, fontweight="bold", color=INK, va="center")
fig.text(0.02, 0.905, "Corpus throughput, latency, batch encoding and decoding · ids identical across all three · × = tuetoken vs the faster of the other two",
         fontsize=17, color=MUTED, va="center")
cores = META["cores"]
fig.text(0.985, 0.97, f"{META['encoding']}  /  {META['cpu']}, {cores} cores", fontsize=14,
         color=MUTED, ha="right", va="center")
lx = 0.985
for k in reversed(LIBS):
    t = fig.text(lx, 0.925, NAME[k], fontsize=15, color=INK, ha="right", va="center")
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    fig.patches.append(matplotlib.patches.Rectangle((bb.x0 - 0.016, 0.917), 0.011, 0.016,
                                                    transform=fig.transFigure, color=C[k]))
    lx = bb.x0 - 0.03
v = META["versions"]
fig.text(0.985, 0.012, f"tuetoken {v['tuetoken']} · tiktoken {v['tiktoken']} · tokenizers {v['tokenizers']} · "
         f"transformers {v['transformers']} · Python {META['python']} · text: OpenWebText",
         fontsize=11, color=MUTED, ha="right")

# ── (a) corpus throughput vs threads ───────────────────────────────────────
ax = axs[0]
cr = R["corpus"]["results"]
for k, mk, ls in (("tuetoken", "o", "-"), ("tiktoken", "s", "--"), ("hf", "^", ":")):
    th = [p["threads"] for p in cr[k]]
    mb = [p["mbps"] for p in cr[k]]
    ax.plot(th, mb, ls, marker=mk, color=C[k], lw=3 if k == "tuetoken" else 2.2, ms=9,
            mfc="white" if k == "tuetoken" else C[k], mew=2.5, zorder=3)
    ax.annotate(fmt_rate(mb[-1]), (th[-1], mb[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=12, fontweight="bold", color=C[k])
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
th = [p["threads"] for p in cr["tuetoken"]]
ax.set_xticks(th)
ax.set_xlim(0, th[-1] * 1.22)
ax.set_xlabel("Threads")
ax.set_ylabel("Throughput (MB/s, log)")
gb = R["corpus"]["bytes"] / (1 << 30)
style(ax, f"(a) Corpus encode, {gb:g} GB · thread scaling")

# ── (b) encode latency ─────────────────────────────────────────────────────
ax = axs[1]
el = R["encode_latency"]
grouped(ax, [size_label(r["size"]) for r in el], el, LIBS, False, "Latency (ms, log)", "Prompt size")
style(ax, "(b) Encode latency (p50)")

# ── (c) batch encode ───────────────────────────────────────────────────────
ax = axs[2]
bt = R["batch"]
grouped(ax, [f"{r['batch']:,}" for r in bt], bt, LIBS, True, "MB/s (log)", "Batch size (documents)")
style(ax, f"(c) Batch encode · {cores} threads")

# ── (d) decode latency ─────────────────────────────────────────────────────
ax = axs[3]
dl = R["decode_latency"]
grouped(ax, [tok_label(r["tokens"]) for r in dl], dl, LIBS, False, "Latency (ms, log)", "Tokens")
style(ax, "(d) Decode latency (p50)")

# ── speedups ───────────────────────────────────────────────────────────────
best = {k: cr[k][-1]["mbps"] for k in LIBS}
one = {k: cr[k][0]["mbps"] for k in LIBS}
summary = [  # (label, vs tiktoken, vs HF)
    (f"Corpus, {cores} threads", best["tuetoken"] / best["tiktoken"], best["tuetoken"] / best["hf"]),
    ("Corpus, 1 thread", one["tuetoken"] / one["tiktoken"], one["tuetoken"] / one["hf"]),
    ("Batch encode", gmean([r["tuetoken"] / r["tiktoken"] for r in bt]), gmean([r["tuetoken"] / r["hf"] for r in bt])),
    ("Encode latency", gmean([r["tiktoken"] / r["tuetoken"] for r in el]), gmean([r["hf"] / r["tuetoken"] for r in el])),
    ("Decode latency", gmean([r["tiktoken"] / r["tuetoken"] for r in dl]), gmean([r["hf"] / r["tuetoken"] for r in dl])),
    ("AutoTokenizer", None, R["chat"]["tuetoken_batch_mbps"] / R["chat"]["transformers_batch_mbps"]),
    ("Chat template", None, R["chat"]["transformers"] / R["chat"]["tuetoken"]),
]

# ── (e) headline ───────────────────────────────────────────────────────────
ax = axs[4]
ax.axis("off")
style(ax, "(e) Peak performance")
ax.yaxis.grid(False)
ax.text(0.5, 0.70, fmt_rate(best["tuetoken"]), ha="center", va="center", fontsize=96,
        fontweight="bold", color=C["tuetoken"], transform=ax.transAxes)
ax.text(0.5, 0.45, f"corpus encode on {cores} cores", ha="center", va="center",
        fontsize=19, color=MUTED, transform=ax.transAxes)
ax.plot([0.08, 0.92], [0.34, 0.34], color=GRID, lw=1.5, transform=ax.transAxes)
tiles = [(summary[0][1], "corpus vs\ntiktoken"), (summary[0][2], "corpus vs\nHuggingFace"),
         (summary[1][1], "1 thread vs\ntiktoken"), (summary[6][2], "chat template\nvs transformers")]
for i, (val, lab) in enumerate(tiles):
    xx = 0.125 + i * 0.25
    ax.text(xx, 0.20, fmt_x(val), ha="center", va="center", fontsize=34, fontweight="bold",
            color=C["tuetoken"], transform=ax.transAxes)
    ax.text(xx, 0.04, lab, ha="center", va="center", fontsize=13, color=MUTED, transform=ax.transAxes)

# ── (f) speedup summary ────────────────────────────────────────────────────
ax = axs[5]
y = np.arange(len(summary))
h = 0.36
for i, (lab, a, b) in enumerate(summary):
    for off, val, k in ((-h / 2, a, "tiktoken"), (h / 2, b, "hf")):
        if val is None:
            continue
        ax.barh(i + off, val, h * 0.92, color=C[k], zorder=2)
        ax.text(val * 1.08, i + off, fmt_x(val), va="center", fontsize=12, fontweight="bold", color=C[k])
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
allv = [v for s in summary for v in s[1:] if v]
ax.set_xlim(1, max(allv) * 3)
ax.set_yticks(y, [s[0] for s in summary], fontsize=13, color=INK)
ax.invert_yaxis()
ax.set_xlabel("tuetoken speedup (log) · green: vs tiktoken · orange: vs HuggingFace / transformers")
style(ax, "(f) Speedup summary")
ax.yaxis.grid(False)
ax.xaxis.grid(True, color=GRID, lw=0.8)

out = os.path.join(ROOT, "bench_tokenizers.png")
fig.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
print("summary:", [(s[0], s[1] and round(s[1], 1), round(s[2], 1)) for s in summary])
