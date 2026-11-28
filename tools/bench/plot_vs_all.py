"""Render publication-grade bench_tokenizers.png from tools/bench/results_vs_all.json.

    python tools/bench/plot_vs_all.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(HERE, "results_vs_all.json")))
META = R["meta"]
CORES = META["cores"]

BG, CARD, INK, MUTED, GRID = "#f8fafc", "#ffffff", "#0f172a", "#64748b", "#e2e8f0"
C = {
    "tuetoken": "#635bff",
    "gigatoken": "#ef4444",
    "fastokens": "#0ea5e9",
    "tiktoken": "#10b981",
    "hf": "#f59e0b",
}
LABEL = {
    "tuetoken": f"tuetoken {META['versions']['tuetoken']}",
    "gigatoken": f"gigatoken {META['versions']['gigatoken']}",
    "fastokens": f"fastokens {META['versions']['fastokens']}",
    "tiktoken": f"tiktoken {META['versions']['tiktoken']}",
    "hf": f"HF tokenizers {META['versions']['tokenizers']}",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.facecolor": CARD,
    "figure.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def style(ax, title, grid_axis="y"):
    ax.set_title(title, loc="left", fontsize=14.5, fontweight="bold", color=INK, pad=12)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=6)


def fmt_x(v):
    if v >= 1000:
        return f"{v:,.0f}×"
    if v >= 10:
        return f"{v:.0f}×"
    return f"{v:.1f}×"


def fmt_rate(mbps):
    return f"{mbps / 1000:.2f} GB/s" if mbps >= 1000 else f"{mbps:.0f} MB/s"


fig = plt.figure(figsize=(24, 14.2), facecolor=BG)
gs = fig.add_gridspec(2, 3, left=0.052, right=0.98, top=0.805, bottom=0.07, hspace=0.40, wspace=0.34)
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

# ── Header & Legend ────────────────────────────────────────────────────────
fig.text(0.025, 0.955, "tuetoken — Head-to-Head Tokenizer Benchmark", fontsize=34, fontweight="bold", color=INK, va="center")
fig.text(
    0.025, 0.910,
    "Bit-for-bit identical token IDs verified across all libraries · Fresh-process cold 1 GB corpus, frontier LLMs, startup, decode & exact O(k) truncation",
    fontsize=14, color=MUTED, va="center",
)
fig.text(
    0.98, 0.958,
    f"{META['cpu']} ({CORES} cores) · Python {META['python']}",
    fontsize=13.5, color=MUTED, ha="right", va="center", fontweight="bold",
)
# Dedicated clean legend row (zero overlap with subtitle)
lx = 0.025
for k in ["tuetoken", "gigatoken", "fastokens", "tiktoken", "hf"]:
    fig.patches.append(mpatches.FancyBboxPatch(
        (lx, 0.862), 0.0095, 0.014,
        boxstyle="round,pad=0.001", transform=fig.transFigure, color=C[k],
    ))
    t = fig.text(lx + 0.014, 0.869, LABEL[k], fontsize=12.8, color=INK, ha="left", va="center",
                 fontweight="bold" if k == "tuetoken" else "normal")
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    lx = bb.x1 + 0.026

fig.text(
    0.98, 0.014,
    "Reproduce: python tools/bench/bench_vs_all.py corpus_1gb.txt && python tools/bench/plot_vs_all.py  ·  Corpus: OpenWebText (o200k_base & HF tokenizer.json)",
    fontsize=10.5, color=MUTED, ha="right",
)

# ── (a) Cold 1 GB Corpus Scaling (o200k_base, fresh process) ───────────────
ax = axs[0]
cold = R["corpus"]["cold"]
for k, mk, ls in (("tuetoken", "o", "-"), ("gigatoken", "D", "-"), ("tiktoken", "s", "--"), ("hf", "^", ":")):
    if k not in cold or not cold[k]:
        continue
    th = [p["threads"] for p in cold[k]]
    mb = [p["mbps"] for p in cold[k]]
    ax.plot(th, mb, ls, marker=mk, color=C[k], lw=3.2 if k == "tuetoken" else 2.2,
            ms=8.5, mfc="white" if k == "tuetoken" else C[k], mew=2.4, zorder=4 if k == "tuetoken" else 3)
    yoff = 6 if k == "tuetoken" else (-4 if k == "gigatoken" else 0)
    ax.annotate(fmt_rate(mb[-1]), (th[-1], mb[-1]), xytext=(8, yoff), textcoords="offset points",
                va="center", fontsize=11.5, fontweight="bold", color=C[k])
if "fastokens" in cold and cold["fastokens"]:
    fp = cold["fastokens"][-1]
    ax.scatter([fp["threads"]], [fp["mbps"]], marker="P", s=95, color=C["fastokens"], zorder=4)
    ax.annotate(fmt_rate(fp["mbps"]), (fp["threads"], fp["mbps"]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=11.5, fontweight="bold", color=C["fastokens"])

tue_cold = cold["tuetoken"][-1]["mbps"]
gt_cold = cold["gigatoken"][-1]["mbps"]
ax.text(
    0.04, 0.43,
    f"tuetoken vs gigatoken ({CORES}T cold): {fmt_x(tue_cold / gt_cold)}",
    transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=C["tuetoken"],
    bbox=dict(boxstyle="round,pad=0.38", facecolor="#eef2ff", edgecolor="#c7d2fe", lw=1),
)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
th_all = [p["threads"] for p in cold["tuetoken"]]
ax.set_xticks(th_all)
ax.set_xlim(0.3, th_all[-1] * 1.32)
ax.set_xlabel("Threads")
ax.set_ylabel("Throughput (MB/s, log)")
gb = R["corpus"]["bytes"] / (1 << 30)
style(ax, f"(a) Cold {gb:.0f} GB corpus encode (fresh process)")

# ── (b) Frontier LLM Throughput (25 MB, all cores) ─────────────────────────
ax = axs[1]
fr = R["frontier"]
y = np.arange(len(fr))
h = 0.26
for idx, (k, off) in enumerate([("tuetoken", -h), ("gigatoken", 0.0), ("hf", h)]):
    vals = [r[k] or 0.0 for r in fr]
    ax.barh(y + off, vals, h * 0.92, color=C[k], zorder=3)
max_fr = max(r["tuetoken"] for r in fr)
for i, r in enumerate(fr):
    tv = r["tuetoken"]
    gv = r["gigatoken"]
    if gv:
        lbl = f"{fmt_rate(tv)} ({fmt_x(tv / gv)} vs gt)"
    else:
        lbl = f"{fmt_rate(tv)} (gt: unsupported)"
    ax.text(tv + max_fr * 0.018, i - h, lbl, va="center", fontsize=10, fontweight="bold", color=C["tuetoken"])
    if gv is None:
        ax.text(max_fr * 0.015, i, "unsupported", va="center", fontsize=9, style="italic", color=C["gigatoken"])
ax.set_yticks(y, [r["model"] for r in fr], fontsize=11, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, max_fr * 1.48)
ax.set_xlabel("Throughput (MB/s, higher is better)")
style(ax, f"(b) Frontier LLMs (25 MB, {CORES} cores)", grid_axis="x")

# ── (c) Startup Latency (tokenizer.json load, ms, log) ─────────────────────
ax = axs[2]
st = R["startup"]
x = np.arange(len(st))
w = 0.27
for j, k in enumerate(("tuetoken", "gigatoken", "hf")):
    vals = [r[k] for r in st]
    ax.bar(x + (j - 1) * w, [v or 0 for v in vals], w * 0.94, color=C[k], zorder=3)
for i, r in enumerate(st):
    sp = r["gigatoken"] / r["tuetoken"]
    top = max(v for v in (r["tuetoken"], r["gigatoken"], r["hf"]) if v)
    ax.text(x[i], top * 1.18, fmt_x(sp), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C["tuetoken"])
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
all_st = [r[k] for r in st for k in ("tuetoken", "gigatoken", "hf") if r[k]]
ax.set_ylim(min(all_st) * 0.5, max(all_st) * 2.6)
ax.set_xticks(x, [r["model"] for r in st], rotation=18, ha="right", fontsize=10)
ax.set_ylabel("Load time (ms, log · lower is better)")
style(ax, "(c) Startup latency (× = vs gigatoken)")

# ── (d) Decode Latency (p50, us, log) ──────────────────────────────────────
ax = axs[3]
dc = R["decode"]
x = np.arange(len(dc))
keys_d = ("tuetoken", "gigatoken", "tiktoken", "fastokens", "hf")
w = 0.165
for j, k in enumerate(keys_d):
    ax.bar(x + (j - 2) * w, [r[k] for r in dc], w * 0.92, color=C[k], zorder=3)
for i, r in enumerate(dc):
    sp_gt = r["gigatoken"] / r["tuetoken"]
    top = max(r[k] for k in keys_d)
    ax.text(x[i], top * 1.22, f"{fmt_x(sp_gt)} vs gt", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=C["tuetoken"])
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
all_d = [r[k] for r in dc for k in keys_d]
ax.set_ylim(min(all_d) * 0.45, max(all_d) * 3.5)
ax.set_xticks(x, [f"{r['tokens']:,} tok" for r in dc], fontsize=11)
ax.set_ylabel("Latency p50 (µs, log · lower is better)")
ax.set_xlabel("Sequence length (Python list[int]; tuetoken is also 2.7–2.8× faster on np.uint32)")
style(ax, "(d) Decode latency (p50, o200k_base)")

# ── (e) Padded Batches, Early-Exit Truncation, Whisper & Offsets ───────────
ax = axs[4]
bt = R["batch_trunc"]
ex = R["extras"]
rows_e = [
    ("50 MB trunc (vs HF)", bt["trunc_50mb_ms"]["hf"] / bt["trunc_50mb_ms"]["tuetoken_right"], C["hf"]),
    ("Whisper decode (vs HF)", ex["whisper_batch_500_ms"]["hf"] / ex["whisper_batch_500_ms"]["tuetoken"], C["hf"]),
    ("1 MB offsets (vs HF)", ex["offsets_1mb_ms"]["hf"] / ex["offsets_1mb_ms"]["tuetoken_buf"], C["hf"]),
    ("o200k startup (vs gt)", st[0]["gigatoken"] / st[0]["tuetoken"], C["gigatoken"]),
    ("10K padded (vs gt)", bt["padded_10k_mbps"]["tuetoken"] / bt["padded_10k_mbps"]["gigatoken"], C["gigatoken"]),
    ("Gemma-3 encode (vs gt)", fr[3]["tuetoken"] / fr[3]["gigatoken"], C["gigatoken"]),
    ("50 MB trunc (vs gt)", bt["trunc_50mb_ms"]["gigatoken"] / bt["trunc_50mb_ms"]["tuetoken_right"], C["gigatoken"]),
    ("100 MB warm (vs gt)", R["corpus"]["warm_100mb"]["tuetoken"] / R["corpus"]["warm_100mb"]["gigatoken"], C["gigatoken"]),
]
ye = np.arange(len(rows_e))
for i, (lbl, sp, col) in enumerate(rows_e):
    ax.barh(i, sp, 0.62, color=col, zorder=3)
    ax.text(sp * 1.15, i, fmt_x(sp), va="center", fontsize=11, fontweight="bold", color=col)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xlim(1, max(r[1] for r in rows_e) * 5.2)
ax.set_yticks(ye, [r[0] for r in rows_e], fontsize=10.5, color=INK)
ax.invert_yaxis()
ax.set_xlabel("Speedup (log · red: vs gigatoken (gt) · orange: vs HuggingFace)")
style(ax, "(e) Workload speedup summary", grid_axis="x")

# ── (f) Headline Summary Card ──────────────────────────────────────────────
ax = axs[5]
ax.set_facecolor(CARD)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_color(GRID)
    ax.spines[s].set_linewidth(1.2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("(f) Peak performance summary", loc="left", fontsize=14.5, fontweight="bold", color=INK, pad=12)

warm_tue = R["corpus"]["warm_100mb"]["tuetoken"]
warm_gt = R["corpus"]["warm_100mb"]["gigatoken"]
ax.text(0.5, 0.77, fmt_rate(warm_tue), ha="center", va="center", fontsize=64,
        fontweight="bold", color=C["tuetoken"], transform=ax.transAxes)
ax.text(0.5, 0.59, f"o200k_base warm ({CORES} cores) · {fmt_rate(tue_cold)} cold 1 GB",
        ha="center", va="center", fontsize=13.5, color=MUTED, transform=ax.transAxes)
ax.plot([0.07, 0.93], [0.49, 0.49], color=GRID, lw=1.4, transform=ax.transAxes)

tiles = [
    (fmt_x(tue_cold / gt_cold), "1 GB cold\nvs gigatoken"),
    (fmt_x(warm_tue / warm_gt), "100 MB warm\nvs gigatoken"),
    (fmt_x(st[0]["gigatoken"] / st[0]["tuetoken"]), "o200k startup\nvs gigatoken"),
    (fmt_x(bt["trunc_50mb_ms"]["hf"] / bt["trunc_50mb_ms"]["tuetoken_right"]), "50 MB trunc=512\nvs HuggingFace"),
]
for i, (val, lab) in enumerate(tiles):
    col_i = i % 2
    row_i = i // 2
    xx = 0.27 if col_i == 0 else 0.73
    yy = 0.34 if row_i == 0 else 0.12
    ax.text(xx, yy + 0.04, val, ha="center", va="center", fontsize=28,
            fontweight="bold", color=C["tuetoken"], transform=ax.transAxes)
    ax.text(xx, yy - 0.065, lab, ha="center", va="center", fontsize=11,
            color=MUTED, transform=ax.transAxes)

out = os.path.join(ROOT, "bench_tokenizers.png")
fig.savefig(out, dpi=150, facecolor=BG)
print("Saved", out)
