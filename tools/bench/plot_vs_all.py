"""Render publication-grade bench_tokenizers.png from tools/bench/results_vs_all.json.

    python tools/bench/plot_vs_all.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(HERE, "results_vs_all.json")))
META = R["meta"]
CORES = META["cores"]

# ── Editorial Color System ────────────────────────────────────────────────
# One signature cobalt for tuetoken, one warm vermilion for gigatoken,
# and three calm graduated cool-slate tones for older baselines.
BG = "#FFFFFF"
STRIP_BG = "#F8FAFC"
INK = "#0F172A"
SUB = "#475569"
MUTED = "#64748B"
RULE = "#E2E8F0"
GRID = "#F1F5F9"

C = {
    "tuetoken": "#1D4ED8",   # signature royal cobalt
    "gigatoken": "#E05A2B",  # refined warm vermilion
    "fastokens": "#475569",  # deep slate
    "tiktoken": "#94A3B8",   # medium slate
    "hf": "#CBD5E1",         # light stone
}
EDGE = {
    "tuetoken": "#1D4ED8",
    "gigatoken": "#E05A2B",
    "fastokens": "#475569",
    "tiktoken": "#64748B",
    "hf": "#94A3B8",
}
LABEL = {
    "tuetoken": f"tuetoken {META['versions']['tuetoken']}",
    "gigatoken": f"gigatoken {META['versions']['gigatoken']}",
    "fastokens": f"fastokens {META['versions']['fastokens']}",
    "tiktoken": f"tiktoken {META['versions']['tiktoken']}",
    "hf": f"HF tokenizers {META['versions']['tokenizers']}",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "SF Pro Text", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9.5,
    "text.color": INK,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": RULE,
    "axes.labelcolor": SUB,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "axes.linewidth": 0.8,
})


def fmt_x(v):
    if v >= 1000:
        return f"{v:,.0f}×"
    if v >= 10:
        return f"{v:.1f}×" if v < 12 else f"{v:.0f}×"
    return f"{v:.1f}×"


def style_ax(ax, tag, title, subtitle, grid_axis="y"):
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.95)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=GRID, lw=0.95)
        ax.yaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis == "x":
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color(RULE)
    else:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(length=0, pad=6, labelsize=9.2)
    ax.text(
        0.0, 1.12, f"{tag}  {title}",
        transform=ax.transAxes, fontsize=12.4, fontweight="bold", color=INK,
        va="bottom", ha="left",
    )
    ax.text(
        0.0, 1.048, subtitle,
        transform=ax.transAxes, fontsize=9.0, color=MUTED,
        va="bottom", ha="left",
    )


fig = plt.figure(figsize=(16.5, 11.4), facecolor=BG)

MARGIN_L = 0.040
MARGIN_R = 0.960
SPAN_W = MARGIN_R - MARGIN_L

# Subtle top editorial rule
fig.patches.append(mpatches.Rectangle(
    (MARGIN_L, 0.985), SPAN_W, 0.0036,
    transform=fig.transFigure, facecolor=C["tuetoken"], edgecolor="none",
))

# ── Header ────────────────────────────────────────────────────────────────
t0 = fig.text(MARGIN_L, 0.946, "tuetoken", fontsize=22, fontweight="bold", color=C["tuetoken"], va="center")
fig.canvas.draw()
bb0 = t0.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(bb0.x1 + 0.007, 0.946, "—  Head-to-Head BPE Tokenizer Benchmark",
         fontsize=21.5, fontweight="bold", color=INK, va="center")

fig.text(
    MARGIN_L, 0.912,
    "Bit-for-bit identical token IDs verified across all five libraries  ·  Cold 1 GB runs use a fresh OS process per data point",
    fontsize=10.2, color=SUB, va="center",
)

fig.text(
    MARGIN_R, 0.952,
    f"{META['cpu']} ({CORES} cores)  ·  Python {META['python']}",
    fontsize=10.2, fontweight="bold", color=INK, ha="right", va="center",
)
fig.text(
    MARGIN_R, 0.927,
    "All axes linear  ·  Also verified on Linux AMD EPYC Zen 4 (AVX-512)",
    fontsize=8.8, color=MUTED, ha="right", va="center",
)

# Horizontal legend bar
lx = MARGIN_L
ly = 0.875
for k in ("tuetoken", "gigatoken", "fastokens", "tiktoken", "hf"):
    fig.patches.append(mpatches.FancyBboxPatch(
        (lx, ly - 0.0055), 0.011, 0.011,
        boxstyle="round,pad=0.0012", transform=fig.transFigure,
        facecolor=C[k], edgecolor=EDGE[k], linewidth=0.8,
    ))
    t = fig.text(
        lx + 0.0155, ly, LABEL[k],
        fontsize=9.6, color=INK if k in ("tuetoken", "gigatoken") else SUB,
        fontweight="bold" if k == "tuetoken" else "normal",
        ha="left", va="center",
    )
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    lx = bb.x1 + 0.025

fig.add_artist(mlines.Line2D(
    [MARGIN_L, MARGIN_R], [0.852, 0.852],
    transform=fig.transFigure, color=RULE, linewidth=0.9,
))

# ── Main 2×2 Grid (aligned Swiss margins so y-labels never clip!) ─────────
gs = fig.add_gridspec(
    2, 2,
    left=0.105, right=MARGIN_R, top=0.790, bottom=0.258,
    hspace=0.47, wspace=0.26,
)

# ── (A) Cold 1 GB Corpus Scaling (Linear GB/s) ────────────────────────────
ax = fig.add_subplot(gs[0, 0])
cold = R["corpus"]["cold"]
gb = R["corpus"]["bytes"] / (1 << 30)

th_t = np.array([p["threads"] for p in cold["tuetoken"]])
gb_t = np.array([p["mbps"] / 1000.0 for p in cold["tuetoken"]])
th_g = np.array([p["threads"] for p in cold["gigatoken"]])
gb_g = np.array([p["mbps"] / 1000.0 for p in cold["gigatoken"]])

ax.fill_between(th_t, gb_g, gb_t, color=C["tuetoken"], alpha=0.075, zorder=2)
ax.plot(th_t, gb_t, "-", color=C["tuetoken"], lw=2.8, marker="o", ms=7.0,
        mfc=BG, mec=C["tuetoken"], mew=2.3, zorder=5)
ax.plot(th_g, gb_g, "-", color=C["gigatoken"], lw=2.2, marker="o", ms=6.0,
        mfc=C["gigatoken"], mec=BG, mew=1.2, zorder=4)

# Direct endpoint labels
ax.annotate(
    f"{gb_t[-1]:.2f} GB/s", (th_t[-1], gb_t[-1]),
    xytext=(10, 0), textcoords="offset points",
    fontsize=10.2, fontweight="bold", color=C["tuetoken"], va="center",
)
ax.annotate(
    f"{gb_g[-1]:.2f} GB/s", (th_g[-1], gb_g[-1]),
    xytext=(10, 0), textcoords="offset points",
    fontsize=9.6, fontweight="bold", color=C["gigatoken"], va="center",
)

# Clean dimension bracket near 14T with text seated comfortably in open shaded wedge
x_br = 13.35
y_hi = float(np.interp(x_br, th_t, gb_t)) - 0.10
y_lo = float(np.interp(x_br, th_g, gb_g)) + 0.10
ax.annotate(
    "", xy=(x_br, y_hi), xytext=(x_br, y_lo),
    arrowprops=dict(arrowstyle="<->", color=C["tuetoken"], lw=1.15, alpha=0.8),
    zorder=4,
)
ax.text(
    x_br - 0.35, 0.5 * (y_hi + y_lo),
    f"{gb_t[-1] / gb_g[-1]:.1f}× faster\nat {CORES}T cold",
    fontsize=9.2, fontweight="bold", color=C["tuetoken"], va="center", ha="right",
)

# fastokens, tiktoken, HF at 14T
if cold.get("fastokens"):
    fp = cold["fastokens"][-1]
    ax.scatter([fp["threads"]], [fp["mbps"] / 1000.0], marker="D", s=46,
               color=C["fastokens"], edgecolor=BG, linewidth=0.9, zorder=4)
    ax.annotate(
        f"fastokens  {fp['mbps']:.0f} MB/s", (fp["threads"], fp["mbps"] / 1000.0),
        xytext=(10, 0), textcoords="offset points",
        fontsize=8.7, color=C["fastokens"], va="center",
    )

tk_14 = cold["tiktoken"][-1]["mbps"]
hf_14 = cold["hf"][-1]["mbps"]
ax.plot([p["threads"] for p in cold["tiktoken"]], [p["mbps"] / 1000.0 for p in cold["tiktoken"]],
        "--", color=C["tiktoken"], lw=1.25, zorder=3)
ax.scatter([CORES], [tk_14 / 1000.0], marker="s", s=34, color=C["tiktoken"],
           edgecolor=EDGE["tiktoken"], linewidth=0.8, zorder=4)
ax.scatter([CORES], [hf_14 / 1000.0], marker="^", s=36, color=C["hf"],
           edgecolor=EDGE["hf"], linewidth=0.8, zorder=4)
ax.annotate(
    f"tiktoken {tk_14:.0f}  ·  HF {hf_14:.0f} MB/s",
    (CORES, tk_14 / 1000.0), xytext=(10, 4), textcoords="offset points",
    fontsize=8.4, color=MUTED, va="center",
)

ax.set_xticks(th_t)
ax.set_xlim(0.2, CORES * 1.48)
ax.set_ylim(-0.08, 5.15)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f} GB/s" if v > 0 else "0"))
ax.set_xlabel("Threads", fontsize=9.2, labelpad=6)
style_ax(ax, "A", f"Cold {gb:.0f} GB corpus scaling", "o200k_base · fresh OS process per point · higher is better", grid_axis="y")

# ── (B) Frontier LLM Throughput (Linear GB/s, aligned vs gt column) ───────
ax = fig.add_subplot(gs[0, 1])
fr = sorted(R["frontier"], key=lambda r: r["tuetoken"], reverse=True)
y = np.arange(len(fr))
bh = 0.34

t_gb = [r["tuetoken"] / 1000.0 for r in fr]
g_gb = [(r["gigatoken"] or 0.0) / 1000.0 for r in fr]

ax.barh(y - bh / 2 - 0.015, t_gb, bh, color=C["tuetoken"], zorder=3)
ax.barh(y + bh / 2 + 0.015, g_gb, bh, color=C["gigatoken"], zorder=3)

# Clean right-aligned "vs gt" speedup column matching Panel C!
COL_X = 11.45
ax.text(COL_X, -0.78, "vs gt", fontsize=8.5, fontweight="bold", color=MUTED, ha="right", va="bottom")

for i, r in enumerate(fr):
    tv = r["tuetoken"] / 1000.0
    gv = r["gigatoken"]
    ax.text(
        tv + 0.14, i - bh / 2 - 0.015, f"{tv:.2f} GB/s",
        va="center", ha="left", fontsize=9.1, fontweight="bold", color=INK,
    )
    if gv is not None:
        gv_gb = gv / 1000.0
        sp = r["tuetoken"] / gv
        ax.text(
            gv_gb + 0.14, i + bh / 2 + 0.015, f"{gv_gb:.2f}",
            va="center", ha="left", fontsize=8.4, color=C["gigatoken"],
        )
        ax.text(
            COL_X, i, fmt_x(sp),
            va="center", ha="right", fontsize=9.4, fontweight="bold", color=C["tuetoken"],
        )
    else:
        ax.text(
            0.14, i + bh / 2 + 0.015, "unsupported in gigatoken",
            va="center", ha="left", fontsize=8.2, style="italic", color=MUTED,
        )
        ax.text(
            COL_X, i, "—",
            va="center", ha="right", fontsize=9.2, color="#94A3B8",
        )

ax.set_yticks(y, [r["model"] for r in fr], fontsize=9.4, color=INK)
ax.invert_yaxis()
ax.set_ylim(len(fr) - 0.45, -0.92)
ax.set_xlim(0, 11.75)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f} GB/s" if 0 < v <= 10 else ("0" if v == 0 else "")))
ax.set_xlabel("Throughput (GB/s)", fontsize=9.2, labelpad=6)
hf_min = min(r["hf"] for r in fr)
hf_max = max(r["hf"] for r in fr)
style_ax(
    ax, "B", "Frontier LLM throughput",
    f"25 MB OpenWebText · {CORES} cores · HF baseline: {hf_min:.0f}–{hf_max:.0f} MB/s (59–927× slower)",
    grid_axis="x",
)

# ── (C) Startup Latency — Horizontal Dumbbell Chart (Linear ms) ───────────
ax = fig.add_subplot(gs[1, 0])
st = sorted(R["startup"], key=lambda r: r["tuetoken"])
yc = np.arange(len(st))

for i, r in enumerate(st):
    tv, gv, hv = r["tuetoken"], r["gigatoken"], r["hf"]
    hi = max(gv, hv)
    ax.hlines(i, tv, hi, color="#E2E8F0", lw=2.8, zorder=2)
    ax.hlines(i, tv, gv, color="#CBD5E1", lw=2.8, zorder=2.5)

ax.scatter([r["hf"] for r in st], yc, s=62, color=C["hf"],
           edgecolor=EDGE["hf"], linewidth=1.1, zorder=3.5)
ax.scatter([r["gigatoken"] for r in st], yc, s=78, color=C["gigatoken"],
           edgecolor=BG, linewidth=1.3, zorder=4.5)
ax.scatter([r["tuetoken"] for r in st], yc, s=90, color=C["tuetoken"],
           edgecolor=BG, linewidth=1.4, zorder=5.5)

COL_C = 392
ax.text(COL_C, -0.78, "vs gt", fontsize=8.5, fontweight="bold", color=MUTED, ha="right", va="bottom")
for i, r in enumerate(st):
    tv, gv = r["tuetoken"], r["gigatoken"]
    # Crisp white halo breaks the gray connector line cleanly around the label
    ax.text(
        tv + 10, i, f"{tv:.1f} ms",
        va="center", ha="left", fontsize=8.8, fontweight="bold", color=C["tuetoken"],
        bbox=dict(facecolor=BG, edgecolor="none", pad=1.4), zorder=5,
    )
    # Uniformly place gigatoken load time cleanly above every orange dot
    ax.text(
        gv, i - 0.30, f"{gv:.0f} ms",
        va="bottom", ha="center", fontsize=8.0, color=C["gigatoken"],
    )
    ax.text(
        COL_C, i, fmt_x(gv / tv),
        va="center", ha="right", fontsize=9.4, fontweight="bold", color=C["tuetoken"],
    )

ax.set_yticks(yc, [r["model"] for r in st], fontsize=9.4, color=INK)
ax.invert_yaxis()
ax.set_ylim(len(st) - 0.45, -0.92)
ax.set_xlim(0, 404)
ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f} ms" if 0 < v <= 300 else ("0" if v == 0 else "")))
ax.set_xlabel("Cold load time (ms)", fontsize=9.2, labelpad=6)
style_ax(
    ax, "C", "Startup latency (tokenizer.json load)",
    "Median load time · ms · lower is better  (tuetoken loads in 7–20 ms)",
    grid_axis="x",
)

# ── (D) Decode Throughput (Linear M tokens/sec) ───────────────────────────
ax = fig.add_subplot(gs[1, 1])
dc = R["decode"]
xd = np.arange(len(dc))
keys_d = ("tuetoken", "gigatoken", "tiktoken", "fastokens", "hf")
w = 0.155

for j, k in enumerate(keys_d):
    vals = [r["tokens"] / r[k] for r in dc]  # tokens / us == M tokens/sec
    ax.bar(
        xd + (j - 2) * w, vals, w * 0.88,
        color=C[k], edgecolor=EDGE[k] if k == "hf" else "none", linewidth=0.6, zorder=3,
    )

for i, r in enumerate(dc):
    t_rate = r["tokens"] / r["tuetoken"]
    g_rate = r["tokens"] / r["gigatoken"]
    sp_gt = r["gigatoken"] / r["tuetoken"]
    lat_str = f"{r['tuetoken']:.2f} µs" if r["tuetoken"] < 1 else f"{r['tuetoken']:.1f} µs"
    ax.text(
        xd[i] - 2 * w, t_rate + 7,
        f"{t_rate:.0f}M\n({lat_str})",
        ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=C["tuetoken"],
        linespacing=1.18,
    )
    # Consistent cobalt color grammar for tuetoken speedup callouts
    ax.text(
        xd[i] + 0.08, g_rate + 14,
        f"{fmt_x(sp_gt)} vs gt",
        ha="center", va="bottom", fontsize=8.7, fontweight="bold", color=C["tuetoken"],
    )

ax.set_xticks(xd, [f"{r['tokens']:,} tokens" for r in dc], fontsize=9.5, color=INK)
ax.set_ylim(0, 305)
ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M" if v > 0 else "0"))
ax.set_xlabel("Sequence length  (Python list[int]  ·  also 2.7–2.8× faster than gt on np.uint32)",
              fontsize=8.6, labelpad=6)
style_ax(
    ax, "D", "Decode throughput (o200k_base)",
    "Derived from p50 latency · M tokens/sec · higher is better",
    grid_axis="y",
)

# ── (E) Bottom Production Workloads Strip (4 Symmetric Visual Columns) ────
strip_y0, strip_h = 0.056, 0.138
fig.patches.append(mpatches.FancyBboxPatch(
    (MARGIN_L, strip_y0), SPAN_W, strip_h,
    boxstyle="round,pad=0.004,rounding_size=0.008",
    transform=fig.transFigure, facecolor=STRIP_BG, edgecolor=RULE, linewidth=0.9,
))

fig.text(
    MARGIN_L, strip_y0 + strip_h + 0.012,
    "E  Production & serving workloads",
    fontsize=11.6, fontweight="bold", color=INK, va="bottom",
)
fig.text(
    MARGIN_R, strip_y0 + strip_h + 0.012,
    f"Same verified {META['cpu']} ({CORES} cores) setup  ·  AutoTokenizer & batch APIs",
    fontsize=8.7, color=MUTED, ha="right", va="bottom",
)

bt = R["batch_trunc"]
ex = R["extras"]
warm = R["corpus"]["warm_100mb"]

cards = [
    {
        "title": "Warm steady-state · 100 MB (higher better)",
        "big": f"{warm['tuetoken'] / 1000:.2f} GB/s",
        "badge": f"{warm['tuetoken'] / warm['gigatoken']:.1f}× vs gt",
        "bars": [
            ("tuetoken", warm["tuetoken"], C["tuetoken"], f"{warm['tuetoken']/1000:.2f} GB/s"),
            ("gigatoken", warm["gigatoken"], C["gigatoken"], f"{warm['gigatoken']/1000:.2f} GB/s"),
            ("fastokens", warm["fastokens"], C["fastokens"], f"{warm['fastokens']/1000:.2f} GB/s"),
        ],
    },
    {
        "title": "10K docs padded · max=512 (higher better)",
        "big": f"{bt['padded_10k_mbps']['tuetoken'] / 1000:.2f} GB/s",
        "badge": f"{bt['padded_10k_mbps']['tuetoken'] / bt['padded_10k_mbps']['gigatoken']:.1f}× vs gt",
        "bars": [
            ("tuetoken", bt["padded_10k_mbps"]["tuetoken"], C["tuetoken"], f"{bt['padded_10k_mbps']['tuetoken']/1000:.2f} GB/s"),
            ("gigatoken", bt["padded_10k_mbps"]["gigatoken"], C["gigatoken"], f"{bt['padded_10k_mbps']['gigatoken']/1000:.2f} GB/s"),
        ],
        "note": "o200k_base · pad_id=0 · 2D NumPy array output",
    },
    {
        "title": "50 MB truncation · max=512 (lower better)",
        "big": f"{bt['trunc_50mb_ms']['tuetoken_right']:.2f} ms",
        "badge": f"{bt['trunc_50mb_ms']['gigatoken'] / bt['trunc_50mb_ms']['tuetoken_right']:.1f}× vs gt · {fmt_x(bt['trunc_50mb_ms']['hf'] / bt['trunc_50mb_ms']['tuetoken_right'])} vs HF",
        "bars": [
            ("tuetoken R/L", bt["trunc_50mb_ms"]["tuetoken_right"], C["tuetoken"],
             f"{bt['trunc_50mb_ms']['tuetoken_right']:.2f} / {bt['trunc_50mb_ms']['tuetoken_left']:.2f} ms"),
            ("gigatoken R", bt["trunc_50mb_ms"]["gigatoken"], C["gigatoken"],
             f"{bt['trunc_50mb_ms']['gigatoken']:.1f} ms (R only)"),
        ],
        "note": f"HF transformers: {bt['trunc_50mb_ms']['hf']:,.0f} ms (Qwen2.5, 1,000 × 50 KB)",
    },
    {
        "title": "Whisper timestamps & 1 MB byte offsets",
        "big": f"{ex['whisper_batch_500_ms']['hf'] / ex['whisper_batch_500_ms']['tuetoken']:.0f}× / {ex['offsets_1mb_ms']['hf'] / ex['offsets_1mb_ms']['tuetoken_buf']:.0f}×",
        "badge": "vs HF",
        "rows": [
            ("Whisper batch 500", f"{ex['whisper_batch_500_ms']['tuetoken']:.3f} ms",
             f"vs {ex['whisper_batch_500_ms']['hf']:.1f} ms HF",
             f"{ex['whisper_batch_500_ms']['hf'] / ex['whisper_batch_500_ms']['tuetoken']:.0f}×"),
            ("1 MB byte offsets", f"{ex['offsets_1mb_ms']['tuetoken_buf']:.2f} ms",
             f"vs {ex['offsets_1mb_ms']['hf']:.0f} ms HF",
             f"{ex['offsets_1mb_ms']['hf'] / ex['offsets_1mb_ms']['tuetoken_buf']:.0f}×"),
        ],
        "note": "Zero-copy Arrow/NumPy offsets & native timestamp decode",
    },
]

col_w = SPAN_W / 4.0
for idx, card in enumerate(cards):
    x0 = MARGIN_L + idx * col_w
    pad_x = 0.014
    if idx > 0:
        fig.add_artist(mlines.Line2D(
            [x0, x0], [strip_y0 + 0.015, strip_y0 + strip_h - 0.015],
            transform=fig.transFigure, color=RULE, linewidth=0.85,
        ))
    fig.text(x0 + pad_x, strip_y0 + strip_h - 0.014, card["title"],
             fontsize=8.4, fontweight="bold", color=SUB, va="top")
    tb = fig.text(x0 + pad_x, strip_y0 + strip_h - 0.036, card["big"],
                  fontsize=15.0, fontweight="bold", color=C["tuetoken"], va="top")
    fig.canvas.draw()
    bb = tb.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.006, strip_y0 + strip_h - 0.042, card["badge"],
             fontsize=8.2, fontweight="bold", color=INK if "vs gt" in card["badge"] else MUTED, va="top")

    if "bars" in card:
        max_v = max(v for _, v, _, _ in card["bars"])
        lbl_w = 0.064
        bar_max_w = col_w - 2 * pad_x - lbl_w - 0.066
        y_top = strip_y0 + 0.056
        step = 0.018 if len(card["bars"]) == 3 else 0.022
        for bi, (lbl, val, col, vtxt) in enumerate(card["bars"]):
            by = y_top - bi * step
            fig.text(x0 + pad_x, by, lbl, fontsize=8.0, color=SUB, va="center")
            bw = max(0.007, bar_max_w * (val / max_v))
            fig.patches.append(mpatches.FancyBboxPatch(
                (x0 + pad_x + lbl_w, by - 0.0040), bw, 0.0080,
                boxstyle="round,pad=0.0005,rounding_size=0.002",
                transform=fig.transFigure, facecolor=col, edgecolor="none",
            ))
            fig.text(x0 + pad_x + lbl_w + bw + 0.005, by, vtxt,
                     fontsize=7.9, fontweight="bold" if bi == 0 else "normal",
                     color=INK if bi == 0 else MUTED, va="center")
        if "note" in card:
            fig.text(x0 + pad_x, strip_y0 + 0.014, card["note"],
                     fontsize=7.6, color=MUTED, va="center")
    elif "rows" in card:
        y_top = strip_y0 + 0.056
        for ri, (name, fast, slow, sp) in enumerate(card["rows"]):
            ry = y_top - ri * 0.022
            fig.text(x0 + pad_x, ry, name, fontsize=8.1, color=SUB, va="center")
            t_fast = fig.text(x0 + pad_x + 0.088, ry, fast, fontsize=8.3,
                              fontweight="bold", color=C["tuetoken"], va="center")
            fig.canvas.draw()
            bbf = t_fast.get_window_extent().transformed(fig.transFigure.inverted())
            fig.text(bbf.x1 + 0.004, ry, slow, fontsize=7.9, color=MUTED, va="center")
            fig.text(x0 + col_w - pad_x, ry, sp, fontsize=8.3,
                     fontweight="bold", color=C["tuetoken"], ha="right", va="center")
        if "note" in card:
            fig.text(x0 + pad_x, strip_y0 + 0.014, card["note"],
                     fontsize=7.6, color=MUTED, va="center")

# ── Footer ────────────────────────────────────────────────────────────────
fig.text(
    MARGIN_L, 0.019,
    "Corpus: OpenWebText (1.00 GB)  ·  OpenAI o200k_base & HuggingFace tokenizer.json  ·  Raw data: tools/bench/results_vs_all.json",
    fontsize=8.3, color=MUTED, va="center",
)
fig.text(
    MARGIN_R, 0.019,
    "Reproduce: python tools/bench/bench_vs_all.py corpus_1gb.txt && python tools/bench/plot_vs_all.py",
    fontsize=8.3, color=MUTED, ha="right", va="center",
)

out = os.path.join(ROOT, "bench_tokenizers.png")
fig.savefig(out, dpi=200, facecolor=BG)
print("Saved", out)
