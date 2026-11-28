"""Render minimalist 16:9 benchmark figure (SVG + 2x PNG) from tools/bench/results_vs_all.json.

    python tools/bench/plot_vs_all.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(HERE, "results_vs_all.json")))
META = R["meta"]
CORES = META["cores"]


def setup_fonts():
    """Register true Regular / Medium / Bold Helvetica Neue faces on macOS if available."""
    ttc = "/System/Library/Fonts/HelveticaNeue.ttc"
    if os.path.exists(ttc):
        try:
            from fontTools.ttLib import TTCollection
            cache_dir = os.path.expanduser("~/.cache/tuetoken/fonts")
            os.makedirs(cache_dir, exist_ok=True)
            faces = [
                (0, "HelveticaNeue-Regular.ttf"),
                (10, "HelveticaNeue-Medium.ttf"),
                (1, "HelveticaNeue-Bold.ttf"),
            ]
            extracted = []
            ttc_obj = None
            for idx, fname in faces:
                out_p = os.path.join(cache_dir, fname)
                if not os.path.exists(out_p):
                    if ttc_obj is None:
                        ttc_obj = TTCollection(ttc)
                    ttc_obj.fonts[idx].save(out_p)
                extracted.append(out_p)
            for p in extracted:
                fm.fontManager.addfont(p)
            return ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
        except Exception:
            pass
    return ["Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


FONT_STACK = setup_fonts()

# ── Palette ───────────────────────────────────────────────────────────────
BG = "#FFFFFF"
INK = "#0F172A"
BLUE = "#1D4ED8"
ORANGE = "#E4572E"
GRAY_BAR = "#CBD5E1"
TICK = "#64748B"
BASELINE = "#E2E8F0"
GRID = "#EEF2F7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "svg.fonttype": "none",
    "font.size": 9.5,
    "text.color": INK,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": TICK,
    "xtick.color": TICK,
    "ytick.color": TICK,
    "axes.linewidth": 0.8,
})

W_REG = "normal"
W_SEMI = "medium"
W_BOLD = "bold"


def draw_rounded_barh(ax, y_center, width, height, color, radius_pt=3.0, zorder=3):
    """Horizontal bar flush on x=0 with rounded right corners (exact radius in points)."""
    if width <= 0:
        return
    fig = ax.figure
    bbox = ax.get_position()
    ax_w_pt = bbox.width * fig.get_figwidth() * 72.0
    ax_h_pt = bbox.height * fig.get_figheight() * 72.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx_per_pt = abs(x1 - x0) / ax_w_pt
    dy_per_pt = abs(y1 - y0) / ax_h_pt

    half_h = height / 2.0
    ry = min(radius_pt * dy_per_pt, half_h)
    rx = min(radius_pt * dx_per_pt, width * 0.48)
    k = 0.55228475
    yb, yt = y_center - half_h, y_center + half_h

    verts = [
        (0.0, yb),
        (width - rx, yb),
        (width - rx * (1 - k), yb),
        (width, yb + ry * (1 - k)),
        (width, yb + ry),
        (width, yt - ry),
        (width, yt - ry * (1 - k)),
        (width - rx * (1 - k), yt),
        (width - rx, yt),
        (0.0, yt),
        (0.0, yb),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def draw_rounded_barv(ax, x_center, height, width, color, radius_pt=2.8, zorder=3):
    """Vertical bar flush on y=0 with rounded top corners (exact radius in points)."""
    if height <= 0:
        return
    fig = ax.figure
    bbox = ax.get_position()
    ax_w_pt = bbox.width * fig.get_figwidth() * 72.0
    ax_h_pt = bbox.height * fig.get_figheight() * 72.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx_per_pt = abs(x1 - x0) / ax_w_pt
    dy_per_pt = abs(y1 - y0) / ax_h_pt

    half_w = width / 2.0
    rx = min(radius_pt * dx_per_pt, half_w)
    ry = min(radius_pt * dy_per_pt, height * 0.48)
    k = 0.55228475
    xl, xr = x_center - half_w, x_center + half_w

    verts = [
        (xl, 0.0),
        (xl, height - ry),
        (xl, height - ry * (1 - k)),
        (xl + rx * (1 - k), height),
        (xl + rx, height),
        (xr - rx, height),
        (xr - rx * (1 - k), height),
        (xr, height - ry * (1 - k)),
        (xr, height - ry),
        (xr, 0.0),
        (xl, 0.0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def clean_spines(ax, baseline="bottom", hgrid=False):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s == baseline)
    ax.spines[baseline].set_color(BASELINE)
    ax.spines[baseline].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    if hgrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    else:
        ax.yaxis.grid(False)
    ax.tick_params(length=0, pad=7, labelsize=9.0, colors=TICK)


# ── Canvas (16:9, pure white, generous margins) ───────────────────────────
fig = plt.figure(figsize=(16.0, 9.0), facecolor=BG)

ML, MR = 0.048, 0.955

# Header: Title (24pt bold) + one subtitle line only
fig.text(ML, 0.922, "tuetoken: BPE tokenizer benchmark",
         fontsize=24, fontweight=W_BOLD, color=INK, va="center")
fig.text(ML, 0.873, "Identical token IDs across all five libraries",
         fontsize=11.5, fontweight=W_REG, color=TICK, va="center")

# ── Layout Coordinates ────────────────────────────────────────────────────
CHART_TOP = 0.778
CHART_BOT = 0.258

# 1) HERO PANEL: Cold 1 GB corpus scaling (~45% of width including direct labels)
# Axes stop cleanly at 14 threads so gridlines & baseline don't run under labels.
ax_hero = fig.add_axes([0.088, CHART_BOT, 0.285, CHART_TOP - CHART_BOT])
cold = R["corpus"]["cold"]
th_t = np.array([p["threads"] for p in cold["tuetoken"]])
gb_t = np.array([p["mbps"] / 1000.0 for p in cold["tuetoken"]])
th_g = np.array([p["threads"] for p in cold["gigatoken"]])
gb_g = np.array([p["mbps"] / 1000.0 for p in cold["gigatoken"]])

ax_hero.set_xlim(0.5, 14.35)
ax_hero.set_ylim(0.0, 4.88)
ax_hero.set_xticks(th_t)
ax_hero.set_yticks([0, 1, 2, 3, 4])
ax_hero.set_yticklabels(["0", "1 GB/s", "2 GB/s", "3 GB/s", "4 GB/s"])
ax_hero.set_xlabel("Threads", fontsize=9.5, fontweight=W_REG, color=TICK, labelpad=8)
clean_spines(ax_hero, baseline="bottom", hgrid=True)

ax_hero.fill_between(th_t, gb_g, gb_t, color=BLUE, alpha=0.085, zorder=2)
ax_hero.plot(th_t, gb_t, color=BLUE, lw=3.0, solid_capstyle="round", zorder=4)
ax_hero.plot(th_g, gb_g, color=ORANGE, lw=1.75, solid_capstyle="round", zorder=3.5)
ax_hero.scatter([th_t[-1]], [gb_t[-1]], s=38, color=BLUE, zorder=5, clip_on=False)
ax_hero.scatter([th_g[-1]], [gb_g[-1]], s=26, color=ORANGE, zorder=4.5, clip_on=False)

# Direct end labels floating cleanly in whitespace right of x=14
lx_end = 14.75
ax_hero.text(lx_end, gb_t[-1] + 0.04, "4.50 GB/s", fontsize=11, fontweight=W_BOLD,
             color=BLUE, va="bottom", clip_on=False)
ax_hero.text(lx_end, gb_t[-1] - 0.05, "tuetoken", fontsize=9.5, fontweight=W_SEMI,
             color=BLUE, va="top", clip_on=False)

ax_hero.text(lx_end, gb_g[-1] + 0.04, "1.79 GB/s", fontsize=11, fontweight=W_SEMI,
             color=ORANGE, va="bottom", clip_on=False)
ax_hero.text(lx_end, gb_g[-1] - 0.05, "gigatoken", fontsize=9.5, fontweight=W_REG,
             color=ORANGE, va="top", clip_on=False)

# One gap annotation only, beside the gap, no arrow
ax_hero.text(
    lx_end, 0.5 * (gb_t[-1] + gb_g[-1]),
    "2.5× faster\nat 14 threads",
    fontsize=10.5, fontweight=W_SEMI, color=BLUE, va="center", ha="left",
    linespacing=1.25, clip_on=False,
)

# Single gray label near the baseline, no separate markers
ax_hero.hlines(0.58, 1.0, 14.0, color=GRAY_BAR, lw=1.15, linestyles="--", zorder=2.5)
ax_hero.text(14.0, 0.28, "others < 0.6 GB/s", fontsize=9.2, fontweight=W_REG,
             color=TICK, ha="right", va="center")

fig.text(0.088, CHART_TOP + 0.024, "Cold 1 GB corpus scaling",
         fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# 2) THROUGHPUT PANEL: LLM tokenizer throughput (top-right, wide)
RIGHT_AX_L = 0.578
ax_thr = fig.add_axes([RIGHT_AX_L, 0.542, MR - RIGHT_AX_L, CHART_TOP - 0.542])
fr = sorted(R["frontier"], key=lambda r: r["tuetoken"], reverse=True)
n_fr = len(fr)

ax_thr.set_xlim(0.0, 10.65)
ax_thr.set_ylim(n_fr - 0.45, -0.55)
ax_thr.set_yticks(np.arange(n_fr))
ax_thr.set_yticklabels([r["model"] for r in fr], fontsize=9.2, color=INK)
ax_thr.set_xticks([0, 3, 6, 9])
ax_thr.set_xticklabels(["0", "3 GB/s", "6 GB/s", "9 GB/s"])
clean_spines(ax_thr, baseline="left", hgrid=False)
ax_thr.tick_params(axis="y", pad=8)

# Consistent ~8.8pt visual blue bar thickness across Throughput and Decode
blue_h = 0.39
orange_h = 0.16
gap_bar = 0.05

for i, r in enumerate(fr):
    tv = r["tuetoken"] / 1000.0
    gv = r["gigatoken"]
    if gv is not None:
        y_blue = i - (orange_h + gap_bar) / 2.0
        y_org = i + (blue_h + gap_bar) / 2.0
        draw_rounded_barh(ax_thr, y_blue, tv, blue_h, BLUE, radius_pt=3.0)
        draw_rounded_barh(ax_thr, y_org, gv / 1000.0, orange_h, ORANGE, radius_pt=1.8)
    else:
        y_blue = float(i)
        draw_rounded_barh(ax_thr, y_blue, tv, blue_h, BLUE, radius_pt=3.0)
    lbl = f"{tv:.2f} GB/s" if i == 0 else f"{tv:.2f}"
    ax_thr.text(tv + 0.17, y_blue, lbl, fontsize=11, fontweight=W_SEMI, color=INK, va="center")

fig.text(RIGHT_AX_L, CHART_TOP + 0.024, "LLM tokenizer throughput",
         fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# 3) STARTUP PANEL: Dumbbell chart (bottom-right-left)
# Axes stop cleanly at 305 ms; right-side speedup labels float outside baseline.
ax_st = fig.add_axes([RIGHT_AX_L, CHART_BOT, 0.145, 0.445 - CHART_BOT])
st = sorted(R["startup"], key=lambda r: r["tuetoken"])
n_st = len(st)

ax_st.set_xlim(-6.0, 306.0)
ax_st.set_ylim(n_st - 0.45, -0.55)
ax_st.set_yticks(np.arange(n_st))
ax_st.set_yticklabels([r["model"] for r in st], fontsize=9.2, color=INK)
ax_st.set_xticks([0, 100, 200, 300])
ax_st.set_xticklabels(["0", "100 ms", "200 ms", "300 ms"])
clean_spines(ax_st, baseline="bottom", hgrid=False)
ax_st.tick_params(axis="y", pad=8)

for i, r in enumerate(st):
    ax_st.hlines(i, r["tuetoken"], r["gigatoken"], color=GRAY_BAR, lw=2.0, zorder=2)

ax_st.scatter([r["gigatoken"] for r in st], np.arange(n_st), s=44, color=ORANGE, zorder=4)
ax_st.scatter([r["tuetoken"] for r in st], np.arange(n_st), s=52, color=BLUE, zorder=5)

for i, r in enumerate(st):
    tv, gv = r["tuetoken"], r["gigatoken"]
    sp = round(gv / tv)
    blue_lbl = f"{tv:.1f} ms" if i == 0 else f"{tv:.1f}"
    ax_st.text(
        tv + 15, i, blue_lbl,
        fontsize=10.2, fontweight=W_SEMI, color=BLUE, va="center", ha="left",
        bbox=dict(facecolor=BG, edgecolor="none", pad=1.2), zorder=6,
    )
    ax_st.text(
        gv + 15, i, f"{sp}×",
        fontsize=11, fontweight=W_SEMI, color=INK, va="center", ha="left", clip_on=False,
    )

ax_st.text(0.0, 1.09, "Startup latency",
           transform=ax_st.transAxes, fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# 4) DECODE PANEL: Grouped bars (bottom-right-right)
DEC_L = 0.788
ax_dec = fig.add_axes([DEC_L, CHART_BOT, MR - DEC_L, 0.445 - CHART_BOT])
dc = R["decode"]
xd = np.arange(len(dc))
keys_d = ("tuetoken", "gigatoken", "tiktoken", "fastokens", "hf")
cols_d = (BLUE, ORANGE, GRAY_BAR, GRAY_BAR, GRAY_BAR)

ax_dec.set_xlim(-0.55, 2.65)
ax_dec.set_ylim(0.0, 305.0)
ax_dec.set_xticks(xd)
ax_dec.set_xticklabels(["50", "1,000", "12,000 tok"])
ax_dec.set_yticks([0, 100, 200, 300])
ax_dec.set_yticklabels(["0", "100M", "200M", "300M"])
clean_spines(ax_dec, baseline="bottom", hgrid=True)

# Match visual blue bar thickness (~8.8pt)
bar_w = 0.160
for j, (k, col) in enumerate(zip(keys_d, cols_d)):
    for i, r in enumerate(dc):
        rate = r["tokens"] / r[k]
        draw_rounded_barv(ax_dec, xd[i] + (j - 2) * bar_w, rate, bar_w * 0.85, col, radius_pt=2.8)

# Label ONLY the blue bars ("109M", "231M", "248M tokens/s") — zero overflow
for i, r in enumerate(dc):
    rate = r["tokens"] / r["tuetoken"]
    bx = xd[i] - 2 * bar_w
    if i == len(dc) - 1:
        ax_dec.text(xd[i] + 2.65 * bar_w, rate + 10, f"{rate:.0f}M tokens/s",
                    fontsize=10.8, fontweight=W_SEMI, color=BLUE, ha="right", va="bottom", clip_on=False)
    else:
        ax_dec.text(bx, rate + 10, f"{rate:.0f}M",
                    fontsize=11, fontweight=W_SEMI, color=BLUE, ha="center", va="bottom")

# Clean non-overlapping gray direct label over the 3 gray bars in group 1
ax_dec.text(xd[0] + 0.02, 33, "others",
            fontsize=8.2, fontweight=W_REG, color=TICK, ha="left", va="bottom")

ax_dec.text(0.0, 1.09, "Decode throughput",
            transform=ax_dec.transAxes, fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# ── Bottom Stat Strip (4 tiles, thin dividers, no boxes) ──────────────────
fig.add_artist(mlines.Line2D([ML, MR], [0.180, 0.180], transform=fig.transFigure, color=BASELINE, lw=0.9))

warm = R["corpus"]["warm_100mb"]
bt = R["batch_trunc"]
ex = R["extras"]

tiles = [
    (f"{warm['tuetoken'] / 1000:.2f} GB/s", "warm steady-state"),
    (f"{bt['padded_10k_mbps']['tuetoken'] / 1000:.2f} GB/s", "10K docs padded"),
    (f"{bt['trunc_50mb_ms']['tuetoken_right']:.2f} ms", "50 MB truncation"),
    (f"{ex['whisper_batch_500_ms']['hf'] / ex['whisper_batch_500_ms']['tuetoken']:.0f}×", "Whisper timestamps vs HF"),
]

span_w = MR - ML
tile_w = span_w / 4.0
for idx, (val, desc) in enumerate(tiles):
    tx = ML + idx * tile_w + (0.014 if idx > 0 else 0.0)
    if idx > 0:
        dx = ML + idx * tile_w
        fig.add_artist(mlines.Line2D([dx, dx], [0.088, 0.156], transform=fig.transFigure, color=BASELINE, lw=0.85))
    fig.text(tx, 0.138, val, fontsize=21, fontweight=W_BOLD, color=BLUE, va="center")
    fig.text(tx, 0.095, desc, fontsize=10.0, fontweight=W_REG, color=TICK, va="center")

# ── Single Footer Line ────────────────────────────────────────────────────
fig.text(
    ML, 0.032,
    f"{META['cpu']}, {CORES} cores · Python {META['python']} · OpenWebText 1 GB · details in README",
    fontsize=9.0, fontweight=W_REG, color=TICK, va="center",
)

out_png = os.path.join(ROOT, "bench_tokenizers.png")
out_svg = os.path.join(ROOT, "bench_tokenizers.svg")
fig.savefig(out_png, dpi=200, facecolor=BG)
fig.savefig(out_svg, facecolor=BG)
print("Saved", out_png)
print("Saved", out_svg)
