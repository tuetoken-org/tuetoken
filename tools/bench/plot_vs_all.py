"""Render minimalist 16:9 benchmark figure (SVG + 2x PNG) from tools/bench/results_vs_all.json.

    python tools/bench/plot_vs_all.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(HERE, "assets")
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
GRAY_LINE = "#94A3B8"
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

LOGO_OPENAI = np.asarray(Image.open(os.path.join(ASSETS, "openai.png")).convert("RGBA"))
LOGO_HF = np.asarray(Image.open(os.path.join(ASSETS, "huggingface.png")).convert("RGBA"))


def place_logo_data(ax, img, xy, size_pt=14.0, zorder=7):
    """Place a crisp RGBA logo centered at data coordinates (xy) with exact visual size in points."""
    # OffsetImage zoom=1 draws 1 image px = 1 pt at 72 dpi
    zoom = size_pt / img.shape[0]
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom, interpolation="lanczos"),
        xy, xycoords="data", frameon=False, pad=0.0, zorder=zorder,
        clip_on=False, annotation_clip=False,
    )
    ax.add_artist(ab)


def place_logo_fig(fig, img, xy_fig, size_pt=12.5, zorder=7):
    """Place a crisp RGBA logo centered at figure-fraction coordinates."""
    zoom = size_pt / img.shape[0]
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom, interpolation="lanczos"),
        xy_fig, xycoords=fig.transFigure, frameon=False, pad=0.0, zorder=zorder,
    )
    fig.add_artist(ab)


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

# 1) HERO PANEL: Cold 1 GB corpus scaling (LOG Y-AXIS so all 5 curves are visible!)
ax_hero = fig.add_axes([0.095, CHART_BOT, 0.260, CHART_TOP - CHART_BOT])
cold = R["corpus"]["cold"]
th_t = np.array([p["threads"] for p in cold["tuetoken"]])
mb_t = np.array([p["mbps"] for p in cold["tuetoken"]])
th_g = np.array([p["threads"] for p in cold["gigatoken"]])
mb_g = np.array([p["mbps"] for p in cold["gigatoken"]])
ft_14 = cold["fastokens"][-1]["mbps"]
th_tk = np.array([p["threads"] for p in cold["tiktoken"]])
mb_tk = np.array([p["mbps"] for p in cold["tiktoken"]])
th_hf = np.array([p["threads"] for p in cold["hf"]])
mb_hf = np.array([p["mbps"] for p in cold["hf"]])

ax_hero.set_yscale("log")
ax_hero.set_xlim(0.5, 14.35)
ax_hero.set_ylim(4.0, 7500.0)
ax_hero.set_xticks(th_t)
ax_hero.set_yticks([10, 100, 1000, 5000])
ax_hero.set_yticklabels(["10 MB/s", "100 MB/s", "1 GB/s", "5 GB/s"])
ax_hero.yaxis.set_minor_locator(mticker.NullLocator())
ax_hero.set_xlabel("Threads", fontsize=9.5, fontweight=W_REG, color=TICK, labelpad=8)
clean_spines(ax_hero, baseline="bottom", hgrid=True)

# Soft fill between tuetoken and gigatoken
ax_hero.fill_between(th_t, mb_g, mb_t, color=BLUE, alpha=0.09, zorder=2)

# All five library lines clearly visible on log scale!
ax_hero.plot(th_t, mb_t, color=BLUE, lw=3.0, solid_capstyle="round", zorder=5)
ax_hero.plot(th_g, mb_g, color=ORANGE, lw=1.8, solid_capstyle="round", zorder=4.5)
ax_hero.hlines(ft_14, 1.0, 14.0, color=GRAY_BAR, lw=1.5, linestyles="--", zorder=3)
ax_hero.plot(th_tk, mb_tk, color=GRAY_LINE, lw=1.6, solid_capstyle="round", zorder=3.5)
ax_hero.plot(th_hf, mb_hf, color=GRAY_BAR, lw=1.6, solid_capstyle="round", zorder=3.2)

# Subtle clean endpoint anchors at 14 threads
ax_hero.scatter([14], [mb_t[-1]], s=38, color=BLUE, zorder=6, clip_on=False)
ax_hero.scatter([14], [mb_g[-1]], s=26, color=ORANGE, zorder=5.5, clip_on=False)
ax_hero.scatter([14], [ft_14], s=22, color=GRAY_LINE, zorder=4, clip_on=False)
ax_hero.scatter([14], [mb_tk[-1]], s=24, color=GRAY_LINE, zorder=4.2, clip_on=False)
ax_hero.scatter([14], [mb_hf[-1]], s=24, color=GRAY_LINE, zorder=4.1, clip_on=False)

# Direct end labels floating cleanly right of x=14
lx_end = 14.85
ax_hero.text(lx_end, mb_t[-1] * 1.04, "4.50 GB/s", fontsize=11, fontweight=W_BOLD,
             color=BLUE, va="bottom", clip_on=False)
ax_hero.text(lx_end, mb_t[-1] * 0.88, "tuetoken", fontsize=9.5, fontweight=W_SEMI,
             color=BLUE, va="top", clip_on=False)

ax_hero.text(lx_end, mb_g[-1] * 1.02, "1.79 GB/s", fontsize=11, fontweight=W_SEMI,
             color=ORANGE, va="bottom", clip_on=False)
ax_hero.text(lx_end, mb_g[-1] * 0.85, "gigatoken · 2.5×", fontsize=9.2, fontweight=W_REG,
             color=ORANGE, va="top", clip_on=False)

ax_hero.text(lx_end, ft_14 * 1.02, f"{ft_14:.0f} MB/s", fontsize=10.2, fontweight=W_SEMI,
             color=TICK, va="bottom", clip_on=False)
ax_hero.text(lx_end, ft_14 * 0.85, f"fastokens · {mb_t[-1] / ft_14:.1f}×", fontsize=9.0, fontweight=W_REG,
             color=TICK, va="top", clip_on=False)

# OpenAI logo + tiktoken direct end label
lx_logo = 15.25
lx_after_logo = 16.35
place_logo_data(ax_hero, LOGO_OPENAI, (lx_logo, mb_tk[-1]), size_pt=15.0)
ax_hero.text(lx_after_logo, mb_tk[-1] * 1.04, f"{mb_tk[-1]:.0f} MB/s", fontsize=10.5, fontweight=W_SEMI,
             color=INK, va="bottom", clip_on=False)
ax_hero.text(lx_after_logo, mb_tk[-1] * 0.84, f"tiktoken · {mb_t[-1] / mb_tk[-1]:.0f}×", fontsize=9.2, fontweight=W_REG,
             color=TICK, va="top", clip_on=False)

# HuggingFace logo + HF tokenizers direct end label
place_logo_data(ax_hero, LOGO_HF, (lx_logo, mb_hf[-1]), size_pt=16.0)
ax_hero.text(lx_after_logo, mb_hf[-1] * 1.04, f"{mb_hf[-1]:.0f} MB/s", fontsize=10.5, fontweight=W_SEMI,
             color=INK, va="bottom", clip_on=False)
ax_hero.text(lx_after_logo, mb_hf[-1] * 0.84, f"HF tokenizers · {mb_t[-1] / mb_hf[-1]:.0f}×", fontsize=9.2, fontweight=W_REG,
             color=TICK, va="top", clip_on=False)

fig.text(0.095, CHART_TOP + 0.024, "Cold 1 GB corpus scaling",
         fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# 2) THROUGHPUT PANEL: LLM tokenizer throughput (top-right, wide)
RIGHT_AX_L = 0.585
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

hf_min = min(r["hf"] for r in fr)
hf_max = max(r["hf"] for r in fr)
sp_min = min(r["tuetoken"] / r["hf"] for r in fr)
sp_max = max(r["tuetoken"] / r["hf"] for r in fr)
t_hf_hdr = fig.text(
    MR, CHART_TOP + 0.026,
    f"HF tokenizers: {hf_min:.0f}–{hf_max:.0f} MB/s ({sp_min:.0f}–{sp_max:.0f}× slower)",
    fontsize=8.8, fontweight=W_REG, color=TICK, ha="right", va="bottom",
)
fig.canvas.draw()
bb_hf = t_hf_hdr.get_window_extent().transformed(fig.transFigure.inverted())
place_logo_fig(fig, LOGO_HF, (bb_hf.x0 - 0.010, CHART_TOP + 0.032), size_pt=12.0)

# 3) STARTUP PANEL: Dumbbell chart with tuetoken, gigatoken, and HF (bottom-right-left)
ax_st = fig.add_axes([RIGHT_AX_L, CHART_BOT, 0.145, 0.445 - CHART_BOT])
st = sorted(R["startup"], key=lambda r: r["tuetoken"])
n_st = len(st)

ax_st.set_xlim(-6.0, 360.0)
ax_st.set_ylim(n_st - 0.45, -0.78)
ax_st.set_yticks(np.arange(n_st))
ax_st.set_yticklabels([r["model"] for r in st], fontsize=9.2, color=INK)
ax_st.set_xticks([0, 150, 300])
ax_st.set_xticklabels(["0", "150 ms", "300 ms"])
clean_spines(ax_st, baseline="bottom", hgrid=False)
ax_st.tick_params(axis="y", pad=8)

for i, r in enumerate(st):
    hi = max(r["gigatoken"], r["hf"])
    ax_st.hlines(i, r["tuetoken"], hi, color=GRAY_BAR, lw=1.9, zorder=2)

ax_st.scatter([r["hf"] for r in st], np.arange(n_st), s=36, color=GRAY_BAR,
              edgecolor=GRAY_LINE, linewidth=0.9, zorder=3.5)
ax_st.scatter([r["gigatoken"] for r in st], np.arange(n_st), s=44, color=ORANGE, zorder=4)
ax_st.scatter([r["tuetoken"] for r in st], np.arange(n_st), s=52, color=BLUE, zorder=5)

# Crisp HF logo & gt label directly above row 0 dots
place_logo_data(ax_st, LOGO_HF, (st[0]["hf"], -0.58), size_pt=10.5)
ax_st.text(st[0]["gigatoken"], -0.42, "gt", fontsize=8.2, fontweight=W_SEMI, color=ORANGE, ha="center", va="bottom")

for i, r in enumerate(st):
    tv, gv, hv = r["tuetoken"], r["gigatoken"], r["hf"]
    hi = max(gv, hv)
    sp = round(hi / tv)
    ax_st.text(
        tv + 15, i, f"{tv:.1f}",
        fontsize=9.6, fontweight=W_SEMI, color=BLUE, va="center", ha="left",
        bbox=dict(facecolor=BG, edgecolor="none", pad=0.9), zorder=6,
    )
    ax_st.text(
        hi + 16, i, f"{sp}×",
        fontsize=10.8, fontweight=W_SEMI, color=INK, va="center", ha="left", clip_on=False,
    )

fig.text(RIGHT_AX_L, 0.445 + 0.017, "Startup latency",
         fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# 4) DECODE PANEL: Grouped bars (bottom-right-right)
DEC_L = 0.790
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

bar_w = 0.160
for j, (k, col) in enumerate(zip(keys_d, cols_d)):
    for i, r in enumerate(dc):
        rate = r["tokens"] / r[k]
        draw_rounded_barv(ax_dec, xd[i] + (j - 2) * bar_w, rate, bar_w * 0.85, col, radius_pt=2.8)

# Label ONLY the blue bars ("109M", "231M", "248M tokens/s")
for i, r in enumerate(dc):
    rate = r["tokens"] / r["tuetoken"]
    bx = xd[i] - 2 * bar_w
    if i == len(dc) - 1:
        ax_dec.text(xd[i] + 2.65 * bar_w, rate + 10, f"{rate:.0f}M tokens/s",
                    fontsize=10.8, fontweight=W_SEMI, color=BLUE, ha="right", va="bottom", clip_on=False)
    else:
        ax_dec.text(bx, rate + 10, f"{rate:.0f}M",
                    fontsize=11, fontweight=W_SEMI, color=BLUE, ha="center", va="bottom")

# Clean stacked direct label with micro OpenAI & HF logos over the 3 gray bars in group 1
place_logo_data(ax_dec, LOGO_OPENAI, (xd[0] - 0.06, 86), size_pt=7.8)
ax_dec.text(xd[0] + 0.03, 86, "tiktoken", fontsize=7.5, fontweight=W_REG, color=TICK, ha="left", va="center")
ax_dec.text(xd[0] + 0.03, 64, "fastokens", fontsize=7.5, fontweight=W_REG, color=TICK, ha="left", va="center")
place_logo_data(ax_dec, LOGO_HF, (xd[0] - 0.06, 42), size_pt=8.4)
ax_dec.text(xd[0] + 0.03, 42, "HF", fontsize=7.5, fontweight=W_REG, color=TICK, ha="left", va="center")

fig.text(DEC_L, 0.445 + 0.017, "Decode throughput",
         fontsize=13, fontweight=W_SEMI, color=INK, va="bottom")

# ── Bottom Stat Strip (4 tiles, thin dividers, no boxes) ──────────────────
fig.add_artist(mlines.Line2D([ML, MR], [0.180, 0.180], transform=fig.transFigure, color=BASELINE, lw=0.9))

warm = R["corpus"]["warm_100mb"]
bt = R["batch_trunc"]
ex = R["extras"]

tiles = [
    (f"{warm['tuetoken'] / 1000:.2f} GB/s", "warm steady-state"),
    (f"{bt['padded_10k_mbps']['tuetoken'] / 1000:.2f} GB/s", "10K docs padded"),
    (f"{bt['trunc_50mb_ms']['tuetoken_right']:.2f} ms",
     f"50 MB truncation · {bt['trunc_50mb_ms']['hf'] / bt['trunc_50mb_ms']['tuetoken_right']:,.0f}× vs HF"),
    (f"{ex['whisper_batch_500_ms']['hf'] / ex['whisper_batch_500_ms']['tuetoken']:.0f}×",
     "Whisper timestamps vs HF"),
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
