"""
2차 해석용 추가 그림 (EMNLP paper)
Fig 4: Qwen scale effect — model size vs bias sensitivity
Fig 5: Sign-flip paired bar — +ΔW vs −ΔW at best layer
Fig 6: Best-layer dot plot — BBQ / CrowS / StereoSet across models
"""

import os as _os_levee
import os
from pathlib import Path as _Path
# Repository root; override paths with LEVEE_* environment variables.
_REPO = _Path(os.environ.get("LEVEE_ROOT", _Path(__file__).resolve().parents[1]))


import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT_DIR   = os.environ.get("LEVEE_FIGDIR", str(_REPO / "results" / "paper_figures"))
PAPER_DIR = os.environ.get("LEVEE_PAPER_FIGDIR", str(_REPO / "paper" / "figures"))
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def _save(fig, name):
    for d in [PAPER_DIR, OUT_DIR]:
        fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"{d}/{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {name}")

BASE_DIR = os.environ.get("LEVEE_RESULTS", str(_REPO / "results"))

FAMILY_COLOR = {"Qwen": "#2196F3", "Gemma": "#4CAF50", "Llama": "#FF5722"}

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep_scores(alias):
    files = sorted(
        glob.glob(f"{BASE_DIR}/sweep_thirds/{alias}/sweep_L*_bbq.json"),
        key=lambda x: int(x.split("_L")[1].split("_")[0])
    )
    layers, scores = [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        layers.append(d["layer_idx"])
        scores.append(d["metrics"]["ambig"]["bias_score"])
    return np.array(layers), np.array(scores)


def load_signflip_scores(alias):
    files = sorted(
        glob.glob(f"{BASE_DIR}/signflip/{alias}/sweep_L*_bbq.json"),
        key=lambda x: int(x.split("_L")[1].split("_")[0])
    )
    layers, scores = [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        layers.append(d["layer_idx"])
        scores.append(d["metrics"]["ambig"]["bias_score"])
    return np.array(layers), np.array(scores)


def load_crows_scores(alias):
    files = sorted(
        glob.glob(f"{BASE_DIR}/sweep_multieval/{alias}/sweep_L*_crows.json"),
        key=lambda x: int(x.split("_L")[1].split("_")[0])
    )
    layers, scores = [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        bs = d["metrics"]["overall"].get("bias_score")
        if bs is not None:
            layers.append(d["layer_idx"])
            scores.append(bs)
    return np.array(layers), np.array(scores)


def load_ss_scores(alias):
    files = sorted(
        glob.glob(f"{BASE_DIR}/sweep_multieval/{alias}/sweep_L*_stereoset.json"),
        key=lambda x: int(x.split("_L")[1].split("_")[0])
    )
    layers, scores = [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        ss = d["metrics"]["overall"].get("ss_score")
        if ss is not None:
            layers.append(d["layer_idx"])
            scores.append(ss)
    return np.array(layers), np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Qwen scale effect
# ─────────────────────────────────────────────────────────────────────────────

QWEN_MODELS = [
    ("qwen_0.5b", "0.5B",  0.5,  24, 0.2521),
    ("qwen_1.5b", "1.5B",  1.5,  28, 0.2725),
    ("qwen_3b",   "3B",    3.0,  36, 0.0503),
    ("qwen_14b",  "14B",  14.0,  48, 0.0129),
]

def plot_scale_effect():
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import TwoSlopeNorm

    N_BINS = 10
    BIN_EDGES = np.linspace(0, 1, N_BINS + 1)
    n_models = len(QWEN_MODELS)

    # Build z-score normalised matrix: 4 models × 10 depth bins
    matrix = np.full((n_models, N_BINS), np.nan)
    for i, (alias, label, size, n, base) in enumerate(QWEN_MODELS):
        _, scores = load_sweep_scores(alias)
        depth = np.arange(len(scores)) / max(len(scores) - 1, 1)
        mu, sd = scores.mean(), scores.std()
        z = (scores - mu) / sd if sd > 1e-8 else np.zeros_like(scores)
        for b in range(N_BINS):
            lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
            mask = (depth >= lo) & (depth <= hi if b == N_BINS - 1 else depth < hi)
            valid = z[mask]
            if len(valid) > 0:
                matrix[i, b] = valid.mean()

    # Size-graduated Qwen blues (light → dark)
    QWEN_BLUES = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
    TEXT_COLORS = ["#1a1a1a", "#1a1a1a", "white", "white"]

    FIG_W, FIG_H = 10.0, 5.5
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1.6, 10, 0.45],
        wspace=0.03,
        left=0.02, right=0.98, bottom=0.17, top=0.94,
    )
    ax_lbl  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cb   = fig.add_subplot(gs[2])

    # ── Left: model name bars (size-graduated blues) ──────────────────────────
    for i, (alias, label, size, n, base) in enumerate(QWEN_MODELS):
        ax_lbl.barh(i, 1, left=0, height=0.96,
                    color=QWEN_BLUES[i], align="center",
                    edgecolor="white", linewidth=0.5)
        ax_lbl.text(0.5, i, f"Qwen {label}",
                    ha="center", va="center", fontsize=9.5,
                    color=TEXT_COLORS[i], fontweight="bold")

    # <3B / ≥3B separator
    ax_lbl.axhline(1.5, color="white", lw=2.0)

    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(-0.5, n_models - 0.5)
    ax_lbl.invert_yaxis()
    ax_lbl.axis("off")

    # ── Right: heatmap ────────────────────────────────────────────────────────
    vabs = max(np.nanpercentile(np.abs(matrix), 95), 0.5)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    im = ax_heat.imshow(
        matrix, aspect="auto", cmap="RdBu_r", norm=norm,
        interpolation="nearest",
        extent=[0, N_BINS, n_models, 0],
    )

    mid_lo, mid_hi = N_BINS / 3, 2 * N_BINS / 3
    ax_heat.axvspan(mid_lo, mid_hi, color="#FFB347", alpha=0.12, zorder=0)
    for xv in (mid_lo, mid_hi):
        ax_heat.axvline(xv, color="#555555", lw=1.2, ls="--", alpha=0.7, zorder=3)

    ax_heat.axhline(2, color="white", lw=2.0, alpha=0.9)

    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    for sp in ax_heat.spines.values():
        sp.set_visible(False)

    # ── Depth bin labels + third labels via fig.text() ───────────────────────
    fig.canvas.draw()
    data_to_fig = ax_heat.transData + fig.transFigure.inverted()
    LABEL_Y_DROP  = 0.012
    REGION_Y_DROP = 0.058

    bin_centers = np.arange(N_BINS) + 0.5
    depth_ticks = [f"{b:.1f}" for b in BIN_EDGES[:-1]]

    for bc, lbl in zip(bin_centers, depth_ticks):
        fx, fy = data_to_fig.transform((bc, n_models))
        fig.text(fx, fy - LABEL_Y_DROP, lbl,
                 ha="center", va="top", fontsize=8.0,
                 transform=fig.transFigure)

    for x0, x1, region_lbl in [
        (0,      mid_lo, "First third"),
        (mid_lo, mid_hi, "Mid third"),
        (mid_hi, N_BINS, "Last third"),
    ]:
        xc = (x0 + x1) / 2
        fx, fy = data_to_fig.transform((xc, n_models))
        bold = (region_lbl == "Mid third")
        fig.text(fx, fy - REGION_Y_DROP, region_lbl,
                 ha="center", va="top", fontsize=8.5,
                 color="#333333", fontweight="bold" if bold else "normal",
                 transform=fig.transFigure)

    fx_l, _ = data_to_fig.transform((0, n_models))
    fx_r, _ = data_to_fig.transform((N_BINS, n_models))
    _, fy_b = data_to_fig.transform((0, n_models))
    fig.text((fx_l + fx_r) / 2, fy_b - 0.10, "Normalized layer depth",
             ha="center", va="top", fontsize=9.5, color="#222222",
             transform=fig.transFigure)

    # <3B / ≥3B group labels — left of model name panel, rotated 90°
    for gy, glabel, gcol in [
        (1.0, "<3B\n(diffuse)",    "#666666"),
        (3.0, "≥3B\n(structured)", "#222222"),
    ]:
        ax_lbl.text(-0.18, gy, glabel, ha="center", va="center",
                    fontsize=7.5, color=gcol, style="italic",
                    rotation=90, clip_on=False,
                    transform=ax_lbl.transData)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("L-BBQ score (z-score)", fontsize=8, labelpad=6)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.text(0.5,  1.04, "High bias", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=6.5, color="#a32424", fontstyle="italic")
    cb.ax.text(0.5, -0.04, "Low bias",  transform=cb.ax.transAxes,
               ha="center", va="top",    fontsize=6.5, color="#1a4f9c", fontstyle="italic")

    _save(fig, "fig4_scale_effect")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Sign-flip paired bar
# ─────────────────────────────────────────────────────────────────────────────

# best layer per model (sweep_thirds ambig minimum)
SIGNFLIP_MODELS = [
    ("qwen_3b",    "qwen_3b",    "Qwen2.5-3B",   "Qwen",  0.0503, 18),
    ("gemma_2b",   "gemma_2b",   "Gemma-2-2B",   "Gemma", 0.1599, 11),
    ("llama32_1b", "llama32_1b", "Llama-3.2-1B", "Llama", 0.2508,  9),
    ("llama32_3b", "llama32_3b", "Llama-3.2-3B", "Llama", 0.1762, 10),
    # Llama-3-8B excluded: sign-flip anomaly (-ΔW < +ΔW < base), consistent with §4.4 MMLU anomaly
    ("llama31_8b", "llama31_8b", "Llama-3.1-8B", "Llama", 0.1180, 11),
]

def get_score_at_layer(alias, experiment, layer_idx):
    pattern = f"{BASE_DIR}/{experiment}/{alias}/sweep_L{layer_idx:02d}_bbq.json"
    files = glob.glob(pattern)
    if not files:
        # try without zero-padding
        pattern2 = f"{BASE_DIR}/{experiment}/{alias}/sweep_L{layer_idx}_bbq.json"
        files = glob.glob(pattern2)
    if not files:
        return None
    with open(files[0]) as fp:
        d = json.load(fp)
    return d["metrics"]["ambig"]["bias_score"]


def plot_signflip():
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    labels    = [m[2] for m in SIGNFLIP_MODELS]
    baselines = [m[4] for m in SIGNFLIP_MODELS]
    pos_vals  = []
    neg_vals  = []

    for sw_alias, sf_alias, label, family, base, best_l in SIGNFLIP_MODELS:
        pos = get_score_at_layer(sw_alias, "sweep_thirds", best_l)
        neg = get_score_at_layer(sf_alias, "signflip",     best_l)
        pos_vals.append(pos if pos is not None else base)
        neg_vals.append(neg if neg is not None else base)

    x = np.arange(len(labels))
    w = 0.26

    # Colorblind-friendly palette: gray / blue / orange
    ax.bar(x - w, baselines, w*0.88, label="Baseline (base model)", color="#78909C", zorder=2)
    ax.bar(x,     pos_vals,  w*0.88, label=r"+$\Delta W$ (alignment direction)",  color="#1976D2", zorder=2)
    ax.bar(x + w, neg_vals,  w*0.88, label=r"$-\Delta W$ (sign-flip)",            color="#F57C00", zorder=2)

    # Grid instead of bar annotations
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#BDBDBD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks([i * 0.05 for i in range(7)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=0, ha="center")
    ax.set_ylabel("L-BBQ score", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(labelsize=9)
    ax.set_ylim(bottom=0, top=0.32)

    _save(fig, "fig5_signflip_paired")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Best-layer dot plot across benchmarks
# ─────────────────────────────────────────────────────────────────────────────

MULTI_MODELS = [
    ("qwen_0.5b",  "Qwen2.5-0.5B",    "Qwen",  24),
    ("qwen_1.5b",  "Qwen2.5-1.5B",    "Qwen",  28),
    ("qwen_3b",    "Qwen2.5-3B",      "Qwen",  36),
    ("qwen_14b",   "Qwen2.5-14B",     "Qwen",  48),
    ("gemma_2b",   "Gemma-2-2B",     "Gemma", 26),
    ("gemma_9b",   "Gemma-2-9B",     "Gemma", 42),
    ("llama32_1b", "Llama-3.2-1B", "Llama", 16),
    ("llama32_3b", "Llama-3.2-3B", "Llama", 28),
    ("llama3_8b",  "Llama-3-8B",   "Llama", 32),
]

def plot_best_layer_dot():
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    y_bbq   = []
    y_crows = []
    y_ss    = []

    def last_argmin(scores):
        # Pick highest-index layer in case of ties (consistent with mode_based.json)
        mn = scores.min()
        return np.where(scores == mn)[0][-1]

    for alias, label, family, n in MULTI_MODELS:
        # BBQ best layer (argmin, last occurrence on tie)
        layers_b, scores_b = load_sweep_scores(alias)
        y_bbq.append(layers_b[last_argmin(scores_b)] / (n - 1))

        # CrowS best layer (argmin bias_score, last on tie)
        layers_c, scores_c = load_crows_scores(alias)
        if len(scores_c) > 0:
            y_crows.append(layers_c[last_argmin(scores_c)] / (n - 1))
        else:
            y_crows.append(None)

        # StereoSet best layer (argmin ss_score, last on tie)
        layers_s, scores_s = load_ss_scores(alias)
        if len(scores_s) > 0:
            y_ss.append(layers_s[last_argmin(scores_s)] / (n - 1))
        else:
            y_ss.append(None)

    x_labels = [m[1] for m in MULTI_MODELS]
    x        = np.arange(len(x_labels))
    fam_cols = [FAMILY_COLOR[m[2]] for m in MULTI_MODELS]

    # plot dots per benchmark
    markers = {"BBQ": ("o", "#1565C0", 80), "CrowS": ("s", "#2E7D32", 80), "SS": ("^", "#BF360C", 80)}

    for i, (xpos, fc) in enumerate(zip(x, fam_cols)):
        # BBQ
        ax.scatter(xpos, y_bbq[i],   marker="o", s=100, color="#1565C0",
                   zorder=4, edgecolors="white", lw=0.5)
        # CrowS
        if y_crows[i] is not None:
            ax.scatter(xpos, y_crows[i], marker="s", s=100, color="#2E7D32",
                       zorder=4, edgecolors="white", lw=0.5)
        # SS
        if y_ss[i] is not None:
            ax.scatter(xpos, y_ss[i],   marker="^", s=100, color="#BF360C",
                       zorder=4, edgecolors="white", lw=0.5)

        # connect with spread line (darker, thicker for visibility)
        pts = [y for y in [y_bbq[i], y_crows[i], y_ss[i]] if y is not None]
        if len(pts) > 1:
            ax.plot([xpos]*len(pts), sorted(pts), color="#444444", lw=1.6, alpha=0.65, zorder=3)

    # thirds bands
    ax.axhspan(0,     1/3, alpha=0.05, color="#2196F3")
    ax.axhspan(1/3,   2/3, alpha=0.08, color="#FF9800")
    ax.axhspan(2/3,   1.0, alpha=0.05, color="#9C27B0")

    ax.axhline(1/3, color="#2196F3", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(2/3, color="#9C27B0", lw=0.6, ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best layer (normalized depth)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1/3, 0.5, 2/3, 1.0])
    ax.set_yticklabels(["0.00", "0.33", "0.50", "0.67", "1.00"], fontsize=8)

    legend_els = [
        Line2D([0],[0], marker="o", color="#1565C0", lw=0, markersize=9, label="BBQ (ambig)"),
        Line2D([0],[0], marker="s", color="#2E7D32", lw=0, markersize=9, label="CrowS-Pairs"),
        Line2D([0],[0], marker="^", color="#BF360C", lw=0, markersize=9, label="StereoSet"),
        Patch(facecolor="#2196F3", alpha=0.2, label="First third"),
        Patch(facecolor="#FF9800", alpha=0.3, label="Mid third"),
        Patch(facecolor="#9C27B0", alpha=0.2, label="Last third"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right",
              ncol=2, framealpha=0.9, edgecolor="#cccccc")
    ax.set_title("")

    _save(fig, "fig6_best_layer_dotplot")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 2nd-level interpretation figures...")
    plot_scale_effect()
    plot_signflip()
    plot_best_layer_dot()
    print("Done.")
