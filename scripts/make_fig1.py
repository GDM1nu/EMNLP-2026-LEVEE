"""
Figure 1: Per-layer BBQ bias score analysis across 11 models.

Generates:
  fig1a  — Normalized boxplot (main message, main text)
  fig1b  — Best-layer positions dot plot (main text)
  fig1c  — Per-model curves small multiples (appendix)
  fig1   — Combined 1a + 1b (2-panel, replaces current Figure 1)

Usage:
    python scripts/make_fig1.py
"""

import os as _os_levee
import os
from pathlib import Path as _Path
# Repository root; override paths with LEVEE_* environment variables.
_REPO = _Path(os.environ.get("LEVEE_ROOT", _Path(__file__).resolve().parents[1]))


import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE = os.environ.get("LEVEE_RESULTS", str(_REPO / "results"))
OUT  = os.environ.get("LEVEE_PAPER_FIGDIR", str(_REPO / "paper" / "figures"))

# ─────────────────────────────────────────────────────────────────
# EMNLP style
# ─────────────────────────────────────────────────────────────────
EMNLP_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FAMILY_COLORS = {
    "Qwen":  "#4C72B0",
    "Gemma": "#55A868",
    "Llama": "#DD8452",
}
MID_COLOR = "#FFB347"
MID_ALPHA = 0.18

# ─────────────────────────────────────────────────────────────────
# Model registry
# (alias, display, family, n_layers, structured)
# ─────────────────────────────────────────────────────────────────
ALL_MODELS = [
    ("qwen_0.5b",  "Qwen2.5-0.5B",     "Qwen",  24, False),
    ("qwen_1.5b",  "Qwen2.5-1.5B",     "Qwen",  28, False),
    ("qwen_3b",    "Qwen2.5-3B",   "Qwen",  36, True),
    ("qwen_14b",   "Qwen2.5-14B",  "Qwen",  48, True),
    ("gemma_2b",   "Gemma-2-2B",    "Gemma", 26, True),
    ("gemma_9b",   "Gemma-2-9B",    "Gemma", 42, True),
    ("gemma3_12b", "Gemma-3-12B",   "Gemma", 48, True),
    ("llama32_1b", "Llama-3.2-1B",  "Llama", 16, True),
    ("llama32_3b", "Llama-3.2-3B",  "Llama", 28, True),
    ("llama3_8b",  "Llama-3-8B",  "Llama", 32, True),
    ("llama31_8b", "Llama-3.1-8B",  "Llama", 32, True),
]

STRUCTURED = [(a, d, f, n, s) for a, d, f, n, s in ALL_MODELS if s]

# base bias 없는 모델에 대한 fallback (SESSION.md Table 2)
BASE_FALLBACK = {"qwen_14b": 0.0129}

# ─────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────

def load_sweep(alias):
    files = sorted(
        glob.glob(f"{BASE}/sweep_thirds/{alias}/sweep_L*_bbq.json"),
        key=lambda x: int(x.split("_L")[1].split("_")[0]),
    )
    layers, scores = [], []
    for f in files:
        d = json.load(open(f))
        layers.append(d["layer_idx"])
        scores.append(d["metrics"]["ambig"]["bias_score"])
    return np.array(layers), np.array(scores)


def get_base_bias(alias):
    f = f"{BASE}/control_sweep/{alias}/ctrl_zero_L00_bbq.json"
    if os.path.exists(f):
        return json.load(open(f))["metrics"]["ambig"]["bias_score"]
    return BASE_FALLBACK.get(alias)


def get_full_bias(alias):
    f = f"{BASE}/block_removal/{alias}/block_full_bbq.json"
    if os.path.exists(f):
        return json.load(open(f))["metrics"]["ambig"]["bias_score"]
    return None


def load_all():
    """모델별 dict 리스트 반환."""
    records = []
    for alias, display, family, n, structured in ALL_MODELS:
        layers, scores = load_sweep(alias)
        base = get_base_bias(alias)
        full = get_full_bias(alias)
        records.append(dict(
            alias=alias, display=display, family=family,
            n=n, structured=structured,
            layers=layers, scores=scores,
            base_bias=base, full_bias=full,
        ))
    assert len(records) == 11, f"Expected 11 models, got {len(records)}"
    return records


def normalize(scores):
    """Within-model z-score normalization."""
    mu, sd = scores.mean(), scores.std()
    if sd < 1e-8:
        return np.zeros_like(scores)
    return (scores - mu) / sd


# ─────────────────────────────────────────────────────────────────
# Fig 1a — Normalized boxplot
# ─────────────────────────────────────────────────────────────────

def make_fig1a(records, ax=None, standalone=True):
    BINS = np.linspace(0, 1, 11)
    bin_data = {i: [] for i in range(10)}

    for r in records:
        if not r["structured"]:
            continue
        z = normalize(r["scores"])
        depths = r["layers"] / (r["n"] - 1)
        for i in range(10):
            lo, hi = BINS[i], BINS[i + 1]
            mask = (depths >= lo) & (depths < hi if i < 9 else depths <= hi)
            bin_data[i].extend(z[mask].tolist())

    if standalone:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
    else:
        fig = None

    positions = np.arange(10) * 0.1  # 0.0, 0.1, ..., 0.9

    # mid-third 음영
    ax.axvspan(0.275, 0.675, alpha=MID_ALPHA, color=MID_COLOR, zorder=0, lw=0)

    # y=0 기준선
    ax.axhline(0, color="dimgray", lw=0.9, ls="-", alpha=0.6, zorder=1)

    box_vals = [bin_data[i] for i in range(10)]
    bp = ax.boxplot(
        box_vals,
        positions=positions,
        widths=0.07,
        patch_artist=True,
        medianprops=dict(color="black", lw=2.0),
        boxprops=dict(facecolor="#90CAF9", alpha=0.85, lw=0.8),
        whiskerprops=dict(lw=1.0, color="steelblue"),
        capprops=dict(lw=1.0, color="steelblue"),
        flierprops=dict(marker="", markersize=0),
    )

    # mid-third 박스 색 강조
    for i, patch in enumerate(bp["boxes"]):
        if 3 <= i <= 6:
            patch.set_facecolor(MID_COLOR)
            patch.set_alpha(0.80)
            for elem in ["whiskers", "caps"]:
                pass  # caps/whiskers color handled globally

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{b:.1f}" for b in BINS[:-1]])
    ax.set_xlim(-0.05, 0.97)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("L-BBQ score (z-score)")

    legend_els = [
        Patch(facecolor=MID_COLOR, alpha=0.7, label="Mid-third (0.33–0.67)"),
        Patch(facecolor="#90CAF9", alpha=0.85, label="Outer thirds"),
        Line2D([0], [0], color="dimgray", lw=0.9, alpha=0.6, label="Model mean"),
    ]
    ax.legend(handles=legend_els, loc="upper right", frameon=False)

    if standalone:
        fig.savefig(f"{OUT}/fig1a_normalized_boxplot.pdf", bbox_inches="tight")
        fig.savefig(f"{OUT}/fig1a_normalized_boxplot.png", bbox_inches="tight", dpi=200)
        print("Saved: fig1a_normalized_boxplot")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Fig 1b — Best-layer positions dot plot
# ─────────────────────────────────────────────────────────────────

def make_fig1b(records, ax=None, standalone=True):
    rows = []
    for r in records:
        best_idx = int(np.argmin(r["scores"]))
        best_depth = r["layers"][best_idx] / (r["n"] - 1)
        rows.append(dict(
            display=r["display"],
            family=r["family"],
            structured=r["structured"],
            best_depth=best_depth,
            best_layer=r["layers"][best_idx],
        ))

    # 순서: Qwen2.5-0.5B→14B, Gemma-2-2B→12B, Llama-1B→8B (아래→위)
    rows = list(reversed(rows))

    if standalone:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
    else:
        fig = None

    # mid-third 음영
    ax.axvspan(0.33, 0.67, alpha=MID_ALPHA, color=MID_COLOR, zorder=0, lw=0)

    for y, row in enumerate(rows):
        color = FAMILY_COLORS[row["family"]]
        if row["structured"]:
            ax.scatter(row["best_depth"], y,
                       s=80, color=color, zorder=3,
                       edgecolors="white", linewidths=0.5)
        else:
            # diffuse 모델: 빈 원
            ax.scatter(row["best_depth"], y,
                       s=80, facecolors="white", edgecolors=color,
                       linewidths=1.5, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["display"] for r in rows], fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Normalized depth of best layer")
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.grid(axis="y", alpha=0)

    # mid-third 텍스트 레이블
    ax.text(0.50, len(rows) - 0.2, "Mid-third", ha="center", va="top",
            fontsize=8, color="darkorange", fontstyle="italic")

    legend_els = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["Qwen"], markersize=8, label="Qwen"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["Gemma"], markersize=8, label="Gemma"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["Llama"], markersize=8, label="Llama"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="white", markeredgecolor="gray",
               markeredgewidth=1.5, markersize=8, label="Diffuse (no clear peak)"),
    ]
    ax.legend(handles=legend_els, loc="lower right", frameon=False, fontsize=8)

    if standalone:
        fig.savefig(f"{OUT}/fig1b_best_layer_positions.pdf", bbox_inches="tight")
        fig.savefig(f"{OUT}/fig1b_best_layer_positions.png", bbox_inches="tight", dpi=200)
        print("Saved: fig1b_best_layer_positions")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Fig 1c — Per-model curves (appendix)
# ─────────────────────────────────────────────────────────────────

def make_fig1c(records):
    structured = [r for r in records if r["structured"]]
    n = len(structured)  # 9
    ncols, nrows = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 7.0))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12,
                        top=0.95, hspace=0.52, wspace=0.38)
    axes = axes.flatten()

    for ax, r in zip(axes, structured):
        color = FAMILY_COLORS[r["family"]]
        layers, scores = r["layers"], r["scores"]
        N = r["n"]
        t1, t2 = N // 3, 2 * N // 3

        ax.axvspan(t1, t2, alpha=MID_ALPHA, color=MID_COLOR, zorder=0, lw=0)

        if r["base_bias"] is not None:
            ax.axhline(r["base_bias"], color="gray", lw=0.9,
                       ls="--", alpha=0.75, zorder=1)
        if r["full_bias"] is not None:
            ax.axhline(r["full_bias"], color="black", lw=0.9,
                       ls=":", alpha=0.75, zorder=1)

        ax.plot(layers, scores, color=color, lw=1.5, zorder=3)

        best_idx = int(np.argmin(scores))
        ax.axvline(layers[best_idx], color="#9C27B0", lw=1.1,
                   ls="--", alpha=0.85, zorder=4)

        ax.set_title(r["display"], fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel("Layer index", fontsize=7.5)
        ax.set_ylabel("L-BBQ score", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, N - 1)

    legend_els = [
        Patch(facecolor=MID_COLOR, alpha=0.6, label="Mid-third"),
        Line2D([0], [0], color="gray",    lw=1.0, ls="--", label="Base model"),
        Line2D([0], [0], color="black",   lw=1.0, ls=":",  label=r"Full $\Delta W$"),
        Line2D([0], [0], color="#9C27B0", lw=1.1, ls="--", label="Best layer"),
        Patch(facecolor=FAMILY_COLORS["Qwen"],  label="Qwen"),
        Patch(facecolor=FAMILY_COLORS["Gemma"], label="Gemma"),
        Patch(facecolor=FAMILY_COLORS["Llama"], label="Llama"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=7,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.01), frameon=False)

    fig.savefig(f"{OUT}/fig1c_per_model_curves.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig1c_per_model_curves.png", bbox_inches="tight", dpi=200)
    print("Saved: fig1c_per_model_curves")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Fig 1 Combined — 1a (top) + 1b (bottom)
# ─────────────────────────────────────────────────────────────────

def make_fig1_combined(records):
    with mpl.rc_context(EMNLP_STYLE):
        fig = plt.figure(figsize=(7.0, 7.2))
        gs = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[1, 1.15],
                               hspace=0.42)

        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])

        # 패널 레이블 — axes title로 분리 (Y축 레이블과 충돌 방지)
        ax_a.set_title("(a)", loc="left", fontsize=11, fontweight="bold", pad=4)
        ax_b.set_title("(b)", loc="left", fontsize=11, fontweight="bold", pad=4)

        make_fig1a(records, ax=ax_a, standalone=False)
        make_fig1b(records, ax=ax_b, standalone=False)

        fig.savefig(f"{OUT}/fig1ab_combined.pdf", bbox_inches="tight")
        fig.savefig(f"{OUT}/fig1ab_combined.png", bbox_inches="tight", dpi=200)
        print("Saved: fig1ab_combined (fig1a + fig1b combined)")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with mpl.rc_context(EMNLP_STYLE):
        print("Loading data...")
        records = load_all()
        print(f"  {len(records)} models loaded")

        print("=== Fig 1a: Normalized boxplot ===")
        make_fig1a(records)

        print("=== Fig 1b: Best-layer positions ===")
        make_fig1b(records)

        print("=== Fig 1c: Per-model curves (appendix) ===")
        make_fig1c(records)

        print("=== Fig 1 Combined ===")
        make_fig1_combined(records)

    print("Done.")
