"""
Paper figures for EMNLP submission.
Fig 1: Per-layer sufficiency sweep (9 models, BBQ ambig bias_score)
Fig 2: Cumulative injection curves (3 models)
Fig 3: Necessity post-hoc group contributions (9 models)
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

OUT_DIR   = os.environ.get("LEVEE_FIGDIR", str(_REPO / "results" / "paper_figures"))
PAPER_DIR = os.environ.get("LEVEE_PAPER_FIGDIR", str(_REPO / "paper" / "figures"))
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def _save(fig, name):
    for d in [PAPER_DIR, OUT_DIR]:
        fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"{d}/{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {name}")

# ── palette ──────────────────────────────────────────────────────────────────
FAMILY_COLOR = {
    "Qwen":   "#2196F3",
    "Gemma":  "#4CAF50",
    "Llama":  "#FF5722",
}
FAMILY_STYLE = {
    "Qwen":  "-",
    "Gemma": "--",
    "Llama": "-.",
}

MODELS = [
    # (alias, display, family, n_layers)
    ("qwen_0.5b",  "Qwen2.5-0.5B",      "Qwen",  24),
    ("qwen_1.5b",  "Qwen2.5-1.5B",      "Qwen",  28),
    ("qwen_3b",    "Qwen2.5-3B",        "Qwen",  36),
    ("qwen_14b",   "Qwen2.5-14B",       "Qwen",  48),
    ("gemma_2b",   "Gemma-2-2B",       "Gemma", 26),
    ("gemma_9b",   "Gemma-2-9B",       "Gemma", 42),
    ("llama32_1b", "Llama-3.2-1B",   "Llama", 16),
    ("llama32_3b", "Llama-3.2-3B",   "Llama", 28),
    ("llama3_8b",  "Llama-3-8B",     "Llama", 32),
]

BASE_DIR = os.environ.get("LEVEE_RESULTS", str(_REPO / "results"))


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep(alias):
    path = f"{BASE_DIR}/sweep_thirds/{alias}"
    files = sorted(glob.glob(f"{path}/sweep_L*_bbq.json"),
                   key=lambda x: int(x.split("_L")[1].split("_")[0]))
    layers, scores = [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        layers.append(d["layer_idx"])
        scores.append(d["metrics"]["ambig"]["bias_score"])
    return np.array(layers), np.array(scores)


def load_cumulative(alias):
    path = f"{BASE_DIR}/cumulative/{alias}"
    files = sorted(glob.glob(f"{path}/cumul_L*_bbq.json"),
                   key=lambda x: int(x.split("_L")[1].split("_")[0]))
    ks, scores = [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        ks.append(d["cumul_k"])
        scores.append(d["metrics"]["ambig"]["bias_score"])
    return np.array(ks), np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Per-layer sufficiency sweep
# Main body: 3 representative models (1×3)
# ─────────────────────────────────────────────────────────────────────────────

# 3 representative models for main body figure (one per family, mid-range scale)
MODELS_3REP = [
    ("qwen_3b",   "Qwen2.5-3B",  "Qwen",  36),
    ("gemma_9b",  "Gemma-2-9B",   "Gemma", 42),
    ("llama3_8b", "Llama-3-8B",   "Llama", 32),
]

def get_base_bias(alias):
    """Base (no-injection) BBQ bias from ctrl_zero layer 0."""
    p = f"{BASE_DIR}/control_sweep/{alias}/ctrl_zero_L00_bbq.json"
    if os.path.exists(p):
        return json.load(open(p))["metrics"]["ambig"]["bias_score"]
    # fallback: necessity_posthoc
    p2 = f"{BASE_DIR}/necessity_posthoc/{alias}_necessity.json"
    if os.path.exists(p2):
        return json.load(open(p2))["baseline_ambig_bias"]
    return None


def get_full_bias(alias):
    """Full ΔW injection BBQ bias from block_removal."""
    p = f"{BASE_DIR}/block_removal/{alias}/block_full_bbq.json"
    if os.path.exists(p):
        return json.load(open(p))["metrics"]["ambig"]["bias_score"]
    return None


def plot_sufficiency():
    """3-model 1×3 figure for main body — with base/full ΔW reference lines."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    for ax, (alias, label, family, n) in zip(axes, MODELS_3REP):
        layers, scores = load_sweep(alias)
        color = FAMILY_COLOR[family]
        base = get_base_bias(alias)
        full = get_full_bias(alias)

        t1 = n // 3
        t2 = 2 * n // 3
        ax.axvspan(0,  t1, alpha=0.06, color="#2196F3")
        ax.axvspan(t1, t2, alpha=0.08, color="#FF9800")
        ax.axvspan(t2, n,  alpha=0.06, color="#9C27B0")

        # Reference lines: base and full ΔW
        if base is not None:
            ax.axhline(base, color="gray", lw=1.2, ls="--",
                       alpha=0.80, zorder=1, label="_base")
        if full is not None:
            ax.axhline(full, color="black", lw=1.2, ls=":",
                       alpha=0.80, zorder=1, label="_full")

        ax.plot(layers, scores, color=color, lw=2.0, zorder=3)
        ax.axvline(layers[np.argmin(scores)], color="red",
                   lw=1.8, ls="--", zorder=4, alpha=0.85)

        ax.set_title(label, fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel("Layer index", fontsize=9)
        ax.set_ylabel("L-BBQ score", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, n - 1)
        ax.spines[['top','right']].set_visible(False)

    legend_els = [
        Patch(facecolor="#2196F3", alpha=0.25, label="First third"),
        Patch(facecolor="#FF9800", alpha=0.35, label="Mid third"),
        Patch(facecolor="#9C27B0", alpha=0.25, label="Last third"),
        Line2D([0], [0], color="red",   lw=1.8, ls="--", label="Best layer (min bias)"),
        Line2D([0], [0], color="gray",  lw=1.2, ls="--", label="Base model (no injection)"),
        Line2D([0], [0], color="black", lw=1.2, ls=":",  label=r"Full $\Delta W$ injection"),
    ]
    plt.subplots_adjust(bottom=0.30, top=0.90, left=0.07, right=0.97, wspace=0.32)
    fig.legend(handles=legend_els, loc="lower center", ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.03))

    _save(fig, "fig1_sufficiency_sweep")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Cumulative injection
# ─────────────────────────────────────────────────────────────────────────────

CUMUL_MODELS = [
    ("gemma_9b",  "Gemma-2-9B",  "Gemma", 42),
    ("llama3_8b", "Llama-3-8B",  "Llama", 32),
    ("qwen_14b",  "Qwen2.5-14B","Qwen",  48),
]

def plot_cumulative():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    for ax, (alias, label, family, n) in zip(axes, CUMUL_MODELS):
        ks, scores = load_cumulative(alias)
        color = FAMILY_COLOR[family]

        t1 = n // 3
        t2 = 2 * n // 3
        ax.axvspan(0,  t1, alpha=0.06, color="#2196F3")
        ax.axvspan(t1, t2, alpha=0.08, color="#FF9800")
        ax.axvspan(t2, n,  alpha=0.06, color="#9C27B0")

        ax.plot(ks, scores, color=color, lw=2, zorder=3)

        # mark peak — vertical dashed line + callout annotation
        peak_k = ks[np.argmax(scores)]
        peak_v = scores.max()
        ax.axvline(peak_k, color="darkorange", lw=2.0, ls="--", zorder=4, alpha=0.9)
        # smart xytext: if peak is in left 30%, place label to the right; else to the left
        offset = max(n * 0.05, 1.5)
        if peak_k < n * 0.30:
            txt_x, ha = peak_k + offset, "left"
        else:
            txt_x, ha = peak_k - offset, "right"
        ax.annotate(f"k={peak_k}\n{peak_v:.3f}",
                    xy=(peak_k, peak_v * 0.92),
                    xytext=(txt_x, peak_v * 0.85),
                    fontsize=7.5, color="#D84315", fontweight="bold",
                    va="top", ha=ha, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="#E64A19", lw=1.2, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="#E64A19",
                                   lw=1.0, alpha=0.8))

        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Layers injected (k)", fontsize=9, labelpad=6)
        ax.set_ylabel("L-BBQ score", fontsize=9, labelpad=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#2196F3", alpha=0.25, label="First third"),
        Patch(facecolor="#FF9800", alpha=0.35, label="Mid third"),
        Patch(facecolor="#9C27B0", alpha=0.25, label="Last third"),
    ]
    plt.subplots_adjust(bottom=0.28, top=0.90, left=0.09, right=0.97, wspace=0.42)
    leg = fig.legend(handles=legend_els, loc="lower center", ncol=3,
                     fontsize=9.5, bbox_to_anchor=(0.5, 0.03),
                     frameon=True, framealpha=0.9, edgecolor="#cccccc",
                     handlelength=1.8, handletextpad=0.6, columnspacing=1.8)

    _save(fig, "fig2_cumulative")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Necessity post-hoc group contributions
# ─────────────────────────────────────────────────────────────────────────────

# From SESSION.md / PAPER_KO.md Table 4
NECESSITY = {
    "Qwen2.5-0.5B":    {"first": -0.0042, "mid": 0.0485,  "last": -0.0184},
    "Qwen2.5-1.5B":    {"first":  0.0005, "mid": 0.0225,  "last":  0.0202},
    "Qwen2.5-3B":      {"first":  0.0171, "mid": 0.1282,  "last":  0.0203},
    "Qwen2.5-14B":     {"first":  0.0245, "mid": 0.0908,  "last":  0.0158},
    "Gemma-2-2B":     {"first": -0.1147, "mid": 0.0343,  "last": -0.0118},
    "Gemma-2-9B":     {"first":  0.0353, "mid": 0.1597,  "last":  0.0062},
    "Llama-3.2-1B": {"first":  0.0468, "mid": 0.1219,  "last":  0.0260},
    "Llama-3.2-3B": {"first": -0.1086, "mid": 0.2881,  "last": -0.0121},  # from necessity_posthoc
    "Llama-3-8B":   {"first": -0.0212, "mid": 0.2093,  "last":  0.0048},
}

def plot_necessity():
    models = list(NECESSITY.keys())
    firsts = [NECESSITY[m]["first"] for m in models]
    mids   = [NECESSITY[m]["mid"]   for m in models]
    lasts  = [NECESSITY[m]["last"]  for m in models]

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.30, top=0.95)

    bars_f = ax.bar(x - w, firsts, w, label="First third", color="#2196F3", alpha=0.85)
    bars_m = ax.bar(x,     mids,   w, label="Mid third",   color="#FF9800", alpha=0.85)
    bars_l = ax.bar(x + w, lasts,  w, label="Last third",  color="#9C27B0", alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Group contribution Σδ (bias score units)", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.legend(fontsize=9.5, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.01),
               frameon=True, framealpha=0.9, edgecolor="#cccccc",
               handlelength=1.8, columnspacing=1.5)

    # annotate mid bars (dominant contributor)
    for bar, v in zip(bars_m, mids):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=6.5, color="#E65100")

    _save(fig, "fig3_necessity_posthoc")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix — Cumulative injection: all 8 models (2×4 grid)
# ─────────────────────────────────────────────────────────────────────────────

CUMUL_ALL = [
    ("qwen_0.5b",  "Qwen2.5-0.5B",  "Qwen",  24),
    ("qwen_1.5b",  "Qwen2.5-1.5B",  "Qwen",  28),
    ("qwen_3b",    "Qwen2.5-3B",    "Qwen",  36),
    ("qwen_14b",   "Qwen2.5-14B",   "Qwen",  48),
    ("gemma_2b",   "Gemma-2-2B",     "Gemma", 26),
    ("gemma_9b",   "Gemma-2-9B",     "Gemma", 42),
    ("llama32_3b", "Llama-3.2-3B",   "Llama", 28),
    ("llama3_8b",  "Llama-3-8B",     "Llama", 32),
]

def plot_cumulative_appendix():
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (alias, label, family, n) in zip(axes, CUMUL_ALL):
        ks, scores = load_cumulative(alias)
        color = FAMILY_COLOR[family]

        t1 = n // 3
        t2 = 2 * n // 3
        ax.axvspan(0,  t1, alpha=0.06, color="#2196F3")
        ax.axvspan(t1, t2, alpha=0.08, color="#FF9800")
        ax.axvspan(t2, n,  alpha=0.06, color="#9C27B0")

        ax.plot(ks, scores, color=color, lw=1.8, zorder=3)

        peak_k = ks[np.argmax(scores)]
        peak_v = scores.max()
        ax.axvline(peak_k, color="darkorange", lw=1.6, ls="--", zorder=4, alpha=0.9)

        offset = max(n * 0.05, 1.0)
        txt_x, ha = (peak_k + offset, "left") if peak_k < n * 0.30 else (peak_k - offset, "right")
        ax.annotate(f"k={peak_k}\n{peak_v:.3f}",
                    xy=(peak_k, peak_v * 0.93),
                    xytext=(txt_x, peak_v * 0.83),
                    fontsize=6.5, color="#D84315", fontweight="bold",
                    va="top", ha=ha, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="#E64A19", lw=1.0, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="#E64A19", lw=0.9, alpha=0.8))

        ax.set_title(label, fontsize=10, fontweight="bold", pad=4)
        ax.set_xlabel("Layers injected (k)", fontsize=8, labelpad=4)
        ax.set_ylabel("L-BBQ score", fontsize=8, labelpad=5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)

    legend_els = [
        Patch(facecolor="#2196F3", alpha=0.25, label="First third"),
        Patch(facecolor="#FF9800", alpha=0.35, label="Mid third"),
        Patch(facecolor="#9C27B0", alpha=0.25, label="Last third"),
    ]
    plt.subplots_adjust(bottom=0.12, top=0.93, left=0.07, right=0.98,
                        wspace=0.42, hspace=0.50)
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.02),
               frameon=True, framealpha=0.9, edgecolor="#cccccc",
               handlelength=1.6, columnspacing=1.5)

    _save(fig, "fig_appendix_cumulative_all")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures...")
    plot_sufficiency()
    plot_cumulative()
    plot_necessity()
    plot_cumulative_appendix()
    print("Done.")
