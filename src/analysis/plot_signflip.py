#!/usr/bin/env python3
"""
Figure 4: Causal direction — sign-flipped ΔW vs. actual ΔW sweep.

Plots per-layer bias for:
  - actual ΔW (from sweep_thirds)
  - sign-flipped −ΔW (from signflip/)

If actual ΔW suppresses bias (↓) and sign-flip amplifies bias (↑),
this confirms the direction is causal, not spurious.

Output: results/analysis/plots/signflip_<alias>.png
        results/analysis/plots/signflip_grid.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.analysis.model_registry import MODEL_ORDER, MODELS, load_baseline

SWEEP_DIR   = Path("results/sweep_thirds")
SIGNFLIP_DIR = Path("results/signflip")
OUT_DIR     = Path("results/analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sweep_bias(result_dir: Path, alias: str) -> dict[int, float] | None:
    model_dir = result_dir / alias
    if not model_dir.exists():
        return None
    rows = {}
    for f in sorted(model_dir.glob("sweep_L*_bbq.json")):
        d = json.loads(f.read_text())
        rows[d["layer_idx"]] = d["metrics"]["ambig"]["bias_score"]
    return rows if rows else None


def plot_single(alias: str, ax=None) -> bool:
    pos = load_sweep_bias(SWEEP_DIR, alias)
    neg = load_sweep_bias(SIGNFLIP_DIR, alias)
    base = load_baseline(alias)

    if pos is None and neg is None:
        return False

    cfg = MODELS[alias]
    n_layers = cfg["n_layers"]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    if base is not None:
        ax.axhline(base, color="#2166ac", linestyle="--", linewidth=1.2,
                   label=f"Base ({base:.3f})", alpha=0.7)

    if pos:
        pos_layers = sorted(pos.keys())
        pos_biases = [pos[l] for l in pos_layers]
        ax.plot(pos_layers, pos_biases, color=cfg["color"], linewidth=2.5,
                label="+ΔW (suppress bias)", zorder=4)
        best = int(np.argmin(pos_biases))
        ax.scatter([pos_layers[best]], [pos_biases[best]], color=cfg["color"],
                   s=60, zorder=5, edgecolors="white", linewidth=0.8)

    if neg:
        neg_layers = sorted(neg.keys())
        neg_biases = [neg[l] for l in neg_layers]
        ax.plot(neg_layers, neg_biases, color="#d73027", linewidth=2,
                linestyle="--", label="−ΔW (sign-flip)", zorder=3)
        worst = int(np.argmax(neg_biases))
        ax.scatter([neg_layers[worst]], [neg_biases[worst]], color="#d73027",
                   s=60, zorder=5, edgecolors="white", linewidth=0.8)

    ax.set_xlim(0, n_layers - 1)
    ax.set_xlabel("Layer index", fontsize=9)
    ax.set_ylabel("Ambig bias score", fontsize=9)
    ax.set_title(cfg["label"], fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    if own_fig:
        fig.tight_layout()
        out = OUT_DIR / f"signflip_{alias}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved → {out}")

    return True


def plot_grid():
    aliases = [a for a in MODEL_ORDER
               if (SWEEP_DIR / a).exists() or (SIGNFLIP_DIR / a).exists()]
    if not aliases:
        print("[skip] no data")
        return

    ncols = 3
    nrows = (len(aliases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes_flat = np.array(axes).flatten() if len(aliases) > 1 else [axes]

    for i, alias in enumerate(aliases):
        plot_single(alias, ax=axes_flat[i])

    for j in range(len(aliases), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Sign-Flip Validation: +ΔW (bias suppression) vs. −ΔW (sign-flipped)\n"
        "Asymmetric response confirms causal direction specificity",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUT_DIR / "signflip_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    print("=== individual signflip plots ===")
    for alias in MODEL_ORDER:
        plot_single(alias)
    print("=== signflip grid ===")
    plot_grid()
    print("=== DONE ===")
