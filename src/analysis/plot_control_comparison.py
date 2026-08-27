#!/usr/bin/env python3
"""
Figure 2: Direction specificity — sweep vs. control conditions.

For each model with control_sweep data, plots per-layer ambig bias for:
  - sweep (actual diff)
  - norm_matched mean ± std (3 seeds)
  - shuffled mean ± std (3 seeds)
  - zero

Output: results/analysis/plots/control_comparison_<alias>.png
        results/analysis/plots/control_comparison_grid.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.analysis.model_registry import MODELS, load_baseline

SWEEP_DIR = Path("results/sweep_thirds")
CTRL_DIR = Path("results/control_sweep")
OUT_DIR = Path("results/analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sweep(alias: str) -> dict[int, float]:
    rows = {}
    for f in sorted((SWEEP_DIR / alias).glob("sweep_L*_bbq.json")):
        d = json.loads(f.read_text())
        rows[d["layer_idx"]] = d["metrics"]["ambig"]["bias_score"]
    return rows


def load_control(alias: str) -> dict[str, dict[int, list[float]]]:
    """Returns {control_type: {layer_idx: [values_per_seed]}}"""
    ctrl_path = CTRL_DIR / alias
    if not ctrl_path.exists():
        return {}

    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in ctrl_path.glob("ctrl_*_bbq.json"):
        d = json.loads(f.read_text())
        ctype = d.get("control_type", "unknown")
        layer = d.get("layer_idx")
        bs = d["metrics"]["ambig"]["bias_score"]
        if layer is not None:
            data[ctype][layer].append(bs)
    return {k: dict(v) for k, v in data.items()}


def plot_single(alias: str, ax=None, show_legend: bool = True) -> bool:
    sweep = load_sweep(alias)
    ctrl = load_control(alias)
    base = load_baseline(alias)

    if not sweep:
        return False

    cfg = MODELS[alias]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5))

    n_layers = cfg["n_layers"]
    layers = sorted(sweep.keys())

    # base reference
    if base is not None:
        ax.axhline(base, color="#2166ac", linestyle="--", linewidth=1.2,
                   label=f"Base ({base:.3f})", alpha=0.7)

    # sweep
    ax.plot(layers, [sweep[l] for l in layers],
            color=cfg["color"], linewidth=2.5, label="Actual ΔW", zorder=4)

    ctrl_styles = {
        "norm_matched": ("#e08214", "-", "Norm-matched"),
        "shuffled":     ("#8073ac", "--", "Shuffled"),
        "zero":         ("#999999", ":", "Zero"),
    }

    for ctype, (color, ls, label) in ctrl_styles.items():
        if ctype not in ctrl:
            continue
        cdata = ctrl[ctype]
        c_layers = sorted(cdata.keys())

        if ctype == "zero":
            vals = [cdata[l][0] for l in c_layers]
            ax.plot(c_layers, vals, color=color, linestyle=ls,
                    linewidth=1.2, label=label, alpha=0.8)
        else:
            means = [np.mean(cdata[l]) for l in c_layers]
            stds = [np.std(cdata[l]) for l in c_layers]
            ax.plot(c_layers, means, color=color, linestyle=ls,
                    linewidth=1.5, label=f"{label} (μ±σ)", alpha=0.9)
            ax.fill_between(c_layers,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.15)

    ax.set_xlim(0, n_layers - 1)
    ax.set_xlabel("Layer index", fontsize=9)
    ax.set_ylabel("Ambig bias score", fontsize=9)
    ax.set_title(cfg["label"], fontsize=10)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")

    if own_fig:
        fig.tight_layout()
        out = OUT_DIR / f"control_comparison_{alias}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved → {out}")

    return True


def plot_grid():
    # only models with control_sweep data
    aliases = [a for a in MODELS if (CTRL_DIR / a).exists()
               and any((CTRL_DIR / a).glob("ctrl_*_bbq.json"))]
    if not aliases:
        print("[skip] no control_sweep data yet")
        return

    n = len(aliases)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, alias in enumerate(aliases):
        plot_single(alias, ax=axes[i], show_legend=(i == 0))

    for j in range(len(aliases), len(axes)):
        axes[j].set_visible(False)

    # shared legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Direction Specificity: Actual ΔW vs. Control Conditions",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "control_comparison_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    print("=== individual control comparison plots ===")
    for alias in MODELS:
        if (CTRL_DIR / alias).exists():
            plot_single(alias)
    print("=== control comparison grid ===")
    plot_grid()
    print("=== DONE ===")
