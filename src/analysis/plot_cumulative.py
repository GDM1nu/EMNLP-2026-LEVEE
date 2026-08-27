#!/usr/bin/env python3
"""
Figure 3: Cumulative formation process.

Plots bias score as layers L0..k are incrementally injected.
X-axis: fraction of layers injected (k / n_layers).
Reference: base model bias (dashed) and full-diff effect (horizontal marker).

Output: results/analysis/plots/cumulative_<alias>.png
        results/analysis/plots/cumulative_grid.png
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

CUMUL_DIR = Path("results/cumulative")
SWEEP_DIR = Path("results/sweep_thirds")
OUT_DIR = Path("results/analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cumulative(alias: str) -> list[dict] | None:
    model_dir = CUMUL_DIR / alias
    if not model_dir.exists():
        return None
    rows = []
    for f in sorted(model_dir.glob("cumul_L*_bbq.json")):
        d = json.loads(f.read_text())
        rows.append({
            "layer_idx": d.get("cumul_k", d.get("layer_idx")),
            "ambig_bias": d["metrics"]["ambig"]["bias_score"],
        })
    rows.sort(key=lambda r: r["layer_idx"])
    return rows if rows else None


def load_full_diff_bias(alias: str) -> float | None:
    """Bias when full diff is applied (last entry of cumulative_summary)."""
    summary = CUMUL_DIR / alias / "cumulative_summary.json"
    if summary.exists():
        d = json.loads(summary.read_text())
        layers = d.get("layers", [])
        if layers:
            return layers[-1].get("ambig_bias")
    return None


def plot_single(alias: str, ax=None) -> bool:
    rows = load_cumulative(alias)
    if rows is None:
        return False

    cfg = MODELS[alias]
    n_layers = cfg["n_layers"]
    base = load_baseline(alias)
    full_bias = load_full_diff_bias(alias)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    ks = [r["layer_idx"] for r in rows]
    biases = [r["ambig_bias"] for r in rows]

    # thirds boundaries
    t1 = n_layers // 3
    t2 = 2 * n_layers // 3
    ax.axvspan(-0.5, t1 - 0.5, color="#deebf7", alpha=0.4, zorder=0)
    ax.axvspan(t1 - 0.5, t2 - 0.5, color="#fee8c8", alpha=0.4, zorder=0)
    ax.axvspan(t2 - 0.5, n_layers - 0.5, color="#e5f5e0", alpha=0.4, zorder=0)
    ax.axvline(t1, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(t2, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.plot(ks, biases, color=cfg["color"], linewidth=2.5,
            label="Cumulative injection", zorder=3)

    if base is not None:
        ax.axhline(base, color="#2166ac", linestyle="--", linewidth=1.2,
                   label=f"Base ({base:.3f})")
    if full_bias is not None:
        ax.axhline(full_bias, color="#d73027", linestyle="-.", linewidth=1.2,
                   label=f"Full diff ({full_bias:.3f})")

    # inflection detection (largest single-layer drop)
    if len(biases) > 1:
        drops = [biases[i] - biases[i+1] for i in range(len(biases)-1)]
        max_drop_idx = int(np.argmax(drops))
        ax.scatter([ks[max_drop_idx]], [biases[max_drop_idx]],
                   color=cfg["color"], s=60, zorder=5,
                   edgecolors="white", linewidth=0.8,
                   label=f"Max drop at L{ks[max_drop_idx]}")

    # annotations for thirds
    ax.text(t1 / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.01,
            "First", ha="center", fontsize=7, color="#2166ac", alpha=0.7)
    ax.text((t1 + t2) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.01,
            "Mid", ha="center", fontsize=7, color="#d95f02", alpha=0.7)
    ax.text((t2 + n_layers) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.01,
            "Last", ha="center", fontsize=7, color="#1a9641", alpha=0.7)

    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xlabel("Layers injected (cumulative, L0..k)", fontsize=9)
    ax.set_ylabel("Ambig bias score", fontsize=9)
    ax.set_title(f"{cfg['label']}  ({len(rows)}/{n_layers} layers)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    if own_fig:
        fig.tight_layout()
        out = OUT_DIR / f"cumulative_{alias}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved → {out}")

    return True


def plot_grid():
    aliases = [a for a in MODEL_ORDER if (CUMUL_DIR / a).exists()
               and any((CUMUL_DIR / a).glob("cumul_L*_bbq.json"))]
    if not aliases:
        print("[skip] no cumulative data yet")
        return

    n = len(aliases)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    plotted = 0
    for i, alias in enumerate(aliases):
        if plot_single(alias, ax=axes_flat[i]):
            plotted += 1

    for j in range(len(aliases), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if plotted == 0:
        plt.close(fig)
        return

    fig.suptitle(
        "Cumulative Layer Injection: Bias Formation Process\n"
        "(blue=first third, orange=mid third, green=last third)",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUT_DIR / "cumulative_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    print("=== individual cumulative plots ===")
    for alias in MODEL_ORDER:
        plot_single(alias)
    print("=== cumulative grid ===")
    plot_grid()
    print("=== DONE ===")
