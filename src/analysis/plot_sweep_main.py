#!/usr/bin/env python3
"""
Figure 1: Per-layer bias sweep curves for all 9 models (3×3 grid).

Each subplot shows ambig bias_score vs normalized depth (layer_idx / n_layers).
Reference lines: base model bias (blue dashed), instruct model bias (green dash-dot).
Peak suppression layer marked with a dot.

Output: results/analysis/plots/sweep_main_grid.png
        results/analysis/plots/sweep_overlay.png
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
from src.analysis.model_registry import MODEL_ORDER, MODELS, load_baseline, load_instruct

SWEEP_DIR = Path("results/sweep_thirds")
OUT_DIR = Path("results/analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sweep(alias: str) -> list[dict] | None:
    model_dir = SWEEP_DIR / alias
    rows = []
    for f in sorted(model_dir.glob("sweep_L*_bbq.json")):
        d = json.loads(f.read_text())
        li = d["layer_idx"]
        rows.append({
            "layer_idx": li,
            "ambig_bias": d["metrics"]["ambig"]["bias_score"],
            "disambig_bias": d["metrics"]["disambig"]["bias_score"],
        })
    rows.sort(key=lambda r: r["layer_idx"])
    return rows if rows else None


def plot_grid():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, alias in enumerate(MODEL_ORDER):
        ax = axes[i]
        cfg = MODELS[alias]
        rows = load_sweep(alias)

        if rows is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(cfg["label"], fontsize=11)
            continue

        layers = [r["layer_idx"] for r in rows]
        ambig = [r["ambig_bias"] for r in rows]
        disambig = [r["disambig_bias"] for r in rows]
        n_layers = cfg["n_layers"]

        base = load_baseline(alias)
        inst = load_instruct(alias)

        # shaded region
        if base is not None and inst is not None:
            ax.axhspan(min(base, inst), max(base, inst),
                       color="#fff3cd", alpha=0.6, zorder=0)

        ax.plot(layers, ambig, color=cfg["color"], linewidth=2,
                label="Ambig (swept)", zorder=3)
        ax.plot(layers, disambig, color=cfg["color"], linewidth=1,
                linestyle=":", alpha=0.5, label="Disambig (swept)", zorder=2)

        # peak suppression marker
        best_idx = int(np.argmin(ambig))
        ax.scatter([layers[best_idx]], [ambig[best_idx]],
                   color=cfg["color"], s=60, zorder=5, edgecolors="white", linewidth=0.8)

        if base is not None:
            ax.axhline(base, color="#2166ac", linestyle="--", linewidth=1.2,
                       label=f"Base ({base:.3f})", zorder=4)
        if inst is not None:
            ax.axhline(inst, color="#1a9641", linestyle="-.", linewidth=1.2,
                       label=f"Instruct ({inst:.3f})", zorder=4)

        peak_layer = layers[best_idx]
        ax.set_title(f"{cfg['label']}  (peak: L{peak_layer})", fontsize=10)
        ax.set_xlim(0, n_layers - 1)
        ax.set_xlabel("Layer index", fontsize=8)
        ax.set_ylabel("Ambig bias score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="upper right", ncol=1)

    fig.suptitle(
        "Layer-wise Task Vector Injection: BBQ Ambiguous Bias Score\n"
        "(dot = peak suppression layer; shaded = base→instruct range)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "sweep_main_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    print("=== sweep_main_grid ===")
    plot_grid()
    print("=== DONE ===")
