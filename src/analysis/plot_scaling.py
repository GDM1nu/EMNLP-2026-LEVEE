#!/usr/bin/env python3
"""
Figure 5: Scaling analysis.

Two plots:
  A. Peak bias reduction vs. model size (log scale)
     y = base_bias - min(ambig_bias across all layers)
  B. Peak suppression layer index vs. model size
     y = layer_idx at which min ambig_bias occurs (+ thirds label)

Shows whether larger models have stronger / more localized suppression.

Output: results/analysis/plots/scaling_peak_reduction.png
        results/analysis/plots/scaling_peak_depth.png
        results/analysis/plots/scaling_combined.png
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

SWEEP_DIR = Path("results/sweep_thirds")
OUT_DIR   = Path("results/analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_COLOR = {
    "qwen":   "#2171b5",
    "gemma":  "#238b45",
    "llama3": "#d94801",
    "llama32": "#fd8d3c",
}

def get_family(alias: str) -> str:
    if alias.startswith("qwen"):    return "qwen"
    if alias.startswith("gemma"):   return "gemma"
    if alias == "llama3_8b":        return "llama3"
    return "llama32"


def collect_stats() -> list[dict]:
    stats = []
    for alias in MODEL_ORDER:
        cfg = MODELS[alias]
        model_dir = SWEEP_DIR / alias
        if not model_dir.exists():
            continue

        rows = []
        for f in sorted(model_dir.glob("sweep_L*_bbq.json")):
            d = json.loads(f.read_text())
            li = d["layer_idx"]
            rows.append({
                "layer_idx": li,
                "ambig_bias": d["metrics"]["ambig"]["bias_score"],
            })
        if not rows:
            continue

        rows.sort(key=lambda r: r["layer_idx"])
        biases = [r["ambig_bias"] for r in rows]
        best_idx = int(np.argmin(biases))
        base = load_baseline(alias)
        n_layers = cfg["n_layers"]
        peak_layer = rows[best_idx]["layer_idx"]
        peak_third = (
            "first" if peak_layer < n_layers // 3 else
            "mid"   if peak_layer < 2 * n_layers // 3 else
            "last"
        )

        stats.append({
            "alias": alias,
            "label": cfg["label"],
            "size_b": cfg["size_b"],
            "color": cfg["color"],
            "family": get_family(alias),
            "n_layers": n_layers,
            "peak_layer_idx": peak_layer,
            "peak_third": peak_third,
            "min_bias": biases[best_idx],
            "base_bias": base,
            "peak_reduction": (base - biases[best_idx]) if base is not None else None,
            "base_ambig": base,
        })
    return stats


def plot_combined(stats: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    families_seen = set()
    for s in stats:
        fam = s["family"]
        label = fam if fam not in families_seen else None
        families_seen.add(fam)

        # left: peak reduction
        if s["peak_reduction"] is not None:
            ax1.scatter(s["size_b"], s["peak_reduction"],
                        color=FAMILY_COLOR[fam], s=120, zorder=5,
                        edgecolors="white", linewidth=0.8, label=label)
            ax1.annotate(s["label"], (s["size_b"], s["peak_reduction"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8)

        # right: peak layer index (absolute)
        ax2.scatter(s["size_b"], s["peak_layer_idx"],
                    color=FAMILY_COLOR[fam], s=120, zorder=5,
                    edgecolors="white", linewidth=0.8,
                    label=label if s["peak_reduction"] is None else None)
        ax2.annotate(f"{s['label']}\n({s['peak_third']})",
                     (s["size_b"], s["peak_layer_idx"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)

    # regression line for peak reduction (models with baseline)
    red_data = [(s["size_b"], s["peak_reduction"]) for s in stats
                if s["peak_reduction"] is not None and s["size_b"] >= 3.0]
    if len(red_data) >= 3:
        xs = np.log([d[0] for d in red_data])
        ys = [d[1] for d in red_data]
        coef = np.polyfit(xs, ys, 1)
        x_range = np.linspace(min(xs), max(xs), 50)
        ax1.plot(np.exp(x_range), np.polyval(coef, x_range),
                 color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Trend (≥3B)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Model size (B params, log scale)", fontsize=11)
    ax1.set_ylabel("Peak bias reduction (base − min_sweep)", fontsize=11)
    ax1.set_title("A. Peak Suppression Magnitude vs. Model Size", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.set_xscale("log")
    ax2.set_xlabel("Model size (B params, log scale)", fontsize=11)
    ax2.set_ylabel("Peak suppression layer index", fontsize=11)
    ax2.set_title("B. Peak Suppression Layer vs. Model Size", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Scaling Analysis: Task Vector Bias Suppression", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "scaling_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


def print_table(stats: list[dict]):
    print(f"\n{'Alias':<14} {'Size':>6} {'Base':>7} {'Min':>7} {'Reduction':>10} {'Peak L':>8} {'Third':>7}")
    print("-" * 64)
    for s in sorted(stats, key=lambda x: x["size_b"]):
        red = f"{s['peak_reduction']:.4f}" if s["peak_reduction"] is not None else "  N/A  "
        base = f"{s['base_bias']:.4f}" if s["base_bias"] is not None else "  N/A "
        print(f"{s['alias']:<14} {s['size_b']:>5.1f}B {base:>7} {s['min_bias']:>7.4f} {red:>10} {s['peak_layer_idx']:>8} {s['peak_third']:>7}")


if __name__ == "__main__":
    stats = collect_stats()
    print_table(stats)
    print("\n=== scaling_combined ===")
    plot_combined(stats)
    print("=== DONE ===")
