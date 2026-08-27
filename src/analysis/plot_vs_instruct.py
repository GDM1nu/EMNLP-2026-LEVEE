#!/usr/bin/env python3
"""
Plot task-vector sweep results relative to instruct baseline.

Three plot types:
  1. normalized_reduction_overlay.png
     y = (base_bias - sweep_bias) / (base_bias - instruct_bias)
     0 = no effect, 1.0 = full instruct effect, >1 = overshot

  2. per-model layer curves with base + instruct reference lines
     showing the "gap" between base and instruct as a shaded band

  3. gap_to_instruct_overlay.png
     y = sweep_bias - instruct_bias  (how far each layer still is from instruct)
"""
from __future__ import annotations
from pathlib import Path

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PROJ_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = str(ARCHIVE)
SWEEP_ROOT = str(PROJ_ROOT / "results/sweep_thirds")
OUT_DIR = str(PROJ_ROOT / "results/plots")

# Model registry
MODELS = {
    "qwen_0.5b": dict(
        label="Qwen 0.5B", color="#9ecae1", n_layers=24,
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_0.5b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_0.5b/baselines/baseline_instruct_bbq.json",
        base_key="ambig", inst_key="ambig",
    ),
    "qwen_1.5b": dict(
        label="Qwen 1.5B", color="#6baed6", n_layers=28,
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_1.5b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_1.5b/baselines/baseline_instruct_bbq.json",
        base_key="ambig", inst_key="ambig",
    ),
    "qwen_3b": dict(
        label="Qwen 3B", color="#2171b5", n_layers=36,
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_3b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_3b/baselines/baseline_instruct_bbq.json",
        base_key="ambig", inst_key="ambig",
    ),
    "qwen_14b": dict(
        label="Qwen 14B", color="#084594", n_layers=48,
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_14b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_14b/baselines/baseline_instruct_bbq.json",
        base_key="ambig", inst_key="ambig",
    ),
    "gemma_2b": dict(
        label="Gemma 2B", color="#a1d99b", n_layers=26,
        base_json=f"{ARCHIVE}/scaling_experiment/gemma_2b/baselines/baseline_gemma2_2b_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/gemma_2b/baselines/baseline_gemma2_2b_it_bbq.json",
        base_key="ambig", inst_key="ambig",
    ),
    "gemma_9b": dict(
        label="Gemma 9B", color="#238b45", n_layers=42,
        base_json=f"{ARCHIVE}/scaling_experiment/gemma_9b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/gemma_9b/baselines/baseline_instruct_bbq.json",
        base_key="ambig", inst_key="ambig",
    ),
    "llama3_8b": dict(
        label="LLaMA 3-8B", color="#d94801", n_layers=32,
        base_json=f"{ARCHIVE}/llama_experiment/results/baselines_logits/bbq_metrics_llama3_8b_base_logits.json",
        inst_json=f"{ARCHIVE}/llama_experiment/results/baselines_logits/bbq_metrics_llama3_8b_instruct_logits.json",
        base_key="ambig", inst_key="ambig",
        flat_metrics=True,   # no outer 'metrics' key
    ),
    # LLaMA 3.2 — no instruct baselines
    "llama32_3b": dict(
        label="LLaMA 3.2-3B", color="#fd8d3c", n_layers=28,
        base_json=None, inst_json=None,
    ),
    "llama32_1b": dict(
        label="LLaMA 3.2-1B", color="#fdae6b", n_layers=16,
        base_json=None, inst_json=None,
    ),
}


# Data loading helpers

def load_bias(path: str, flat_metrics: bool = False) -> float:
    d = json.load(open(path))
    if flat_metrics:
        return d["ambig"]["bias_score"]
    return d["metrics"]["ambig"]["bias_score"]


def load_sweep(alias: str) -> list[dict]:
    """Return sorted list of {layer_idx, ambig_bias} from per-layer JSON files."""
    pattern = os.path.join(SWEEP_ROOT, alias, "sweep_L*_bbq.json")
    rows = []
    for p in glob.glob(pattern):
        d = json.load(open(p))
        rows.append(dict(
            layer_idx=d["layer_idx"],
            ambig_bias=d["metrics"]["ambig"]["bias_score"],
        ))
    rows.sort(key=lambda x: x["layer_idx"])
    return rows


# Plot 1: normalized reduction overlay

def plot_normalized_overlay():
    fig, ax = plt.subplots(figsize=(10, 6))

    has_plot = False
    for alias, cfg in MODELS.items():
        if cfg["base_json"] is None or not os.path.exists(cfg["base_json"]):
            continue
        if cfg["inst_json"] is None or not os.path.exists(cfg["inst_json"]):
            continue

        flat = cfg.get("flat_metrics", False)
        base_bias = load_bias(cfg["base_json"], flat)
        inst_bias = load_bias(cfg["inst_json"], flat)
        gap = base_bias - inst_bias
        if abs(gap) < 1e-6:
            continue

        rows = load_sweep(alias)
        if not rows:
            continue

        layers = [r["layer_idx"] for r in rows]
        norm_red = [(base_bias - r["ambig_bias"]) / gap for r in rows]

        ax.plot(layers, norm_red, color=cfg["color"], linewidth=2,
                label=cfg["label"], alpha=0.9)
        # mark best layer
        best_idx = int(np.argmax(norm_red))
        ax.scatter([layers[best_idx]], [norm_red[best_idx]],
                   color=cfg["color"], s=60, zorder=5)
        has_plot = True

    if not has_plot:
        print("[skip] no data for normalized overlay")
        return

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Base (no effect)")
    ax.axhline(1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.8,
               label="Full Instruct")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("Normalized Bias Reduction\n(0=no effect, 1=full instruct)", fontsize=12)
    ax.set_title("Single-Layer Task Vector: Reduction Toward Instruct Baseline", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "normalized_reduction_overlay.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved → {out}")


# Plot 2: gap to instruct overlay

def plot_gap_to_instruct():
    fig, ax = plt.subplots(figsize=(10, 6))

    for alias, cfg in MODELS.items():
        if cfg["base_json"] is None or not os.path.exists(cfg.get("inst_json") or ""):
            continue

        flat = cfg.get("flat_metrics", False)
        inst_bias = load_bias(cfg["inst_json"], flat)

        rows = load_sweep(alias)
        if not rows:
            continue

        layers = [r["layer_idx"] for r in rows]
        gap = [r["ambig_bias"] - inst_bias for r in rows]

        ax.plot(layers, gap, color=cfg["color"], linewidth=2,
                label=cfg["label"], alpha=0.9)
        best_idx = int(np.argmin([abs(g) for g in gap]))
        ax.scatter([layers[best_idx]], [gap[best_idx]],
                   color=cfg["color"], s=60, zorder=5)

    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.5,
               label="Instruct baseline (gap=0)")
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("sweep_bias − instruct_bias", fontsize=12)
    ax.set_title("Remaining Gap to Instruct Baseline per Layer", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "gap_to_instruct_overlay.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved → {out}")


# Plot 3: per-model layer curves with reference lines

def plot_individual_with_refs():
    for alias, cfg in MODELS.items():
        rows = load_sweep(alias)
        if not rows:
            continue

        layers = [r["layer_idx"] for r in rows]
        biases = [r["ambig_bias"] for r in rows]
        n_layers = cfg["n_layers"]

        flat = cfg.get("flat_metrics", False)
        base_bias = (load_bias(cfg["base_json"], flat)
                     if cfg.get("base_json") and os.path.exists(cfg["base_json"]) else None)
        inst_bias = (load_bias(cfg["inst_json"], flat)
                     if cfg.get("inst_json") and os.path.exists(cfg["inst_json"] or "") else None)

        fig, ax = plt.subplots(figsize=(9, 5))

        # shaded region between base and instruct
        if base_bias is not None and inst_bias is not None:
            ax.axhspan(inst_bias, base_bias,
                       color="gold", alpha=0.15, label="Base→Instruct range")

        ax.plot(layers, biases, color=cfg["color"], linewidth=2.5,
                label="Single-layer sweep")

        # best layer marker
        best_idx = int(np.argmin(biases))
        ax.scatter([layers[best_idx]], [biases[best_idx]],
                   color=cfg["color"], s=80, zorder=6,
                   label=f"Best layer (L{layers[best_idx]}, bias={biases[best_idx]:.4f})")

        if base_bias is not None:
            ax.axhline(base_bias, color="steelblue", linestyle="--",
                       linewidth=1.5, label=f"Base model ({base_bias:.4f})")
        if inst_bias is not None:
            ax.axhline(inst_bias, color="forestgreen", linestyle="-.",
                       linewidth=1.5, label=f"Instruct model ({inst_bias:.4f})")

            # annotate gap at best layer
            gap_pct = (base_bias - biases[best_idx]) / max(base_bias - inst_bias, 1e-6) * 100
            ax.annotate(
                f"{gap_pct:.0f}% of instruct\neffect achieved",
                xy=(layers[best_idx], biases[best_idx]),
                xytext=(layers[best_idx] + n_layers * 0.05,
                        biases[best_idx] + (base_bias - inst_bias) * 0.3),
                fontsize=8, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
            )

        ax.set_xlabel("Layer index", fontsize=11)
        ax.set_ylabel("Ambig Bias Score", fontsize=11)
        ax.set_title(f"{cfg['label']} — Layer-wise Task Vector vs. Instruct Gap", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, n_layers - 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = os.path.join(OUT_DIR, f"{alias}_depth_curve.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved → {out}")


# Plot 4: strong models side-by-side (2×3 grid)

def plot_strong_grid():
    strong = ["qwen_3b", "gemma_2b", "gemma_9b", "llama3_8b", "qwen_14b", "llama32_3b"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, alias in enumerate(strong):
        ax = axes[i]
        cfg = MODELS[alias]
        rows = load_sweep(alias)
        if not rows:
            ax.set_visible(False); continue

        layers = [r["layer_idx"] for r in rows]
        biases = [r["ambig_bias"] for r in rows]
        n_layers = cfg["n_layers"]
        flat = cfg.get("flat_metrics", False)

        base_bias = (load_bias(cfg["base_json"], flat)
                     if cfg.get("base_json") and os.path.exists(cfg["base_json"]) else None)
        inst_bias = (load_bias(cfg["inst_json"], flat)
                     if cfg.get("inst_json") and os.path.exists(cfg.get("inst_json") or "") else None)

        if base_bias is not None and inst_bias is not None:
            ax.axhspan(inst_bias, base_bias, color="gold", alpha=0.18)

        ax.plot(layers, biases, color=cfg["color"], linewidth=2)
        best_idx = int(np.argmin(biases))
        ax.scatter([layers[best_idx]], [biases[best_idx]],
                   color=cfg["color"], s=70, zorder=5)

        if base_bias is not None:
            ax.axhline(base_bias, color="steelblue", linestyle="--", linewidth=1.2)
            ax.text(0.02, base_bias, "base", va="bottom", ha="left",
                    fontsize=7, color="steelblue", transform=ax.get_yaxis_transform())
        if inst_bias is not None:
            ax.axhline(inst_bias, color="forestgreen", linestyle="-.", linewidth=1.2)
            ax.text(0.02, inst_bias, "instruct", va="top", ha="left",
                    fontsize=7, color="forestgreen", transform=ax.get_yaxis_transform())
            gap_pct = (base_bias - biases[best_idx]) / max(base_bias - inst_bias, 1e-6) * 100
            ax.set_title(f"{cfg['label']}  (best: {gap_pct:.0f}% of instruct Δ)", fontsize=10)
        else:
            ax.set_title(cfg["label"], fontsize=10)

        ax.set_xlim(0, n_layers - 1)
        ax.set_xlabel("Layer index", fontsize=9)
        ax.set_ylabel("Ambig Bias", fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Single-Layer Sweep vs. Base/Instruct Gap (gold = base→instruct range)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "strong_models_vs_instruct.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Normalized reduction overlay ===")
    plot_normalized_overlay()
    print("=== Gap to instruct overlay ===")
    plot_gap_to_instruct()
    print("=== Individual layer curves (with ref lines) ===")
    plot_individual_with_refs()
    print("=== Strong models grid ===")
    plot_strong_grid()
    print("=== DONE ===")
