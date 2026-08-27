#!/usr/bin/env python3
"""
Comprehensive visualization for all CPU analysis results that lack plots.
Generates:
  1. bbq_category_localization  → layer_index/bbq_category/
  2. cumulative_marginal        → layer_index/cumulative/
  3. perlayer_signflip_symmetry → layer_index/signflip/
  4. participation_ratio        → layer_index/weight_space/
  5. layer_acf                  → layer_index/weight_space/
  6. ties_sign_conflict         → layer_index/weight_space/
  7. dare_sparsity              → layer_index/weight_space/
"""
from __future__ import annotations
from pathlib import Path
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


PROJ_ROOT = Path(__file__).resolve().parents[2]

BASE     = str(PROJ_ROOT)
ADIR     = f"{BASE}/results/analysis"
PDIR     = f"{ADIR}/plots/layer_index"

THIRD_COLORS = {"first": "#4472C4", "mid": "#C00000", "last": "#70AD47"}
THIRD_LABELS = {"first": "First-third", "mid": "Mid-third", "last": "Last-third"}
MODEL_LABELS = {
    "qwen_0.5b": "Qwen\n0.5B", "qwen_1.5b": "Qwen\n1.5B", "qwen_3b": "Qwen\n3B",
    "qwen_14b": "Qwen\n14B", "gemma_2b": "Gemma\n2B", "gemma_9b": "Gemma\n9B",
    "gemma3_12b": "Gemma3\n12B", "llama32_1b": "LLaMA3.2\n1B",
    "llama32_3b": "LLaMA3.2\n3B", "llama3_8b": "LLaMA3\n8B", "llama31_8b": "LLaMA3.1\n8B",
}

def load(fname: str) -> dict:
    return json.load(open(f"{ADIR}/{fname}"))

def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.replace(BASE+'/', '')}")

def third_bg(ax, n_layers: int, alpha: float = 0.08):
    f = n_layers // 3
    ax.axvspan(-0.5, f - 0.5, color=THIRD_COLORS["first"], alpha=alpha)
    ax.axvspan(f - 0.5, 2*f - 0.5, color=THIRD_COLORS["mid"], alpha=alpha)
    ax.axvspan(2*f - 0.5, n_layers - 0.5, color=THIRD_COLORS["last"], alpha=alpha)


# 1. BBQ Category Localization

def plot_bbq_category():
    print("\n[1] BBQ Category Localization")
    d = load("bbq_category_localization.json")
    results = d["results"]

    all_cats = sorted(set(
        cat for r in results for cat in r["per_category"].keys()
    ))
    models = [r["model"] for r in results]
    model_labels = [MODEL_LABELS.get(m, m) for m in models]

    # Map third → int for colormap
    third_to_int = {"first": 0, "mid": 1, "last": 2}
    cmap = ListedColormap([THIRD_COLORS["first"], THIRD_COLORS["mid"], THIRD_COLORS["last"]])

    mat = np.full((len(all_cats), len(models)), -1, dtype=float)
    for j, r in enumerate(results):
        for i, cat in enumerate(all_cats):
            if cat in r["per_category"]:
                mat[i, j] = third_to_int.get(r["per_category"][cat]["best_third"], -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [4, 1]})

    ax = axes[0]
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, fontsize=8)
    ax.set_yticks(range(len(all_cats)))
    ax.set_yticklabels([c.replace("_", " ") for c in all_cats], fontsize=9)
    ax.set_title("Dominant Third per BBQ Category × Model", fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=THIRD_LABELS[t])
               for t in ["first", "mid", "last"]]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              bbox_to_anchor=(1.0, -0.12), ncol=3)

    # Annotate cells with first letter of third
    for i in range(len(all_cats)):
        for j in range(len(models)):
            if mat[i, j] >= 0:
                label = ["F", "M", "L"][int(mat[i, j])]
                ax.text(j, i, label, ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

    # Summary bar: how many categories each model has as mid
    ax2 = axes[1]
    mid_counts = []
    for r in results:
        cnt = sum(1 for v in r["per_category"].values() if v["best_third"] == "mid")
        mid_counts.append(cnt)
    total_cats = [len(r["per_category"]) for r in results]
    bars = ax2.barh(range(len(models)), mid_counts, color=THIRD_COLORS["mid"], alpha=0.8)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(model_labels, fontsize=8)
    ax2.set_xlabel("# categories\n(mid dominant)", fontsize=9)
    ax2.set_title("Mid-dominant\ncount", fontsize=10)
    for i, (v, total) in enumerate(zip(mid_counts, total_cats)):
        ax2.text(v + 0.1, i, f"{v}/{total}", va="center", fontsize=8)
    ax2.set_xlim(0, max(total_cats) + 1)

    fig.suptitle("BBQ Category-level Bias Localization", fontsize=13, y=1.02)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/bbq_category/bbq_category_heatmap.png")


# 2. Cumulative Marginal Gains

def plot_cumulative_marginal():
    print("\n[2] Cumulative Marginal Gains")
    d = load("cumulative_marginal.json")
    results = d["results"]

    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        pairs = r["layer_pairs"]
        layers = [p["layer_idx"] for p in pairs]
        gains  = [p["marginal_gain"] for p in pairs]
        n_layers = r["n_layers"]
        f = n_layers // 3

        colors = [
            THIRD_COLORS["first"] if l < f else
            THIRD_COLORS["mid"]   if l < 2*f else
            THIRD_COLORS["last"]
            for l in layers
        ]
        ax.bar(layers, gains, color=colors, alpha=0.8, width=0.8)
        ax.axhline(0, color="black", lw=0.6)
        third_bg(ax, n_layers, alpha=0.07)

        rho = r["spearman_sweep_marginal"]
        p   = r["spearman_sweep_marginal_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        dom = r["dominant_third_marginal"]
        ax.set_title(
            f"{MODEL_LABELS.get(r['model'], r['model'])}\nρ={rho:+.2f}{sig}  dom={dom}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Layer idx", fontsize=8)
        ax.set_ylabel("Marginal gain" if idx % ncols == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=THIRD_LABELS[t])
               for t in ["first", "mid", "last"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cumulative Injection Marginal Gains per Layer\n"
                 "(ρ = Spearman vs sweep bias; negative ρ = sweep-effective layers have high marginal gain)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/cumulative/cumulative_marginal_gains.png")


# 3. Per-layer Sign-flip Symmetry

def plot_perlayer_signflip():
    print("\n[3] Per-layer Sign-flip Symmetry")
    d = load("perlayer_signflip_symmetry.json")
    results = d["results"]

    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        metrics = r["layer_metrics"]
        layers  = [m["layer_idx"] for m in metrics]
        scores  = [m["sym_score"] for m in metrics]
        n_layers = r["n_layers"]

        third_bg(ax, n_layers, alpha=0.1)
        ax.plot(layers, scores, color="#333333", lw=1.2, zorder=3)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="perfect sym (1.0)")

        best_sym_third = r["most_symmetric_third"]
        rho = r["spearman_idx_sym"]
        p   = r["spearman_idx_sym_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(
            f"{MODEL_LABELS.get(r['model'], r['model'])}\nbest={best_sym_third}  ρ={rho:+.2f}{sig}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Layer idx", fontsize=8)
        ax.set_ylabel("Sym score (↓=symmetric)" if idx % ncols == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=THIRD_LABELS[t])
               for t in ["first", "mid", "last"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-layer Sign-flip Symmetry Score\n"
                 "(lower = more symmetric = ΔW inversion ↔ behavior inversion)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/signflip/perlayer_sym_score.png")


# 4. Participation Ratio per Layer

def plot_participation_ratio():
    print("\n[4] Participation Ratio")
    d = load("participation_ratio.json")
    results = d["results"]

    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        metrics  = r["layer_metrics"]
        layers   = [m["layer_idx"] for m in metrics]
        pr_vals  = [m["pr"] for m in metrics]
        n_layers = r["n_layers"]

        colors = [
            THIRD_COLORS["first"] if m["layer_idx"] < n_layers//3 else
            THIRD_COLORS["mid"]   if m["layer_idx"] < 2*(n_layers//3) else
            THIRD_COLORS["last"]
            for m in metrics
        ]
        ax.bar(layers, pr_vals, color=colors, alpha=0.8, width=0.8)
        third_bg(ax, n_layers, alpha=0.07)

        dom = r["dominant_third_pr"]
        rho = r["spearman_pr_bias"]
        p   = r["spearman_pr_bias_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(
            f"{MODEL_LABELS.get(r['model'], r['model'])}\ndom={dom}  ρ={rho:+.2f}{sig}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Layer idx", fontsize=8)
        ax.set_ylabel("Participation Ratio" if idx % ncols == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=THIRD_LABELS[t])
               for t in ["first", "mid", "last"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ΔW Participation Ratio per Layer\n"
                 "(higher PR = more directions contribute equally; last-dominant = concentrated energy)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/weight_space/participation_ratio.png")


# 5. Layer ACF Summary

def plot_layer_acf():
    print("\n[5] Layer ACF")
    d = load("layer_acf.json")
    results = d["results"]
    models  = [r["model"] for r in results]
    labels  = [MODEL_LABELS.get(m, m) for m in models]

    acf1      = [r["acf_lag1"]     for r in results]
    acf2      = [r["acf_lag2"]     for r in results]
    spikiness = [r["mid_spikiness"] for r in results]
    min_thirds = [r["min_third"]   for r in results]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ACF lag1
    ax = axes[0]
    bar_colors = [THIRD_COLORS[t] for t in min_thirds]
    ax.bar(x, acf1, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ACF lag-1"); ax.set_title("Bias Profile Autocorrelation (lag 1)\n(positive = smooth transition)")
    for i, v in enumerate(acf1):
        ax.text(i, v + 0.01 * np.sign(v), f"{v:.2f}", ha="center", fontsize=7)

    # ACF lag2
    ax = axes[1]
    ax.bar(x, acf2, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ACF lag-2"); ax.set_title("Bias Profile Autocorrelation (lag 2)")
    for i, v in enumerate(acf2):
        ax.text(i, v + 0.01 * np.sign(v), f"{v:.2f}", ha="center", fontsize=7)

    # Mid spikiness
    ax = axes[2]
    spike_colors = ["#C00000" if s > 1.0 else "#AAAAAA" for s in spikiness]
    ax.bar(x, spikiness, color=spike_colors, alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="spikiness=1 (baseline)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mid spikiness"); ax.set_title("Mid-third Spikiness\n(>1 = sharper dip than global avg)")
    ax.legend(fontsize=8)
    for i, v in enumerate(spikiness):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=f"min_third={t}")
               for t in ["first", "mid", "last"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Layer-distance ACF of Sweep Bias Profile", fontsize=12, y=1.02)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/weight_space/layer_acf_summary.png")


# 6. TIES Sign Conflict — per-layer coherence

def plot_ties_sign_conflict():
    print("\n[6] TIES Sign Conflict")
    d = load("ties_sign_conflict.json")
    results = d["results"]

    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        metrics  = r["layer_metrics"]
        layers   = [m["layer_idx"] for m in metrics]
        coh      = [m["sign_coherence"] for m in metrics]
        n_layers = r["n_layers"]

        colors = [
            THIRD_COLORS["first"] if m["layer_idx"] < n_layers//3 else
            THIRD_COLORS["mid"]   if m["layer_idx"] < 2*(n_layers//3) else
            THIRD_COLORS["last"]
            for m in metrics
        ]
        ax.bar(layers, [v * 1000 for v in coh], color=colors, alpha=0.8, width=0.8)
        third_bg(ax, n_layers, alpha=0.07)

        dom = r["dominant_coherence"]
        rho = r["spearman_coherence_bias"]
        p   = r["spearman_coherence_bias_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(
            f"{MODEL_LABELS.get(r['model'], r['model'])}\ndom={dom}  ρ={rho:+.2f}{sig}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Layer idx", fontsize=8)
        ax.set_ylabel("Sign coherence (×10⁻³)" if idx % ncols == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=THIRD_LABELS[t])
               for t in ["first", "mid", "last"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("TIES Sign Coherence per Layer\n"
                 "(≈ 0 everywhere = full-rank ΔW, 50/50 sign split; no directional structure)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/weight_space/ties_sign_coherence.png")


# 7. DARE Sparsity (mean_abs per layer)

def plot_dare_sparsity():
    print("\n[7] DARE Sparsity")
    d = load("dare_sparsity.json")
    results = d["results"]

    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        metrics  = r["layer_metrics"]
        layers   = [m["layer_idx"] for m in metrics]
        mean_abs = [m["mean_abs"] for m in metrics]
        n_layers = r["n_layers"]
        f = n_layers // 3

        colors = [
            THIRD_COLORS["first"] if l < f else
            THIRD_COLORS["mid"]   if l < 2*f else
            THIRD_COLORS["last"]
            for l in layers
        ]
        ax.bar(layers, mean_abs, color=colors, alpha=0.8, width=0.8)
        third_bg(ax, n_layers, alpha=0.07)

        dom = r["dominant_third_high_magnitude"]
        rho = r["spearman_meanabs_bias"]
        p   = r["spearman_meanabs_bias_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(
            f"{MODEL_LABELS.get(r['model'], r['model'])}\ndom={dom}  ρ={rho:+.2f}{sig}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Layer idx", fontsize=8)
        ax.set_ylabel("Mean |ΔW|" if idx % ncols == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    patches = [mpatches.Patch(color=THIRD_COLORS[t], label=THIRD_LABELS[t])
               for t in ["first", "mid", "last"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("DARE Sparsity: Mean |ΔW| per Layer\n"
                 "(last-dominant = weight magnitude concentrated late, same as dw_norm/PR)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    savefig(fig, f"{PDIR}/weight_space/dare_sparsity.png")


if __name__ == "__main__":
    plot_bbq_category()
    plot_cumulative_marginal()
    plot_perlayer_signflip()
    plot_participation_ratio()
    plot_layer_acf()
    plot_ties_sign_conflict()
    plot_dare_sparsity()
    print("\nAll done.")
