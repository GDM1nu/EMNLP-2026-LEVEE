"""
Per-layer ΔW Cosine Similarity Analysis

Three analyses in one pass:
  1. Per-layer L2 norm vs BBQ ambig bias_score correlation
  2. cos_sim(ΔW_l, ΔW_global_mean) vs BBQ bias_score correlation
  3. Cross-model thirds pattern comparison (equal thirds, Rogers 2020)

Output:
  results/analysis/cosine_sim.pkl   — full data
  results/analysis/cosine_sim.json  — summary (correlations, peak layers)
  results/analysis/plots/cosine_*   — visualizations
"""
import re, json, pickle, os
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJ_ROOT = Path(__file__).resolve().parents[2]


# Config

MODELS = {
    "qwen_0.5b":  {"n_layers": 24, "family": "qwen",
                   "diff": str(ARCHIVE / "scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt")},
    "qwen_1.5b":  {"n_layers": 28, "family": "qwen",
                   "diff": str(ARCHIVE / "scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt")},
    "qwen_3b":    {"n_layers": 36, "family": "qwen",
                   "diff": str(ARCHIVE / "scaling_experiment/qwen_3b/diff_vector/diff_vector.pt")},
    "qwen_14b":   {"n_layers": 48, "family": "qwen",
                   "diff": str(ARCHIVE / "scaling_experiment/qwen_14b/diff_vector/diff_vector.pt")},
    "gemma_2b":   {"n_layers": 26, "family": "gemma",
                   "diff": str(ARCHIVE / "scaling_experiment/gemma_2b/diff_vector/diff_vector.pt")},
    "gemma_9b":   {"n_layers": 42, "family": "gemma",
                   "diff": str(ARCHIVE / "scaling_experiment/gemma_9b/diff_vector/diff_vector.pt")},
    "llama32_1b": {"n_layers": 16, "family": "llama",
                   "diff": str(ARCHIVE / "llama_experiment/models/llama32_1b_diff/llama32_1b_diff.pt")},
    "llama32_3b": {"n_layers": 28, "family": "llama",
                   "diff": str(ARCHIVE / "llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt")},
    "llama3_8b":  {"n_layers": 32, "family": "llama",
                   "diff": str(ARCHIVE / "llama_experiment/models/llama_diff_vector/llama_diff_vector.pt")},
}

SWEEP_DIR  = Path(str(PROJ_ROOT / "results/sweep_thirds"))
OUT_DIR    = Path(str(PROJ_ROOT / "results/analysis"))
PLOT_DIR   = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# baseline BBQ ambig bias_score per model
ARCHIVE = ARCHIVE
BASELINES = {
    "qwen_0.5b":  ARCHIVE / "scaling_experiment/qwen_0.5b/baselines/baseline_base_bbq.json",
    "qwen_1.5b":  ARCHIVE / "scaling_experiment/qwen_1.5b/baselines/baseline_base_bbq.json",
    "qwen_3b":    ARCHIVE / "scaling_experiment/qwen_3b/baselines/baseline_base_bbq.json",
    "qwen_14b":   ARCHIVE / "scaling_experiment/qwen_14b/baselines/baseline_base_bbq.json",
    "gemma_2b":   ARCHIVE / "scaling_experiment/gemma_2b/baselines/baseline_gemma2_2b_base_bbq.json",
    "gemma_9b":   ARCHIVE / "scaling_experiment/gemma_9b/baselines/baseline_base_bbq.json",
    "llama3_8b":  ARCHIVE / "llama_experiment/results/baselines/bbq_metrics_llama3_8b_base.json",
    "llama32_1b": None,   # baseline not available
    "llama32_3b": None,
}

def load_baseline_bias(alias: str) -> float | None:
    path = BASELINES.get(alias)
    if path is None or not Path(path).exists():
        return None
    try:
        d = json.loads(Path(path).read_text())
        # support two JSON structures
        if "metrics" in d:
            return float(d["metrics"]["ambig"]["bias_score"])
        return float(d["ambig"]["bias_score"])
    except Exception:
        return None


# Helpers

def layer_vectors(diff: dict, n_layers: int) -> list[torch.Tensor]:
    """Flatten and concat all parameters per layer into a list of (D_l,) tensors."""
    buckets = defaultdict(list)
    for key, tensor in diff.items():
        m = re.search(r"\.(\d+)\.", key)
        if m:
            idx = int(m.group(1))
            if idx < n_layers:
                buckets[idx].append(tensor.float().reshape(-1))
    return [torch.cat(buckets[i]) if i in buckets else torch.zeros(1)
            for i in range(n_layers)]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return (a @ b / (na * nb)).item()


def load_bbq_sweep(alias: str, n_layers: int) -> list[float | None]:
    """Load per-layer ambig bias_score; incomplete sweep layers return None."""
    scores = [None] * n_layers
    sweep_path = SWEEP_DIR / alias
    if not sweep_path.exists():
        return scores
    for f in sweep_path.glob("sweep_L*_bbq.json"):
        try:
            d = json.loads(f.read_text())
            l = d["layer_idx"]
            scores[l] = d["metrics"]["ambig"]["bias_score"]
        except Exception:
            pass
    return scores


def pearson_skip_none(xs, ys):
    """Compute Pearson r, skipping None values."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    a, b = zip(*pairs)
    r, p = pearsonr(a, b)
    return float(r), float(p)


def spearman_skip_none(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    a, b = zip(*pairs)
    r, p = spearmanr(a, b)
    return float(r), float(p)


# Analysis 1 & 2: per-model L2 norm + cos_sim vs BBQ sweep

def analyze_per_model(alias: str, cfg: dict) -> dict:
    n = cfg["n_layers"]
    print(f"\n[{alias}] loading diff...", end=" ", flush=True)
    diff = torch.load(cfg["diff"], map_location="cpu", weights_only=False)

    vecs = layer_vectors(diff, n)          # list of (D_l,) tensors
    norms = [v.norm().item() for v in vecs]

    # global mean vector (mean across layers, normalized to unit before averaging
    # to avoid large-layer dominance — direction-only mean)
    unit_vecs = [v / v.norm() if v.norm() > 0 else v for v in vecs]
    global_mean = torch.stack(unit_vecs).mean(dim=0)

    cos_to_mean = [cosine_sim(v, global_mean) for v in vecs]

    bbq_raw = load_bbq_sweep(alias, n)

    # baseline delta: delta(l) = sweep(l) - baseline
    baseline = load_baseline_bias(alias)
    if baseline is not None:
        bbq_delta = [s - baseline if s is not None else None for s in bbq_raw]
    else:
        bbq_delta = bbq_raw  # no baseline: use raw scores

    n_done = sum(1 for s in bbq_raw if s is not None)
    print(f"done. sweep {n_done}/{n} layers | baseline={baseline:.4f}" if baseline else
          f"done. sweep {n_done}/{n} layers | baseline=None")

    # correlation analysis (delta-based)
    pearson_norm_bbq,  pp_nb  = pearson_skip_none(norms,       bbq_delta)
    pearson_cos_bbq,   pp_cb  = pearson_skip_none(cos_to_mean, bbq_delta)
    spearman_norm_bbq, sp_nb  = spearman_skip_none(norms,       bbq_delta)
    spearman_cos_bbq,  sp_cb  = spearman_skip_none(cos_to_mean, bbq_delta)

    # peak suppression layer: minimum delta (strongest bias reduction)
    done_idx = [i for i, s in enumerate(bbq_delta) if s is not None]
    peak_suppress_layer = min(done_idx, key=lambda i: bbq_delta[i]) if done_idx else None
    peak_norm_layer     = int(np.argmax(norms))
    peak_coslow_layer   = int(np.argmin(cos_to_mean))

    result = dict(
        alias=alias, n_layers=n, family=cfg["family"],
        norms=norms,
        cos_to_mean=cos_to_mean,
        bbq_bias_score=bbq_raw,
        bbq_delta=bbq_delta,
        baseline_bias=baseline,
        layer_indices=list(range(n)),
        n_sweep_done=n_done,
        peak_suppress_layer=peak_suppress_layer,
        peak_norm_layer=peak_norm_layer,
        peak_coslow_layer=peak_coslow_layer,
        correlations=dict(
            pearson_norm_bbq=pearson_norm_bbq,   pearson_norm_bbq_p=pp_nb,
            pearson_cos_bbq=pearson_cos_bbq,     pearson_cos_bbq_p=pp_cb,
            spearman_norm_bbq=spearman_norm_bbq, spearman_norm_bbq_p=sp_nb,
            spearman_cos_bbq=spearman_cos_bbq,   spearman_cos_bbq_p=sp_cb,
        ),
    )

    print(f"  peak_suppress L{peak_suppress_layer} / {n}"
          if peak_suppress_layer is not None else "  peak: sweep incomplete")
    print(f"  peak_norm     L{peak_norm_layer} / {n}")
    print(f"  Pearson(norm, Δbias): r={pearson_norm_bbq:.3f}  p={pp_nb:.3f}")
    print(f"  Pearson(cos,  Δbias): r={pearson_cos_bbq:.3f}  p={pp_cb:.3f}")
    return result


# Analysis 3: cross-model thirds-based comparison (equal thirds, Rogers 2020)

def thirds_summary(per_model: dict) -> dict:
    """
    Cross-model comparison using equal thirds (Rogers 2020, Geva 2021/2022).

    Summarises each model as [first_third_mean, mid_third_mean, last_third_mean].
    Equal thirds follow Rogers et al. (TACL 2020) as the standard for
    cross-architecture comparison.

    Returns:
      per_model_thirds: alias → {first, mid, last} mean delta + dominant pattern
      cross_model:      pairwise thirds-pattern agreement between models
    """
    aliases = [a for a in per_model if per_model[a]["n_sweep_done"] >= 8]
    per_model_thirds = {}

    for alias in aliases:
        r   = per_model[alias]
        n   = r["n_layers"]
        delta = r["bbq_delta"]

        t1 = n // 3
        t2 = 2 * n // 3
        groups = {
            "first": delta[:t1],
            "mid":   delta[t1:t2],
            "last":  delta[t2:],
        }
        thirds_mean = {}
        for g, vals in groups.items():
            valid = [v for v in vals if v is not None]
            thirds_mean[g] = float(np.mean(valid)) if valid else float("nan")

        # dominant third: the one with the strongest bias suppression (minimum delta)
        dominant = min(thirds_mean, key=lambda k: thirds_mean[k])
        layer_ranges = {
            "first": (0, t1 - 1),
            "mid":   (t1, t2 - 1),
            "last":  (t2, n - 1),
        }

        per_model_thirds[alias] = dict(
            n_layers=n,
            family=MODELS[alias]["family"],
            thirds_mean=thirds_mean,
            dominant_third=dominant,
            layer_ranges=layer_ranges,
        )
        print(f"  {alias:<14} first={thirds_mean['first']:+.4f}  "
              f"mid={thirds_mean['mid']:+.4f}  last={thirds_mean['last']:+.4f}  "
              f"→ dominant={dominant}")

    # cross-model: Pearson r over the thirds vector [first,mid,last] (dof=1; interpret cautiously)
    cross_model = {}
    for a1, a2 in combinations(aliases, 2):
        t1v = [per_model_thirds[a1]["thirds_mean"][g] for g in ("first","mid","last")]
        t2v = [per_model_thirds[a2]["thirds_mean"][g] for g in ("first","mid","last")]
        r, p = pearson_skip_none(t1v, t2v)
        same_family = MODELS[a1]["family"] == MODELS[a2]["family"]

        # pattern match: do both models share the same dominant third?
        pattern_match = per_model_thirds[a1]["dominant_third"] == per_model_thirds[a2]["dominant_third"]

        key = f"{a1}_vs_{a2}"
        cross_model[key] = dict(
            model_a=a1, model_b=a2,
            same_family=same_family,
            thirds_a=t1v, thirds_b=t2v,
            pearson_r=r, pearson_p=p,
            pattern_match=pattern_match,
            dominant_a=per_model_thirds[a1]["dominant_third"],
            dominant_b=per_model_thirds[a2]["dominant_third"],
        )

    return {"per_model": per_model_thirds, "cross_model": cross_model}


# Plotting

def plot_per_model(per_model: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping plots")
        return

    # Figure 1: per-model dual-axis (norm + bias_score)
    n_models = len(per_model)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (alias, r) in zip(axes, per_model.items()):
        layers = r["layer_indices"]
        norms  = r["norms"]
        bbq    = r["bbq_bias_score"]
        n      = r["n_layers"]

        # normalize norms to [0,1] for overlay
        max_n = max(norms) if max(norms) > 0 else 1
        norm_scaled = [x / max_n for x in norms]

        ax2 = ax.twinx()
        ax.plot(layers, norm_scaled, color="steelblue", alpha=0.7, label="ΔW norm (scaled)")
        ax.plot(layers, r["cos_to_mean"], color="orange", alpha=0.7, linestyle="--", label="cos_sim to mean")

        done = [(layers[i], bbq[i]) for i in range(n) if bbq[i] is not None]
        if done:
            dx, dy = zip(*done)
            ax2.plot(dx, dy, color="crimson", linewidth=1.5, label="BBQ bias_score")

        ax.set_title(f"{alias} (n={n})", fontsize=9)
        ax.set_xlabel("layer index")
        ax.set_ylabel("norm / cos_sim", fontsize=7)
        ax2.set_ylabel("BBQ bias_score", fontsize=7, color="crimson")
        ax.set_ylim(-0.1, 1.3)

        # vertical lines for peaks
        if r["peak_suppress_layer"] is not None:
            ax.axvline(r["peak_suppress_layer"], color="crimson", alpha=0.3, linestyle=":")
        ax.axvline(r["peak_norm_layer"], color="steelblue", alpha=0.3, linestyle=":")

    # merge legends
    handles = [
        plt.Line2D([0], [0], color="steelblue", label="ΔW L2 norm (scaled)"),
        plt.Line2D([0], [0], color="orange", linestyle="--", label="cos_sim to mean ΔW"),
        plt.Line2D([0], [0], color="crimson", label="BBQ ambig bias_score"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Per-layer ΔW statistics vs BBQ bias_score", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cosine_per_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved cosine_per_model.png")

    # Figure 2: correlation summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    aliases = list(per_model.keys())
    r_norm  = [per_model[a]["correlations"]["pearson_norm_bbq"] for a in aliases]
    r_cos   = [per_model[a]["correlations"]["pearson_cos_bbq"]  for a in aliases]

    x = np.arange(len(aliases))
    w = 0.35
    axes[0].bar(x - w/2, r_norm, w, label="Pearson(L2 norm, bias)", color="steelblue")
    axes[0].bar(x + w/2, r_cos,  w, label="Pearson(cos_sim, bias)", color="orange")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(aliases, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title("Correlation: ΔW stats vs BBQ bias_score")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-1, 1)

    # peak layer comparison (layer index, not depth)
    bias_layers = [per_model[a]["peak_suppress_layer"] or 0 for a in aliases]
    norm_layers = [per_model[a]["peak_norm_layer"]       for a in aliases]
    axes[1].scatter(norm_layers, bias_layers, c="steelblue", s=80, zorder=3)
    for i, a in enumerate(aliases):
        axes[1].annotate(a, (norm_layers[i], bias_layers[i]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    max_l = max(max(norm_layers), max(bias_layers)) + 1
    axes[1].plot([0, max_l], [0, max_l], "k--", alpha=0.3, label="y=x")
    axes[1].set_xlabel("peak norm layer (index)")
    axes[1].set_ylabel("peak bias_score layer (index)")
    axes[1].set_title("Peak alignment: L2 norm vs BBQ sweep")
    axes[1].set_xlim(0, max_l); axes[1].set_ylim(0, max_l)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cosine_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved cosine_correlation.png")


def plot_cross_model(cross: dict, thirds_per_model: dict):
    """Cross-model comparison plot using equal thirds (Rogers 2020)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch


    except ImportError:
        return

    aliases = list(thirds_per_model.keys())
    DISPLAY = {
        "qwen_0.5b":"Qwen 0.5B","qwen_1.5b":"Qwen 1.5B","qwen_3b":"Qwen 3B",
        "qwen_14b":"Qwen 14B","gemma_2b":"Gemma 2B","gemma_9b":"Gemma 9B",
        "llama32_1b":"LLaMA3.2-1B","llama32_3b":"LLaMA3.2-3B","llama3_8b":"LLaMA3-8B",
    }

    # Figure 1: 9-model thirds mean delta heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    thirds_names = ["first", "mid", "last"]
    matrix = np.array([
        [thirds_per_model[a]["thirds_mean"][t] for t in thirds_names]
        for a in aliases
    ])
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(matrix.min()), abs(matrix.max())),
                   vmax= max(abs(matrix.min()), abs(matrix.max())))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["First third", "Mid third", "Last third"], fontsize=11)
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels([DISPLAY.get(a, a) for a in aliases], fontsize=10)
    for i, a in enumerate(aliases):
        for j, t in enumerate(thirds_names):
            v = thirds_per_model[a]["thirds_mean"][t]
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > matrix.std() else "black")
    plt.colorbar(im, ax=ax, label="Mean Δbias_score (sweep − baseline)")
    ax.set_title("Cross-model thirds pattern: mean Δbias_score per third\n"
                 "(blue=suppression↓, red=amplification↑)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cosine_cross_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] saved cosine_cross_model.png (thirds heatmap)")

    # Figure 2: dominant third agreement matrix
    n = len(aliases)
    match_matrix = np.zeros((n, n))
    for i, a1 in enumerate(aliases):
        for j, a2 in enumerate(aliases):
            if i == j:
                match_matrix[i, j] = 1.0
            else:
                key = f"{a1}_vs_{a2}" if f"{a1}_vs_{a2}" in cross else f"{a2}_vs_{a1}"
                if key in cross:
                    match_matrix[i, j] = 1.0 if cross[key]["pattern_match"] else 0.0

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.imshow(match_matrix, cmap="Greens", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([DISPLAY.get(a, a) for a in aliases], rotation=40, ha="right", fontsize=8)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([DISPLAY.get(a, a) for a in aliases], fontsize=8)
    ax2.set_title("Cross-model: dominant third pattern match\n(green=same dominant third)", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "cosine_cross_model_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("[plot] saved cosine_cross_model_bar.png (pattern match matrix)")


# Main

def main():
    print("=" * 60)
    print("Analysis 1 & 2: per-layer ΔW stats vs BBQ sweep")
    print("=" * 60)
    per_model = {}
    for alias, cfg in MODELS.items():
        per_model[alias] = analyze_per_model(alias, cfg)

    print("\n" + "=" * 60)
    print("Analysis 3: cross-model thirds-based comparison (Rogers 2020)")
    print("=" * 60)
    thirds_result = thirds_summary(per_model)
    cross = thirds_result["cross_model"]

    # summary statistics
    print("\n" + "=" * 60)
    print("CORRELATION SUMMARY")
    print("=" * 60)
    valid = [(a, r) for a, r in per_model.items() if r["n_sweep_done"] >= 8]
    if valid:
        mean_r_norm = np.mean([r["correlations"]["pearson_norm_bbq"] for _, r in valid])
        mean_r_cos  = np.mean([r["correlations"]["pearson_cos_bbq"]  for _, r in valid])
        print(f"Mean Pearson(L2 norm, BBQ bias): {mean_r_norm:.3f}  (n={len(valid)} models)")
        print(f"Mean Pearson(cos_sim, BBQ bias): {mean_r_cos:.3f}  (n={len(valid)} models)")

    # thirds pattern match rate
    same_match  = [v["pattern_match"] for v in cross.values() if v["same_family"]]
    cross_match = [v["pattern_match"] for v in cross.values() if not v["same_family"]]
    if same_match:
        print(f"Thirds pattern match — same family:  {sum(same_match)}/{len(same_match)}")
    if cross_match:
        print(f"Thirds pattern match — cross family: {sum(cross_match)}/{len(cross_match)}")

    # save
    pkl_path = OUT_DIR / "cosine_sim.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"per_model": per_model, "cross_model": cross}, f)
    print(f"\nSaved → {pkl_path}")

    # JSON summary (per_model only; cross_model is too large)
    summary = {}
    for alias, r in per_model.items():
        summary[alias] = {
            "n_layers": r["n_layers"],
            "family": r["family"],
            "n_sweep_done": r["n_sweep_done"],
            "peak_suppress_layer": r["peak_suppress_layer"],
            "peak_norm_layer":     r["peak_norm_layer"],
            "peak_coslow_layer":   r["peak_coslow_layer"],
            "correlations":        r["correlations"],
        }
    cross_summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("thirds_a", "thirds_b")}
                     for k, v in cross.items()}
    summary["_cross_model"] = cross_summary
    summary["_thirds_per_model"] = {
        a: thirds_result["per_model"][a] for a in thirds_result["per_model"]
    }

    json_path = OUT_DIR / "cosine_sim.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved → {json_path}")

    # plots
    print("\nGenerating plots...")
    plot_per_model(per_model)
    plot_cross_model(cross, thirds_result["per_model"])
    print("Done.")


if __name__ == "__main__":
    main()
