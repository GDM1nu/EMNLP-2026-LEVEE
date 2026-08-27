"""
Per-layer ΔW Frobenius norm profile analysis.

For each layer i:
  frob_norm  = sqrt(Σ w²)  over all 2D weight matrices in that layer
  frob_norm_rel = frob_norm / mean(frob_norm across all layers)  [relative, model-normalized]

Then:
  - Spearman(frob_norm, bias_curve)  — does larger ΔW → more bias transfer?
  - Per-third means of norm
  - Ratio: actual ΔW effect / norm (direction efficiency)

Complements analyze_dw_svd.py:
  SVD → structure/direction of ΔW
  Norm → magnitude of ΔW
  Together → §4.6 "direction vs magnitude" control experiment

CPU-only. No SVD. Very fast (~1 min/model).
Output: results/analysis/dw_norm.json + printed table
"""
import re, json, glob, sys, traceback
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent
ARCHIVE = ARCHIVE

MODELS = {
    "qwen_0.5b":  {"n_layers": 24, "diff": ARCHIVE/"scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt"},
    "qwen_1.5b":  {"n_layers": 28, "diff": ARCHIVE/"scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt"},
    "qwen_3b":    {"n_layers": 36, "diff": ARCHIVE/"scaling_experiment/qwen_3b/diff_vector/diff_vector.pt"},
    "qwen_14b":   {"n_layers": 48, "diff": ARCHIVE/"scaling_experiment/qwen_14b/diff_vector/diff_vector.pt"},
    "gemma_2b":   {"n_layers": 26, "diff": ARCHIVE/"scaling_experiment/gemma_2b/diff_vector/diff_vector.pt"},
    "gemma_9b":   {"n_layers": 42, "diff": ARCHIVE/"scaling_experiment/gemma_9b/diff_vector/diff_vector.pt"},
    "llama32_1b": {"n_layers": 16, "diff": ARCHIVE/"llama_experiment/models/llama32_1b_diff/llama32_1b_diff.pt"},
    "llama32_3b": {"n_layers": 28, "diff": ARCHIVE/"llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt"},
    "llama3_8b":  {"n_layers": 32, "diff": ARCHIVE/"llama_experiment/models/llama_diff_vector/llama_diff_vector.pt"},
}

LAYER_RE = re.compile(r"(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.")


def layer_norm_profile(diff: dict, n_layers: int):
    """
    Returns list of dicts, one per layer.
    frob_norm = sqrt(Σ_matrices Σ_elements w²)  [combined across all 2D matrices in layer]
    n_params  = total number of parameters (elements) in 2D matrices of that layer
    """
    buckets = defaultdict(list)
    for key, tensor in diff.items():
        m = LAYER_RE.search(key)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= n_layers:
            continue
        if tensor.ndim != 2:
            continue
        buckets[idx].append((key, tensor))

    results = []
    for layer_idx in range(n_layers):
        matrices = buckets.get(layer_idx, [])
        if not matrices:
            results.append({
                "layer_idx": layer_idx,
                "n_matrices": 0,
                "frob_norm": float("nan"),
                "frob_norm_sq": float("nan"),
                "n_params": 0,
            })
            continue

        frob_sq_total = 0.0
        n_params_total = 0
        n_matrices = 0
        for key, tensor in matrices:
            w = tensor.float()
            frob_sq_total += float(w.pow(2).sum())
            n_params_total += w.numel()
            n_matrices += 1

        results.append({
            "layer_idx": layer_idx,
            "n_matrices": n_matrices,
            "frob_norm": float(np.sqrt(frob_sq_total)),
            "frob_norm_sq": frob_sq_total,
            "n_params": n_params_total,
        })
    return results


def load_bbq_curve(model_key):
    scores = []
    summary = PROJ / f"results/sweep_thirds/{model_key}/sweep_summary.json"
    if summary.exists():
        with open(summary) as f:
            d = json.load(f)
        for l in d.get("layers", []):
            try:
                bs = l["benchmarks"]["bbq"]["ambig"]["bias_score"]
                scores.append((l["layer_idx"], bs))
            except (KeyError, TypeError):
                pass
    if not scores:
        for fp in sorted(glob.glob(str(PROJ / f"results/sweep_thirds/{model_key}/sweep_L*_bbq.json"))):
            with open(fp) as f:
                ld = json.load(f)
            try:
                bs = ld["metrics"]["ambig"]["bias_score"]
                scores.append((ld["layer_idx"], bs))
            except (KeyError, TypeError):
                pass
    if not scores:
        return None
    scores.sort()
    return [s[1] for s in scores]


def layer_to_third(idx, n):
    if idx < n // 3: return "first"
    if idx < 2 * n // 3: return "mid"
    return "last"


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def analyze_model(model_key):
    cfg = MODELS[model_key]
    diff_path = cfg["diff"]
    n_layers = cfg["n_layers"]

    if not diff_path.exists():
        print(f"  [{model_key}] diff not found: {diff_path}")
        return None

    print(f"  [{model_key}] loading {diff_path.stat().st_size/1e9:.1f}GB ...", end=" ", flush=True)
    diff = torch.load(diff_path, map_location="cpu", weights_only=True)
    print(f"norm ({n_layers} layers) ...", end=" ", flush=True)

    layer_metrics = layer_norm_profile(diff, n_layers)
    del diff

    bias_curve = load_bbq_curve(model_key)

    norms     = [l["frob_norm"] for l in layer_metrics]
    norms_arr = np.array(norms)
    mean_norm = float(np.nanmean(norms_arr))

    # Relative norm (normalized within model so we can compare across models)
    norms_rel = [float(v / mean_norm) if not np.isnan(v) else float("nan") for v in norms]

    # Spearman vs bias curve
    def spear(a, b):
        a_arr = np.array(a)
        if b is not None and len(b) == n_layers and not np.any(np.isnan(a_arr)):
            return spearmanr(a_arr, b)
        return float("nan"), float("nan")

    r_norm, p_norm = spear(norms, bias_curve)
    r_rel,  p_rel  = spear(norms_rel, bias_curve)

    # Per-third means (absolute norm)
    thirds = {"first": [], "mid": [], "last": []}
    thirds_rel = {"first": [], "mid": [], "last": []}
    for l, nr in zip(layer_metrics, norms_rel):
        t = layer_to_third(l["layer_idx"], n_layers)
        if not np.isnan(l["frob_norm"]):
            thirds[t].append(l["frob_norm"])
        if not np.isnan(nr):
            thirds_rel[t].append(nr)

    third_means     = {t: float(np.mean(v)) if v else float("nan") for t, v in thirds.items()}
    third_means_rel = {t: float(np.mean(v)) if v else float("nan") for t, v in thirds_rel.items()}
    dominant        = max(third_means, key=third_means.get)   # highest norm third
    dominant_rel    = max(third_means_rel, key=third_means_rel.get)

    # Direction efficiency: bias_effect / norm
    # For each layer: |bias_curve[i] - mean(bias_curve)| / frob_norm
    # High = small norm, large bias effect → direction-driven
    eff_scores = []
    if bias_curve and len(bias_curve) == n_layers:
        bias_arr   = np.array(bias_curve)
        bias_mean  = float(np.nanmean(bias_arr))
        bias_dev   = np.abs(bias_arr - bias_mean)   # deviation from mean bias
        for i, l in enumerate(layer_metrics):
            fn = l["frob_norm"]
            if not np.isnan(fn) and fn > 0:
                eff_scores.append(float(bias_dev[i] / fn))
            else:
                eff_scores.append(float("nan"))

    print(f"ρ(norm,bias)={r_norm:+.3f}p={p_norm:.3f}  "
          f"dom_norm={dominant}  mean_norm={mean_norm:.3f}", flush=True)

    return {
        "model": model_key,
        "n_layers": n_layers,
        "mean_frob_norm": mean_norm,
        "layer_metrics": [
            {**lm, "frob_norm_rel": nr, "direction_efficiency": es}
            for lm, nr, es in zip(
                layer_metrics, norms_rel,
                eff_scores if eff_scores else [float("nan")] * n_layers
            )
        ],
        "spearman_norm_bias":    round(r_norm, 4),
        "spearman_norm_bias_p":  round(p_norm, 4),
        "spearman_norm_rel_bias":   round(r_rel, 4),
        "spearman_norm_rel_bias_p": round(p_rel, 4),
        "third_means_norm":     third_means,
        "third_means_norm_rel": third_means_rel,
        "dominant_third_high_norm": dominant,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS.keys())
    print(f"Analyzing {len(models)} models — Frobenius norm profile (CPU, no SVD)\n")

    out_dir = PROJ / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dw_norm.json"

    all_results = []
    for m in models:
        try:
            r = analyze_model(m)
            if r:
                all_results.append(r)
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                sys.stdout.flush()
        except Exception as e:
            print(f"\n  [{m}] FAILED: {e}", flush=True)
            traceback.print_exc()

    if not all_results:
        print("No results.")
        return


    # Table 1: Per-third Frobenius norm (relative) + Spearman

    print("\n=== [1] Frobenius norm (relative, per-third) + Spearman vs BBQ bias ===")
    print(f"{'Model':<14} {'first':>7} {'mid':>7} {'last':>7}  {'dom':>5}  "
          f"{'ρ(norm,bias)':>13} {'sig':>4}  {'mean‖ΔW‖':>10}")
    print("-" * 80)
    for r in all_results:
        tm = r["third_means_norm_rel"]
        print(f"{r['model']:<14} {tm['first']:>7.4f} {tm['mid']:>7.4f} {tm['last']:>7.4f}  "
              f"{r['dominant_third_high_norm']:>5}  "
              f"{r['spearman_norm_bias']:>+13.3f} {sig_stars(r['spearman_norm_bias_p']):>4}  "
              f"{r['mean_frob_norm']:>10.4f}")


    # Table 2: Hypothesis — does high-norm = high bias effect?

    print("\n=== [2] Hypothesis: high-norm third = mid-third? ===")
    mid_high = sum(1 for r in all_results if r["dominant_third_high_norm"] == "mid")
    print(f"  mid-third has highest norm: {mid_high}/{len(all_results)} models")

    print("\n=== [3] Spearman sign summary ===")
    pos = sum(1 for r in all_results if r["spearman_norm_bias"] > 0)
    sig = sum(1 for r in all_results if r["spearman_norm_bias_p"] < 0.05)
    print(f"  ρ > 0 (larger norm → higher bias): {pos}/{len(all_results)} models")
    print(f"  significant (p<0.05): {sig}/{len(all_results)} models")
    print(f"  interpretation: {'norm predicts bias' if pos > len(all_results)//2 else 'direction matters more than magnitude'}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
