#!/usr/bin/env python3
"""
Cumulative marginal gain vs sweep bias Spearman analysis.

For each model:
  - marginal_gain[k] = cumul_bias[k] - cumul_bias[k-1]  (k >= 1)
  - sweep_bias[k]    = single-layer sweep bias at layer k
  - Spearman(sweep_bias, marginal_gain)
  - third_means for marginal_gain

Output: results/analysis/cumulative_marginal.json
"""
from pathlib import Path
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr


PROJ_ROOT = Path(__file__).resolve().parents[2]

BASE = str(PROJ_ROOT)
OUT = os.path.join(BASE, "results/analysis/cumulative_marginal.json")

MODELS = [
    ("qwen_0.5b",  24),
    ("qwen_1.5b",  28),
    ("qwen_3b",    36),
    ("gemma_2b",   26),
    ("llama32_3b", 28),
    ("llama3_8b",  32),
    ("gemma_9b",   42),
    ("qwen_14b",   48),
]


def load_cumulative(alias):
    """Load cumulative bias scores per layer k. Returns list of (k, bias_score)."""
    d = os.path.join(BASE, "results/cumulative", alias)
    if not os.path.isdir(d):
        return None
    pairs = []
    for fn in sorted(os.listdir(d)):
        m = re.match(r"cumul_L(\d+)_bbq\.json", fn)
        if not m:
            continue
        k = int(m.group(1))
        with open(os.path.join(d, fn)) as f:
            data = json.load(f)
        bias = data["metrics"]["ambig"]["bias_score"]
        pairs.append((k, bias))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_sweep(alias):
    """Load single-layer sweep bias per layer. Returns dict {layer_idx: bias_score}."""
    d = os.path.join(BASE, "results/sweep_thirds", alias)
    if not os.path.isdir(d):
        return None
    result = {}
    for fn in os.listdir(d):
        m = re.match(r"sweep_L(\d+)_bbq\.json", fn)
        if not m:
            continue
        layer = int(m.group(1))
        with open(os.path.join(d, fn)) as f:
            data = json.load(f)
        result[layer] = data["metrics"]["ambig"]["bias_score"]
    return result


def dominant_third(values_by_layer, n_layers):
    """Return which third (first/mid/last) has the highest mean value."""
    n3 = n_layers // 3
    groups = {
        "first": [v for l, v in values_by_layer if l < n3],
        "mid":   [v for l, v in values_by_layer if n3 <= l < 2 * n3],
        "last":  [v for l, v in values_by_layer if l >= 2 * n3],
    }
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in groups.items()}
    return max(means, key=means.get), {k: round(v, 6) for k, v in means.items()}


def analyze(alias, n_layers_expected):
    cumul = load_cumulative(alias)
    sweep = load_sweep(alias)
    if not cumul or not sweep:
        print(f"  [{alias}] missing data — skip")
        return None
    if len(cumul) < 3:
        print(f"  [{alias}] too few cumulative layers ({len(cumul)}) — skip")
        return None

    # marginal gain: bias reduction when adding layer k (positive = good)
    layer_pairs = []
    for i in range(1, len(cumul)):
        k = cumul[i][0]
        mg = cumul[i - 1][1] - cumul[i][1]
        sw = sweep.get(k)
        if sw is None:
            continue
        layer_pairs.append({"layer_idx": k, "sweep_bias": round(sw, 6), "marginal_gain": round(mg, 6)})

    if len(layer_pairs) < 3:
        print(f"  [{alias}] too few matched pairs ({len(layer_pairs)}) — skip")
        return None

    sw_vals = [p["sweep_bias"] for p in layer_pairs]
    mg_vals = [p["marginal_gain"] for p in layer_pairs]
    rho, pval = spearmanr(sw_vals, mg_vals)

    n_layers = cumul[-1][0] + 1  # infer from max layer index
    dom, third_means = dominant_third(
        [(p["layer_idx"], p["marginal_gain"]) for p in layer_pairs], n_layers
    )

    sig = ""
    if pval < 0.001: sig = "***"
    elif pval < 0.01: sig = "**"
    elif pval < 0.05: sig = "*"
    elif pval < 0.1: sig = "†"

    print(f"  [{alias}] n={len(layer_pairs)}, ρ={rho:.3f}{sig}, p={pval:.3f}, "
          f"marginal_dom={dom}, n_layers={n_layers}")

    return {
        "model": alias,
        "n_layers": n_layers,
        "n_pairs": len(layer_pairs),
        "spearman_sweep_marginal": round(rho, 4),
        "spearman_sweep_marginal_p": round(pval, 4),
        "third_means_marginal": third_means,
        "dominant_third_marginal": dom,
        "layer_pairs": layer_pairs,
    }


def main():
    print("=== compute_cumulative_marginal ===")
    results = []
    for alias, n_layers in MODELS:
        r = analyze(alias, n_layers)
        if r:
            results.append(r)

    out = {"n_models": len(results), "results": results}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT}  ({len(results)} models)")

    # summary table
    print("\n--- Summary ---")
    print(f"{'model':<14} {'N':>4} {'rho':>7} {'p':>7} {'sig':>4} {'mg_dom':<8}")
    for r in results:
        sig = ""
        p = r["spearman_sweep_marginal_p"]
        if p < 0.001: sig = "***"
        elif p < 0.01: sig = "**"
        elif p < 0.05: sig = "*"
        elif p < 0.1: sig = "†"
        print(f"{r['model']:<14} {r['n_pairs']:>4} {r['spearman_sweep_marginal']:>7.3f} "
              f"{p:>7.3f} {sig:>4} {r['dominant_third_marginal']:<8}")


if __name__ == "__main__":
    main()
