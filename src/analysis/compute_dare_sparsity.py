#!/usr/bin/env python3
"""
DARE Sparsity Analysis — ΔW magnitude distribution per layer.

Based on: Yu et al. (2024) "Language Models are Super Mario: Absorbing Abilities
from pathlib import Path
from Homologous Models as a Free Lunch" (ICML 2024).

DARE showed that most ΔW parameters are near-zero and can be dropped without
harming performance. We use this framework to characterize ΔW sparsity by layer:
- Does the mid-third have larger-magnitude (less sparse) parameters?
- Does sparsity profile correlate with BBQ bias transfer effect?

Metrics per layer:
  - sparsity_p10: fraction of |ΔW| < 10th percentile of the full ΔW
  - density_top1pct: fraction of params contributing to top 1% cumulative magnitude
  - mean_abs, p50, p90, p99 of |ΔW|
  - thirds_mean: average mean_abs by third (first/mid/last)

Output: results/analysis/dare_sparsity.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

BASE = str(PROJ_ROOT)
OUT  = f"{BASE}/results/analysis/dare_sparsity.json"

ARCHIVE = str(ARCHIVE)
sys.path.insert(0, f"{ARCHIVE}/scaling_experiment/scripts")
from model_utils import get_num_layers, resolve_model_path


PROJ_ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("qwen_0.5b",  "Qwen/Qwen2.5-0.5B",          f"{ARCHIVE}/scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt"),
    ("qwen_1.5b",  "Qwen/Qwen2.5-1.5B",           f"{ARCHIVE}/scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt"),
    ("qwen_3b",    "Qwen/Qwen2.5-3B",             f"{ARCHIVE}/scaling_experiment/qwen_3b/diff_vector/diff_vector.pt"),
    ("gemma_2b",   "google/gemma-2-2b",            f"{ARCHIVE}/scaling_experiment/gemma_2b/diff_vector/diff_vector.pt"),
    ("gemma_9b",   "google/gemma-2-9b",            f"{ARCHIVE}/scaling_experiment/gemma_9b/diff_vector/diff_vector.pt"),
    ("llama32_3b", "meta-llama/Llama-3.2-3B",      f"{ARCHIVE}/llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt"),
    ("llama3_8b",  "meta-llama/Meta-Llama-3-8B",   f"{ARCHIVE}/llama_experiment/models/llama_diff_vector/llama_diff_vector.pt"),
]

SWEEP_DIR = f"{BASE}/results/sweep_thirds"


def layer_bias_scores(alias: str) -> dict[int, float]:
    files = sorted(
        __import__("glob").glob(f"{SWEEP_DIR}/{alias}/sweep_L*_bbq.json")
    )
    out = {}
    for f in files:
        d = json.load(open(f))
        l = d.get("layer_idx")
        s = d.get("metrics", {}).get("ambig", {}).get("bias_score")
        if l is not None and s is not None:
            out[l] = s
    return out


def thirds(n: int) -> dict[str, set]:
    return {
        "first": set(range(0, n // 3)),
        "mid":   set(range(n // 3, 2 * n // 3)),
        "last":  set(range(2 * n // 3, n)),
    }


def analyze_model(alias, model_name, diff_path) -> dict:
    print(f"\n[{alias}] loading diff ({os.path.getsize(diff_path)/1e9:.1f}GB)...", flush=True)
    diff = torch.load(diff_path, map_location="cpu", weights_only=False)

    n_layers = get_num_layers(resolve_model_path(model_name) or model_name)
    t = thirds(n_layers)
    bias_scores = layer_bias_scores(alias)

    # group params by layer index
    layer_params: dict[int, list] = {i: [] for i in range(n_layers)}
    for key, tensor in diff.items():
        # extract layer index from key (e.g. "model.layers.12.mlp.gate_proj.weight")
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    li = int(parts[i + 1])
                    if 0 <= li < n_layers:
                        layer_params[li].append(tensor.float().abs().numpy().ravel())
                except ValueError:
                    pass
                break

    layer_metrics = []
    all_abs = []  # global ΔW magnitude pool for global threshold

    # first pass: collect all values for global percentile threshold
    print(f"  computing global magnitude distribution...", flush=True)
    sample_vals = []
    for li in range(n_layers):
        if not layer_params[li]:
            continue
        v = np.concatenate(layer_params[li])
        # sample 10k per layer for global stats (memory efficient)
        if len(v) > 10000:
            idx = np.random.choice(len(v), 10000, replace=False)
            sample_vals.append(v[idx])
        else:
            sample_vals.append(v)
    global_abs = np.concatenate(sample_vals)
    global_p10  = float(np.percentile(global_abs, 10))
    global_p50  = float(np.percentile(global_abs, 50))
    global_p90  = float(np.percentile(global_abs, 90))

    # second pass: per-layer metrics
    print(f"  computing per-layer sparsity...", flush=True)
    for li in range(n_layers):
        if not layer_params[li]:
            continue
        v = np.concatenate(layer_params[li])
        n_params = len(v)

        mean_abs = float(v.mean())
        p50_abs  = float(np.percentile(v, 50))
        p90_abs  = float(np.percentile(v, 90))
        p99_abs  = float(np.percentile(v, 99))

        # DARE sparsity: fraction of params below global p10 threshold
        sparsity_global_p10 = float((v < global_p10).mean())

        # top-1% density: what fraction of params account for top 1% magnitude
        # (params sorted descending; cumsum until 1% of total sum)
        total_mass = v.sum()
        if total_mass > 0:
            sorted_v = np.sort(v)[::-1]
            cumsum = np.cumsum(sorted_v)
            cutoff = 0.01 * total_mass
            top1pct_count = int(np.searchsorted(cumsum, cutoff)) + 1
            density_top1pct = top1pct_count / n_params
        else:
            density_top1pct = 0.0

        layer_metrics.append(dict(
            layer_idx=li,
            n_params=n_params,
            mean_abs=round(mean_abs, 6),
            p50_abs=round(p50_abs, 6),
            p90_abs=round(p90_abs, 6),
            p99_abs=round(p99_abs, 6),
            sparsity_global_p10=round(sparsity_global_p10, 4),
            density_top1pct=round(density_top1pct, 6),
            bias_score=round(bias_scores.get(li, float("nan")), 6) if li in bias_scores else None,
        ))
        if li % 8 == 0:
            print(f"    L{li:02d}: mean_abs={mean_abs:.4f}  sparsity={sparsity_global_p10:.3f}  density_top1={density_top1pct:.4f}", flush=True)

    # thirds summary
    ts = thirds(n_layers)
    thirds_mean = {}
    for name, idxs in ts.items():
        vals = [m["mean_abs"] for m in layer_metrics if m["layer_idx"] in idxs]
        thirds_mean[name] = round(float(np.mean(vals)), 6) if vals else None

    dominant = max(thirds_mean, key=lambda k: thirds_mean[k] or 0)

    # correlation with bias score
    pairs = [(m["mean_abs"], m["bias_score"]) for m in layer_metrics if m["bias_score"] is not None]
    if len(pairs) >= 5:
        from scipy.stats import spearmanr
        xs, ys = zip(*pairs)
        sp_r, sp_p = spearmanr(xs, ys)
    else:
        sp_r, sp_p = float("nan"), float("nan")

    del diff
    import gc; gc.collect()



    return dict(
        model=alias,
        n_layers=n_layers,
        global_p10=round(global_p10, 6),
        global_p50=round(global_p50, 6),
        global_p90=round(global_p90, 6),
        thirds_mean_abs=thirds_mean,
        dominant_third_high_magnitude=dominant,
        spearman_meanabs_bias=round(float(sp_r), 4),
        spearman_meanabs_bias_p=round(float(sp_p), 4),
        layer_metrics=layer_metrics,
    )


def main():
    np.random.seed(0)
    results = []
    for alias, model_name, diff_path in MODELS:
        if not os.path.exists(diff_path):
            print(f"[skip] {alias}: diff not found at {diff_path}")
            continue
        r = analyze_model(alias, model_name, diff_path)
        results.append(r)
        print(f"  thirds mean_abs: {r['thirds_mean_abs']}  dominant={r['dominant_third_high_magnitude']}")
        print(f"  Spearman(mean_abs, bias): r={r['spearman_meanabs_bias']:.3f}  p={r['spearman_meanabs_bias_p']:.3f}")

    output = dict(
        n_models=len(results),
        reference="Yu et al. (ICML 2024) DARE: Language Models are Super Mario",
        metric_desc=dict(
            sparsity_global_p10="fraction of |ΔW| below global 10th-percentile threshold",
            density_top1pct="fraction of params needed to cover top 1% of total |ΔW| mass",
            mean_abs="mean |ΔW| per layer (proxy for magnitude concentration)",
        ),
        results=results,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
