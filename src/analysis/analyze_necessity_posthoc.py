"""
Post-hoc necessity analysis from per-layer sweep results.

For each model, loads sweep_summary.json and computes per-layer bias delta
(baseline_bias - layer_bias), then aggregates by thirds to estimate group
contributions under a linearity assumption.

Outputs:
  results/necessity_posthoc/<alias>_necessity.json

Usage:
  python src/analysis/analyze_necessity_posthoc.py
  python src/analysis/analyze_necessity_posthoc.py --models qwen_0.5b qwen_3b
  python src/analysis/analyze_necessity_posthoc.py --alias gemma_2b --baseline 0.312
"""

import argparse
import json
import os
from pathlib import Path

ARCHIVE   = PROJ_ROOT.parent / "archive"  # path to model weight storage

SWEEP_DIR = Path("results/sweep_thirds")
OUT_DIR   = Path("results/necessity_posthoc")

BASELINE_PATHS = {
    "qwen_0.5b":  str(ARCHIVE / "scaling_experiment/qwen_0.5b/baselines/baseline_base_bbq.json"),
    "qwen_1.5b":  str(ARCHIVE / "scaling_experiment/qwen_1.5b/baselines/baseline_base_bbq.json"),
    "qwen_3b":    str(ARCHIVE / "scaling_experiment/qwen_3b/baselines/baseline_base_bbq.json"),
    "qwen_14b":   str(ARCHIVE / "scaling_experiment/qwen_14b/baselines/baseline_base_bbq.json"),
    "gemma_9b":   str(ARCHIVE / "scaling_experiment/gemma_9b/baselines/baseline_base_bbq.json"),
}


def load_baseline(alias: str, override: float | None) -> float | None:
    if override is not None:
        return override
    path = BASELINE_PATHS.get(alias)
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["metrics"]["ambig"]["bias_score"]


def compute_thirds(n_layers: int) -> dict:
    t1 = n_layers // 3
    t2 = 2 * n_layers // 3
    return {
        "first": list(range(0, t1)),
        "mid":   list(range(t1, t2)),
        "last":  list(range(t2, n_layers)),
    }


def analyze_model(alias: str, baseline_override: float | None = None) -> dict | None:
    summary_path = SWEEP_DIR / alias / "sweep_summary.json"
    if not summary_path.exists():
        print(f"[skip] {alias}: sweep_summary.json not found")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    n_layers = summary["n_layers"]
    baseline = load_baseline(alias, baseline_override)
    if baseline is None:
        print(f"[skip] {alias}: no baseline — provide --baseline <value>")
        return None

    thirds = compute_thirds(n_layers)

    # per-layer bias score and delta
    layers_data = []
    if summary["layers"]:
        for entry in summary["layers"]:
            l = entry["layer_idx"]
            bias = entry["benchmarks"]["bbq"]["ambig"]["bias_score"]
            delta = baseline - bias
            layers_data.append({"layer_idx": l, "bias_score": bias, "delta": delta})
    else:
        # fallback: read individual layer files
        import glob


        layer_files = sorted(glob.glob(str(SWEEP_DIR / alias / "sweep_L*_bbq.json")))
        for f in layer_files:
            with open(f) as fp:
                d = json.load(fp)
            l = d["layer_idx"]
            bias = d["metrics"]["ambig"]["bias_score"]
            delta = baseline - bias
            layers_data.append({"layer_idx": l, "bias_score": bias, "delta": delta})

    # sort by layer_idx just in case
    layers_data.sort(key=lambda x: x["layer_idx"])

    # group contribution (sum of deltas, linearity approximation)
    group_results = {}
    for name, indices in thirds.items():
        layer_deltas = [layers_data[i]["delta"] for i in indices if i < len(layers_data)]
        layer_biases = [layers_data[i]["bias_score"] for i in indices if i < len(layers_data)]
        best_layer = indices[int(__import__("numpy").argmin(layer_biases))]
        group_results[name] = {
            "layers": indices,
            "sum_delta": round(sum(layer_deltas), 6),
            "mean_delta": round(sum(layer_deltas) / len(layer_deltas), 6),
            "best_layer": best_layer,
            "best_layer_bias": round(min(layer_biases), 6),
        }

    total_sum = sum(v["sum_delta"] for v in group_results.values())

    # relative contribution (% of total)
    for name in group_results:
        s = group_results[name]["sum_delta"]
        group_results[name]["relative_contribution"] = round(s / total_sum, 4) if total_sum != 0 else 0.0

    result = {
        "alias": alias,
        "n_layers": n_layers,
        "baseline_ambig_bias": baseline,
        "thirds": {
            "first": thirds["first"],
            "mid":   thirds["mid"],
            "last":  thirds["last"],
        },
        "group_contributions": group_results,
        "layers": layers_data,
        "note": "Linearity approximation: sum(delta_l for l in group). True necessity requires held-out injection runs.",
    }

    # print summary
    print(f"\n=== {alias} (n_layers={n_layers}, baseline={baseline:.4f}) ===")
    print(f"  {'third':<8} {'layers':<14} {'sum_delta':>10} {'relative':>10} {'best_layer':>12} {'best_bias':>10}")
    for name in ["first", "mid", "last"]:
        g = group_results[name]
        lr = f"{g['layers'][0]}-{g['layers'][-1]}"
        print(f"  {name:<8} {lr:<14} {g['sum_delta']:>10.4f} {g['relative_contribution']:>10.2%} {g['best_layer']:>12} {g['best_layer_bias']:>10.4f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Aliases to process (default: all in sweep_thirds/)")
    parser.add_argument("--alias", help="Single alias (use with --baseline)")
    parser.add_argument("--baseline", type=float, help="Manual baseline ambig_bias_score override")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.alias:
        aliases = [args.alias]
    elif args.models:
        aliases = args.models
    else:
        aliases = sorted(p.name for p in SWEEP_DIR.iterdir() if p.is_dir())

    for alias in aliases:
        override = args.baseline if args.alias == alias or (args.baseline and len(aliases) == 1) else None
        result = analyze_model(alias, override)
        if result is None:
            continue
        out_path = OUT_DIR / f"{alias}_necessity.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → saved: {out_path}")


if __name__ == "__main__":
    main()
