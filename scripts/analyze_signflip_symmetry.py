"""
Sign-flip symmetry analysis.

For each model with signflip data, compute:
  Δ+ = baseline - sweep_best  (suppression by +ΔW at best layer)
  Δ− = signflip_at_best_layer - baseline  (amplification by −ΔW at best layer)
  symmetry_score = |Δ+ − Δ−| / max(|Δ+|, |Δ−|)   → 0=symmetric, 1=asymmetric

Output: results/analysis/signflip_symmetry.json + printed table
"""
import json, glob, statistics
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

# Alignment method per model (from published technical reports)
ALIGNMENT = {
    "qwen_3b":    "DPO/GRPO",
    "qwen_14b":   "DPO/GRPO",
    "gemma_2b":   "RLHF(SFT+PPO)",
    "gemma_9b":   "RLHF(SFT+PPO)",
    "llama32_1b": "SFT+PPO+DPO",
    "llama32_3b": "SFT+PPO+DPO",
    "llama3_8b":  "SFT+PPO+DPO",
}

# Hardcoded baselines from control_sweep zero or archive base scores
BASELINES = {
    "qwen_3b":    None,   # compute from zero control
    "qwen_14b":   0.0130,
    "gemma_2b":   None,
    "gemma_9b":   0.1163,
    "llama32_1b": None,
    "llama32_3b": None,
    "llama3_8b":  0.1274,
}


def get_baseline_from_zero(model_key):
    pattern = str(PROJ / f"results/control_sweep/{model_key}/ctrl_zero_L*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    scores = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        # handle two possible structures
        metrics = d.get("metrics", d)
        ambig = metrics.get("ambig", {})
        if "bias_score" in ambig:
            scores.append(ambig["bias_score"])
    return statistics.mean(scores) if scores else None


def _bbq_ambig(d):
    """Extract ambig bias_score from either 'benchmarks' or 'metrics' structure."""
    if "benchmarks" in d:
        return d["benchmarks"]["bbq"]["ambig"]["bias_score"]
    return d["metrics"]["ambig"]["bias_score"]


def get_sweep_best(model_key):
    summary_path = PROJ / f"results/sweep_thirds/{model_key}/sweep_summary.json"
    scores = []
    if summary_path.exists():
        with open(summary_path) as f:
            d = json.load(f)
        layers = d.get("layers", [])
        for l in layers:
            try:
                scores.append((l["layer_idx"], _bbq_ambig(l)))
            except (KeyError, TypeError):
                pass
    # fallback: individual layer files
    if not scores:
        import glob as _glob
        for fp in sorted(_glob.glob(str(PROJ / f"results/sweep_thirds/{model_key}/sweep_L*_bbq.json"))):
            with open(fp) as f:
                ld = json.load(f)
            try:
                scores.append((ld["layer_idx"], _bbq_ambig(ld)))
            except (KeyError, TypeError):
                pass
    if not scores:
        return None, None
    return min(scores, key=lambda x: x[1])


def get_signflip_at_layer(model_key, layer_idx):
    # try summary first
    summary_path = PROJ / f"results/signflip/{model_key}/sweep_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            d = json.load(f)
        for l in d.get("layers", []):
            if l["layer_idx"] == layer_idx:
                try:
                    return _bbq_ambig(l)
                except (KeyError, TypeError):
                    pass
    # fallback: individual file
    ind_path = PROJ / f"results/signflip/{model_key}/sweep_L{layer_idx:02d}_bbq.json"
    if ind_path.exists():
        with open(ind_path) as f:
            d = json.load(f)
        try:
            return _bbq_ambig(d)
        except (KeyError, TypeError):
            pass
    return None


def main():
    results = []

    for model_key in sorted(ALIGNMENT.keys()):
        signflip_dir = PROJ / f"results/signflip/{model_key}"
        if not signflip_dir.exists():
            continue

        # baseline
        baseline = BASELINES[model_key]
        if baseline is None:
            baseline = get_baseline_from_zero(model_key)
        if baseline is None:
            print(f"[WARN] {model_key}: no baseline, skipping")
            continue

        # sweep best
        best_layer, sweep_best = get_sweep_best(model_key)
        if best_layer is None:
            print(f"[WARN] {model_key}: no sweep_thirds data")
            continue

        # signflip at best layer
        sf_score = get_signflip_at_layer(model_key, best_layer)
        if sf_score is None:
            print(f"[WARN] {model_key}: signflip L{best_layer:02d} not found")
            continue

        delta_plus  = baseline - sweep_best      # suppression (+ = suppressed)
        delta_minus = sf_score - baseline         # amplification (+ = amplified)
        denom = max(abs(delta_plus), abs(delta_minus))
        symmetry = abs(delta_plus - delta_minus) / denom if denom > 0 else float("nan")

        if delta_plus > 0 and delta_minus > 0:
            verdict = "symmetric" if symmetry < 0.3 else ("asymmetric" if symmetry < 0.7 else "anomaly")
        elif delta_plus > 0 and delta_minus <= 0:
            verdict = "anomaly(-ΔW also suppresses)"
        else:
            verdict = "anomaly(+ΔW amplifies)"

        results.append({
            "model": model_key,
            "alignment": ALIGNMENT[model_key],
            "baseline": round(baseline, 4),
            "best_layer": best_layer,
            "sweep_best": round(sweep_best, 4),
            "signflip_at_best": round(sf_score, 4),
            "delta_plus": round(delta_plus, 4),
            "delta_minus": round(delta_minus, 4),
            "symmetry_score": round(symmetry, 4),
            "verdict": verdict,
        })

    # print table
    header = f"{'Model':<14} {'Align':<16} {'Base':>6} {'BestL':>6} {'+ΔW best':>9} {'-ΔW@best':>9} {'Δ+':>7} {'Δ-':>7} {'symm':>6} {'Verdict'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<14} {r['alignment']:<16} {r['baseline']:>6.4f} "
              f"L{r['best_layer']:02d}   {r['sweep_best']:>9.4f} {r['signflip_at_best']:>9.4f} "
              f"{r['delta_plus']:>7.4f} {r['delta_minus']:>7.4f} {r['symmetry_score']:>6.3f}  {r['verdict']}")

    out_dir = PROJ / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signflip_symmetry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
