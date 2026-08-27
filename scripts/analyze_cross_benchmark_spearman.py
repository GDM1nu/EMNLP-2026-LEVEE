"""
Cross-benchmark best-layer Spearman correlation.

For each of 11 models, find the best layer (minimum bias score) on:
  - BBQ      (sweep_thirds, ambig bias_score, lower = less biased)
  - CrowS    (sweep_multieval, overall bias_score, lower = less stereo-preferred)
  - StereoSet(sweep_multieval, overall ss_score,   lower = less stereotyped)

Then compute Spearman r across models for each benchmark pair.

Output: results/analysis/cross_benchmark_spearman.json + table
"""
import json, glob
from pathlib import Path
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent

MODELS_ORDER = [
    "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_14b",
    "gemma_2b", "gemma_9b", "gemma3_12b",
    "llama32_1b", "llama32_3b", "llama3_8b", "llama31_8b",
]

N_LAYERS = {
    "qwen_0.5b": 24, "qwen_1.5b": 28, "qwen_3b": 36, "qwen_14b": 48,
    "gemma_2b": 26, "gemma_9b": 42, "gemma3_12b": 42,
    "llama32_1b": 16, "llama32_3b": 28, "llama3_8b": 32, "llama31_8b": 32,
}


def _bbq_ambig(d):
    if "benchmarks" in d:
        return d["benchmarks"]["bbq"]["ambig"]["bias_score"]
    return d["metrics"]["ambig"]["bias_score"]


def get_bbq_best(model_key):
    scores = []
    summary = PROJ / f"results/sweep_thirds/{model_key}/sweep_summary.json"
    if summary.exists():
        with open(summary) as f:
            d = json.load(f)
        for l in d.get("layers", []):
            try:
                scores.append((l["layer_idx"], _bbq_ambig(l)))
            except (KeyError, TypeError):
                pass
    if not scores:
        import glob as _g
        for fp in sorted(_g.glob(str(PROJ / f"results/sweep_thirds/{model_key}/sweep_L*_bbq.json"))):
            with open(fp) as f:
                ld = json.load(f)
            try:
                scores.append((ld["layer_idx"], _bbq_ambig(ld)))
            except (KeyError, TypeError):
                pass
    if not scores:
        return None, None
    return min(scores, key=lambda x: x[1])


def get_crows_best(model_key):
    files = sorted(glob.glob(str(PROJ / f"results/sweep_multieval/{model_key}/sweep_L*_crows.json")))
    if not files:
        return None, None
    scores = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        layer_idx = d["layer_idx"]
        bias = d["metrics"]["overall"]["bias_score"]
        scores.append((layer_idx, bias))
    best = min(scores, key=lambda x: x[1])
    return best


def get_ss_best(model_key):
    files = sorted(glob.glob(str(PROJ / f"results/sweep_multieval/{model_key}/sweep_L*_stereoset.json")))
    if not files:
        return None, None
    scores = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        layer_idx = d["layer_idx"]
        ss = d["metrics"]["overall"]["ss_score"]
        scores.append((layer_idx, ss))
    best = min(scores, key=lambda x: x[1])
    return best


def main():
    records = []

    for model_key in MODELS_ORDER:
        n = N_LAYERS.get(model_key, None)
        bbq_l, bbq_s = get_bbq_best(model_key)
        crows_l, crows_s = get_crows_best(model_key)
        ss_l, ss_s = get_ss_best(model_key)

        records.append({
            "model": model_key,
            "n_layers": n,
            "bbq_best_layer": bbq_l,   "bbq_best_score": round(bbq_s, 4) if bbq_s is not None else None,
            "crows_best_layer": crows_l, "crows_best_score": round(crows_s, 2) if crows_s is not None else None,
            "ss_best_layer": ss_l,     "ss_best_score": round(ss_s, 2) if ss_s is not None else None,
        })

    # Print table
    print(f"{'Model':<14} {'N':>3}  {'BBQ L':>6} {'BBQ s':>7}  {'CrowS L':>7} {'CrowS s':>8}  {'SS L':>5} {'SS s':>7}")
    print("-" * 70)
    for r in records:
        bl = f"L{r['bbq_best_layer']:02d}" if r['bbq_best_layer'] is not None else "N/A"
        cl = f"L{r['crows_best_layer']:02d}" if r['crows_best_layer'] is not None else "N/A"
        sl = f"L{r['ss_best_layer']:02d}" if r['ss_best_layer'] is not None else "N/A"
        bs = f"{r['bbq_best_score']:.4f}" if r['bbq_best_score'] is not None else "N/A"
        cs = f"{r['crows_best_score']:.2f}" if r['crows_best_score'] is not None else "N/A"
        ss = f"{r['ss_best_score']:.2f}" if r['ss_best_score'] is not None else "N/A"
        print(f"{r['model']:<14} {r['n_layers']:>3}  {bl:>6} {bs:>7}  {cl:>7} {cs:>8}  {sl:>5} {ss:>7}")

    # Spearman correlations (models with all three benchmarks)
    complete = [r for r in records if all(r[k] is not None for k in ["bbq_best_layer","crows_best_layer","ss_best_layer"])]
    print(f"\nComplete models (all 3 benchmarks): {len(complete)}")

    if len(complete) >= 3:
        bbq_layers   = [r["bbq_best_layer"]   for r in complete]
        crows_layers = [r["crows_best_layer"] for r in complete]
        ss_layers    = [r["ss_best_layer"]    for r in complete]

        r_bc, p_bc = spearmanr(bbq_layers, crows_layers)
        r_bs, p_bs = spearmanr(bbq_layers, ss_layers)
        r_cs, p_cs = spearmanr(crows_layers, ss_layers)

        def stars(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return "ns"

        print("\nSpearman correlations (best layer index, cross-model):")
        print(f"  r(BBQ, CrowS) = {r_bc:+.3f}  p={p_bc:.3f} {stars(p_bc)}")
        print(f"  r(BBQ, SS)    = {r_bs:+.3f}  p={p_bs:.3f} {stars(p_bs)}")
        print(f"  r(CrowS, SS)  = {r_cs:+.3f}  p={p_cs:.3f} {stars(p_cs)}")

        spearman_results = {
            "n_models": len(complete),
            "models_included": [r["model"] for r in complete],
            "r_bbq_crows": round(r_bc, 4), "p_bbq_crows": round(p_bc, 4), "sig_bbq_crows": stars(p_bc),
            "r_bbq_ss":    round(r_bs, 4), "p_bbq_ss":    round(p_bs, 4), "sig_bbq_ss":    stars(p_bs),
            "r_crows_ss":  round(r_cs, 4), "p_crows_ss":  round(p_cs, 4), "sig_crows_ss":  stars(p_cs),
        }
    else:
        spearman_results = {}
        print("[WARN] Not enough complete models for Spearman")

    # Best layer as fraction of n_layers (depth)
    print("\nBest layer position (fraction of N):")
    print(f"{'Model':<14} {'BBQ':>6} {'CrowS':>7} {'SS':>6}")
    for r in complete:
        n = r["n_layers"]
        bb = r["bbq_best_layer"] / (n-1) if n > 1 else 0
        cb = r["crows_best_layer"] / (n-1) if n > 1 else 0
        sb = r["ss_best_layer"] / (n-1) if n > 1 else 0
        print(f"{r['model']:<14} {bb:>6.2f} {cb:>7.2f} {sb:>6.2f}")

    out_dir = PROJ / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_benchmark_spearman.json"
    with open(out_path, "w") as f:
        json.dump({"per_model": records, "spearman": spearman_results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
