"""
Mode-based layer analysis.

Instead of global minimum (best layer), find ALL local minima (modes/dips)
in each model's per-layer bias curve.

Key questions:
  1. Do local minima cluster in the same third(s) across benchmarks,
     even when global minima differ? (explains non-significant Spearman)
  2. How many modes does each model have per benchmark?
  3. Do models within the same family share mode positions?

Output: results/analysis/mode_based.json + tables
"""
import json, glob
from pathlib import Path
import numpy as np
from scipy.signal import argrelmin
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

FAMILY = {
    "qwen_0.5b": "qwen", "qwen_1.5b": "qwen", "qwen_3b": "qwen", "qwen_14b": "qwen",
    "gemma_2b": "gemma", "gemma_9b": "gemma", "gemma3_12b": "gemma",
    "llama32_1b": "llama", "llama32_3b": "llama", "llama3_8b": "llama", "llama31_8b": "llama",
}


# Data loaders


def _bbq_ambig(d):
    if "benchmarks" in d:
        return d["benchmarks"]["bbq"]["ambig"]["bias_score"]
    return d["metrics"]["ambig"]["bias_score"]


def load_bbq_curve(model_key):
    """Returns (layer_indices, scores) sorted by layer."""
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
        for fp in sorted(glob.glob(str(PROJ / f"results/sweep_thirds/{model_key}/sweep_L*_bbq.json"))):
            with open(fp) as f:
                ld = json.load(f)
            try:
                scores.append((ld["layer_idx"], _bbq_ambig(ld)))
            except (KeyError, TypeError):
                pass
    if not scores:
        return None, None
    scores.sort()
    return [s[0] for s in scores], [s[1] for s in scores]


def load_crows_curve(model_key):
    files = sorted(glob.glob(str(PROJ / f"results/sweep_multieval/{model_key}/sweep_L*_crows.json")))
    if not files:
        return None, None
    scores = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        scores.append((d["layer_idx"], d["metrics"]["overall"]["bias_score"]))
    scores.sort()
    return [s[0] for s in scores], [s[1] for s in scores]


def load_ss_curve(model_key):
    files = sorted(glob.glob(str(PROJ / f"results/sweep_multieval/{model_key}/sweep_L*_stereoset.json")))
    if not files:
        return None, None
    scores = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        scores.append((d["layer_idx"], d["metrics"]["overall"]["ss_score"]))
    scores.sort()
    return [s[0] for s in scores], [s[1] for s in scores]


# Mode finding


def find_local_minima(scores, order=1):
    """Find local minima indices using scipy argrelmin.
    order=1: each point compared to its immediate neighbors.
    Also include global minimum if not already in local minima.
    """
    arr = np.array(scores)
    local_min_idx = list(argrelmin(arr, order=order)[0])
    global_min_idx = int(np.argmin(arr))
    if global_min_idx not in local_min_idx:
        local_min_idx.append(global_min_idx)
    return sorted(local_min_idx)


def layer_to_third(layer_idx, n_layers):
    """Return 'first', 'mid', or 'last' given layer index and total layers."""
    boundary1 = n_layers // 3
    boundary2 = 2 * n_layers // 3
    if layer_idx < boundary1:
        return "first"
    elif layer_idx < boundary2:
        return "mid"
    else:
        return "last"


def thirds_coverage(mode_indices, n_layers):
    """Which thirds contain at least one mode?"""
    thirds = set(layer_to_third(i, n_layers) for i in mode_indices)
    return sorted(thirds)


def mode_overlap(modes_a, modes_b, window=2):
    """Count how many modes in A are within ±window of any mode in B."""
    count = 0
    for a in modes_a:
        if any(abs(a - b) <= window for b in modes_b):
            count += 1
    return count


# Main analysis


def analyze_model(model_key):
    n = N_LAYERS.get(model_key)
    if n is None:
        return None

    bbq_layers, bbq_scores = load_bbq_curve(model_key)
    crows_layers, crows_scores = load_crows_curve(model_key)
    ss_layers, ss_scores = load_ss_curve(model_key)

    result = {"model": model_key, "n_layers": n, "family": FAMILY.get(model_key)}

    for bench, layers, scores in [("bbq", bbq_layers, bbq_scores),
                                   ("crows", crows_layers, crows_scores),
                                   ("ss", ss_layers, ss_scores)]:
        if scores is None:
            result[bench] = None
            continue

        modes_idx = find_local_minima(scores, order=1)
        mode_layers = [layers[i] for i in modes_idx]
        global_min_layer = layers[int(np.argmin(scores))]
        global_min_score = min(scores)

        thirds = thirds_coverage(mode_layers, n)
        third_counts = {"first": 0, "mid": 0, "last": 0}
        for li in mode_layers:
            third_counts[layer_to_third(li, n)] += 1

        result[bench] = {
            "n_modes": len(mode_layers),
            "mode_layers": mode_layers,
            "mode_thirds": thirds,
            "third_counts": third_counts,
            "global_min_layer": global_min_layer,
            "global_min_score": round(global_min_score, 4),
        }

    # Cross-benchmark mode overlap (window=2 layers)
    if result.get("bbq") and result.get("crows"):
        result["overlap_bbq_crows"] = mode_overlap(
            result["bbq"]["mode_layers"], result["crows"]["mode_layers"])
    if result.get("bbq") and result.get("ss"):
        result["overlap_bbq_ss"] = mode_overlap(
            result["bbq"]["mode_layers"], result["ss"]["mode_layers"])
    if result.get("crows") and result.get("ss"):
        result["overlap_crows_ss"] = mode_overlap(
            result["crows"]["mode_layers"], result["ss"]["mode_layers"])

    return result


def dominant_third(third_counts):
    return max(third_counts, key=third_counts.get)


def main():
    all_results = []
    for m in MODELS_ORDER:
        r = analyze_model(m)
        if r:
            all_results.append(r)


    # Table 1: Mode count and dominant third per model/benchmark

    print("=== Table 1: Mode count and dominant third ===")
    print(f"{'Model':<14} {'N':>3}  "
          f"{'BBQ modes':>9} {'BBQ dom':>8}  "
          f"{'CrowS modes':>11} {'CrowS dom':>10}  "
          f"{'SS modes':>8} {'SS dom':>7}")
    print("-" * 80)
    for r in all_results:
        bbq_n  = r["bbq"]["n_modes"]  if r.get("bbq")   else "-"
        bbq_d  = dominant_third(r["bbq"]["third_counts"])  if r.get("bbq")   else "-"
        cr_n   = r["crows"]["n_modes"] if r.get("crows")  else "-"
        cr_d   = dominant_third(r["crows"]["third_counts"]) if r.get("crows") else "-"
        ss_n   = r["ss"]["n_modes"]    if r.get("ss")     else "-"
        ss_d   = dominant_third(r["ss"]["third_counts"])   if r.get("ss")     else "-"
        print(f"{r['model']:<14} {r['n_layers']:>3}  "
              f"{str(bbq_n):>9} {bbq_d:>8}  "
              f"{str(cr_n):>11} {cr_d:>10}  "
              f"{str(ss_n):>8} {ss_d:>7}")


    # Table 2: Does mid-third contain a mode? (cross-benchmark consistency)

    print("\n=== Table 2: Mid-third contains mode? ===")
    print(f"{'Model':<14} {'BBQ∋mid':>8} {'CrowS∋mid':>10} {'SS∋mid':>7}  {'all 3':>6}")
    print("-" * 50)
    for r in all_results:
        bbq_mid   = "✓" if r.get("bbq")   and "mid" in r["bbq"]["mode_thirds"]   else "✗"
        crows_mid = "✓" if r.get("crows") and "mid" in r["crows"]["mode_thirds"] else "✗"
        ss_mid    = "✓" if r.get("ss")    and "mid" in r["ss"]["mode_thirds"]    else "✗"
        all3 = "✓" if bbq_mid == crows_mid == ss_mid == "✓" else ""
        print(f"{r['model']:<14} {bbq_mid:>8} {crows_mid:>10} {ss_mid:>7}  {all3:>6}")


    # Table 3: Mode overlap between benchmarks (within ±2 layers)

    print("\n=== Table 3: Cross-benchmark mode overlap (±2 layers) ===")
    print(f"{'Model':<14} {'BBQ∩CrowS':>10} {'BBQ∩SS':>7} {'CrowS∩SS':>9}")
    print("-" * 45)
    for r in all_results:
        bc = r.get("overlap_bbq_crows", "-")
        bs = r.get("overlap_bbq_ss", "-")
        cs = r.get("overlap_crows_ss", "-")
        print(f"{r['model']:<14} {str(bc):>10} {str(bs):>7} {str(cs):>9}")


    # Summary: Spearman on dominant-third instead of exact layer

    print("\n=== Spearman: dominant third (0=first, 1=mid, 2=last) ===")
    third_to_int = {"first": 0, "mid": 1, "last": 2}
    bbq_thirds, crows_thirds, ss_thirds = [], [], []
    complete = [r for r in all_results
                if r.get("bbq") and r.get("crows") and r.get("ss")]
    for r in complete:
        bbq_thirds.append(third_to_int[dominant_third(r["bbq"]["third_counts"])])
        crows_thirds.append(third_to_int[dominant_third(r["crows"]["third_counts"])])
        ss_thirds.append(third_to_int[dominant_third(r["ss"]["third_counts"])])

    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    if len(complete) >= 3:
        r_bc, p_bc = spearmanr(bbq_thirds, crows_thirds)
        r_bs, p_bs = spearmanr(bbq_thirds, ss_thirds)
        r_cs, p_cs = spearmanr(crows_thirds, ss_thirds)
        print(f"  r(BBQ_dom, CrowS_dom) = {r_bc:+.3f}  p={p_bc:.3f} {stars(p_bc)}")
        print(f"  r(BBQ_dom, SS_dom)    = {r_bs:+.3f}  p={p_bs:.3f} {stars(p_bs)}")
        print(f"  r(CrowS_dom, SS_dom)  = {r_cs:+.3f}  p={p_cs:.3f} {stars(p_cs)}")

    # mid-third coverage rate
    print("\n=== Mid-third mode coverage rate ===")
    for bench in ["bbq", "crows", "ss"]:
        models_with_data = [r for r in all_results if r.get(bench)]
        has_mid = sum(1 for r in models_with_data if "mid" in r[bench]["mode_thirds"])
        print(f"  {bench:6s}: {has_mid}/{len(models_with_data)} models have a mode in mid-third")

    # Save
    out_dir = PROJ / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mode_based.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
