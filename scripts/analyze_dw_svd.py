"""
Per-layer ΔW SVD analysis (AdaLoRA-style effective rank).

For each layer i, for each 2D weight matrix W in {Q,K,V,O,gate,up,down}:
  σ = singular values of W
  effective_rank = exp(H(σ / ‖σ‖₁))   [entropy-based, Aghajanyan 2021]
  spectral_norm  = σ_max
  nuclear_norm   = ‖σ‖₁

Per-layer aggregation: mean over all 2D matrices in that layer.

Then correlate per-layer metrics with bias suppression curve (sweep_thirds).

Hypothesis: mid-third layers have lower effective rank (more structured ΔW)
            → concentrated directional change → more effective bias transfer.

CPU-only. Loads diff_vector.pt with map_location='cpu'.

Output: results/analysis/dw_svd.json + printed table
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


def effective_rank(singular_values: np.ndarray) -> float:
    """Entropy-based effective rank (Aghajanyan et al. 2021)."""
    s = singular_values[singular_values > 0]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def layer_svd_metrics(diff: dict, n_layers: int):
    """
    Returns list of dicts, one per layer, with aggregated SVD metrics.
    Only processes 2D tensors (weight matrices); skips 1D (bias, layernorm).
    """
    buckets = defaultdict(list)
    for key, tensor in diff.items():
        m = LAYER_RE.search(key)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= n_layers:
            continue
        if tensor.ndim != 2:   # skip 1D (bias, layernorm)
            continue
        buckets[idx].append((key, tensor))

    results = []
    for layer_idx in range(n_layers):
        matrices = buckets.get(layer_idx, [])
        if not matrices:
            results.append({"layer_idx": layer_idx, "n_matrices": 0,
                            "eff_rank_mean": float("nan"),
                            "spectral_norm_mean": float("nan"),
                            "nuclear_norm_mean": float("nan"),
                            "eff_rank_norm_mean": float("nan")})
            continue

        eff_ranks, spec_norms, nuc_norms, eff_rank_norms = [], [], [], []
        stable_ranks, top_sv_concs = [], []
        for key, tensor in matrices:
            W = tensor.float().numpy()
            try:
                s = np.linalg.svd(W, compute_uv=False, full_matrices=False)
            except np.linalg.LinAlgError:
                print(f"\n  [WARN] LinAlgError on {key}, skipping", flush=True)
                continue
            except Exception as e:
                print(f"\n  [WARN] SVD failed on {key}: {e}, skipping", flush=True)
                continue
            er = effective_rank(s)
            er_norm = er / min(W.shape)
            frob_sq = float((s ** 2).sum())       # ||W||_F^2 = Σσᵢ²
            sigma1  = float(s[0])
            # stable rank = ||W||_F^2 / σ₁²  (≥1, low → one dominant direction)
            st_rank = frob_sq / (sigma1 ** 2) if sigma1 > 0 else float("nan")
            # top-SV concentration = σ₁ / Σσᵢ  (high → one direction dominates)
            nuc = float(s.sum())
            top_conc = sigma1 / nuc if nuc > 0 else float("nan")

            eff_ranks.append(er)
            eff_rank_norms.append(er_norm)
            spec_norms.append(sigma1)
            nuc_norms.append(nuc)
            stable_ranks.append(st_rank)
            top_sv_concs.append(top_conc)

        def _mean(lst): return float(np.mean(lst)) if lst else float("nan")
        results.append({
            "layer_idx": layer_idx,
            "n_matrices": len(eff_ranks),
            "eff_rank_mean":      _mean(eff_ranks),
            "spectral_norm_mean": _mean(spec_norms),
            "nuclear_norm_mean":  _mean(nuc_norms),
            "eff_rank_norm_mean": _mean(eff_rank_norms),
            "stable_rank_mean":   _mean(stable_ranks),
            "top_sv_conc_mean":   _mean(top_sv_concs),
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


def analyze_model(model_key):
    cfg = MODELS[model_key]
    diff_path = cfg["diff"]
    n_layers = cfg["n_layers"]

    if not diff_path.exists():
        print(f"  [{model_key}] diff not found: {diff_path}")
        return None

    print(f"  [{model_key}] loading {diff_path.stat().st_size/1e9:.1f}GB ...", end=" ", flush=True)
    diff = torch.load(diff_path, map_location="cpu", weights_only=True)
    print(f"SVD ({n_layers} layers) ...", end=" ", flush=True)

    layer_metrics = layer_svd_metrics(diff, n_layers)
    del diff  # free RAM immediately

    bias_curve = load_bbq_curve(model_key)

    eff_ranks    = [l["eff_rank_norm_mean"] for l in layer_metrics]
    spec_norms   = [l["spectral_norm_mean"] for l in layer_metrics]
    stable_ranks = [l["stable_rank_mean"]   for l in layer_metrics]
    top_sv_concs = [l["top_sv_conc_mean"]   for l in layer_metrics]

    def spear(a, b):
        if b and len(b) == n_layers:
            return spearmanr(a, b)
        return float("nan"), float("nan")

    r_er, p_er = spear(eff_ranks,    bias_curve)
    r_sn, p_sn = spear(spec_norms,   bias_curve)
    r_sr, p_sr = spear(stable_ranks, bias_curve)
    r_tc, p_tc = spear(top_sv_concs, bias_curve)

    # Per-third means for stable_rank (low stable_rank = structured = potentially more effective)
    thirds_sr = {"first": [], "mid": [], "last": []}
    thirds_er = {"first": [], "mid": [], "last": []}
    for l in layer_metrics:
        t = layer_to_third(l["layer_idx"], n_layers)
        if not np.isnan(l["stable_rank_mean"]):
            thirds_sr[t].append(l["stable_rank_mean"])
        if not np.isnan(l["eff_rank_norm_mean"]):
            thirds_er[t].append(l["eff_rank_norm_mean"])

    third_means_sr = {t: float(np.mean(v)) if v else float("nan") for t, v in thirds_sr.items()}
    third_means_er = {t: float(np.mean(v)) if v else float("nan") for t, v in thirds_er.items()}
    dominant_sr = min(third_means_sr, key=third_means_sr.get)
    dominant_er = min(third_means_er, key=third_means_er.get)

    print(f"ρ(er,bias)={r_er:+.3f}p={p_er:.3f}  ρ(sr,bias)={r_sr:+.3f}p={p_sr:.3f}  "
          f"ρ(tc,bias)={r_tc:+.3f}p={p_tc:.3f}  dom_sr={dominant_sr}")

    return {
        "model": model_key,
        "n_layers": n_layers,
        "layer_metrics": layer_metrics,
        "spearman_eff_rank_bias":   round(r_er, 4),
        "spearman_eff_rank_bias_p": round(p_er, 4),
        "spearman_spec_norm_bias":  round(r_sn, 4),
        "spearman_spec_norm_bias_p":round(p_sn, 4),
        "spearman_stable_rank_bias":  round(r_sr, 4),
        "spearman_stable_rank_bias_p":round(p_sr, 4),
        "spearman_top_sv_conc_bias":  round(r_tc, 4),
        "spearman_top_sv_conc_bias_p":round(p_tc, 4),
        "third_means_eff_rank_norm": third_means_er,
        "third_means_stable_rank":   third_means_sr,
        "dominant_third_low_er":  dominant_er,
        "dominant_third_low_sr":  dominant_sr,
    }


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="single model key, or omit for all")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS.keys())
    print(f"Analyzing {len(models)} models (CPU-only, no GPU needed)\n")

    out_dir = PROJ / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dw_svd.json"

    # Load any existing partial results so we can resume
    all_results = []
    done_models = set()
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if isinstance(existing, list):
                for r in existing:
                    if isinstance(r, dict) and "spearman_stable_rank_bias" in r:
                        all_results.append(r)
                        done_models.add(r["model"])
                if done_models:
                    print(f"  Resuming — already have: {sorted(done_models)}")
        except Exception:
            pass

    for m in models:
        if m in done_models:
            print(f"  [{m}] already done, skipping")
            continue
        try:
            r = analyze_model(m)
            if r:
                all_results.append(r)
                # Intermediate save after every model
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                sys.stdout.flush()
        except Exception as e:
            print(f"\n  [{m}] FAILED: {e}", flush=True)
            traceback.print_exc()

    if not all_results:
        print("No results.")
        return

    # Summary table — effective rank
    print("\n=== [1] Effective rank norm (eff_rank/min(m,n)) by third ===")
    print(f"{'Model':<14} {'first':>7} {'mid':>7} {'last':>7}  {'dom':>6}  {'ρ(er,bias)':>11} {'sig':>4}")
    print("-" * 70)
    for r in all_results:
        tm = r["third_means_eff_rank_norm"]
        print(f"{r['model']:<14} {tm['first']:>7.4f} {tm['mid']:>7.4f} {tm['last']:>7.4f}  "
              f"{r['dominant_third_low_er']:>6}  "
              f"{r['spearman_eff_rank_bias']:>+11.3f} {sig_stars(r['spearman_eff_rank_bias_p']):>4}")

    # Summary table — stable rank
    print("\n=== [2] Stable rank (||W||_F^2 / σ₁^2) by third ===")
    print(f"{'Model':<14} {'first':>8} {'mid':>8} {'last':>8}  {'dom':>6}  {'ρ(sr,bias)':>11} {'sig':>4}  {'ρ(tc,bias)':>11} {'sig':>4}")
    print("-" * 90)
    for r in all_results:
        tm = r["third_means_stable_rank"]
        print(f"{r['model']:<14} {tm['first']:>8.2f} {tm['mid']:>8.2f} {tm['last']:>8.2f}  "
              f"{r['dominant_third_low_sr']:>6}  "
              f"{r['spearman_stable_rank_bias']:>+11.3f} {sig_stars(r['spearman_stable_rank_bias_p']):>4}  "
              f"{r['spearman_top_sv_conc_bias']:>+11.3f} {sig_stars(r['spearman_top_sv_conc_bias_p']):>4}")

    # Hypothesis checks
    print("\n=== Hypothesis: mid-third = lowest stable rank (= most directional ΔW) ===")
    mid_low_sr = sum(1 for r in all_results if r["dominant_third_low_sr"] == "mid")
    mid_low_er = sum(1 for r in all_results if r["dominant_third_low_er"] == "mid")
    print(f"  stable_rank  — mid lowest: {mid_low_sr}/{len(all_results)} models")
    print(f"  eff_rank_norm— mid lowest: {mid_low_er}/{len(all_results)} models")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
