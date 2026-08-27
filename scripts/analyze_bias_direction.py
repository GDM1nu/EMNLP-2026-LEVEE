"""
Bias Direction Extraction (BDE).

Uses sweep_thirds bias scores as supervision to extract the bias-specific
weight direction from ΔW matrices — in weight space, without running the model.

Algorithm:
  w_i  = bias_score[i] - mean(bias_score)     # signed bias sensitivity per layer
  B    = Σᵢ w_i × vec(ΔW_i)                   # bias direction vector (weight space)
  align_i = cos(vec(ΔW_i), B)                 # alignment of each layer with bias axis

LOO validation (non-circular):
  B_(-i) = B - w_i × vec(ΔW_i)               # leave-one-out bias direction
  align_LOO_i = cos(vec(ΔW_i), B_(-i))        # validated alignment

Per-parameter-type bias energy:
  B_type = Σᵢ w_i × vec(ΔW_i^type)           # per-type bias direction
  energy_type = ‖B_type‖² / Σ_t ‖B_t‖²      # fraction of bias in that param type

Outputs:
  - layer_alignment (LOO-validated, non-circular)
  - per-type bias energy: which weight type (q/k/v/o/gate/up/down) carries most bias
  - B_top1_energy: rank-1 approximation quality of B (how concentrated the bias axis is)
  - Spearman(align_LOO, bias_curve) — validation

Interpretation:
  High align_LOO_i → that layer's ΔW points in the bias direction → bias hub
  High energy_type → that parameter type is the main carrier of bias change

CPU-only. Uses diff_vector.pt + sweep_thirds JSON.
Output: results/analysis/bias_direction.json
"""
import re, json, glob, sys, traceback
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from scipy.stats import spearmanr

PROJ    = Path(__file__).resolve().parent.parent
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

LAYER_RE      = re.compile(r"(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.")
PARAM_TYPE_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|qkv_proj|query_key_value|query|key|value|dense|fc1|fc2)\.weight$"
)


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
    if idx < n // 3:     return "first"
    if idx < 2 * n // 3: return "mid"
    return "last"


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def analyze_model(model_key):
    cfg       = MODELS[model_key]
    diff_path = cfg["diff"]
    n_layers  = cfg["n_layers"]

    if not diff_path.exists():
        print(f"  [{model_key}] diff not found")
        return None

    bias_curve = load_bbq_curve(model_key)
    if bias_curve is None or len(bias_curve) != n_layers:
        print(f"  [{model_key}] bias curve missing or wrong length")
        return None

    print(f"  [{model_key}] loading {diff_path.stat().st_size/1e9:.1f}GB ...", end=" ", flush=True)
    diff = torch.load(diff_path, map_location="cpu", weights_only=True)


    # Build per-layer tensor lists

    buckets      = defaultdict(list)   # layer_idx → [(key, tensor)]
    type_buckets = defaultdict(list)   # layer_idx → [(param_type, tensor)]

    for key, tensor in diff.items():
        m = LAYER_RE.search(key)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= n_layers or tensor.ndim != 2:
            continue
        buckets[idx].append((key, tensor))
        mt = PARAM_TYPE_RE.search(key)
        ptype = mt.group(1) if mt else "other"
        type_buckets[idx].append((ptype, tensor))

    del diff


    # Flatten each layer into a single float32 numpy vector

    print(f"BDE ({n_layers} layers) ...", end=" ", flush=True)

    layer_vecs = {}   # layer_idx → 1D np.float32 array
    for idx in range(n_layers):
        mats = buckets.get(idx, [])
        if not mats:
            continue
        parts = [t.float().numpy().ravel() for _, t in mats]
        layer_vecs[idx] = np.concatenate(parts)

    valid_layers = sorted(layer_vecs.keys())
    if len(valid_layers) < 3:
        print("too few layers")
        return None

    # Bias weights: w_i = b_i - mean(b)
    b_arr  = np.array([bias_curve[i] for i in valid_layers], dtype=np.float64)
    w_arr  = b_arr - b_arr.mean()   # centered


    # Build global bias direction B = Σᵢ w_i × v_i

    dim = max(v.shape[0] for v in layer_vecs.values())
    # All layers should have same dim (same architecture). Use the actual dims.
    B = np.zeros(layer_vecs[valid_layers[0]].shape, dtype=np.float64)
    for i, idx in enumerate(valid_layers):
        v = layer_vecs[idx].astype(np.float64)
        if v.shape == B.shape:
            B += w_arr[i] * v

    B_norm = np.linalg.norm(B)


    # LOO alignment: align_LOO_i = cos(v_i, B - w_i*v_i)
    # Non-circular validation

    alignments_loo = []
    alignments_raw = []
    for i, idx in enumerate(valid_layers):
        v = layer_vecs[idx].astype(np.float64)
        if v.shape != B.shape:
            alignments_loo.append(float("nan"))
            alignments_raw.append(float("nan"))
            continue
        raw = cosine(v, B)
        B_loo = B - w_arr[i] * v
        loo   = cosine(v, B_loo)
        alignments_raw.append(raw)
        alignments_loo.append(loo)


    # Per-parameter-type bias energy

    type_B = defaultdict(lambda: None)
    for i, idx in enumerate(valid_layers):
        for ptype, tensor in type_buckets.get(idx, []):
            v_t = tensor.float().numpy().ravel().astype(np.float64) * w_arr[i]
            if type_B[ptype] is None:
                type_B[ptype] = v_t.copy()
            elif type_B[ptype].shape == v_t.shape:
                type_B[ptype] += v_t

    total_energy = sum(float(np.dot(v, v)) for v in type_B.values() if v is not None)
    type_energy  = {}
    for ptype, v in type_B.items():
        if v is not None and total_energy > 0:
            type_energy[ptype] = round(float(np.dot(v, v)) / total_energy, 4)

    top_type = max(type_energy, key=type_energy.get) if type_energy else "?"


    # B rank-1 quality: top-SV energy of B reshaped to matrix
    # We approximate: reshape B to 2D and compute top singular value

    b1_energy = float("nan")
    try:
        B_sq = int(np.sqrt(len(B)))
        if B_sq * B_sq <= len(B):
            B_mat = B[:B_sq * B_sq].reshape(B_sq, B_sq)
            s     = np.linalg.svd(B_mat, compute_uv=False, full_matrices=False)
            b1_energy = round(float(s[0] ** 2 / (s ** 2).sum()), 4)
    except Exception:
        pass


    # Spearman(align_LOO, bias_curve)  — validated

    loo_arr  = np.array(alignments_loo)
    b_valid  = np.array([bias_curve[i] for i in valid_layers])
    mask     = ~np.isnan(loo_arr)
    r_val, p_val = (float("nan"), float("nan"))
    if mask.sum() >= 4:
        r_val, p_val = spearmanr(loo_arr[mask], b_valid[mask])


    # Per-third mean alignment

    thirds = {"first": [], "mid": [], "last": []}
    for i, idx in enumerate(valid_layers):
        t = layer_to_third(idx, n_layers)
        if not np.isnan(alignments_loo[i]):
            thirds[t].append(alignments_loo[i])
    third_means = {t: float(np.mean(v)) if v else float("nan") for t, v in thirds.items()}
    dominant    = max(third_means, key=lambda k: abs(third_means[k])
                      if not np.isnan(third_means[k]) else 0)

    print(f"ρ(align_LOO,bias)={r_val:+.3f}p={p_val:.3f}  "
          f"top_type={top_type}({type_energy.get(top_type,0):.2f})  dom={dominant}", flush=True)

    return {
        "model":     model_key,
        "n_layers":  n_layers,
        "spearman_align_bias":   round(r_val, 4),
        "spearman_align_bias_p": round(p_val, 4),
        "dominant_third":        dominant,
        "third_means_align_loo": third_means,
        "top_param_type":        top_type,
        "type_energy":           type_energy,
        "B_rank1_energy":        b1_energy,
        "layer_data": [
            {
                "layer_idx":    idx,
                "third":        layer_to_third(idx, n_layers),
                "bias_score":   float(bias_curve[idx]),
                "bias_weight":  float(w_arr[i]),
                "align_raw":    round(alignments_raw[i], 4),
                "align_loo":    round(alignments_loo[i], 4),
            }
            for i, idx in enumerate(valid_layers)
        ],
    }


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS.keys())
    print(f"Bias Direction Extraction — {len(models)} models (CPU-only)\n")

    out_dir  = PROJ / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_direction.json"

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


    # Summary table

    print("\n=== Bias Direction Extraction — Summary ===")
    print(f"{'Model':<14} {'first':>7} {'mid':>7} {'last':>7}  {'dom':>5}  "
          f"{'ρ(LOO,bias)':>12} {'sig':>4}  {'top_type':<12} {'B_r1':>6}")
    print("-" * 80)
    for r in all_results:
        tm = r["third_means_align_loo"]
        print(f"{r['model']:<14} "
              f"{tm['first']:>7.4f} {tm['mid']:>7.4f} {tm['last']:>7.4f}  "
              f"{r['dominant_third']:>5}  "
              f"{r['spearman_align_bias']:>+12.3f} {sig_stars(r['spearman_align_bias_p']):>4}  "
              f"{r['top_param_type']:<12} {r['B_rank1_energy']:>6.4f}")

    # Hypothesis check
    print("\n=== Hypothesis: mid-third has highest |alignment| ===")
    mid_dom = sum(1 for r in all_results if r["dominant_third"] == "mid")
    print(f"  mid dominant: {mid_dom}/{len(all_results)} models")

    print("\n=== Top parameter type frequency ===")
    from collections import Counter
    top_types = Counter(r["top_param_type"] for r in all_results)
    for t, cnt in top_types.most_common():
        print(f"  {t:<15}: {cnt}/{len(all_results)} models")

    sig_count = sum(1 for r in all_results if r["spearman_align_bias_p"] < 0.05)
    print(f"\n=== ρ(align_LOO, bias) significant (p<0.05): {sig_count}/{len(all_results)} ===")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
