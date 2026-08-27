#!/usr/bin/env python3
"""
TIES Sign Conflict Analysis — per-layer direction coherence of ΔW.

Based on: Yadav et al. (2023) "TIES-Merging: Resolving Interference When Merging Models"
(NeurIPS 2023)

TIES resolves sign conflicts across multiple task vectors by majority voting.
Here we adapt the metric to single-model setting:

  sign_coherence = |mean(ΔW)| / mean(|ΔW|)   ∈ [0, 1]
    → 1: all params point same direction (perfectly coherent)
    → 0: positive/negative params cancel perfectly (noisy)

  pos_frac = fraction of positive parameters
  sign_entropy = H(pos_frac, 1-pos_frac)   ∈ [0, 1]  (bits normalized)
    → 0: all same sign, → 1: perfectly 50/50

  trim_survival = after keeping top-20% magnitude (TIES "trim" step),
                  what fraction of survivors agree with layer-level elected sign?

Key question: Is mid-third sign-coherence systematically higher?
  If yes → mid-third ΔW is more "directional" (bias-specific signal)
  If no  → magnitude (norm) vs coherence (sign) are dissociable

Output: results/analysis/ties_sign_conflict.json
"""
from __future__ import annotations
from pathlib import Path
import json, os, sys
import numpy as np
import torch

BASE  = str(PROJ_ROOT)
OUT   = f"{BASE}/results/analysis/ties_sign_conflict.json"
ARCHIVE = str(ARCHIVE)
sys.path.insert(0, f"{ARCHIVE}/scaling_experiment/scripts")
from model_utils import get_num_layers, resolve_model_path


PROJ_ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("qwen_0.5b",  "Qwen/Qwen2.5-0.5B",        f"{ARCHIVE}/scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt"),
    ("qwen_1.5b",  "Qwen/Qwen2.5-1.5B",         f"{ARCHIVE}/scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt"),
    ("qwen_3b",    "Qwen/Qwen2.5-3B",            f"{ARCHIVE}/scaling_experiment/qwen_3b/diff_vector/diff_vector.pt"),
    ("qwen_14b",   "Qwen/Qwen2.5-14B",           f"{ARCHIVE}/scaling_experiment/qwen_14b/diff_vector/diff_vector.pt"),
    ("gemma_2b",   "google/gemma-2-2b",           f"{ARCHIVE}/scaling_experiment/gemma_2b/diff_vector/diff_vector.pt"),
    ("gemma_9b",   "google/gemma-2-9b",           f"{ARCHIVE}/scaling_experiment/gemma_9b/diff_vector/diff_vector.pt"),
    ("llama32_3b", "meta-llama/Llama-3.2-3B",     f"{ARCHIVE}/llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt"),
    ("llama3_8b",  "meta-llama/Meta-Llama-3-8B",  f"{ARCHIVE}/llama_experiment/models/llama_diff_vector/llama_diff_vector.pt"),
]

SWEEP_DIR = f"{BASE}/results/sweep_thirds"
TRIM_FRAC = 0.20  # keep top-20% by magnitude (TIES default)


def layer_bias_scores(alias: str) -> dict[int, float]:
    import glob
    out = {}
    for f in sorted(glob.glob(f"{SWEEP_DIR}/{alias}/sweep_L*_bbq.json")):
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

    # group params by layer
    layer_params: dict[int, list] = {i: [] for i in range(n_layers)}
    for key, tensor in diff.items():
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    li = int(parts[i + 1])
                    if 0 <= li < n_layers:
                        layer_params[li].append(tensor.float().numpy().ravel())
                except ValueError:
                    pass
                break

    layer_metrics = []
    print(f"  computing per-layer sign coherence...", flush=True)

    for li in range(n_layers):
        if not layer_params[li]:
            continue
        v = np.concatenate(layer_params[li])
        n_params = len(v)

        # sign coherence: |E[v]| / E[|v|]
        mean_v   = float(v.mean())
        mean_abs = float(np.abs(v).mean())
        sign_coherence = abs(mean_v) / (mean_abs + 1e-12)

        # pos_frac & sign entropy
        pos_frac = float((v > 0).mean())
        neg_frac = 1.0 - pos_frac
        eps = 1e-9
        sign_entropy = -(pos_frac * np.log2(pos_frac + eps) +
                         neg_frac * np.log2(neg_frac + eps))  # bits, max=1 at 50/50

        # TIES trim: keep top-TRIM_FRAC by |v|, then check sign agreement with elected sign
        n_keep = max(1, int(n_params * TRIM_FRAC))
        top_idx  = np.argpartition(np.abs(v), -n_keep)[-n_keep:]
        top_vals = v[top_idx]
        elected_sign = 1.0 if top_vals.mean() >= 0 else -1.0
        trim_survival = float((np.sign(top_vals) == elected_sign).mean())

        layer_metrics.append(dict(
            layer_idx=li,
            n_params=n_params,
            sign_coherence=round(sign_coherence, 6),
            sign_entropy=round(sign_entropy, 6),
            pos_frac=round(pos_frac, 6),
            trim_survival=round(trim_survival, 6),
            mean_abs=round(mean_abs, 6),
            bias_score=round(bias_scores.get(li, float("nan")), 6) if li in bias_scores else None,
        ))
        if li % 8 == 0:
            print(f"    L{li:02d}: coherence={sign_coherence:.4f}  entropy={sign_entropy:.4f}"
                  f"  trim_surv={trim_survival:.4f}", flush=True)

    # thirds summary
    ts = thirds(n_layers)
    thirds_coherence = {}
    thirds_entropy   = {}
    thirds_survival  = {}
    for name, idxs in ts.items():
        coh  = [m["sign_coherence"] for m in layer_metrics if m["layer_idx"] in idxs]
        ent  = [m["sign_entropy"]   for m in layer_metrics if m["layer_idx"] in idxs]
        surv = [m["trim_survival"]  for m in layer_metrics if m["layer_idx"] in idxs]
        thirds_coherence[name] = round(float(np.mean(coh)),  6) if coh  else None
        thirds_entropy[name]   = round(float(np.mean(ent)),  6) if ent  else None
        thirds_survival[name]  = round(float(np.mean(surv)), 6) if surv else None

    dom_coh  = max(thirds_coherence, key=lambda k: thirds_coherence[k] or 0)
    dom_surv = max(thirds_survival,  key=lambda k: thirds_survival[k]  or 0)

    # Spearman with bias score
    pairs_coh  = [(m["sign_coherence"], m["bias_score"]) for m in layer_metrics if m["bias_score"] is not None]
    pairs_surv = [(m["trim_survival"],  m["bias_score"]) for m in layer_metrics if m["bias_score"] is not None]

    from scipy.stats import spearmanr
    if len(pairs_coh) >= 5:
        sp_coh_r,  sp_coh_p  = spearmanr(*zip(*pairs_coh))
        sp_surv_r, sp_surv_p = spearmanr(*zip(*pairs_surv))
    else:
        sp_coh_r = sp_coh_p = sp_surv_r = sp_surv_p = float("nan")

    del diff
    import gc; gc.collect()



    return dict(
        model=alias,
        n_layers=n_layers,
        thirds_coherence=thirds_coherence,
        thirds_entropy=thirds_entropy,
        thirds_survival=thirds_survival,
        dominant_coherence=dom_coh,
        dominant_survival=dom_surv,
        spearman_coherence_bias=round(float(sp_coh_r),  4),
        spearman_coherence_bias_p=round(float(sp_coh_p),4),
        spearman_survival_bias=round(float(sp_surv_r),  4),
        spearman_survival_bias_p=round(float(sp_surv_p),4),
        layer_metrics=layer_metrics,
    )


def main():
    np.random.seed(0)
    results = []
    for alias, model_name, diff_path in MODELS:
        if not os.path.exists(diff_path):
            print(f"[skip] {alias}: {diff_path}")
            continue
        r = analyze_model(alias, model_name, diff_path)
        results.append(r)
        print(f"  thirds_coherence : {r['thirds_coherence']}  dominant={r['dominant_coherence']}")
        print(f"  thirds_survival  : {r['thirds_survival']}  dominant={r['dominant_survival']}")
        print(f"  Spearman(coherence, bias): r={r['spearman_coherence_bias']:.3f}  p={r['spearman_coherence_bias_p']:.3f}")

    output = dict(
        n_models=len(results),
        reference="Yadav et al. (NeurIPS 2023) TIES-Merging",
        trim_frac=TRIM_FRAC,
        metric_desc=dict(
            sign_coherence="|mean(ΔW)| / mean(|ΔW|) — directional consistency [0→noisy, 1→coherent]",
            sign_entropy="H(pos_frac, neg_frac) in bits — 0=all same sign, 1=perfectly mixed",
            trim_survival="fraction of top-20%% params agreeing with elected sign after TIES trim",
        ),
        results=results,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
