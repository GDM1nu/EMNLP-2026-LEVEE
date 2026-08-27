#!/usr/bin/env python3
"""
BBQ bias probing–bias score correlation analysis.

Correlates per-layer linear probe accuracy (correct_answer_3class)
with per-layer ΔW injection bias score (ambig bias_score from sweep_thirds).

Limitation: only llama3_8b has probing results; bias_binary label mode
is not yet implemented so we use correct_answer_3class as a proxy.

Output: results/analysis/probe_bias_corr.json
"""
from __future__ import annotations
from pathlib import Path
import json
import glob
import os
from scipy.stats import spearmanr, pearsonr
import numpy as np


PROJ_ROOT = Path(__file__).resolve().parents[2]

BASE = str(PROJ_ROOT)
PROBE_DIR = f"{BASE}/results/probing"
SWEEP_DIR = f"{BASE}/results/sweep_thirds"
OUT = f"{BASE}/results/analysis/probe_bias_corr.json"


def load_probe(alias: str) -> dict[int, float]:
    """layer_idx -> probe accuracy"""
    files = glob.glob(f"{PROBE_DIR}/{alias}_*.json")
    if not files:
        return {}
    d = json.load(open(files[0]))
    return {r["layer_idx"]: r["accuracy"] for r in d["results"]}


def load_sweep(alias: str) -> dict[int, float]:
    """layer_idx -> ambig bias_score"""
    files = sorted(glob.glob(f"{SWEEP_DIR}/{alias}/sweep_L*_bbq.json"))
    out = {}
    for f in files:
        d = json.load(open(f))
        layer = d.get("layer_idx")
        score = d.get("metrics", {}).get("ambig", {}).get("bias_score")
        if layer is not None and score is not None:
            out[layer] = score
    return out


def correlate(alias: str) -> dict | None:
    probe = load_probe(alias)
    sweep = load_sweep(alias)
    if not probe or not sweep:
        return None

    common = sorted(set(probe) & set(sweep))
    if len(common) < 5:
        return None

    x = np.array([probe[l] for l in common])   # probe accuracy
    y = np.array([sweep[l] for l in common])    # bias score

    sp_r, sp_p = spearmanr(x, y)
    pe_r, pe_p = pearsonr(x, y)

    return dict(
        alias=alias,
        n_layers=len(common),
        probe_label="correct_answer_3class",
        bias_metric="ambig_bias_score",
        spearman_r=round(float(sp_r), 4),
        spearman_p=round(float(sp_p), 4),
        pearson_r=round(float(pe_r), 4),
        pearson_p=round(float(pe_p), 4),
        probe_mean=round(float(x.mean()), 4),
        probe_std=round(float(x.std()), 4),
        bias_mean=round(float(y.mean()), 4),
        bias_std=round(float(y.std()), 4),
        per_layer=[
            dict(layer=l, probe_acc=round(float(probe[l]), 4),
                 bias_score=round(float(sweep[l]), 4))
            for l in common
        ],
    )


def main():
    # find models with probing data
    probe_files = glob.glob(f"{PROBE_DIR}/*.json")
    aliases = []
    for f in probe_files:
        name = os.path.basename(f)
        # extract alias: e.g. llama3_8b_bbq_age_probe -> llama3_8b
        parts = name.replace(".json", "").split("_")
        # match against sweep_thirds dirs
        sweep_models = os.listdir(SWEEP_DIR)
        for m in sweep_models:
            if name.startswith(m):
                aliases.append(m)
                break

    aliases = sorted(set(aliases))
    print(f"Models with probing data: {aliases}")

    results = []
    for alias in aliases:
        r = correlate(alias)
        if r:
            results.append(r)
            sig = "***" if r["spearman_p"] < 0.001 else ("**" if r["spearman_p"] < 0.01 else ("*" if r["spearman_p"] < 0.05 else "ns"))
            print(f"\n[{alias}] n={r['n_layers']} layers")
            print(f"  Spearman r={r['spearman_r']:.4f}  p={r['spearman_p']:.4f} {sig}")
            print(f"  Pearson  r={r['pearson_r']:.4f}  p={r['pearson_p']:.4f}")
            print(f"  probe acc: {r['probe_mean']:.4f} ± {r['probe_std']:.4f}")
            print(f"  bias score: {r['bias_mean']:.4f} ± {r['bias_std']:.4f}")
        else:
            print(f"[{alias}] skipped (insufficient data)")

    output = dict(
        n_models=len(results),
        probe_label="correct_answer_3class",
        bias_metric="ambig_bias_score (sweep_thirds, ambig context)",
        note=(
            "bias_binary label mode not yet implemented — correct_answer_3class "
            "used as proxy (predicts A/B/C, not bias direction directly). "
            "Only llama3_8b has probing data as of 2026-05-06."
        ),
        results=results,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
