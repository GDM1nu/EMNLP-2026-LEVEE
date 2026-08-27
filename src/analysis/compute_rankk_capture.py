"""
LoRA-style rank-k capture analysis.

For each model × layer: what fraction of ||ΔW||²_F is captured by top-k singular vectors?
Low k → high concentration → bias signal is low-rank and transferable.

Output: results/analysis/rankk_capture.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

# model registry
ARCHIVE = ARCHIVE

MODELS: dict[str, dict] = {
    "qwen_0.5b":  {"path": ARCHIVE / "scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt",  "n_layers": 24,  "group": "le1.5b"},
    "qwen_1.5b":  {"path": ARCHIVE / "scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt",  "n_layers": 28,  "group": "le1.5b"},
    "llama32_1b": {"path": ARCHIVE / "llama_experiment/models/llama32_1b_diff/llama32_1b_diff.pt", "n_layers": 16, "group": "le1.5b"},
    "qwen_3b":    {"path": ARCHIVE / "scaling_experiment/qwen_3b/diff_vector/diff_vector.pt",    "n_layers": 36,  "group": "ge3b"},
    "llama32_3b": {"path": ARCHIVE / "llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt", "n_layers": 28, "group": "ge3b"},
    "gemma_2b":   {"path": ARCHIVE / "scaling_experiment/gemma_2b/diff_vector/diff_vector.pt",   "n_layers": 26,  "group": "ge3b"},
    "gemma_9b":   {"path": ARCHIVE / "scaling_experiment/gemma_9b/diff_vector/diff_vector.pt",   "n_layers": 42,  "group": "ge3b"},
    "llama3_8b":  {"path": ARCHIVE / "llama_experiment/models/llama_diff_vector/llama_diff_vector.pt", "n_layers": 32, "group": "ge3b"},
    "qwen_14b":   {"path": ARCHIVE / "scaling_experiment/qwen_14b/diff_vector/diff_vector.pt",   "n_layers": 48,  "group": "ge3b"},
}

K_VALUES  = [1, 2, 5, 10, 20, 50, 100]
SVD_COMPS = 120   # randomized SVD components (≥ max K_VALUES)
OUT_PATH  = Path("results/analysis/rankk_capture.json")


def _layer_matrices(diff: dict[str, torch.Tensor], layer_idx: int) -> dict[str, np.ndarray]:
    """Return 2-D weight matrices for a given layer index."""
    pat = re.compile(rf"model\.layers\.{layer_idx}\.(.+)\.weight$")
    out = {}
    for k, v in diff.items():
        m = pat.match(k)
        if m and v.ndim == 2:
            out[m.group(1)] = v.float().numpy()
    return out


def _rankk_for_matrix(W: np.ndarray) -> dict:
    """SVD-based rank-k capture for one weight matrix."""
    fro_sq = float(np.sum(W ** 2))
    if fro_sq < 1e-15:
        return {}

    max_k = min(SVD_COMPS, W.shape[0], W.shape[1])
    _, sv, _ = randomized_svd(W, n_components=max_k, random_state=42, n_iter=4)
    cumsum = np.cumsum(sv ** 2)
    total  = cumsum[-1]   # sum of top-max_k squares (≈ fro_sq for large enough max_k)

    result: dict = {}
    for k in K_VALUES:
        actual_k = min(k, len(sv))
        result[str(k)] = round(float(cumsum[actual_k - 1] / fro_sq), 6)

    # smallest k that explains ≥ 90 % of variance
    thr = 0.9 * fro_sq
    k90 = int(np.searchsorted(cumsum, thr) + 1)
    result["k90"] = min(k90, len(sv))

    return result


def process_model(alias: str, cfg: dict) -> dict:
    path     = cfg["path"]
    n_layers = cfg["n_layers"]
    group    = cfg["group"]

    print(f"  loading {alias} ({n_layers}L) …", flush=True)
    diff = torch.load(path, map_location="cpu", weights_only=True)

    layers_out = {}
    for li in tqdm(range(n_layers), desc=f"    {alias}", leave=False):
        mats = _layer_matrices(diff, li)
        if not mats:
            continue

        per_mod: dict = {}
        for mod_name, W in mats.items():
            r = _rankk_for_matrix(W)
            if r:
                per_mod[mod_name] = r

        if not per_mod:
            continue

        # Average capture across modules at this layer
        avg: dict = {}
        for k in [str(v) for v in K_VALUES]:
            vals = [per_mod[m][k] for m in per_mod if k in per_mod[m]]
            avg[k] = round(float(np.mean(vals)), 6) if vals else None
        k90_vals = [per_mod[m]["k90"] for m in per_mod if "k90" in per_mod[m]]
        avg["k90"] = round(float(np.mean(k90_vals)), 2) if k90_vals else None

        layers_out[str(li)] = {"avg": avg, "per_module": per_mod}

    # Thirds summary (avg k90 per third)
    third = n_layers // 3
    thirds: dict = {}
    for name, sl in [("early", range(0, third)),
                     ("mid",   range(third, 2 * third)),
                     ("late",  range(2 * third, n_layers))]:
        vals = [layers_out[str(li)]["avg"]["k90"]
                for li in sl
                if str(li) in layers_out and layers_out[str(li)]["avg"]["k90"] is not None]
        thirds[name] = round(float(np.mean(vals)), 2) if vals else None

    return {
        "n_layers": n_layers,
        "group":    group,
        "thirds_k90": thirds,
        "layers":   layers_out,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    for alias, cfg in MODELS.items():
        print(f"\n[rank-k] {alias}")
        results[alias] = process_model(alias, cfg)

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_PATH}")

    # Quick summary table
    print("\n=== thirds k90 summary ===")
    print(f"{'model':<14} {'group':<8} {'early':>7} {'mid':>7} {'late':>7}")
    for alias, r in results.items():
        t = r["thirds_k90"]
        print(f"{alias:<14} {r['group']:<8} {t['early']:>7.1f} {t['mid']:>7.1f} {t['late']:>7.1f}")


if __name__ == "__main__":
    main()
