"""
Task-vector-style cross-model ΔW alignment.

Two complementary metrics:

1. Energy profile Spearman correlation
   Per model: layer-wise ||ΔW||²_F summed across modules, normalized to sum=1.
   Correlate profiles between model pairs at matched normalized depth.
   ≥3B models should cluster (high r); ≤1.5B should be noisier.

2. Module-type fraction profile correlation
   For each module type (gate_proj, q_proj, …): per-layer fraction of that module's
   energy out of total layer energy → profile → pairwise Spearman across models.
   gate_proj having the highest cross-model correlation = architecture-agnostic carrier.

Output: results/analysis/crossmodel_alignment.json
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# model registry
ARCHIVE = ARCHIVE

MODELS: dict[str, dict] = {
    "qwen_0.5b":  {"path": ARCHIVE / "scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt",  "n_layers": 24, "group": "le1.5b"},
    "qwen_1.5b":  {"path": ARCHIVE / "scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt",  "n_layers": 28, "group": "le1.5b"},
    "llama32_1b": {"path": ARCHIVE / "llama_experiment/models/llama32_1b_diff/llama32_1b_diff.pt", "n_layers": 16, "group": "le1.5b"},
    "qwen_3b":    {"path": ARCHIVE / "scaling_experiment/qwen_3b/diff_vector/diff_vector.pt",    "n_layers": 36, "group": "ge3b"},
    "llama32_3b": {"path": ARCHIVE / "llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt", "n_layers": 28, "group": "ge3b"},
    "gemma_2b":   {"path": ARCHIVE / "scaling_experiment/gemma_2b/diff_vector/diff_vector.pt",   "n_layers": 26, "group": "ge3b"},
    "gemma_9b":   {"path": ARCHIVE / "scaling_experiment/gemma_9b/diff_vector/diff_vector.pt",   "n_layers": 42, "group": "ge3b"},
    "llama3_8b":  {"path": ARCHIVE / "llama_experiment/models/llama_diff_vector/llama_diff_vector.pt", "n_layers": 32, "group": "ge3b"},
    "qwen_14b":   {"path": ARCHIVE / "scaling_experiment/qwen_14b/diff_vector/diff_vector.pt",   "n_layers": 48, "group": "ge3b"},
}

N_BINS   = 20   # normalized depth bins for cross-model comparison
OUT_PATH = Path("results/analysis/crossmodel_alignment.json")

# canonical module-type short names (regex → label)
MOD_PATTERNS = [
    (re.compile(r"mlp\.gate_proj"),             "gate_proj"),
    (re.compile(r"mlp\.up_proj"),               "up_proj"),
    (re.compile(r"mlp\.down_proj"),             "down_proj"),
    (re.compile(r"self_attn\.q_proj"),          "q_proj"),
    (re.compile(r"self_attn\.k_proj"),          "k_proj"),
    (re.compile(r"self_attn\.v_proj"),          "v_proj"),
    (re.compile(r"self_attn\.o_proj"),          "o_proj"),
]


def _mod_label(key: str) -> str | None:
    for pat, label in MOD_PATTERNS:
        if pat.search(key):
            return label
    return None


def extract_profiles(alias: str, cfg: dict) -> dict:
    """Load diff vector, return per-layer energy + per-module-type fraction profiles."""
    diff = torch.load(cfg["path"], map_location="cpu", weights_only=True)
    n    = cfg["n_layers"]

    # layer_energy[li] = total ||ΔW||²_F for layer li
    layer_energy   = np.zeros(n)
    # mod_energy[mod][li] = ||ΔW_mod||²_F for that module at layer li
    mod_energy: dict[str, np.ndarray] = {m: np.zeros(n) for _, m in MOD_PATTERNS}

    for key, tensor in diff.items():
        if tensor.ndim != 2:
            continue
        m = re.search(r"model\.layers\.(\d+)\.", key)
        if not m:
            continue
        li  = int(m.group(1))
        if li >= n:
            continue
        fro_sq = float(tensor.float().pow(2).sum())
        layer_energy[li] += fro_sq

        label = _mod_label(key)
        if label:
            mod_energy[label][li] += fro_sq

    # Normalized energy profile (sum to 1)
    total = layer_energy.sum()
    energy_profile = (layer_energy / total).tolist() if total > 0 else layer_energy.tolist()

    # Normalized depth of each layer
    norm_depth = [(li + 0.5) / n for li in range(n)]

    # Thirds
    t = n // 3
    thirds = {
        "early": float(layer_energy[:t].sum() / total),
        "mid":   float(layer_energy[t:2*t].sum() / total),
        "late":  float(layer_energy[2*t:].sum() / total),
    }

    # Per-module fraction of layer energy (handle zero-energy layers)
    mod_fractions: dict[str, list] = {}
    for label, arr in mod_energy.items():
        fracs = []
        for li in range(n):
            if layer_energy[li] > 0:
                fracs.append(float(arr[li] / layer_energy[li]))
            else:
                fracs.append(0.0)
        mod_fractions[label] = fracs

    return {
        "n_layers":      n,
        "group":         cfg["group"],
        "norm_depth":    norm_depth,
        "energy_profile": energy_profile,
        "thirds":        thirds,
        "mod_fractions": mod_fractions,
    }


def _bin_profile(norm_depth: list[float], profile: list[float], n_bins: int) -> np.ndarray:
    """Average profile values into n_bins uniform normalized-depth bins."""
    bins = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    edges  = np.linspace(0, 1, n_bins + 1)
    for d, v in zip(norm_depth, profile):
        b = min(int(d * n_bins), n_bins - 1)
        bins[b]   += v
        counts[b] += 1
    mask = counts > 0
    bins[mask] /= counts[mask]
    return bins


def pairwise_spearman(profiles_dict: dict[str, dict],
                      profile_key: str,
                      group_filter: str | None = None) -> list[dict]:
    """Compute pairwise Spearman r for a profile across models."""
    aliases = [a for a, p in profiles_dict.items()
               if group_filter is None or p["group"] == group_filter]
    results = []
    for a1, a2 in combinations(aliases, 2):
        p1 = profiles_dict[a1]
        p2 = profiles_dict[a2]
        # Bin both to N_BINS for comparable length
        b1 = _bin_profile(p1["norm_depth"], p1[profile_key], N_BINS)
        b2 = _bin_profile(p2["norm_depth"], p2[profile_key], N_BINS)
        r, pval = spearmanr(b1, b2)
        results.append({"a": a1, "b": a2, "r": round(float(r), 4), "p": round(float(pval), 6)})
    return results


def mod_pairwise_spearman(profiles_dict: dict, mod_label: str,
                          group_filter: str | None = None) -> list[dict]:
    aliases = [a for a, p in profiles_dict.items()
               if (group_filter is None or p["group"] == group_filter)
               and mod_label in p["mod_fractions"]]
    results = []
    for a1, a2 in combinations(aliases, 2):
        p1 = profiles_dict[a1]
        p2 = profiles_dict[a2]
        b1 = _bin_profile(p1["norm_depth"], p1["mod_fractions"][mod_label], N_BINS)
        b2 = _bin_profile(p2["norm_depth"], p2["mod_fractions"][mod_label], N_BINS)
        r, pval = spearmanr(b1, b2)
        results.append({"a": a1, "b": a2, "r": round(float(r), 4), "p": round(float(pval), 6)})
    return results


def _mean_r(pairs: list[dict]) -> float:
    return round(float(np.mean([x["r"] for x in pairs])), 4) if pairs else float("nan")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract profiles
    print("Extracting ΔW profiles …")
    profiles: dict[str, dict] = {}
    for alias, cfg in tqdm(MODELS.items()):
        profiles[alias] = extract_profiles(alias, cfg)

    # 2. Energy profile Spearman
    print("Computing energy profile Spearman …")
    ep_ge3b  = pairwise_spearman(profiles, "energy_profile", "ge3b")
    ep_le15b = pairwise_spearman(profiles, "energy_profile", "le1.5b")
    ep_all   = pairwise_spearman(profiles, "energy_profile", None)

    energy_profile_spearman = {
        "ge3b_pairs":    ep_ge3b,
        "le1.5b_pairs":  ep_le15b,
        "all_pairs":     ep_all,
        "ge3b_mean_r":   _mean_r(ep_ge3b),
        "le1.5b_mean_r": _mean_r(ep_le15b),
        "all_mean_r":    _mean_r(ep_all),
    }

    # 3. Module-type fraction Spearman
    print("Computing module-type fraction Spearman …")
    module_type_spearman: dict = {}
    for _, mod_label in MOD_PATTERNS:
        ge3b_pairs  = mod_pairwise_spearman(profiles, mod_label, "ge3b")
        le15b_pairs = mod_pairwise_spearman(profiles, mod_label, "le1.5b")
        module_type_spearman[mod_label] = {
            "ge3b_pairs":    ge3b_pairs,
            "le1.5b_pairs":  le15b_pairs,
            "ge3b_mean_r":   _mean_r(ge3b_pairs),
            "le1.5b_mean_r": _mean_r(le15b_pairs),
        }

    # 4. Save
    # Strip heavy per_module from profiles for storage (keep only mod_fractions summary)
    per_model_out = {
        alias: {
            "n_layers":      p["n_layers"],
            "group":         p["group"],
            "thirds":        p["thirds"],
            "mod_fractions_thirds": {
                mod: {
                    "early": round(float(np.mean(p["mod_fractions"][mod][:p["n_layers"]//3])), 6),
                    "mid":   round(float(np.mean(p["mod_fractions"][mod][p["n_layers"]//3:2*p["n_layers"]//3])), 6),
                    "late":  round(float(np.mean(p["mod_fractions"][mod][2*p["n_layers"]//3:])), 6),
                }
                for mod in p["mod_fractions"]
            },
        }
        for alias, p in profiles.items()
    }

    out = {
        "n_bins":                  N_BINS,
        "energy_profile_spearman": energy_profile_spearman,
        "module_type_spearman":    module_type_spearman,
        "per_model":               per_model_out,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {OUT_PATH}")

    # 5. Summary
    print("\n=== Energy profile Spearman (mean r) ===")
    print(f"  ≥3B  pairs: {energy_profile_spearman['ge3b_mean_r']:+.3f}  (n={len(ep_ge3b)})")
    print(f"  ≤1.5B pairs: {energy_profile_spearman['le1.5b_mean_r']:+.3f}  (n={len(ep_le15b)})")

    print("\n=== Module-type ge3b mean Spearman r ===")
    rows = [(mod, v["ge3b_mean_r"]) for mod, v in module_type_spearman.items()]
    for mod, r in sorted(rows, key=lambda x: -x[1]):
        print(f"  {mod:<20} {r:+.3f}")


if __name__ == "__main__":
    main()
