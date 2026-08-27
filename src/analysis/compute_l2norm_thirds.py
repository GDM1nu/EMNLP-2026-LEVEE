"""
Per-layer L2 norm of diff vectors, grouped by equal thirds.
Output: results/analysis/l2norm_thirds.pkl
"""
import re
import pickle
from pathlib import Path
from collections import defaultdict

import torch


PROJ_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE   = PROJ_ROOT.parent / "archive"  # path to model weight storage

MODELS = {
    "qwen_0.5b":  {"n_layers": 24, "diff": str(ARCHIVE / "scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt")},
    "qwen_1.5b":  {"n_layers": 28, "diff": str(ARCHIVE / "scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt")},
    "qwen_3b":    {"n_layers": 36, "diff": str(ARCHIVE / "scaling_experiment/qwen_3b/diff_vector/diff_vector.pt")},
    "qwen_14b":   {"n_layers": 48, "diff": str(ARCHIVE / "scaling_experiment/qwen_14b/diff_vector/diff_vector.pt")},
    "gemma_2b":   {"n_layers": 26, "diff": str(ARCHIVE / "scaling_experiment/gemma_2b/diff_vector/diff_vector.pt")},
    "gemma_9b":   {"n_layers": 42, "diff": str(ARCHIVE / "scaling_experiment/gemma_9b/diff_vector/diff_vector.pt")},
    "llama32_1b": {"n_layers": 16, "diff": str(ARCHIVE / "llama_experiment/models/llama32_1b_diff/llama32_1b_diff.pt")},
    "llama32_3b": {"n_layers": 28, "diff": str(ARCHIVE / "llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt")},
    "llama3_8b":  {"n_layers": 32, "diff": str(ARCHIVE / "llama_experiment/models/llama_diff_vector/llama_diff_vector.pt")},
}

OUT = Path(str(PROJ_ROOT / "results/analysis"))
OUT.mkdir(parents=True, exist_ok=True)


def compute_thirds(n):
    t1, t2 = n // 3, 2 * n // 3
    return {
        "first_third": list(range(0, t1)),
        "mid_third":   list(range(t1, t2)),
        "last_third":  list(range(t2, n)),
    }


def layer_norms(diff: dict, n_layers: int) -> list[float]:
    """Sum of squared Frobenius norms per layer index."""
    layer_sq = defaultdict(float)
    for key, tensor in diff.items():
        m = re.search(r"\.(\d+)\.", key)
        if m:
            idx = int(m.group(1))
            if idx < n_layers:
                layer_sq[idx] += tensor.float().norm().item() ** 2
    return [layer_sq[i] ** 0.5 for i in range(n_layers)]


results = {}
for alias, cfg in MODELS.items():
    n = cfg["n_layers"]
    print(f"[{alias}] loading diff... ", end="", flush=True)
    diff = torch.load(cfg["diff"], map_location="cpu", weights_only=False)
    norms = layer_norms(diff, n)
    thirds = compute_thirds(n)

    third_norms = {
        k: sum(norms[i] for i in idx_list) / len(idx_list)
        for k, idx_list in thirds.items()
    }

    results[alias] = {
        "n_layers":    n,
        "norms":       norms,          # per-layer L2 norm
        "thirds":      thirds,         # layer index groups
        "third_norms": third_norms,    # mean norm per third
        "max_norm_layer": int(max(range(n), key=lambda i: norms[i])),
        "max_norm_val":   max(norms),
    }
    print(f"done. max_norm=L{results[alias]['max_norm_layer']} ({results[alias]['max_norm_val']:.3f})")
    print(f"       first={third_norms['first_third']:.3f}  mid={third_norms['mid_third']:.3f}  last={third_norms['last_third']:.3f}")

out_path = OUT / "l2norm_thirds.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved → {out_path}")
