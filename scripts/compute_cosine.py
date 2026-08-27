"""
Within-model layer-pair ΔW cosine similarity analysis.

Three statistics computed:
  1. N×N cosine similarity matrix  C[i,j] = cos(ΔW_i, ΔW_j)
  2. Distance-binned mean cosine   mean(C[i,j] for |i-j|=k), k=1..N-1
  3. Reference-direction cosine    ref = normalize(mean(unit_ΔW_i for i < N//2))
                                   ref_cosine[i] = cos(ΔW_i, ref)

Output: results/cosine/{model_key}_cosine.npz
Auto-report: results/cosine/SUMMARY.md
"""
import re, sys, argparse, time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "results" / "cosine"
LOG_DIR = PROJ / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE = ARCHIVE

MODELS = {
    "qwen_0.5b": {
        "n_layers": 24, "family": "qwen",
        "diff": ARCHIVE / "scaling_experiment/qwen_0.5b/diff_vector/diff_vector.pt",
    },
    "qwen_1.5b": {
        "n_layers": 28, "family": "qwen",
        "diff": ARCHIVE / "scaling_experiment/qwen_1.5b/diff_vector/diff_vector.pt",
    },
    "qwen_3b": {
        "n_layers": 36, "family": "qwen",
        "diff": ARCHIVE / "scaling_experiment/qwen_3b/diff_vector/diff_vector.pt",
    },
    "qwen_14b": {
        "n_layers": 48, "family": "qwen",
        "diff": ARCHIVE / "scaling_experiment/qwen_14b/diff_vector/diff_vector.pt",
    },
    "gemma_2b": {
        "n_layers": 26, "family": "gemma",
        "diff": ARCHIVE / "scaling_experiment/gemma_2b/diff_vector/diff_vector.pt",
    },
    "gemma_9b": {
        "n_layers": 42, "family": "gemma",
        "diff": ARCHIVE / "scaling_experiment/gemma_9b/diff_vector/diff_vector.pt",
    },
    "llama32_1b": {
        "n_layers": 16, "family": "llama",
        "diff": ARCHIVE / "llama_experiment/models/llama32_1b_diff/llama32_1b_diff.pt",
    },
    "llama32_3b": {
        "n_layers": 28, "family": "llama",
        "diff": ARCHIVE / "llama_experiment/models/llama32_3b_diff/llama32_3b_diff.pt",
    },
    "llama3_8b": {
        "n_layers": 32, "family": "llama",
        "diff": ARCHIVE / "llama_experiment/models/llama_diff_vector/llama_diff_vector.pt",
    },
}


# Core


def layer_vectors(diff: dict, n_layers: int) -> list:
    """Flatten and concat all parameters per layer into a list of float32 1D tensors."""
    buckets = defaultdict(list)
    for key, tensor in diff.items():
        m = re.search(r"\.(\d+)\.", key)
        if m:
            idx = int(m.group(1))
            if idx < n_layers:
                buckets[idx].append(tensor.float().reshape(-1))
    vecs = []
    for i in range(n_layers):
        if i in buckets:
            vecs.append(torch.cat(buckets[i]))
        else:
            print(f"  ⚠ layer {i} has no keys — using zeros", flush=True)
            vecs.append(torch.zeros(1))
    return vecs


def compute_cosine_analysis(model_key: str, cfg: dict) -> dict:
    n = cfg["n_layers"]
    diff_path = cfg["diff"]

    print(f"\n[{model_key}] loading diff ({Path(diff_path).stat().st_size / 1e9:.1f} GB)...",
          end=" ", flush=True)
    t0 = time.time()
    diff = torch.load(diff_path, map_location="cpu", weights_only=False)
    print(f"done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  extracting {n} layer vectors...", end=" ", flush=True)
    t0 = time.time()
    vecs = layer_vectors(diff, n)
    del diff  # free memory immediately
    print(f"done ({time.time()-t0:.1f}s)", flush=True)

    # norms
    norms = torch.tensor([v.norm().item() for v in vecs], dtype=torch.float32)

    # unit vectors — for layers with zero norm, set to zero (cosine undefined)
    unit_vecs = []
    for i, v in enumerate(vecs):
        norm = norms[i].item()
        if norm > 0:
            unit_vecs.append(v / norm)
        else:
            unit_vecs.append(torch.zeros_like(v))

    # N×N cosine matrix
    # Stack unit vecs: shape differs per layer, cannot directly stack into (N, D)
    # Compute pairwise dot products iteratively (memory-safe for large D)
    print(f"  computing {n}×{n} cosine matrix...", end=" ", flush=True)
    t0 = time.time()
    C = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(i, n):
            # unit_vecs[i] and unit_vecs[j] may have different lengths if bucket sizes differ
            # (shouldn't happen, but guard)
            u, v_ = unit_vecs[i], unit_vecs[j]
            if u.shape == v_.shape:
                val = float((u * v_).sum().item())
            else:
                val = float("nan")
            val = max(-1.0, min(1.0, val))  # clamp numerical noise
            C[i, j] = val
            C[j, i] = val
    print(f"done ({time.time()-t0:.1f}s)", flush=True)

    # sanity check
    diag_max_err = (C.diagonal() - 1.0).abs().max().item()
    if diag_max_err > 1e-4:
        print(f"  ⚠ diagonal max error: {diag_max_err:.2e}")
    assert C.shape == (n, n)
    assert torch.allclose(C, C.T, atol=1e-5), "C not symmetric"

    # Distance-binned mean cosine
    distance_mean = torch.zeros(n - 1, dtype=torch.float32)
    distance_count = torch.zeros(n - 1, dtype=torch.int32)
    for i in range(n):
        for j in range(i + 1, n):
            k = j - i  # distance 1..N-1
            distance_mean[k - 1] += C[i, j].item()
            distance_count[k - 1] += 1
    for k in range(n - 1):
        if distance_count[k] > 0:
            distance_mean[k] /= distance_count[k]

    # Reference-direction cosine
    half = n // 2
    # ref = mean of unit first-half layer vectors
    ref = torch.stack(unit_vecs[:half]).mean(dim=0)
    ref_norm = ref.norm()
    if ref_norm > 0:
        ref = ref / ref_norm

    ref_cosine = torch.tensor(
        [float((unit_vecs[i] * ref).sum().item()) for i in range(n)],
        dtype=torch.float32,
    )
    ref_cosine = ref_cosine.clamp(-1.0, 1.0)

    # summary stats
    # off-diagonal mean (upper triangle)
    triu_idx = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    off_diag_vals = C[triu_idx]
    mean_offdiag = float(off_diag_vals.mean().item())

    # first-half mean for ref_cosine
    ref_first = float(ref_cosine[:half].mean().item())
    ref_last  = float(ref_cosine[half:].mean().item())
    sign_flips = int(((ref_cosine[:-1] > 0) & (ref_cosine[1:] <= 0)).sum().item() +
                     ((ref_cosine[:-1] <= 0) & (ref_cosine[1:] > 0)).sum().item())

    print(f"  mean off-diag cosine: {mean_offdiag:.4f}")
    print(f"  ref_cosine first-half: {ref_first:.4f}  last-half: {ref_last:.4f}")
    print(f"  ref_cosine sign flips: {sign_flips}")

    return dict(
        C=C.numpy(),
        distance_mean=distance_mean.numpy(),
        ref_cosine=ref_cosine.numpy(),
        n_layers=n,
        model_key=model_key,
        family=cfg["family"],
        mean_offdiag=mean_offdiag,
        ref_first=ref_first,
        ref_last=ref_last,
        sign_flips=sign_flips,
    )


def run_model(model_key: str, skip_existing: bool = False) -> dict | None:
    cfg = MODELS[model_key]
    out_path = OUT_DIR / f"{model_key}_cosine.npz"

    if skip_existing and out_path.exists():
        print(f"[{model_key}] skipping — {out_path.name} already exists")
        return None

    log_path = LOG_DIR / f"cosine_{model_key}.log"
    import subprocess, shutil
    # just run inline; caller redirects if needed
    result = compute_cosine_analysis(model_key, cfg)

    np.savez(
        out_path,
        C=result["C"],
        distance_mean=result["distance_mean"],
        ref_cosine=result["ref_cosine"],
        n_layers=np.array(result["n_layers"]),
        model_key=np.array(result["model_key"]),
    )
    print(f"  → saved {out_path}")
    return result


# SUMMARY.md


def load_result_from_npz(model_key: str, cfg: dict) -> dict | None:
    """Load summary stats from existing .npz (used when compute was skipped)."""
    path = OUT_DIR / f"{model_key}_cosine.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    n = int(d["n_layers"])
    C = d["C"]
    rc = d["ref_cosine"]
    half = n // 2
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_offdiag = float(C[triu].mean())
    return dict(
        model_key=model_key,
        family=cfg["family"],
        n_layers=n,
        C=C,
        distance_mean=d["distance_mean"],
        ref_cosine=rc,
        mean_offdiag=mean_offdiag,
        ref_first=float(rc[:half].mean()),
        ref_last=float(rc[half:].mean()),
        sign_flips=int(((rc[:-1] > 0) & (rc[1:] <= 0)).sum() +
                       ((rc[:-1] <= 0) & (rc[1:] > 0)).sum()),
    )


def write_summary(results: list[dict]):
    MODELS_ORDER = list(MODELS.keys())
    # merge with any existing npz not in results
    result_map = {r["model_key"]: r for r in results}
    for key, cfg in MODELS.items():
        if key not in result_map:
            r = load_result_from_npz(key, cfg)
            if r is not None:
                result_map[key] = r
    ordered = [result_map[k] for k in MODELS_ORDER if k in result_map]

    lines = [
        "# Cosine Analysis Summary",
        "",
        "## Model-wise statistics",
        "",
        "| Model | n_layers | Mean off-diag cosine | ref_cos first-half | null (1/√(N/2)) | ref_cos last-half | Sign-flip layers |",
        "|-------|----------|---------------------|-------------------|-----------------|------------------|-----------------|",
    ]
    for r in ordered:
        null_val = 1.0 / (r["n_layers"] // 2) ** 0.5
        lines.append(
            f"| {r['model_key']} | {r['n_layers']} | {r['mean_offdiag']:+.5f} | "
            f"{r['ref_first']:.4f} | {null_val:.4f} | "
            f"{r['ref_last']:.4f} | {r['sign_flips']} |"
        )

    lines += [
        "",
        "## Reference-direction analysis",
        "",
        "| Model | min ref_cosine | at layer | max ref_cosine | at layer |",
        "|-------|----------------|----------|----------------|----------|",
    ]
    for r in ordered:
        rc = r["ref_cosine"]
        lines.append(
            f"| {r['model_key']} | {rc.min():.4f} | L{int(rc.argmin())} | "
            f"{rc.max():.4f} | L{int(rc.argmax())} |"
        )

    lines += [
        "",
        "## ⚠️ Methodological note",
        "",
        "All pairwise layer cosines ≈ 0 (layers are mutually near-orthogonal in ΔW space).",
        "Under this condition, cos(v_i, normalize(mean(v_j))) = 1/√N — **not a signal.**",
        "ref_cosine first-half values match the null exactly (see null column).",
        "To detect genuine direction coherence, SVD/PCA with null baseline is required.",
        "",
        "## Key observations",
        "",
        "- [ ] LLaMA 3-8B: verify L16 cosine reversal (current: ref_cosine min at L26, magnitude ≈ 0)",
        "- [ ] Gemma 9B (cumulative works well): verify first-half all-positive pattern",
        "- [ ] Qwen 0.5B/1.5B (cumulative fails): verify cosine pattern",
        "- [ ] **TODO**: re-measure direction coherence with SVD analysis",
        "",
        "> Generated by `scripts/compute_cosine.py`",
    ]

    path = OUT_DIR / "SUMMARY.md"
    path.write_text("\n".join(lines))
    print(f"\n→ SUMMARY.md saved to {path} ({len(ordered)}/9 models)")


# CLI


def main():
    parser = argparse.ArgumentParser(description="Within-model layer-pair cosine analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODELS.keys()), help="single model")
    group.add_argument("--all", action="store_true", help="all 9 models sequentially")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip if .npz already exists")
    args = parser.parse_args()

    if args.model:
        keys = [args.model]
    else:
        keys = list(MODELS.keys())

    completed = []
    for key in keys:
        r = run_model(key, skip_existing=args.skip_existing)
        if r is not None:
            completed.append(r)

    write_summary(completed)

    print("\n=== Done ===")
    for r in completed:
        print(f"  {r['model_key']}: N={r['n_layers']}, "
              f"mean_offdiag={r['mean_offdiag']:.4f}, "
              f"ref_first={r['ref_first']:.4f}, ref_last={r['ref_last']:.4f}")


if __name__ == "__main__":
    main()
