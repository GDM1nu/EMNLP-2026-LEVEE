"""
ΔW direction coherence via C-matrix eigendecomposition.

C = M_unit @ M_unit.T  (Gram matrix of unit-normalized layer ΔW vectors)
Already stored in results/cosine/{model}_cosine.npz.

For each model:
  1. Eigendecomposition of C
  2. Permutation null (off-diagonal shuffle, K=1000)
  3. Spearman correlation: PC1 loadings vs BBQ sweep bias_score

Output:
  results/cosine/{model}_svd.npz
  results/cosine/SVD_SUMMARY.md
"""
import json, glob, argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJ    = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "results" / "cosine"

MODELS_ORDER = [
    "qwen_0.5b","qwen_1.5b","qwen_3b","qwen_14b",
    "gemma_2b","gemma_9b","llama32_1b","llama32_3b","llama3_8b",
]

N_LAYERS = {
    "qwen_0.5b": 24, "qwen_1.5b": 28, "qwen_3b": 36, "qwen_14b": 48,
    "gemma_2b": 26, "gemma_9b": 42,
    "llama32_1b": 16, "llama32_3b": 28, "llama3_8b": 32,
}


def load_cosine_npz(model_key):
    path = OUT_DIR / f"{model_key}_cosine.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    C = data["C"].astype(np.float64)
    n = C.shape[0]
    return C, n


def compute_eig(C, n):
    lam_asc, V_asc = np.linalg.eigh(C)
    sort_idx = np.argsort(lam_asc)[::-1]
    lam = lam_asc[sort_idx]
    V   = V_asc[:, sort_idx]
    evr = lam / n          # trace(C) = N
    cumevr = np.cumsum(evr)
    pc1 = V[:, 0]
    if pc1.mean() < 0:
        pc1 = -pc1
    return lam, V, evr, cumevr, pc1


def permutation_null(C, n, K=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    idx_i, idx_j = np.triu_indices(n, k=1)
    offdiag = C[idx_i, idx_j].copy()
    C_null = np.eye(n, dtype=np.float64)
    null_lam1 = np.empty(K)
    for k in range(K):
        perm = rng.permutation(offdiag)
        C_null[idx_i, idx_j] = perm
        C_null[idx_j, idx_i] = perm
        null_lam1[k] = np.linalg.eigvalsh(C_null)[-1]
    return null_lam1


def load_bbq_bias(model_key, n):
    pattern = str(PROJ / f"results/sweep_thirds/{model_key}/sweep_L*.json")
    files = sorted(glob.glob(pattern))
    if len(files) != n:
        # try summary
        summary = PROJ / f"results/sweep_thirds/{model_key}/sweep_summary.json"
        if summary.exists():
            with open(summary) as f:
                d = json.load(f)
            layers = sorted(d["layers"], key=lambda l: l["layer_idx"])
            return np.array([l["benchmarks"]["bbq"]["ambig"]["bias_score"] for l in layers])
        return None
    scores = []
    for fp in sorted(files):
        with open(fp) as f:
            d = json.load(f)
        scores.append(d["metrics"]["ambig"]["bias_score"])
    return np.array(scores)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def analyze_model(model_key, K=1000, rng=None, skip_existing=False):
    out_path = OUT_DIR / f"{model_key}_svd.npz"
    if skip_existing and out_path.exists():
        print(f"  [{model_key}] skip (exists)")
        return np.load(out_path, allow_pickle=True)

    C, n = load_cosine_npz(model_key)
    if C is None:
        print(f"  [{model_key}] no cosine npz, skip")
        return None

    print(f"  [{model_key}] n={n}, eigendecomposition...", end=" ", flush=True)
    lam, V, evr, cumevr, pc1 = compute_eig(C, n)

    print(f"permutation null (K={K})...", end=" ", flush=True)
    null_lam1 = permutation_null(C, n, K=K, rng=rng)
    null_mean = null_lam1.mean()
    null_std  = null_lam1.std()
    z_score   = (lam[0] - null_mean) / null_std if null_std > 0 else float("nan")
    pval      = (null_lam1 >= lam[0]).sum() / K

    bias_arr = load_bbq_bias(model_key, n)
    if bias_arr is not None and len(bias_arr) == n:
        r_bias, p_bias = spearmanr(pc1, bias_arr)
    else:
        r_bias, p_bias = float("nan"), float("nan")
        bias_arr = np.full(n, float("nan"))

    print(f"λ₁={lam[0]:.3f} z={z_score:.2f} p={pval:.3f} ρ(bias)={r_bias:.3f}")

    np.savez(out_path,
             eigenvalues=lam, eigenvectors=V, evr=evr, cumevr=cumevr, pc1=pc1,
             lambda1_null=null_lam1,
             lambda1_null_mean=null_mean, lambda1_null_std=null_std,
             lambda1_z=z_score, lambda1_pval=pval,
             spearman_pc1_bias=r_bias, spearman_pc1_bias_pval=p_bias,
             bias_arr=bias_arr, n_layers=n)
    return np.load(out_path, allow_pickle=True)


def write_summary(models):
    rows = []
    for m in models:
        path = OUT_DIR / f"{m}_svd.npz"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        lam1      = float(d["eigenvalues"][0])
        n         = int(d["n_layers"])
        evr1      = float(d["evr"][0])
        null_mean = float(d["lambda1_null_mean"])
        null_std  = float(d["lambda1_null_std"])
        z         = float(d["lambda1_z"])
        pval      = float(d["lambda1_pval"])
        r_bias    = float(d["spearman_pc1_bias"])
        p_bias    = float(d["spearman_pc1_bias_pval"])
        rows.append((m, n, lam1, evr1, null_mean, z, pval, r_bias, p_bias))

    lines = ["# SVD Direction Coherence Summary\n",
             f"Generated: 9 models\n\n",
             "## Table 1: Eigenvalue Analysis\n\n",
             "| Model | N | λ₁ | λ₁/N | null_mean | z | p | sig |\n",
             "|-------|---|-----|------|-----------|---|---|-----|\n"]
    for m, n, lam1, evr1, null_mean, z, pval, _, _ in rows:
        lines.append(f"| {m} | {n} | {lam1:.3f} | {evr1:.3f} | {null_mean:.3f} | {z:.2f} | {pval:.3f} | {sig_stars(pval)} |\n")

    lines += ["\n## Table 2: PC1 vs BBQ Bias Correlation\n\n",
              "| Model | ρ(PC1, BBQ bias) | p | sig |\n",
              "|-------|-----------------|---|-----|\n"]
    for m, n, _, _, _, _, _, r_bias, p_bias in rows:
        lines.append(f"| {m} | {r_bias:+.3f} | {p_bias:.3f} | {sig_stars(p_bias)} |\n")

    lines += ["\n## Methodological Note\n\n",
              "C = M_unit @ M_unit.T (Gram matrix of unit-normalized per-layer ΔW vectors).\n",
              "Null distribution: permute off-diagonal elements of C, K=1000.\n",
              "λ₁/N = explained variance ratio of PC1. Uniform null → λ₁/N ≈ 1/N.\n"]

    summary_path = OUT_DIR / "SVD_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"\nSummary → {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.all or args.model is None:
        models = MODELS_ORDER
    else:
        models = [args.model]

    print(f"Analyzing {len(models)} models (K={args.K}, seed={args.seed})")
    for m in models:
        analyze_model(m, K=args.K, rng=rng, skip_existing=args.skip_existing)

    write_summary(MODELS_ORDER)


if __name__ == "__main__":
    main()
