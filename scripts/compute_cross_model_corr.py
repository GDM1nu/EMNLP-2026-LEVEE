"""
Cross-model BBQ sweep curve correlation.

Interpolates per-layer sweep curves for all 11 models onto a normalized depth
grid (0–1, 100 points) and computes pairwise Pearson r between models.

Output:
  results/cosine/cross_model_corr.json   — r/p table
  results/cosine/cross_model_corr.png    — heatmap
"""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d


PROJ = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJ / "results" / "sweep_thirds"
OUT_DIR = PROJ / "results" / "cosine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen_0.5b":   {"n_layers": 24,  "display": "Qwen2.5 0.5B"},
    "qwen_1.5b":   {"n_layers": 28,  "display": "Qwen2.5 1.5B"},
    "qwen_3b":     {"n_layers": 36,  "display": "Qwen2.5 3B"},
    "qwen_14b":    {"n_layers": 48,  "display": "Qwen2.5 14B"},
    "gemma_2b":    {"n_layers": 26,  "display": "Gemma2 2B"},
    "gemma_9b":    {"n_layers": 42,  "display": "Gemma2 9B"},
    "gemma3_12b":  {"n_layers": 48,  "display": "Gemma3 12B"},
    "llama32_1b":  {"n_layers": 16,  "display": "LLaMA3.2 1B"},
    "llama32_3b":  {"n_layers": 28,  "display": "LLaMA3.2 3B"},
    "llama3_8b":   {"n_layers": 32,  "display": "LLaMA3 8B"},
    "llama31_8b":  {"n_layers": 32,  "display": "LLaMA3.1 8B"},
}

GRID_SIZE = 100  # interpolation points along normalized depth
GRID = np.linspace(0, 1, GRID_SIZE)


def load_sweep_curve(alias: str, n_layers: int) -> np.ndarray | None:
    """Load per-layer ambig bias_score and interpolate to a 100-point normalized-depth array."""
    sweep_path = SWEEP_DIR / alias
    if not sweep_path.exists():
        print(f"  [{alias}] sweep directory missing — skipping")
        return None

    layer_scores = {}
    for f in sweep_path.glob("sweep_L*_bbq.json"):
        try:
            d = json.loads(f.read_text())
            l = int(d["layer_idx"])
            s = float(d["metrics"]["ambig"]["bias_score"])
            layer_scores[l] = s
        except Exception:
            pass

    if len(layer_scores) < 4:
        print(f"  [{alias}] only {len(layer_scores)} sweep layers — skipping (too few)")
        return None

    layers = sorted(layer_scores)
    scores = [layer_scores[l] for l in layers]
    depths = [l / (n_layers - 1) for l in layers]

    # linear interpolation, no boundary extrapolation
    f_interp = interp1d(depths, scores, kind="linear",
                        bounds_error=False, fill_value=(scores[0], scores[-1]))
    curve = f_interp(GRID)

    print(f"  [{alias}] {len(layer_scores)}/{n_layers} layers, "
          f"bias_score range [{min(scores):.4f}, {max(scores):.4f}]")
    return curve


def compute_all_curves() -> dict[str, np.ndarray]:
    print("=== loading and interpolating sweep curves ===")
    curves = {}
    for alias, cfg in MODELS.items():
        curve = load_sweep_curve(alias, cfg["n_layers"])
        if curve is not None:
            # zero-variance curve: Pearson r undefined → mark in heatmap
            if np.std(curve) < 1e-6:
                print(f"  [{alias}] zero variance — will show '—' in heatmap")
                curves[alias] = None
            else:
                curves[alias] = curve
    return curves


def compute_pairwise(curves: dict) -> dict:
    """Compute pairwise Pearson r and p-value for all model pairs."""
    print("\n=== pairwise Pearson r ===")
    aliases = list(curves.keys())
    results = {}

    for i, a in enumerate(aliases):
        for j, b in enumerate(aliases):
            if j <= i:
                continue
            ca, cb = curves[a], curves[b]
            if ca is None or cb is None:
                r, p = None, None
                print(f"  {a} vs {b}: — (zero-variance model)")
            else:
                r, p = pearsonr(ca, cb)
                r, p = float(r), float(p)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {a:14} vs {b:14}  r={r:+.3f}  p={p:.3f}{sig}")
            results[f"{a}__vs__{b}"] = {"model_a": a, "model_b": b, "r": r, "p": p}

    return results


def save_json(curves: dict, pairwise: dict):
    out = {
        "models": {
            alias: {
                "n_layers": MODELS[alias]["n_layers"],
                "display": MODELS[alias]["display"],
                "n_sweep_layers": int(np.sum(~np.isnan(curves[alias]))) if curves[alias] is not None else 0,
                "has_curve": curves[alias] is not None,
            }
            for alias in curves
        },
        "grid_size": GRID_SIZE,
        "pairwise": pairwise,
    }
    path = OUT_DIR / "cross_model_corr.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> saved: {path}")


def plot_heatmap(curves: dict, pairwise: dict):
    aliases = [a for a in MODELS if a in curves]
    n = len(aliases)
    labels = [MODELS[a]["display"] for a in aliases]

    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    alias2i = {a: i for i, a in enumerate(aliases)}
    for key, v in pairwise.items():
        a, b = v["model_a"], v["model_b"]
        r = v["r"]
        if r is None:
            continue
        i, j = alias2i[a], alias2i[b]
        mat[i, j] = r
        mat[j, i] = r

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center", fontsize=7, color="white")
            elif np.isfinite(mat[i, j]):
                v = mat[i, j]
                color = "white" if abs(v) > 0.6 else "black"
                # look up p-value
                key1 = f"{aliases[min(i,j)]}__vs__{aliases[max(i,j)]}"
                pv = pairwise.get(key1, {}).get("p", None)
                sig = "**" if pv is not None and pv < 0.01 else "*" if pv is not None and pv < 0.05 else ""
                ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#aaaaaa")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(
        f"Cross-model BBQ sweep curve correlation (Pearson r, n={GRID_SIZE})\n"
        "normalized depth interpolation | * p<0.05  ** p<0.01",
        fontsize=10
    )
    fig.tight_layout()

    out = OUT_DIR / "cross_model_corr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> saved: {out}")


def print_summary(pairwise: dict):
    valid = [(v["r"], v["p"], v["model_a"], v["model_b"])
             for v in pairwise.values() if v["r"] is not None]
    valid.sort(key=lambda x: -abs(x[0]))

    print("\n=== significant pairs (p<0.05) ===")
    sig = [(r, p, a, b) for r, p, a, b in valid if p < 0.05]
    for r, p, a, b in sig:
        same = MODELS[a].get("family", a.split("_")[0]) == MODELS[b].get("family", b.split("_")[0])
        tag = "same" if same else "cross"
        print(f"  {a:14} vs {b:14}  r={r:+.3f}  p={p:.3f}  [{tag}-family]")

    print(f"\nsignificant pairs: {len(sig)}/{len(valid)}")


if __name__ == "__main__":
    curves = compute_all_curves()
    pairwise = compute_pairwise(curves)
    save_json(curves, pairwise)
    plot_heatmap(curves, pairwise)
    print_summary(pairwise)
    print("\n=== done ===")
