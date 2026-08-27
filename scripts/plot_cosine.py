"""
Cosine similarity visualization — 3 figures.

Plot A: 9-model N×N heatmap grid
Plot B: Distance-decay overlay (all models, family-colored)
Plot C: Ref-direction cosine 3×3 grid
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "results" / "cosine"

MODELS_ORDER = [
    "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_14b",
    "gemma_2b", "gemma_9b",
    "llama32_1b", "llama32_3b", "llama3_8b",
]

DISPLAY = {
    "qwen_0.5b":  "Qwen 0.5B",  "qwen_1.5b":  "Qwen 1.5B",
    "qwen_3b":    "Qwen 3B",    "qwen_14b":   "Qwen 14B",
    "gemma_2b":   "Gemma 2B",   "gemma_9b":   "Gemma 9B",
    "llama32_1b": "LLaMA3.2-1B","llama32_3b": "LLaMA3.2-3B",
    "llama3_8b":  "LLaMA3-8B",
}

FAMILY_COLORS = {
    # Qwen: blue tones
    "qwen_0.5b": "#aec6cf", "qwen_1.5b": "#6aabbb", "qwen_3b": "#2980b9", "qwen_14b": "#1a5276",
    # Gemma: green tones
    "gemma_2b": "#82c785", "gemma_9b": "#1e8449",
    # LLaMA: orange/red tones
    "llama32_1b": "#f9a87f", "llama32_3b": "#e05c1a", "llama3_8b": "#7b241c",
}

N_LAYERS = {
    "qwen_0.5b": 24, "qwen_1.5b": 28, "qwen_3b": 36, "qwen_14b": 48,
    "gemma_2b": 26, "gemma_9b": 42,
    "llama32_1b": 16, "llama32_3b": 28, "llama3_8b": 32,
}


def load_npz(model_key: str) -> dict | None:
    path = OUT_DIR / f"{model_key}_cosine.npz"
    if not path.exists():
        print(f"  ⚠ {model_key}: {path.name} not found — skipping")
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "C": d["C"],
        "distance_mean": d["distance_mean"],
        "ref_cosine": d["ref_cosine"],
        "n_layers": int(d["n_layers"]),
    }


# Plot A: N×N heatmap grid


def plot_A_heatmap(data: dict):
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes = axes.flatten()

    for ax, key in zip(axes, MODELS_ORDER):
        if key not in data:
            ax.set_visible(False)
            continue
        d = data[key]
        C = d["C"]
        n = d["n_layers"]
        t1, t2 = n // 3, 2 * n // 3

        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", origin="upper", aspect="equal")
        ax.set_title(DISPLAY[key], fontsize=10, fontweight="bold")
        ax.set_xlabel("Layer index", fontsize=7)
        ax.set_ylabel("Layer index", fontsize=7)
        ax.tick_params(labelsize=6)

        # thirds dashed lines
        for boundary in [t1, t2]:
            ax.axhline(boundary - 0.5, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(boundary - 0.5, color="k", linewidth=0.8, linestyle="--", alpha=0.6)

        # thirds labels
        for pos, label in [(t1 // 2, "first"), ((t1 + t2) // 2, "mid"), ((t2 + n) // 2, "last")]:
            ax.text(pos, -1.5, label, ha="center", va="bottom", fontsize=5, color="gray",
                    transform=ax.get_xaxis_transform())

    # shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="cosine similarity")

    fig.suptitle("Layer-pair ΔW cosine similarity (within-model)", fontsize=13, y=1.0)
    out = OUT_DIR / "fig_A_heatmap_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot A] saved {out.name}")


# Plot B: distance-decay overlay


def plot_B_distance_decay(data: dict):
    fig, ax = plt.subplots(figsize=(9, 5))

    for key in MODELS_ORDER:
        if key not in data:
            continue
        d = data[key]
        dm = d["distance_mean"]
        n = d["n_layers"]
        # x: normalized distance [0, 1)
        x = np.arange(1, n) / (n - 1) if n > 1 else np.array([0.0])
        ax.plot(x, dm, color=FAMILY_COLORS[key], label=DISPLAY[key], linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Normalized layer distance |i−j| / (N−1)", fontsize=11)
    ax.set_ylabel("Mean cosine similarity", fontsize=11)
    ax.set_title("Mean cosine vs normalized layer distance", fontsize=12)
    ax.text(0.5, -0.12,
            "Higher = more coherent direction across nearby layers",
            ha="center", transform=ax.transAxes, fontsize=9, color="gray",
            style="italic")

    # family legend patches + line legend
    qwen_patch  = mpatches.Patch(color="#2980b9",  label="Qwen (0.5B→14B)")
    gemma_patch = mpatches.Patch(color="#1e8449",  label="Gemma (2B→9B)")
    llama_patch = mpatches.Patch(color="#e05c1a",  label="LLaMA (1B→8B)")
    ax.legend(handles=[qwen_patch, gemma_patch, llama_patch],
              loc="upper right", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig_B_distance_decay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot B] saved {out.name}")


# Plot C: ref-direction cosine 3×3 grid


THIRDS_COLORS = {"first": "#4a90d9", "mid": "#e6892b", "last": "#3cae6f"}

def plot_C_ref_direction(data: dict):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for ax, key in zip(axes, MODELS_ORDER):
        if key not in data:
            ax.set_visible(False)
            continue
        d = data[key]
        rc = d["ref_cosine"]
        n = d["n_layers"]
        t1, t2 = n // 3, 2 * n // 3
        layers = np.arange(n)

        # shade thirds
        ax.axvspan(-0.5, t1 - 0.5, alpha=0.10, color=THIRDS_COLORS["first"])
        ax.axvspan(t1 - 0.5, t2 - 0.5, alpha=0.10, color=THIRDS_COLORS["mid"])
        ax.axvspan(t2 - 0.5, n - 0.5, alpha=0.10, color=THIRDS_COLORS["last"])

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.plot(layers, rc, color=FAMILY_COLORS[key], linewidth=1.8)

        # mark sign-flip points (cosine crosses zero)
        for i in range(len(rc) - 1):
            if rc[i] * rc[i + 1] < 0:
                ax.axvline(i + 0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

        # LLaMA 3-8B special annotation
        if key == "llama3_8b":
            # find layer closest to reversal around L16
            dip_idx = int(np.argmin(rc))
            ax.axvline(dip_idx, color="crimson", linewidth=1.2, linestyle="-", alpha=0.8)
            ax.annotate(
                "reversal",
                xy=(dip_idx, rc[dip_idx]),
                xytext=(dip_idx + 1.5, rc[dip_idx] + 0.05),
                fontsize=6, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
            )

        ax.set_title(DISPLAY[key], fontsize=10, fontweight="bold")
        ax.set_xlabel("Layer index", fontsize=7)
        ax.set_ylabel("cos(ΔW_i, ref)", fontsize=7)
        ax.set_xlim(-0.5, n - 0.5)
        ax.tick_params(labelsize=6)

        # thirds labels on x-axis
        ax.text((t1 // 2) / n, -0.13, "first", ha="center", transform=ax.transAxes,
                fontsize=6, color=THIRDS_COLORS["first"])
        ax.text(((t1 + t2) / 2) / n, -0.13, "mid", ha="center", transform=ax.transAxes,
                fontsize=6, color=THIRDS_COLORS["mid"])
        ax.text(((t2 + n) / 2) / n, -0.13, "last", ha="center", transform=ax.transAxes,
                fontsize=6, color=THIRDS_COLORS["last"])

    fig.suptitle("cos(ΔW_i, ref_first_half) — direction coherence with first-half mean",
                 fontsize=12, y=1.0)

    # legend for thirds
    patches = [mpatches.Patch(color=THIRDS_COLORS[t], alpha=0.4, label=f"{t} third")
               for t in ("first", "mid", "last")]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(hspace=0.5, wspace=0.35)
    out = OUT_DIR / "fig_C_ref_direction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot C] saved {out.name}")


# Main


def main():
    data = {}
    for key in MODELS_ORDER:
        d = load_npz(key)
        if d is not None:
            data[key] = d

    if not data:
        print("No .npz files found. Run compute_cosine.py first.")
        sys.exit(1)

    print(f"Loaded {len(data)}/9 models: {list(data.keys())}")

    plot_A_heatmap(data)
    plot_B_distance_decay(data)
    plot_C_ref_direction(data)

    print("\n=== All figures saved to", OUT_DIR, "===")


if __name__ == "__main__":
    main()
