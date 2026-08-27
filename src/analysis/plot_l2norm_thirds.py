"""
Plot per-layer L2 norms with thirds shading + bar chart of third means.
Output: results/analysis/plots/l2norm_thirds_*.png
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA = Path(str(PROJ_ROOT / "results/analysis/l2norm_thirds.pkl"))
OUT  = Path(str(PROJ_ROOT / "results/analysis/plots"))
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA, "rb") as f:
    data = pickle.load(f)

COLORS = {"first_third": "#4C72B0", "mid_third": "#DD8452", "last_third": "#55A868"}
LABELS = {"first_third": "First third", "mid_third": "Mid third", "last_third": "Last third"}

# 1. Per-layer line plots (one per model)
for alias, d in data.items():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    n = d["n_layers"]
    norms = d["norms"]
    thirds = d["thirds"]

    for third, color in COLORS.items():
        idx = thirds[third]
        ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, alpha=0.12, color=color, label=LABELS[third])

    ax.plot(range(n), norms, color="black", linewidth=1.2, zorder=3)
    ax.scatter(range(n), norms, color="black", s=18, zorder=4)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("L2 norm")
    ax.set_title(f"{alias}  (n_layers={n})")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-0.5, n - 0.5)
    plt.tight_layout()
    fig.savefig(OUT / f"l2norm_perlayer_{alias}.png", dpi=150)
    plt.close(fig)
    print(f"saved: l2norm_perlayer_{alias}.png")

# 2. Overlay: all models normalized (0-1)
fig, ax = plt.subplots(figsize=(8, 4))
for alias, d in data.items():
    norms = np.array(d["norms"])
    n = d["n_layers"]
    xs = np.array(range(n)) / (n - 1)
    mn, mx = norms.min(), norms.max()
    normed = (norms - mn) / (mx - mn + 1e-12)
    ax.plot(xs, normed, alpha=0.7, linewidth=1.2, label=alias)

ax.axvspan(0,     1/3, alpha=0.07, color=COLORS["first_third"])
ax.axvspan(1/3,   2/3, alpha=0.07, color=COLORS["mid_third"])
ax.axvspan(2/3,   1.0, alpha=0.07, color=COLORS["last_third"])
ax.set_xlabel("Relative depth (thirds boundary)")
ax.set_ylabel("Normalized L2 norm")
ax.set_title("Per-layer L2 norm — all models (normalized)")
ax.legend(fontsize=7, ncol=3, loc="upper left")
ax.set_xlim(0, 1)
ax.axvline(1/3, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(2/3, color="gray", linewidth=0.8, linestyle="--")
plt.tight_layout()
fig.savefig(OUT / "l2norm_overlay.png", dpi=150)
plt.close(fig)
print("saved: l2norm_overlay.png")

# 3. Bar chart: mean norm per third
aliases = list(data.keys())
x = np.arange(len(aliases))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4))
for i, third in enumerate(["first_third", "mid_third", "last_third"]):
    vals = [data[a]["third_norms"][third] for a in aliases]
    ax.bar(x + i * width, vals, width, label=LABELS[third], color=COLORS[third], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(aliases, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Mean L2 norm")
ax.set_title("Mean L2 norm per third — all models")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "l2norm_thirds_bar.png", dpi=150)
plt.close(fig)
print("saved: l2norm_thirds_bar.png")
