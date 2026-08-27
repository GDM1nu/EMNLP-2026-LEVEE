"""
ATLAS-inspired figures for bias_transfer_v2 paper.
Reference: Adiga et al. (2025) "Attention Speaks Volumes: Localizing and
           Mitigating Bias in Language Models", ACL 2025.

Fig 1 (ATLAS Fig 3 analog): Per-layer ΔW injection score heatmap
  - X: normalized layer depth [0, 1], third boundaries marked
  - Y: 11 models grouped by architecture family
  - Color: BBQ ambiguous bias score (lower = better debiasing)
  - Contrast: ATLAS finds last-third via attention; we find mid-third via ΔW

Fig 2 (ATLAS Fig 4 analog): Block removal comparison bar chart
  - Analog to ATLAS's "top-k vs. random/other layers" intervention comparison
  - 5 models × 3 conditions: -mid, -first, -last
  - Shows mid-third is consistently the most critical block (ΔBBQ highest)

Usage:
    conda activate bias_transfer_clean
    cd $PROJECT_ROOT
    python scripts/make_atlas_inspired_figures.py

Output: results/paper_figures/atlas_fig{1,2}.{pdf,png}
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = "results/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Model registry
# (alias, display_label, family, n_layers)
MODELS = [
    ("qwen_0.5b",  "Qwen-0.5B",     "Qwen",  24),
    ("qwen_1.5b",  "Qwen-1.5B",     "Qwen",  28),
    ("qwen_3b",    "Qwen-3B",       "Qwen",  36),
    ("qwen_14b",   "Qwen-14B",      "Qwen",  48),
    ("gemma_2b",   "Gemma-2-2B",    "Gemma", 26),
    ("gemma_9b",   "Gemma-2-9B",    "Gemma", 42),
    ("llama32_1b", "LLaMA-3.2-1B",  "LLaMA", 16),
    ("llama32_3b", "LLaMA-3.2-3B",  "LLaMA", 28),
    ("llama3_8b",  "LLaMA-3-8B",    "LLaMA", 32),
    ("llama31_8b", "LLaMA-3.1-8B",  "LLaMA", 32),
    ("gemma3_12b", "Gemma-3-12B",   "Gemma", 48),
]

FAMILY_COLOR = {
    "Qwen":  "#4472C4",
    "Gemma": "#C00000",
    "LLaMA": "#70AD47",
}

SWEEP_DIR = "results/sweep_thirds"


def load_sweep_scores(alias, n_layers):
    """Load per-layer BBQ ambiguous bias scores for a model."""
    scores = []
    layers = []
    for li in range(n_layers):
        path = f"{SWEEP_DIR}/{alias}/sweep_L{li:02d}_bbq.json"
        if not os.path.exists(path):
            layers.append(li)
            scores.append(np.nan)
            continue
        d = json.load(open(path))
        bs = d["metrics"]["ambig"]["bias_score"]
        layers.append(li)
        scores.append(bs)
    return np.array(layers), np.array(scores)


# Figure 1: Per-layer ΔW injection heatmap
print("Building Fig 1: Per-layer ΔW injection heatmap...")

# Resolution: interpolate all models to 100 normalized bins
N_BINS = 100
heatmap_data = []          # shape: (n_models, N_BINS)
row_labels = []
row_families = []
model_n_layers = []

for alias, label, family, n_layers in MODELS:
    layers, scores = load_sweep_scores(alias, n_layers)
    if np.all(np.isnan(scores)):
        print(f"  [skip] {alias}: no data")
        continue
    # normalize layer index to [0, 1]
    norm_layers = layers / (n_layers - 1)
    # interpolate to N_BINS
    bins = np.linspace(0, 1, N_BINS)
    interp = np.interp(bins, norm_layers[~np.isnan(scores)],
                       scores[~np.isnan(scores)])
    heatmap_data.append(interp)
    row_labels.append(label)
    row_families.append(family)
    model_n_layers.append(n_layers)

heatmap_data = np.array(heatmap_data)   # (n_models, N_BINS)
n_models = len(row_labels)

# Global color scale: clip at 99th percentile for visual range
vmin = 0.0
vmax = np.nanpercentile(heatmap_data, 95)

fig, ax = plt.subplots(figsize=(11, 4.8))

im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r',
               vmin=vmin, vmax=vmax, interpolation='nearest',
               extent=[0, 1, n_models - 0.5, -0.5])

# Third boundaries
for x, lw, ls, lbl in [(1/3, 2.0, '--', 'First | Mid'), (2/3, 2.0, '--', 'Mid | Last')]:
    ax.axvline(x, color='white', lw=lw, ls=ls, alpha=0.9)
    ax.text(x, -0.7, lbl, ha='center', va='bottom', fontsize=7.5,
            color='white', fontweight='bold', transform=ax.get_xaxis_transform())

# Third region labels on top
ax.text(1/6, 1.01, 'First third', ha='center', va='bottom', fontsize=8.5,
        transform=ax.transAxes, color='#555555')
ax.text(0.5, 1.01, 'Mid third', ha='center', va='bottom', fontsize=8.5,
        transform=ax.transAxes, color='#555555', fontweight='bold')
ax.text(5/6, 1.01, 'Last third', ha='center', va='bottom', fontsize=8.5,
        transform=ax.transAxes, color='#555555')

# Shading for mid third
ax.axvspan(1/3, 2/3, color='white', alpha=0.07)

# Y-axis: model labels with family color bars
ax.set_yticks(range(n_models))
ax.set_yticklabels(row_labels, fontsize=9)

# Color the y-tick labels by family
for i, (label, family) in enumerate(zip(row_labels, row_families)):
    ax.get_yticklabels()[i].set_color(FAMILY_COLOR[family])
    ax.get_yticklabels()[i].set_fontweight('bold')

# X-axis
ax.set_xlabel("Normalized layer depth", fontsize=10)
ax.set_xticks([0, 1/3, 0.5, 2/3, 1.0])
ax.set_xticklabels(['0', '⅓', '½', '⅔', '1'], fontsize=9)

# Colorbar
cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
cb.set_label("BBQ ambiguous bias score\n(lower = better debiasing)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Family legend
legend_patches = [mpatches.Patch(color=FAMILY_COLOR[f], label=f)
                  for f in ['Qwen', 'Gemma', 'LLaMA']]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
          framealpha=0.85, title='Architecture', title_fontsize=8.5)

# Annotation: contrast with ATLAS
ax.annotate("Weight-space (ΔW) localization:\nmid-third dominates",
            xy=(0.5, 0.5), xytext=(0.72, 0.15),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=8.0, color='#1a1a1a',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='#aaaaaa', alpha=0.9))

ax.set_title(
    "Per-Layer ΔW Injection Score Heatmap — All 11 Models\n"
    r"(cf. ATLAS Fig. 3: attention scores concentrate in last third; "
    r"ΔW injection localizes to mid third)",
    fontsize=9.5, pad=10)

plt.tight_layout()
out1 = f"{OUT_DIR}/atlas_fig1_heatmap.pdf"
fig.savefig(out1, bbox_inches='tight')
fig.savefig(out1.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")
plt.close(fig)


# Figure 2: Block removal grouped bar chart
print("Building Fig 2: Block removal bar chart...")

BLOCK_DATA = {
    # model_label: (base, full, delta_mid, delta_first, delta_last)
    # delta_X = BBQ after removing X - full_injection
    "Qwen-3B":       (0.050, 0.000, +0.019, +0.000, +0.000),
    "LLaMA-3.2-3B":  (0.176, 0.035, +0.215, -0.011, -0.002),
    "Gemma-2-2B":    (0.160, 0.016, +0.169, -0.006, +0.000),
    "Gemma-2-9B":    (0.116, 0.000, +0.057, +0.001, -0.000),
    "LLaMA-3-8B":    (0.127, 0.068, +0.115, -0.011, -0.005),
}
# models exceeding base after mid removal (counter-balancing)
EXCEEDS_BASE = {"LLaMA-3.2-3B", "Gemma-2-2B", "LLaMA-3-8B"}

models_br = list(BLOCK_DATA.keys())
x = np.arange(len(models_br))
width = 0.24

fig, ax = plt.subplots(figsize=(9, 4.8))

color_mid   = "#C00000"   # red  — bias-critical
color_first = "#4472C4"   # blue — capability-critical (MMLU)
color_last  = "#70AD47"   # green — negligible

bars_mid   = [BLOCK_DATA[m][2] for m in models_br]
bars_first = [BLOCK_DATA[m][3] for m in models_br]
bars_last  = [BLOCK_DATA[m][4] for m in models_br]

b1 = ax.bar(x - width, bars_mid,   width, label='Remove mid-third',   color=color_mid,   alpha=0.88, edgecolor='white', lw=0.5)
b2 = ax.bar(x,         bars_first, width, label='Remove first-third',  color=color_first, alpha=0.88, edgecolor='white', lw=0.5)
b3 = ax.bar(x + width, bars_last,  width, label='Remove last-third',   color=color_last,  alpha=0.88, edgecolor='white', lw=0.5)

# Mark models where post-removal bias exceeds base (counter-balancing marker)
for i, m in enumerate(models_br):
    if m in EXCEEDS_BASE:
        ax.annotate("*", xy=(x[i] - width, bars_mid[i]),
                    xytext=(x[i] - width, bars_mid[i] + 0.005),
                    ha='center', va='bottom', fontsize=13, color=color_mid,
                    fontweight='bold')

ax.axhline(0, color='black', lw=0.8, ls='-')
ax.set_xticks(x)
ax.set_xticklabels(models_br, fontsize=9.5)
ax.set_ylabel(r"$\Delta$BBQ bias score (vs. full injection)", fontsize=10)
ax.set_xlabel("Model", fontsize=10)

ax.legend(fontsize=9.5, loc='upper right', framealpha=0.9)

# Second y-axis annotation: ATLAS comparison note
ax.text(0.02, 0.97,
        "Analog to ATLAS Fig. 4:\nATLAS compares top-k attention layers vs. random;\n"
        "we compare block removal across weight-space thirds.\n"
        "Mid-third removal is consistently most critical (* = exceeds base bias).",
        transform=ax.transAxes, va='top', ha='left', fontsize=7.8,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='#cccccc', alpha=0.92))

ax.set_title(
    "Block Removal Comparison: Which Layer-Third Is Most Critical?\n"
    r"($\Delta$BBQ > 0 = more biased after removal; mid-third removal is dominant)",
    fontsize=10, pad=8)

plt.tight_layout()
out2 = f"{OUT_DIR}/atlas_fig2_block_removal.pdf"
fig.savefig(out2, bbox_inches='tight')
fig.savefig(out2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")
plt.close(fig)

print("\nDone. Both ATLAS-inspired figures saved.")
