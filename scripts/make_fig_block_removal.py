"""
Block Removal figure — necessity + BBQ-MMLU dissociation in one plot.

Two-panel figure:
  Top:    BBQ ambig bias_score per model × condition (necessity evidence)
  Bottom: MMLU accuracy per model × condition (dissociation evidence)

Usage (from repo root):
    conda activate bias_transfer_clean
    python scripts/make_fig_block_removal.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('results/paper_figures', exist_ok=True)

# ── data ─────────────────────────────────────────────────────────────
MODELS   = ['qwen_3b', 'llama32_3b', 'gemma_2b', 'gemma_9b', 'llama3_8b']
LABELS   = {
    'qwen_3b':    'Qwen2.5-3B',
    'llama32_3b': 'Llama-3.2-3B',
    'gemma_2b':   'Gemma-2-2B',
    'gemma_9b':   'Gemma-2-9B',
    'llama3_8b':  'Llama-3-8B†',
}
CONDS    = ['full', 'minus_first', 'minus_mid', 'minus_last']
CLABEL   = {'full': 'Full', 'minus_first': '−First', 'minus_mid': '−Mid', 'minus_last': '−Last'}
COLOR    = {'full': '#AAAAAA', 'minus_first': '#4472C4', 'minus_mid': '#C00000', 'minus_last': '#70AD47'}
BASE     = 'results/block_removal'

bbq, mmlu = {}, {}
for alias in MODELS:
    d = os.path.join(BASE, alias)
    bbq[alias]  = {c: json.load(open(f'{d}/block_{c}_bbq.json'))['metrics']['ambig']['bias_score']
                   for c in CONDS}
    mmlu[alias] = {c: json.load(open(f'{d}/block_{c}_mmlu.json'))['mmlu_acc']
                   for c in CONDS}

# ── layout ───────────────────────────────────────────────────────────
n_models  = len(MODELS)
n_conds   = len(CONDS)
bar_w     = 0.18
x         = np.arange(n_models)
offsets   = np.linspace(-(n_conds-1)/2, (n_conds-1)/2, n_conds) * bar_w

fig, (ax_bbq, ax_mmlu) = plt.subplots(2, 1, figsize=(8.5, 5.0),
                                       gridspec_kw={'hspace': 0.48})

# ── top: BBQ bias_score ──────────────────────────────────────────────
for ci, cond in enumerate(CONDS):
    vals = [bbq[m][cond] for m in MODELS]
    bars = ax_bbq.bar(x + offsets[ci], vals, bar_w,
                      color=COLOR[cond], label=CLABEL[cond],
                      edgecolor='white', linewidth=0.5, zorder=3)

ax_bbq.axhline(0, color='black', lw=0.7, zorder=2)
ax_bbq.set_xticks(x)
ax_bbq.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
ax_bbq.set_ylabel('L-BBQ Score', fontsize=9, fontweight='bold')
ax_bbq.set_title('Block Removal: Necessity (BBQ)', fontsize=10, fontweight='bold', pad=4)
ax_bbq.grid(axis='y', lw=0.4, alpha=0.5, zorder=0)
ax_bbq.spines[['top','right']].set_visible(False)
ax_bbq.tick_params(axis='both', labelsize=8.5)

# annotate Llama-3-8B† footnote
ax_bbq.text(x[-1], bbq['llama3_8b']['minus_mid'] + 0.005,
            '†', ha='center', va='bottom', fontsize=9, color='#C00000')

# ── bottom: MMLU accuracy ────────────────────────────────────────────
for ci, cond in enumerate(CONDS):
    vals = [mmlu[m][cond] for m in MODELS]
    ax_mmlu.bar(x + offsets[ci], vals, bar_w,
                color=COLOR[cond], label=CLABEL[cond],
                edgecolor='white', linewidth=0.5, zorder=3)

# highlight Gemma-2-9B −first as hero
hero_val = mmlu['gemma_9b']['minus_first']
hero_x   = x[MODELS.index('gemma_9b')] + offsets[CONDS.index('minus_first')]
ax_mmlu.annotate(f'−35.0%p\n(MMLU)',
                 xy=(hero_x, hero_val),
                 xytext=(hero_x - 0.55, hero_val - 0.08),
                 fontsize=7.5, color='#4472C4', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#4472C4', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#4472C4', lw=0.8))

ax_mmlu.set_xticks(x)
ax_mmlu.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
ax_mmlu.set_ylabel('MMLU Accuracy', fontsize=9, fontweight='bold')
ax_mmlu.set_title('Block Removal: Capability (MMLU)', fontsize=10, fontweight='bold', pad=4)
ax_mmlu.grid(axis='y', lw=0.4, alpha=0.5, zorder=0)
ax_mmlu.spines[['top','right']].set_visible(False)
ax_mmlu.tick_params(axis='both', labelsize=8.5)
ax_mmlu.set_ylim(0.15, 0.75)

# ── shared legend ────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLOR[c], label=CLABEL[c]) for c in CONDS]
plt.subplots_adjust(bottom=0.14, top=0.95, left=0.10, right=0.97)
fig.legend(handles=patches, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

fig.text(0.5, -0.06,
         '† Llama-3-8B full injection degrades MMLU (known anomaly); excluded from dissociation analysis.',
         ha='center', fontsize=7.5, color='#555555', style='italic')

plt.savefig('results/paper_figures/fig_block_removal.png', dpi=150, bbox_inches='tight')
plt.savefig('results/paper_figures/fig_block_removal.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig_block_removal.pdf', bbox_inches='tight')
plt.close()
print("✅ fig_block_removal saved → results/paper_figures/ + paper/figures/")
