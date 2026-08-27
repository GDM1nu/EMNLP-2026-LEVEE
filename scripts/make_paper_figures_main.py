"""
Main paper figures (3종) — run from repo root with bias_transfer_clean env.

Fig 1: BBQ Category Localization Heatmap
Fig 2: BBQ–MMLU Double Dissociation Scatter (block removal + block only)
Fig 3: Per-layer Sign-flip Symmetry Grid

Usage:
    conda activate bias_transfer_clean
    python scripts/make_paper_figures_main.py

Output: results/paper_figures/fig_{category_localization,bbq_mmlu_dissociation,perlayer_signflip_symmetry}.{png,pdf}
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings; warnings.filterwarnings('ignore')

os.makedirs('results/paper_figures', exist_ok=True)

MODEL_ORDER = ['qwen_0.5b','qwen_1.5b','qwen_3b','qwen_14b',
               'gemma_2b','gemma_9b','gemma3_12b',
               'llama32_1b','llama32_3b','llama3_8b','llama31_8b']
MODEL_LABEL = {
    'qwen_0.5b':'Qwen2.5-0.5B','qwen_1.5b':'Qwen2.5-1.5B','qwen_3b':'Qwen2.5-3B','qwen_14b':'Qwen2.5-14B',
    'gemma_2b':'Gemma-2-2B','gemma_9b':'Gemma-2-9B','gemma3_12b':'Gemma-3-12B',
    'llama32_1b':'Llama-3.2-1B','llama32_3b':'Llama-3.2-3B',
    'llama3_8b':'Llama-3-8B','llama31_8b':'Llama-3.1-8B',
}
# Muted, colorblind-friendly palette (CB-safe: blue / vermillion / teal-green)
THIRD_COLOR = {'first': '#5B8DB8', 'mid': '#C9503F', 'last': '#5E9E68'}
FAMILY_MAP = {
    'qwen_0.5b':'Qwen','qwen_1.5b':'Qwen','qwen_3b':'Qwen','qwen_14b':'Qwen',
    'gemma_2b':'Gemma','gemma_9b':'Gemma','gemma3_12b':'Gemma',
    'llama32_1b':'Llama','llama32_3b':'Llama','llama3_8b':'Llama','llama31_8b':'Llama',
}
FAMILY_COLORS = {'Qwen': '#4C72B0', 'Gemma': '#55A868', 'Llama': '#DD8452'}

# ─────────────────────────────────────────────────────────────────
# FIG 1: BBQ Category Localization Heatmap
# ─────────────────────────────────────────────────────────────────
d_cat = json.load(open('results/analysis/bbq_category_localization.json'))
CAT_ORDER = ['Age','Disability_status','Physical_appearance','Gender_identity',
             'Race_ethnicity','Race_x_gender','Race_x_SES','Nationality',
             'Sexual_orientation','Religion','SES']
CAT_LABEL = {'Age':'Age','Disability_status':'Disability\nStatus',
             'Physical_appearance':'Physical\nAppearance',
             'Gender_identity':'Gender\nIdentity',
             'Race_ethnicity':'Race/\nEthnicity',
             'Race_x_gender':'Race×\nGender','Race_x_SES':'Race×\nSES',
             'Nationality':'Nationality','Sexual_orientation':'Sexual\nOrientation',
             'Religion':'Religion','SES':'SES'}
THIRD_NUM = {'first':0,'mid':1,'last':2}

present = {v['model']: v for v in d_cat['results']}
models_plot = [m for m in MODEL_ORDER if m in present]

mat = np.full((len(models_plot), len(CAT_ORDER)), np.nan)
for mi, m in enumerate(models_plot):
    for ci, cat in enumerate(CAT_ORDER):
        t = present[m]['per_category'].get(cat, {}).get('best_third')
        if t: mat[mi, ci] = THIRD_NUM[t]

import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox

n_models = len(models_plot)
n_cats = len(CAT_ORDER)
families = [FAMILY_MAP[m] for m in models_plot]

# ── Layout ───────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 13.0, 8.0
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 10], wspace=0.02,
                       left=0.02, right=0.99, bottom=0.20, top=0.94)
ax_lbl  = fig.add_subplot(gs[0])
ax_heat = fig.add_subplot(gs[1])

# ── Model name column ────────────────────────────────────────────────────────
for i, m in enumerate(models_plot):
    fam = FAMILY_MAP[m]
    ax_lbl.barh(i, 1, left=0, height=0.96,
                color=FAMILY_COLORS[fam], align='center',
                edgecolor='white', linewidth=0.4)
    ax_lbl.text(0.5, i, MODEL_LABEL[m],
                ha='center', va='center', fontsize=8.8,
                color='white', fontweight='bold')

prev = families[0]
for i, fam in enumerate(families[1:], start=1):
    if fam != prev:
        ax_lbl.axhline(i - 0.5, color='white', lw=1.5)
    prev = fam

ax_lbl.set_xlim(0, 1)
ax_lbl.set_ylim(-0.5, n_models - 0.5)
ax_lbl.invert_yaxis()
ax_lbl.axis('off')

# ── Heatmap ──────────────────────────────────────────────────────────────────
cmap = ListedColormap([THIRD_COLOR['first'], THIRD_COLOR['mid'], THIRD_COLOR['last']])
ax_heat.imshow(mat, aspect='auto', cmap=cmap, vmin=-0.5, vmax=2.5,
               interpolation='nearest')

prev = families[0]
for i, fam in enumerate(families[1:], start=1):
    if fam != prev:
        ax_heat.axhline(i - 0.5, color='white', lw=1.8, ls='--', alpha=0.85)
    prev = fam

ax_heat.set_xticks([])
ax_heat.set_yticks([])
for sp in ax_heat.spines.values():
    sp.set_visible(False)

# Cell text
for i in range(n_models):
    for j in range(n_cats):
        if not np.isnan(mat[i, j]):
            ax_heat.text(j, i, ['F', 'M', 'L'][int(mat[i, j])],
                         ha='center', va='center', fontsize=7.5,
                         color='white', fontweight='normal')

# SES separator
ax_heat.axvline(9.5, color='white', lw=2.0, ls='--', alpha=0.8)
ax_heat.text(10.0, n_models - 0.2, 'SES\n(exception)', fontsize=7.0,
             color='white', ha='center', va='top', style='italic')

# ── Category labels via fig.text() ───────────────────────────────────────────
fig.canvas.draw()
data_to_fig = ax_heat.transData + fig.transFigure.inverted()
LABEL_DROP = 0.012

for ci, cat in enumerate(CAT_ORDER):
    fx, fy = data_to_fig.transform((ci, n_models - 0.5))
    fig.text(fx, fy - LABEL_DROP,
             CAT_LABEL[cat],
             ha='center', va='top',
             rotation=0, fontsize=8.5,
             transform=fig.transFigure)

# ── Legend ────────────────────────────────────────────────────────────────────
leg_patches = [mpatches.Patch(color=THIRD_COLOR[t], label=f'{t.capitalize()} third')
               for t in ['first', 'mid', 'last']]
fig.legend(handles=leg_patches, loc='lower center', fontsize=9.5,
           framealpha=0.92, edgecolor='#cccccc',
           ncol=3, handlelength=1.5, columnspacing=1.8,
           bbox_to_anchor=(0.6, 0.02))

# ── Save with fixed figure size (no bbox crop) ────────────────────────────────
# Fixed FIG_W×FIG_H guarantees every element is within bounds.
# pad_inches is irrelevant here since we're not using bbox_inches='tight'.
for path in ('results/paper_figures/fig_category_localization.png',
             'results/paper_figures/fig_category_localization.pdf',
             'paper/figures/fig_category_localization.pdf'):
    dpi = 150 if path.endswith('.png') else None
    kw = dict(dpi=dpi) if dpi else {}
    fig.savefig(path, **kw)

plt.close()
print("✅ Fig 1 saved")

# ─────────────────────────────────────────────────────────────────
# FIG 2: BBQ-MMLU Dissociation Scatter
# ─────────────────────────────────────────────────────────────────
models5 = ['qwen_3b','llama32_3b','gemma_2b','gemma_9b','llama3_8b']
labels5  = {'qwen_3b':'Qwen2.5-3B','llama32_3b':'Llama-3.2-3B',
            'gemma_2b':'Gemma-2-2B','gemma_9b':'Gemma-2-9B','llama3_8b':'Llama-3-8B†'}
MCOLOR = {'qwen_3b':'#4472C4','llama32_3b':'#ED7D31',
          'gemma_2b':'#A5A5A5','gemma_9b':'#FFC000','llama3_8b':'#9B59B6'}
MARKER_T = {'first':'s','mid':'o','last':'^'}

# Table 4a: block removal
rem = {}
for m in models5:
    full_bbq = json.load(open(f'results/block_removal/{m}/block_full_bbq.json'))['metrics']['ambig']['bias_score']
    full_mmlu = json.load(open(f'results/block_removal/{m}/block_full_mmlu.json'))['mmlu_acc']
    for cond in ['mid','first','last']:
        cbbq = json.load(open(f'results/block_removal/{m}/block_minus_{cond}_bbq.json'))['metrics']['ambig']['bias_score']
        cmmlu = json.load(open(f'results/block_removal/{m}/block_minus_{cond}_mmlu.json'))['mmlu_acc']
        rem[(m,cond)] = (cbbq - full_bbq, cmmlu - full_mmlu)

# Table 4b: block only
only = {}
for m in models5:
    base_mmlu = json.load(open(f'results/block_removal/{m}/block_only_baseline_mmlu.json'))['mmlu_acc']
    for cond in ['first','mid','last']:
        bbq = json.load(open(f'results/block_removal/{m}/block_only_{cond}_bbq.json'))['metrics']['ambig']['bias_score']
        dmmlu = json.load(open(f'results/block_removal/{m}/block_only_only_{cond}_mmlu.json'))['mmlu_acc'] - base_mmlu
        only[(m,cond)] = (bbq, dmmlu)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

for ax, dataset, xlabel, ylabel, title, annotate_cond in [
    (axes[0], rem,
     'ΔL-BBQ score (removed − full)',
     'ΔMMLU accuracy (removed − full)',
     '(a) Block Removal (necessity)', 'mid'),
    (axes[1], only,
     'L-BBQ score (only that third)',
     'ΔMMLU from uninstructed base model',
     '(b) Block-Only Injection (sufficiency)', 'first'),
]:
    ax.axhline(0, color='#888', lw=0.8, ls='--', alpha=0.5)
    if dataset is rem: ax.axvline(0, color='#888', lw=0.8, ls='--', alpha=0.5)

    for (m, cond), (dx, dy) in dataset.items():
        is_anom = (m == 'llama3_8b')
        ax.scatter(dx, dy, c=MCOLOR[m], marker=MARKER_T[cond],
                   s=130, alpha=0.35 if is_anom else 0.92,
                   zorder=1 if is_anom else 4,
                   edgecolors='white', linewidths=0.8)
        if cond == annotate_cond and not is_anom:
            if abs(dx) > 0.02 or abs(dy) > 0.03:
                ax.annotate(labels5[m], (dx, dy),
                            textcoords='offset points', xytext=(6, 4),
                            fontsize=7.5, color=MCOLOR[m], fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)  # sub-panel (a)/(b) label kept
    ax.tick_params(labelsize=8.5)
    ax.spines[['top','right']].set_visible(False)

# quadrant annotations on left panel
axes[0].text(0.97, 0.97, 'bias↑  cap≈\n(mid)', transform=axes[0].transAxes,
             ha='right', va='top', fontsize=8, color='#C00000',
             bbox=dict(boxstyle='round,pad=0.2', fc='#FFF0F0', alpha=0.7))
axes[0].text(0.55, 0.03, 'bias≈  cap↓\n(first)', transform=axes[0].transAxes,
             ha='left', va='bottom', fontsize=8, color='#4472C4',
             bbox=dict(boxstyle='round,pad=0.2', fc='#F0F0FF', alpha=0.7))

model_h = [mpatches.Patch(color=MCOLOR[m], label=labels5[m]) for m in models5]
marker_h = [plt.scatter([],[],c='#555',marker=MARKER_T[t],s=80,label=f'{t.capitalize()}-third')
            for t in ['first','mid','last']]
plt.tight_layout(rect=[0, 0.11, 1, 1])
fig.legend(handles=model_h+marker_h, loc='lower center', ncol=8,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.02), framealpha=0.9)
plt.savefig('results/paper_figures/fig_bbq_mmlu_dissociation.png', dpi=150, bbox_inches='tight')
plt.savefig('results/paper_figures/fig_bbq_mmlu_dissociation.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 2 saved")

# ─────────────────────────────────────────────────────────────────
# FIG 2b: BBQ-MMLU Dissociation Curve (line graph)
# ─────────────────────────────────────────────────────────────────
# Load absolute values per model per condition
rem_abs = {}
for m in models5:
    full_bbq = json.load(open(f'results/block_removal/{m}/block_full_bbq.json'))['metrics']['ambig']['bias_score']
    full_mmlu = json.load(open(f'results/block_removal/{m}/block_full_mmlu.json'))['mmlu_acc']
    rem_abs[m] = {'Full': (full_bbq, full_mmlu)}
    for cond in ['first', 'mid', 'last']:
        cbbq  = json.load(open(f'results/block_removal/{m}/block_minus_{cond}_bbq.json'))['metrics']['ambig']['bias_score']
        cmmlu = json.load(open(f'results/block_removal/{m}/block_minus_{cond}_mmlu.json'))['mmlu_acc']
        rem_abs[m][f'-{cond.capitalize()}'] = (cbbq, cmmlu)

only_abs = {}
for m in models5:
    base_mmlu = json.load(open(f'results/block_removal/{m}/block_only_baseline_mmlu.json'))['mmlu_acc']
    only_abs[m] = {}
    for cond in ['first', 'mid', 'last']:
        bbq   = json.load(open(f'results/block_removal/{m}/block_only_{cond}_bbq.json'))['metrics']['ambig']['bias_score']
        dmmlu = json.load(open(f'results/block_removal/{m}/block_only_only_{cond}_mmlu.json'))['mmlu_acc'] - base_mmlu
        only_abs[m][f'{cond.capitalize()}-only'] = (bbq, dmmlu)

LSMAP = {m: ls for m, ls in zip(models5, ['-','--','-.',':',(0,(3,1,1,1))])}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

# Panel (a): Block Removal — absolute BBQ and MMLU across conditions
conds_rem = ['Full', '-First', '-Mid', '-Last']
x_rem = np.arange(len(conds_rem))
ax_a  = axes[0]
ax_a2 = ax_a.twinx()

for m in models5:
    bbq_vals  = [rem_abs[m][c][0] for c in conds_rem]
    mmlu_vals = [rem_abs[m][c][1] for c in conds_rem]
    ax_a.plot(x_rem, bbq_vals,  ls=LSMAP[m], color='#C00000', lw=2,   alpha=0.82, label=labels5[m])
    ax_a2.plot(x_rem, mmlu_vals, ls=LSMAP[m], color='#2060C0', lw=1.5, alpha=0.65)

# shade -Mid (BBQ spike) and -First (MMLU drop)
ax_a.axvspan(1.5, 2.5, alpha=0.08, color='#C00000')   # -Mid
ax_a.axvspan(0.5, 1.5, alpha=0.08, color='#2060C0')   # -First
ax_a.set_xticks(x_rem)
ax_a.set_xticklabels(conds_rem, fontsize=9.5)
ax_a.set_ylabel('L-BBQ score', color='#C00000', fontsize=10)
ax_a2.set_ylabel('MMLU accuracy', color='#2060C0', fontsize=10)
ax_a.tick_params(axis='y', labelcolor='#C00000')
ax_a2.tick_params(axis='y', labelcolor='#2060C0')
ax_a.set_title('(a) Block Removal', fontsize=11, fontweight='bold')
ax_a.set_xlim(-0.3, len(conds_rem)-0.7)

# Panel (b): Block-Only Injection — BBQ (left) and ΔMMLU (right)
conds_only = ['First-only', 'Mid-only', 'Last-only']
x_only = np.arange(len(conds_only))
ax_b  = axes[1]
ax_b2 = ax_b.twinx()

for m in models5:
    bbq_vals  = [only_abs[m][c][0] for c in conds_only]
    mmlu_vals = [only_abs[m][c][1] for c in conds_only]
    ax_b.plot(x_only, bbq_vals,  ls=LSMAP[m], color='#C00000', lw=2,   alpha=0.82)
    ax_b2.plot(x_only, mmlu_vals, ls=LSMAP[m], color='#2060C0', lw=1.5, alpha=0.65)

ax_b.axvspan(0.5, 1.5, alpha=0.08, color='#C00000')   # Mid-only (lowest BBQ)
ax_b.axvspan(-0.5, 0.5, alpha=0.08, color='#2060C0')  # First-only (highest MMLU)
ax_b.set_xticks(x_only)
ax_b.set_xticklabels(conds_only, fontsize=9.5)
ax_b.set_ylabel('L-BBQ score', color='#C00000', fontsize=10)
ax_b2.set_ylabel('ΔMMLU from base', color='#2060C0', fontsize=10)
ax_b.tick_params(axis='y', labelcolor='#C00000')
ax_b2.tick_params(axis='y', labelcolor='#2060C0')
ax_b.axhline(0, color='#2060C0', lw=0.7, ls=':', alpha=0.5)
ax_b.set_title('(b) Block-Only Injection', fontsize=11, fontweight='bold')
ax_b.set_xlim(-0.3, len(conds_only)-0.7)

# shared legend: model lines + metric color legend
model_h = [plt.Line2D([0],[0], color='gray', ls=LSMAP[m], lw=2, label=labels5[m]) for m in models5]
metric_h = [
    plt.Line2D([0],[0], color='#C00000', lw=2, label='BBQ bias (left axis)'),
    plt.Line2D([0],[0], color='#2060C0', lw=1.5, label='MMLU (right axis)'),
]
plt.subplots_adjust(bottom=0.20, top=0.90, left=0.08, right=0.93, wspace=0.35)
fig.legend(handles=model_h+metric_h, loc='lower center', ncol=7,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.02), framealpha=0.9)

plt.savefig('results/paper_figures/fig_bbq_mmlu_dissociation_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('results/paper_figures/fig_bbq_mmlu_dissociation_curve.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 2b (curve) saved")

# ─────────────────────────────────────────────────────────────────
# FIG 7: BBQ-MMLU Dissociation Arrow Plot (redesigned)
# ─────────────────────────────────────────────────────────────────
MCOLOR7 = {
    'qwen_3b':    '#4477AA',
    'llama32_3b': '#EE6677',
    'gemma_2b':   '#999999',
    'gemma_9b':   '#CCBB44',
    'llama3_8b':  '#AA3377',
}
labels7 = {
    'qwen_3b':    'Qwen2.5-3B',
    'llama32_3b': 'Llama-3.2-3B',
    'gemma_2b':   'Gemma-2-2B',
    'gemma_9b':   'Gemma-2-9B',
    'llama3_8b':  'Llama-3-8B†',
}
HERO_ARROWS = {('llama32_3b', 'mid'), ('gemma_9b', 'first')}

# Wider figure: extra right margin for external legends
fig, ax = plt.subplots(figsize=(5.8, 4.2))

# crosshair
ax.axhline(0, color='#bbb', lw=0.8, ls='--', zorder=0)
ax.axvline(0, color='#bbb', lw=0.8, ls='--', zorder=0)
ax.scatter([0], [0], color='black', s=18, zorder=6, clip_on=False)

# draw arrows: last → first → mid (mid rendered on top)
for m in models5:
    color = MCOLOR7[m]
    is_qwen = m.startswith('qwen')
    for cond in ['last', 'first', 'mid']:
        dx, dy = rem[(m, cond)]
        is_hero = (m, cond) in HERO_ARROWS
        is_last = cond == 'last'
        # Thickness hierarchy: hero > mid > first >> last
        lw = (2.8 if is_hero
              else 2.0 if cond == 'mid'
              else 1.4 if cond == 'first'
              else 0.7)
        # Transparency: Qwen slightly faded to reduce overlap; last very faint
        alpha = (0.25 if is_last
                 else 0.62 if (is_qwen and not is_hero)
                 else 1.0  if is_hero
                 else 0.85)
        style  = '->' if cond in ('mid', 'last') else '-|>'
        zorder = 5 if is_hero else (3 if not is_last else 1)
        ax.annotate(
            '', xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle=style, color=color, lw=lw, alpha=alpha,
                mutation_scale=12 if is_hero else (8 if not is_last else 5),
            ),
            zorder=zorder,
        )

# ── Hero callout: Llama-3.2-3B −mid ─────────────────────────────
# arrow tip at ≈ (0.215, −0.048); callout box just above tip
dx_lm, dy_lm = rem[('llama32_3b', 'mid')]
ax.annotate(
    f'−mid\nΔBBQ {dx_lm:+.3f}\nΔMMLU {dy_lm:+.3f}',
    xy=(dx_lm, dy_lm),
    xytext=(0.170, 0.040),
    fontsize=6.5, fontweight='bold', color=MCOLOR7['llama32_3b'],
    ha='center', va='bottom', zorder=9,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
              edgecolor=MCOLOR7['llama32_3b'], lw=1.5, alpha=0.97),
    arrowprops=dict(arrowstyle='-', color=MCOLOR7['llama32_3b'],
                    lw=0.9, alpha=0.75),
)

# ── Hero callout: Gemma-2-9B −first ──────────────────────────────
# arrow tip at ≈ (0.001, −0.350); callout box close to tip, right side
dx_gf, dy_gf = rem[('gemma_9b', 'first')]
ax.annotate(
    f'−first\nΔMMLU {dy_gf:+.3f}\nΔBBQ ≈ 0',
    xy=(dx_gf, dy_gf),
    xytext=(0.040, -0.300),
    fontsize=6.5, fontweight='bold', color=MCOLOR7['gemma_9b'],
    ha='left', va='center', zorder=9,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
              edgecolor=MCOLOR7['gemma_9b'], lw=1.5, alpha=0.97),
    arrowprops=dict(arrowstyle='-', color=MCOLOR7['gemma_9b'],
                    lw=0.9, alpha=0.75),
)

# Quadrant labels: small, muted, italic
for xp, yp, ha, va, txt in [
    (0.98, 0.98, 'right', 'top',    'bias↑ cap↑'),
    (0.02, 0.98, 'left',  'top',    'bias↓ cap↑'),
    (0.02, 0.02, 'left',  'bottom', 'bias↓ cap↓'),
    (0.98, 0.02, 'right', 'bottom', 'bias↑ cap↓'),
]:
    ax.text(xp, yp, txt, transform=ax.transAxes, fontsize=5.8,
            color='#c0c0c0', style='italic', ha=ha, va=va)

ax.set_xlim(-0.02, 0.25)
ax.set_ylim(-0.41, 0.07)
ax.set_xlabel('ΔL-BBQ (relative to full vector)', fontsize=8.5, fontweight='bold')
ax.set_ylabel('ΔMMLU (relative to full vector)', fontsize=8.5, fontweight='bold')
ax.tick_params(labelsize=7)
ax.spines['top'].set_color('#ccc')
ax.spines['right'].set_color('#ccc')

# ── Legend 1: Models — outside plot, anchored to upper-right ─────
leg1_h = [
    plt.Line2D([0], [0], color=MCOLOR7[m], lw=2.2, marker='o', markersize=5,
               label=labels7[m])
    for m in models5
]
leg1 = ax.legend(
    handles=leg1_h,
    title='Model', title_fontsize=7,
    fontsize=6.5, framealpha=0.95, handlelength=1.8,
    loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
)
ax.add_artist(leg1)

# ── Legend 2: Conditions — outside plot, below model legend ──────
leg2_h = [
    plt.Line2D([0], [0], color='#444', lw=2.2,              label='−mid  (bias-critical)'),
    plt.Line2D([0], [0], color='#444', lw=1.4, ls='--',     label='−first (cap-critical)'),
    plt.Line2D([0], [0], color='#bbb', lw=0.7, alpha=0.55,  label='−last  (negligible)'),
]
ax.legend(
    handles=leg2_h,
    title='Condition', title_fontsize=7,
    fontsize=6.5, framealpha=0.95, handlelength=1.8,
    loc='upper left', bbox_to_anchor=(1.02, 0.42), borderaxespad=0,
)

plt.tight_layout()
plt.savefig('results/paper_figures/fig7_arrow_plot.pdf', bbox_inches='tight')
plt.savefig('results/paper_figures/fig7_arrow_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/fig7_arrow_plot.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 7 (arrow plot) redesigned + saved")

# ─────────────────────────────────────────────────────────────────
# FIG 3: Per-layer Signflip Symmetry
# ─────────────────────────────────────────────────────────────────
d_sym = json.load(open('results/analysis/perlayer_signflip_symmetry.json'))
records = d_sym['results']
n = len(records)
ncols = 4; nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.4),
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
axes = axes.flatten()

# Shared legend at top center
leg_patches = [mpatches.Patch(color=THIRD_COLOR[t], alpha=0.85,
                               label=f'{t.capitalize()} third')
               for t in ['first', 'mid', 'last']]
fig.legend(handles=leg_patches, loc='upper center', fontsize=11,
           bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True,
           framealpha=0.92, edgecolor='#cccccc',
           handlelength=1.5, handletextpad=0.6, columnspacing=1.8)

for idx, v in enumerate(records):
    ax = axes[idx]
    lm = v['layer_metrics']
    L  = v['n_layers']
    xs = [x['layer_idx'] for x in lm]
    ys = [x['sym_score']  for x in lm]
    ts = [x['third']      for x in lm]
    t1, t2 = L // 3, 2 * L // 3

    # Vertical boundary lines only — no background shading
    for xb in (t1 - 0.5, t2 - 0.5):
        ax.axvline(xb, color='#888888', lw=0.9, ls='--', alpha=0.65, zorder=0)

    # Per-model mean reference line
    ax.axhline(float(np.mean(ys)), color='#bbbbbb', lw=0.9, ls=':', zorder=0)

    # Line segments colored by third
    for i in range(len(xs) - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], c=THIRD_COLOR[ts[i]], lw=2.2, alpha=0.9)
    # Scatter markers
    for x, y, t in zip(xs, ys, ts):
        ax.scatter(x, y, c=THIRD_COLOR[t], s=35, zorder=4, alpha=0.95, edgecolors='none')

    most = v['most_symmetric_third']
    tc   = THIRD_COLOR[most]
    ax.set_title(MODEL_LABEL.get(v['model'], v['model']),
                 fontsize=11, color=tc, fontweight='bold', pad=5)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Layer', fontsize=10, labelpad=4)
    ax.set_ylabel('Sym score', fontsize=10, labelpad=4)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune='both'))
    ax.spines[['top', 'right']].set_visible(False)

for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.savefig('results/paper_figures/fig_perlayer_signflip_symmetry.png', dpi=150, bbox_inches='tight')
plt.savefig('results/paper_figures/fig_perlayer_signflip_symmetry.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig_perlayer_signflip_symmetry.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 3 saved")
print("\nAll figures → results/paper_figures/")
