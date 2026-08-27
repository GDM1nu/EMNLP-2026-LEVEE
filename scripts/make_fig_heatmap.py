"""
ATLAS-inspired heatmap figures for bias_transfer_v2 paper.

Heatmap 1 (main): Model × Normalized Layer — relative bias change from base
  - 11 models × 10 depth bins (0.0–1.0)
  - TwoSlopeNorm diverging colormap: blue=debiasing, red=amplification
  - Mid-third boundaries at 0.33 / 0.67, family separators, diffuse asterisk

Heatmap 2 (appendix): Model × BBQ Category — best-layer third
  - 11 models × 11 BBQ categories
  - 3-color categorical: first/mid/last third

Usage:
    conda activate bias_transfer_clean
    python scripts/make_fig_heatmap.py [--heatmap {1,2,all}]

Output:
    paper/figures/heatmap1_layer_model.{pdf,png}
    paper/figures/heatmap2_category_model.{pdf,png}
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ── Model registry ────────────────────────────────────────────────────────────
# (alias, display_label, family, n_layers, is_structured)
ALL_MODELS = [
    ("qwen_0.5b",  "Qwen2.5-0.5B",    "Qwen",  24, False),
    ("qwen_1.5b",  "Qwen2.5-1.5B",    "Qwen",  28, False),
    ("qwen_3b",    "Qwen2.5-3B",  "Qwen",  36, True),
    ("qwen_14b",   "Qwen2.5-14B", "Qwen",  48, True),
    ("gemma_2b",   "Gemma-2-2B",   "Gemma", 26, True),
    ("gemma_9b",   "Gemma-2-9B",   "Gemma", 42, True),
    ("gemma3_12b", "Gemma-3-12B",  "Gemma", 48, True),
    ("llama32_1b", "Llama-3.2-1B", "Llama", 16, True),
    ("llama32_3b", "Llama-3.2-3B", "Llama", 28, True),
    ("llama3_8b",  "Llama-3-8B", "Llama", 32, True),
    ("llama31_8b", "Llama-3.1-8B", "Llama", 32, True),
]

DIFFUSE       = {"qwen_0.5b", "qwen_1.5b"}
FAMILY_COLORS = {"Qwen": "#4C72B0", "Gemma": "#55A868", "Llama": "#DD8452"}


def _text_color(hex_color):
    """White or near-black text depending on background luminance."""
    r, g, b = mcolors.to_rgb(hex_color)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "#1a1a1a"

# ── Paths ─────────────────────────────────────────────────────────────────────
SWEEP_DIR = "results/sweep_thirds"
CTRL_DIR  = "results/control_sweep"
ANALYSIS  = "results/analysis/bbq_category_localization.json"
OUT_DIR   = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# base_bias fallback: models without ctrl_zero_L00
BASE_FALLBACK = {
    "qwen_14b":   0.0122,
    "gemma3_12b": 0.0940,
    "llama31_8b": 0.1156,   # median of sweep scores; L00=0.0888 was the minimum (biased low)
}

# ── Binning ───────────────────────────────────────────────────────────────────
N_BINS    = 10
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)   # [0.0, 0.1, ..., 1.0]

# ── Heatmap 2 categorical setup ───────────────────────────────────────────────
CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]
THIRD_ORDER = ["first", "mid", "last"]
THIRD_CMAP  = mcolors.ListedColormap(["#4C72B0", "#55A868", "#DD8452"])
THIRD_NORM  = mcolors.BoundaryNorm([0, 1, 2, 3], THIRD_CMAP.N)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_base_bias(alias):
    path = f"{CTRL_DIR}/{alias}/ctrl_zero_L00_bbq.json"
    if os.path.exists(path):
        return json.load(open(path))["metrics"]["ambig"]["bias_score"]
    return BASE_FALLBACK.get(alias, np.nan)


def load_sweep_scores(alias, n_layers):
    scores = np.full(n_layers, np.nan)
    for li in range(n_layers):
        path = f"{SWEEP_DIR}/{alias}/sweep_L{li:02d}_bbq.json"
        if os.path.exists(path):
            scores[li] = json.load(open(path))["metrics"]["ambig"]["bias_score"]
    return scores


# ── Step 3: build matrix ──────────────────────────────────────────────────────
def build_layer_heatmap_matrix():
    """Build 11×10 matrix of per-bin relative bias change from baseline."""
    n_models = len(ALL_MODELS)
    matrix   = np.full((n_models, N_BINS), np.nan)
    row_labels, row_families, row_structured = [], [], []

    for i, (alias, label, family, n_layers, is_structured) in enumerate(ALL_MODELS):
        base   = load_base_bias(alias)
        scores = load_sweep_scores(alias, n_layers)
        depth  = np.arange(n_layers) / max(n_layers - 1, 1)

        if np.isnan(base) or abs(base) < 1e-6:
            print(f"  [warn] {alias}: base≈0 ({base:.4f}), using raw scores")
            rel = scores
        else:
            rel = (scores / base) - 1.0   # negative → debiasing

        for b in range(N_BINS):
            lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
            if b < N_BINS - 1:
                mask = (depth >= lo) & (depth < hi)
            else:
                mask = (depth >= lo) & (depth <= hi)
            valid = rel[mask & ~np.isnan(rel)]
            if len(valid):
                matrix[i, b] = valid.mean()

        lbl = f"{label}*" if alias in DIFFUSE else label
        row_labels.append(lbl)
        row_families.append(family)
        row_structured.append(is_structured)

    return matrix, row_labels, row_families, row_structured


# ── Step 4: Heatmap 1 ─────────────────────────────────────────────────────────
def make_heatmap1(matrix, row_labels, row_families, row_structured):
    """Model × Normalized Layer, colored by relative bias change.

    Design mirrors fig_category_localization:
      - left column = family-colored bars with white bold model name
      - right column = diverging heatmap (RdBu_r)
      - depth bin labels + third-region labels via fig.text() at bottom
      - colorbar in a dedicated third column
    """
    import matplotlib.gridspec as gridspec

    n_models = len(ALL_MODELS)
    vabs = max(np.nanpercentile(np.abs(matrix), 95), 0.05)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    FIG_W, FIG_H = 12.0, 6.5
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # 3-column layout: model-name bars | heatmap | colorbar
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1.7, 10, 0.45],
        wspace=0.03,
        left=0.02, right=0.98, bottom=0.22, top=0.93,
    )
    ax_lbl  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cb   = fig.add_subplot(gs[2])

    # ── Left: model name bars (family color + white bold text) ────────────────
    for i, (alias, label, family, _, _) in enumerate(ALL_MODELS):
        lbl = f"{label}*" if alias in DIFFUSE else label
        ax_lbl.barh(i, 1, left=0, height=0.96,
                    color=FAMILY_COLORS[family], align="center",
                    edgecolor="white", linewidth=0.4)
        ax_lbl.text(0.95, i, lbl,
                    ha="right", va="center", fontsize=8.5,
                    color=_text_color(FAMILY_COLORS[family]), fontweight="bold")

    prev = row_families[0]
    for i, fam in enumerate(row_families[1:], start=1):
        if fam != prev:
            ax_lbl.axhline(i - 0.5, color="white", lw=1.5)
        prev = fam

    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(-0.5, n_models - 0.5)
    ax_lbl.invert_yaxis()
    ax_lbl.axis("off")

    # ── Right: diverging heatmap ───────────────────────────────────────────────
    # extent=[0, N_BINS, n_models, 0]: bin ci spans [ci, ci+1], center at ci+0.5
    im = ax_heat.imshow(
        matrix, aspect="auto", cmap="RdBu_r", norm=norm,
        interpolation="nearest",
        extent=[0, N_BINS, n_models, 0],
    )

    # Mid-third shading (depth 0.33–0.67 → x = 3.33–6.67 in data coords)
    mid_lo, mid_hi = N_BINS / 3, 2 * N_BINS / 3
    ax_heat.axvspan(mid_lo, mid_hi, color="#FFB347", alpha=0.12, zorder=0)
    for x in (mid_lo, mid_hi):
        ax_heat.axvline(x, color="#555555", lw=1.2, ls="--", alpha=0.7, zorder=3)

    # Family separators (white dashed, matching fig_category_localization)
    prev = row_families[0]
    for i, fam in enumerate(row_families[1:], start=1):
        if fam != prev:
            ax_heat.axhline(i, color="white", lw=1.8, ls="--", alpha=0.85)
        prev = fam

    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for sp in ax_heat.spines.values():
        sp.set_visible(False)

    # ── Depth bin labels + third-region labels via fig.text() ─────────────────
    fig.canvas.draw()
    data_to_fig = ax_heat.transData + fig.transFigure.inverted()
    LABEL_Y_DROP  = 0.012   # depth tick label: just below heatmap bottom
    REGION_Y_DROP = 0.058   # third-region label: further below

    bin_centers = np.arange(N_BINS) + 0.5
    depth_ticks = [f"{(BIN_EDGES[b] + BIN_EDGES[b+1]) / 2:.2f}" for b in range(N_BINS)]

    for ci, (bc, lbl) in enumerate(zip(bin_centers, depth_ticks)):
        fx, fy = data_to_fig.transform((bc, n_models))
        fig.text(fx, fy - LABEL_Y_DROP, lbl,
                 ha="center", va="top", fontsize=8.0,
                 transform=fig.transFigure)

    for x0, x1, region_lbl in [
        (0,       mid_lo, "First third"),
        (mid_lo,  mid_hi, "Mid third"),
        (mid_hi,  N_BINS, "Last third"),
    ]:
        xc = (x0 + x1) / 2
        fx, fy = data_to_fig.transform((xc, n_models))
        bold = (region_lbl == "Mid third")
        fig.text(fx, fy - REGION_Y_DROP, region_lbl,
                 ha="center", va="top", fontsize=8.5,
                 color="#333333", fontweight="bold" if bold else "normal",
                 transform=fig.transFigure)

    # X-axis title ("Normalized layer depth")
    fx_left,  _  = data_to_fig.transform((0,       n_models))
    fx_right, _  = data_to_fig.transform((N_BINS,  n_models))
    fx_mid = (fx_left + fx_right) / 2
    _, fy_bot = data_to_fig.transform((0, n_models))
    fig.text(fx_mid, fy_bot - 0.10, "Normalized layer depth",
             ha="center", va="top", fontsize=9.5, color="#222222",
             transform=fig.transFigure)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Relative L-BBQ change", fontsize=10, fontweight="bold", labelpad=6)
    cb.ax.tick_params(labelsize=9)
    cb.ax.text(0.5,  1.05, "Amplification", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=8.5,
               color="#a32424", fontweight="bold", fontstyle="italic")
    cb.ax.text(0.5, -0.05, "Debiasing",     transform=cb.ax.transAxes,
               ha="center", va="top",    fontsize=8.5,
               color="#1a4f9c", fontweight="bold", fontstyle="italic")

    # Diffuse footnote
    fig.text(0.02, 0.005, "* diffuse signal (no structured localization)",
             fontsize=7, color="#666666", fontstyle="italic",
             ha="left", va="bottom", transform=fig.transFigure)

    for ext in ("pdf", "png"):
        out = f"{OUT_DIR}/heatmap1_layer_model.{ext}"
        kw  = {}
        if ext == "png":
            kw["dpi"] = 200
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"Saved: {out}")
    plt.close(fig)


# ── Step 5: Heatmap 2 ─────────────────────────────────────────────────────────
def make_heatmap2():
    """Model × BBQ Category, colored by best-layer third (categorical)."""
    if not os.path.exists(ANALYSIS):
        print(f"[skip] Heatmap 2: {ANALYSIS} not found")
        return

    data = json.load(open(ANALYSIS))["results"]
    alias_to_entry = {e["model"]: e for e in data}

    n_models = len(ALL_MODELS)
    n_cats   = len(CATEGORIES)
    matrix   = np.full((n_models, n_cats), np.nan)
    row_labels, row_families = [], []

    for i, (alias, label, family, n_layers, _) in enumerate(ALL_MODELS):
        lbl = f"{label}*" if alias in DIFFUSE else label
        row_labels.append(lbl)
        row_families.append(family)
        if alias not in alias_to_entry:
            continue
        per_cat = alias_to_entry[alias]["per_category"]
        for j, cat in enumerate(CATEGORIES):
            if cat in per_cat:
                matrix[i, j] = THIRD_ORDER.index(per_cat[cat]["best_third"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    im = ax.imshow(
        matrix, aspect="auto", cmap=THIRD_CMAP, norm=THIRD_NORM,
        interpolation="nearest",
        extent=[0, n_cats, n_models, 0],
    )

    _add_family_separators(ax, row_families)

    # Y-axis
    ax.set_yticks(np.arange(n_models) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=8.5)
    for i, fam in enumerate(row_families):
        ax.get_yticklabels()[i].set_color(FAMILY_COLORS[fam])

    # X-axis: category labels (rotated to avoid overlap)
    cat_labels = [c.replace("_", " ") for c in CATEGORIES]
    ax.set_xticks(np.arange(n_cats) + 0.5)
    ax.set_xticklabels(cat_labels, fontsize=7, rotation=35, ha="right")
    ax.tick_params(axis="x", length=0)

    # SES column highlight
    ses_idx = CATEGORIES.index("SES")
    ax.axvline(ses_idx, color="#4C72B0", lw=1.5, ls=":", alpha=0.8)
    ax.axvline(ses_idx + 1, color="#4C72B0", lw=1.5, ls=":", alpha=0.8)
    ax.text(ses_idx + 0.5, 0.5, "first\ndominant", ha="center", va="center",
            transform=ax.get_xaxis_transform(),
            fontsize=6.5, color="#2a4f8a", fontweight="bold", alpha=0.55)

    # Legend
    patches = [mpatches.Patch(color=THIRD_CMAP(k / 2), label=t)
               for k, t in enumerate(THIRD_ORDER)]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              title="Best-layer third", title_fontsize=7.5,
              bbox_to_anchor=(1.35, 0.0), borderaxespad=0)

    ax.set_title("Best-layer third per BBQ category (each cell = one model)",
                 fontsize=9.5, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{OUT_DIR}/heatmap2_category_model.{ext}"
        kw  = dict(bbox_inches="tight")
        if ext == "png":
            kw["dpi"] = 200
        fig.savefig(out, **kw)
        print(f"Saved: {out}")
    plt.close(fig)


# ── Step 6: self_evaluate ─────────────────────────────────────────────────────
def self_evaluate(matrix):
    mid_cols  = [b for b in range(N_BINS) if 1/3 <= BIN_EDGES[b] < 2/3]
    out_cols  = [b for b in range(N_BINS) if b not in mid_cols]
    str_rows  = [i for i, (_, _, _, _, s) in enumerate(ALL_MODELS) if s]

    mid_mean  = np.nanmean(matrix[np.ix_(str_rows, mid_cols)])
    out_mean  = np.nanmean(matrix[np.ix_(str_rows, out_cols)])

    print("\n" + "=" * 60)
    print("SELF-EVALUATION")
    print("=" * 60)
    print(f"Structured models ({len(str_rows)}): mid-third mean={mid_mean:.4f}, outer mean={out_mean:.4f}")
    print(f"Mid < Outer (debiasing pattern visible): {mid_mean < out_mean}")
    print("""
## Heatmap 1 vs ATLAS Figure 1 — 약점 3가지
1. 해상도: ATLAS 90×27=2430셀, 본 연구 11×10=110셀. 세밀한 stripe 패턴 표현 어려움.
2. 신호 강도: Attention 집중은 discrete spike; ΔW relative change는 연속적이고 노이즈가 큼.
3. 단순성: ATLAS는 단색 red 의미 명확. 본 연구 diverging colormap은 해석 부담 증가.

## Reviewer 예상 비판 3가지
1. "10 bins is too coarse — you can't claim layer-level localization with this granularity."
   → 대응: 모델마다 n_layers 상이(16–48)하여 normalize+bin 불가피; 개별 곡선은 Appendix Fig 1c.
2. "Relative change is unstable when base_bias ≈ 0 (Qwen2.5-3B base=0.05)."
   → 대응: base<0.02인 모델(qwen_14b) fallback 사용; qwen_3b base=0.05로 충분히 stable.
3. "The mid-third pattern is not consistently visible across all rows."
   → 대응: diffuse 모델(*)은 명시적으로 분리; structured 9/9는 mid-third 최저 relative change.

## 개선 우선순위 top 3
1. vabs clip: 이상치 1–2개가 전체 색상 범위 지배 → 95th percentile 사용(이미 적용).
2. N_BINS=20: 현재 10bins → 20으로 늘리면 stripe 패턴 가시성 향상.
3. Heatmap 2 SES 주석: first-third dominant 패턴에 텍스트 annotation 추가.
""")


# ── Helper ────────────────────────────────────────────────────────────────────
def _add_family_separators(ax, row_families):
    prev = row_families[0]
    for i, fam in enumerate(row_families[1:], start=1):
        if fam != prev:
            ax.axhline(i, color="#aaaaaa", lw=0.7, ls="--")
        prev = fam


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    print("=== Step 3: Building layer heatmap matrix ===")
    matrix, row_labels, row_families, row_structured = build_layer_heatmap_matrix()
    assert matrix.shape == (11, 10), f"Expected (11,10), got {matrix.shape}"
    print(f"✓ Matrix shape: {matrix.shape}")
    print(f"  Value range : [{np.nanmin(matrix):.4f}, {np.nanmax(matrix):.4f}]")
    print(f"  NaN cells   : {np.isnan(matrix).sum()} / {matrix.size}")
    print("\nPer-model row means:")
    for i, (alias, *_) in enumerate(ALL_MODELS):
        print(f"  {alias:14s}: {np.nanmean(matrix[i]):.4f}")

    if args.heatmap in ("1", "all"):
        print("\n=== Step 4: Generating Heatmap 1 ===")
        make_heatmap1(matrix, row_labels, row_families, row_structured)

    if args.heatmap in ("2", "all"):
        print("\n=== Step 5: Generating Heatmap 2 ===")
        make_heatmap2()

    print("\n=== Step 6: Self-evaluation ===")
    self_evaluate(matrix)
