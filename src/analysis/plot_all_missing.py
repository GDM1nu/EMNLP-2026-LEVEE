#!/usr/bin/env python3
"""Generate plots for all analysis results."""
from __future__ import annotations
from pathlib import Path
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


PROJ_ROOT = Path(__file__).resolve().parents[2]

OUT = str(PROJ_ROOT / "results/analysis/plots")
os.makedirs(OUT, exist_ok=True)

BASE = str(PROJ_ROOT)
ANALYSIS = f"{BASE}/results/analysis"

PALETTE = {
    "first": "#4472C4", "mid": "#C00000", "last": "#70AD47",
    "full":  "#2E75B6",
    "minus_first": "#4472C4", "minus_mid": "#C00000",
    "minus_last":  "#70AD47", "minus_best": "#FF7F0F",
    "only_first":  "#4472C4", "only_mid":  "#C00000", "only_last": "#70AD47",
}
MODEL_ORDER = ["qwen_0.5b","qwen_1.5b","qwen_3b","qwen_14b",
               "gemma_2b","gemma_9b","llama32_1b","llama32_3b","llama3_8b"]
MODEL_LABELS = {
    "qwen_0.5b":"Qwen 0.5B","qwen_1.5b":"Qwen 1.5B","qwen_3b":"Qwen 3B","qwen_14b":"Qwen 14B",
    "gemma_2b":"Gemma 2B","gemma_9b":"Gemma 9B",
    "llama32_1b":"LLaMA3.2 1B","llama32_3b":"LLaMA3.2 3B","llama3_8b":"LLaMA3 8B",
}


# 1. Block Removal BBQ — 5 models × 5 conditions grouped bar

def plot_block_removal_bbq():
    models  = ["qwen_3b","llama32_3b","gemma_2b","gemma_9b","llama3_8b"]
    mlabels = ["Qwen 3B","LLaMA3.2 3B","Gemma 2B","Gemma 9B","LLaMA3 8B"]
    conds   = ["full","minus_first","minus_mid","minus_last","minus_best"]
    colors  = ["#2E75B6","#4472C4","#C00000","#70AD47","#FF7F0F"]

    data = {}
    for m in models:
        data[m] = {}
        for c in conds:
            f = f"{BASE}/results/block_removal/{m}/block_{c}_bbq.json"
            try:
                data[m][c] = json.load(open(f))["metrics"]["ambig"]["bias_score"]
            except: data[m][c] = 0

    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
    fig.suptitle("Block Removal — BBQ Ambiguous Bias Score\n(lower = more debiased)", fontsize=14, y=1.02)

    for ax, m, ml in zip(axes, models, mlabels):
        vals = [data[m][c] for c in conds]
        bars = ax.bar(range(5), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(ml, fontsize=10, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_xticklabels(["full","−first","−mid","−last","−best"], fontsize=8, rotation=30)
        ax.set_ylabel("bias_score", fontsize=8)
        # highlight minus_mid
        bars[2].set_edgecolor("black"); bars[2].set_linewidth(1.5)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.02, f"{v:.4f}", ha="center", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(f"{OUT}/block_removal_bbq_grouped.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ block_removal_bbq_grouped.png")


# 2. Block Only BBQ — completed models only (3+1)

def plot_block_only_bbq():
    models  = ["qwen_3b","llama32_3b","gemma_2b","gemma_9b","llama3_8b"]
    mlabels = ["Qwen 3B","LLaMA3.2 3B","Gemma 2B","Gemma 9B","LLaMA3 8B"]
    conds_only = ["only_first","only_mid","only_last"]
    full_vals  = [0.0002, 0.0348, 0.0156, 0.0001, 0.0675]

    data = {}
    for m in models:
        data[m] = {}
        for c in conds_only:
            f = f"{BASE}/results/block_removal/{m}/block_{c}_bbq.json"
            try:
                data[m][c] = json.load(open(f))["metrics"]["ambig"]["bias_score"]
            except: data[m][c] = None

    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
    fig.suptitle("Block Only — BBQ Ambiguous Bias Score (Sufficiency)\n(lower = that third alone is sufficient for debiasing)", fontsize=14, y=1.02)

    for ax, m, ml, fv in zip(axes, models, mlabels, full_vals):
        vals = [data[m][c] for c in conds_only]
        has_data = [v is not None for v in vals]
        colors_ = [PALETTE[c] for c in conds_only]
        x = range(3)
        bars = ax.bar(x, [v if v is not None else 0 for v in vals],
                      color=[c if h else "#DDDDDD" for c,h in zip(colors_,has_data)],
                      edgecolor="white")
        # full reference line
        ax.axhline(fv, color="#2E75B6", linestyle="--", linewidth=1.2, label=f"full={fv:.4f}")
        ax.set_title(ml, fontsize=10, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["only\nfirst","only\nmid","only\nlast"], fontsize=8)
        ax.set_ylabel("bias_score", fontsize=8)
        bars[1].set_edgecolor("black"); bars[1].set_linewidth(1.5)
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(i, v + max([vv for vv in vals if vv] + [0.001])*0.03,
                        f"{v:.4f}", ha="center", fontsize=7.5)
            else:
                ax.text(i, 0.002, "pending", ha="center", fontsize=7, color="#999999")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/block_only_bbq_grouped.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ block_only_bbq_grouped.png")


# 3. BDE — param_type energy distribution (9 models, stacked bar)

def plot_bde():
    d = json.load(open(f"{ANALYSIS}/bias_direction.json"))
    param_types = ["gate_proj","up_proj","down_proj","q_proj","o_proj","v_proj","k_proj"]
    pt_colors   = ["#C00000","#ED7D31","#FFC000","#4472C4","#70AD47","#7030A0","#AAAAAA"]

    models = [m["model"] for m in d]
    energies = {pt: [] for pt in param_types}
    for m in d:
        te = m["type_energy"]
        for pt in param_types:
            energies[pt].append(te.get(pt, 0))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    for pt, color in zip(param_types, pt_colors):
        vals = np.array(energies[pt])
        ax.bar(x, vals, bottom=bottoms, color=color, label=pt, edgecolor="white", linewidth=0.3)
        # label if >5%
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(i, b + v/2, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m,m) for m in models], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Energy fraction", fontsize=10)
    ax.set_title("BDE — ΔW Parameter Type Energy Distribution\n(gate_proj ranks #1 in 9/9 models)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{OUT}/bde_param_type_energy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ bde_param_type_energy.png")

    # 2nd plot: per-layer alignment score for ≥3B models
    big_models = [m for m in d if m["model"] in ["qwen_3b","gemma_2b","gemma_9b","llama32_3b","llama3_8b"]]
    fig, axes = plt.subplots(1, len(big_models), figsize=(16, 4), sharey=False)
    fig.suptitle("BDE — Per-layer ΔW Bias Alignment Score (≥3B models)", fontsize=12)
    for ax, m in zip(axes, big_models):
        layers = m["layer_data"]
        n = m["n_layers"]
        xs = [l["layer_idx"] for l in layers]
        ys = [l["align_loo"] for l in layers]
        colors_ = [PALETTE[l["third"]] for l in layers]
        ax.bar(xs, ys, color=colors_, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(MODEL_LABELS.get(m["model"], m["model"]), fontsize=9, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("align_loo", fontsize=8)
        # third boundaries
        ax.axvline(n//3-0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(2*n//3-0.5, color="gray", linestyle=":", linewidth=0.8)
    patches = [mpatches.Patch(color=PALETTE[t], label=t) for t in ["first","mid","last"]]
    axes[-1].legend(handles=patches, fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/bde_per_layer_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ bde_per_layer_alignment.png")


# 4. dw_norm — Frobenius norm per layer (9-model grid)

def plot_dw_norm():
    d = json.load(open(f"{ANALYSIS}/dw_norm.json"))
    models_present = [m for m in d if m["model"] in MODEL_ORDER]
    models_present.sort(key=lambda m: MODEL_ORDER.index(m["model"]) if m["model"] in MODEL_ORDER else 99)

    n = len(models_present)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*3.5))
    axes = axes.flatten()
    fig.suptitle("ΔW Frobenius Norm by Layer\n(color: first=blue / mid=red / last=green)", fontsize=13)

    for ax, m in zip(axes, models_present):
        n_layers = m["n_layers"]
        lms = m["layer_metrics"]
        xs = [l["layer_idx"] for l in lms]
        ys = [l["frob_norm"] for l in lms]
        colors_ = [PALETTE["first"] if x < n_layers//3
                   else (PALETTE["mid"] if x < 2*n_layers//3 else PALETTE["last"])
                   for x in xs]
        ax.bar(xs, ys, color=colors_, edgecolor="none", alpha=0.85)
        ax.set_title(f"{MODEL_LABELS.get(m['model'],m['model'])}\nρ={m['spearman_norm_bias']:.3f}", fontsize=9)
        ax.set_xlabel("Layer", fontsize=7); ax.set_ylabel("||ΔW||_F", fontsize=7)
        ax.axvline(n_layers//3-0.5, color="gray", lw=0.8, ls=":")
        ax.axvline(2*n_layers//3-0.5, color="gray", lw=0.8, ls=":")

    for ax in axes[len(models_present):]: ax.set_visible(False)
    patches = [mpatches.Patch(color=PALETTE[t], label=f"{t}-third") for t in ["first","mid","last"]]
    fig.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/dw_norm_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ dw_norm_per_layer.png")

    # thirds summary bar
    thirds_data = {"model":[],"first":[],"mid":[],"last":[]}
    for m in models_present:
        n_layers = m["n_layers"]
        lms = m["layer_metrics"]
        f = np.mean([l["frob_norm"] for l in lms if l["layer_idx"] < n_layers//3])
        mi = np.mean([l["frob_norm"] for l in lms if n_layers//3 <= l["layer_idx"] < 2*n_layers//3])
        la = np.mean([l["frob_norm"] for l in lms if l["layer_idx"] >= 2*n_layers//3])
        thirds_data["model"].append(MODEL_LABELS.get(m["model"],m["model"]))
        thirds_data["first"].append(f); thirds_data["mid"].append(mi); thirds_data["last"].append(la)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(thirds_data["model"])); w = 0.25
    ax.bar(x-w, thirds_data["first"], w, color=PALETTE["first"], label="first-third")
    ax.bar(x,   thirds_data["mid"],   w, color=PALETTE["mid"],   label="mid-third")
    ax.bar(x+w, thirds_data["last"],  w, color=PALETTE["last"],  label="last-third")
    ax.set_xticks(x); ax.set_xticklabels(thirds_data["model"], rotation=30, ha="right")
    ax.set_ylabel("Mean ||ΔW||_F"); ax.set_title("ΔW Frobenius Norm — Mean by Third\n(norm dominant ≠ behavioral dominant)")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/dw_norm_thirds_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ dw_norm_thirds_bar.png")


# 5. dw_svd — stable_rank & eff_rank per layer (≥3B grid)

def plot_dw_svd():
    d = json.load(open(f"{ANALYSIS}/dw_svd.json"))
    big = [m for m in d if m["model"] in ["qwen_3b","gemma_2b","gemma_9b","llama32_3b","llama3_8b"]]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("ΔW SVD — Stable Rank & Top-SV Concentration by Layer", fontsize=13)

    for col, m in enumerate(big):
        n_layers = m["n_layers"]
        lms = m["layer_metrics"]
        xs = [l["layer_idx"] for l in lms]
        sr = [l["stable_rank_mean"] for l in lms]
        tc = [l["top_sv_conc_mean"] for l in lms]
        colors_ = [PALETTE["first"] if x < n_layers//3
                   else (PALETTE["mid"] if x < 2*n_layers//3 else PALETTE["last"])
                   for x in xs]

        # stable rank
        ax = axes[0][col]
        ax.bar(xs, sr, color=colors_, edgecolor="none", alpha=0.85)
        ax.set_title(MODEL_LABELS.get(m["model"],m["model"]), fontsize=9, fontweight="bold")
        ax.set_ylabel("stable_rank" if col==0 else "", fontsize=8)
        ax.axvline(n_layers//3-0.5, color="gray", lw=0.8, ls=":")
        ax.axvline(2*n_layers//3-0.5, color="gray", lw=0.8, ls=":")

        # top-sv concentration
        ax2 = axes[1][col]
        ax2.bar(xs, tc, color=colors_, edgecolor="none", alpha=0.85)
        ax2.set_ylabel("top_sv_conc" if col==0 else "", fontsize=8)
        ax2.set_xlabel("Layer", fontsize=8)
        ax2.axvline(n_layers//3-0.5, color="gray", lw=0.8, ls=":")
        ax2.axvline(2*n_layers//3-0.5, color="gray", lw=0.8, ls=":")

    patches = [mpatches.Patch(color=PALETTE[t], label=f"{t}-third") for t in ["first","mid","last"]]
    fig.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/dw_svd_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ dw_svd_per_layer.png")


# 6. rank-k capture — thirds comparison

def plot_rankk():
    d = json.load(open(f"{ANALYSIS}/rankk_capture.json"))
    # format: {model_name: {n_layers, thirds_k90: {early,mid,late}, layers: {idx: {avg: {k: val}}}}}

    model_keys = [k for k in d.keys() if k in MODEL_ORDER]
    model_keys.sort(key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    if not model_keys:
        print("skip rankk (no known models)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rank-k Capture — Top-k Singular Values Explain ||ΔW||²_F", fontsize=12)

    # per-layer k=10 capture for big models
    big_keys = [m for m in model_keys if m in ["qwen_3b","gemma_2b","llama32_3b","llama3_8b","gemma_9b"]]
    ax = axes[0]
    for mk in big_keys:
        m = d[mk]
        layers = m["layers"]
        xs = sorted(int(k) for k in layers.keys())
        k10 = [layers[str(x)]["avg"].get("10", 0) for x in xs]
        ax.plot(xs, k10, label=MODEL_LABELS.get(mk, mk), alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("k=10 capture ratio")
    ax.set_title("Top-10 SV Capture per Layer"); ax.legend(fontsize=7)

    # thirds_k90 bar (early/mid/late → first/mid/last naming)
    ax2 = axes[1]
    mnames = [MODEL_LABELS.get(mk, mk) for mk in model_keys]
    x = np.arange(len(model_keys)); w = 0.25
    third_map = [("early", PALETTE["first"], "early-third"),
                 ("mid",   PALETTE["mid"],   "mid-third"),
                 ("late",  PALETTE["last"],  "late-third")]
    for i, (tkey, color, label) in enumerate(third_map):
        vals = [d[mk]["thirds_k90"].get(tkey, 0) for mk in model_keys]
        ax2.bar(x + (i-1)*w, vals, w, color=color, label=label)
    ax2.set_xticks(x); ax2.set_xticklabels(mnames, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Mean k90 threshold"); ax2.set_title("k90 Capture Threshold by Third"); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/rankk_capture_by_third.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ rankk_capture_by_third.png")


# 7. cross_benchmark_spearman — benchmark cross-correlation matrix

def plot_cross_bench():
    d = json.load(open(f"{ANALYSIS}/cross_benchmark_spearman.json"))

    # expect: r values between BBQ/CrowS/StereoSet
    benchmarks = ["BBQ","CrowS","StereoSet"]
    # try to extract correlation matrix
    try:
        corr = d.get("correlation_matrix") or d.get("spearman_matrix")
        if corr is None:
            # flat format
            r_bc = d.get("r_bbq_crows", d.get("spearman_bbq_crows", 0))
            r_bs = d.get("r_bbq_ss",    d.get("spearman_bbq_ss",    0))
            r_cs = d.get("r_crows_ss",  d.get("spearman_crows_ss",  0))
            p_bc = d.get("p_bbq_crows", 1); p_bs = d.get("p_bbq_ss", 1); p_cs = d.get("p_crows_ss", 1)
            mat = np.array([[1.0, r_bc, r_bs],[r_bc, 1.0, r_cs],[r_bs, r_cs, 1.0]])
            pmat = np.array([[0,p_bc,p_bs],[p_bc,0,p_cs],[p_bs,p_cs,0]])
        else:
            mat = np.array(corr); pmat = np.zeros_like(mat)
    except:
        print("skip cross_bench (format issue)")
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(benchmarks); ax.set_yticklabels(benchmarks)
    for i in range(3):
        for j in range(3):
            sig = ""
            if i!=j:
                p = pmat[i,j]
                sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else " ns"))
            ax.text(j, i, f"{mat[i,j]:.3f}\n{sig}", ha="center", va="center",
                    fontsize=11, color="white" if abs(mat[i,j])>0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Cross-Benchmark Spearman\n(best-layer rank correlation)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cross_benchmark_spearman.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ cross_benchmark_spearman.png")


# 8. mode_based — verify mid-third mode presence

def plot_mode_based():
    d = json.load(open(f"{ANALYSIS}/mode_based.json"))
    entries = d if isinstance(d, list) else d.get("results", [])
    if not entries:
        print("skip mode_based (empty)")
        return

    # benchmarks are keys: bbq, crows, ss
    benchmarks = ["bbq", "crows", "ss"]
    bm_labels  = {"bbq": "BBQ", "crows": "CrowS", "ss": "StereoSet"}

    entries_known = [e for e in entries if e["model"] in MODEL_ORDER]
    entries_known.sort(key=lambda e: MODEL_ORDER.index(e["model"]))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bm = ["#2E75B6", "#C00000", "#70AD47"]
    markers = ["o", "s", "^"]

    for bi, bm in enumerate(benchmarks):
        xs, ys = [], []
        for e in entries_known:
            bdata = e.get(bm)
            if bdata is None: continue
            best = bdata.get("global_min_layer")
            n_layers = e.get("n_layers", 1)
            if best is not None:
                xs.append(MODEL_ORDER.index(e["model"]))
                ys.append(best / n_layers)
        if xs:
            ax.scatter(xs, ys, label=bm_labels[bm], color=colors_bm[bi],
                       marker=markers[bi], s=80, zorder=3)
            ax.plot(xs, ys, color=colors_bm[bi], alpha=0.4, linewidth=1)

    ax.axhspan(1/3, 2/3, alpha=0.08, color=PALETTE["mid"], label="mid-third region")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ORDER], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Global min layer (normalized depth 0->1)")
    ax.set_title("Mode-based Analysis — Global Min Layer Position\n(mid-third: 0.33-0.67)")
    ax.legend(fontsize=9); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUT}/mode_based_best_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ mode_based_best_layer.png")


# 9. signflip_symmetry — symmetry_score per model

def plot_signflip_sym():
    d = json.load(open(f"{ANALYSIS}/signflip_symmetry.json"))
    entries = d if isinstance(d, list) else d.get("results", [])
    if not entries:
        print("skip signflip_symmetry"); return

    models_s = [e["model"] for e in entries]
    sym = [e.get("symmetry_score", e.get("symmetry", 0)) for e in entries]
    delta_pos = [e.get("delta_pos", e.get("max_pos_bias",0)) for e in entries]
    delta_neg = [e.get("delta_neg", e.get("min_neg_bias",0)) for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sign-flip Symmetry Analysis", fontsize=12)

    # symmetry score bar
    ax = axes[0]
    x = np.arange(len(models_s))
    colors_ = [PALETTE["mid"] if s < 0.3 else ("#4472C4" if s < 0.6 else "#70AD47") for s in sym]
    bars = ax.bar(x, sym, color=colors_, edgecolor="white")
    ax.axhline(0.25, color="gray", ls="--", lw=1, label="qwen_3b (symmetric)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m,m) for m in models_s], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("symmetry_score"); ax.set_title("Symmetry Score\n(lower = more asymmetric)")
    for i, v in enumerate(sym):
        ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()

    # Δpos vs Δneg scatter
    ax2 = axes[1]
    ax2.scatter(delta_pos, [abs(v) for v in delta_neg],
                c=[MODEL_ORDER.index(m) if m in MODEL_ORDER else 0 for m in models_s],
                cmap="tab10", s=80, zorder=3)
    for i, m in enumerate(models_s):
        ax2.annotate(MODEL_LABELS.get(m,m), (delta_pos[i], abs(delta_neg[i])),
                     fontsize=7, xytext=(4,4), textcoords="offset points")
    ax2.plot([0, max(delta_pos)*1.1], [0, max(delta_pos)*1.1],
             "k--", lw=0.8, label="perfect symmetry")
    ax2.set_xlabel("Δ bias (positive ΔW)"); ax2.set_ylabel("|Δ bias (negative ΔW)|")
    ax2.set_title("Asymmetry: +ΔW vs −ΔW effect"); ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/signflip_symmetry.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ signflip_symmetry.png")


# 10. crossmodel_alignment — module-type Spearman heatmap

def plot_crossmodel_align():
    d = json.load(open(f"{ANALYSIS}/crossmodel_alignment.json"))
    # format: {module_type_spearman: {mod: {ge3b_mean_r, le1.5b_mean_r}}, per_model: {name: {thirds, ...}}}
    try:
        mts = d.get("module_type_spearman", {})
        all_types = ["gate_proj","up_proj","down_proj","q_proj","k_proj","v_proj","o_proj"]
        all_types = [t for t in all_types if t in mts]
        if not all_types:
            print("skip crossmodel_alignment (no module_type_spearman)")
            return

        ge3b_r  = [mts[t]["ge3b_mean_r"]   for t in all_types]
        le15b_r = [mts[t]["le1.5b_mean_r"]  for t in all_types]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Cross-model ΔW Alignment — Spearman r by Module Type", fontsize=12)

        # module type mean r bar
        ax = axes[0]
        x = np.arange(len(all_types)); w = 0.35
        ax.bar(x - w/2, ge3b_r,  w, color="#2E75B6", label=">=3B pairs")
        ax.bar(x + w/2, le15b_r, w, color="#70AD47", label="<=1.5B pairs")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(all_types, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean Spearman r"); ax.set_title("Module-type Mean r\n(gate_proj highest for >=3B)")
        ax.legend(fontsize=9)
        for i, (g, s) in enumerate(zip(ge3b_r, le15b_r)):
            ax.text(i-w/2, g + (0.02 if g >= 0 else -0.06), f"{g:.2f}", ha="center", fontsize=7.5, color="#2E75B6")
            ax.text(i+w/2, s + (0.02 if s >= 0 else -0.06), f"{s:.2f}", ha="center", fontsize=7.5, color="#70AD47")

        # per-model thirds energy
        pm = d.get("per_model", {})
        model_keys = [k for k in pm.keys() if k in MODEL_ORDER]
        model_keys.sort(key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
        if model_keys:
            ax2 = axes[1]
            x2 = np.arange(len(model_keys)); w2 = 0.25
            for i, (tkey, color, label) in enumerate([("early", PALETTE["first"], "early"),
                                                       ("mid",   PALETTE["mid"],   "mid"),
                                                       ("late",  PALETTE["last"],  "late")]):
                vals = [pm[mk]["thirds"].get(tkey, 0) for mk in model_keys]
                ax2.bar(x2 + (i-1)*w2, vals, w2, color=color, label=label)
            ax2.set_xticks(x2)
            ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_keys],
                                 rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("Alignment fraction"); ax2.set_title("Per-model ΔW Thirds Alignment")
            ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{OUT}/crossmodel_alignment_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("✓ crossmodel_alignment_heatmap.png")
    except Exception as e:
        print(f"skip crossmodel_alignment: {e}")


# 11. probe_bias_corr — scatter

def plot_probe_corr():
    d = json.load(open(f"{ANALYSIS}/probe_bias_corr.json"))
    if not d["results"]: print("skip probe_corr"); return
    r = d["results"][0]
    layers = r["per_layer"]
    xs = [l["probe_acc"] for l in layers]
    ys = [l["bias_score"] for l in layers]
    layer_ids = [l["layer"] for l in layers]

    n_layers = len(layers)
    colors_ = [PALETTE["first"] if l["layer"] < n_layers//3
               else (PALETTE["mid"] if l["layer"] < 2*n_layers//3 else PALETTE["last"])
               for l in layers]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xs, ys, c=colors_, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    for i, (x, y, li) in enumerate(zip(xs, ys, layer_ids)):
        if li % 4 == 0:
            ax.annotate(f"L{li}", (x, y), fontsize=6.5,
                        xytext=(3,3), textcoords="offset points", color="#555555")

    # regression line
    m_, b_ = np.polyfit(xs, ys, 1)
    xline = np.linspace(min(xs), max(xs), 100)
    ax.plot(xline, m_*xline+b_, "k--", lw=1, alpha=0.5)

    patches = [mpatches.Patch(color=PALETTE[t], label=f"{t}-third") for t in ["first","mid","last"]]
    ax.legend(handles=patches, fontsize=8)
    ax.set_xlabel("Probe accuracy (correct_answer_3class)", fontsize=10)
    ax.set_ylabel("BBQ ambig bias_score", fontsize=10)
    ax.set_title(f"Probe Accuracy vs Bias Score — LLaMA3 8B\n"
                 f"Spearman r={r['spearman_r']:.3f}  p={r['spearman_p']:.3f} (ns)\n"
                 f"Note: label_mode=correct_answer_3class (bias_binary not yet implemented)", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/probe_bias_corr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ probe_bias_corr_scatter.png")


# 12. Block Removal — Necessity vs Sufficiency comparison summary

def plot_necessity_sufficiency_summary():
    models_done  = ["qwen_3b","llama32_3b","gemma_2b"]
    mlabels_done = ["Qwen 3B","LLaMA3.2 3B","Gemma 2B"]
    full_vals    = [0.0002, 0.0348, 0.0156]
    minus_mid    = [0.0190, 0.2493, 0.1843]
    only_mid     = [0.0007, 0.0203, 0.0126]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models_done)); w = 0.22
    ax.bar(x-w, full_vals,  w, color="#2E75B6", label="full (all layers)")
    ax.bar(x,   minus_mid,  w, color="#C00000", label="minus_mid (necessity: mid removed)")
    ax.bar(x+w, only_mid,   w, color="#70AD47", label="only_mid (sufficiency: mid only)")

    for i, (fv, mm, om) in enumerate(zip(full_vals, minus_mid, only_mid)):
        ax.text(i-w, fv+0.003,  f"{fv:.4f}", ha="center", fontsize=7.5, color="#2E75B6")
        ax.text(i,   mm+0.003,  f"{mm:.4f}", ha="center", fontsize=7.5, color="#C00000", fontweight="bold")
        ax.text(i+w, om+0.003,  f"{om:.4f}", ha="center", fontsize=7.5, color="#70AD47")

    ax.set_xticks(x); ax.set_xticklabels(mlabels_done, fontsize=11)
    ax.set_ylabel("BBQ ambiguous bias_score", fontsize=10)
    ax.set_title("Necessity × Sufficiency — Double Dissociation\n"
                 "minus_mid↑↑ (necessity) + only_mid≈full (sufficiency) → mid-third = functional locus", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/necessity_sufficiency_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ necessity_sufficiency_summary.png")


# RUN ALL

if __name__ == "__main__":
    print("=== Generating all missing plots ===\n")
    plot_block_removal_bbq()
    plot_block_only_bbq()
    plot_bde()
    plot_dw_norm()
    plot_dw_svd()
    plot_rankk()
    plot_cross_bench()
    plot_mode_based()
    plot_signflip_sym()
    plot_crossmodel_align()
    plot_probe_corr()
    plot_necessity_sufficiency_summary()
    print(f"\n=== Done. Saved to {OUT} ===")
