#!/usr/bin/env python3
"""
ΔW Norm vs BBQ/MMLU Dissociation Analysis.

Hypothesis: the third with highest ΔW Frobenius norm predicts BBQ bias sensitivity
but NOT MMLU capability sensitivity — i.e., ΔW magnitude concentration is bias-specific.

For each model:
  - dw_norm thirds: which third has highest mean_frob_norm?
  - block_removal BBQ: Δbias_score per condition (minus_first/mid/last vs full)
  - block_removal MMLU: Δmmlu_acc per condition (minus_first/mid/last vs full)

Expected finding:
  - High-norm third removal → large BBQ Δ (bias sensitive)
  - High-norm third removal → small MMLU Δ (capability spared)
  = weight magnitude concentration is bias-specific

Runs with BBQ-only if MMLU not yet available. Full dissociation plot once MMLU arrives.

Output: results/analysis/norm_mmlu_dissociation.json
        results/analysis/plots/norm_mmlu_dissociation.png
"""
from __future__ import annotations
from pathlib import Path
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PROJ_ROOT = Path(__file__).resolve().parents[2]

BASE     = str(PROJ_ROOT)
ANALYSIS = f"{BASE}/results/analysis"
BLOCK_DIR = f"{BASE}/results/block_removal"
OUT_JSON  = f"{ANALYSIS}/norm_mmlu_dissociation.json"
OUT_PLOT  = f"{ANALYSIS}/plots/norm_mmlu_dissociation.png"

MODELS = ["qwen_3b", "llama32_3b", "gemma_2b", "gemma_9b", "llama3_8b"]
MODEL_LABELS = {
    "qwen_3b": "Qwen 3B", "llama32_3b": "LLaMA3.2 3B",
    "gemma_2b": "Gemma 2B", "gemma_9b": "Gemma 9B", "llama3_8b": "LLaMA3 8B",
}
CONDS = ["minus_first", "minus_mid", "minus_last"]
COND_COLORS = {"minus_first": "#4472C4", "minus_mid": "#C00000", "minus_last": "#70AD47"}
COND_LABELS = {"minus_first": "−first", "minus_mid": "−mid", "minus_last": "−last"}


def load_dw_norm_thirds() -> dict[str, dict]:
    d = json.load(open(f"{ANALYSIS}/dw_norm.json"))
    out = {}
    for m in d:
        alias = m["model"]
        if alias not in MODELS:
            continue
        n = m["n_layers"]
        lms = m["layer_metrics"]
        f  = np.mean([l["frob_norm"] for l in lms if l["layer_idx"] < n // 3])
        mi = np.mean([l["frob_norm"] for l in lms if n // 3 <= l["layer_idx"] < 2 * n // 3])
        la = np.mean([l["frob_norm"] for l in lms if l["layer_idx"] >= 2 * n // 3])
        out[alias] = {
            "n_layers": n,
            "norm_thirds": {"first": round(float(f), 4), "mid": round(float(mi), 4), "last": round(float(la), 4)},
            "dominant_norm": max([("first", f), ("mid", mi), ("last", la)], key=lambda x: x[1])[0],
        }
    return out


def load_block_bbq(alias: str) -> dict:
    out = {}
    for cond in ["full"] + CONDS:
        path = f"{BLOCK_DIR}/{alias}/block_{cond}_bbq.json"
        if os.path.exists(path):
            out[cond] = json.load(open(path))["metrics"]["ambig"]["bias_score"]
    return out


def load_block_mmlu(alias: str) -> dict | None:
    path = f"{BLOCK_DIR}/{alias}/block_mmlu_summary.json"
    if os.path.exists(path):
        d = json.load(open(path))
        # format: {"results": [{"condition": "full", "mmlu_acc": ...}, ...]}
        results = d.get("results", [])
        if results:
            out = {r["condition"]: r["mmlu_acc"] for r in results if "mmlu_acc" in r}
            # normalize condition names: "minus_first" -> "minus_first"
            normalized = {}
            for k, v in out.items():
                key = k.replace("block_", "")
                normalized[key] = v
            return normalized if normalized else None
    # fallback: per-condition files
    out = {}
    for cond in ["full"] + CONDS:
        p = f"{BLOCK_DIR}/{alias}/block_{cond}_mmlu.json"
        if os.path.exists(p):
            d = json.load(open(p))
            acc = d.get("mmlu_acc") or d.get("accuracy") or d.get("overall_acc")
            if acc is None:
                metrics = d.get("metrics", {})
                acc = metrics.get("mmlu_acc") or metrics.get("accuracy")
            if acc is not None:
                out[cond] = acc
    return out if out else None


def main():
    norm_data = load_dw_norm_thirds()
    records = []
    has_mmlu = False

    for alias in MODELS:
        if alias not in norm_data:
            print(f"[skip] {alias}: no dw_norm data")
            continue

        bbq  = load_block_bbq(alias)
        mmlu = load_block_mmlu(alias)
        if mmlu:
            has_mmlu = True

        nd = norm_data[alias]
        r = dict(
            model=alias,
            n_layers=nd["n_layers"],
            norm_thirds=nd["norm_thirds"],
            dominant_norm=nd["dominant_norm"],
        )

        # BBQ delta (vs full)
        if "full" in bbq:
            r["bbq_full"] = round(bbq["full"], 6)
            r["bbq_delta"] = {}
            for c in CONDS:
                if c in bbq:
                    r["bbq_delta"][c] = round(bbq[c] - bbq["full"], 6)

        # MMLU delta (vs full)
        if mmlu and "full" in mmlu:
            r["mmlu_full"] = round(mmlu["full"], 4)
            r["mmlu_delta"] = {}
            for c in CONDS:
                if c in mmlu:
                    r["mmlu_delta"][c] = round(mmlu[c] - mmlu["full"], 4)

        records.append(r)
        print(f"\n[{alias}] dominant_norm={nd['dominant_norm']}  norms={nd['norm_thirds']}")
        if "bbq_delta" in r:
            print(f"  BBQ Δ: {r['bbq_delta']}")
        if "mmlu_delta" in r:
            print(f"  MMLU Δ: {r['mmlu_delta']}")

    # Save JSON
    output = dict(
        has_mmlu=has_mmlu,
        hypothesis=(
            "High-norm third removal → large BBQ Δ (bias-sensitive) "
            "but small MMLU Δ (capability-spared) = bias-specific weight concentration"
        ),
        models=records,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON → {OUT_JSON}")

    # Plot
    n_models = len(records)
    if n_models == 0:
        print("No data to plot.")
        return

    if has_mmlu:
        fig, axes = plt.subplots(3, n_models, figsize=(4 * n_models, 10))
        row_titles = ["ΔW Norm by Third", "BBQ Δ (bias)", "MMLU Δ (capability)"]
    else:
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 7))
        row_titles = ["ΔW Norm by Third", "BBQ Δ (bias)"]

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "ΔW Norm vs Bias/Capability Dissociation\n"
        "(high-norm third: bias-sensitive but capability-spared?)",
        fontsize=13, y=1.01
    )

    for col, r in enumerate(records):
        alias = r["model"]
        nd = r["norm_thirds"]
        norm_vals = [nd["first"], nd["mid"], nd["last"]]
        dom = r["dominant_norm"]
        norm_colors = [
            ("#C00000" if t == dom else "#AAAAAA")
            for t in ["first", "mid", "last"]
        ]

        # Row 0: norm
        ax0 = axes[0][col]
        bars = ax0.bar(["first", "mid", "last"], norm_vals, color=norm_colors, edgecolor="white")
        ax0.set_title(MODEL_LABELS.get(alias, alias), fontsize=10, fontweight="bold")
        ax0.set_ylabel("Mean ||ΔW||_F" if col == 0 else "", fontsize=8)
        for i, v in enumerate(norm_vals):
            ax0.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=7.5)
        ax0.set_xlabel("Third", fontsize=8)

        # Row 1: BBQ delta
        ax1 = axes[1][col]
        if "bbq_delta" in r:
            bd = r["bbq_delta"]
            cond_vals = [bd.get(c, 0) for c in CONDS]
            bar_colors = [COND_COLORS[c] for c in CONDS]
            bars1 = ax1.bar(CONDS, cond_vals, color=bar_colors, edgecolor="white")
            ax1.axhline(0, color="black", lw=0.7)
            ax1.set_ylabel("Δ bias_score" if col == 0 else "", fontsize=8)
            ax1.set_xticklabels([COND_LABELS[c] for c in CONDS], fontsize=8)
            # highlight dominant norm third's removal
            dom_cond = f"minus_{dom}"
            for i, c in enumerate(CONDS):
                if c == dom_cond:
                    bars1[i].set_edgecolor("black"); bars1[i].set_linewidth(2)
            for i, v in enumerate(cond_vals):
                ax1.text(i, v + (max(cond_vals) * 0.04 if v >= 0 else min(cond_vals) * 0.04),
                         f"{v:+.3f}", ha="center", fontsize=7.5, fontweight="bold")
        else:
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("Condition", fontsize=8)

        # Row 2: MMLU delta (if available)
        if has_mmlu:
            ax2 = axes[2][col]
            if "mmlu_delta" in r:
                md = r["mmlu_delta"]
                cond_vals2 = [md.get(c, 0) for c in CONDS]
                bars2 = ax2.bar(CONDS, cond_vals2, color=[COND_COLORS[c] for c in CONDS], edgecolor="white")
                ax2.axhline(0, color="black", lw=0.7)
                ax2.set_ylabel("Δ MMLU acc" if col == 0 else "", fontsize=8)
                ax2.set_xticklabels([COND_LABELS[c] for c in CONDS], fontsize=8)
                dom_cond = f"minus_{dom}"
                for i, c in enumerate(CONDS):
                    if c == dom_cond:
                        bars2[i].set_edgecolor("black"); bars2[i].set_linewidth(2)
                for i, v in enumerate(cond_vals2):
                    ax2.text(i, v + (max(abs(x) for x in cond_vals2 + [0.001]) * 0.08 * (1 if v >= 0 else -1)),
                             f"{v:+.3f}", ha="center", fontsize=7.5)
            else:
                ax2.text(0.5, 0.5, "MMLU\nincomplete", ha="center", va="center",
                         transform=ax2.transAxes, fontsize=10, color="#999999")
            ax2.set_xlabel("Condition", fontsize=8)

    # Row labels
    for row, title in enumerate(row_titles):
        axes[row][0].set_ylabel(f"{title}\n{axes[row][0].get_ylabel()}", fontsize=8)

    # Legend
    patches = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c]) for c in CONDS]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {OUT_PLOT}")
    if not has_mmlu:
        print("  (MMLU row shows placeholder — re-run after block_removal_mmlu completes)")


if __name__ == "__main__":
    main()
