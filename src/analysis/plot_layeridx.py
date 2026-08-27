"""
Layer-index-based plots for all experiments.
X-axis: actual layer index (L0, L1, ..., LN-1)
Outputs: results/analysis/plots/layer_index/

Usage:
  python src/analysis/plot_layeridx.py
  python src/analysis/plot_layeridx.py --only sweep signflip
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = Path("results")
OUT  = Path("results/analysis/plots/layer_index")

ALIASES = [
    "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_14b",
    "gemma_2b", "gemma_9b", "gemma3_12b",
    "llama32_1b", "llama32_3b", "llama3_8b", "llama31_8b",
]

DISPLAY = {
    "qwen_0.5b":   "Qwen2.5 0.5B",
    "qwen_1.5b":   "Qwen2.5 1.5B",
    "qwen_3b":     "Qwen2.5 3B",
    "qwen_14b":    "Qwen2.5 14B",
    "gemma_2b":    "Gemma2 2B",
    "gemma_9b":    "Gemma2 9B",
    "gemma3_12b":  "Gemma3 12B",
    "llama32_1b":  "LLaMA3.2 1B",
    "llama32_3b":  "LLaMA3.2 3B",
    "llama3_8b":   "LLaMA3 8B",
    "llama31_8b":  "LLaMA3.1 8B",
}

COLORS = {
    "qwen_0.5b":   "#1f77b4",
    "qwen_1.5b":   "#aec7e8",
    "qwen_3b":     "#ff7f0e",
    "qwen_14b":    "#ffbb78",
    "gemma_2b":    "#2ca02c",
    "gemma_9b":    "#98df8a",
    "gemma3_12b":  "#17becf",
    "llama32_1b":  "#d62728",
    "llama32_3b":  "#ff9896",
    "llama3_8b":   "#9467bd",
    "llama31_8b":  "#c5b0d5",
}

BASELINES = {
    "gemma_2b":   0.1599,
    "gemma_9b":   0.1163,
    "llama32_1b": 0.2508,
    "llama3_8b":  0.1274,
    "qwen_0.5b":  0.2521,
    "qwen_1.5b":  0.2725,
    "qwen_14b":   0.0129,
    "qwen_3b":    0.0503,
}

# data loaders

def load_sweep(alias):
    """Returns list of (layer_idx, ambig_bias, disambig_bias) sorted by layer."""
    d = BASE / "sweep_thirds" / alias
    files = sorted(glob.glob(str(d / "sweep_L*_bbq.json")))
    rows = []
    for f in files:
        with open(f) as fp:
            rec = json.load(fp)
        rows.append((
            rec["layer_idx"],
            rec["metrics"]["ambig"]["bias_score"],
            rec["metrics"]["disambig"]["bias_score"],
        ))
    rows.sort(key=lambda x: x[0])
    return rows


def load_signflip(alias):
    d = BASE / "signflip" / alias
    files = sorted(glob.glob(str(d / "sweep_L*_bbq.json")))
    rows = []
    for f in files:
        with open(f) as fp:
            rec = json.load(fp)
        rows.append((
            rec["layer_idx"],
            rec["metrics"]["ambig"]["bias_score"],
            rec["metrics"]["disambig"]["bias_score"],
        ))
    rows.sort(key=lambda x: x[0])
    return rows


def load_cumulative(alias):
    """Returns list of (cumul_k, ambig_bias, disambig_bias) sorted by k."""
    d = BASE / "cumulative" / alias
    files = sorted(glob.glob(str(d / "cumul_L*_bbq.json")))
    rows = []
    for f in files:
        with open(f) as fp:
            rec = json.load(fp)
        rows.append((
            rec["cumul_k"],
            rec["metrics"]["ambig"]["bias_score"],
            rec["metrics"]["disambig"]["bias_score"],
        ))
    rows.sort(key=lambda x: x[0])
    return rows


def load_control(alias):
    """Returns dict: condition → list of (layer_idx, ambig_bias)."""
    d = BASE / "control_sweep" / alias
    if not d.exists():
        return {}
    cond_data = {}
    for f in sorted(glob.glob(str(d / "ctrl_*_bbq.json"))):
        fname = Path(f).stem
        # e.g. ctrl_norm_matched_seed0_L05
        parts = fname.split("_L")[0]  # ctrl_norm_matched_seed0
        # condition key: drop 'ctrl_'
        cond = parts[len("ctrl_"):]   # norm_matched_seed0
        with open(f) as fp:
            rec = json.load(fp)
        cond_data.setdefault(cond, []).append((
            rec["layer_idx"],
            rec["metrics"]["ambig"]["bias_score"],
        ))
    for k in cond_data:
        cond_data[k].sort(key=lambda x: x[0])
    return cond_data


# helpers

def _set_layer_xticks(ax, n_layers, max_ticks=10):
    step = max(1, n_layers // max_ticks)
    ticks = list(range(0, n_layers, step))
    if (n_layers - 1) not in ticks:
        ticks.append(n_layers - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"L{t}" for t in ticks], fontsize=7, rotation=45, ha="right")


def _thirds_spans(n_layers, ax, alpha=0.07):
    t1 = n_layers // 3
    t2 = 2 * n_layers // 3
    ax.axvspan(-0.5, t1 - 0.5, color="#1f77b4", alpha=alpha, label="first third")
    ax.axvspan(t1 - 0.5, t2 - 0.5, color="#ff7f0e", alpha=alpha, label="mid third")
    ax.axvspan(t2 - 0.5, n_layers - 0.5, color="#2ca02c", alpha=alpha, label="last third")


# plot functions

def plot_sweep_individual(alias):
    rows = load_sweep(alias)
    if not rows:
        print(f"[skip sweep] {alias}: no data")
        return
    layers, ambig, disambig = zip(*rows)
    n = max(layers) + 1
    baseline = BASELINES.get(alias)

    fig, ax = plt.subplots(figsize=(8, 4))
    _thirds_spans(n, ax)
    ax.plot(layers, ambig, "o-", color="#1f77b4", ms=4, lw=1.5, label="ambig bias")
    ax.plot(layers, disambig, "s--", color="#ff7f0e", ms=4, lw=1.2, label="disambig bias")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1, label=f"base={baseline:.4f}")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Bias score")
    ax.set_title(f"{DISPLAY[alias]} — Layer-wise sweep (single layer +ΔW)")
    _set_layer_xticks(ax, n)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "sweep" / f"sweep_{alias}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_sweep_grid():
    available = [a for a in ALIASES if (BASE / "sweep_thirds" / a).exists()]
    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, alias in enumerate(available):
        rows = load_sweep(alias)
        if not rows:
            axes[i].set_visible(False)
            continue
        layers, ambig, disambig = zip(*rows)
        nl = max(layers) + 1
        baseline = BASELINES.get(alias)
        ax = axes[i]
        _thirds_spans(nl, ax)
        ax.plot(layers, ambig, "o-", color="#1f77b4", ms=3, lw=1.2, label="ambig")
        ax.plot(layers, disambig, "s--", color="#ff7f0e", ms=3, lw=1, label="disambig")
        if baseline is not None:
            ax.axhline(baseline, color="gray", ls=":", lw=0.8)
        ax.set_title(DISPLAY[alias], fontsize=9)
        ax.set_xlabel("Layer index", fontsize=7)
        ax.set_ylabel("Bias", fontsize=7)
        _set_layer_xticks(ax, nl, max_ticks=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Layer-wise sweep (single layer +ΔW) — all models", fontsize=12)
    fig.tight_layout()
    out = OUT / "sweep" / "sweep_all_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_signflip_individual(alias):
    pos_rows = load_sweep(alias)
    neg_rows = load_signflip(alias)
    if not pos_rows or not neg_rows:
        print(f"[skip signflip] {alias}: missing data")
        return
    pos_layers, pos_ambig, _ = zip(*pos_rows)
    neg_layers, neg_ambig, _ = zip(*neg_rows)
    n = max(max(pos_layers), max(neg_layers)) + 1
    baseline = BASELINES.get(alias)

    fig, ax = plt.subplots(figsize=(8, 4))
    _thirds_spans(n, ax)
    ax.plot(pos_layers, pos_ambig, "o-", color="#1f77b4", ms=4, lw=1.5, label="+ΔW (bias xfer)")
    ax.plot(neg_layers, neg_ambig, "s--", color="#d62728", ms=4, lw=1.5, label="−ΔW (sign-flip)")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1, label=f"base={baseline:.4f}")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Ambig bias score")
    ax.set_title(f"{DISPLAY[alias]} — Sign-flip comparison (+ΔW vs −ΔW)")
    _set_layer_xticks(ax, n)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "signflip" / f"signflip_{alias}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_signflip_grid():
    sf_aliases = [a for a in ALIASES if (BASE / "signflip" / a).exists()]
    ncols = 3
    nrows = (len(sf_aliases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, alias in enumerate(sf_aliases):
        pos_rows = load_sweep(alias)
        neg_rows = load_signflip(alias)
        if not pos_rows or not neg_rows:
            axes[i].set_visible(False)
            continue
        pos_layers, pos_ambig, _ = zip(*pos_rows)
        neg_layers, neg_ambig, _ = zip(*neg_rows)
        n = max(max(pos_layers), max(neg_layers)) + 1
        baseline = BASELINES.get(alias)
        ax = axes[i]
        _thirds_spans(n, ax)
        ax.plot(pos_layers, pos_ambig, "o-", color="#1f77b4", ms=3, lw=1.2, label="+ΔW")
        ax.plot(neg_layers, neg_ambig, "s--", color="#d62728", ms=3, lw=1.2, label="−ΔW")
        if baseline is not None:
            ax.axhline(baseline, color="gray", ls=":", lw=0.8)
        ax.set_title(DISPLAY[alias], fontsize=9)
        ax.set_xlabel("Layer index", fontsize=7)
        ax.set_ylabel("Ambig bias", fontsize=7)
        _set_layer_xticks(ax, n, max_ticks=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Sign-flip comparison (+ΔW vs −ΔW) — all models", fontsize=12)
    fig.tight_layout()
    out = OUT / "signflip" / "signflip_all_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_cumulative_individual(alias):
    rows = load_cumulative(alias)
    if not rows:
        print(f"[skip cumulative] {alias}: no data")
        return
    ks, ambig, disambig = zip(*rows)
    baseline = BASELINES.get(alias)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, ambig, "o-", color="#1f77b4", ms=4, lw=1.5, label="ambig bias")
    ax.plot(ks, disambig, "s--", color="#ff7f0e", ms=4, lw=1.2, label="disambig bias")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1, label=f"base={baseline:.4f}")
    ax.set_xlabel("k (number of layers injected, from L0)")
    ax.set_ylabel("Bias score")
    ax.set_title(f"{DISPLAY[alias]} — Cumulative injection (L0…Lk)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "cumulative" / f"cumulative_{alias}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_cumulative_grid():
    cumul_aliases = [a for a in ALIASES if (BASE / "cumulative" / a).exists()]
    ncols = 3
    nrows = (len(cumul_aliases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, alias in enumerate(cumul_aliases):
        rows = load_cumulative(alias)
        if not rows:
            axes[i].set_visible(False)
            continue
        ks, ambig, disambig = zip(*rows)
        baseline = BASELINES.get(alias)
        ax = axes[i]
        ax.plot(ks, ambig, "o-", color="#1f77b4", ms=3, lw=1.2, label="ambig")
        ax.plot(ks, disambig, "s--", color="#ff7f0e", ms=3, lw=1, label="disambig")
        if baseline is not None:
            ax.axhline(baseline, color="gray", ls=":", lw=0.8)
        ax.set_title(DISPLAY[alias], fontsize=9)
        ax.set_xlabel("k (layers injected)", fontsize=7)
        ax.set_ylabel("Bias", fontsize=7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cumulative injection (L0…Lk) — all models", fontsize=12)
    fig.tight_layout()
    out = OUT / "cumulative" / "cumulative_all_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_control_individual(alias):
    cond_data = load_control(alias)
    sweep_rows = load_sweep(alias)
    if not cond_data and not sweep_rows:
        print(f"[skip control] {alias}: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if sweep_rows:
        layers, ambig, _ = zip(*sweep_rows)
        n = max(layers) + 1
        _thirds_spans(n, ax)
        ax.plot(layers, ambig, "o-", color="#1f77b4", ms=4, lw=1.8, label="+ΔW (bias xfer)", zorder=5)

    ctrl_colors = {
        "norm_matched_seed0": "#2ca02c",
        "norm_matched_seed1": "#98df8a",
        "norm_matched_seed2": "#b5cf6b",
        "shuffled_seed0": "#d62728",
        "shuffled_seed1": "#ff9896",
        "zero": "#7f7f7f",
    }
    for cond, rows in sorted(cond_data.items()):
        ly, am = zip(*rows)
        c = ctrl_colors.get(cond, "#aec7e8")
        ls = "--" if "seed" in cond else ":"
        ax.plot(ly, am, ls, color=c, ms=3, lw=1.2, label=f"ctrl_{cond}")

    baseline = BASELINES.get(alias)
    if baseline is not None:
        ax.axhline(baseline, color="black", ls=":", lw=1, label=f"base={baseline:.4f}")

    n_layers = max(layers) + 1 if sweep_rows else 1
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Ambig bias score")
    ax.set_title(f"{DISPLAY[alias]} — Control comparison")
    _set_layer_xticks(ax, n_layers)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "control" / f"control_{alias}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_necessity_bar():
    """Bar chart of group contributions per model from necessity_posthoc."""
    nec_dir = BASE / "necessity_posthoc"
    files = sorted(glob.glob(str(nec_dir / "*_necessity.json")))
    if not files:
        print("[skip necessity bar] no files")
        return

    aliases_found = []
    first_vals, mid_vals, last_vals = [], [], []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        gc = d["group_contributions"]
        aliases_found.append(d["alias"])
        first_vals.append(gc["first"]["mean_delta"])
        mid_vals.append(gc["mid"]["mean_delta"])
        last_vals.append(gc["last"]["mean_delta"])

    x = np.arange(len(aliases_found))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, first_vals, w, label="first third", color="#1f77b4", alpha=0.85)
    ax.bar(x,     mid_vals,   w, label="mid third",   color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, last_vals,  w, label="last third",  color="#2ca02c", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(a, a) for a in aliases_found], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean delta (baseline − layer_bias)")
    ax.set_title("Post-hoc necessity: mean Δbias per third")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT / "necessity" / "necessity_bar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_necessity_individual(alias):
    nec_file = BASE / "necessity_posthoc" / f"{alias}_necessity.json"
    if not nec_file.exists():
        print(f"[skip necessity] {alias}: no file")
        return
    with open(nec_file) as f:
        d = json.load(f)
    layers_data = d["layers"]
    if not layers_data:
        return
    idx = [r["layer_idx"] for r in layers_data]
    delta = [r["delta"] for r in layers_data]
    baseline = d["baseline_ambig_bias"]
    n = d["n_layers"]

    fig, ax = plt.subplots(figsize=(8, 4))
    _thirds_spans(n, ax)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in delta]
    ax.bar(idx, delta, color=colors, alpha=0.8, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Δbias (baseline − layer_bias)")
    ax.set_title(f"{DISPLAY[alias]} — Per-layer bias delta (baseline={baseline:.4f})")
    _set_layer_xticks(ax, n)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT / "necessity" / f"necessity_{alias}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# main

SECTIONS = ["sweep", "signflip", "cumulative", "control", "necessity"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=SECTIONS, help="Sections to run (default: all)")
    args = parser.parse_args()
    run = set(args.only) if args.only else set(SECTIONS)

    OUT.mkdir(parents=True, exist_ok=True)

    if "sweep" in run:
        print("\n=== Sweep plots ===")
        for alias in ALIASES:
            plot_sweep_individual(alias)
        plot_sweep_grid()

    if "signflip" in run:
        print("\n=== Signflip plots ===")
        sf_aliases = [a for a in ALIASES if (BASE / "signflip" / a).exists()]
        for alias in sf_aliases:
            plot_signflip_individual(alias)
        plot_signflip_grid()

    if "cumulative" in run:
        print("\n=== Cumulative plots ===")
        cumul_aliases = [a for a in ALIASES if (BASE / "cumulative" / a).exists()]
        for alias in cumul_aliases:
            plot_cumulative_individual(alias)
        plot_cumulative_grid()

    if "control" in run:
        print("\n=== Control plots ===")
        ctrl_aliases = [a for a in ALIASES if (BASE / "control_sweep" / a).exists()]
        for alias in ctrl_aliases:
            plot_control_individual(alias)

    if "necessity" in run:
        print("\n=== Necessity plots ===")
        for alias in ALIASES:
            plot_necessity_individual(alias)
        plot_necessity_bar()

    print("\nDone.")


if __name__ == "__main__":
    main()
