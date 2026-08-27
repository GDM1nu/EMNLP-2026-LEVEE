"""
Generate missing layer_index/ visualizations:
  - l2norm: per-model per-layer, overlay, thirds bar
  - scaling: delta vs params, peak layer vs params
  - probing: LLaMA 3-8B BBQ Age per-layer accuracy
  - cosine: cross-model correlation heatmap
  - efficiency: single-layer injection coverage bar

Output: results/analysis/plots/layer_index/<section>/
"""
import json
import pickle
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = Path("results")
OUT  = Path("results/analysis/plots/layer_index")
OUT.mkdir(parents=True, exist_ok=True)

ALIASES = [
    "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_14b",
    "gemma_2b", "gemma_9b",
    "llama32_1b", "llama32_3b", "llama3_8b",
]

DISPLAY = {
    "qwen_0.5b":  "Qwen2.5 0.5B",
    "qwen_1.5b":  "Qwen2.5 1.5B",
    "qwen_3b":    "Qwen2.5 3B",
    "qwen_14b":   "Qwen2.5 14B",
    "gemma_2b":   "Gemma3 2B",
    "gemma_9b":   "Gemma3 9B",
    "llama32_1b": "LLaMA3.2 1B",
    "llama32_3b": "LLaMA3.2 3B",
    "llama3_8b":  "LLaMA3 8B",
}

COLORS = {
    "qwen_0.5b":  "#1f77b4",
    "qwen_1.5b":  "#aec7e8",
    "qwen_3b":    "#ff7f0e",
    "qwen_14b":   "#ffbb78",
    "gemma_2b":   "#2ca02c",
    "gemma_9b":   "#98df8a",
    "llama32_1b": "#d62728",
    "llama32_3b": "#ff9896",
    "llama3_8b":  "#9467bd",
}

FAMILY_COLOR = {
    "qwen":   "#2171b5",
    "gemma":  "#238b45",
    "llama3": "#d94801",
    "llama32": "#fd8d3c",
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

SIZE_B = {
    "qwen_0.5b":  0.5,
    "qwen_1.5b":  1.5,
    "qwen_3b":    3.0,
    "qwen_14b":   14.0,
    "gemma_2b":   2.0,
    "gemma_9b":   9.0,
    "llama32_1b": 1.0,
    "llama32_3b": 3.0,
    "llama3_8b":  8.0,
}


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
    ax.axvspan(-0.5, t1 - 0.5,          color="#1f77b4", alpha=alpha)
    ax.axvspan(t1 - 0.5, t2 - 0.5,     color="#ff7f0e", alpha=alpha)
    ax.axvspan(t2 - 0.5, n_layers - 0.5, color="#2ca02c", alpha=alpha)


def get_family(alias):
    if alias.startswith("qwen"):   return "qwen"
    if alias.startswith("gemma"):  return "gemma"
    if alias == "llama3_8b":       return "llama3"
    return "llama32"


# 1. L2 norm

def plot_l2norm():
    data_path = BASE.parent / "results/analysis/l2norm_thirds.pkl"
    if not data_path.exists():
        data_path = Path("results/analysis/l2norm_thirds.pkl")
    if not data_path.exists():
        print("[skip l2norm] l2norm_thirds.pkl not found")
        return
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    out_dir = OUT / "l2norm"
    out_dir.mkdir(parents=True, exist_ok=True)

    THIRD_COLORS = {"first_third": "#4C72B0", "mid_third": "#DD8452", "last_third": "#55A868"}
    THIRD_LABELS = {"first_third": "First third", "mid_third": "Mid third", "last_third": "Last third"}

    # 1a. Per-model per-layer line plot (layer index x-axis)
    for alias, d in data.items():
        n = d["n_layers"]
        norms = d["norms"]
        thirds = d["thirds"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for third, color in THIRD_COLORS.items():
            idx = thirds[third]
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, alpha=0.12, color=color,
                       label=THIRD_LABELS[third])
        ax.plot(range(n), norms, color="black", lw=1.2, zorder=3)
        ax.scatter(range(n), norms, color="black", s=18, zorder=4)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("ΔW L2 norm")
        ax.set_title(f"{DISPLAY.get(alias, alias)} — ΔW L2 norm per layer")
        _set_layer_xticks(ax, n)
        ax.set_xlim(-0.5, n - 0.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"l2norm_{alias}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved: {out}")

    # 1b. Overlay (layer index, not normalized depth)
    fig, ax = plt.subplots(figsize=(10, 5))
    for alias, d in data.items():
        norms = np.array(d["norms"])
        n = d["n_layers"]
        mn, mx = norms.min(), norms.max()
        normed = (norms - mn) / (mx - mn + 1e-12)
        ax.plot(range(n), normed, alpha=0.75, lw=1.5,
                color=COLORS.get(alias, "#333333"),
                label=DISPLAY.get(alias, alias))
    ax.set_xlabel("Layer index (absolute)")
    ax.set_ylabel("Normalized L2 norm (0–1 per model)")
    ax.set_title("ΔW L2 norm per layer — all models (min-max normalized)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "l2norm_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")

    # 1c. Thirds bar chart
    aliases = list(data.keys())
    x = np.arange(len(aliases))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (third, color) in enumerate(THIRD_COLORS.items()):
        vals = [data[a]["third_norms"][third] for a in aliases]
        ax.bar(x + i * w, vals, w, label=THIRD_LABELS[third], color=color, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([DISPLAY.get(a, a) for a in aliases], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean ΔW L2 norm")
    ax.set_title("Mean ΔW L2 norm per third — all models")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "l2norm_thirds_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# 2. Scaling

def _load_sweep_stats():
    rows = []
    for alias in ALIASES:
        d = BASE / "sweep_thirds" / alias
        files = sorted(glob.glob(str(d / "sweep_L*_bbq.json")))
        if not files:
            continue
        layer_data = []
        for f in files:
            with open(f) as fp:
                rec = json.load(fp)
            layer_data.append((rec["layer_idx"], rec["metrics"]["ambig"]["bias_score"]))
        layer_data.sort(key=lambda x: x[0])
        layers, biases = zip(*layer_data)
        best_idx = int(np.argmin(biases))
        n = max(layers) + 1
        peak_layer = layers[best_idx]
        base = BASELINES.get(alias)
        rows.append({
            "alias": alias,
            "size_b": SIZE_B[alias],
            "n_layers": n,
            "peak_layer": peak_layer,
            "min_bias": biases[best_idx],
            "base_bias": base,
            "delta": biases[0] - biases[best_idx],  # max - min approx
            "peak_reduction": (base - biases[best_idx]) if base else None,
            "family": get_family(alias),
        })
    return rows


def plot_scaling():
    stats = _load_sweep_stats()
    if not stats:
        print("[skip scaling] no sweep data")
        return
    out_dir = OUT / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    families_seen = set()
    for s in sorted(stats, key=lambda x: x["size_b"]):
        fam = s["family"]
        fc = FAMILY_COLOR[fam]
        lbl = fam.capitalize() if fam not in families_seen else None
        families_seen.add(fam)

        # Left: peak reduction vs size
        if s["peak_reduction"] is not None:
            ax1.scatter(s["size_b"], s["peak_reduction"],
                        color=fc, s=120, zorder=5, edgecolors="white", lw=0.8, label=lbl)
            ax1.annotate(DISPLAY.get(s["alias"], s["alias"]).split()[-1],
                         (s["size_b"], s["peak_reduction"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8)

        # Right: peak layer index vs size
        ax2.scatter(s["size_b"], s["peak_layer"],
                    color=fc, s=120, zorder=5, edgecolors="white", lw=0.8,
                    label=lbl if s["peak_reduction"] is None else None)
        third = ("first" if s["peak_layer"] < s["n_layers"] // 3
                 else "mid" if s["peak_layer"] < 2 * s["n_layers"] // 3
                 else "last")
        ax2.annotate(f"L{s['peak_layer']}\n({third})",
                     (s["size_b"], s["peak_layer"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)

    # trend line (≥3B)
    red_data = [(s["size_b"], s["peak_reduction"]) for s in stats
                if s["peak_reduction"] is not None and s["size_b"] >= 3.0]
    if len(red_data) >= 3:
        xs = np.log([d[0] for d in red_data])
        ys = [d[1] for d in red_data]
        coef = np.polyfit(xs, ys, 1)
        x_range = np.linspace(min(xs), max(xs), 50)
        ax1.plot(np.exp(x_range), np.polyval(coef, x_range),
                 color="gray", ls="--", lw=1, alpha=0.6, label="Trend (≥3B)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Model size (B params, log scale)", fontsize=11)
    ax1.set_ylabel("Peak bias reduction (base − min_sweep)", fontsize=11)
    ax1.set_title("A. Peak Suppression Magnitude vs. Model Size", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.set_xscale("log")
    ax2.set_xlabel("Model size (B params, log scale)", fontsize=11)
    ax2.set_ylabel("Peak suppression layer index", fontsize=11)
    ax2.set_title("B. Peak Suppression Layer (absolute index) vs. Model Size", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Scaling Analysis: Task Vector Bias Suppression", fontsize=13)
    fig.tight_layout()
    out = out_dir / "scaling_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")

    # Also: delta vs size scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in sorted(stats, key=lambda x: x["size_b"]):
        fam = s["family"]
        ax.scatter(s["size_b"], s["peak_reduction"] if s["peak_reduction"] else s["min_bias"] * 0,
                   color=FAMILY_COLOR[fam], s=100, zorder=5, edgecolors="white", lw=0.8)
        if s["peak_reduction"] is not None:
            ax.annotate(DISPLAY.get(s["alias"], s["alias"]),
                        (s["size_b"], s["peak_reduction"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
    handles = [mpatches.Patch(color=v, label=k) for k, v in FAMILY_COLOR.items()]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B params)", fontsize=11)
    ax.set_ylabel("Peak bias reduction (base − min_sweep)", fontsize=11)
    ax.set_title("Peak bias suppression vs. model size (layer-index basis)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "scaling_reduction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# 3. Probing accuracy

def plot_probing():
    probe_file = BASE / "probing" / "llama3_8b_bbq_age_probe.json"
    if not probe_file.exists():
        print("[skip probing] file not found")
        return
    with open(probe_file) as f:
        d = json.load(f)
    results = d["results"]
    layers = [r["layer_idx"] for r in results]
    accs   = [r["accuracy"] for r in results]
    n = max(layers) + 1
    chance = 1.0 / 3.0  # 3-class

    out_dir = OUT / "probing"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _thirds_spans(n, ax)
    ax.plot(layers, accs, "o-", color="#9467bd", ms=5, lw=1.8, label="probe accuracy")
    ax.axhline(chance, color="gray", ls="--", lw=1.2, label=f"chance ({chance:.3f})")
    best_l = layers[int(np.argmax(accs))]
    best_a = max(accs)
    ax.annotate(f"L{best_l}\n({best_a:.3f})",
                xy=(best_l, best_a), xytext=(best_l + 1, best_a + 0.005),
                fontsize=8, color="#9467bd",
                arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.8))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Probe accuracy (3-class)")
    ax.set_title("LLaMA 3-8B — BBQ Age probe accuracy per layer\n(correct_answer_3class, n=500)")
    _set_layer_xticks(ax, n)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.25, 0.45)
    fig.tight_layout()
    out = out_dir / "probing_llama3_8b_bbq_age.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# 4. Cross-model correlation heatmap

def plot_cosine_heatmap():
    cos_file = Path("results/analysis/cosine_sim.json")
    if not cos_file.exists():
        print("[skip cosine heatmap] cosine_sim.json not found")
        return
    with open(cos_file) as f:
        data = json.load(f)
    out_dir = OUT / "cosine"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model Pearson r(norm, bias) bar
    aliases = [a for a in data.keys() if not a.startswith("_")]
    r_norm  = [data[a]["correlations"]["pearson_norm_bbq"] for a in aliases]
    p_norm  = [data[a]["correlations"]["pearson_norm_bbq_p"] for a in aliases]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_colors = ["#d62728" if r < 0 else "#1f77b4" for r in r_norm]
    bars = ax.bar(range(len(aliases)), r_norm, color=bar_colors, alpha=0.8)
    for i, (r, p) in enumerate(zip(r_norm, p_norm)):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(i, r + (0.01 if r >= 0 else -0.02), sig, ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(aliases)))
    ax.set_xticklabels([DISPLAY.get(a, a) for a in aliases], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (ΔW L2 norm vs bias suppression)")
    ax.set_title("Correlation: ΔW L2 norm vs. per-layer bias suppression\n(* p<0.05, ** p<0.01)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "cosine_norm_bias_corr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")

    # Cross-model BBQ curve correlation heatmap (from RESULTS_DEEP data)
    # Using hardcoded values from RESULTS_DEEP.md section 5
    aliases_cm = [
        "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_14b",
        "gemma_2b", "gemma_9b",
        "llama32_1b", "llama32_3b", "llama3_8b",
    ]
    # r matrix (symmetric, diagonal=1)
    # order matches aliases_cm
    r_vals = {
        ("qwen_0.5b","qwen_1.5b"): 0.399,
        ("qwen_0.5b","qwen_3b"):   0.478,
        ("qwen_0.5b","qwen_14b"):  None,
        ("qwen_0.5b","gemma_2b"):  0.458,
        ("qwen_0.5b","gemma_9b"):  0.786,
        ("qwen_0.5b","llama32_1b"):0.188,
        ("qwen_0.5b","llama32_3b"):0.760,
        ("qwen_0.5b","llama3_8b"): 0.427,
        ("qwen_1.5b","qwen_3b"):   0.441,
        ("qwen_1.5b","qwen_14b"):  None,
        ("qwen_1.5b","gemma_2b"):  0.073,
        ("qwen_1.5b","gemma_9b"):  0.339,
        ("qwen_1.5b","llama32_1b"):0.047,
        ("qwen_1.5b","llama32_3b"):0.372,
        ("qwen_1.5b","llama3_8b"): 0.531,
        ("qwen_3b","qwen_14b"):    None,
        ("qwen_3b","gemma_2b"):    0.302,
        ("qwen_3b","gemma_9b"):    0.262,
        ("qwen_3b","llama32_1b"):  0.566,
        ("qwen_3b","llama32_3b"):  0.258,
        ("qwen_3b","llama3_8b"):   0.640,
        ("qwen_14b","gemma_2b"):   None,
        ("qwen_14b","gemma_9b"):   None,
        ("qwen_14b","llama32_1b"): None,
        ("qwen_14b","llama32_3b"): None,
        ("qwen_14b","llama3_8b"):  None,
        ("gemma_2b","gemma_9b"):   0.363,
        ("gemma_2b","llama32_1b"):-0.103,
        ("gemma_2b","llama32_3b"): 0.619,
        ("gemma_2b","llama3_8b"):  0.281,
        ("gemma_9b","llama32_1b"): 0.014,
        ("gemma_9b","llama32_3b"): 0.671,
        ("gemma_9b","llama3_8b"):  0.483,
        ("llama32_1b","llama32_3b"):0.051,
        ("llama32_1b","llama3_8b"):0.145,
        ("llama32_3b","llama3_8b"):0.295,
    }

    n = len(aliases_cm)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    alias2idx = {a: i for i, a in enumerate(aliases_cm)}
    for (a, b), r in r_vals.items():
        if r is None:
            continue
        i, j = alias2idx[a], alias2idx[b]
        mat[i, j] = r
        mat[j, i] = r

    fig, ax = plt.subplots(figsize=(9, 8))
    # Use only finite values for colormap range
    finite = mat[np.isfinite(mat)]
    vmin, vmax = -0.2, 0.9
    im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                val = mat[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
            elif i == j:
                pass
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#999999")

    labels = [DISPLAY.get(a, a).replace(" ", "\n") for a in aliases_cm]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Cross-model BBQ sweep curve correlation (Pearson r)", fontsize=11)
    fig.tight_layout()
    out = out_dir / "cross_model_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# 5. Coverage efficiency

def plot_efficiency():
    # Data from RESULTS.md single-layer injection efficiency table
    # (base→instruct gap, sweep best reduction, coverage ratio)
    eff_data = [
        {"alias": "qwen_0.5b",  "base_instruct": 0.173, "sweep_red": 0.017, "coverage": 0.10},
        {"alias": "qwen_3b",    "base_instruct": 0.050, "sweep_red": 0.023, "coverage": 0.45},
        {"alias": "qwen_14b",   "base_instruct": 0.009, "sweep_red": 0.009, "coverage": 1.00},
        {"alias": "gemma_2b",   "base_instruct": 0.144, "sweep_red": 0.035, "coverage": 0.24},
        {"alias": "gemma_9b",   "base_instruct": 0.116, "sweep_red": 0.031, "coverage": 0.27},
        {"alias": "llama3_8b",  "base_instruct": 0.067, "sweep_red": 0.089, "coverage": 1.33},
    ]
    labels = [DISPLAY.get(d["alias"], d["alias"]) for d in eff_data]
    coverages = [d["coverage"] for d in eff_data]
    colors = ["#d62728" if c > 1.0 else "#2ca02c" if c >= 0.5 else "#1f77b4"
              for c in coverages]

    out_dir = OUT / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), coverages, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="100% coverage (= instruct effect)")
    for i, (bar, c) in enumerate(zip(bars, coverages)):
        ax.text(i, c + 0.02, f"{c:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Single-layer injection coverage\n(sweep best reduction / base-instruct gap)")
    ax.set_title("Single-layer injection efficiency: coverage of base→instruct gap")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.55)
    ax.grid(True, alpha=0.3, axis="y")
    p1 = mpatches.Patch(color="#1f77b4", alpha=0.85, label="<50%")
    p2 = mpatches.Patch(color="#2ca02c", alpha=0.85, label="50–100%")
    p3 = mpatches.Patch(color="#d62728", alpha=0.85, label=">100% (exceeds instruct)")
    ax.legend(handles=[p1, p2, p3], fontsize=8, loc="upper left")
    fig.tight_layout()
    out = out_dir / "efficiency_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")

    # Grouped bar: base→instruct vs sweep reduction
    x = np.arange(len(eff_data))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [d["base_instruct"] for d in eff_data], w,
           label="Base → Instruct gap", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, [d["sweep_red"] for d in eff_data], w,
           label="Single-layer sweep best reduction", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Bias score reduction (ambig)")
    ax.set_title("Base→Instruct gap vs. single-layer sweep best reduction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "efficiency_grouped_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# 6. Sweep overlay (compact, paper-ready)

def plot_sweep_overlay_compact():
    """Overlay of ambig bias for all models, normalized to [0,1] per model."""
    out_dir = OUT / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for alias in ALIASES:
        d = BASE / "sweep_thirds" / alias
        files = sorted(glob.glob(str(d / "sweep_L*_bbq.json")))
        if not files:
            continue
        layer_data = []
        for f in files:
            with open(f) as fp:
                rec = json.load(fp)
            layer_data.append((rec["layer_idx"], rec["metrics"]["ambig"]["bias_score"]))
        layer_data.sort()
        layers, biases = zip(*layer_data)
        n = max(layers) + 1
        biases = np.array(biases)
        mn, mx = biases.min(), biases.max()
        normed = (biases - mn) / (mx - mn + 1e-12)
        xs = [l / (n - 1) for l in layers]
        ax.plot(xs, normed, alpha=0.75, lw=1.5,
                color=COLORS.get(alias, "#333"), label=DISPLAY.get(alias, alias))

    ax.axvline(1/3, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(2/3, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvspan(0, 1/3, alpha=0.05, color="#1f77b4")
    ax.axvspan(1/3, 2/3, alpha=0.05, color="#ff7f0e")
    ax.axvspan(2/3, 1.0, alpha=0.05, color="#2ca02c")
    ax.set_xlabel("Relative layer depth (0=first, 1=last)")
    ax.set_ylabel("Normalized ambig bias (0=min, 1=max per model)")
    ax.set_title("Layer-wise bias suppression — all 9 models (normalized overlay)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "sweep_overlay_normalized.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


# main

if __name__ == "__main__":
    print("=== L2 norm plots ===")
    plot_l2norm()

    print("\n=== Scaling plots ===")
    plot_scaling()

    print("\n=== Probing plot ===")
    plot_probing()

    print("\n=== Cosine heatmap ===")
    plot_cosine_heatmap()

    print("\n=== Efficiency plots ===")
    plot_efficiency()

    print("\n=== Sweep overlay normalized ===")
    plot_sweep_overlay_compact()

    print("\nAll done.")
