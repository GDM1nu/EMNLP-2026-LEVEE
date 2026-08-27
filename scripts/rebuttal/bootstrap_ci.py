#!/usr/bin/env python
"""
C1 — Stratified multinomial bootstrap CIs for BBQ ambig bias_score (GPU-free).

Per-item predictions were not saved, but every result JSON stores per-category
ambig counts (n_biased / n_anti_biased / n_unknown / total). Each ambig item
falls into exactly one outcome class, so the empirical per-category outcome
distribution is a multinomial — resampling it stratified by category is
equivalent to a nonparametric bootstrap over items.

Caveat (state this in the rebuttal): condition contrasts (e.g. minus_mid vs
full) are computed from *independent* bootstrap draws. The two conditions
share the same item set, so the true paired variance is smaller — these
difference CIs are conservative (wider). If a conservative CI already
excludes 0, the claim is safe.

Usage:
    python scripts/rebuttal2/bootstrap_ci.py                # scan block_removal + base
    python scripts/rebuttal2/bootstrap_ci.py --B 20000
    python scripts/rebuttal2/bootstrap_ci.py --models qwen_3b llama3_8b

Output:
    results/rebuttal2/bootstrap_ci.json   (machine-readable)
    results/rebuttal2/bootstrap_ci.md     (rebuttal-ready tables)
"""

import os as _os
from pathlib import Path as _Path
# Repository root; override with LEVEE_ROOT if the layout differs.
REPO_ROOT = _Path(_os.environ.get("LEVEE_ROOT", _Path(__file__).resolve().parents[2]))
# Directory holding pre-computed results (see results/README.md).
RESULTS_DIR = _Path(_os.environ.get("LEVEE_RESULTS", REPO_ROOT / "results"))
# Directory holding model weights / diff vectors.
WEIGHTS_DIR = _Path(_os.environ.get("LEVEE_WEIGHTS", REPO_ROOT / "weights"))
# BBQ jsonl (58,492 items). Download instructions in README.
BBQ_JSONL = _Path(_os.environ.get("LEVEE_BBQ", REPO_ROOT / "data" / "bbq_all.jsonl"))

import argparse
import glob
import json
import os

import numpy as np

BASE = str(REPO_ROOT)
BLOCK_DIR = os.path.join(BASE, "results", "block_removal")
CTRL_DIR = os.path.join(BASE, "results", "control_sweep")
OUT_DIR = os.path.join(BASE, "results", "rebuttal2")

# block_removal condition files (main paper Table 4/5 + audit)
CONDITIONS = {
    "full": "block_full_bbq.json",
    "minus_first": "block_minus_first_bbq.json",
    "minus_mid": "block_minus_mid_bbq.json",   # == outer-only (audit stage b)
    "minus_last": "block_minus_last_bbq.json",
    "only_first": "block_only_first_bbq.json",
    "only_mid": "block_only_mid_bbq.json",     # audit stage c
    "only_last": "block_only_last_bbq.json",
}

# contrasts quoted in the paper / audit protocol: (a, b) -> CI of bias(a) - bias(b)
CONTRASTS = [
    ("minus_mid", "full"),        # necessity: removing mid raises bias
    ("minus_first", "full"),
    ("minus_last", "full"),
    ("minus_mid", "base"),        # active suppression / audit fail criterion
    ("minus_mid", "minus_first"), # mid removal dominates outer removal
    ("minus_mid", "minus_last"),
    ("only_mid", "base"),         # mid-only sufficiency
]


def find_bbq_metrics(d):
    """Handle both result shapes: {'metrics': {...}} and {'benchmarks': {'bbq': {...}}}."""
    if "metrics" in d and "ambig" in d.get("metrics", {}):
        return d["metrics"]
    if "benchmarks" in d and "bbq" in d["benchmarks"]:
        return d["benchmarks"]["bbq"]
    raise KeyError("no BBQ metrics block found")


def ambig_counts_by_category(metrics):
    """(C, 4) int array of [n_biased, n_anti_biased, n_unknown, n_other] per category."""
    rows = []
    for cat in sorted(metrics["by_category"]):
        a = metrics["by_category"][cat]["ambig"]
        n_b, n_a, n_u, tot = a["n_biased"], a["n_anti_biased"], a["n_unknown"], a["total"]
        rows.append((n_b, n_a, n_u, tot - n_b - n_a - n_u))
    arr = np.array(rows, dtype=np.int64)
    assert (arr >= 0).all(), "negative residual outcome count — check by_category integrity"
    return arr


def bias_score_from_totals(tot4):
    """tot4: (..., 4) summed counts -> ambig bias_score, replicating run_eval formula."""
    n_b, n_a, n_u = tot4[..., 0], tot4[..., 1], tot4[..., 2]
    total = tot4.sum(-1)
    denom = n_b + n_a + n_u
    acc = n_u / total
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (n_b - n_a) / denom * (1 - acc), 0.0)


def bootstrap(counts, B, rng):
    """counts: (C, 4). Returns (B,) bootstrap bias_scores, stratified by category."""
    totals = np.zeros((B, 4))
    for c in range(counts.shape[0]):
        n = int(counts[c].sum())
        if n == 0:
            continue
        p = counts[c] / n
        totals += rng.multinomial(n, p, size=B)
    return bias_score_from_totals(totals)


def point_estimate(counts):
    return float(bias_score_from_totals(counts.sum(0).astype(float)))


def load_base_counts(alias):
    """Base model = ctrl_zero (zero diff). Any layer file is the same base model.

    NOTE (C6, 2026-07-08): ctrl_zero is ~0.001-0.002 higher than the paper's
    'base' column for Qwen-family + Gemma-9B (the paper base is a direct
    base eval, not the zero-diff injection round-trip). Audit Pass/Fail and
    all -mid deltas are identical under both, but do NOT quote this base CI
    next to the paper's base value — they will disagree at the 3rd decimal.
    """
    files = sorted(glob.glob(os.path.join(CTRL_DIR, alias, "ctrl_zero_L*_bbq.json")))
    if not files:
        return None
    with open(files[0]) as f:
        return ambig_counts_by_category(find_bbq_metrics(json.load(f)))


def ci(samples, level=0.95):
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    aliases = args.models or sorted(os.listdir(BLOCK_DIR))
    report = {}
    md = ["# Bootstrap 95% CIs — BBQ ambig bias_score",
          f"\nB={args.B}, stratified by category, multinomial over outcome classes.",
          "Contrast CIs use independent draws → conservative (see script docstring).\n"]

    for alias in aliases:
        mdir = os.path.join(BLOCK_DIR, alias)
        if not os.path.isdir(mdir):
            continue
        cond_counts, cond_boot, entry = {}, {}, {"conditions": {}, "contrasts": {}}

        base = load_base_counts(alias)
        if base is not None:
            cond_counts["base"] = base

        for cond, fname in CONDITIONS.items():
            path = os.path.join(mdir, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                cond_counts[cond] = ambig_counts_by_category(find_bbq_metrics(json.load(f)))

        for cond, counts in cond_counts.items():
            boot = bootstrap(counts, args.B, rng)
            cond_boot[cond] = boot
            lo, hi = ci(boot)
            entry["conditions"][cond] = {
                "bias_score": point_estimate(counts), "ci95": [lo, hi]}

        for a, b in CONTRASTS:
            if a in cond_boot and b in cond_boot:
                diff = cond_boot[a] - cond_boot[b]
                lo, hi = ci(diff)
                pt = entry["conditions"][a]["bias_score"] - entry["conditions"][b]["bias_score"]
                entry["contrasts"][f"{a} - {b}"] = {
                    "delta": pt, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}

        report[alias] = entry

        md.append(f"\n## {alias}\n")
        md.append("| condition | bias_score | 95% CI |")
        md.append("|---|---|---|")
        for cond in ["base"] + list(CONDITIONS):
            if cond in entry["conditions"]:
                e = entry["conditions"][cond]
                md.append(f"| {cond} | {e['bias_score']:.4f} | [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}] |")
        if entry["contrasts"]:
            md.append("\n| contrast | Δ | 95% CI | excludes 0 |")
            md.append("|---|---|---|---|")
            for name, e in entry["contrasts"].items():
                mark = "**yes**" if e["excludes_zero"] else "no"
                md.append(f"| {name} | {e['delta']:+.4f} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {mark} |")

    with open(os.path.join(OUT_DIR, "bootstrap_ci.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(OUT_DIR, "bootstrap_ci.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"wrote {OUT_DIR}/bootstrap_ci.{{json,md}} ({len(report)} models)")
    for alias, entry in report.items():
        key = "minus_mid - full"
        if key in entry["contrasts"]:
            e = entry["contrasts"][key]
            flag = "SIG" if e["excludes_zero"] else "n.s."
            print(f"  {alias:12s} minus_mid-full Δ={e['delta']:+.4f} "
                  f"CI=[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}] {flag}")


if __name__ == "__main__":
    main()
