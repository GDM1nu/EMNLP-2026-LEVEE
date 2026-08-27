#!/usr/bin/env python3
"""
Block Removal experiment — tests necessity of layer groups.

Injects full ΔW minus a specific block, evaluates BBQ bias.

Conditions (all run from the same base model, apply→eval→unapply):
  full        → all N layers injected
  minus_first → all except first-third  [0,   N//3)
  minus_mid   → all except mid-third    [N//3, 2N//3)
  minus_last  → all except last-third   [2N//3, N)
  minus_best  → all except sweep_best layer  (requires --best_layer)

Interpretation:
  full → low bias (instruct-like).
  If minus_mid raises bias back toward baseline → mid is NECESSARY.
  Compare across thirds: which removal hurts most?

Output per condition: block_<cond>_bbq.json
Summary: block_summary.json
"""
from __future__ import annotations
from pathlib import Path

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from typing import Set

import numpy as np
import torch
import transformers

PROJ_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE   = PROJ_ROOT.parent / "archive"  # path to model weight storage
ARCHIVE   = PROJ_ROOT.parent / "archive"  # model weight storage

ARCHIVE_SCRIPTS = str(ARCHIVE / "scaling_experiment/scripts")
sys.path.insert(0, ARCHIVE_SCRIPTS)

from model_utils import (
    get_num_layers, load_model_and_tokenizer, resolve_model_path,
    analyze_matching, apply_delta, sha256_file,
)
from eval_bbq_shared import (
    load_bbq_hf, compute_metrics as bbq_metrics,
    print_results as bbq_print, evaluate_batched_logits,
)


def thirds(n: int) -> dict[str, Set[int]]:
    f = set(range(0, n // 3))
    m = set(range(n // 3, 2 * n // 3))
    l = set(range(2 * n // 3, n))
    return {"first": f, "mid": m, "last": l}


def run_condition(
    label: str,
    target: Set[int],
    model, tok, diff, diff_sha, bbq_data,
    device, args, n_layers, all_layers,
) -> dict | None:
    out_json = os.path.join(args.output_dir, f"block_{label}_bbq.json")
    if os.path.exists(out_json):
        print(f"[skip] {label}")
        with open(out_json) as f:
            return json.load(f)

    print(f"\n>>> {label}  target={len(target)}/{n_layers} layers", flush=True)

    match_stats = analyze_matching(model, diff, target)
    apply_stats = apply_delta(model, diff, target, alpha=args.alpha, subtract=False)
    print(f"  applied={apply_stats['applied_params']} params", flush=True)

    results, trunc = evaluate_batched_logits(
        model, tok, bbq_data, device, args.batch_size, args.max_length
    )
    metrics = bbq_metrics(results)
    bbq_print(metrics, label)

    # unapply immediately
    apply_delta(model, diff, target, alpha=args.alpha, subtract=True)

    excluded = sorted(all_layers - target)
    payload = dict(
        exp=f"block_{label}_bbq",
        model=args.model,
        n_layers=n_layers,
        condition=label,
        layers_injected=sorted(target),
        layers_excluded=excluded,
        alpha=args.alpha,
        diff_sha256=diff_sha,
        metrics=metrics,
        trunc_stats=trunc,
        match_stats=match_stats,
        apply_stats=apply_stats,
        timestamp=datetime.now().isoformat(),
        env=dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            cuda=str(torch.version.cuda),
        ),
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  → {out_json}", flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description="Block removal experiment (necessity test)")
    p.add_argument("--model",      required=True)
    p.add_argument("--diff",       required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--max_length", type=int,   default=512)
    p.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--alpha",      type=float, default=1.0)
    p.add_argument("--best_layer", type=int,   default=-1,
                   help="Sweep-best layer index for minus_best condition. -1 = skip.")
    p.add_argument("--seed",       type=int,   default=0,
                   help="Global random seed for reproducibility.")
    p.add_argument("--conditions", default="full,minus_first,minus_mid,minus_last,minus_best",
                   help="Comma-separated list of conditions to run.")
    args = p.parse_args()

    # determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    requested_conditions = set(args.conditions.split(","))

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)
    all_layers = set(range(n_layers))
    t = thirds(n_layers)

    print(f"[block_removal] model={args.model}  n_layers={n_layers}")
    print(f"  first: L{min(t['first'])}..L{max(t['first'])}  ({len(t['first'])} layers)")
    print(f"  mid:   L{min(t['mid'])}..L{max(t['mid'])}  ({len(t['mid'])} layers)")
    print(f"  last:  L{min(t['last'])}..L{max(t['last'])}  ({len(t['last'])} layers)")
    if args.best_layer >= 0:
        print(f"  best_layer: L{args.best_layer}")

    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu,
    )

    print("\n[preload] BBQ...", flush=True)
    bbq_data = load_bbq_hf()
    print(f"[preload] BBQ: {len(bbq_data)} samples", flush=True)

    common = dict(
        model=model, tok=tok, diff=diff, diff_sha=diff_sha,
        bbq_data=bbq_data, device=device, args=args,
        n_layers=n_layers, all_layers=all_layers,
    )

    all_conditions = [
        ("full",        all_layers),
        ("minus_first", all_layers - t["first"]),
        ("minus_mid",   all_layers - t["mid"]),
        ("minus_last",  all_layers - t["last"]),
        # group-level sufficiency: only one third injected
        ("only_first",  t["first"]),
        ("only_mid",    t["mid"]),
        ("only_last",   t["last"]),
    ]
    if args.best_layer >= 0:
        all_conditions.append(("minus_best", all_layers - {args.best_layer}))

    conditions = [(label, target) for label, target in all_conditions
                  if label in requested_conditions]

    all_results = []
    for label, target in conditions:
        r = run_condition(label, target, **common)
        if r:
            ambig = (r.get("metrics") or {}).get("ambig", {}).get("bias_score")
            all_results.append(dict(condition=label, ambig_bias=ambig,
                                    n_injected=len(target)))

    # summary
    print("\n=== Block Removal Summary ===")
    print(f"{'condition':<14} {'injected':>8} {'ambig_bias':>12}")
    print("-" * 38)
    for row in all_results:
        bs = row["ambig_bias"]
        bs_str = f"{bs:.4f}" if bs is not None else "  None"
        print(f"{row['condition']:<14} {row['n_injected']:>8} {bs_str:>12}")

    summary_path = os.path.join(args.output_dir, "block_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha,
            best_layer=args.best_layer,
            diff_sha256=diff_sha,
            timestamp=datetime.now().isoformat(),
            results=all_results,
        ), f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {summary_path}")

    del model, tok, diff
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
