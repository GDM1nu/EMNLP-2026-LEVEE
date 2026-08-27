#!/usr/bin/env python3
"""
Cumulative layer sweep: apply diff to layers 0..k for each k, evaluate BBQ.

Unlike the single-layer sweep (one layer at a time), this accumulates:
  k=0  → diff applied to {L0}
  k=1  → diff applied to {L0, L1}
  ...
  k=N  → diff applied to {L0, ..., LN}  (= full diff)

Model weights are modified in-place incrementally (one new layer added per step),
avoiding repeated apply/unapply cycles.
"""
from __future__ import annotations
from pathlib import Path

import argparse
import gc
import json
import os
import sys
from datetime import datetime

import torch
import transformers

PROJ_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE   = PROJ_ROOT.parent / "archive"  # path to model weight storage
ARCHIVE   = PROJ_ROOT.parent / "archive"  # model weight storage

# reuse archive utilities
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


def main():
    p = argparse.ArgumentParser(description="Cumulative layer sweep (L0..k)")
    p.add_argument("--model",      required=True)
    p.add_argument("--diff",       required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--max_length", type=int,   default=512)
    p.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--alpha",      type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)
    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu,
    )

    print(f"[cumulative_sweep] model={args.model}  n_layers={n_layers}")
    print("[preload] BBQ...")
    bbq_data = load_bbq_hf()
    print(f"[preload] BBQ: {len(bbq_data)} samples")

    env = dict(torch=torch.__version__,
               transformers=transformers.__version__,
               cuda=str(torch.version.cuda))

    all_results = []

    for k in range(n_layers):
        out_json = os.path.join(args.output_dir, f"cumul_L{k:02d}_bbq.json")
        if os.path.exists(out_json):
            print(f"[skip] cumul_L{k:02d}")
            # still need to apply this layer to keep model state consistent
            apply_delta(model, diff, {k}, alpha=args.alpha, subtract=False)
            continue

        # apply diff for layer k only (layers 0..k-1 already applied)
        apply_stats = apply_delta(model, diff, {k}, alpha=args.alpha, subtract=False)

        print(f"\n>>> cumul_L{k:02d}/{n_layers-1}"
              f"  new_params={apply_stats['applied_params']}")

        results, trunc = evaluate_batched_logits(
            model, tok, bbq_data, device, args.batch_size, args.max_length)
        metrics = bbq_metrics(results)
        bbq_print(metrics, f"cumul_L{k:02d}")

        payload = dict(
            exp=f"cumul_L{k:02d}_bbq",
            model=args.model, model_path=model_path,
            n_layers=n_layers,
            cumul_k=k,
            layers_applied=list(range(k + 1)),
            alpha=args.alpha,
            diff_sha256=diff_sha,
            metrics=metrics,
            trunc_stats=trunc,
            apply_stats=apply_stats,
            timestamp=datetime.now().isoformat(),
            env=env,
        )
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  → {out_json}")

        all_results.append(dict(
            k=k,
            ambig_bias=metrics.get("ambig_bias"),
            disambig_bias=metrics.get("disambig_bias"),
            accuracy=metrics.get("accuracy"),
        ))

    summary_path = os.path.join(args.output_dir, "cumulative_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha,
            diff_sha256=diff_sha,
            timestamp=datetime.now().isoformat(), env=env,
            layers=all_results,
        ), f, indent=2, ensure_ascii=False)
    print(f"\n[summary] → {summary_path}")

    del model, tok, diff
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
