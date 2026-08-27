#!/usr/bin/env python3
"""
Block-Only MMLU — capability check for group-sufficiency conditions.

For each "only_*" condition, injects ΔW at ONLY that third of layers
and measures MMLU capability. Complements block_removal_mmlu.py
(which removes one third from full injection) and block_removal BBQ.

Design:
  baseline   → no injection (base model)
  only_first → inject ΔW at first-third  [0,   N//3)
  only_mid   → inject ΔW at mid-third    [N//3, 2N//3)
  only_last  → inject ΔW at last-third   [2N//3, N)

Motivation:
  block_only BBQ showed mid-only achieves 99–110% of full debiasing effect.
  This script checks whether mid-only injection also preserves MMLU capability,
  establishing mid-third as both sufficient for bias reduction and safe for
  capability retention.

MMLU subset: 14/57 subcategories, same as run_block_removal_mmlu.py.
  STEM (4): abstract_algebra, high_school_mathematics,
            high_school_physics, high_school_computer_science
  Humanities (3): philosophy, high_school_world_history, moral_scenarios
  Social Sci (4): sociology, high_school_psychology,
                  high_school_government_and_politics, high_school_statistics
  Professional (3): professional_law, medical_genetics, global_facts

Reproducibility:
  --seed controls Python / NumPy / PyTorch / CUDA seeds.
  torch.backends.cudnn.deterministic = True, benchmark = False.
  CUBLAS_WORKSPACE_CONFIG=:4096:8 enforced for deterministic matmul.
  diff SHA-256 logged per run.

Output per condition: block_only_{cond}_mmlu.json
Summary: block_only_mmlu_summary.json

Usage:
  python run_block_only_mmlu.py \\
      --model Qwen/Qwen2.5-3B \\
      --diff  /path/to/diff_vector.pt \\
      --output_dir results/block_removal/qwen_3b \\
      --gpu 1 --batch_size 4 --seed 0
"""
from __future__ import annotations
from pathlib import Path

import argparse
import gc
import json
import os
import random
import subprocess
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
    apply_delta, sha256_file,
)

MMLU_SUBSET = [
    # STEM (4)
    "mmlu_abstract_algebra",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_physics",
    "mmlu_high_school_computer_science",
    # Humanities (3)
    "mmlu_philosophy",
    "mmlu_high_school_world_history",
    "mmlu_moral_scenarios",
    # Social Sciences (4)
    "mmlu_sociology",
    "mmlu_high_school_psychology",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_statistics",
    # Professional (3)
    "mmlu_professional_law",
    "mmlu_medical_genetics",
    "mmlu_global_facts",
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def thirds(n: int) -> dict[str, Set[int]]:
    return {
        "first": set(range(0, n // 3)),
        "mid":   set(range(n // 3, 2 * n // 3)),
        "last":  set(range(2 * n // 3, n)),
    }


def make_injected_lm(model, tokenizer, gpu: int, batch_size: int):
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.model import CacheHook

    class InjectedHFLM(HFLM):
        def __init__(self, model, tokenizer, gpu, batch_size):
            self._rank = 0
            self._world_size = 1
            self.cache_hook = CacheHook(None)
            self._model = model
            self._device = torch.device(f"cuda:{gpu}")
            self._config = model.config
            self.tokenizer = tokenizer
            self._max_length = getattr(model.config, "max_position_embeddings", 2048)
            self.custom_prefix_token_id = None
            self.batch_size_per_gpu = batch_size
            self.batch_sizes = {}
            self.batch_schedule = 1
            self.max_batch_size = 64
            self.add_bos_token = getattr(tokenizer, "add_bos_token", False)
            self.truncation = False
            self.logits_cache = True
            self.vocab_size = tokenizer.vocab_size
            self.pretrained = model
            self.delta = None
            self.peft = None
            self.revision = "main"
            self.think_end_token = None
            self.chat_template_args = None
            self.softmax_dtype = None
            self.mixed_precision_dtype = None

        def _detect_batch_size(self, requests=None, pos=0):
            return self.batch_size_per_gpu

    return InjectedHFLM(model, tokenizer, gpu, batch_size)


def eval_mmlu_subset(lm, tasks: list[str]) -> tuple[float | None, dict[str, float]]:
    from lm_eval import evaluator
    from lm_eval.tasks import get_task_dict, TaskManager




    tm = TaskManager()
    task_dict = get_task_dict(tasks, task_manager=tm)

    def _set_fewshot(d, n=5):
        for v in d.values():
            if isinstance(v, dict):
                _set_fewshot(v, n)
            elif hasattr(v, "config"):
                v.config.num_fewshot = n

    _set_fewshot(task_dict)
    results = evaluator.evaluate(lm=lm, task_dict=task_dict, log_samples=False)

    subtask_scores: dict[str, float] = {}
    scores = []
    if "results" in results:
        for k, v in results["results"].items():
            for acc_key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if acc_key in v:
                    subtask_scores[k] = float(v[acc_key])
                    scores.append(float(v[acc_key]))
                    break

    overall = sum(scores) / len(scores) if scores else None
    return overall, subtask_scores


def run_condition(
    label: str,
    target: Set[int],
    model, tok, diff, diff_sha,
    args, n_layers, all_layers, git_rev,
) -> dict | None:
    out_json = os.path.join(args.output_dir, f"block_only_{label}_mmlu.json")
    if os.path.exists(out_json):
        print(f"[skip] {label} (already exists: {out_json})")
        with open(out_json) as f:
            return json.load(f)

    print(f"\n>>> {label}  injected={len(target)}/{n_layers} layers  "
          f"({sorted(target)[:3]}...)", flush=True)

    if target:
        apply_stats = apply_delta(model, diff, target, alpha=args.alpha, subtract=False)
        print(f"  applied={apply_stats['applied_params']} params", flush=True)
    else:
        apply_stats = {"applied_params": 0, "note": "baseline_no_injection"}

    lm = make_injected_lm(model, tok, args.gpu, args.batch_size)
    mmlu_acc, subtask_scores = eval_mmlu_subset(lm, MMLU_SUBSET)
    del lm

    if target:
        apply_delta(model, diff, target, alpha=args.alpha, subtract=True)

    excluded = sorted(all_layers - target)
    payload = dict(
        exp=f"block_only_{label}_mmlu",
        model=args.model,
        n_layers=n_layers,
        condition=label,
        layers_injected=sorted(target),
        layers_excluded=excluded,
        n_injected=len(target),
        alpha=args.alpha,
        seed=args.seed,
        mmlu_tasks=MMLU_SUBSET,
        n_mmlu_tasks=len(MMLU_SUBSET),
        diff_sha256=diff_sha,
        git_hash=git_rev,
        mmlu_acc=mmlu_acc,
        subtask_scores=subtask_scores,
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
    acc_s = f"{mmlu_acc:.4f}" if mmlu_acc is not None else "None"
    print(f"  MMLU-subset={acc_s}  → {out_json}", flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(
        description="Block-only MMLU: inject ΔW at one third only, measure capability."
    )
    p.add_argument("--model",      required=True, help="HuggingFace model name or path")
    p.add_argument("--diff",       required=True, help="Path to diff_vector.pt")
    p.add_argument("--output_dir", required=True, help="Directory for output JSON files")
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--alpha",      type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--conditions", default="baseline,only_first,only_mid,only_last",
                   help="Comma-separated conditions to run.")
    args = p.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    git_rev = git_hash()
    requested = set(args.conditions.split(","))

    n_layers   = get_num_layers(resolve_model_path(args.model) or args.model)
    all_layers = set(range(n_layers))
    t          = thirds(n_layers)

    print(f"[block_only_mmlu] model={args.model}  n_layers={n_layers}  git={git_rev}")
    print(f"  first: L{min(t['first'])}..L{max(t['first'])}  ({len(t['first'])} layers)")
    print(f"  mid:   L{min(t['mid'])}..L{max(t['mid'])}  ({len(t['mid'])} layers)")
    print(f"  last:  L{min(t['last'])}..L{max(t['last'])}  ({len(t['last'])} layers)")
    print(f"  MMLU tasks ({len(MMLU_SUBSET)}): {MMLU_SUBSET[:3]}...")
    print(f"  seed={args.seed}  alpha={args.alpha}  dtype={args.dtype}")

    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)
    print(f"  diff SHA256={diff_sha[:16]}...")

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu,
    )

    all_conditions = [
        ("baseline",    set()),          # no injection — base model capability
        ("only_first",  t["first"]),
        ("only_mid",    t["mid"]),
        ("only_last",   t["last"]),
    ]
    conditions = [(label, target) for label, target in all_conditions
                  if label in requested]

    common = dict(
        model=model, tok=tok, diff=diff, diff_sha=diff_sha,
        args=args, n_layers=n_layers, all_layers=all_layers, git_rev=git_rev,
    )

    all_results = []
    for label, target in conditions:
        r = run_condition(label, target, **common)
        if r:
            all_results.append(dict(
                condition=label,
                n_injected=len(target),
                mmlu_acc=r.get("mmlu_acc"),
            ))

    print("\n=== Block-Only MMLU Summary ===")
    print(f"  model={args.model}  git={git_rev}  seed={args.seed}")
    print(f"  tasks: {len(MMLU_SUBSET)} subcategories (14/57 MMLU)")
    print(f"{'condition':<14} {'injected':>8} {'mmlu_acc':>10}")
    print("-" * 36)
    for row in all_results:
        acc = row["mmlu_acc"]
        acc_s = f"{acc:.4f}" if acc is not None else "   None"
        print(f"{row['condition']:<14} {row['n_injected']:>8} {acc_s:>10}")

    summary_path = os.path.join(args.output_dir, "block_only_mmlu_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha, seed=args.seed,
            git_hash=git_rev,
            mmlu_tasks=MMLU_SUBSET,
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
