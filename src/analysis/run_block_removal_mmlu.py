#!/usr/bin/env python3
"""
Block Removal MMLU — capability preservation check.

Evaluates MMLU on each block removal condition (full / minus_first / minus_mid /
minus_last) using the same in-memory apply→eval→unapply loop as run_block_removal.py.

MMLU subset (14 subcategories, 4 domains):
  STEM (4):          abstract_algebra, high_school_mathematics,
                     high_school_physics, high_school_computer_science
  Humanities (4):    philosophy, world_history, moral_scenarios,
                     high_school_history
  Social Sci (3):    sociology, high_school_psychology,
                     high_school_government_and_politics
  Professional (3):  professional_law, medical_genetics, global_facts

14/57 categories ≈ 4× faster than full MMLU; covers all four Hendrycks et al.
(2021) domains so domain-level performance is representable.
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
    apply_delta, sha256_file,
)

# MMLU subset
# 14 representative subcategories across all 4 Hendrycks et al. (2021) domains.
# Chosen to cover objective/factual knowledge; less likely to be confounded by
# the bias signal targeted by our ΔW intervention.
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
    # Professional / Other (3)
    "mmlu_professional_law",
    "mmlu_medical_genetics",
    "mmlu_global_facts",
]


# lm_eval in-memory wrapper

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


# thirds (identical to run_block_removal.py)

def thirds(n: int) -> dict[str, Set[int]]:
    f = set(range(0, n // 3))
    m = set(range(n // 3, 2 * n // 3))
    l = set(range(2 * n // 3, n))
    return {"first": f, "mid": m, "last": l}


# per-condition MMLU run

def run_condition(
    label: str,
    target: Set[int],
    model, tok, diff, diff_sha,
    args, n_layers, all_layers,
) -> dict | None:
    out_json = os.path.join(args.output_dir, f"block_{label}_mmlu.json")
    if os.path.exists(out_json):
        print(f"[skip] {label} (already done)")
        with open(out_json) as f:
            return json.load(f)

    print(f"\n>>> {label}  target={len(target)}/{n_layers} layers", flush=True)

    apply_stats = apply_delta(model, diff, target, alpha=args.alpha, subtract=False)
    print(f"  applied={apply_stats['applied_params']} params", flush=True)

    lm = make_injected_lm(model, tok, args.gpu, args.batch_size)
    mmlu_acc, subtask_scores = eval_mmlu_subset(lm, MMLU_SUBSET)
    del lm

    apply_delta(model, diff, target, alpha=args.alpha, subtract=True)

    excluded = sorted(all_layers - target)
    payload = dict(
        exp=f"block_{label}_mmlu",
        model=args.model,
        n_layers=n_layers,
        condition=label,
        layers_injected=sorted(target),
        layers_excluded=excluded,
        alpha=args.alpha,
        mmlu_tasks=MMLU_SUBSET,
        diff_sha256=diff_sha,
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
    p = argparse.ArgumentParser(description="Block removal MMLU capability check")
    p.add_argument("--model",      required=True)
    p.add_argument("--diff",       required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--dtype",      default="bfloat16")
    p.add_argument("--alpha",      type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--conditions", default="full,minus_first,minus_mid,minus_last",
                   help="Comma-separated conditions to run.")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    requested = set(args.conditions.split(","))

    n_layers   = get_num_layers(resolve_model_path(args.model) or args.model)
    all_layers = set(range(n_layers))
    t          = thirds(n_layers)

    print(f"[block_removal_mmlu] model={args.model}  n_layers={n_layers}")
    print(f"  first: L{min(t['first'])}..L{max(t['first'])}  ({len(t['first'])} layers)")
    print(f"  mid:   L{min(t['mid'])}..L{max(t['mid'])}  ({len(t['mid'])} layers)")
    print(f"  last:  L{min(t['last'])}..L{max(t['last'])}  ({len(t['last'])} layers)")
    print(f"  MMLU tasks ({len(MMLU_SUBSET)}): {MMLU_SUBSET[:3]}...")

    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu,
    )

    all_conditions = [
        ("full",        all_layers),
        ("minus_first", all_layers - t["first"]),
        ("minus_mid",   all_layers - t["mid"]),
        ("minus_last",  all_layers - t["last"]),
    ]
    conditions = [(label, target) for label, target in all_conditions
                  if label in requested]

    common = dict(
        model=model, tok=tok, diff=diff, diff_sha=diff_sha,
        args=args, n_layers=n_layers, all_layers=all_layers,
    )

    all_results = []
    for label, target in conditions:
        r = run_condition(label, target, **common)
        if r:
            all_results.append(dict(
                condition=label,
                mmlu_acc=r.get("mmlu_acc"),
                n_injected=len(target),
            ))

    print("\n=== Block Removal MMLU Summary ===")
    print(f"  tasks: {len(MMLU_SUBSET)} subcategories (14/57 MMLU)")
    print(f"{'condition':<14} {'injected':>8} {'mmlu_acc':>10}")
    print("-" * 36)
    for row in all_results:
        acc = row["mmlu_acc"]
        acc_s = f"{acc:.4f}" if acc is not None else "   None"
        print(f"{row['condition']:<14} {row['n_injected']:>8} {acc_s:>10}")

    summary_path = os.path.join(args.output_dir, "block_mmlu_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha,
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
