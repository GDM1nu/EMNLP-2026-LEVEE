#!/usr/bin/env python3
"""
run_eval.py — Unified evaluation entry point.

A single script to run any combination of {baseline, layerwise, random}
× {bbq, crows, stereoset}.  Replaces the need for separate eval_bbq_*.py,
eval_crows_*.py, eval_stereoset_*.py scripts.

Examples
--------
# Baseline (no diff vector)
python run_eval.py baseline \
    --model Qwen/Qwen2.5-0.5B --label base \
    --benchmarks bbq crows stereoset

# Layerwise
python run_eval.py layerwise \
    --model Qwen/Qwen2.5-0.5B --diff diff_vector.pt \
    --layer_condition mid_8 \
    --benchmarks bbq crows stereoset

# Random control
python run_eval.py random \
    --model Qwen/Qwen2.5-0.5B --diff diff_vector.pt \
    --control_type norm_matched --seed 0 \
    --benchmarks bbq
"""
from __future__ import annotations

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
import gc
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

import torch
import transformers

from model_utils import (
    compute_and_save_diff, compute_layer_groups, get_num_layers,
    load_model_and_tokenizer, resolve_model_path,
    analyze_matching, apply_delta, make_control_diff, sha256_file,
)
from eval_bbq_shared import (
    load_bbq, load_bbq_hf, compute_metrics as bbq_metrics,
    print_results as bbq_print, evaluate_batched_logits,
)
from eval_crows_shared import (
    load_crows_pairs, evaluate_crows, compute_crows_metrics,
)
from eval_stereoset_shared import (
    load_stereoset, evaluate_stereoset, compute_stereoset_metrics,
)

# ======================================================================
# Data paths (defaults — override with flags)
# ======================================================================
_BBQ_DIR     = str(BBQ_JSONL.parent / "bbq")
_CROWS_CSV   = str(BBQ_JSONL.parent / "crows_pairs" / "crows_pairs_anonymized.csv")
_STEREO_JSON = str(BBQ_JSONL.parent / "stereoset" / "test.json")


def _env_info() -> dict:
    return dict(
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda=str(torch.version.cuda),
    )


# ======================================================================
# Sub-command: baseline
# ======================================================================
def cmd_baseline(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu, quantize=args.quantize, multi_gpu=args.multi_gpu,
    )
    device = next(model.parameters()).device if (args.quantize is not None or args.multi_gpu) else f"cuda:{args.gpu}"

    for bench in args.benchmarks:
        tag = f"baseline_{args.label}_{bench}"
        out_json = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(out_json):
            print(f"[skip] {tag}"); continue

        print(f"\n>>> {tag}")
        if bench == "bbq":
            data = load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
            results, trunc = evaluate_batched_logits(
                model, tok, data, str(device), args.batch_size, args.max_length)
            metrics = bbq_metrics(results)
            bbq_print(metrics, tag)
            payload = dict(metrics=metrics, trunc_stats=trunc)

        elif bench == "crows":
            dataset = load_crows_pairs(args.crows_path)
            results = evaluate_crows(model, tok, dataset, str(device), args.batch_size)
            metrics = compute_crows_metrics(results)
            print(f"  CrowS bias_score = {metrics['overall']['bias_score']:.2f}%")
            payload = dict(metrics=metrics)

        elif bench == "stereoset":
            dataset = load_stereoset(args.stereoset_path)
            results = evaluate_stereoset(model, tok, dataset, str(device), args.batch_size)
            metrics = compute_stereoset_metrics(results)
            o = metrics["overall"]
            print(f"  StereoSet LM={o['lm_score']:.1f} SS={o['ss_score']:.1f} iCAT={o['icat']:.1f}")
            payload = dict(metrics=metrics)
        else:
            continue

        payload.update(dict(
            exp=tag, model=args.model, model_path=model_path,
            timestamp=datetime.now().isoformat(), env=_env_info(),
        ))
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  → {out_json}")

    del model, tok; gc.collect(); torch.cuda.empty_cache()


# ======================================================================
# Sub-command: layerwise
# ======================================================================
def cmd_layerwise(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)
    groups   = compute_layer_groups(n_layers)
    target   = set(groups[args.layer_condition])

    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu, quantize=args.quantize, multi_gpu=args.multi_gpu,
    )
    device = next(model.parameters()).device if (args.quantize is not None or args.multi_gpu) else f"cuda:{args.gpu}"

    match_stats = analyze_matching(model, diff, target)
    apply_stats = apply_delta(model, diff, target, alpha=args.alpha)
    print(f"  applied {apply_stats['applied_params']} params to layers {sorted(target)}")
    del diff; gc.collect()

    for bench in args.benchmarks:
        tag = f"layerwise_{args.layer_condition}_{bench}"
        out_json = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(out_json):
            print(f"[skip] {tag}"); continue

        print(f"\n>>> {tag}")
        if bench == "bbq":
            data = load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
            results, trunc = evaluate_batched_logits(
                model, tok, data, str(device), args.batch_size, args.max_length)
            metrics = bbq_metrics(results)
            bbq_print(metrics, tag)
            payload = dict(metrics=metrics, trunc_stats=trunc)

        elif bench == "crows":
            dataset = load_crows_pairs(args.crows_path)
            results = evaluate_crows(model, tok, dataset, str(device), args.batch_size)
            metrics = compute_crows_metrics(results)
            print(f"  CrowS bias_score = {metrics['overall']['bias_score']:.2f}%")
            payload = dict(metrics=metrics)

        elif bench == "stereoset":
            dataset = load_stereoset(args.stereoset_path)
            results = evaluate_stereoset(model, tok, dataset, str(device), args.batch_size)
            metrics = compute_stereoset_metrics(results)
            o = metrics["overall"]
            print(f"  StereoSet LM={o['lm_score']:.1f} SS={o['ss_score']:.1f} iCAT={o['icat']:.1f}")
            payload = dict(metrics=metrics)
        else:
            continue

        payload.update(dict(
            exp=tag, model=args.model, n_layers=n_layers,
            layer_condition=args.layer_condition,
            target_layers=sorted(target),
            all_groups={k: [v[0], v[-1]] for k, v in groups.items()},
            diff_sha256=diff_sha, model_path=model_path,
            match_stats=match_stats, apply_stats=apply_stats,
            timestamp=datetime.now().isoformat(), env=_env_info(),
        ))
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  → {out_json}")

    del model, tok; gc.collect(); torch.cuda.empty_cache()


# ======================================================================
# Sub-command: sweep  (full per-layer sweep, one layer at a time)
# ======================================================================
def _sweep_eval_bench(bench, model, tok, device, args, preloaded_bbq=None) -> tuple[dict, dict]:
    """Run one benchmark and return (metrics, extra_payload_fields)."""
    if bench == "bbq":
        data = preloaded_bbq if preloaded_bbq is not None else (
            load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
        )
        results, trunc = evaluate_batched_logits(
            model, tok, data, str(device), args.batch_size, args.max_length)
        metrics = bbq_metrics(results)
        bbq_print(metrics, bench)
        return metrics, dict(trunc_stats=trunc)

    if bench == "crows":
        dataset = load_crows_pairs(args.crows_path)
        results = evaluate_crows(model, tok, dataset, str(device), args.batch_size)
        metrics = compute_crows_metrics(results)
        print(f"  CrowS bias_score = {metrics['overall']['bias_score']:.2f}%")
        return metrics, {}

    if bench == "stereoset":
        dataset = load_stereoset(args.stereoset_path)
        results = evaluate_stereoset(model, tok, dataset, str(device), args.batch_size)
        metrics = compute_stereoset_metrics(results)
        o = metrics["overall"]
        print(f"  StereoSet LM={o['lm_score']:.1f} SS={o['ss_score']:.1f} iCAT={o['icat']:.1f}")
        return metrics, {}

    raise ValueError(f"Unknown benchmark: {bench}")


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    """Parse a --layers spec into a sorted layer index list.

    "" / "all"  -> every layer
    "0-31"      -> inclusive range
    "0,4,8"     -> explicit list
    Ranges and lists may be mixed: "0-7,20,30-33".
    Used to shard one model's sweep across several GPU groups.
    """
    if not spec or spec.strip().lower() == "all":
        return list(range(n_layers))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [i for i in out if not (0 <= i < n_layers)]
    if bad:
        raise ValueError(f"--layers out of range for n_layers={n_layers}: {sorted(bad)}")
    return sorted(out)


def _stratified_subsample(data, max_samples: int, seed: int):
    """Stratified BBQ subsample by (category, context_condition).

    Byte-identical protocol to scripts/rebuttal3 run_block_removal.py so that
    subsampled sweep numbers are directly comparable to the rebuttal3 8k block
    results. Returns the input unchanged when max_samples is 0 or >= len(data).
    """
    if not max_samples or max_samples >= len(data):
        return data
    strata = defaultdict(list)
    for it in data:
        strata[(it.get("category"), it.get("context_condition", "unknown"))].append(it)
    n_str = len(strata)
    per = max(1, max_samples // n_str)
    rng = random.Random(seed)
    out = []
    for key in sorted(strata):   # identical ordering to rebuttal3 run_block_removal.py
        items = strata[key]
        if len(items) <= per:
            out.extend(items)
        else:
            out.extend(rng.sample(items, per))
    print(f"[preload] BBQ subsampled -> {len(out)} items "
          f"({n_str} strata x ~{per}, seed={seed})", flush=True)
    return out


def _sweep_meta_guard(out_dir: str, n_eval: int, n_layers: int, diff_sha: str):
    """Refuse to mix different BBQ sample sizes inside one output directory.

    Subsampled and full-58k per-layer JSONs share the sweep_L##_bbq.json name,
    and cmd_sweep resumes by skipping files that already exist — so writing both
    into one directory would silently produce a half-subsample, half-full
    profile. This writes SWEEP_META.json on first use and hard-fails on mismatch.
    """
    path = os.path.join(out_dir, "SWEEP_META.json")
    meta = dict(n_eval_samples=n_eval, n_layers=n_layers, diff_sha256=diff_sha)
    if os.path.exists(path):
        prev = json.load(open(path))
        for k in ("n_eval_samples", "n_layers", "diff_sha256"):
            if prev.get(k) != meta[k]:
                raise SystemExit(
                    f"[ABORT] {path} says {k}={prev.get(k)!r} but this run has "
                    f"{meta[k]!r}. Use a separate --output_dir (e.g. "
                    f"..._sub8k vs ..._full58k) instead of mixing them."
                )
    else:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
    return meta


def cmd_sweep(args):
    """Full per-layer sweep: apply diff to one layer at a time and evaluate.

    Design:
      - Model loaded once; diff applied / unapplied per iteration (subtract=True).
      - match_stats recorded per layer to verify key alignment.
      - Floating-point drift monitored via a reference parameter snapshot.
      - Per-layer JSONs written incrementally; sweep_summary.json written at end.
      - Safe to resume: layers whose output files already exist are skipped.
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)

    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu, quantize=args.quantize, multi_gpu=args.multi_gpu,
    )
    device = next(model.parameters()).device if (args.quantize is not None or args.multi_gpu) else f"cuda:{args.gpu}"

    # --- drift reference: snapshot one parameter before any modification ---
    _drift_name, _drift_ref = next(
        (n, p.detach().clone().cpu())
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    print(f"[drift_ref] tracking '{_drift_name}'")

    # preload BBQ once to avoid reloading on every layer iteration
    _preloaded_bbq = None
    if "bbq" in args.benchmarks:
        _preloaded_bbq = load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
        print(f"[preload] BBQ: {len(_preloaded_bbq)} samples (full)")
        _preloaded_bbq = _stratified_subsample(
            _preloaded_bbq, getattr(args, "max_samples", 0), getattr(args, "sample_seed", 0))
        _sweep_meta_guard(out_dir, len(_preloaded_bbq), n_layers, diff_sha)

    all_results: list[dict] = []

    layer_plan = _parse_layers(getattr(args, "layers", ""), n_layers)
    print(f"[plan] evaluating {len(layer_plan)}/{n_layers} layers: {layer_plan}", flush=True)

    for layer_idx in layer_plan:
        target = {layer_idx}

        pending = [
            bench for bench in args.benchmarks
            if not os.path.exists(
                os.path.join(out_dir, f"sweep_L{layer_idx:02d}_{bench}.json")
            )
        ]
        if not pending:
            print(f"[skip] layer {layer_idx:02d}/{n_layers - 1} (all benchmarks done)")
            continue

        # --- verify key alignment before touching weights ---
        match_stats = analyze_matching(model, diff, target)

        # --- apply diff to this layer only ---
        apply_stats = apply_delta(model, diff, target, alpha=args.alpha, subtract=False)
        print(f"\n>>> layer {layer_idx:02d}/{n_layers - 1}"
              f"  matched={match_stats.get('matched_in_layers', '?')}"
              f"  applied={apply_stats['applied_params']} params")

        layer_entry = dict(
            layer_idx=layer_idx,
            match_stats=match_stats,
            apply_stats=apply_stats,
            benchmarks={},
        )

        for bench in pending:
            tag      = f"sweep_L{layer_idx:02d}_{bench}"
            out_json = os.path.join(out_dir, f"{tag}.json")
            print(f"  [{bench}]")

            metrics, extra = _sweep_eval_bench(bench, model, tok, device, args, preloaded_bbq=_preloaded_bbq)
            layer_entry["benchmarks"][bench] = metrics

            payload = dict(metrics=metrics, **extra)
            payload.update(dict(
                exp=tag, model=args.model, model_path=model_path,
                n_layers=n_layers,
                layer_idx=layer_idx,
                alpha=args.alpha,
                diff_sha256=diff_sha,
                match_stats=match_stats,
                apply_stats=apply_stats,
                timestamp=datetime.now().isoformat(), env=_env_info(),
            ))
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"  → {out_json}")

        all_results.append(layer_entry)

        # --- unapply: explicit subtract to restore model weights ---
        apply_delta(model, diff, target, alpha=args.alpha, subtract=True)

        # --- drift check: confirm weights returned to reference state ---
        _cur = dict(model.named_parameters())[_drift_name].detach().cpu()
        drift = (_cur - _drift_ref).abs().max().item()
        print(f"[drift_check] layer={layer_idx:02d}  max_abs_diff={drift:.2e}")

    # --- aggregate summary across all evaluated layers ---
    # Sharded runs each cover only part of the model, so they must not clobber a
    # single sweep_summary.json. Shards write sweep_summary_L<a>-<b>.json; the
    # canonical per-layer evidence is the sweep_L##_bbq.json files either way,
    # and tools/analyze_sweep.py rebuilds the full profile from those.
    if getattr(args, "layers", ""):
        tag = f"L{layer_plan[0]:02d}-{layer_plan[-1]:02d}"
        summary_path = os.path.join(out_dir, f"sweep_summary_{tag}.json")
    else:
        summary_path = os.path.join(out_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha,
            diff_sha256=diff_sha,
            layer_plan=layer_plan,
            n_eval_samples=(len(_preloaded_bbq) if _preloaded_bbq is not None else None),
            max_samples=getattr(args, "max_samples", 0),
            sample_seed=getattr(args, "sample_seed", 0),
            timestamp=datetime.now().isoformat(), env=_env_info(),
            layers=all_results,
        ), f, indent=2, ensure_ascii=False)
    print(f"\n[summary] → {summary_path}")

    del model, tok, diff; gc.collect(); torch.cuda.empty_cache()


# ======================================================================
# Sub-command: loo_sweep  (Leave-One-Out)
# ======================================================================
def cmd_loo_sweep(args):
    """Leave-One-Out sweep: inject ΔW into ALL layers except one, evaluate.

    For each excluded layer L:
      - target = {0, 1, ..., N-1} - {L}
      - apply ΔW to all other layers → evaluate BBQ → unapply

    This measures *necessity*: if bias returns when L is excluded,
    L is a causally necessary layer for bias suppression.
    Contrast with cmd_sweep (sufficiency: one layer alone).
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)

    diff     = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff)

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu, quantize=args.quantize, multi_gpu=args.multi_gpu,
    )
    device = next(model.parameters()).device if (args.quantize is not None or args.multi_gpu) else f"cuda:{args.gpu}"

    _drift_name, _drift_ref = next(
        (n, p.detach().clone().cpu())
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    print(f"[drift_ref] tracking '{_drift_name}'")

    _preloaded_bbq = None
    if "bbq" in args.benchmarks:
        _preloaded_bbq = load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
        print(f"[preload] BBQ: {len(_preloaded_bbq)} samples")

    all_results: list[dict] = []

    for layer_idx in range(n_layers):
        # LOO: all layers except this one
        target = set(range(n_layers)) - {layer_idx}

        pending = [
            bench for bench in args.benchmarks
            if not os.path.exists(
                os.path.join(out_dir, f"loo_L{layer_idx:02d}_{bench}.json")
            )
        ]
        if not pending:
            print(f"[skip] loo layer_excluded={layer_idx:02d}/{n_layers - 1} (all benchmarks done)")
            continue

        match_stats = analyze_matching(model, diff, target)
        apply_stats = apply_delta(model, diff, target, alpha=args.alpha, subtract=False)
        print(f"\n>>> loo exclude=L{layer_idx:02d}/{n_layers - 1}"
              f"  injected={len(target)} layers"
              f"  applied={apply_stats['applied_params']} params")

        layer_entry = dict(
            layer_excluded=layer_idx,
            layers_injected=sorted(target),
            match_stats=match_stats,
            apply_stats=apply_stats,
            benchmarks={},
        )

        for bench in pending:
            tag      = f"loo_L{layer_idx:02d}_{bench}"
            out_json = os.path.join(out_dir, f"{tag}.json")
            print(f"  [{bench}]")

            metrics, extra = _sweep_eval_bench(bench, model, tok, device, args, preloaded_bbq=_preloaded_bbq)
            layer_entry["benchmarks"][bench] = metrics

            payload = dict(metrics=metrics, **extra)
            payload.update(dict(
                exp=tag, model=args.model, model_path=model_path,
                n_layers=n_layers,
                layer_excluded=layer_idx,
                layers_injected=sorted(target),
                alpha=args.alpha,
                diff_sha256=diff_sha,
                match_stats=match_stats,
                apply_stats=apply_stats,
                timestamp=datetime.now().isoformat(), env=_env_info(),
            ))
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"  → {out_json}")

        all_results.append(layer_entry)

        apply_delta(model, diff, target, alpha=args.alpha, subtract=True)

        _cur = dict(model.named_parameters())[_drift_name].detach().cpu()
        drift = (_cur - _drift_ref).abs().max().item()
        print(f"[drift_check] excluded=L{layer_idx:02d}  max_abs_diff={drift:.2e}")

    summary_path = os.path.join(out_dir, "loo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha,
            diff_sha256=diff_sha,
            timestamp=datetime.now().isoformat(), env=_env_info(),
            layers=all_results,
        ), f, indent=2, ensure_ascii=False)
    print(f"\n[summary] → {summary_path}")

    del model, tok, diff; gc.collect(); torch.cuda.empty_cache()


# ======================================================================
# Sub-command: random
# ======================================================================
def cmd_random(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)
    groups   = compute_layer_groups(n_layers)
    target   = set(groups[args.layer_condition])

    diff_orig = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha  = sha256_file(args.diff)
    diff_ctrl = make_control_diff(diff_orig, args.control_type, args.seed)
    del diff_orig; gc.collect()

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu, quantize=args.quantize, multi_gpu=args.multi_gpu,
    )
    device = next(model.parameters()).device if (args.quantize is not None or args.multi_gpu) else f"cuda:{args.gpu}"

    apply_stats = apply_delta(model, diff_ctrl, target, alpha=args.alpha)
    print(f"  applied {apply_stats['applied_params']} control params")
    del diff_ctrl; gc.collect()

    seed_tag = "" if args.control_type == "zero" else f"_seed{args.seed}"

    for bench in args.benchmarks:
        tag = f"random_{args.control_type}{seed_tag}_{args.layer_condition}_{bench}"
        out_json = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(out_json):
            print(f"[skip] {tag}"); continue

        print(f"\n>>> {tag}")
        if bench == "bbq":
            data = load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
            results, trunc = evaluate_batched_logits(
                model, tok, data, str(device), args.batch_size, args.max_length)
            metrics = bbq_metrics(results)
            bbq_print(metrics, tag)
            payload = dict(metrics=metrics, trunc_stats=trunc)

        elif bench == "crows":
            dataset = load_crows_pairs(args.crows_path)
            results = evaluate_crows(model, tok, dataset, str(device), args.batch_size)
            metrics = compute_crows_metrics(results)
            payload = dict(metrics=metrics)

        elif bench == "stereoset":
            dataset = load_stereoset(args.stereoset_path)
            results = evaluate_stereoset(model, tok, dataset, str(device), args.batch_size)
            metrics = compute_stereoset_metrics(results)
            payload = dict(metrics=metrics)
        else:
            continue

        payload.update(dict(
            exp=tag, model=args.model, n_layers=n_layers,
            control_type=args.control_type, seed=args.seed,
            layer_condition=args.layer_condition,
            target_layers=sorted(target),
            diff_sha256=diff_sha, apply_stats=apply_stats,
            timestamp=datetime.now().isoformat(), env=_env_info(),
        ))
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  → {out_json}")

    del model, tok; gc.collect(); torch.cuda.empty_cache()


# ======================================================================
# Sub-command: random_sweep
# ======================================================================
def cmd_random_sweep(args):
    """Per-layer random control sweep: same loop as cmd_sweep but injects a
    control diff (norm_matched / shuffled / zero) into one layer at a time."""
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    n_layers = get_num_layers(resolve_model_path(args.model) or args.model)

    diff_orig = torch.load(args.diff, map_location="cpu", weights_only=False)
    diff_sha  = sha256_file(args.diff)
    diff_ctrl = make_control_diff(diff_orig, args.control_type, args.seed)
    del diff_orig; gc.collect()

    model, tok, model_path = load_model_and_tokenizer(
        args.model, dtype=args.dtype, gpu=args.gpu, quantize=args.quantize, multi_gpu=args.multi_gpu,
    )
    device = next(model.parameters()).device if (args.quantize is not None or args.multi_gpu) else f"cuda:{args.gpu}"

    _drift_name, _drift_ref = next(
        (n, p.detach().clone().cpu())
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    print(f"[drift_ref] tracking '{_drift_name}'")

    _preloaded_bbq = None
    if "bbq" in args.benchmarks:
        _preloaded_bbq = load_bbq(args.bbq_dir) if os.path.isdir(args.bbq_dir) else load_bbq_hf()
        print(f"[preload] BBQ: {len(_preloaded_bbq)} samples")

    seed_tag = "" if args.control_type == "zero" else f"_seed{args.seed}"
    all_results: list[dict] = []

    for layer_idx in range(n_layers):
        target = {layer_idx}

        pending = [
            bench for bench in args.benchmarks
            if not os.path.exists(
                os.path.join(out_dir, f"ctrl_{args.control_type}{seed_tag}_L{layer_idx:02d}_{bench}.json")
            )
        ]
        if not pending:
            print(f"[skip] layer {layer_idx:02d}/{n_layers - 1} (all benchmarks done)")
            continue

        match_stats = analyze_matching(model, diff_ctrl, target)
        apply_stats = apply_delta(model, diff_ctrl, target, alpha=args.alpha, subtract=False)
        print(f"\n>>> layer {layer_idx:02d}/{n_layers - 1}"
              f"  control={args.control_type}{seed_tag}"
              f"  applied={apply_stats['applied_params']} params")

        layer_entry = dict(
            layer_idx=layer_idx,
            match_stats=match_stats,
            apply_stats=apply_stats,
            benchmarks={},
        )

        for bench in pending:
            tag      = f"ctrl_{args.control_type}{seed_tag}_L{layer_idx:02d}_{bench}"
            out_json = os.path.join(out_dir, f"{tag}.json")
            print(f"  [{bench}]")

            metrics, extra = _sweep_eval_bench(bench, model, tok, device, args, preloaded_bbq=_preloaded_bbq)
            layer_entry["benchmarks"][bench] = metrics

            payload = dict(metrics=metrics, **extra)
            payload.update(dict(
                exp=tag, model=args.model, model_path=model_path,
                n_layers=n_layers,
                layer_idx=layer_idx,
                control_type=args.control_type,
                seed=args.seed,
                alpha=args.alpha,
                diff_sha256=diff_sha,
                match_stats=match_stats,
                apply_stats=apply_stats,
                timestamp=datetime.now().isoformat(), env=_env_info(),
            ))
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"  → {out_json}")

        all_results.append(layer_entry)

        apply_delta(model, diff_ctrl, target, alpha=args.alpha, subtract=True)

        _cur = dict(model.named_parameters())[_drift_name].detach().cpu()
        drift = (_cur - _drift_ref).abs().max().item()
        print(f"[drift_check] layer={layer_idx:02d}  max_abs_diff={drift:.2e}")

    summary_path = os.path.join(out_dir, "ctrl_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(
            model=args.model, model_path=model_path,
            n_layers=n_layers, alpha=args.alpha,
            control_type=args.control_type, seed=args.seed,
            diff_sha256=diff_sha,
            timestamp=datetime.now().isoformat(), env=_env_info(),
            layers=all_results,
        ), f, indent=2, ensure_ascii=False)
    print(f"\n[summary] → {summary_path}")

    del model, tok, diff_ctrl; gc.collect(); torch.cuda.empty_cache()


# ======================================================================
# Argparse
# ======================================================================
def main():
    p = argparse.ArgumentParser(
        description="Unified diff-vector bias-transfer evaluation")
    sub = p.add_subparsers(dest="mode", required=True)

    # --- shared args ---
    def add_common(sp):
        sp.add_argument("--model", required=True)
        sp.add_argument("--benchmarks", nargs="+", default=["bbq", "crows", "stereoset"],
                        choices=["bbq", "crows", "stereoset"])
        sp.add_argument("--output_dir", required=True)
        sp.add_argument("--gpu", type=int, default=0)
        sp.add_argument("--batch_size", type=int, default=8)
        sp.add_argument("--max_length", type=int, default=512)
        sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
        sp.add_argument("--quantize", default=None, choices=["4bit", "8bit"])
        sp.add_argument("--multi_gpu", action="store_true", default=False,
                        help="Spread model across all available GPUs (device_map=auto, full precision)")
        sp.add_argument("--bbq_dir", default=_BBQ_DIR)
        sp.add_argument("--crows_path", default=_CROWS_CSV)
        sp.add_argument("--stereoset_path", default=_STEREO_JSON)

    # baseline
    sp_b = sub.add_parser("baseline")
    add_common(sp_b)
    sp_b.add_argument("--label", required=True, help="e.g. base / instruct")

    # layerwise
    sp_l = sub.add_parser("layerwise")
    add_common(sp_l)
    sp_l.add_argument("--diff", required=True)
    sp_l.add_argument("--layer_condition", required=True,
                      choices=["first_4", "first_8", "mid_8", "last_8",
                               "last_4", "all", "minus_mid_8",
                               "minus_first_4", "minus_first_8", "minus_last_8", "minus_last_4",
                               "first_third", "mid_third", "last_third",
                               "minus_first_third", "minus_mid_third", "minus_last_third"])
    sp_l.add_argument("--alpha", type=float, default=1.0)

    # sweep (full per-layer sweep)
    sp_s = sub.add_parser("sweep",
                          help="Apply diff to each layer individually and evaluate (full sweep)")
    add_common(sp_s)
    sp_s.add_argument("--diff", required=True)
    sp_s.add_argument("--alpha", type=float, default=1.0)
    # --- LEVEE camera-ready additions (per-layer sweep on large models) ---
    sp_s.add_argument("--layers", default="",
                      help="Layer shard for this worker: '0-21', '0,4,8', or "
                           "'0-7,20-23'. Empty/'all' = every layer. Lets one "
                           "model's sweep be split across several GPU groups.")
    sp_s.add_argument("--max_samples", type=int, default=0,
                      help="If >0, stratified-subsample BBQ to ~this many items "
                           "(balanced over category x context_condition, seeded). "
                           "Same protocol as rebuttal3 run_block_removal.py. "
                           "0 = full 58,492.")
    sp_s.add_argument("--sample_seed", type=int, default=0,
                      help="Seed for --max_samples stratified subsampling.")

    # random
    sp_r = sub.add_parser("random")
    add_common(sp_r)
    sp_r.add_argument("--diff", required=True)
    sp_r.add_argument("--control_type", required=True,
                      choices=["norm_matched", "shuffled", "zero"])
    sp_r.add_argument("--seed", type=int, default=0)
    sp_r.add_argument("--layer_condition", default="mid_third",
                      choices=["first_4", "first_8", "mid_8", "last_8",
                               "last_4", "all", "minus_mid_8",
                               "minus_first_4", "minus_first_8", "minus_last_8", "minus_last_4",
                               "first_third", "mid_third", "last_third",
                               "minus_first_third", "minus_mid_third", "minus_last_third"])
    sp_r.add_argument("--subtract", action="store_true", default=False,
                      help="Instruct 모델에서 랜덤 벡터 subtract")
    sp_r.add_argument("--inst_model", default=None,
                      help="subtract 시 사용할 Instruct 모델")
    sp_r.add_argument("--alpha", type=float, default=1.0)

    # loo_sweep (leave-one-out sweep)
    sp_loo = sub.add_parser("loo_sweep",
                            help="Inject ΔW into ALL-except-one layer and evaluate (necessity)")
    add_common(sp_loo)
    sp_loo.add_argument("--diff", required=True)
    sp_loo.add_argument("--alpha", type=float, default=1.0)

    # random_sweep (per-layer control sweep)
    sp_rs = sub.add_parser("random_sweep",
                           help="Apply control diff to each layer individually and evaluate")
    add_common(sp_rs)
    sp_rs.add_argument("--diff", required=True)
    sp_rs.add_argument("--control_type", required=True,
                       choices=["norm_matched", "shuffled", "zero"])
    sp_rs.add_argument("--seed", type=int, default=0)
    sp_rs.add_argument("--alpha", type=float, default=1.0)

    args = p.parse_args()
    {"baseline": cmd_baseline, "layerwise": cmd_layerwise,
     "random": cmd_random, "sweep": cmd_sweep,
     "random_sweep": cmd_random_sweep,
     "loo_sweep": cmd_loo_sweep}[args.mode](args)


if __name__ == "__main__":
    main()
