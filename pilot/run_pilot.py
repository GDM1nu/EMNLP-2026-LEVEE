#!/usr/bin/env python3
"""
Sentinel pilot (PILOT_PLAN.md, steps 8-10): Qwen2.5-3B / Gemma-2-9B /
Llama-3-8B, 10 conditions each (base, full, 1 representative layer per
third, minus_first/mid/last, only_first/mid/last), full BBQ 58,492.

Reuses the SAME apply_delta/evaluate_batched_logits machinery as the
original results/block_removal/ and results/sweep_thirds/ runs (imported
unmodified from emnlp/perlayer_a6000/engine/) -- only the bias LABEL is
recomputed, via scorer.official_bbq.classify_prediction(), alongside the old
label, on the SAME model predictions. Saves per-example JSONL for every
condition (step 7).

Usage:
    conda activate bias_transfer_clean
    python run_pilot.py --model qwen_3b   --gpu 0
    python run_pilot.py --model gemma_9b  --gpu 0
    python run_pilot.py --model llama3_8b --gpu 1
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path
# Repository root; override with LEVEE_ROOT if the layout differs.
REPO_ROOT = _Path(_os.environ.get("LEVEE_ROOT", _Path(__file__).resolve().parents[1]))
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
import sys
from datetime import datetime

import numpy as np
import torch
import transformers

ENGINE = str(REPO_ROOT / "src" / "eval")
sys.path.insert(0, ENGINE)
sys.path.insert(0, str(REPO_ROOT))

from model_utils import (
    get_num_layers, load_model_and_tokenizer, resolve_model_path,
    analyze_matching, apply_delta, sha256_file,
)
from eval_bbq_shared import (
    load_bbq_hf, download_bbq, load_bbq, compute_metrics as old_bbq_metrics,
    evaluate_batched_logits,
)
from scorer.official_bbq import classify_prediction, compute_official_metrics

MODELS = {
    "qwen_3b":   dict(hf="Qwen/Qwen2.5-3B",
                       diff=str(WEIGHTS_DIR / "qwen_3b" / "diff_vector.pt"),
                       expect_sha="502eca5e74791ae2f46b0cdd5f20dda8bce81f0b343eb705f6818ae52e781f92"),
    "gemma_9b":  dict(hf="google/gemma-2-9b",
                       diff=str(WEIGHTS_DIR / "gemma_9b" / "diff_vector.pt"),
                       expect_sha="cf8cd0acfa88c27d662f23ea29d4d0e4af3b30fcf765d4274081778f36f5e42b"),
    "llama31_8b": dict(hf="meta-llama/Meta-Llama-3.1-8B",
                        diff=str(WEIGHTS_DIR / "llama31_8b" / "diff_vector.pt"),
                        expect_sha="4d705c99e7b3f74696d0362a1bd5a925614f2ba7146922acf6a4fe38016f46c2"),
}

# representative best-per-third layer under the OLD metric (docs/PILOT_PLAN.md)
REP_LAYERS = {
    "qwen_3b":    dict(first=1,  mid=18, last=24),
    "gemma_9b":   dict(first=13, mid=20, last=29),
    "llama31_8b": dict(first=9,  mid=11, last=23),
}


def thirds(n):
    return dict(first=set(range(0, n // 3)),
                mid=set(range(n // 3, 2 * n // 3)),
                last=set(range(2 * n // 3, n)))


def run_condition(label, target, model, tok, diff, diff_sha, bbq_data, device, args,
                   n_layers, all_layers, out_dir, model_key, layer_desc):
    agg_path = os.path.join(out_dir, f"{label}_official_bbq.json")
    jsonl_path = os.path.join(out_dir, f"{label}_examples.jsonl")
    if os.path.exists(agg_path):
        print(f"[skip] {label}")
        return json.load(open(agg_path))

    print(f"\n>>> {label}  target={len(target)}/{n_layers} layers", flush=True)
    match_stats = analyze_matching(model, diff, target) if target else {}
    apply_stats = (apply_delta(model, diff, target, alpha=args.alpha, subtract=False)
                   if target else dict(applied_params=0, bucket_counts={}))
    print(f"  applied={apply_stats['applied_params']} params", flush=True)

    raw_results, trunc = evaluate_batched_logits(
        model, tok, bbq_data, device, args.batch_size, args.max_length)

    if target:
        apply_delta(model, diff, target, alpha=args.alpha, subtract=True)  # unapply

    item_by_key = {(it["category"], it["example_id"]): it for it in bbq_data}
    with open(jsonl_path, "w") as f:
        for r, it in zip(raw_results, bbq_data):
            pred = r["prediction_index"]
            off_label = classify_prediction(it, pred)
            r["bias_type_official"] = off_label
            f.write(json.dumps(dict(
                example_id=it["example_id"], category=r["category"],
                question_polarity=r["question_polarity"],
                context_condition=r["context_condition"],
                prediction_index=pred, correct=r["correct"],
                is_unknown=(off_label == "unknown"),
                is_bias_aligned=(off_label == "biased"),
                model=model_key, layer=layer_desc, condition=label,
            )) + "\n")

    official_metrics = compute_official_metrics([
        dict(category=r["category"], context_condition=r["context_condition"],
             correct=r["correct"], prediction_index=r["prediction_index"],
             bias_type_official=r["bias_type_official"]) for r in raw_results])
    old_metrics = old_bbq_metrics([
        {k: v for k, v in r.items() if k != "bias_type_official"} for r in raw_results
    ])

    payload = dict(
        exp=f"pilot_{label}", model=model_key, model_hf=args.model_hf,
        n_layers=n_layers, condition=label, layer_desc=layer_desc,
        layers_injected=sorted(target), layers_excluded=sorted(all_layers - target),
        alpha=args.alpha, diff_sha256=diff_sha,
        old_metrics=old_metrics, official_metrics=official_metrics,
        trunc_stats=trunc, match_stats=match_stats, apply_stats=apply_stats,
        per_example_jsonl=jsonl_path,
        timestamp=datetime.now().isoformat(),
        env=dict(torch=torch.__version__, transformers=transformers.__version__,
                  cuda=str(torch.version.cuda)),
    )
    with open(agg_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  official ambig bias_score = {official_metrics['ambig']['bias_score']:.4f}"
          f"  (old = {old_metrics['ambig']['bias_score']:.4f})")
    print(f"  -> {agg_path}", flush=True)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", default="",
                    help="Comma-separated subset of condition labels to run "
                         "(base,full,L<n>_first_rep,L<n>_mid_rep,L<n>_last_rep,"
                         "minus_first,minus_mid,minus_last,only_first,only_mid,only_last). "
                         "Empty = run all 11. Used to split one model's conditions "
                         "across two GPUs in parallel. Mutually exclusive with --layers.")
    p.add_argument("--layers", default="",
                    help="Full per-layer sweep instead of the 11-condition light pilot: "
                         "a range like '0-20' (inclusive) or a comma list '0,4,9'. "
                         "Produces one single-layer condition 'L<n>' per layer, same "
                         "apply->eval->unapply + per-example JSONL + old/paper/official "
                         "triple-metric machinery as the light pilot. Mutually exclusive "
                         "with --conditions.")
    args = p.parse_args()
    assert not (args.conditions and args.layers), "--conditions and --layers are mutually exclusive"

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = MODELS[args.model]
    args.model_hf = cfg["hf"]
    subdir = "pilot_sweep" if args.layers else "pilot"
    out_dir = str(RESULTS_DIR / subdir / args.model)
    os.makedirs(out_dir, exist_ok=True)

    diff_sha = sha256_file(cfg["diff"])
    assert diff_sha == cfg["expect_sha"], (
        f"diff SHA mismatch for {args.model}: got {diff_sha}, expected {cfg['expect_sha']} "
        f"-- this must be the SAME ΔW used for the original sweep_thirds/block_removal run")
    print(f"[ok] diff_sha256 matches original sweep run: {diff_sha}")

    n_layers = get_num_layers(resolve_model_path(cfg["hf"]) or cfg["hf"])
    all_layers = set(range(n_layers))
    t = thirds(n_layers)
    rep = REP_LAYERS[args.model]

    diff = torch.load(cfg["diff"], map_location="cpu", weights_only=False)
    model, tok, model_path = load_model_and_tokenizer(
        cfg["hf"], dtype=args.dtype, gpu=args.gpu)
    device = f"cuda:{args.gpu}"

    print("[preload] BBQ...")
    _bbq_dir = os.environ.get(
        "LEVEE_BBQ_DIR",
        str(BBQ_JSONL.parent / "bbq"))
    if os.path.isdir(_bbq_dir) and os.listdir(_bbq_dir):
        bbq_data = load_bbq(_bbq_dir)
    else:
        download_bbq(_bbq_dir)
        bbq_data = load_bbq(_bbq_dir)
    if not bbq_data:
        bbq_data = load_bbq_hf()
    print(f"[preload] BBQ: {len(bbq_data)} samples")
    assert len(bbq_data) == 58492, f"expected full 58,492-item BBQ, got {len(bbq_data)}"

    if args.layers:
        if "-" in args.layers and "," not in args.layers:
            lo, hi = args.layers.split("-")
            layer_list = list(range(int(lo), int(hi) + 1))
        else:
            layer_list = [int(x) for x in args.layers.split(",")]
        assert all(0 <= L < n_layers for L in layer_list), \
            f"layer(s) out of range for n_layers={n_layers}: {layer_list}"
        conditions = [(f"L{L:02d}", {L}, f"L{L}") for L in layer_list]
        print(f"[sweep] running {len(conditions)} single-layer conditions: L{layer_list[0]}..L{layer_list[-1]}")
    else:
        conditions = [
            ("base",        set(),                     "none"),
            ("full",        all_layers,                 "all"),
            (f"L{rep['first']}_first_rep", {rep["first"]}, f"L{rep['first']}"),
            (f"L{rep['mid']}_mid_rep",     {rep["mid"]},   f"L{rep['mid']}"),
            (f"L{rep['last']}_last_rep",   {rep["last"]},  f"L{rep['last']}"),
            ("minus_first", all_layers - t["first"],     "minus_first_third"),
            ("minus_mid",   all_layers - t["mid"],       "minus_mid_third"),
            ("minus_last",  all_layers - t["last"],      "minus_last_third"),
            ("only_first",  t["first"],                  "first_third"),
            ("only_mid",    t["mid"],                    "mid_third"),
            ("only_last",   t["last"],                   "last_third"),
        ]
        if args.conditions:
            wanted = set(args.conditions.split(","))
            conditions = [c for c in conditions if c[0] in wanted]
            print(f"[filter] running {len(conditions)}/{11} conditions: {[c[0] for c in conditions]}")

    common = dict(model=model, tok=tok, diff=diff, diff_sha=diff_sha, bbq_data=bbq_data,
                  device=device, args=args, n_layers=n_layers, all_layers=all_layers,
                  out_dir=out_dir, model_key=args.model)

    results = {}
    for label, target, layer_desc in conditions:
        results[label] = run_condition(label, target, layer_desc=layer_desc, **common)

    print(f"\n=== {args.model} pilot done, {len(results)} conditions -> {out_dir} ===")

    del model, tok, diff
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
