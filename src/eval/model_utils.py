#!/usr/bin/env python3
"""
model_utils.py — Shared utilities for diff-vector bias transfer scaling experiments.

Provides:
    - Automatic layer-group computation (proportional to model depth)
    - Diff-vector computation, application, and random-control generation
    - Reproducible model loading with explicit dtype / device control

All evaluation logic (BBQ, CrowS-Pairs, StereoSet) lives in the respective
eval_*.py scripts; this module is deliberately benchmark-agnostic.

Author : KDM
Project: Diff Vector Bias Transfer (EMNLP submission)
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



import gc
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_CACHE = _os.environ.get("HF_HOME", str(REPO_ROOT / "hub"))
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# ---------------------------------------------------------------------------
# Layer-group computation
# ---------------------------------------------------------------------------

def get_num_layers(model_name_or_path: str) -> int:
    """Return ``num_hidden_layers`` from the model config."""
    cfg = AutoConfig.from_pretrained(
        model_name_or_path, cache_dir=HF_CACHE, trust_remote_code=True
    )
    # Gemma3 (multimodal): num_hidden_layers is nested inside text_config
    if hasattr(cfg, "text_config"):
        tc = cfg.text_config
        if isinstance(tc, dict):
            return tc["num_hidden_layers"]
        return tc.num_hidden_layers
    return cfg.num_hidden_layers


def compute_layer_groups(n_layers: int) -> Dict[str, List[int]]:
    """Compute layer groups proportional to the model depth.

    Legacy groups (kept for backward compatibility):
        first_4  ≈ first 14%      first_8  ≈ first 29%
        mid_8    ≈ middle 28%     last_8   ≈ last 29%
        last_4   ≈ last 14%       all / minus_mid_8

    Equal thirds (no gaps, full coverage):
        first_third = L0      ~ L(N//3 - 1)
        mid_third   = L(N//3) ~ L(2*N//3 - 1)
        last_third  = L(2*N//3) ~ L(N-1)
        Remainder (N % 3 layers) absorbed into last_third.

    Returns a dict  ``{group_name: [layer_indices]}``.
    """
    first_4_end = max(4, round(n_layers * 0.14))
    first_8_end = max(8, round(n_layers * 0.29))

    mid_start = round(n_layers * 0.36)
    mid_end   = round(n_layers * 0.64)
    if mid_end - mid_start < 4:          # guarantee ≥4 layers
        mid_start = n_layers // 2 - 2
        mid_end   = n_layers // 2 + 2

    last_8_start = n_layers - max(8, round(n_layers * 0.29))
    last_4_start = n_layers - max(4, round(n_layers * 0.14))

    # equal thirds — no gaps, full coverage
    t1 = n_layers // 3
    t2 = 2 * n_layers // 3

    groups = {
        "first_4":     list(range(0, first_4_end)),
        "first_8":     list(range(0, first_8_end)),
        "mid_8":       list(range(mid_start, mid_end)),
        "last_8":      list(range(last_8_start, n_layers)),
        "last_4":      list(range(last_4_start, n_layers)),
        "all":         list(range(0, n_layers)),
        "first_third": list(range(0, t1)),
        "mid_third":   list(range(t1, t2)),
        "last_third":  list(range(t2, n_layers)),
    }
    mid_set = set(groups["mid_8"])
    groups["minus_mid_8"]   = [i for i in range(n_layers) if i not in mid_set]
    first_4_set = set(groups["first_4"])
    first_8_set = set(groups["first_8"])
    last_8_set  = set(groups["last_8"])
    last_4_set  = set(groups["last_4"])
    groups["minus_first_4"] = [i for i in range(n_layers) if i not in first_4_set]
    groups["minus_first_8"] = [i for i in range(n_layers) if i not in first_8_set]
    groups["minus_last_8"]  = [i for i in range(n_layers) if i not in last_8_set]
    groups["minus_last_4"]  = [i for i in range(n_layers) if i not in last_4_set]
    mid_third_set   = set(groups["mid_third"])
    first_third_set = set(groups["first_third"])
    last_third_set  = set(groups["last_third"])
    groups["minus_mid_third"]   = [i for i in range(n_layers) if i not in mid_third_set]
    groups["minus_first_third"] = [i for i in range(n_layers) if i not in first_third_set]
    groups["minus_last_third"]  = [i for i in range(n_layers) if i not in last_third_set]
    return groups


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def resolve_model_path(model_name: str) -> str:
    """Return local snapshot path if cached, else the HF repo id."""
    escaped = model_name.replace("/", "--")
    snap_dir = os.path.join(HF_CACHE, f"models--{escaped}", "snapshots")
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            return os.path.join(snap_dir, snaps[-1])
    return model_name                       # will trigger HF download


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    gpu: int = 0,
    quantize: Optional[str] = None,        # "4bit" | "8bit" | None
    multi_gpu: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load a causal-LM + tokenizer with explicit dtype / device.

    Parameters
    ----------
    quantize : str, optional
        ``"4bit"`` or ``"8bit"`` for bitsandbytes quantisation (70B).
        Default ``None`` loads in *dtype* precision.
    multi_gpu : bool, optional
        Use device_map="auto" to spread across all available GPUs (full precision).
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model_path  = resolve_model_path(model_name)

    load_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        cache_dir=HF_CACHE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    elif multi_gpu:
        load_kwargs["dtype"] = torch_dtype
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch_dtype
        load_kwargs["device_map"] = {"": gpu}

    print(f"[load] {model_name}  (dtype={dtype}, gpu={gpu}, quant={quantize}, multi_gpu={multi_gpu})")
    print(f"       path = {model_path}")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,  # HF repo id — avoids incomplete snapshot issues
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_path


# ---------------------------------------------------------------------------
# Diff-vector operations
# ---------------------------------------------------------------------------

def _layer_idx(name: str) -> Optional[int]:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _bucketize(name: str) -> str:
    if "self_attn" in name or ".attn" in name:
        return "attn"
    if ".mlp" in name or "feed_forward" in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def analyze_matching(
    model: AutoModelForCausalLM,
    diff: Dict[str, torch.Tensor],
    target_layers: Set[int],
) -> dict:
    """Report how many diff keys match model parameters in *target_layers*."""
    total = match = match_in = 0
    bucket: Counter = Counter()
    for n, _ in model.named_parameters():
        total += 1
        if n in diff:
            match += 1
            li = _layer_idx(n)
            if li is not None and li in target_layers:
                match_in += 1
                bucket[_bucketize(n)] += 1
    return dict(
        total_params=total, matched_total=match,
        matched_in_layers=match_in, bucket_counts=dict(bucket),
    )


@torch.no_grad()
def apply_delta(
    model: AutoModelForCausalLM,
    diff: Dict[str, torch.Tensor],
    target_layers: Set[int],
    alpha: float = 1.0,
    subtract: bool = False,
) -> dict:
    """Add (or subtract) the diff vector to *model* for layers in *target_layers*."""
    applied = 0
    bucket: Counter = Counter()
    for n, p in model.named_parameters():
        d = diff.get(n)
        if d is None:
            continue
        li = _layer_idx(n)
        if li is None or li not in target_layers:
            continue
        upd = d.to(device=p.device, dtype=p.dtype)
        if subtract:
            p.data.sub_(alpha * upd)
        else:
            p.data.add_(alpha * upd)
        applied += 1
        bucket[_bucketize(n)] += 1
    return dict(applied_params=applied, bucket_counts=dict(bucket))


def compute_and_save_diff(
    base_model: str,
    instruct_model: str,
    output_path: str,
    dtype: str = "bfloat16",
) -> None:
    """Compute Δ = Instruct − Base and save as ``.pt``."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"[skip] diff already exists: {output_path}")
        return

    save_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    # --- base ---
    print(f"[diff] loading base: {base_model}")
    base_path = resolve_model_path(base_model)
    m_base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
        cache_dir=HF_CACHE, trust_remote_code=True,
    )
    base_sd = {n: p.data.clone() for n, p in m_base.named_parameters()}
    del m_base; gc.collect()

    # --- instruct ---
    print(f"[diff] loading instruct: {instruct_model}")
    inst_path = resolve_model_path(instruct_model)
    m_inst = AutoModelForCausalLM.from_pretrained(
        inst_path, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
        cache_dir=HF_CACHE, trust_remote_code=True,
    )
    inst_sd = {n: p.data.clone() for n, p in m_inst.named_parameters()}
    del m_inst; gc.collect()

    # --- diff ---
    common = set(base_sd) & set(inst_sd)
    diff = {k: (inst_sd[k] - base_sd[k]).to(save_dtype) for k in common}
    del base_sd, inst_sd; gc.collect()

    torch.save(diff, output_path)
    meta = dict(
        base=base_model, instruct=instruct_model,
        n_params=len(diff), sha256=sha256_file(output_path),
        timestamp=datetime.now().isoformat(),
    )
    with open(output_path.replace(".pt", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[diff] saved {len(diff)} tensors → {output_path}")


# ---------------------------------------------------------------------------
# Random-control generation
# ---------------------------------------------------------------------------

def make_control_diff(
    diff: Dict[str, torch.Tensor],
    control_type: str,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Generate a control diff vector.

    Parameters
    ----------
    control_type : ``"norm_matched"`` | ``"shuffled"`` | ``"zero"``
    """
    rng = torch.Generator().manual_seed(seed)
    ctrl: Dict[str, torch.Tensor] = {}
    for key, tensor in diff.items():
        t = tensor.float()
        if control_type == "norm_matched":
            noise = torch.randn(t.shape, generator=rng, dtype=torch.float32)
            n_orig = t.norm()
            n_noise = noise.norm()
            if n_noise > 0:
                noise = noise * (n_orig / n_noise)
            ctrl[key] = noise.to(tensor.dtype)
        elif control_type == "shuffled":
            flat = t.view(-1)
            idx = torch.randperm(flat.numel(), generator=rng)
            ctrl[key] = flat[idx].view(t.shape).to(tensor.dtype)
        elif control_type == "zero":
            ctrl[key] = torch.zeros_like(tensor)
        else:
            raise ValueError(f"unknown control_type: {control_type}")
    return ctrl
