"""
Re-evaluation wrapper: reuses the ORIGINAL (unmodified) GPU forward-pass code
from bias_transfer_v2's eval_bbq_shared.evaluate_batched_logits() -- that
function is not buggy, it correctly records prediction_index/gold_index/
question_polarity/category/context_condition per example. Only the bias
LABEL it originally attached (via determine_bias()) was wrong.

This module: (a) re-labels every prediction with the corrected, verified
scorer.official_bbq.classify_prediction(), (b) saves a full per-example JSONL
(never done before -- see docs/BBQ_POLARITY_BUG.md), (c) aggregates with
scorer.official_bbq.compute_official_metrics().

NOT YET RUN. This requires GPU + model loading; held for explicit go-ahead
per user instruction (steps 8-11: CPU verification must PASS first -- it has,
see tests/exhaustive_equivalence_test.py -- and only a 3-model sentinel pilot
runs before any full 11-model re-evaluation).
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


import json
import os
import sys
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

from scorer.official_bbq import classify_prediction, compute_official_metrics


def evaluate_and_save(
    model, tokenizer, data: List[dict], device: str, batch_size: int,
    max_length: int, *, model_name: str, layer: Optional[str], condition: str,
    out_dir: str, save_logits: bool = True,
) -> dict:
    """Run the (unmodified) forced-choice logit eval, re-label with the
    corrected official scorer, save per-example JSONL + aggregate metrics.

    out_dir/{model_name}_{condition}_examples.jsonl  -- one line per example:
        example_id, category, question_polarity, context_condition,
        prediction_index, correct, is_unknown, is_bias_aligned,
        model, layer, condition, [choice_logits]
    out_dir/{model_name}_{condition}_official_bbq.json -- aggregate metrics
        (compute_official_metrics output), plus the OLD (buggy) aggregate
        for side-by-side comparison.
    """
    from eval_bbq_shared import evaluate_batched_logits, compute_metrics as old_compute_metrics

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{model_name}_{condition}"
    jsonl_path = os.path.join(out_dir, f"{tag}_examples.jsonl")
    agg_path = os.path.join(out_dir, f"{tag}_official_bbq.json")

    # evaluate_batched_logits returns per-example dicts with category,
    # context_condition, question_polarity, gold_index, prediction_index,
    # correct, bias_type (OLD, polarity-blind -- overwritten below),
    # generated_text. It does NOT currently return per-candidate logits;
    # `save_logits` is left as a documented gap (see NOTE below) rather than
    # silently claimed -- evaluate_batched_logits would need a small,
    # separate patch to also return `scores_by_item` per example if logits
    # are required. Not done here: no GPU run has happened yet to need it.
    raw_results, trunc = evaluate_batched_logits(
        model, tokenizer, data, device, batch_size, max_length)

    item_by_key = {(it["category"], it["example_id"]): it for it in data}

    with open(jsonl_path, "w") as f:
        for r, it in zip(raw_results, data):
            pred = r["prediction_index"]
            official_label = classify_prediction(it, pred)
            r["bias_type_current_formula"] = r.pop("bias_type")  # old, polarity-blind label
            r["bias_type_official"] = official_label
            row = dict(
                example_id=it["example_id"], category=r["category"],
                question_polarity=r["question_polarity"],
                context_condition=r["context_condition"],
                prediction_index=pred, correct=r["correct"],
                is_unknown=(official_label == "unknown"),
                is_bias_aligned=(official_label == "biased"),
                model=model_name, layer=layer, condition=condition,
            )
            f.write(json.dumps(row) + "\n")

    for r in raw_results:
        pass  # raw_results already mutated in place above with bias_type_official

    official_metrics = compute_official_metrics(
        [dict(category=r["category"], context_condition=r["context_condition"],
              correct=r["correct"], prediction_index=r["prediction_index"],
              bias_type_official=r["bias_type_official"]) for r in raw_results])
    old_metrics = old_compute_metrics([
        dict(r, bias_type=r["bias_type_current_formula"]) for r in raw_results])

    payload = dict(
        official_metrics=official_metrics, old_metrics=old_metrics,
        trunc_stats=trunc, model=model_name, layer=layer, condition=condition,
        timestamp=datetime.now().isoformat(),
        per_example_jsonl=jsonl_path,
    )
    with open(agg_path, "w") as f:
        json.dump(payload, f, indent=2)

    return payload
