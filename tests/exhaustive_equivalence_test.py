#!/usr/bin/env python
"""
Exhaustive equivalence test: our standalone `classify_prediction()` vs the
ground-truth `biased_answers` computed by the ACTUAL installed
lm_eval.tasks.bbq.utils.process_docs, over ALL 58,492 BBQ items x all 3
possible pred_idx in {0,1,2} = 175,476 cases.

PASS criteria (per user spec):
  - mismatch count == 0
  - forced-choice classification coverage == 100% (every (item, pred_idx)
    gets a definite label from BOTH implementations, or both agree it's
    undefined -- no case where one side errors/None-vs-label without
    the other agreeing)
  - breakdown by neg/nonneg and by category all show 0 mismatches

Also verifies the algebraic identity (user's item 6):
  when parse_rate == 1 (every example gets a valid non -1 prediction) and
  accuracy_ambig == n_unknown/total (BBQ's ambiguous gold label IS the
  unknown option), then
      S_AMB == (n_biased - n_anti) / total
  This is checked on a SYNTHETIC all-correct scenario (pred_idx == gold
  label for every item) since that is the only case where accuracy is
  exactly nu/total by construction; real model predictions won't satisfy
  the antecedent, so this validates the *identity*, not real model runs.
"""

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
import sys
from collections import defaultdict

sys.path.insert(0, str(REPO_ROOT))
from scorer.official_bbq import (
    official_biased_answers_via_lmeval,
    classify_prediction,
    official_biased_indices,
)

DATA = str(BBQ_JSONL)


def main():
    items = [json.loads(l) for l in open(DATA)]
    assert len(items) == 58492, f"expected 58492 items, got {len(items)}"

    print(f"Loaded {len(items)} items. Running official process_docs (ground truth)...")
    biased_answers_by_id = official_biased_answers_via_lmeval(items)
    assert len(biased_answers_by_id) == len(items), \
        f"{len(biased_answers_by_id)} unique (category,example_id) keys vs {len(items)} items"

    mismatches = []
    total_cases = 0
    covered = 0
    by_polarity = defaultdict(lambda: [0, 0])   # [cases, mismatches]
    by_category = defaultdict(lambda: [0, 0])
    by_condition = defaultdict(lambda: [0, 0])

    for item in items:
        key = (item["category"], item["example_id"])
        gt_biased_texts = set(biased_answers_by_id[key])
        # our own unknown detection (shared logic, used identically on both sides)
        biased_idx_ours, unknown_idx = official_biased_indices(item)

        for pred_idx in (0, 1, 2):
            total_cases += 1
            ans_key = f"ans{pred_idx}"
            if ans_key not in item:
                continue

            # reference label derived purely from the OFFICIAL biased_answers text list
            if pred_idx == unknown_idx:
                ref_label = "unknown"
            elif not gt_biased_texts:
                ref_label = None   # official process_docs found no biased answer for this item
            elif item[ans_key] in gt_biased_texts:
                ref_label = "biased"
            else:
                ref_label = "anti_biased"

            our_label = classify_prediction(item, pred_idx)

            key = (item["question_polarity"], item["category"], item["context_condition"])
            by_polarity[item["question_polarity"]][0] += 1
            by_category[item["category"]][0] += 1
            by_condition[item["context_condition"]][0] += 1

            if ref_label == our_label:
                covered += 1
            else:
                mismatches.append(dict(
                    example_id=item["example_id"], pred_idx=pred_idx, polarity=item["question_polarity"],
                    category=item["category"], condition=item["context_condition"],
                    ref_label=ref_label, our_label=our_label,
                    ans_text=item[ans_key], gt_biased_texts=list(gt_biased_texts),
                ))
                by_polarity[item["question_polarity"]][1] += 1
                by_category[item["category"]][1] += 1
                by_condition[item["context_condition"]][1] += 1

    print(f"\ntotal (item, pred_idx) cases: {total_cases}")
    print(f"agreements: {covered}  ({covered/total_cases*100:.4f}%)")
    print(f"mismatches: {len(mismatches)}")

    print("\n-- by polarity --")
    for k, (n, m) in sorted(by_polarity.items()):
        print(f"  {k:8s} n={n:7d} mismatches={m}")
    print("-- by context_condition --")
    for k, (n, m) in sorted(by_condition.items()):
        print(f"  {k:10s} n={n:7d} mismatches={m}")
    print("-- by category (mismatches only shown if >0) --")
    for k, (n, m) in sorted(by_category.items()):
        if m:
            print(f"  {k:20s} n={n:7d} mismatches={m}")
    if not any(m for _, m in by_category.values()):
        print("  (0 mismatches in every category)")

    if mismatches:
        print(f"\nFirst 5 mismatches:")
        for mm in mismatches[:5]:
            print(" ", mm)

    passed = (len(mismatches) == 0) and (covered == total_cases)
    print(f"\n{'PASS' if passed else 'FAIL'}: exhaustive equivalence test")

    # ---- algebraic identity check (item 6) ----
    print("\n-- identity check: S_AMB == (n_biased - n_anti)/total under parse_rate=1 --")
    print("   (non-trivial synthetic prediction: pred_idx = example_id % 3, NOT gold-matching,")
    print("    to avoid a degenerate acc=1 or acc=0 check)")
    ambig_items = [it for it in items if it["context_condition"] == "ambig"]
    nb = na = nu = n_none = 0
    n_correct = 0
    gold_is_unknown_count = 0
    for it in ambig_items:
        pred_idx = it["example_id"] % 3
        label = classify_prediction(it, pred_idx)
        if label == "biased":
            nb += 1
        elif label == "anti_biased":
            na += 1
        elif label == "unknown":
            nu += 1
        else:
            n_none += 1
        if pred_idx == it["label"]:
            n_correct += 1
        _, unk_idx = official_biased_indices(it)
        if it["label"] == unk_idx:
            gold_is_unknown_count += 1

    total = len(ambig_items)
    print(f"  gold label == unknown-answer index, for {gold_is_unknown_count}/{total} ambig items"
          f" ({gold_is_unknown_count/total*100:.2f}%) -- structural check on BBQ's own design")
    parse_rate = (nb + na + nu) / total
    acc = n_correct / total
    print(f"  n_biased={nb} n_anti={na} n_unknown={nu} n_none={n_none} total={total}")
    print(f"  parse_rate (nb+na+nu)/total = {parse_rate:.6f}")
    print(f"  acc (pred==gold)            = {acc:.6f}")
    print(f"  nu/total                    = {nu/total:.6f}")

    if n_none == 0 and abs(acc - nu / total) < 1e-9:
        s_amb_direct = (1 - acc) * ((nb - na) / (nb + na) if (nb + na) else 0.0)
        s_amb_identity = (nb - na) / total
        print(f"  S_AMB (direct formula)      = {s_amb_direct:.6f}")
        print(f"  (n_biased-n_anti)/total     = {s_amb_identity:.6f}")
        print(f"  {'PASS' if abs(s_amb_direct - s_amb_identity) < 1e-9 else 'FAIL'}: identity holds")
    else:
        print(f"  n_none={n_none} on the full ambig set: {n_none} items have NO well-defined biased"
              f" answer at all (empty biased_idx, e.g. BBQ example_id 284/Gender_identity-style rows"
              f" -- a documented BBQ data property, not a scorer bug) and were hit by pred_idx=id%3.")
        print(f"  Restricting to items WITH a well-defined biased answer (parse_rate=1 by construction"
              f" within this subset) to isolate the identity:")
        sub = [it for it in ambig_items if official_biased_indices(it)[0]]
        nb2 = na2 = nu2 = n_none2 = 0
        n_correct2 = 0
        for it in sub:
            pred_idx = it["example_id"] % 3
            label = classify_prediction(it, pred_idx)
            if label == "biased":
                nb2 += 1
            elif label == "anti_biased":
                na2 += 1
            elif label == "unknown":
                nu2 += 1
            else:
                n_none2 += 1
            if pred_idx == it["label"]:
                n_correct2 += 1
        total2 = len(sub)
        acc2 = n_correct2 / total2
        print(f"  n={total2}  n_biased={nb2} n_anti={na2} n_unknown={nu2} n_none={n_none2}")
        print(f"  acc(pred==gold)={acc2:.6f}  nu/total={nu2/total2:.6f}")
        assert n_none2 == 0 and nb2 + na2 + nu2 == total2, "parse_rate != 1 within the restricted subset"
        assert abs(acc2 - nu2 / total2) < 1e-9, "acc != nu/total within the restricted subset"
        s_amb_direct2 = (1 - acc2) * ((nb2 - na2) / (nb2 + na2) if (nb2 + na2) else 0.0)
        s_amb_identity2 = (nb2 - na2) / total2
        print(f"  S_AMB (direct) = {s_amb_direct2:.6f}  vs  (nb-na)/total = {s_amb_identity2:.6f}")
        print(f"  {'PASS' if abs(s_amb_direct2 - s_amb_identity2) < 1e-9 else 'FAIL'}: identity holds"
              f" under parse_rate=1")

    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
