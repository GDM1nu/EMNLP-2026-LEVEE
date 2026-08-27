#!/usr/bin/env python3
"""
Validate all experiment result files.

Checks:
  - Expected files exist (correct count per model)
  - JSON parseable
  - Metrics in valid range [0, 1]
  - sweep_summary.json present where expected
  - Prints a summary table with pass/warn/fail per experiment phase
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path("results")

# n_layers only (summary checked separately via summary_name)
SWEEP_THIRDS_EXPECTED = {
    "qwen_0.5b": 24, "qwen_1.5b": 28, "qwen_3b": 36, "qwen_14b": 48,
    "gemma_2b": 26, "gemma_9b": 42,
    "llama32_1b": 16, "llama32_3b": 28, "llama3_8b": 32,
}

# norm_matched×3 + shuffled×3 + zero×1 = 7 conditions, each has n_layers files
CONTROL_SWEEP_EXPECTED = {
    "qwen_0.5b": 7 * 24, "qwen_1.5b": 7 * 28, "llama32_1b": 7 * 16,
    "qwen_3b": 7 * 36, "llama32_3b": 7 * 28, "gemma_2b": 7 * 26,
    "gemma_9b": 7 * 42, "llama3_8b": 7 * 32, "qwen_14b": 7 * 48,
}

# n_layers + 1 summary
CUMULATIVE_EXPECTED = {
    "qwen_3b": 36, "gemma_2b": 26, "llama32_3b": 28,
    "gemma_9b": 42, "llama3_8b": 32,
}

SIGNFLIP_EXPECTED = {
    "qwen_3b": 36, "llama32_1b": 16, "llama32_3b": 28,
    "gemma_2b": 26, "llama3_8b": 32, "gemma_9b": 42, "qwen_14b": 48,
}

OK = "\033[32m✅\033[0m"
WARN = "\033[33m⚠️\033[0m"
FAIL = "\033[31m❌\033[0m"


def check_json_metrics(path: Path) -> list[str]:
    errors = []
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return [f"JSON parse error: {e}"]

    m = d.get("metrics", {})
    for split in ("overall", "ambig", "disambig"):
        if split not in m:
            continue
        bs = m[split].get("bias_score")
        acc = m[split].get("accuracy")
        if bs is not None and not (0.0 <= bs <= 1.0):
            errors.append(f"{split}.bias_score={bs:.4f} out of range")
        if acc is not None and not (0.0 <= acc <= 1.0):
            errors.append(f"{split}.accuracy={acc:.4f} out of range")
    return errors


def check_phase(phase_name: str, result_dir: Path, expected: dict[str, int],
                glob_pattern: str, summary_name: str | None = None) -> bool:
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    all_ok = True
    for alias, exp_count in sorted(expected.items()):
        model_dir = result_dir / alias
        if not model_dir.exists():
            print(f"  {FAIL} {alias}: directory missing")
            all_ok = False
            continue

        files = list(model_dir.glob(glob_pattern))
        count = len(files)

        # check summary if required
        summary_ok = True
        if summary_name:
            summary_path = model_dir / summary_name
            if not summary_path.exists():
                summary_ok = False

        # check for metric errors in a sample (all files if <50, else first+last 5)
        check_files = files if len(files) <= 50 else files[:5] + files[-5:]
        metric_errors = []
        for f in check_files:
            errs = check_json_metrics(f)
            if errs:
                metric_errors.append(f"{f.name}: {', '.join(errs)}")

        if count == exp_count and summary_ok and not metric_errors:
            print(f"  {OK} {alias}: {count}/{exp_count} files")
        elif count == 0:
            print(f"  {FAIL} {alias}: 0/{exp_count} files — NOT STARTED")
            all_ok = False
        elif count < exp_count:
            pct = count / exp_count * 100
            status = WARN if count > 0 else FAIL
            print(f"  {status} {alias}: {count}/{exp_count} ({pct:.0f}%) — INCOMPLETE")
            all_ok = False
        else:
            print(f"  {OK} {alias}: {count}/{exp_count} files", end="")

        if not summary_ok:
            print(f"    {WARN} {summary_name} missing")
            all_ok = False
        if metric_errors:
            for e in metric_errors[:3]:
                print(f"    {FAIL} {e}")
            if len(metric_errors) > 3:
                print(f"    ... and {len(metric_errors)-3} more errors")
            all_ok = False

        if count == exp_count and summary_ok and not metric_errors:
            pass  # newline already printed
        else:
            pass

    return all_ok


def main():
    print("\n" + "=" * 60)
    print("  BIAS TRANSFER v2 — Result Validation")
    print("=" * 60)

    results = {}

    results["sweep_thirds"] = check_phase(
        "Phase 1: sweep_thirds (BBQ per-layer)",
        BASE / "sweep_thirds",
        SWEEP_THIRDS_EXPECTED,
        glob_pattern="sweep_L*_bbq.json",
        summary_name="sweep_summary.json",
    )

    results["control_sweep"] = check_phase(
        "Phase 2: control_sweep (direction specificity)",
        BASE / "control_sweep",
        CONTROL_SWEEP_EXPECTED,
        glob_pattern="ctrl_*_bbq.json",
        summary_name=None,
    )

    results["cumulative"] = check_phase(
        "Phase 3: cumulative (formation process)",
        BASE / "cumulative",
        CUMULATIVE_EXPECTED,
        glob_pattern="cumul_L*_bbq.json",
        summary_name="cumulative_summary.json",
    )

    results["signflip"] = check_phase(
        "Phase 4: signflip (causal direction)",
        BASE / "signflip",
        SIGNFLIP_EXPECTED,
        glob_pattern="sweep_L*_bbq.json",
        summary_name="sweep_summary.json",
    )

    # multieval
    print(f"\n{'='*60}")
    print("  Phase 5: sweep_multieval (CrowS + StereoSet)")
    print(f"{'='*60}")
    multieval_dir = BASE / "sweep_multieval"
    if not multieval_dir.exists() or not any(multieval_dir.iterdir()):
        print(f"  {FAIL} NOT STARTED — 0/9 models")
        results["multieval"] = False
    else:
        print(f"  {WARN} PARTIAL")
        results["multieval"] = False

    # summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for phase, ok in results.items():
        icon = OK if ok else FAIL
        print(f"  {icon} {phase}")

    n_fail = sum(1 for v in results.values() if not v)
    if n_fail == 0:
        print(f"\n  All phases complete. Ready for analysis.")
    else:
        print(f"\n  {n_fail} phase(s) incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
