#!/usr/bin/env python3
"""
MMLU Evaluation with Diff Vector — Qwen / Gemma / LLaMA 3.0
lm_eval HFLM 상속, in-memory 모델 직접 주입 (저장 없음)
조건: baseline / mid_8 / all / minus_mid_8
"""
import os, re, json, argparse, torch, gc, hashlib
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    compute_layer_groups, get_num_layers, load_model_and_tokenizer,
    resolve_model_path, apply_delta, sha256_file,
)

def parse_mmlu_acc(results):
    if "results" not in results: return None
    r = results["results"]
    for key in ["mmlu", "mmlu_average"]:
        if key in r:
            for acc_key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if acc_key in r[key]: return float(r[key][acc_key])
    scores = []
    for k, v in r.items():
        if "mmlu" in k:
            for acc_key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if acc_key in v: scores.append(float(v[acc_key])); break
    return sum(scores) / len(scores) if scores else None

def make_injected_lm(model, tokenizer, gpu, batch_size):
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.model import CacheHook

    class InjectedHFLM(HFLM):
        def __init__(self, model, tokenizer, gpu, batch_size):
            self._rank               = 0
            self._world_size         = 1
            self.cache_hook          = CacheHook(None)
            self._model              = model
            self._device             = torch.device(f"cuda:{gpu}")
            self._config             = model.config
            self.tokenizer           = tokenizer
            self._max_length         = getattr(model.config, "max_position_embeddings", 2048)
            self.custom_prefix_token_id = None
            self.batch_size_per_gpu  = batch_size
            self.batch_sizes         = {}
            self.batch_schedule      = 1
            self.max_batch_size      = 64
            self.add_bos_token       = getattr(tokenizer, "add_bos_token", False)
            self.truncation          = False
            self.logits_cache        = True
            self.vocab_size          = tokenizer.vocab_size
            self.pretrained          = model
            self.delta               = None
            self.peft                = None
            self.revision            = "main"
            self.think_end_token     = None
            self.chat_template_args  = None
            self.softmax_dtype       = None
            self.mixed_precision_dtype = None

        def _detect_batch_size(self, requests=None, pos=0):
            return self.batch_size_per_gpu

    return InjectedHFLM(model, tokenizer, gpu, batch_size)

def run_single(exp_name, model_name, diff, target_layers, subtract, args, result_dir):
    out_json = os.path.join(result_dir, f"{exp_name}_summary.json")
    if os.path.exists(out_json):
        print(f"[SKIP] {exp_name}"); return

    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'='*60}\n[EXP] {exp_name}\n{'='*60}")

    model, tokenizer, _ = load_model_and_tokenizer(
        model_name, gpu=args.gpu, dtype=args.dtype)

    if diff is not None and target_layers:
        applied = apply_delta(model, diff, target_layers, subtract=subtract)
        print(f"  Applied {applied} params")
    else:
        print(f"  No diff (baseline)")

    print(f"  Injecting into lm_eval HFLM...")
    lm = make_injected_lm(model, tokenizer, args.gpu, args.batch_size)

    print(f"  Running MMLU 5-shot...")
    from lm_eval import evaluator
    from lm_eval.tasks import get_task_dict

    from lm_eval.tasks import TaskManager
    tm        = TaskManager()
    task_dict = get_task_dict(["mmlu"], task_manager=tm)
    def _set_fewshot(d, n=5):
        for v in d.values():
            if isinstance(v, dict): _set_fewshot(v, n)
            elif hasattr(v, "config"): v.config.num_fewshot = n
    _set_fewshot(task_dict)
    lm_results = evaluator.evaluate(
        lm=lm, task_dict=task_dict, log_samples=False)

    mmlu_acc = parse_mmlu_acc(lm_results)
    print(f"  MMLU acc: {mmlu_acc:.4f}" if mmlu_acc else "  MMLU acc: 파싱 실패")

    subtask_scores = {}
    if "results" in lm_results:
        for k, v in lm_results["results"].items():
            if "mmlu" in k:
                for acc_key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                    if acc_key in v: subtask_scores[k] = float(v[acc_key]); break

    summary = {
        "exp": exp_name, "model": model_name,
        "target_layers": sorted(target_layers) if target_layers else [],
        "subtract": subtract, "timestamp": datetime.now().isoformat(),
        "mmlu_acc": mmlu_acc, "subtask_scores": subtask_scores,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_json}")
    del model, tokenizer, lm; gc.collect(); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True, help="HF model name (base)")
    ap.add_argument("--diff_path",  required=True, help="diff vector .pt 경로")
    ap.add_argument("--merged_model", default=None, help="minus_mid_8용 merged 경로")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--gpu",        type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype",      default="bf16")
    ap.add_argument("--alpha",      type=float, default=1.0)
    ap.add_argument("--layer_condition", default=None,
                    choices=["first_4","first_8","mid_8","last_8","last_4",
                             "all","minus_mid_8","minus_first_4",
                             "minus_first_8","minus_last_8","minus_last_4"],
                    help="단일 조건 실행 시 지정. 없으면 기존 4개 조건 전체 실행")
    args = ap.parse_args()

    n_layers = get_num_layers(args.model)
    groups   = compute_layer_groups(n_layers)
    print(f"Model: {args.model} | n_layers={n_layers} | mid_8={groups['mid_8']}")

    print(f"Loading diff: {args.diff_path}")
    diff     = torch.load(args.diff_path, map_location="cpu", weights_only=False)
    diff_sha = sha256_file(args.diff_path)
    print(f"  {len(diff)} tensors | {diff_sha[:16]}...")

    if args.layer_condition:
        minus_conds = ["minus_first_4","minus_first_8","minus_mid_8","minus_last_8","minus_last_4"]
        is_minus = args.layer_condition in minus_conds
        if is_minus:
            base_key = args.layer_condition.replace("minus_","")
            target = set(i for i in range(n_layers) if i not in set(groups.get(base_key, groups["mid_8"])))
            inst_model = args.merged_model if args.merged_model else args.model
            conditions = [(args.layer_condition, inst_model, diff, target, True)]
        else:
            conditions = [(args.layer_condition, args.model, diff, set(groups[args.layer_condition]), False)]
    else:
        conditions = [
            ("baseline",          args.model, None, set(),                False),
            ("groupA_mid_8",      args.model, diff, set(groups["mid_8"]), False),
            ("groupA_all",        args.model, diff, set(groups["all"]),   False),
        ]
    if args.merged_model and os.path.exists(args.merged_model):
        conditions.append(
            ("groupB_minus_mid_8", args.merged_model, diff, set(groups["mid_8"]), True))
    else:
        print(f"⚠️  merged_model 없음, minus_mid_8 스킵")

    model_key = args.model.replace("/", "_")
    for cond_name, model_name, d, target_layers, subtract in conditions:
        run_single(f"{model_key}_{cond_name}",
                   model_name, d, target_layers, subtract, args,
                   args.result_dir)

    print(f"\n{'='*60}\n최종 요약\n{'='*60}")
    for cond_name, *_ in conditions:
        model_key = args.model.replace("/", "_")
        exp_name  = f"{model_key}_{cond_name}"
        out_json  = os.path.join(args.result_dir, f"{exp_name}_summary.json")
        if os.path.exists(out_json):
            d = json.load(open(out_json))
            acc = d.get("mmlu_acc")
            print(f"  {cond_name:<25} MMLU={acc:.4f}" if acc else f"  {cond_name:<25} MMLU=None")

if __name__ == "__main__":
    main()
