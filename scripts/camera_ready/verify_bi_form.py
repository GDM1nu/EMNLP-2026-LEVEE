#!/usr/bin/env python
r"""S_L-BBQ = (1 - Acc_amb) * (1/N) * sum_i b_i   가 기존 표기와 수치적으로 동일한가.

b_i = +1 (biased) / -1 (anti-biased) / 0 (UNKNOWN)

핵심 쟁점은 N 의 정의다.
  N_assigned = n_s + n_a + n_n   (매퍼가 라벨을 붙인 응답 수)
  N_total    = ambiguous 문항 총수 (29,246)
legacy 매퍼는 일부 응답에 라벨을 붙이지 못하므로 두 값이 다르다.
따라서 두 후보를 모두 계산해 어느 쪽이 기존 bias_score 와 일치하는지 본다.
"""
import json, os, glob

# Paths resolve to the repository root by default; override with env vars.
# Pre-computed results are not redistributed here (see results/README.md).
ROOT = os.environ.get("LEVEE_RESULTS_ROOT") or os.environ.get(
    "LEVEE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONDS = ["full","minus_first","minus_mid","minus_last","only_first","only_mid","only_last"]
B, K = "results/block_removal", "results/rebuttal3/final_58k_20260711/results_full58k"
MODELS = [("Qwen2.5-0.5B",f"{B}/qwen_0.5b"),("Qwen2.5-1.5B",f"{B}/qwen_1.5b"),
          ("Qwen2.5-3B",f"{B}/qwen_3b"),("Qwen2.5-14B",f"{B}/qwen_14b"),
          ("Gemma-2-2B",f"{B}/gemma_2b"),("Gemma-2-9B",f"{B}/gemma_9b"),
          ("Gemma-3-12B",f"{B}/gemma3_12b"),("Llama-3.2-1B",f"{B}/llama32_1b"),
          ("Llama-3.2-3B",f"{B}/llama32_3b"),("Llama-3.1-8B",f"{B}/llama31_8b"),
          ("Llama-3-8B",f"{B}/llama3_8b"),("Mistral-7B",f"{B}/mistral_7b"),
          ("Mistral-24B",f"{B}/mistral_small_24b"),("Qwen2.5-32B",f"{K}/qwen2.5-32b"),
          ("Gemma-4-31B",f"{K}/gemma-4-31b")]

TOL = 1e-6          # 저장값이 round(...,6) 이므로 그 해상도까지만 요구한다
n_ok_assigned = n_ok_total = n = 0
worst_a = worst_t = 0.0
rows = []
for name, d in MODELS:
    for c in CONDS:
        p = os.path.join(ROOT, d, f"block_{c}_bbq.json")
        if not os.path.exists(p): continue
        a = json.load(open(p))["metrics"]["ambig"]
        ns, na, nu, acc, stored = (a["n_biased"], a["n_anti_biased"], a["n_unknown"],
                                   a["accuracy"], a["bias_score"])
        sum_b = ns - na                      # (+1)*ns + (-1)*na + 0*nu
        N_assigned = ns + na + nu
        N_total    = a["total"]
        v_assigned = (1 - acc) * sum_b / N_assigned
        v_total    = (1 - acc) * sum_b / N_total
        da, dt = abs(v_assigned - stored), abs(v_total - stored)
        worst_a = max(worst_a, da); worst_t = max(worst_t, dt)
        n_ok_assigned += da <= TOL; n_ok_total += dt <= TOL; n += 1
        rows.append((name, c, stored, v_assigned, v_total, N_assigned, N_total))

if n == 0:
    raise SystemExit(
        f"결과 파일을 찾지 못했다 (ROOT={ROOT}).\n"
        "  검사가 통과한 것이 아니라 실행되지 않은 것이다.\n"
        "  LEVEE_RESULTS_ROOT 를 pre-computed results 가 있는 경로로 지정할 것.")

print("== 집계 수준 검증 (기존 legacy 결과 전체) ==")
print(f"조건 총수 {n}")
print(f"  N = n_s+n_a+n_n (배정된 응답) 로 계산 -> 일치 {n_ok_assigned}/{n}   최대오차 {worst_a:.2e}")
print(f"  N = ambiguous 총수            로 계산 -> 일치 {n_ok_total}/{n}   최대오차 {worst_t:.2e}")
ex = rows[0]
print(f"\n예: {ex[0]} / {ex[1]}   stored={ex[2]:.6f}  N_assigned={ex[5]:,}({ex[3]:.6f})  N_total={ex[6]:,}({ex[4]:.6f})")
