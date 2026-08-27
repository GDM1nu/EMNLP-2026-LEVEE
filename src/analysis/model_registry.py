"""
Shared model metadata for all analysis and plot scripts.
"""

ARCHIVE = str(ARCHIVE)

MODELS = {
    "qwen_0.5b": dict(
        label="Qwen 0.5B", size_b=0.5, n_layers=24, color="#9ecae1",
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_0.5b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_0.5b/baselines/baseline_instruct_bbq.json",
        flat_metrics=False,
    ),
    "qwen_1.5b": dict(
        label="Qwen 1.5B", size_b=1.5, n_layers=28, color="#6baed6",
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_1.5b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_1.5b/baselines/baseline_instruct_bbq.json",
        flat_metrics=False,
    ),
    "qwen_3b": dict(
        label="Qwen 3B", size_b=3.0, n_layers=36, color="#2171b5",
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_3b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_3b/baselines/baseline_instruct_bbq.json",
        flat_metrics=False,
    ),
    "qwen_14b": dict(
        label="Qwen 14B", size_b=14.0, n_layers=48, color="#084594",
        base_json=f"{ARCHIVE}/scaling_experiment/qwen_14b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/qwen_14b/baselines/baseline_instruct_bbq.json",
        flat_metrics=False,
    ),
    "gemma_2b": dict(
        label="Gemma 2B", size_b=2.0, n_layers=26, color="#74c476",
        base_json=f"{ARCHIVE}/scaling_experiment/gemma_2b/baselines/baseline_gemma2_2b_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/gemma_2b/baselines/baseline_gemma2_2b_it_bbq.json",
        flat_metrics=False,
    ),
    "gemma_9b": dict(
        label="Gemma 9B", size_b=9.0, n_layers=42, color="#238b45",
        base_json=f"{ARCHIVE}/scaling_experiment/gemma_9b/baselines/baseline_base_bbq.json",
        inst_json=f"{ARCHIVE}/scaling_experiment/gemma_9b/baselines/baseline_instruct_bbq.json",
        flat_metrics=False,
    ),
    "llama32_1b": dict(
        label="LLaMA 3.2-1B", size_b=1.0, n_layers=16, color="#fdae6b",
        base_json=None,
        inst_json=None,
        flat_metrics=False,
    ),
    "llama32_3b": dict(
        label="LLaMA 3.2-3B", size_b=3.0, n_layers=28, color="#fd8d3c",
        base_json=None,
        inst_json=None,
        flat_metrics=False,
    ),
    "llama3_8b": dict(
        label="LLaMA 3-8B", size_b=8.0, n_layers=32, color="#d94801",
        base_json=f"{ARCHIVE}/llama_experiment/results/baselines_logits/bbq_metrics_llama3_8b_base_logits.json",
        inst_json=None,
        flat_metrics=True,
    ),
}

MODEL_ORDER = [
    "qwen_0.5b", "qwen_1.5b", "llama32_1b",
    "gemma_2b", "qwen_3b", "llama32_3b",
    "gemma_9b", "llama3_8b", "qwen_14b",
]


def load_baseline(alias: str) -> float | None:
    import json, os
    cfg = MODELS[alias]
    path = cfg.get("base_json")
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    if cfg["flat_metrics"]:
        return d["ambig"]["bias_score"]
    return d["metrics"]["ambig"]["bias_score"]


def load_instruct(alias: str) -> float | None:
    import json, os
    cfg = MODELS[alias]
    path = cfg.get("inst_json")
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    if cfg["flat_metrics"]:
        return d["ambig"]["bias_score"]
    return d["metrics"]["ambig"]["bias_score"]
