# Pre-computed Results

This directory contains pre-computed experimental results (JSON only, ~40 MB)
used to generate the figures and tables in the paper. Model weights, the BBQ
data, and per-example predictions are not included. Local filesystem prefixes
were stripped from the `model_path` fields; the remaining
`models--<org>--<name>/snapshots/<hash>` string identifies the exact HuggingFace
snapshot that was used.

Generated outputs (`results/paper_validation/`, `results/analysis/plots/`) are
git-ignored; rerun `python scripts/paper_validation.py` to recreate them.

## Directory Structure

```
results/
├── sweep_thirds/          # Per-layer ΔW injection results (Figure 1, 2, 3)
│   └── {model}/
│       └── sweep_L{idx}_bbq.json
├── block_removal/         # Block removal results (Table 4, Figure 8)
│   └── {model}/
│       ├── block_full_bbq.json
│       ├── block_minus_first_bbq.json
│       ├── block_minus_mid_bbq.json
│       ├── block_minus_last_bbq.json
│       ├── block_only_first_bbq.json
│       ├── block_only_mid_bbq.json
│       ├── block_only_last_bbq.json
│       ├── block_full_mmlu.json
│       └── block_minus_{first,mid,last}_mmlu.json
├── signflip/              # −ΔW injection results (Figure 13)
│   └── {model}/
│       └── sweep_L{idx}_bbq.json
├── control_sweep/         # Zero-injection and norm-matched controls (Table 3)
│   └── {model}/
│       └── ctrl_zero_L{idx}_bbq.json
├── sweep_multieval/       # CrowS-Pairs and StereoSet sweep (Appendix)
│   └── {model}/
│       ├── sweep_L{idx}_crows.json
│       └── sweep_L{idx}_stereoset.json
└── analysis/              # Derived statistics
    ├── probe_bias_corr.json
    ├── cosine_sim.json
    └── plots/
```

## JSON Schema

### `sweep_L{idx}_bbq.json`

```json
{
  "layer_idx": 12,
  "model": "qwen_3b",
  "metrics": {
    "ambig": {
      "bias_score": 0.0277,
      "accuracy": 0.72
    },
    "disambig": {
      "bias_score": 0.013,
      "accuracy": 0.85
    }
  }
}
```

### `block_{condition}_mmlu.json`

```json
{
  "model": "gemma_9b",
  "condition": "minus_first",
  "mmlu_acc": 0.312
}
```

## Models

Results are stored per model using the following aliases:

| Alias | Model |
|-------|-------|
| `qwen_0.5b` | Qwen2.5-0.5B-Instruct |
| `qwen_1.5b` | Qwen2.5-1.5B-Instruct |
| `qwen_3b` | Qwen2.5-3B-Instruct |
| `qwen_14b` | Qwen2.5-14B-Instruct |
| `gemma_2b` | Gemma-2-2B-it |
| `gemma_9b` | Gemma-2-9B-it |
| `gemma3_12b` | Gemma-3-12B-it |
| `llama32_1b` | Llama-3.2-1B-Instruct |
| `llama32_3b` | Llama-3.2-3B-Instruct |
| `llama3_8b` | Meta-Llama-3-8B-Instruct |
| `llama31_8b` | Meta-Llama-3.1-8B-Instruct |
