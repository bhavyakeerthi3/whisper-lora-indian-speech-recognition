# Results

This directory holds experiment outputs from WER/CER evaluation runs.

## Target metrics (benchmark-based estimates)

| Model | Method | Telugu WER | Hindi WER | English WER | Overall WER | Relative improvement |
|---|---|---:|---:|---:|---:|---:|
| whisper-small | Zero-shot baseline | ~68% | ~52% | ~14% | ~45% | — |
| whisper-small | LoRA (r=8) | ~24% | ~18% | ~9% | ~17% | ~65% |
| whisper-small | QLoRA (4-bit) | ~26% | ~20% | ~10% | ~19% | ~60% |

> These are target metrics based on published comparable results. Replace with values produced by `src/indic_whisper_asr/evaluation/evaluate_asr.py` on your full dataset before reporting as final.

## Files to add after running experiments

- `baseline_whisper_metrics.json` — zero-shot evaluation output
- `lora_metrics.json` — LoRA fine-tuned evaluation output
- `qlora_metrics.json` — QLoRA evaluation output
- `language_breakdown.csv` — per-language WER/CER breakdown
- `training_resource_usage.csv` — GPU memory, training time, trainable params

## How to generate

```bash
# Baseline (zero-shot)
python src/indic_whisper_asr/evaluation/evaluate_asr.py \
  --model_path openai/whisper-small \
  --manifest data/sample_manifest.csv \
  --language te \
  --output results/baseline_whisper_metrics.json

# LoRA fine-tuned
python src/indic_whisper_asr/evaluation/evaluate_asr.py \
  --model_path outputs/whisper-tiny-indic-lora-smoke \
  --manifest data/sample_manifest.csv \
  --language te \
  --output results/lora_metrics.json
```
