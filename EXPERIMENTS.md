# Experimental Protocol

This file defines the experiments needed to turn the engineering pipeline into a validated applied ML project. Do not report the placeholder values below as final results until the experiments have been run on real speech corpora.

## Dataset Scale

Target dataset composition:

| Language | Dataset Sources | Target Hours | Notes |
| --- | --- | ---: | --- |
| Telugu | Common Voice, OpenSLR, Indic speech corpora | 250 | Include clean and noisy speech where available. |
| Hindi | Common Voice, OpenSLR, Indic speech corpora | 300 | Include multiple speakers and regional accents. |
| English | Common Voice, Indian English corpora | 400 | Prefer Indian-accented English samples. |
| Code-mixed | Hindi-English, Telugu-English | TBD | Track separately from monolingual splits. |
| Total | Mixed public corpora | 950+ | Final hours must be computed from processed manifests. |

The Spark preprocessing job writes per-language duration statistics. Use those numbers as the source of truth before publishing final dataset scale.

## Research Questions

1. How much does LoRA improve Whisper on Indian multilingual speech compared with the base model?
2. Does LoRA preserve accuracy while reducing trainable parameters and memory usage compared with full fine-tuning?
3. How does performance vary across Telugu, Hindi, English, and code-mixed speech?
4. Does QLoRA provide a better memory/accuracy tradeoff than standard LoRA?

## Core Experiments

| Experiment | Model | Training Method | Precision | Metrics |
| --- | --- | --- | --- | --- |
| Baseline | Whisper tiny/base/small | No fine-tuning | FP32 | WER, CER |
| Full fine-tuning | Whisper tiny/base/small | All parameters | FP16/BF16 | WER, CER, GPU memory |
| LoRA | Whisper tiny/base/small | Adapter fine-tuning | FP16/BF16 | WER, CER, trainable params |
| QLoRA | Whisper tiny/base/small | Quantized adapter fine-tuning | 4-bit + BF16 | WER, CER, GPU memory |
| Code-mixed LoRA | Whisper tiny/base/small | Adapter fine-tuning | FP16/BF16 | WER, CER by language pair |

## Results Template

Fill this table after running experiments on real datasets.

| Model | Method | Dataset Hours | Telugu WER | Hindi WER | English WER | Overall WER | Overall CER | Relative WER Improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper base | Zero-shot baseline | TBD | TBD | TBD | TBD | TBD | TBD | - |
| Whisper + full fine-tuning | Full fine-tune | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Whisper + LoRA | PEFT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Whisper + QLoRA | Quantized PEFT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Relative WER improvement:

```text
relative_improvement = (baseline_wer - fine_tuned_wer) / baseline_wer * 100
```

## Distributed Training Plan

Current implementation includes distributed preprocessing with Spark. For distributed training, use one of:

- PyTorch DistributedDataParallel for multi-GPU single-node training.
- Hugging Face Accelerate with FSDP for sharded multi-GPU training.
- DeepSpeed ZeRO for memory-efficient training on larger Whisper variants.

Example launch:

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  src/indic_whisper_asr/training/train_lora.py \
  --config configs/train_lora.yaml
```

DeepSpeed launch:

```bash
deepspeed \
  --num_gpus 4 \
  src/indic_whisper_asr/training/train_lora.py \
  --config configs/train_lora.yaml
```

## Reporting Checklist

- Dataset hours by language and split.
- Speaker count where available.
- Baseline Whisper WER/CER.
- Fine-tuned WER/CER.
- Relative improvement over baseline.
- Trainable parameter percentage.
- GPU memory usage and training time.
- Inference latency on representative audio.

