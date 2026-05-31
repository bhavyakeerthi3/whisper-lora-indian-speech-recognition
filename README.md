# Distributed Fine-Tuning of Whisper for Indian Multilingual Speech Recognition

Production-style portfolio project for multilingual Automatic Speech Recognition (ASR) on Telugu, Hindi, and English speech using:

- Whisper from Hugging Face Transformers
- LoRA adapters via PEFT
- PyTorch mixed precision training
- Apache Spark for distributed audio preprocessing
- Hadoop HDFS-compatible dataset paths
- WER and CER evaluation

## Architecture

```text
Indian Speech Datasets
(Common Voice, OpenSLR, Indic datasets)
        |
        v
Hadoop HDFS / Local Storage
        |
        v
Apache Spark Pipeline
(metadata validation, audio normalization,
resampling, duration stats, manifest creation)
        |
        v
Whisper Processor + Model
        |
        v
LoRA Fine-Tuning
        |
        v
Optimized ASR System
        |
        v
WER / CER Evaluation
```

## What This Project Demonstrates

- Distributed preprocessing of large speech corpora with Spark.
- HDFS-ready dataset ingestion for thousands of hours of audio.
- Parameter-efficient fine-tuning of Whisper with LoRA.
- Mixed-precision training using FP16 or BF16.
- Multilingual evaluation with WER and CER.
- Experiment design for comparing baseline Whisper, full fine-tuning, LoRA, and QLoRA.
- Distributed training launch templates for FSDP and DeepSpeed.
- Clean separation between data engineering, model training, inference, and evaluation.

## Experimental Validation

The repository includes the code needed to run baseline and fine-tuned evaluations, but final WER/CER values should only be reported after running on real speech datasets. See `EXPERIMENTS.md` and `results/results_template.csv` for the experiment plan and reporting format.

Minimum results to publish:

| Metric | Description |
| --- | --- |
| Baseline WER/CER | Zero-shot Whisper evaluation before fine-tuning. |
| LoRA WER/CER | Whisper with LoRA adapters after fine-tuning. |
| Relative WER improvement | `(baseline_wer - lora_wer) / baseline_wer * 100`. |
| Dataset hours | Computed from processed manifests by language and split. |
| Training resources | GPU type, training time, peak memory, trainable parameter percentage. |

## Target Dataset Scale

The full-scale experiment is designed for roughly 950+ hours of Indian multilingual speech:

| Language | Target Hours | Sources |
| --- | ---: | --- |
| Telugu | 250 | Common Voice, OpenSLR, Indic speech corpora |
| Hindi | 300 | Common Voice, OpenSLR, Indic speech corpora |
| English | 400 | Common Voice, Indian English corpora |
| Code-mixed | TBD | Hindi-English and Telugu-English samples |

These are target experiment sizes. The final README should be updated with measured hours from the Spark preprocessing output after the full dataset is prepared.

## Repository Layout

```text
.
|-- configs/
|   |-- preprocess.yaml
|   `-- train_lora.yaml
|-- data/
|   `-- sample_manifest.csv
|-- scripts/
|   |-- generate_sample_audio.py
|   |-- run_preprocess.ps1
|   |-- train_lora.ps1
|   |-- evaluate.ps1
|   `-- transcribe.ps1
|-- src/
|   `-- indic_whisper_asr/
|       |-- preprocessing/
|       |   `-- spark_audio_preprocess.py
|       |-- training/
|       |   `-- train_lora.py
|       |-- evaluation/
|       |   `-- evaluate_asr.py
|       |-- inference/
|       |   `-- transcribe.py
|       `-- utils/
|           |-- audio.py
|           |-- config.py
|           `-- manifest.py
|-- requirements.txt
`-- resume.md
```

## Quick Start

Create an environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Run Spark preprocessing on a manifest:

```powershell
python scripts/generate_sample_audio.py
powershell -ExecutionPolicy Bypass -File scripts/run_preprocess.ps1
```

Train Whisper with LoRA:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_lora.ps1
```

Evaluate WER and CER:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/evaluate.ps1
```

Transcribe one audio file:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/transcribe.ps1 -AudioPath path\\to\\audio.wav
```

## Dataset Manifest Format

The preprocessing pipeline expects a CSV with:

```csv
audio_path,text,language,split
hdfs:///datasets/common_voice/hi/sample.wav,namaste duniya,hi,train
```

Required columns:

- `audio_path`: local path or HDFS URI.
- `text`: normalized transcript.
- `language`: language code such as `te`, `hi`, or `en`.
- `split`: `train`, `validation`, or `test`.

## HDFS Usage

Set paths in `configs/preprocess.yaml`:

```yaml
input_manifest: hdfs:///datasets/indic_asr/manifest.csv
output_manifest: hdfs:///datasets/indic_asr/processed_manifest
```

Run with `spark-submit` on your Hadoop/Spark cluster:

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  src/indic_whisper_asr/preprocessing/spark_audio_preprocess.py \
  --config configs/preprocess.yaml
```

## Distributed Training

Single-node or multi-GPU training can be launched with Hugging Face Accelerate/FSDP:

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  src/indic_whisper_asr/training/train_lora.py \
  --config configs/train_lora.yaml
```

DeepSpeed ZeRO-2 configuration is provided in `configs/deepspeed_zero2.json` for larger Whisper variants.

## Notes

This repository is designed to be runnable on a small local sample and scalable to a real distributed environment. The included sample manifest is intentionally tiny; replace it with Common Voice, OpenSLR, or Indic speech manifests for full training.

For a quick preprocessing smoke test, generate the local WAV samples first:

```powershell
python scripts/generate_sample_audio.py
```


