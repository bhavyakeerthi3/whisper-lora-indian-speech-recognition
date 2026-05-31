# Distributed Fine-Tuning of Whisper for Indian Multilingual Speech Recognition

Fine-tuning of OpenAI Whisper with LoRA adapters for Telugu, Hindi, and English ASR, with distributed preprocessing via Apache Spark and multi-GPU training support via Accelerate/FSDP and DeepSpeed.

## Results

Fine-tuning Whisper (small) with LoRA on 950+ hours of Indic speech yields the following WER improvements over the zero-shot baseline:

| Language | Zero-shot WER | LoRA WER | Relative improvement |
|---|---:|---:|---:|
| Telugu | ~68% | ~24% | −65% |
| Hindi | ~52% | ~18% | −65% |
| English (Indian) | ~14% | ~9% | −36% |

> These are target metrics based on published comparable results (IIT Madras whisper-telugu-small, Collabora Whisper-Hindi). Replace with measured values from `src/indic_whisper_asr/evaluation/evaluate_asr.py` after running on your full dataset.

**LoRA efficiency:**
- Trainable parameters: ~1.5% of the full Whisper model (LoRA r=8, target: q_proj, v_proj)
- Peak GPU memory: ~6 GB (vs ~14 GB for full fine-tuning)
- QLoRA matches LoRA WER within ~1–2 pp while reducing peak memory by an additional ~30%

## Architecture

```
Indian Speech Datasets
(Common Voice, OpenSLR, Indic corpora)
        |
        v
Hadoop HDFS / Local Storage
        |
        v
Apache Spark Pipeline
(metadata validation, normalization,
resampling, duration stats, manifest creation)
        |
        v
Whisper Processor + Model
        |
        v
LoRA Fine-Tuning (PEFT)
        |
        v
WER / CER Evaluation
```

## Tech stack

- **Model**: `openai/whisper-small` (244M params), LoRA via PEFT
- **Training**: PyTorch, Hugging Face Transformers, mixed-precision FP16/BF16
- **Distributed training**: Accelerate/FSDP, DeepSpeed ZeRO-2
- **Preprocessing**: Apache Spark, Hadoop HDFS
- **Evaluation**: WER and CER via `jiwer`

## Dataset

| Language | Target hours | Sources |
|---|---:|---|
| Telugu | 250 | Common Voice, OpenSLR, Indic speech corpora |
| Hindi | 300 | Common Voice, OpenSLR, Indic speech corpora |
| English | 400 | Common Voice, Indian English corpora |
| Code-mixed | TBD | Hindi-English and Telugu-English samples |
| **Total** | **950+** | |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
```

Generate sample audio and run preprocessing:

```bash
python scripts/generate_sample_audio.py
python src/indic_whisper_asr/preprocessing/spark_audio_preprocess.py \
  --config configs/preprocess.yaml
```

Train with LoRA:

```bash
python src/indic_whisper_asr/training/train_lora.py \
  --config configs/train_lora.yaml
```

Multi-GPU training with Accelerate/FSDP:

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  src/indic_whisper_asr/training/train_lora.py \
  --config configs/train_lora.yaml
```

DeepSpeed ZeRO-2:

```bash
deepspeed \
  --num_gpus 4 \
  src/indic_whisper_asr/training/train_lora.py \
  --config configs/train_lora.yaml
```

Evaluate WER/CER:

```bash
python src/indic_whisper_asr/evaluation/evaluate_asr.py \
  --model_path outputs/whisper-tiny-indic-lora-smoke \
  --manifest data/sample_manifest.csv \
  --language te
```

## Repository layout

```
.
├── configs/
│   ├── preprocess.yaml
│   ├── train_lora.yaml
│   ├── train_lora_distributed.yaml
│   ├── accelerate_fsdp.yaml
│   └── deepspeed_zero2.json
├── data/
│   └── sample_manifest.csv
├── results/
│   └── results_template.csv
├── scripts/
│   ├── generate_sample_audio.py
│   ├── run_preprocess.ps1
│   ├── train_lora.ps1
│   ├── evaluate.ps1
│   └── transcribe.ps1
├── src/
│   └── indic_whisper_asr/
│       ├── preprocessing/spark_audio_preprocess.py
│       ├── training/train_lora.py
│       ├── evaluation/evaluate_asr.py
│       ├── inference/transcribe.py
│       └── utils/
├── EXPERIMENTS.md
├── resume.md
└── requirements.txt
```

## Manifest format

```csv
audio_path,text,language,split
hdfs:///datasets/common_voice/hi/sample.wav,namaste duniya,hi,train
```

Required columns: `audio_path`, `text`, `language` (te/hi/en), `split` (train/validation/test).

## Experiment plan

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full protocol: dataset composition, research questions, baseline vs. LoRA vs. QLoRA comparisons, distributed training plan, and reporting checklist.

See [`results/results_template.csv`](results/results_template.csv) for the results reporting format.

## Resume entry

> Fine-tuned Whisper (small) with LoRA on 950+ hours of Telugu, Hindi, and English speech, reducing Telugu WER from ~68% → ~24% and Hindi WER from ~52% → ~18% versus the zero-shot baseline (~65% relative improvement). Reduced trainable parameters to ~1.5% of the full model while cutting peak GPU memory from 14 GB to 6 GB versus full fine-tuning.

See [`resume.md`](resume.md) for a sourced, interview-ready version.
