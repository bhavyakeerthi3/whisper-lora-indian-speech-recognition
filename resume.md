# Resume Entry

## Distributed Fine-Tuning of Whisper for Indian Multilingual Speech Recognition

**Tech Stack:** PyTorch, Whisper, LoRA, Hugging Face Transformers, Apache Spark, Hadoop HDFS, Accelerate/FSDP, DeepSpeed

- Built a multilingual Automatic Speech Recognition system using Whisper and LoRA adapters for Telugu, Hindi, and English speech recognition.
- Designed Spark-based distributed audio preprocessing pipelines for normalization, silence removal, resampling, metadata cleaning, and dataset validation over large-scale speech corpora stored in Hadoop HDFS.
- Applied parameter-efficient fine-tuning with LoRA and mixed precision training to reduce GPU memory usage and accelerate training/inference.
- Evaluated transcription quality using Word Error Rate and Character Error Rate on multilingual datasets including Common Voice, OpenSLR, and Indic speech corpora.
- Added experiment plans for comparing baseline Whisper, full fine-tuning, LoRA, QLoRA, and code-mixed speech recognition.

## Metrics-Oriented Version

- Built a multilingual ASR pipeline using Whisper and LoRA for Telugu, Hindi, and English speech datasets.
- Designed Spark and HDFS-based preprocessing for large-scale speech data, including distributed normalization, resampling, validation, and per-language duration statistics.
- Implemented mixed-precision LoRA fine-tuning and prepared distributed training configurations using Accelerate/FSDP and DeepSpeed.
- Compared parameter-efficient fine-tuning approaches for low-resource Indian language speech recognition using WER and CER evaluation.

## Version With Verified Metrics

Use this version only after running full experiments and replacing placeholders:

- Built a multilingual ASR system using Whisper and LoRA across `TBD` hours of Telugu, Hindi, English, and code-mixed speech.
- Reduced WER from `TBD` to `TBD` compared with the baseline Whisper model, achieving `TBD%` relative improvement.
- Reduced trainable parameters to `TBD%` of the full Whisper model while maintaining competitive CER/WER.
- Scaled preprocessing with Spark and HDFS across `TBD` audio files and `TBD` speakers.
