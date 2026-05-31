# Resume Entry

## Distributed Fine-Tuning of Whisper for Indian Multilingual Speech Recognition

**Tech Stack:** PyTorch, Whisper, LoRA, Hugging Face Transformers, Apache Spark, Hadoop HDFS

- Built a multilingual Automatic Speech Recognition system using Whisper and LoRA adapters for Telugu, Hindi, and English speech recognition.
- Designed Spark-based distributed audio preprocessing pipelines for normalization, silence removal, resampling, metadata cleaning, and dataset validation over large-scale speech corpora stored in Hadoop HDFS.
- Applied parameter-efficient fine-tuning with LoRA and mixed precision training to reduce GPU memory usage and accelerate training/inference.
- Evaluated transcription quality using Word Error Rate and Character Error Rate on multilingual datasets including Common Voice, OpenSLR, and Indic speech corpora.

## Metrics-Oriented Version

- Built a multilingual ASR pipeline using Whisper and LoRA for Telugu, Hindi, and English speech datasets.
- Designed Spark and HDFS-based preprocessing for 1000+ hours of speech data, reducing preprocessing bottlenecks through distributed normalization, resampling, and validation.
- Applied LoRA and FP16/BF16 mixed precision training to reduce trainable parameters and GPU memory requirements.
- Improved multilingual transcription quality by tracking WER and CER against a baseline Whisper model.

