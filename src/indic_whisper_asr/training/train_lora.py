import argparse
import inspect
from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from indic_whisper_asr.utils.config import load_yaml
from indic_whisper_asr.utils.audio import load_audio_mono


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": item["input_features"]} for item in features]
        label_features = [{"input_ids": item["labels"]} for item in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_manifest_dataset(path: str):
    data_files = {
        "train": f"{path}/part-*",
        "validation": f"{path}/part-*",
        "test": f"{path}/part-*",
    }
    dataset = load_dataset("parquet", data_files=data_files)
    return dataset.filter(lambda row: row["split"] in {"train", "validation", "test"})


def train(config_path: str) -> None:
    config = load_yaml(config_path)
    model_name = config.get("model_name_or_path", "openai/whisper-small")

    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=None if config.get("language") == "multilingual" else config.get("language"),
        task=config.get("task", "transcribe"),
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    lora_config = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_config.get("r", 16)),
            lora_alpha=int(lora_config.get("alpha", 32)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("parquet", data_files=f"{config['processed_manifest']}/*.parquet")["train"]

    train_split = config.get("train_split", "train")
    eval_split = config.get("eval_split", "validation")
    train_data = dataset.filter(lambda row: row["split"] == train_split)
    eval_data = dataset.filter(lambda row: row["split"] == eval_split)

    if config.get("max_train_samples"):
        train_data = train_data.select(range(min(int(config["max_train_samples"]), len(train_data))))
    if config.get("max_eval_samples"):
        eval_data = eval_data.select(range(min(int(config["max_eval_samples"]), len(eval_data))))

    def prepare_batch(batch):
        audio_array, sample_rate = load_audio_mono(batch["audio_path"], target_sample_rate=16000)
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sample_rate
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_data = train_data.map(prepare_batch, remove_columns=train_data.column_names)
    eval_data = eval_data.map(prepare_batch, remove_columns=eval_data.column_names)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

    training_kwargs = {
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters
        else "evaluation_strategy": "steps"
    }
    if config.get("deepspeed"):
        training_kwargs["deepspeed"] = config["deepspeed"]

    args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        warmup_steps=int(config.get("warmup_steps", 100)),
        num_train_epochs=float(config.get("num_train_epochs", 3)),
        max_steps=int(config["max_steps"]) if config.get("max_steps") is not None else -1,
        fp16=bool(config.get("fp16", True)),
        bf16=bool(config.get("bf16", False)),
        **training_kwargs,
        eval_steps=int(config.get("eval_steps", 250)),
        save_steps=int(config.get("save_steps", 250)),
        logging_steps=int(config.get("logging_steps", 25)),
        predict_with_generate=True,
        generation_max_length=int(config.get("generation_max_length", 225)),
        remove_unused_columns=False,
        report_to=[],
    )

    trainer_kwargs = {
        "processing_class"
        if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters
        else "tokenizer": processor.feature_extractor
    }
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_lora.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args().config)
