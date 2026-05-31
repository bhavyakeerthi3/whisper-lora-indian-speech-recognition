import argparse
import json
from pathlib import Path

import evaluate
import torch
from datasets import Audio, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def evaluate_model(base_model: str, adapter_path: str, manifest_path: str, split: str, output_path: str) -> None:
    processor = WhisperProcessor.from_pretrained(adapter_path)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = load_dataset("parquet", data_files=f"{manifest_path}/*.parquet")["train"]
    dataset = dataset.filter(lambda row: row["split"] == split)
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

    predictions = []
    references = []

    for row in tqdm(dataset, desc=f"Evaluating {split}"):
        audio = row["audio_path"]
        inputs = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            generated_ids = model.generate(inputs)
        predictions.append(processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        references.append(row["text"])

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    metrics = {
        "split": split,
        "num_samples": len(references),
        "wer": wer_metric.compute(predictions=predictions, references=references),
        "cer": cer_metric.compute(predictions=predictions, references=references),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--adapter-path", default="outputs/whisper-small-indic-lora")
    parser.add_argument("--manifest-path", default="outputs/processed_manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-path", default="outputs/evaluation_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        manifest_path=args.manifest_path,
        split=args.split,
        output_path=args.output_path,
    )

