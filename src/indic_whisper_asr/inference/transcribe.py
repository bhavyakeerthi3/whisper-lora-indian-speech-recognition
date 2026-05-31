import argparse

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from indic_whisper_asr.utils.audio import load_audio_mono


def transcribe(base_model: str, adapter_path: str, audio_path: str) -> str:
    processor = WhisperProcessor.from_pretrained(adapter_path)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    audio, sample_rate = load_audio_mono(audio_path, target_sample_rate=16000)
    inputs = processor.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features.to(device))
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--adapter-path", default="outputs/whisper-small-indic-lora")
    parser.add_argument("--audio-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(transcribe(args.base_model, args.adapter_path, args.audio_path))

