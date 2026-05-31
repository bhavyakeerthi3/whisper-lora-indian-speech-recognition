from pathlib import Path

import librosa
import soundfile as sf


def load_audio_mono(path: str, target_sample_rate: int = 16000) -> tuple[list[float], int]:
    audio, sample_rate = librosa.load(path, sr=target_sample_rate, mono=True)
    return audio.tolist(), sample_rate


def duration_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def normalize_audio_file(input_path: str, output_path: str, target_sample_rate: int = 16000) -> None:
    audio, sample_rate = librosa.load(input_path, sr=target_sample_rate, mono=True)
    peak = max(abs(audio).max(), 1e-8)
    audio = audio / peak * 0.95
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)

