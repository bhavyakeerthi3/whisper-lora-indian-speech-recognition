import math
import wave
from pathlib import Path


def write_tone(path: Path, frequency: float, duration_seconds: float = 1.2, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    amplitude = 0.25
    total_samples = int(duration_seconds * sample_rate)

    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(total_samples):
            sample = math.sin(2 * math.pi * frequency * index / sample_rate)
            value = int(amplitude * 32767 * sample)
            handle.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def main() -> None:
    samples = {
        "data/audio/te_sample.wav": 220.0,
        "data/audio/hi_sample.wav": 330.0,
        "data/audio/en_sample.wav": 440.0,
    }
    for file_name, frequency in samples.items():
        write_tone(Path(file_name), frequency)
        print(f"wrote {file_name}")


if __name__ == "__main__":
    main()

