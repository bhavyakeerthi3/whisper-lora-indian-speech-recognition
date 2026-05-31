import csv
import json
import wave
from pathlib import Path

import pandas as pd
import yaml


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def main() -> None:
    config = yaml.safe_load(Path("configs/preprocess.yaml").read_text(encoding="utf-8"))
    manifest_path = Path(config["input_manifest"])
    output_dir = Path("outputs/processed_manifest_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            audio_path = Path(row["audio_path"])
            if not audio_path.exists():
                continue
            duration = wav_duration_seconds(audio_path)
            if not (config["min_duration_seconds"] <= duration <= config["max_duration_seconds"]):
                continue
            if row["language"] not in set(config["allowed_languages"]):
                continue
            row["duration_seconds"] = round(duration, 3)
            rows.append(row)

    frame = pd.DataFrame(rows)
    parquet_path = output_dir / "part-00000.parquet"
    csv_path = output_dir / "processed_manifest.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)

    stats = (
        frame.groupby(["language", "split"], as_index=False)
        .agg(num_samples=("audio_path", "count"), hours=("duration_seconds", lambda x: round(x.sum() / 3600, 6)))
        .sort_values(["language", "split"])
    )

    summary = {
        "input_manifest": str(manifest_path),
        "valid_samples": int(len(frame)),
        "output_parquet": str(parquet_path),
        "output_csv": str(csv_path),
        "stats": stats.to_dict(orient="records"),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

