import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from indic_whisper_asr.utils.config import load_yaml


def build_spark(app_name: str = "indic-whisper-audio-preprocess") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def add_duration_column(frame, target_sample_rate: int):
    def estimate_duration(path: str) -> float:
        try:
            import soundfile as sf

            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except Exception:
            # HDFS paths are commonly validated in cluster jobs where codecs are installed.
            # Keep failed probes as -1 so downstream filters remove them.
            return -1.0

    duration_udf = F.udf(estimate_duration, DoubleType())
    return frame.withColumn("duration_seconds", duration_udf(F.col("audio_path")))


def preprocess(config_path: str) -> None:
    config = load_yaml(config_path)
    spark = build_spark()

    input_manifest = config["input_manifest"]
    output_manifest = config["output_manifest"]
    allowed_languages = config.get("allowed_languages", ["te", "hi", "en"])
    min_duration = float(config.get("min_duration_seconds", 0.5))
    max_duration = float(config.get("max_duration_seconds", 30.0))
    text_min_chars = int(config.get("text_min_chars", 1))
    text_max_chars = int(config.get("text_max_chars", 300))
    target_sample_rate = int(config.get("target_sample_rate", 16000))

    frame = spark.read.option("header", True).csv(input_manifest)

    required = {"audio_path", "text", "language", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    cleaned = (
        frame.dropna(subset=["audio_path", "text", "language", "split"])
        .withColumn("text", F.trim(F.lower(F.col("text"))))
        .withColumn("language", F.lower(F.col("language")))
        .withColumn("split", F.lower(F.col("split")))
        .filter(F.col("language").isin(allowed_languages))
        .filter(F.col("split").isin(["train", "validation", "test"]))
        .filter(F.length("text").between(text_min_chars, text_max_chars))
        .dropDuplicates(["audio_path", "text", "language", "split"])
    )

    validated = add_duration_column(cleaned, target_sample_rate).filter(
        F.col("duration_seconds").between(min_duration, max_duration)
    )

    stats = (
        validated.groupBy("language", "split")
        .agg(
            F.count("*").alias("num_samples"),
            F.round(F.sum("duration_seconds") / 3600.0, 3).alias("hours"),
            F.round(F.avg("duration_seconds"), 3).alias("avg_duration_seconds"),
        )
        .orderBy("language", "split")
    )

    stats.show(truncate=False)
    if os.name == "nt" and not output_manifest.startswith(("hdfs://", "s3://", "s3a://")):
        output_dir = Path(output_manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        local_frame = validated.toPandas()
        local_frame.to_parquet(output_dir / "part-00000.parquet", index=False)
        local_frame.to_csv(output_dir / "processed_manifest.csv", index=False)
    else:
        validated.write.mode("overwrite").parquet(output_manifest)
    spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preprocess.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.config)
