param(
  [string]$Config = "configs/preprocess.yaml"
)

$env:PYTHONPATH = "src"
$env:PYSPARK_PYTHON = (Get-Command python).Source
$env:PYSPARK_DRIVER_PYTHON = (Get-Command python).Source
python -m indic_whisper_asr.preprocessing.spark_audio_preprocess --config $Config
