param(
  [string]$Config = "configs/train_lora.yaml"
)

$env:PYTHONPATH = "src"
python -m indic_whisper_asr.training.train_lora --config $Config

