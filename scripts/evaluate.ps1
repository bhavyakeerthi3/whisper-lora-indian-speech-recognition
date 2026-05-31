param(
  [string]$BaseModel = "openai/whisper-small",
  [string]$AdapterPath = "outputs/whisper-small-indic-lora",
  [string]$ManifestPath = "outputs/processed_manifest",
  [string]$Split = "test",
  [string]$OutputPath = "outputs/evaluation_metrics.json"
)

$env:PYTHONPATH = "src"
python -m indic_whisper_asr.evaluation.evaluate_asr --base-model $BaseModel --adapter-path $AdapterPath --manifest-path $ManifestPath --split $Split --output-path $OutputPath

