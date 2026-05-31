param(
  [Parameter(Mandatory=$true)][string]$AudioPath,
  [string]$BaseModel = "openai/whisper-small",
  [string]$AdapterPath = "outputs/whisper-small-indic-lora"
)

$env:PYTHONPATH = "src"
python -m indic_whisper_asr.inference.transcribe --base-model $BaseModel --adapter-path $AdapterPath --audio-path $AudioPath

