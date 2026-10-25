param(
    [string]$Python = ".venv-vlm\Scripts\python.exe",
    [string]$Device = "cuda",
    [string]$OutputDir = "experiments\rabbit_streaming_memory\outputs\checkpoints",
    [string]$SelectionManifest = "experiments\rabbit_streaming_memory\manifests\training_selection_proposed.json",
    [string]$AuditManifest = "experiments\rabbit_streaming_memory\manifests\audit_full_2026-10-24.json"
)
$ErrorActionPreference = "Stop"
$modes = @("t0", "t1", "t2")
$seeds = @(42, 43, 44)
foreach ($mode in $modes) {
    foreach ($seed in $seeds) {
        $logDir = Join-Path $OutputDir "logs"
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $log = Join-Path $logDir "${mode}_seed${seed}.log"
        & $Python "experiments/rabbit_streaming_memory/src/stream_trainer.py" `
            --mode $mode --seed $seed --device $Device `
            --selection-manifest $SelectionManifest --audit-manifest $AuditManifest `
            --output-dir $OutputDir 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $mode seed=$seed (exit=$LASTEXITCODE); see $log"
        }
    }
}
& $Python "experiments/rabbit_streaming_memory/src/summarize_training_matrix.py" --checkpoint-root $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Training summary failed (exit=$LASTEXITCODE)" }
