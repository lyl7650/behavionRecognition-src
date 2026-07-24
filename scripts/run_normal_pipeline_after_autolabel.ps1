param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$AutoLabelPythonPath = "D:\Anaconda\envs\auto-label\python.exe",
    [string]$ArPythonPath = "D:\Anaconda\envs\AR\python.exe",
    [ValidateRange(1, 86400)]
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "normal_pipeline_watcher_lib.ps1")

function Write-WatcherLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $script:WatcherLogPath -Value $line
}

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description not found: $Path"
    }
}

function Start-LoggedPythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StepName,
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    Write-WatcherLog "Starting ${StepName}: $ScriptPath"

    $exitCode = Invoke-LoggedProcess `
        -FilePath $PythonPath `
        -ArgumentList @($ScriptPath) `
        -WorkingDirectory $RepoRoot `
        -LogPath $LogPath

    Write-WatcherLog "${StepName} finished with exit code: $exitCode"

    if ($exitCode -ne 0) {
        throw "${StepName} failed, see log: $LogPath"
    }
}

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $RepoRoot "logs\overnight\$timestamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$script:WatcherLogPath = Join-Path $logDir "watcher.log"
$trainLogPath = Join-Path $logDir "train.log"
$distillLogPath = Join-Path $logDir "distill.log"

Assert-PathExists -Path $RepoRoot -Description "Repo root"
Assert-PathExists -Path $AutoLabelPythonPath -Description "auto-label Python"
Assert-PathExists -Path $ArPythonPath -Description "AR Python"

$plan = Get-NormalPipelineCommandPlan -RepoRoot $RepoRoot -ArPythonPath $ArPythonPath
Assert-PathExists -Path $plan.Train.Arguments[0] -Description "Train script"
Assert-PathExists -Path $plan.Distill.Arguments[0] -Description "Distill script"

Write-WatcherLog "Watcher started. Repo: $RepoRoot"
Write-WatcherLog "Watching interpreter: $AutoLabelPythonPath"
Write-WatcherLog "Training interpreter: $ArPythonPath"
Write-WatcherLog "Log directory: $logDir"

try {
    $initialSnapshot = Get-PythonProcessesWithPath
    $trackedPids = @(Get-TrackedAutoLabelPids -Processes $initialSnapshot -AutoLabelPythonPath $AutoLabelPythonPath)

    if ($trackedPids.Count -gt 0) {
        Write-WatcherLog "Tracking current auto-label PID(s): $($trackedPids -join ', ')"
    }
    else {
        Write-WatcherLog "No active auto-label PID found at startup. Training will start immediately."
    }

    while (-not (Test-TrackedPidsFinished -TrackedPids $trackedPids -LivePids @((Get-PythonProcessesWithPath).Id))) {
        Write-WatcherLog "Tracked auto-label PID(s) still running: $($trackedPids -join ', '). Retry in $PollSeconds seconds."
        Start-Sleep -Seconds $PollSeconds
    }

    if ($trackedPids.Count -gt 0) {
        Write-WatcherLog "Tracked auto-label PID(s) have exited. Starting train then distill."
    }
    else {
        Write-WatcherLog "Starting train then distill."
    }

    Start-LoggedPythonStep -StepName "train" -PythonPath $plan.Train.Path -ScriptPath $plan.Train.Arguments[0] -LogPath $trainLogPath
    Start-LoggedPythonStep -StepName "distill" -PythonPath $plan.Distill.Path -ScriptPath $plan.Distill.Arguments[0] -LogPath $distillLogPath

    Write-WatcherLog "Pipeline finished."
}
catch {
    Write-WatcherLog "Watcher failed: $($_.Exception.Message)"
    throw
}
