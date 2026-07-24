$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$helperPath = Join-Path $repoRoot "scripts\normal_pipeline_watcher_lib.ps1"

. $helperPath

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw "ASSERT TRUE FAILED: $Message"
    }
}

function Assert-Equal {
    param(
        $Actual,
        $Expected,
        [string]$Message
    )

    if ($Actual -ne $Expected) {
        throw "ASSERT EQUAL FAILED: $Message`nExpected: $Expected`nActual: $Actual"
    }
}

$processes = @(
    [pscustomobject]@{ Path = "D:\Anaconda\envs\auto-label\python.exe"; Id = 1001 },
    [pscustomobject]@{ Path = "D:\Anaconda\envs\AR\python.exe"; Id = 2002 }
)

Assert-True (Test-AutoLabelProcessActive -Processes $processes -AutoLabelPythonPath "D:\Anaconda\envs\auto-label\python.exe") "???? auto-label ??????"
Assert-True (-not (Test-AutoLabelProcessActive -Processes $processes -AutoLabelPythonPath "D:\Anaconda\envs\missing\python.exe")) "???????????"

$plan = Get-NormalPipelineCommandPlan -RepoRoot "E:\ActionRecognition\behavionRecognition-src" -ArPythonPath "D:\Anaconda\envs\AR\python.exe"

Assert-Equal $plan.Train.Path "D:\Anaconda\envs\AR\python.exe" "????? AR ?????"
Assert-Equal $plan.Train.Arguments[0] "E:\ActionRecognition\behavionRecognition-src\train\train_finetune_normal_3.py" "?????????"
Assert-Equal $plan.Distill.Arguments[0] "E:\ActionRecognition\behavionRecognition-src\distill\distill_normal_3.py" "?????????"

$none = @(
    [pscustomobject]@{ Path = "D:\Anaconda\envs\AR\python.exe"; Id = 3003 }
)

$firstCheck = Test-DoubleCheckAutoLabelInactive -ProcessSnapshots @($processes, $none) -AutoLabelPythonPath "D:\Anaconda\envs\auto-label\python.exe"
$secondCheck = Test-DoubleCheckAutoLabelInactive -ProcessSnapshots @($none, $none) -AutoLabelPythonPath "D:\Anaconda\envs\auto-label\python.exe"

Assert-Equal $firstCheck $false "???????? auto-label ??????????"
Assert-Equal $secondCheck $true "???????? auto-label ??????????"

$trackedPids = Get-TrackedAutoLabelPids -Processes @(
    [pscustomobject]@{ Id = 27800; Path = "D:\Anaconda\envs\auto-label\python.exe"; CommandLine = '"D:\Anaconda\envs\auto-label\python.exe" script_a.py' },
    [pscustomobject]@{ Id = 27801; Path = "D:\Anaconda\envs\auto-label\python.exe"; CommandLine = '"D:\Anaconda\envs\auto-label\python.exe" script_b.py' },
    [pscustomobject]@{ Id = 32000; Path = "D:\Anaconda\envs\AR\python.exe"; CommandLine = '"D:\Anaconda\envs\AR\python.exe" train.py' }
) -AutoLabelPythonPath "D:\Anaconda\envs\auto-label\python.exe"

Assert-Equal ($trackedPids -join ",") "27800,27801" "?????? auto-label ?????? PID"
Assert-Equal (Test-TrackedPidsFinished -TrackedPids @(27800, 27801) -LivePids @(27801, 32000)) $false "???? PID ????????????"
Assert-Equal (Test-TrackedPidsFinished -TrackedPids @(27800, 27801) -LivePids @(32000)) $true "?????? PID ?????????"

$tempDir = Join-Path $env:TEMP ("watcher-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
$logPath = Join-Path $tempDir "combined.log"
$hostExe = (Get-Process -Id $PID).Path
$exitCode = Invoke-LoggedProcess -FilePath $hostExe -ArgumentList @("-NoProfile", "-Command", "Write-Output 'stdout-line'; [Console]::Error.WriteLine('stderr-line'); exit 3") -WorkingDirectory $tempDir -LogPath $logPath
$logText = [System.IO.File]::ReadAllText($logPath)

Assert-Equal $exitCode 3 "??????????"
Assert-True ($logText.Contains("stdout-line")) "????????????"
Assert-True ($logText.Contains("stderr-line")) "????????????"

Write-Host "All watcher tests passed."
