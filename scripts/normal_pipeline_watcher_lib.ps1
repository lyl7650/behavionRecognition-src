function Test-AutoLabelProcessActive {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Processes,
        [Parameter(Mandatory = $true)]
        [string]$AutoLabelPythonPath
    )

    $normalizedTarget = [System.IO.Path]::GetFullPath($AutoLabelPythonPath).ToLowerInvariant()

    foreach ($process in $Processes) {
        if (-not $process.PSObject.Properties.Name.Contains("Path")) {
            continue
        }

        $path = $process.Path
        if ([string]::IsNullOrWhiteSpace($path)) {
            continue
        }

        try {
            $normalizedPath = [System.IO.Path]::GetFullPath($path).ToLowerInvariant()
        }
        catch {
            continue
        }

        if ($normalizedPath -eq $normalizedTarget) {
            return $true
        }
    }

    return $false
}

function Get-TrackedAutoLabelPids {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Processes,
        [Parameter(Mandatory = $true)]
        [string]$AutoLabelPythonPath
    )

    $normalizedTarget = [System.IO.Path]::GetFullPath($AutoLabelPythonPath).ToLowerInvariant()
    $tracked = New-Object System.Collections.Generic.List[int]

    foreach ($process in $Processes) {
        $candidatePaths = New-Object System.Collections.Generic.List[string]

        if ($process.PSObject.Properties.Name.Contains("Path") -and -not [string]::IsNullOrWhiteSpace($process.Path)) {
            $candidatePaths.Add($process.Path)
        }

        if ($process.PSObject.Properties.Name.Contains("CommandLine") -and -not [string]::IsNullOrWhiteSpace($process.CommandLine)) {
            $commandLine = $process.CommandLine.Trim()
            if ($commandLine.StartsWith('"')) {
                $endQuote = $commandLine.IndexOf('"', 1)
                if ($endQuote -gt 1) {
                    $candidatePaths.Add($commandLine.Substring(1, $endQuote - 1))
                }
            }
            else {
                $firstToken = ($commandLine -split "\s+", 2)[0]
                if (-not [string]::IsNullOrWhiteSpace($firstToken)) {
                    $candidatePaths.Add($firstToken)
                }
            }
        }

        foreach ($candidatePath in $candidatePaths) {
            try {
                $normalizedCandidate = [System.IO.Path]::GetFullPath($candidatePath).ToLowerInvariant()
            }
            catch {
                continue
            }

            if ($normalizedCandidate -eq $normalizedTarget) {
                $tracked.Add([int]$process.Id)
                break
            }
        }
    }

    return @($tracked.ToArray() | Sort-Object -Unique)
}

function Test-DoubleCheckAutoLabelInactive {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$ProcessSnapshots,
        [Parameter(Mandatory = $true)]
        [string]$AutoLabelPythonPath
    )

    foreach ($snapshot in $ProcessSnapshots) {
        if (Test-AutoLabelProcessActive -Processes $snapshot -AutoLabelPythonPath $AutoLabelPythonPath) {
            return $false
        }
    }

    return $true
}

function Test-TrackedPidsFinished {
    param(
        [Parameter(Mandatory = $true)]
        [int[]]$TrackedPids,
        [Parameter(Mandatory = $true)]
        [int[]]$LivePids
    )

    if ($TrackedPids.Count -eq 0) {
        return $true
    }

    foreach ($trackedPid in $TrackedPids) {
        if ($LivePids -contains $trackedPid) {
            return $false
        }
    }

    return $true
}

function Get-NormalPipelineCommandPlan {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$ArPythonPath
    )

    $repoRootPath = [System.IO.Path]::GetFullPath($RepoRoot)
    $pythonPath = [System.IO.Path]::GetFullPath($ArPythonPath)

    return [pscustomobject]@{
        Train = [pscustomobject]@{
            Path = $pythonPath
            Arguments = @((Join-Path $repoRootPath "train\train_finetune_normal_3.py"))
        }
        Distill = [pscustomobject]@{
            Path = $pythonPath
            Arguments = @((Join-Path $repoRootPath "distill\distill_normal_3.py"))
        }
    }
}

function Get-PythonProcessesWithPath {
    $processes = @()

    foreach ($process in (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue)) {
        $processes += [pscustomobject]@{
            Id = [int]$process.ProcessId
            Path = $process.ExecutablePath
            CommandLine = $process.CommandLine
        }
    }

    return $processes
}

function Invoke-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $parent = Split-Path -Parent $LogPath
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    if (Test-Path -LiteralPath $LogPath) {
        Remove-Item -LiteralPath $LogPath -Force
    }

    $stdoutPath = "$LogPath.stdout.tmp"
    $stderrPath = "$LogPath.stderr.tmp"

    foreach ($tempPath in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force
        }
    }

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -NoNewWindow `
        -PassThru `
        -Wait

    foreach ($tempPath in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $tempPath) {
            Get-Content -LiteralPath $tempPath | Add-Content -LiteralPath $LogPath
            Remove-Item -LiteralPath $tempPath -Force
        }
    }

    return $process.ExitCode
}
