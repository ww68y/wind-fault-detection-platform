$ErrorActionPreference = "Stop"

$bundleRoot = "E:\Desktop\training_bundle_v05_balanced"
$currentRunName = "v5_1_no_leak_1024_b24_w4_nocache_from_v04_corrosion_x4_crack_x2_50e"
$expectedEpochs = 50
$pollSeconds = 300
$maxWaitHours = 8

Set-Location $bundleRoot

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$timestamp] $Message"
}

function Test-TrainingPython {
    param([string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        return $false
    }
    try {
        & $PythonPath -c "import torch, ultralytics; raise SystemExit(0 if torch.cuda.is_available() else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-TrainingPython {
    $pythonCandidates = @()
    if ($env:WIND_TRAIN_PYTHON) {
        $pythonCandidates += $env:WIND_TRAIN_PYTHON
    }
    $pythonCandidates += @(
        ".\.venv\Scripts\python.exe",
        "E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe",
        "C:\Users\li152\anaconda3\python.exe"
    )

    foreach ($candidate in $pythonCandidates) {
        if (Test-TrainingPython $candidate) {
            return $candidate
        }
    }
    throw "No CUDA-ready Python environment found."
}

function Get-LastEpoch {
    param([string]$RunName)
    $resultsCsv = Join-Path $bundleRoot "runs\detect\runs\wind_fault\$RunName\results.csv"
    if (-not (Test-Path -LiteralPath $resultsCsv)) {
        return -1
    }
    $rows = Import-Csv -LiteralPath $resultsCsv
    if (-not $rows -or $rows.Count -eq 0) {
        return -1
    }
    $last = $rows | Select-Object -Last 1
    return [int]$last.epoch
}

function Wait-ForRunComplete {
    param(
        [string]$RunName,
        [int]$ExpectedEpochs,
        [int]$PollSeconds,
        [int]$MaxWaitHours
    )

    $deadline = (Get-Date).AddHours($MaxWaitHours)
    $runDir = Join-Path $bundleRoot "runs\detect\runs\wind_fault\$RunName"
    Write-Log "Waiting for current run: $RunName"

    while ((Get-Date) -lt $deadline) {
        $lastEpoch = Get-LastEpoch $RunName
        $resultsPng = Join-Path $runDir "results.png"
        $bestPt = Join-Path $runDir "weights\best.pt"
        $lastPt = Join-Path $runDir "weights\last.pt"

        Write-Log "Current run last epoch: $lastEpoch / $ExpectedEpochs"
        if ($lastEpoch -ge $ExpectedEpochs -and
            (Test-Path -LiteralPath $resultsPng) -and
            (Test-Path -LiteralPath $bestPt) -and
            (Test-Path -LiteralPath $lastPt)) {
            Write-Log "Current run appears complete."
            return
        }

        Start-Sleep -Seconds $PollSeconds
    }

    throw "Timed out waiting for current run to finish: $RunName"
}

function Start-V51Experiment {
    param(
        [string]$OutputDir,
        [int]$CorrosionRepeat,
        [int]$CrackRepeat,
        [string]$RunName
    )

    $python = Get-TrainingPython
    Write-Log "Using Python: $python"
    Write-Log "Building dataset: $OutputDir corrosion_repeat=$CorrosionRepeat crack_repeat=$CrackRepeat"
    & $python scripts\build_v5_1_dataset.py `
        --output $OutputDir `
        --corrosion-repeat $CorrosionRepeat `
        --crack-repeat $CrackRepeat

    Write-Log "Starting training: $RunName"
    & $python scripts\train_yolo.py `
        --data "$OutputDir\data.yaml" `
        --model models\v04_negative_1024_best.pt `
        --epochs 50 `
        --imgsz 1024 `
        --batch 24 `
        --workers 4 `
        --cache false `
        --optimizer AdamW `
        --lr0 0.0005 `
        --lrf 0.05 `
        --close-mosaic 15 `
        --project runs\wind_fault `
        --name $RunName `
        --device 0
    Write-Log "Finished training: $RunName"
}

Write-Log "Tonight V5.1 queue started."
Wait-ForRunComplete `
    -RunName $currentRunName `
    -ExpectedEpochs $expectedEpochs `
    -PollSeconds $pollSeconds `
    -MaxWaitHours $maxWaitHours

Start-V51Experiment `
    -OutputDir "dataset_v5_1_no_leak_c3_crack2" `
    -CorrosionRepeat 3 `
    -CrackRepeat 2 `
    -RunName "v5_1_no_leak_1024_b24_w4_nocache_from_v04_corrosion_x3_crack_x2_50e"

Start-V51Experiment `
    -OutputDir "dataset_v5_1_no_leak_c4_crack1" `
    -CorrosionRepeat 4 `
    -CrackRepeat 1 `
    -RunName "v5_1_no_leak_1024_b24_w4_nocache_from_v04_corrosion_x4_crack_x1_50e"

Write-Log "Tonight V5.1 queue finished."
