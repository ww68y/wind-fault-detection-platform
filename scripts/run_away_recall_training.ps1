param(
    [int]$Epochs640 = 80,
    [int]$Epochs832 = 80,
    [string]$Device = "0",
    [switch]$Cpu,
    [switch]$Skip832,
    [switch]$SkipDatasetBuild
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TaskDir = Join-Path $ProjectRoot "runs\wind_fault\logs\recall_task_$Stamp"
$LogPath = Join-Path $TaskDir "recall_task.log"
$SummaryPath = Join-Path $TaskDir "recall_task_summary.txt"
$StopFile = Join-Path $ProjectRoot "STOP_RECALL_TASK.txt"

New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Yolo = Join-Path $ProjectRoot ".venv\Scripts\yolo.exe"
$DeviceArg = if ($Cpu) { "cpu" } else { $Device }
$SeedModel = "runs\detect\runs\wind_fault\hardcase\v02_improve_corrosion_640_b4_30e_20260604_212514\weights\best.pt"
if (-not (Test-Path $SeedModel)) {
    $SeedModel = "runs\detect\runs\wind_fault\hardcase\v02_hardcase_640_b4_40e_20260604_190625\weights\best.pt"
}
if (-not (Test-Path $SeedModel)) {
    $SeedModel = "yolov8n.pt"
}

function Write-TaskLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Test-StopRequested {
    if (Test-Path $StopFile) {
        Write-TaskLog "Stop file detected: $StopFile"
        return $true
    }
    return $false
}

function Invoke-Logged {
    param(
        [string]$StepName,
        [string]$Exe,
        [string[]]$Arguments
    )
    if (Test-StopRequested) {
        throw "Stop requested before step: $StepName"
    }
    Write-TaskLog "START $StepName"
    Write-TaskLog ("COMMAND {0} {1}" -f $Exe, ($Arguments -join " "))
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
    Write-TaskLog "END $StepName exit=$exitCode"
    if ($exitCode -ne 0) {
        throw "Step failed: $StepName exit=$exitCode"
    }
}

function Invoke-ModelEval {
    param(
        [string]$Name,
        [string]$BestModel,
        [int]$ImageSize,
        [int]$Batch
    )
    Invoke-Logged `
        -StepName "$Name val" `
        -Exe $Yolo `
        -Arguments @(
            "detect", "val",
            "model=$BestModel",
            "data=datasets\wind_blade_defect_v03_recall\data.yaml",
            "split=val",
            "imgsz=$ImageSize",
            "batch=$Batch",
            "device=$DeviceArg",
            "project=runs\wind_fault\overnight_eval",
            "name=$Name`_val"
        )

    foreach ($conf in @("0.3", "0.2")) {
        $confName = $conf.Replace(".", "")
        $PredictName = "$Name`_conf$confName"
        Invoke-Logged `
            -StepName "$Name hard_cases conf=$conf" `
            -Exe $Python `
            -Arguments @(
                "scripts\predict_yolo.py",
                "--model", $BestModel,
                "--source", "hard_cases\missed_samples",
                "--conf", $conf,
                "--project", "runs\wind_fault\overnight_hardcases",
                "--name", $PredictName,
                "--device", $DeviceArg
            )
        Invoke-Logged `
            -StepName "$Name hard_cases error mining conf=$conf" `
            -Exe $Python `
            -Arguments @(
                "scripts\mine_detection_errors.py",
                "--data", "datasets\wind_blade_defect\data.yaml",
                "--images", "hard_cases\missed_samples",
                "--labels", "datasets\wind_blade_defect\labels\test",
                "--report", "runs\wind_fault\overnight_hardcases\$PredictName\reports\detection_report.json",
                "--output", "runs\wind_fault\overnight_hardcases\$PredictName\error_mining",
                "--iou", "0.5",
                "--low-conf", "0.45"
            )
    }
}

try {
    Write-TaskLog "Recall training task started."
    Write-TaskLog "Task dir: $TaskDir"
    Write-TaskLog "Seed model: $SeedModel"
    Write-TaskLog "Device: $DeviceArg"

    if (-not $SkipDatasetBuild) {
        Invoke-Logged `
            -StepName "build v03 recall dataset" `
            -Exe $Python `
            -Arguments @("scripts\prepare_v03_recall_dataset.py", "--overwrite")
    }

    $Name640 = "v03_recall_640_b4_${Epochs640}e_$Stamp"
    Invoke-Logged `
        -StepName "train $Name640" `
        -Exe $Python `
        -Arguments @(
            "scripts\train_yolo.py",
            "--data", "datasets\wind_blade_defect_v03_recall\data.yaml",
            "--model", $SeedModel,
            "--epochs", "$Epochs640",
            "--imgsz", "640",
            "--batch", "4",
            "--project", "runs\wind_fault\overnight",
            "--name", $Name640,
            "--device", $DeviceArg
        )

    $Best640 = "runs\detect\runs\wind_fault\overnight\$Name640\weights\best.pt"
    Invoke-ModelEval -Name $Name640 -BestModel $Best640 -ImageSize 640 -Batch 4

    $Best832Seed = if (Test-Path $Best640) { $Best640 } else { $SeedModel }
    if (-not $Skip832) {
        $Name832 = "v03_recall_832_b2_${Epochs832}e_$Stamp"
        Invoke-Logged `
            -StepName "train $Name832" `
            -Exe $Python `
            -Arguments @(
                "scripts\train_yolo.py",
                "--data", "datasets\wind_blade_defect_v03_recall\data.yaml",
                "--model", $Best832Seed,
                "--epochs", "$Epochs832",
                "--imgsz", "832",
                "--batch", "2",
                "--project", "runs\wind_fault\overnight",
                "--name", $Name832,
                "--device", $DeviceArg
            )

        $Best832 = "runs\detect\runs\wind_fault\overnight\$Name832\weights\best.pt"
        Invoke-ModelEval -Name $Name832 -BestModel $Best832 -ImageSize 832 -Batch 2
    }

    @(
        "status: complete",
        "task_dir: $TaskDir",
        "log: $LogPath",
        "seed_model: $SeedModel",
        "v03_640_model: $Best640",
        "v03_832_model: $(if ($Skip832) { 'skipped' } else { $Best832 })",
        "val_results_dir: runs\detect\runs\wind_fault\overnight_eval",
        "hardcase_results_dir: runs\detect\runs\wind_fault\overnight_hardcases"
    ) | Set-Content -Path $SummaryPath -Encoding UTF8
    Write-TaskLog "Recall training task complete."
    Write-TaskLog "Summary: $SummaryPath"
}
catch {
    @(
        "status: failed",
        "task_dir: $TaskDir",
        "log: $LogPath",
        "error: $($_.Exception.Message)"
    ) | Set-Content -Path $SummaryPath -Encoding UTF8
    Write-TaskLog "FAILED $($_.Exception.Message)"
    exit 1
}
