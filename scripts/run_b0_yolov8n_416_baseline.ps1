param(
    [string]$Python = "",
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [int]$Epochs = 50,
    [int]$Batch = 16,
    [string]$Device = "0",
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $ExternalGpuPython = "E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe"
    $ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $ExternalGpuPython) {
        $Python = $ExternalGpuPython
    } elseif (Test-Path $ProjectPython) {
        $Python = $ProjectPython
    } else {
        $Python = "python"
    }
}

$RunRoot = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\b0_yolov8n_416_baseline"
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "b0_yolov8n_416_baseline_$Stamp.status.log"

$TrainScript = Join-Path $ProjectRoot "scripts\train_yolo.py"
$AnalyzeScript = Join-Path $ProjectRoot "scripts\analyze_yolop2_model.py"
$DataYaml = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative\data.yaml"
$ModelSeed = Join-Path $ProjectRoot "yolov8n.pt"
$V6Best = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\msf_yolo_full_innovation\msf_full_p2_shape_hard_finetune_60e_amp_b12_from_warmup_20260606_182744\weights\best.pt"

$RunName = "b0_vanilla_yolov8n_416_b${Batch}_${Epochs}e_$Stamp"
$RunDir = Join-Path $RunRoot $RunName
$B0Best = Join-Path $RunDir "weights\best.pt"
$CompareDir = Join-Path $RunDir "v6_vs_b0_analysis"
$AnalysisBatch = [Math]::Min($Batch, 12)

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $StatusLog -Value $line -Encoding UTF8
}

if (-not (Test-Path $TrainScript)) {
    throw "Train script not found: $TrainScript"
}
if (-not (Test-Path $AnalyzeScript)) {
    throw "Analyze script not found: $AnalyzeScript"
}
if (-not (Test-Path $DataYaml)) {
    throw "Data yaml not found: $DataYaml"
}
if (-not (Test-Path $ModelSeed)) {
    throw "YOLO seed not found: $ModelSeed"
}
if (-not (Test-Path $V6Best)) {
    throw "V6 best.pt not found: $V6Best"
}
if ($Epochs -lt 1) {
    throw "Epochs must be greater than 0. Ultralytics treats epochs=0 as its default training length."
}
if ($Batch -lt 1) {
    throw "Batch must be greater than 0."
}

Write-Status "B0 YOLOv8n-416 baseline queue started."
Write-Status "Python: $Python"
Write-Status "Data: $DataYaml"
Write-Status "Model seed: $ModelSeed"
Write-Status "Run: $RunDir"
Write-Status "V6 candidate: $V6Best"
Write-Status "Analysis batch: $AnalysisBatch"

Write-Status "Stage 1/2: train B0 YOLOv8n-416."
& $Python $TrainScript `
    --data $DataYaml `
    --model $ModelSeed `
    --imgsz 416 `
    --epochs $Epochs `
    --batch $Batch `
    --project $RunRoot `
    --name $RunName `
    --device $Device `
    --workers $Workers

if ($LASTEXITCODE -ne 0) {
    Write-Status "B0 training failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
if (-not (Test-Path $B0Best)) {
    Write-Status "B0 training finished but best.pt was not found: $B0Best"
    exit 2
}
Write-Status "Stage 1/2 finished. B0 best: $B0Best"

Write-Status "Stage 2/2: compare V6 against B0 with native evaluation sizes."
& $Python $AnalyzeScript `
    --candidate $V6Best `
    --baseline $B0Best `
    --data $DataYaml `
    --output $CompareDir `
    --imgsz 1024 `
    --baseline-imgsz 416 `
    --candidate-imgsz 1024 `
    --batch $AnalysisBatch `
    --conf 0.45 `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    Write-Status "V6 vs B0 analysis failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status "Stage 2/2 finished. Report: $(Join-Path $CompareDir 'analysis.md')"
Write-Status "B0 YOLOv8n-416 baseline queue finished."
