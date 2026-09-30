param(
    [string]$Python = "E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe",
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\yolop2_strategy"
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "p2_strategy_queue_$Stamp.status.log"

$TrainScript = Join-Path $ProjectRoot "scripts\train_yolop2.py"
$AnalyzeScript = Join-Path $ProjectRoot "scripts\analyze_yolop2_model.py"
$DataYaml = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative\data.yaml"
$ModelConfig = Join-Path $ProjectRoot "configs\yolov8n-p2-wind.yaml"
$V4Seed = Join-Path $ProjectRoot "models\v04_negative_1024_best.pt"

$WarmupName = "p2_head_warmup_15e_amp_b16_from_v04_$Stamp"
$FinetuneName = "p2_full_finetune_80e_amp_b16_from_warmup_$Stamp"
$WarmupDir = Join-Path $RunRoot $WarmupName
$FinetuneDir = Join-Path $RunRoot $FinetuneName
$WarmupBest = Join-Path $WarmupDir "weights\best.pt"
$FinetuneBest = Join-Path $FinetuneDir "weights\best.pt"
$AnalysisDir = Join-Path $FinetuneDir "analysis"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $StatusLog -Value $line -Encoding UTF8
}

Write-Status "P2 strategy queue started."
Write-Status "RunRoot: $RunRoot"
Write-Status "Warmup: $WarmupName"
Write-Status "Finetune: $FinetuneName"

Write-Status "Stage 1/3: warmup started."
& $Python $TrainScript `
    --data $DataYaml `
    --model-config $ModelConfig `
    --pretrained $V4Seed `
    --imgsz 1024 `
    --epochs 15 `
    --batch 16 `
    --project $RunRoot `
    --name $WarmupName `
    --device 0 `
    --workers 4 `
    --optimizer AdamW `
    --lr0 0.001 `
    --lrf 0.1 `
    --close-mosaic 5 `
    --freeze 10 `
    --patience 15 `
    --amp true

if ($LASTEXITCODE -ne 0) {
    Write-Status "Warmup failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
if (-not (Test-Path $WarmupBest)) {
    Write-Status "Warmup finished but best.pt was not found: $WarmupBest"
    exit 2
}
Write-Status "Stage 1/3: warmup finished. Best: $WarmupBest"

Write-Status "Stage 2/3: full finetune started."
& $Python $TrainScript `
    --data $DataYaml `
    --model-config $ModelConfig `
    --pretrained $WarmupBest `
    --imgsz 1024 `
    --epochs 80 `
    --batch 16 `
    --project $RunRoot `
    --name $FinetuneName `
    --device 0 `
    --workers 4 `
    --optimizer AdamW `
    --lr0 0.00025 `
    --lrf 0.05 `
    --cos-lr `
    --close-mosaic 20 `
    --patience 30 `
    --amp true

if ($LASTEXITCODE -ne 0) {
    Write-Status "Finetune failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
if (-not (Test-Path $FinetuneBest)) {
    Write-Status "Finetune finished but best.pt was not found: $FinetuneBest"
    exit 3
}
Write-Status "Stage 2/3: full finetune finished. Best: $FinetuneBest"

Write-Status "Stage 3/3: analysis started."
& $Python $AnalyzeScript `
    --candidate $FinetuneBest `
    --baseline $V4Seed `
    --data $DataYaml `
    --output $AnalysisDir `
    --imgsz 1024 `
    --batch 16 `
    --conf 0.45 `
    --device 0

if ($LASTEXITCODE -ne 0) {
    Write-Status "Analysis failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status "Stage 3/3: analysis finished. Report: $(Join-Path $AnalysisDir 'analysis.md')"
Write-Status "P2 strategy queue finished."
