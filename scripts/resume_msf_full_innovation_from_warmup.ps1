param(
    [Parameter(Mandatory = $true)]
    [string]$WarmupBest,
    [string]$Python = "E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe",
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\msf_yolo_full_innovation"
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "msf_full_innovation_resume_$Stamp.status.log"

$TrainScript = Join-Path $ProjectRoot "scripts\train_msf_yolo_full.py"
$AnalyzeScript = Join-Path $ProjectRoot "scripts\analyze_yolop2_model.py"
$BladeSliceScript = Join-Path $ProjectRoot "scripts\evaluate_blade_slice.py"
$DataYaml = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative\data.yaml"
$DatasetRoot = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative"
$ModelConfig = Join-Path $ProjectRoot "configs\yolov8n-p2-wind.yaml"
$V4Seed = Join-Path $ProjectRoot "models\v04_negative_1024_best.pt"

$FinetuneName = "msf_full_p2_shape_hard_finetune_60e_amp_b12_from_warmup_$Stamp"
$FinetuneDir = Join-Path $RunRoot $FinetuneName
$FinetuneBest = Join-Path $FinetuneDir "weights\best.pt"
$AnalysisDir = Join-Path $FinetuneDir "analysis"
$BladeSliceDir = Join-Path $FinetuneDir "blade_slice_eval"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $StatusLog -Value $line -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $WarmupBest)) {
    throw "Warmup best.pt not found: $WarmupBest"
}

Write-Status "MSF-YOLO full innovation resume started."
Write-Status "WarmupBest: $WarmupBest"
Write-Status "Finetune: $FinetuneName"

Write-Status "Stage 2/4: full-innovation finetune started."
& $Python $TrainScript `
    --data $DataYaml `
    --model-config $ModelConfig `
    --pretrained $WarmupBest `
    --imgsz 1024 `
    --epochs 60 `
    --batch 12 `
    --project $RunRoot `
    --name $FinetuneName `
    --device 0 `
    --workers 4 `
    --optimizer AdamW `
    --lr0 0.00018 `
    --lrf 0.05 `
    --cos-lr `
    --close-mosaic 20 `
    --patience 25 `
    --amp true `
    --cls-pw 0.35 `
    --hard-reweight true `
    --corrosion-weight 3.0 `
    --crack-weight 2.0 `
    --small-target-weight 1.5 `
    --small-target-max-area 0.012 `
    --small-target-class-ids 0 3 `
    --shape-aware true `
    --shape-area-threshold 0.012 `
    --shape-small-gain 0.35 `
    --shape-slender-gain 0.25 `
    --shape-slender-ratio 3.0 `
    --shape-max-weight 1.7 `
    --multi-scale 0.10 `
    --mosaic 1.0 `
    --copy-paste 0.05 `
    --scale 0.60

if ($LASTEXITCODE -ne 0) {
    Write-Status "Finetune failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
if (-not (Test-Path $FinetuneBest)) {
    Write-Status "Finetune finished but best.pt was not found: $FinetuneBest"
    exit 3
}
Write-Status "Stage 2/4: finetune finished. Best: $FinetuneBest"

Write-Status "Stage 3/4: V4 baseline comparison started."
& $Python $AnalyzeScript `
    --candidate $FinetuneBest `
    --baseline $V4Seed `
    --data $DataYaml `
    --output $AnalysisDir `
    --imgsz 1024 `
    --batch 12 `
    --conf 0.45 `
    --device 0

if ($LASTEXITCODE -ne 0) {
    Write-Status "Analysis failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Write-Status "Stage 3/4: analysis finished. Report: $(Join-Path $AnalysisDir 'analysis.md')"

Write-Status "Stage 4/4: Blade-Slice evaluation started."
& $Python $BladeSliceScript `
    --model $FinetuneBest `
    --dataset $DatasetRoot `
    --data $DataYaml `
    --splits val test_clean `
    --output $BladeSliceDir `
    --imgsz 1024 `
    --tile-size 640 `
    --overlap 0.25 `
    --conf 0.45 `
    --low-conf 0.45 `
    --eval-iou 0.5 `
    --nms-iou 0.5 `
    --device 0 `
    --modes normal blade_slice_fused

if ($LASTEXITCODE -ne 0) {
    Write-Status "Blade-Slice evaluation failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Write-Status "Stage 4/4: Blade-Slice evaluation finished. Report: $(Join-Path $BladeSliceDir 'summary.md')"
Write-Status "MSF-YOLO full innovation resume finished."
