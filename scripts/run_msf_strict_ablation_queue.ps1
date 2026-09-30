param(
    [string]$Python = "E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe",
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [int]$WarmupEpochs = 15,
    [int]$FinetuneEpochs = 80,
    [int]$Batch = 16,
    [switch]$SkipP2Control,
    [switch]$SkipBladeScan,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\msf_yolo_strict_ablation"
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "msf_strict_ablation_queue_$Stamp.status.log"

$TrainScript = Join-Path $ProjectRoot "scripts\train_msf_yolo_full.py"
$AnalyzeScript = Join-Path $ProjectRoot "scripts\analyze_yolop2_model.py"
$BladeSliceScript = Join-Path $ProjectRoot "scripts\evaluate_blade_slice.py"
$SummaryScript = Join-Path $ProjectRoot "scripts\summarize_strict_ablation.py"
$DataYaml = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative\data.yaml"
$DatasetRoot = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative"
$ModelConfig = Join-Path $ProjectRoot "configs\yolov8n-p2-wind.yaml"
$V4Seed = Join-Path $ProjectRoot "models\v04_negative_1024_best.pt"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $StatusLog -Value $line -Encoding UTF8
}

function Invoke-Checked {
    param(
        [string]$Stage,
        [scriptblock]$Command
    )
    Write-Status "$Stage started."
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Status "$Stage failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
    Write-Status "$Stage finished."
}

function Invoke-TrainPair {
    param(
        [string]$Variant,
        [string]$ShapeAware,
        [string]$HardReweight,
        [double]$ClsPw
    )

    $WarmupName = "${Variant}_warmup_${WarmupEpochs}e_amp_b${Batch}_from_v4_$Stamp"
    $FinetuneName = "${Variant}_finetune_${FinetuneEpochs}e_amp_b${Batch}_from_warmup_$Stamp"
    $WarmupDir = Join-Path $RunRoot $WarmupName
    $FinetuneDir = Join-Path $RunRoot $FinetuneName
    $WarmupBest = Join-Path $WarmupDir "weights\best.pt"
    $FinetuneBest = Join-Path $FinetuneDir "weights\best.pt"
    $AnalysisDir = Join-Path $FinetuneDir "analysis"

    Write-Status "Variant $Variant configured: shape_aware=$ShapeAware hard_reweight=$HardReweight cls_pw=$ClsPw"
    Write-Status "Variant $Variant warmup: $WarmupName"
    Write-Status "Variant $Variant finetune: $FinetuneName"

    Invoke-Checked "Variant $Variant warmup" {
        & $Python $TrainScript `
            --data $DataYaml `
            --model-config $ModelConfig `
            --pretrained $V4Seed `
            --imgsz 1024 `
            --epochs $WarmupEpochs `
            --batch $Batch `
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
            --amp true `
            --cls-pw $ClsPw `
            --hard-reweight $HardReweight `
            --corrosion-weight 3.0 `
            --crack-weight 2.0 `
            --small-target-weight 1.5 `
            --small-target-max-area 0.012 `
            --small-target-class-ids 0 3 `
            --shape-aware $ShapeAware `
            --shape-area-threshold 0.012 `
            --shape-small-gain 0.35 `
            --shape-slender-gain 0.25 `
            --shape-slender-ratio 3.0 `
            --shape-max-weight 1.7 `
            --multi-scale 0.10 `
            --mosaic 1.0 `
            --copy-paste 0.05 `
            --scale 0.60
    }

    if (-not (Test-Path $WarmupBest)) {
        Write-Status "Variant $Variant warmup best.pt not found: $WarmupBest"
        exit 2
    }

    Invoke-Checked "Variant $Variant finetune" {
        & $Python $TrainScript `
            --data $DataYaml `
            --model-config $ModelConfig `
            --pretrained $WarmupBest `
            --imgsz 1024 `
            --epochs $FinetuneEpochs `
            --batch $Batch `
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
            --amp true `
            --cls-pw $ClsPw `
            --hard-reweight $HardReweight `
            --corrosion-weight 3.0 `
            --crack-weight 2.0 `
            --small-target-weight 1.5 `
            --small-target-max-area 0.012 `
            --small-target-class-ids 0 3 `
            --shape-aware $ShapeAware `
            --shape-area-threshold 0.012 `
            --shape-small-gain 0.35 `
            --shape-slender-gain 0.25 `
            --shape-slender-ratio 3.0 `
            --shape-max-weight 1.7 `
            --multi-scale 0.10 `
            --mosaic 1.0 `
            --copy-paste 0.05 `
            --scale 0.60
    }

    if (-not (Test-Path $FinetuneBest)) {
        Write-Status "Variant $Variant finetune best.pt not found: $FinetuneBest"
        exit 3
    }

    Invoke-Checked "Variant $Variant V4 comparison" {
        & $Python $AnalyzeScript `
            --candidate $FinetuneBest `
            --baseline $V4Seed `
            --data $DataYaml `
            --output $AnalysisDir `
            --imgsz 1024 `
            --batch $Batch `
            --conf 0.45 `
            --device 0
    }
}

function Invoke-BladeSliceScan {
    $ScanRoot = Join-Path $RunRoot "blade_slice_scan_$Stamp"
    New-Item -ItemType Directory -Force -Path $ScanRoot | Out-Null
    $TileSizes = @(512, 640, 768, 896)
    foreach ($Tile in $TileSizes) {
        $OutputDir = Join-Path $ScanRoot "tile_${Tile}_overlap025"
        Invoke-Checked "Blade-Slice scan tile=$Tile overlap=0.25" {
            & $Python $BladeSliceScript `
                --model $V4Seed `
                --dataset $DatasetRoot `
                --data $DataYaml `
                --splits val test_clean `
                --output $OutputDir `
                --imgsz 1024 `
                --tile-size $Tile `
                --overlap 0.25 `
                --conf 0.45 `
                --low-conf 0.45 `
                --eval-iou 0.5 `
                --nms-iou 0.5 `
                --device 0 `
                --modes blade_slice_fused
        }
    }
}

Write-Status "MSF-YOLO strict ablation queue started."
Write-Status "RunRoot: $RunRoot"
Write-Status "Stamp: $Stamp"
Write-Status "WarmupEpochs=$WarmupEpochs FinetuneEpochs=$FinetuneEpochs Batch=$Batch"

if ($DryRun) {
    Write-Status "DryRun requested; queue configuration validated without starting training."
    exit 0
}

if (-not $SkipP2Control) {
    Invoke-TrainPair -Variant "p2_plain_control" -ShapeAware "false" -HardReweight "false" -ClsPw 0.0
}

Invoke-TrainPair -Variant "p2_shape_only" -ShapeAware "true" -HardReweight "false" -ClsPw 0.0
Invoke-TrainPair -Variant "p2_hard_only" -ShapeAware "false" -HardReweight "true" -ClsPw 0.35
Invoke-TrainPair -Variant "p2_shape_hard" -ShapeAware "true" -HardReweight "true" -ClsPw 0.35

if (-not $SkipBladeScan) {
    Invoke-BladeSliceScan
}

Invoke-Checked "Strict ablation summary generation" {
    & $Python $SummaryScript `
        --run-root $RunRoot `
        --output (Join-Path $RunRoot "strict_ablation_summary_$Stamp")
}

Write-Status "MSF-YOLO strict ablation queue finished."
