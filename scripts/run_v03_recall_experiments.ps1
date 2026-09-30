param(
    [ValidateSet("all", "v02_640", "v03_640", "v03_832")]
    [string]$Experiment = "all",
    [int]$Epochs = 80,
    [string]$Device = "0",
    [switch]$Cpu,
    [switch]$BuildDataset
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($BuildDataset) {
    python scripts/prepare_v03_recall_dataset.py --overwrite
}

$deviceArgs = @()
if ($Cpu) {
    $deviceArgs = @("--device", "cpu")
} elseif ($Device -ne "") {
    $deviceArgs = @("--device", $Device)
}

function Invoke-RecallTrain {
    param(
        [string]$Key,
        [string]$Data,
        [int]$ImageSize,
        [int]$Batch
    )

    if ($Experiment -ne "all" -and $Experiment -ne $Key) {
        return
    }

    python scripts/train_yolo.py `
        --data $Data `
        --model yolov8n.pt `
        --epochs $Epochs `
        --imgsz $ImageSize `
        --batch $Batch `
        --project runs/wind_fault `
        --name $Key `
        @deviceArgs
}

Invoke-RecallTrain `
    -Key "v02_640" `
    -Data "datasets/wind_blade_defect_v02_improve/data.yaml" `
    -ImageSize 640 `
    -Batch 4

Invoke-RecallTrain `
    -Key "v03_640" `
    -Data "datasets/wind_blade_defect_v03_recall/data.yaml" `
    -ImageSize 640 `
    -Batch 4

Invoke-RecallTrain `
    -Key "v03_832" `
    -Data "datasets/wind_blade_defect_v03_recall/data.yaml" `
    -ImageSize 832 `
    -Batch 2
