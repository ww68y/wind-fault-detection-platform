$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment not found: $Python" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt"
    exit 1
}

Write-Host "Checking CUDA..." -ForegroundColor Cyan
& $Python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$CudaOk = (& $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')")[-1]
if ($CudaOk.Trim() -ne "1") {
    Write-Host "CUDA is not available. Update/restart driver first, then run this script again." -ForegroundColor Yellow
    exit 1
}

$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $ProjectRoot "runs\wind_fault\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Training {
    param(
        [string]$RunName,
        [int]$Epochs,
        [int]$ImageSize,
        [int]$Batch
    )

    $LogPath = Join-Path $LogDir "$RunName.log"
    Write-Host "Starting training: $RunName" -ForegroundColor Green
    Write-Host "Log file: $LogPath"
    Write-Host "Parameters: epochs=$Epochs imgsz=$ImageSize batch=$Batch device=0"

    & $Python scripts\train_yolo.py `
        --data datasets\wind_blade_defect\data.yaml `
        --model yolov8n.pt `
        --epochs $Epochs `
        --imgsz $ImageSize `
        --batch $Batch `
        --name $RunName `
        --device 0 2>&1 | Tee-Object -FilePath $LogPath

    return $LASTEXITCODE
}

$PrimaryName = "gpu_overnight_${TimeStamp}"
$PrimaryExit = Invoke-Training -RunName $PrimaryName -Epochs 80 -ImageSize 640 -Batch 4

if ($PrimaryExit -eq 0) {
    Write-Host "Training finished successfully: runs\wind_fault\$PrimaryName" -ForegroundColor Green
    exit 0
}

Write-Host "Primary training failed. Retrying with safer low-memory settings..." -ForegroundColor Yellow
$FallbackName = "gpu_overnight_${TimeStamp}_fallback"
$FallbackExit = Invoke-Training -RunName $FallbackName -Epochs 80 -ImageSize 512 -Batch 2

if ($FallbackExit -eq 0) {
    Write-Host "Fallback training finished successfully: runs\wind_fault\$FallbackName" -ForegroundColor Green
    exit 0
}

Write-Host "Training failed. Check logs in: $LogDir" -ForegroundColor Red
exit $FallbackExit
