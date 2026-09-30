$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -e .

$datasetPath = (Resolve-Path ".\dataset").Path.Replace("\", "/")
@"
path: $datasetPath
train: images/train
val: images/val
test: images/test_clean

names:
  0: crack
  1: hole
  2: spalling
  3: corrosion
"@ | Set-Content -LiteralPath ".\dataset\data.yaml" -Encoding UTF8

$cudaAvailable = .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
if (($cudaAvailable | Select-Object -Last 1).Trim() -ne "True") {
    throw "CUDA is not available in this venv. Install CUDA PyTorch first, for example: .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
}

.\.venv\Scripts\python.exe scripts\train_yolo.py `
    --data dataset\data.yaml `
    --model models\v03_recall_832_best.pt `
    --epochs 80 `
    --imgsz 832 `
    --batch 16 `
    --workers 4 `
    --project runs\wind_fault `
    --name v05_balanced_832_b16_80e `
    --device 0
