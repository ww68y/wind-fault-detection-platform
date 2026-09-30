$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -e .

.\.venv\Scripts\python.exe scripts\train_yolo.py `
    --data dataset\data.yaml `
    --model models\v03_recall_832_best.pt `
    --epochs 60 `
    --imgsz 832 `
    --batch 2 `
    --project runs\wind_fault `
    --name v04_negative_832_b2_60e `
    --device 0
