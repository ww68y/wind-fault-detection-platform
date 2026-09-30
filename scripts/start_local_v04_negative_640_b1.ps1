$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$env:PYTHONUTF8 = "1"

.\.venv\Scripts\python.exe scripts\train_yolo.py `
    --data datasets\wind_blade_defect_v04_negative\data.yaml `
    --model runs\detect\runs\wind_fault\overnight\v03_recall_832_b2_80e_20260604_224650\weights\best.pt `
    --epochs 60 `
    --imgsz 640 `
    --batch 1 `
    --workers 0 `
    --project runs\wind_fault\local `
    --name v04_negative_local_640_b1_60e `
    --device 0
