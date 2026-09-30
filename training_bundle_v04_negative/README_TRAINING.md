# v04 negative training bundle

This bundle is prepared for training on another Windows computer.

## Contents

- `dataset/`: portable YOLO dataset with confirmed negative samples added
- `models/v03_recall_832_best.pt`: seed model from the current best v03_832 run
- `scripts/train_yolo.py`: training entry
- `src/wind_fault/`: minimal project package used by the training script
- `requirements.txt` and `pyproject.toml`: Python dependencies and package config

## Recommended commands

Open PowerShell in this folder and run:

```powershell
.\start_training_832.ps1
```

If the other computer has a strong NVIDIA GPU, you can run the 1024 small-defect experiment:

```powershell
.\start_training_1024.ps1
```

For a weaker GPU, run the lighter 640 experiment:

```powershell
.\start_training_640.ps1
```

Suggested choice:

- GPU 4GB to 6GB: `.\start_training_640.ps1`
- GPU 8GB to 10GB: `.\start_training_832.ps1`
- GPU 12GB or better: `.\start_training_1024.ps1`

## Manual command

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python scripts\train_yolo.py `
  --data dataset\data.yaml `
  --model models\v03_recall_832_best.pt `
  --epochs 60 `
  --imgsz 832 `
  --batch 2 `
  --project runs\wind_fault `
  --name v04_negative_832_b2_60e `
  --device 0
```

Use `--device cpu` only if the target computer has no NVIDIA GPU. CPU training will be very slow.

## Chinese quick start

1. Copy this whole folder to the other computer.
2. Install Python 3.10 or 3.11.
3. Open PowerShell in this folder.
4. Run one of these commands:

```powershell
.\start_training_832.ps1
```

or, for a stronger GPU:

```powershell
.\start_training_1024.ps1
```

After training, the best model will be saved under:

```text
runs\wind_fault\experiment_name\weights\best.pt
```

## Goal

This v04 dataset mainly adds confirmed negative samples, so the expected improvement is fewer false positives. After training, compare:

- validation precision and recall
- hard-case false positive count
- corrosion recall
- real image behavior at `conf=0.4` to `0.45`
