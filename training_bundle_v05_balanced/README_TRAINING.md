# v05 balanced training bundle

This bundle is prepared for the stronger Windows training computer.

## Contents

- `dataset/`: v05 balanced YOLO dataset
- `models/v03_recall_832_best.pt`: recommended seed model
- `models/v04_partial_best.pt`: interrupted local v04 seed model for comparison
- `scripts/train_yolo.py`: training entry
- `src/wind_fault/`: project package used by the training script
- `requirements.txt` and `pyproject.toml`: Python dependencies and package config

## Recommended command

Open PowerShell in this folder and run:

```powershell
.\start_training_1024_b8.ps1
```

If 1024 is too slow or reports CUDA out of memory, run:

```powershell
.\start_training_832_b16.ps1
```

For an extra comparison run that starts from the interrupted local v04 model:

```powershell
.\start_training_1024_b8_from_v04_partial.ps1
```

## Dataset

The dataset keeps the existing validation and test splits unchanged for fair comparison.

- train: 7611 images
- val: 571 images
- test_clean: 543 images

Class mapping:

- `0 crack`
- `1 hole`
- `2 spalling`
- `3 corrosion`

The best model will be saved under:

```text
runs\wind_fault\experiment_name\weights\best.pt
```
