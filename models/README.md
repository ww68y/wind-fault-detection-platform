# Model Seeds

This directory stores candidate model weights that can be used as training seeds.

## v04_negative_1024_best.pt

Source:

```text
E:\Desktop\training_bundle_v04_negative\runs\detect\runs\wind_fault\v04_negative_1024_b2_80e-3\weights\best.pt
```

Purpose:

- Candidate training seed for later v05 / MSF-YOLO experiments.
- Trained from `models\v03_recall_832_best.pt` on the v04 negative-sample dataset.
- Main expected value: lower false positives and stronger crack behavior.
- It is not the current default web/API model.

SHA256:

```text
7DCCE600F7CAD150EDFF0F59365BE901891688DC9BADFC3BF1394B8B6E189476
```

Reference validation from the same v04 dataset:

| Split / imgsz | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| val / 640 | 0.963 | 0.938 | 0.960 | 0.823 |
| test_clean / 640 | 0.972 | 0.976 | 0.983 | 0.856 |

Example usage:

```powershell
python scripts\train_yolo.py `
  --data datasets\wind_blade_defect_v05_balanced\data.yaml `
  --model models\v04_negative_1024_best.pt `
  --epochs 80 `
  --imgsz 832 `
  --batch 8 `
  --project runs\wind_fault `
  --name v05_from_v04_negative_seed `
  --device 0
```
