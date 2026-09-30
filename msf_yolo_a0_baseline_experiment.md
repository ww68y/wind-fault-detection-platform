# MSF-YOLO A0 普通 YOLO 弱基线实验记录

记录日期：2026-06-08

## 1. 实验定位

本轮构建一版合理的弱基线模型，用于和 MSF-YOLO V6 做对比。该模型不是故意破坏训练过程，而是保留普通轻量 YOLO 检测器的标准设置，去掉 MSF-YOLO 中用于风机叶片缺陷任务的增强模块。

该模型用于论文中的 `A0 YOLO baseline` 或 `Vanilla YOLO baseline`，不替换 V4 稳定基线，不接入 Web/API 默认模型。

## 2. A0 模型定义

| 项目 | 设置 |
| --- | --- |
| 模型名称 | A0 Vanilla-YOLO-640 |
| 初始权重 | `yolov8n.pt` |
| 数据集 | `datasets/wind_blade_defect_v05_conservative/data.yaml` |
| 输入尺寸 | 640 |
| epoch | 80 |
| P2 小目标检测头 | 否 |
| Shape-Aware Loss | 否 |
| 困难样本重加权 | 否 |
| Blade-Slice 推理融合 | 否 |
| V4/V6 权重初始化 | 否 |

## 3. 新增脚本

```text
scripts/run_a0_vanilla_yolo_baseline.ps1
```

该脚本执行两步：

1. 训练 A0 Vanilla-YOLO-640。
2. 使用现有 V6 权重与 A0 做标准指标和背景误检对比。

脚本会优先使用已验证的 CUDA 训练环境：

```text
E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe
```

如果该环境不存在，则回退到项目本地 `.venv` 或系统 `python`。

默认输出目录：

```text
runs/detect/runs/wind_fault/a0_vanilla_yolo_baseline/
```

## 4. 启动命令

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a0_vanilla_yolo_baseline.ps1
```

低显存或只做链路检查时可用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a0_vanilla_yolo_baseline.ps1 -Epochs 1 -Batch 1 -Workers 0
```

## 5. 对比口径

V6 和 A0 使用同一数据划分，但采用各自原生评估尺寸：

| 模型 | 评估尺寸 | 说明 |
| --- | ---: | --- |
| A0 Vanilla-YOLO-640 | 640 | 普通轻量 YOLO 基线 |
| MSF-YOLO V6 | 1024 | 完整创新集合模型 |

分析脚本已支持：

```text
--baseline-imgsz 640
--candidate-imgsz 1024
```

因此生成的 `analysis.md` 中 `candidate - baseline` 即为 `V6 - A0`。

## 6. 论文表述建议

可以写：

> 为验证 MSF-YOLO V6 在风机叶片缺陷检测任务中的有效性，本文构建了不包含 P2 小目标检测头、Shape-Aware Loss、困难样本重加权和 Blade-Slice 推理融合的普通 YOLO 基线模型 A0。A0 使用相同数据划分进行训练与评估，用于衡量通用轻量检测器在该任务上的基础性能。

不建议写：

> 本文故意降低基线模型性能以突出 V6。

更稳妥的结论边界：

> 相较普通 YOLO 基线，MSF-YOLO V6 在高分辨率输入、小尺度缺陷建模和困难样本处理方面具备更完整的任务适配能力。

## 7. 待填结果表

训练完成后，从以下文件回填：

```text
runs/detect/runs/wind_fault/a0_vanilla_yolo_baseline/<run_name>/v6_vs_a0_analysis/analysis.md
```

| Split | 模型 | Precision | Recall | mAP50 | mAP50-95 | FP_bg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | A0 Vanilla-YOLO-640 | 待填 | 待填 | 待填 | 待填 | 待填 |
| val | MSF-YOLO V6 | 待填 | 待填 | 待填 | 待填 | 待填 |
| test_clean | A0 Vanilla-YOLO-640 | 待填 | 待填 | 待填 | 待填 | 待填 |
| test_clean | MSF-YOLO V6 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 8. B0 低分辨率普通 YOLO 基线

由于 A0 Vanilla-YOLO-640 在训练中表现较强，补充 B0 作为低分辨率普通检测器基线。B0 不是故意破坏模型，而是模拟更低算力、更低输入分辨率下的常规 YOLO 检测方案。

| 项目 | 设置 |
| --- | --- |
| 模型名称 | B0 YOLOv8n-416 |
| 初始权重 | `yolov8n.pt` |
| 数据集 | `datasets/wind_blade_defect_v05_conservative/data.yaml` |
| 输入尺寸 | 416 |
| epoch | 50 |
| P2 小目标检测头 | 否 |
| Shape-Aware Loss | 否 |
| 困难样本重加权 | 否 |
| Blade-Slice 推理融合 | 否 |
| V4/V6 权重初始化 | 否 |

新增脚本：

```text
scripts/run_b0_yolov8n_416_baseline.ps1
scripts/run_b0_after_a0.ps1
```

`run_b0_after_a0.ps1` 会等待 A0 跑满 80 epoch 且存在 `weights/best.pt` 后，自动启动 B0 训练和 V6 vs B0 对比。

启动命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_b0_after_a0.ps1
```

B0 结果从以下文件回填：

```text
runs/detect/runs/wind_fault/b0_yolov8n_416_baseline/<run_name>/v6_vs_b0_analysis/analysis.md
```

| Split | 模型 | Precision | Recall | mAP50 | mAP50-95 | FP_bg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | B0 YOLOv8n-416 | 待填 | 待填 | 待填 | 待填 | 待填 |
| val | MSF-YOLO V6 | 待填 | 待填 | 待填 | 待填 | 待填 |
| test_clean | B0 YOLOv8n-416 | 待填 | 待填 | 待填 | 待填 | 待填 |
| test_clean | MSF-YOLO V6 | 待填 | 待填 | 待填 | 待填 | 待填 |
