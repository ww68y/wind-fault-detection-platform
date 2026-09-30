# MSF-YOLO C0 低分辨率实时 YOLO 基线实验记录

记录日期：2026-06-09

## 1. 实验定位

C0 用于补充一个更低输入分辨率的常规 YOLO 对照模型，定位为“低分辨率实时部署 baseline”。它不是故意破坏训练过程，而是在相同数据划分上使用更小输入尺寸，模拟低算力或实时巡检场景中小缺陷细节被压缩后的检测性能。

## 2. C0 模型定义

| 项目 | 设置 |
| --- | --- |
| 模型名称 | C0 YOLOv8n-320 |
| 初始权重 | `yolov8n.pt` |
| 数据集 | `datasets/wind_blade_defect_v05_conservative/data.yaml` |
| 输入尺寸 | 320 |
| epoch | 50 |
| batch | 16 |
| P2 小目标检测头 | 否 |
| Shape-Aware Loss | 否 |
| 困难样本重加权 | 否 |
| Blade-Slice 推理融合 | 否 |
| V4/V6 权重初始化 | 否 |

## 3. 新增脚本

```text
scripts/run_c0_yolov8n_320_baseline.ps1
scripts/run_c0_after_b0.ps1
```

`run_c0_after_b0.ps1` 会等待 B0 跑满 50 epoch、生成 `weights/best.pt`，并完成 `v6_vs_b0_analysis/analysis.md` 后再启动 C0，避免 B0 分析和 C0 训练同时抢占 GPU。

## 4. 对比口径

| 模型 | 评估尺寸 | 说明 |
| --- | ---: | --- |
| C0 YOLOv8n-320 | 320 | 低分辨率实时 YOLO 基线 |
| MSF-YOLO V6 | 1024 | 完整创新集合模型 |

生成的报告位置：

```text
runs/detect/runs/wind_fault/c0_yolov8n_320_baseline/<run_name>/v6_vs_c0_analysis/analysis.md
```

## 5. 待回填结果表

| Split | 模型 | Precision | Recall | mAP50 | mAP50-95 | FP_bg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | C0 YOLOv8n-320 | 待填 | 待填 | 待填 | 待填 | 待填 |
| val | MSF-YOLO V6 | 待填 | 待填 | 待填 | 待填 | 待填 |
| test_clean | C0 YOLOv8n-320 | 待填 | 待填 | 待填 | 待填 | 待填 |
| test_clean | MSF-YOLO V6 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 6. 论文表述边界

推荐表述：

> C0 用于评估低输入分辨率条件下常规轻量 YOLO 检测器的性能边界，从而分析 MSF-YOLO 在高分辨率输入与小目标特征增强设计上的必要性。

避免表述：

> C0 是故意降低性能的弱模型，用于衬托 V6。
