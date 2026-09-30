# MSF-YOLO YOLO-P2 小目标检测头实验记录

记录日期：2026-06-06

## 1. 实验目的

本轮进入 MSF-YOLO 的 YOLO-P2 小目标检测头实验。

目标是验证在 V4 稳定基线的基础上，增加 P2/4 小目标检测分支后，是否能提升小裂纹、小缺陷、远距离叶片缺陷的召回能力。

注意：

- 本轮是新结构实验，不修改现有模型。
- 不把新权重复制到 `models/`。
- 不替换 Web/API 默认模型。
- 训练输出只保存在独立 `runs` 目录下。

## 2. 模型结构

新增配置：

```text
configs/yolov8n-p2-wind.yaml
```

来源：

```text
Ultralytics 8.4.60 cfg/models/v8/yolov8-p2.yaml
```

本项目修改：

```text
nc: 4
Detect(P2, P3, P4, P5)
```

结构摘要：

```text
YOLOv8n-p2-wind
layers: 161
parameters: 2,927,088
GFLOPs: 12.4
```

相对普通 YOLOv8n，P2 模型新增了 stride=4 的检测分支，理论上更适合非常小的缺陷目标。

## 3. 训练入口

新增脚本：

```text
scripts/train_yolop2.py
```

脚本能力：

- 从项目内 P2 YAML 构建新结构
- 支持 `--pretrained` 做部分权重初始化
- 支持 `--fraction` 做小样本 smoke
- 默认 `amp=false`，避免离线环境下 Ultralytics AMP 检查联网下载

## 4. Smoke 验证

Smoke 命令使用：

```text
imgsz: 640
epochs: 1
batch: 2
fraction: 0.02
pretrained: models/v04_negative_1024_best.pt
```

验证结论：

- P2 YAML 可以正常解析。
- V4 权重可以部分迁移到 P2 新结构。
- GPU/CUDA 可以正常训练。
- smoke 结果不作为模型效果判断。

权重迁移日志：

```text
Transferred 219/437 items from pretrained weights
```

## 5. 正式训练任务

当前正式任务已启动。

任务名：

```text
yolov8n_p2_1024_b8_from_v04_50e_20260606_120535
```

数据集：

```text
datasets/wind_blade_defect_v05_conservative/data.yaml
```

种子权重：

```text
models/v04_negative_1024_best.pt
```

训练配置：

```text
imgsz: 1024
epochs: 50
batch: 8
workers: 4
optimizer: AdamW
lr0: 0.0005
lrf: 0.05
close_mosaic: 15
patience: 20
amp: false
device: 0
```

输出目录：

```text
runs/detect/runs/wind_fault/yolop2/yolov8n_p2_1024_b8_from_v04_50e_20260606_120535
```

日志：

```text
logs/yolov8n_p2_1024_b8_from_v04_50e_20260606_120535.out.log
logs/yolov8n_p2_1024_b8_from_v04_50e_20260606_120535.err.log
```

当前状态：

```text
finished
```

已确认：

- 50 epochs 已完成。
- best.pt / last.pt 已保存。
- 已完成 V4 baseline vs YOLO-P2 candidate 对比分析。
- 没有写入 `models/`。

## 6. 评估标准

训练完成后必须至少对比：

- V4 negative 基线
- YOLO-P2 best.pt

对比 split：

```text
val
test_clean
background empty samples
```

核心指标：

- Precision
- Recall
- mAP50
- mAP50-95
- crack recall
- corrosion recall
- 背景负样本误检数

判断标准：

- 如果 P2 只提升 Recall 但明显增加背景误检，不应替换 V4。
- 如果 P2 提升 crack 小目标但 corrosion 不变，需要单独记录为小目标方向有效，但不能称为整体替代。
- 如果 P2 同时提升 test_clean mAP50-95、crack recall，且背景误检不恶化，才进入下一轮候选模型评估。

## 7. 注意事项

本轮不是 Blade-Slice，也不是困难样本重加权。

它对应 MSF-YOLO 的结构创新方向：

```text
YOLO-P2 small-object detection head
```

训练完成后，再决定是否继续：

- P2 + Blade-Slice 推理复测
- P2 + corrosion 失败样本再训练
- P2 + Shape-Aware Loss

## 8. 完成后模型分析

分析报告：

```text
runs/detect/runs/wind_fault/yolop2/yolov8n_p2_1024_b8_from_v04_50e_20260606_120535/analysis/analysis.md
```

候选权重：

```text
runs/detect/runs/wind_fault/yolop2/yolov8n_p2_1024_b8_from_v04_50e_20260606_120535/weights/best.pt
```

### val 对比

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V4 negative | 0.959868 | 0.941097 | 0.958265 | 0.809383 |
| YOLO-P2 | 0.949076 | 0.906795 | 0.936949 | 0.786777 |
| P2 - V4 | -0.010792 | -0.034302 | -0.021316 | -0.022605 |

### test_clean 对比

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V4 negative | 0.957699 | 0.979452 | 0.979929 | 0.840205 |
| YOLO-P2 | 0.960331 | 0.944363 | 0.967849 | 0.822908 |
| P2 - V4 | +0.002633 | -0.035089 | -0.012080 | -0.017298 |

### test_clean 分类 mAP50-95

| 类别 | V4 negative | YOLO-P2 | P2 - V4 |
| --- | ---: | ---: | ---: |
| crack | 0.861346 | 0.845635 | -0.015711 |
| hole | 0.894491 | 0.880761 | -0.013730 |
| spalling | 0.972025 | 0.973740 | +0.001715 |
| corrosion | 0.632960 | 0.591494 | -0.041466 |

### 背景负样本误检

| Split | 模型 | 空标注图 | 误检数 | 误检类别 |
| --- | --- | ---: | ---: | --- |
| val | V4 negative | 17 | 0 | {} |
| val | YOLO-P2 | 17 | 1 | hole 1 |
| test_clean | V4 negative | 18 | 1 | spalling 1 |
| test_clean | YOLO-P2 | 18 | 2 | corrosion 1, spalling 1 |

## 9. 本轮结论

本轮 YOLO-P2 不建议替换 V4，也不建议加入当前主模型链路。

主要原因：

- test_clean mAP50-95 从 0.840205 下降到 0.822908。
- test_clean Recall 从 0.979452 下降到 0.944363，漏检风险变大。
- crack 小目标方向没有体现收益，crack mAP50-95 从 0.861346 降到 0.845635。
- corrosion 下降更明显，从 0.632960 降到 0.591494。
- 背景负样本误检从 1 增加到 2，val 也新增 1 个 hole 误检。

保留判断：

- V4 negative 仍然是当前稳定基线。
- YOLO-P2 结构可以保留为实验方向，但本轮训练结果不成立。
- P2 若继续，应先解决初始化和训练策略问题，而不是直接作为候选模型。

下一步建议：

- 不继续基于本轮 P2 权重做发布评估。
- 可尝试 P2 + AMP + 更大 batch 的训练稳定性测试，但前提是先解决离线 AMP 检查问题。
- 如果继续 P2，建议先做更短的对照实验：普通 YOLOv8n 同数据同配置 vs YOLOv8n-P2，而不是直接和强 V4 seed 比。
- 当前更优先做 corrosion / crack 失败样本归因。

## 10. P2 训练策略改进队列

启动日期：2026-06-06

本轮按新的训练策略继续尝试，不覆盖上一轮 P2 结果。

已解决问题：

- 本机已找到 `E:\Desktop\training_bundle_v04_negative\yolo26n.pt`。
- 已复制到项目根目录：

```text
yolo26n.pt
```

用途：

- 让 Ultralytics `amp=true` 检查读取本地权重。
- 避免训练启动时联网下载 `yolo26n.pt`。

AMP / batch 稳定性测试：

| 测试 | 结果 | 备注 |
| --- | --- | --- |
| batch=12, amp=true | 通过 | AMP checks passed，显存约 5.95GB |
| batch=16, amp=true | 通过 | AMP checks passed，显存约 8.11GB |

新队列脚本：

```text
scripts/run_p2_strategy_queue.ps1
```

队列任务：

```text
Stage 1: P2_head_warmup_15e
Stage 2: P2_full_finetune_80e
Stage 3: V4 vs P2 自动分析
```

当前队列 stamp：

```text
20260606_144058
```

状态日志：

```text
logs/p2_strategy_queue_20260606_144058.status.log
```

输出日志：

```text
logs/p2_strategy_queue_20260606_144058.out.log
logs/p2_strategy_queue_20260606_144058.err.log
```

Warmup 任务名：

```text
p2_head_warmup_15e_amp_b16_from_v04_20260606_144058
```

Warmup 配置：

```text
imgsz: 1024
epochs: 15
batch: 16
amp: true
freeze: 10
optimizer: AdamW
lr0: 0.001
lrf: 0.1
close_mosaic: 5
```

Full finetune 任务名：

```text
p2_full_finetune_80e_amp_b16_from_warmup_20260606_144058
```

Full finetune 配置：

```text
imgsz: 1024
epochs: 80
batch: 16
amp: true
freeze: none
optimizer: AdamW
lr0: 0.00025
lrf: 0.05
cos_lr: true
close_mosaic: 20
```

当前状态：

```text
Stage 1/3 warmup running
```

已确认：

- AMP checks passed。
- GPU 正常训练。
- warmup 输出目录已生成。
- 本轮仍不写入 `models/`，不接入主模型链路。
