# MSF-YOLO V6 全创新集合模型实验记录

记录日期：2026-06-06

## 1. 实验定位

本轮目标是构建一个“集合全部创新点”的实验模型，用于证明 MSF-YOLO 方向已经具备完整训练与评估链路。

注意：

- 本模型是实验模型，不替代当前 V4 稳定基线。
- 不复制到 `models/`。
- 不接入 Web/API 默认模型。
- 不用于说明 V4 已经集成这些创新点。
- 效果暂时不作为启动前提，后续只作为消融和论文记录使用。

## 2. 已接入的创新点

### 2.1 训练阶段真正进入权重的部分

| 创新点 | 接入方式 | 是否进入 `.pt` 权重训练 |
| --- | --- | --- |
| YOLO-P2 小目标检测头 | 使用 `configs/yolov8n-p2-wind.yaml`，Detect(P2,P3,P4,P5) | 是 |
| 小目标增强 | 使用 `imgsz=1024`、`multi_scale=0.10`、mosaic/copy-paste/scale 增强 | 是 |
| 困难样本重加权 | 自定义训练采样器，对 crack、corrosion、小目标样本提高采样权重 | 是 |
| 类别重加权 | 使用 `cls_pw=0.35` 触发类别权重 BCE | 是 |
| Shape-Aware Loss | 自定义 bbox/DFL loss，对小框、细长框提高定位损失权重 | 是 |

### 2.2 推理/评估阶段接入的部分

| 创新点 | 接入方式 | 是否写进 `.pt` 权重 |
| --- | --- | --- |
| Blade-Slice | 训练完成后自动做 normal vs blade_slice_fused 复测 | 否 |

Blade-Slice 是推理策略，不是模型结构本身。因此它不能说“写入了权重”，只能说“作为 MSF-YOLO 推理增强模块参与评估”。

## 3. 新增文件

训练入口：

```text
scripts/train_msf_yolo_full.py
```

队列入口：

```text
scripts/run_msf_full_innovation_queue.ps1
```

输出目录：

```text
runs/detect/runs/wind_fault/msf_yolo_full_innovation/
```

## 4. 训练策略

本轮采用两阶段训练，避免直接全量扰动：

### Stage 1：warmup

```text
epochs: 10
batch: 12
imgsz: 1024
amp: true
freeze: 10
optimizer: AdamW
lr0: 0.0008
lrf: 0.1
close_mosaic: 5
```

作用：

- 从 V4 稳定基线迁移权重。
- 冻结部分 backbone。
- 先让新增 P2 head、Shape-Aware loss 和重加权采样链路稳定起来。

### Stage 2：full finetune

```text
epochs: 60
batch: 12
imgsz: 1024
amp: true
freeze: none
optimizer: AdamW
lr0: 0.00018
lrf: 0.05
cos_lr: true
close_mosaic: 20
patience: 25
```

作用：

- 从 warmup best.pt 继续。
- 全模型低学习率微调。
- 生成最终全创新集合版候选权重。

## 5. 评估策略

训练完成后自动执行：

1. V4 baseline vs V6 candidate 标准指标对比。
2. `val` / `test_clean` 双 split 评估。
3. 背景空标签样本误检统计。
4. Blade-Slice normal vs fused 推理复测。

核心判断指标：

- Precision
- Recall
- mAP50
- mAP50-95
- 背景负样本误检
- crack / corrosion 分类别表现

## 6. Smoke 验证

已完成 1 epoch / 2% 数据 smoke。

验证通过：

- P2 模型结构可构建。
- V4 权重可迁移到 P2 结构。
- 自定义 hard-sample sampler 已生效。
- `cls_pw` 类别权重已生效。
- Shape-Aware bbox loss 可参与训练。
- AMP 检查通过。
- CUDA/GPU 训练可运行。

Smoke 仅证明链路可跑，不作为效果判断。

## 7. 对外表述建议

可以说：

> 在 V4 稳定基线基础上，项目进一步构建了 MSF-YOLO 全创新集合实验模型，综合接入 YOLO-P2 小目标检测头、小目标增强、困难样本重加权、Shape-Aware Loss，并在训练后进行 Blade-Slice 推理增强评估。

不建议说：

> V4 已经集成 YOLO-P2、Blade-Slice、Shape-Aware Loss 和困难样本重加权。

更准确的区分是：

- V4 是当前稳定部署基线。
- V6/MSF-Full 是集合创新点的实验模型。
- Blade-Slice 属于推理增强，不属于权重结构。

