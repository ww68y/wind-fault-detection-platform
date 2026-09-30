# MSF-YOLO V4 Blade-Slice 评估记录

记录日期：2026-06-06

## 1. 任务目的

本次任务验证 MSF-YOLO 第一阶段中的 Blade-Slice 推理增强是否值得继续推进。

本轮不是训练新模型，也不修改模型权重；只在 V4 稳定基线上做普通推理与切片融合推理的对比评估。

基线模型：

```text
models/v04_negative_1024_best.pt
```

评估数据：

```text
datasets/wind_blade_defect_v05_conservative/images/val
datasets/wind_blade_defect_v05_conservative/images/test_clean
```

输出目录：

```text
runs/wind_fault/msf_blade_slice_v4_eval_20260606_114017
```

评估脚本：

```text
scripts/evaluate_blade_slice.py
```

## 2. 评估配置

```text
imgsz: 1024
tile_size: 640
overlap: 0.25
conf: 0.45
eval_iou: 0.5
nms_iou: 0.5
device: 0
```

对比模式：

- normal：V4 普通全图推理
- blade_slice_fused：V4 全图推理 + 切片推理 + NMS 融合

## 3. 总体结果

| Split | Mode | Recall | Precision | Missed | False positives | Predictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | normal | 0.868624 | 0.899200 | 85 | 63 | 625 |
| val | blade_slice_fused | 0.867079 | 0.880691 | 86 | 76 | 637 |
| test_clean | normal | 0.888325 | 0.903614 | 66 | 56 | 581 |
| test_clean | blade_slice_fused | 0.890017 | 0.872305 | 65 | 77 | 603 |

## 4. 分类召回对比

### val

| Class | normal recall | blade_slice_fused recall | 变化 |
| --- | ---: | ---: | ---: |
| corrosion | 0.438596 | 0.438596 | 0 |
| crack | 0.918552 | 0.918552 | 0 |
| hole | 0.957944 | 0.953271 | -0.004673 |
| spalling | 0.832258 | 0.832258 | 0 |

### test_clean

| Class | normal recall | blade_slice_fused recall | 变化 |
| --- | ---: | ---: | ---: |
| corrosion | 0.542373 | 0.542373 | 0 |
| crack | 0.937198 | 0.942029 | +0.004831 |
| hole | 1.000000 | 1.000000 | 0 |
| spalling | 0.815603 | 0.815603 | 0 |

## 5. 误检变化

val：

| Mode | False positives | 主要误检类别 |
| --- | ---: | --- |
| normal | 63 | 需查看 normal error report |
| blade_slice_fused | 76 | spalling 39, crack 17, corrosion 13, hole 7 |

test_clean：

| Mode | False positives | 主要误检类别 |
| --- | ---: | --- |
| normal | 56 | spalling 37, crack 7, hole 7, corrosion 5 |
| blade_slice_fused | 77 | spalling 37, crack 27, hole 8, corrosion 5 |

关键观察：

- Blade-Slice 融合在 test_clean 上只多召回 1 个 crack。
- test_clean 的 false positives 从 56 增加到 77，主要新增在 crack 类。
- val 上没有召回收益，反而 missed 从 85 增到 86，false positives 从 63 增到 76。
- corrosion 没有受益，两个 split 的 corrosion recall 均无变化。

## 6. 结论

默认参数下的 V4 + Blade-Slice 融合暂不适合作为正式推理方案。

原因：

- 召回收益极小，只在 test_clean 上多召回 1 个 crack。
- Precision 明显下降。
- 误检增加明显，尤其 test_clean 的 crack 误检从 7 增到 27。
- corrosion 没有改善，说明当前 corrosion 瓶颈不是简单切片放大能解决的。

本轮不建议：

- 不建议把 Blade-Slice 融合接入默认 Web/API 推理。
- 不建议基于这组默认参数直接做发布。
- 不建议据此启动新的训练模型。

## 7. 下一步

Blade-Slice 方向仍可保留，但需要先缩小验证范围：

- 只在高分辨率、远距离、小目标图像上单独评估。
- 单独评估 tile_size 512 / 768 / 896，而不是直接全量融合。
- 增加切片结果的置信度门控，例如切片分支 conf 高于全图分支。
- 对切片新增的 false positive 做人工复核，重点看 crack 和 spalling。

模型方向上，当前更值得继续做：

- corrosion 失败样本归因
- crack 小目标专项样本整理
- YOLO-P2 或小目标检测头实验
- Shape-Aware Loss 前的 mask 候选复核

本次 Blade-Slice 评估记录到此为止。V4 仍然保留为当前稳定基线。
