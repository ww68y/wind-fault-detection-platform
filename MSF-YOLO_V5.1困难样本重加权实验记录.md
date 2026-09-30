# MSF-YOLO V5.1 困难样本重加权实验记录

记录日期：2026-06-06

## 1. 实验目的

本轮实验属于 MSF-YOLO 第一阶段探索，目标不是替换当前线上/项目默认模型，而是验证在 V4 稳定基线之上，使用困难样本重加权是否能带来稳定收益。

重点观察方向：

- corrosion 类召回和 mAP50-95 是否提升
- crack 类是否能继续保持或提升
- val / test_clean 是否同时稳定
- 背景负样本误检是否恶化
- 是否值得把新模型加入项目模型目录

当前结论：本轮 V5.1 重加权实验暂不替换 V4，也不加入现有主模型链路，避免模型污染。

## 2. 基线模型

V4 negative 1024 是当前最稳的 MSF-YOLO 第一阶段基线。

项目内种子模型：

```text
models/v04_negative_1024_best.pt
```

原始来源：

```text
E:\Desktop\training_bundle_v04_negative\runs\detect\runs\wind_fault\v04_negative_1024_b2_80e-3\weights\best.pt
```

模型定位：

- 当前作为后续训练种子之一保留
- 不作为 Web/API 默认模型强制替换
- 后续 Blade-Slice 推理评估优先基于该模型做

## 3. 数据与训练设置

V5.1 使用 no-leak 数据集，训练集进行困难样本重加权，val 和 test_clean 保持不变，用于与 V4 公平对比。

数据配置：

```text
E:\Desktop\training_bundle_v05_balanced\dataset_v5_1_no_leak\data.yaml
```

基础训练配置：

```text
imgsz: 1024
epochs: 50
batch: 24
workers: 4
optimizer: AdamW
lr0: 0.0005
lrf: 0.05
close_mosaic: 15
seed model: V4 negative 1024 best.pt
```

训练任务：

| 实验名 | 重加权策略 | 说明 |
| --- | --- | --- |
| c4_crack2 | corrosion x4 + crack x2 | 最激进的 corrosion 加权，同时增强 crack |
| c3_crack2 | corrosion x3 + crack x2 | 中等 corrosion 加权，同时增强 crack |
| c4_crack1 | corrosion x4 + crack x1 | 强 corrosion 加权，弱化 crack 加权 |

对应权重位置：

```text
E:\Desktop\training_bundle_v05_balanced\runs\detect\runs\wind_fault\v5_1_no_leak_1024_b24_w4_nocache_from_v04_corrosion_x4_crack_x2_50e\weights\best.pt
E:\Desktop\training_bundle_v05_balanced\runs\detect\runs\wind_fault\v5_1_no_leak_1024_b24_w4_nocache_from_v04_corrosion_x3_crack_x2_50e\weights\best.pt
E:\Desktop\training_bundle_v05_balanced\runs\detect\runs\wind_fault\v5_1_no_leak_1024_b24_w4_nocache_from_v04_corrosion_x4_crack_x1_50e\weights\best.pt
```

## 4. val 对比结果

| 模型 | Precision | Recall | mAP50 | mAP50-95 | crack | hole | spalling | corrosion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 negative | 0.959837 | 0.939978 | 0.958147 | 0.809191 | 0.8450 | 0.8455 | 0.9677 | 0.5786 |
| V5.1 c4_crack2 | 0.942658 | 0.939314 | 0.955072 | 0.817708 | 0.8797 | 0.8597 | 0.9733 | 0.5583 |
| V5.1 c3_crack2 | 0.953004 | 0.906663 | 0.951125 | 0.810248 | 0.8686 | 0.8589 | 0.9725 | 0.5411 |
| V5.1 c4_crack1 | 0.935648 | 0.920396 | 0.953369 | 0.808548 | 0.8617 | 0.8592 | 0.9701 | 0.5432 |

val 观察：

- c4_crack2 的总 mAP50-95 比 V4 高，但 Precision 明显下降。
- 三个 V5.1 方案的 corrosion mAP50-95 都低于 V4。
- c3_crack2 的 Precision 接近 V4，但 Recall 明显下降。
- c4_crack1 没有体现出对 corrosion 的有效改善，整体也不优于 V4。

## 5. test_clean 对比结果

| 模型 | Precision | Recall | mAP50 | mAP50-95 | crack | hole | spalling | corrosion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 negative | 0.957717 | 0.979439 | 0.979917 | 0.840210 | 0.8613 | 0.8945 | 0.9720 | 0.6330 |
| V5.1 c4_crack2 | 0.928644 | 0.969080 | 0.959200 | 0.830396 | 0.8853 | 0.8946 | 0.9814 | 0.5603 |
| V5.1 c3_crack2 | 0.944713 | 0.961271 | 0.976619 | 0.840086 | 0.8749 | 0.8925 | 0.9795 | 0.6134 |
| V5.1 c4_crack1 | 0.950160 | 0.951758 | 0.977855 | 0.838515 | 0.8702 | 0.8870 | 0.9826 | 0.6142 |

test_clean 观察：

- V4 的 Recall、mAP50、mAP50-95 仍是最稳。
- c3_crack2 的总 mAP50-95 最接近 V4，但 Recall 下降，corrosion 仍低于 V4。
- c4_crack2 虽然 crack 提升明显，但 Precision、mAP50、mAP50-95 和 corrosion 都下降，不适合替换。
- c4_crack1 也没有超过 V4，且 Recall 下降更明显。

## 6. 背景负样本误检方面

误检统计口径：conf=0.45，在空标注背景图上统计预测框数量。

| 模型 | val empty 17 张 | test_clean empty 18 张 | 主要误检类别 |
| --- | ---: | ---: | --- |
| V4 negative | 0 | 1 | test_clean: spalling 1 |
| V5 balanced | 0 | 1 | test_clean: spalling 1 |
| V5.1 c4_crack2 | 2 | 1 | val: hole 1, corrosion 1; test_clean: spalling 1 |
| V5.1 c3_crack2 | 1 | 1 | val: corrosion 1; test_clean: spalling 1 |
| V5.1 c4_crack1 | 0 | 1 | test_clean: spalling 1 |

背景负样本观察：

- V4 在 val empty 上没有误检，是目前背景抑制最稳定的模型之一。
- c4_crack2 在 val empty 上新增 2 个误检，说明激进重加权会带来一定误检风险。
- c3_crack2 新增 1 个 corrosion 误检，说明 corrosion 重加权需要控制强度。
- c4_crack1 的背景误检没有恶化，但主指标没有超过 V4。

## 7. 结论

本轮 corrosion / crack 重加权实验已经完成，两个用户关心的组合都已跑完：

- corrosion x3 + crack x2：已跑完，效果最接近 V4，但没有形成稳定提升。
- corrosion x4 + crack x1：已跑完，背景误检尚可，但主指标和 Recall 不如 V4。

综合 val、test_clean 和背景负样本误检结果，本轮不建议把 V5.1 三个模型加入当前已有模型目录，也不建议替换 V4。

保留判断：

- V4 negative 继续作为 MSF-YOLO 第一阶段基线。
- V5.1 c3_crack2 可作为“重加权方向参考”，但不是可发布模型。
- c4_crack2 证明过度重加权会引入误检和 Precision 下降。
- c4_crack1 证明单纯提高 corrosion 权重不能解决 corrosion 泛化问题。

## 8. 下一步建议

优先级 1：基于 V4 做 Blade-Slice 推理评估。

理由：

- Blade-Slice 是推理增强，不会污染当前模型权重。
- V4 是当前最稳基线，适合作为第一阶段 MSF-YOLO 推理增强入口。
- 先验证切片对小裂纹、小缺陷和远距离叶片缺陷是否有效，再决定是否需要训练侧结构改动。

优先级 2：整理 corrosion 失败样本。

理由：

- 本轮重加权没有真正提升 corrosion。
- corrosion 问题更像数据质量、标注一致性、场景差异和外观边界不清导致的问题。
- 下一步应先做误检/漏检样本分组，而不是继续盲目增加 corrosion oversampling。

优先级 3：Shape-Aware Loss 前先复核 mask 候选集。

当前候选集：

```text
datasets/msf_shape_mask_candidates
```

注意：

- 当前 manifest 中仍有旧电脑路径痕迹，需要后续修正。
- 所有 mask 候选应先人工复核，不建议直接进入训练。

本轮实验记录到此为止。V5.1 结果用于算法路线判断，不进入当前主模型链路。
