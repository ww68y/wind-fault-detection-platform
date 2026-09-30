# MSF-YOLO 小目标子集评估实验记录

记录日期：2026-06-09

## 1. 实验目的

整体 `val/test_clean` 中存在较多中大目标，普通 YOLOv8n 在该评测口径下已经表现很强。为验证 MSF-YOLO 的 P2 浅层检测分支、Shape-Aware Loss 和困难样本机制是否真正服务于小目标缺陷，需要补充小目标子集评估。

## 2. 小目标定义

采用 YOLO 标注中的归一化框面积：

```text
area = bbox_width * bbox_height
```

默认评估两档阈值：

| 子集 | 判定条件 | 含义 |
| --- | --- | --- |
| small-1pct | area <= 0.01 | 框面积不超过图像面积 1% |
| tiny-0.5pct | area <= 0.005 | 框面积不超过图像面积 0.5% |

## 3. 评估口径

脚本只将小框作为 GT 参与 AP/Recall 计算。同一张图中的大框作为 ignore 区域处理：如果预测框只命中大目标，不计为小目标误检。这可以避免“同图存在大缺陷”导致小目标评估被不公平惩罚。

## 4. 新增脚本

```text
scripts/evaluate_small_object_subset.py
scripts/run_small_object_subset_eval.ps1
scripts/run_small_eval_after_c0.ps1
```

手动运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_small_object_subset_eval.ps1 -IncludeC0
```

等待 C0 完成后自动运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_small_eval_after_c0.ps1
```

## 5. 输出文件

默认输出到：

```text
paper_assets_msf_yolo/small_object_eval_<stamp>/
```

主要文件：

```text
small_object_eval.md
small_object_eval.csv
small_object_eval.json
subset_val_area_le_0p01.csv
subset_test_clean_area_le_0p01.csv
subset_val_area_le_0p005.csv
subset_test_clean_area_le_0p005.csv
```

## 6. 论文表述边界

推荐表述：

> 考虑到整体测试集中存在较多中大尺度缺陷，本文进一步构建小目标子集评估，仅统计归一化面积小于 1% 和 0.5% 的缺陷框，以检验所提多尺度浅层检测结构在微小缺陷场景下的有效性。

避免表述：

> 为了让 MSF-YOLO 指标更好，本文重新筛选了更有利的测试集。

## 7. 本轮评估结果

输出目录：

```text
paper_assets_msf_yolo/small_object_eval_20260609_134953/
```

整体说明：该评估只统计小目标 GT，大目标作为 ignore 区域，不把命中大目标的检测框计为小目标误检。

### 7.1 area <= 0.01

| Split | 模型 | Precision@0.5/conf | Recall@0.5/conf | AP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| val | V6 | 0.9452 | 0.8922 | 0.7188 | 0.5249 |
| val | A0 | 0.9765 | 0.8966 | 0.7601 | 0.5283 |
| val | B0 | 0.9673 | 0.8922 | 0.7377 | 0.5327 |
| val | C0 | 0.9670 | 0.8836 | 0.7414 | 0.5181 |
| test_clean | V6 | 0.9650 | 0.9698 | 0.9712 | 0.7720 |
| test_clean | A0 | 0.9949 | 0.9849 | 0.9911 | 0.7691 |
| test_clean | B0 | 0.9898 | 0.9749 | 0.9823 | 0.7536 |
| test_clean | C0 | 0.9898 | 0.9799 | 0.9962 | 0.7496 |

结论：在 `test_clean` 的 1% 小目标子集上，V6 的 mAP50-95 最高，分别高于 A0/B0/C0 约 0.0028/0.0184/0.0223。但 V6 的 Precision、Recall 和 AP50 不占优，因此只能表述为“高 IoU 综合定位指标更优”，不能写成全面领先。

### 7.2 area <= 0.005

| Split | 模型 | Precision@0.5/conf | Recall@0.5/conf | AP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| val | V6 | 0.9603 | 0.8683 | 0.6495 | 0.4768 |
| val | A0 | 1.0000 | 0.8743 | 0.6873 | 0.5131 |
| val | B0 | 0.9864 | 0.8683 | 0.6459 | 0.4865 |
| val | C0 | 0.9863 | 0.8623 | 0.6629 | 0.4611 |
| test_clean | V6 | 0.9917 | 0.9675 | 0.9907 | 0.7443 |
| test_clean | A0 | 1.0000 | 0.9837 | 0.9986 | 0.7607 |
| test_clean | B0 | 1.0000 | 0.9675 | 0.9860 | 0.7329 |
| test_clean | C0 | 1.0000 | 0.9675 | 1.0000 | 0.6973 |

结论：在 `test_clean` 的 0.5% 极小目标子集上，V6 高于 B0/C0，但低于 A0。该结果支持“V6 相对低分辨率 YOLO 基线具有更好的极小目标定位稳定性”，但不支持“V6 超过所有普通 YOLO 基线”。

## 8. 可写入论文的保守结论

可写：

> 在整体测试集上，普通 YOLOv8n 基线已表现出较强竞争力；进一步的小目标子集评估显示，MSF-YOLO 在 `test_clean` 的面积小于 1% 缺陷框上取得最高 mAP50-95，并在面积小于 0.5% 的极小目标子集上优于 416/320 输入的低分辨率 YOLO 基线，说明 P2 高分辨率检测分支与形态感知设计对小尺度缺陷定位具有一定收益。

不建议写：

> MSF-YOLO 在所有小目标指标上全面优于 A0/B0/C0。
