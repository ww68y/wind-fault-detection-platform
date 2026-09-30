# 风机叶片缺陷检测工程迁移与 Codex 交接说明

生成目的：把当前电脑上的工程完整迁移到另一台电脑，并让另一台电脑上的 Codex 能快速理解项目现状、算法目标和下一步工作。

## 1. 当前项目定位

本项目是风机叶片表面缺陷检测平台，当前基础算法为 Ultralytics YOLO 系列目标检测模型，检测类别统一为：

- crack：裂纹 / 划痕 / 细裂纹类
- hole：孔洞
- spalling：剥落 / 表面损伤
- corrosion：腐蚀 / 侵蚀

当前正在推进的算法创新方向为：

**MSF-YOLO：面向风机叶片缺陷检测的形态引导小目标融合 YOLO 算法**

核心模块规划：

- YOLO-P2 小目标检测头
- Blade-Slice 高分辨率重叠切片融合推理
- Shape-Aware Loss 形态感知损失
- Difficulty-Aware Hard Cases Reweighting 困难样本重加权训练

## 2. V5 模型定位

V5 不是最终算法名，而是 MSF-YOLO 的阶段性训练基座。

V5 的主要目标：

- 提升真实巡检图片泛化能力
- 提升 corrosion / erosion 类稳定性
- 提升小裂纹、小损伤、scratch / craze / hide_craze 等样本召回能力
- 通过 Dirt 等负样本减少误检
- 保留旧 val 和 test_clean，方便与旧模型公平对比

当前可将 V5 表述为：

**v05_real_scene_generalization：真实场景泛化增强模型**

后续实验建议：

```text
旧稳定模型
vs v04_negative
vs v05_real_scene
vs v05 + Blade-Slice
vs MSF-YOLO 完整版
```

## 3. 当前电脑上已经完成的 MSF-YOLO 准备工作

已经新增脚本：

```text
scripts/prepare_shape_mask_candidates.py
```

用途：

- 从 YOLO 框标注中筛选高价值缺陷样本
- 生成缺陷裁剪图
- 生成矩形初始 mask
- 生成 GrabCut 伪 mask
- 生成 overlay 复核图
- 生成 manifest.csv 复核清单

已经生成候选集：

```text
datasets/msf_shape_mask_candidates
```

数量：

```text
crack: 60
corrosion: 60
hole: 60
spalling: 60
total: 240
```

复核入口：

```text
datasets/msf_shape_mask_candidates/overlays/crack
datasets/msf_shape_mask_candidates/overlays/corrosion
datasets/msf_shape_mask_candidates/overlays/hole
datasets/msf_shape_mask_candidates/overlays/spalling
```

复核清单：

```text
datasets/msf_shape_mask_candidates/manifest.csv
```

review_status 建议取值：

```text
accepted：伪 mask 基本可用
edited：缺陷存在，但 mask 或框需要后续修改
rejected：类别不对、缺陷不明显、标签噪声或不适合训练
```

复核原则：

- 看不清、不像、不确定的样本不要进入 Shape-Aware Loss 训练
- crack 和 corrosion 优先复核
- 形态感知数据宁愿少一点，也要准确

## 4. 迁移时必须保留的目录和文件

建议完整复制整个工程目录：

```text
wind-fault-detection-platform
```

重点必须保留：

```text
configs/
scripts/
src/
tests/
datasets/
hard_cases/
runs/
release/
logs/
training_bundle_v04_negative/
training_bundle_v05_balanced/
*.txt
*.md
requirements.txt
pyproject.toml
yolov8n.pt
```

尤其不要漏掉：

```text
datasets/wind_blade_defect_v05_conservative
datasets/msf_shape_mask_candidates
training_bundle_v05_balanced
hard_cases
```

可以不复制，或复制后重新生成：

```text
.venv/
__pycache__/
.pytest_cache/
*.pyc
```

## 5. 推荐迁移方式

### 方式 A：移动硬盘或 U 盘直接复制

适合数据集较大时使用。

复制整个目录：

```text
C:\Users\yys\Desktop\风机\wind-fault-detection-platform
```

到另一台电脑，例如：

```text
D:\wind-fault-detection-platform
```

### 方式 B：PowerShell 压缩包

在当前电脑的 `C:\Users\yys\Desktop\风机` 下执行：

```powershell
Compress-Archive -Path .\wind-fault-detection-platform -DestinationPath .\wind-fault-detection-platform_transfer.zip -Force
```

如果压缩包太大，优先用方式 A。

### 方式 C：robocopy 镜像复制

适合复制到移动硬盘，例如目标盘为 `E:\wind-fault-detection-platform`：

```powershell
robocopy .\wind-fault-detection-platform E:\wind-fault-detection-platform /E /XD .venv __pycache__ .pytest_cache /XF *.pyc
```

注意：不要把目标设错，避免覆盖别的项目。

## 6. 新电脑环境初始化

进入项目目录：

```powershell
cd D:\wind-fault-detection-platform
```

创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

验证项目：

```powershell
python -m pytest
```

如果要用 GPU 训练，需要确认 CUDA 版 PyTorch 可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 7. 另一台电脑上的 Codex 启动提示词

把下面这段直接发给另一台电脑上的 Codex：

```text
请先阅读项目根目录下的《项目迁移与Codex交接说明.md》，然后继续接手这个风机叶片缺陷检测项目。

当前项目基础模型是 YOLO 风机叶片四类缺陷检测，类别为 crack、hole、spalling、corrosion。

当前算法创新方向是 MSF-YOLO：面向风机叶片缺陷检测的形态引导小目标融合 YOLO 算法。

V5 是当前真实场景泛化增强训练版本，不是最终算法名。V5 的目标是提升真实图片泛化、corrosion 稳定性、小裂纹召回，并通过 Dirt 等负样本降低误检。

当前已经准备了像素级形态感知的候选集：
datasets/msf_shape_mask_candidates

请先确认：
1. V5 训练是否完成；
2. V5 的 best.pt、results.csv、results.png、confusion_matrix 等是否已保存；
3. 用旧 val 和 test_clean 对旧模型、v04_negative、v05 做公平评估；
4. 继续复核 datasets/msf_shape_mask_candidates/overlays 下的样本，把 manifest.csv 中 review_status 标为 accepted、edited 或 rejected；
5. 后续在 V5 基础上逐步实现 Blade-Slice、YOLO-P2 和 Shape-Aware Loss。

不要把 V5 当作算法名，算法名统一使用 MSF-YOLO。
```

## 8. V5 训练完成后的优先工作

1. 封版 V5 产物：

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
PR_curve / F1_curve
args.yaml
data.yaml
```

2. 做公平评估：

```text
旧稳定模型 vs v04_negative vs v05
```

重点指标：

```text
Precision
Recall
mAP50
mAP50-95
crack recall
corrosion recall
false positive count
hard_cases 漏检数量
Dirt 负样本误检数量
```

3. 继续 MSF-YOLO 路线：

```text
V5
→ V5 + Blade-Slice
→ V5 + YOLO-P2
→ V5 + 框级 Shape-Aware Loss
→ 后续 mask 级 Shape-Aware Loss
```

## 9. 重要注意事项

- 不要直接用未复核的伪 mask 训练像素级形态损失。
- 不确定、不清楚、不像缺陷的 overlay 样本标为 rejected。
- 当前数据足够支撑 MSF-YOLO v1：Blade-Slice、YOLO-P2、困难样本重加权、框级 Shape-Aware Loss。
- 如果要做更高级的 mask 级 Shape-Aware Loss，需要先复核并清洗 `datasets/msf_shape_mask_candidates`。
- 另一台电脑如果已经在训练 V5，不要覆盖它的 `runs/` 训练输出，先备份再合并。

