# 风机叶片缺陷检测平台

这是一个用于后续迭代的风机叶片缺陷检测工程骨架。第一版重点不是追求一次性做完，而是先把数据整理、YOLO 训练、推理、报告输出和 Web API 的基础链路搭起来。

当前项目已经可以作为一个本地检测与展示原型使用：支持风机叶片图片、批量图片和视频检测，提供 FastAPI 接口与网页工作台，保存检测结果和历史记录，并支持漏检样本整理、DTU 1024 高分辨率切片数据接入和后续 MSF-YOLO 算法迭代。

当前默认缺陷类别为 `crack`（裂纹）、`hole`（孔洞）、`spalling`（剥落）和 `corrosion`（腐蚀）。训练数据、模型权重和运行产物默认不提交到 Git 仓库，便于控制仓库体积；具体获取和放置方式见各目录说明。

## 已实现功能

- 基于 Ultralytics YOLO 的训练、验证和推理流程
- 单张图片、批量图片和视频检测
- FastAPI 后端、浏览器检测工作台和 `/health`、`/detect` 接口
- JSON、Markdown 和网页形式的检测报告
- 历史任务、关键帧结果和困难/漏检样本分析
- COCO、VOC、LabelMe 标注转换与高分辨率图片切片
- DTU 1024 数据接入，以及 MSF-YOLO（小目标、切片、困难样本和形态感知方向）实验记录

## 参考项目分工

- YOLO 训练流程参考：<https://github.com/memari-majid/Wind-Turbine-Blade-Defect-Detection-with-YOLO-Models>
- 风机叶片表面缺陷数据集参考：<https://github.com/zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset>
- DTU 无人机巡检标注/1024 切片参考：<https://github.com/imadgohar/DTU-annotations>

## 当前工程结构

```text
wind-fault-detection-platform/
├── configs/default.yaml                 # 默认训练/推理配置
├── datasets/README.md                   # 数据集放置和整理说明
├── scripts/
│   ├── train_yolo.py                    # YOLO 训练入口
│   └── predict_yolo.py                  # YOLO 推理入口
├── src/wind_fault/
│   ├── api/app.py                       # FastAPI 检测接口
│   ├── data/convert_coco_json.py        # COCO/DTU JSON 转 YOLO
│   ├── data/convert_voc_xml.py          # VOC XML 转 YOLO
│   ├── data/dataset.py                  # 数据集通用工具
│   ├── data/slice_images.py             # 高分辨率图片切片
│   ├── model/yolo_runner.py             # ultralytics 调用封装
│   └── reports/prediction_report.py     # 推理结果报告
├── tests/test_dataset_utils.py
├── pyproject.toml
└── requirements.txt
```

## 快速开始

```powershell
cd wind-fault-detection-platform
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

准备 YOLO 数据集后，目录建议整理成：

```text
datasets/wind_blade_defect/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

训练：

```powershell
python scripts/train_yolo.py --data datasets/wind_blade_defect/data.yaml --model yolov8n.pt --epochs 80 --imgsz 640 --batch 16
```

如果已经安装 CUDA 版 PyTorch，可以指定 GPU：

```powershell
python scripts/train_yolo.py --data datasets/wind_blade_defect/data.yaml --model yolov8n.pt --epochs 80 --imgsz 640 --batch 8 --device 0
```

通宵训练可以用脚本启动。脚本会先检查 GPU，优先使用 `640 + batch 4`，如果显存不足会降到 `512 + batch 2` 重试：

```powershell
.\scripts\start_overnight_training.ps1
```

推理：

```powershell
python scripts/predict_yolo.py --model runs/wind_fault/baseline/weights/best.pt --source test.jpg --conf 0.3
```

漏检样本分析：

```powershell
python scripts/analyze_missed_samples.py `
  --data datasets/wind_blade_defect/data.yaml `
  --images datasets/wind_blade_defect/images/test `
  --labels datasets/wind_blade_defect/labels/test `
  --report runs/wind_fault/test_batch_80e/reports/detection_report.json `
  --output runs/wind_fault/test_batch_80e/reports/missed_samples
```

启动 Web API：

```powershell
$env:WIND_MODEL_PATH="runs/wind_fault/baseline/weights/best.pt"
uvicorn wind_fault.api.app:app --reload
```

检测接口：

```text
GET  /health
POST /detect
```

`POST /detect` 使用 `multipart/form-data` 上传 `file` 字段。

## 推荐迭代路线

1. 先下载风机叶片表面缺陷数据集，整理成 YOLO 格式。
2. 用 `yolov8n.pt` 跑通第一版训练和推理。
3. 加入 DTU 的 1024 切片数据，提高细小裂纹和长条形缺陷检出能力。
4. 用 Web API 保存检测记录，生成故障报告。
5. 再逐步加前端页面、用户管理、客户端下发和运维闭环。
