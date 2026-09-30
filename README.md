# 风机叶片缺陷检测平台

[English README](README_EN.md)

一个面向风电叶片巡检场景的计算机视觉工程。项目以 Ultralytics YOLO 为基础，围绕“数据整理、模型训练、图片/视频推理、结果报告和 Web API”搭建了一套可以持续迭代的本地检测平台。

这个项目的重点不是只提供一个单次训练脚本，而是把模型能力接入到相对完整的业务链路中：用户上传图片或视频，系统执行检测，保存原始文件和推理结果，生成 JSON/Markdown/HTML 报告，并为后续的漏检分析、维修建议和模型迭代保留结构化数据。

## 项目概览

风机叶片在长期运行过程中可能出现裂纹、孔洞、剥落和腐蚀等表面缺陷。人工巡检成本较高，而且细小缺陷、远距离目标和复杂背景容易造成漏检。这个项目尝试使用目标检测模型辅助巡检人员完成初筛，并通过报告和历史记录支持后续复核。

当前默认支持以下四类缺陷：

| 类别 | 含义 | 典型问题 |
| --- | --- | --- |
| `crack` | 裂纹 | 叶片表面细长裂纹、结构性裂纹 |
| `hole` | 孔洞 | 局部穿孔或明显空洞 |
| `spalling` | 剥落 | 表面材料剥落、破损 |
| `corrosion` | 腐蚀 | 腐蚀斑、腐蚀区域 |

项目当前定位为本地检测和展示原型。训练数据、模型权重和运行产物默认不提交到仓库，以控制仓库体积并避免上传大文件；具体数据准备方式见 `datasets/README.md`。

## 核心功能

### 检测与推理

- 支持单张图片检测；
- 支持批量图片检测；
- 支持视频检测，并生成检测结果视频和关键帧；
- 支持摄像头/视频流实时检测脚本；
- 支持通过置信度阈值控制检测结果；
- 支持 CPU 或 CUDA 设备。

### Web API 与工作台

- 基于 FastAPI 提供 HTTP API；
- 内置浏览器检测工作台；
- 自动保存上传文件、渲染结果、报告和历史记录；
- 支持图片、批量图片和视频三种检测流程；
- 支持健康检查、历史记录、报告详情和评估页面；
- 检测结果可通过浏览器直接查看。

### 报告与运维闭环

- 输出结构化 JSON 报告；
- 输出便于阅读的 Markdown 报告；
- 生成 HTML 检测结果页；
- 汇总缺陷类别、置信度、边框和风险等级；
- 支持关键帧和检测结果图片；
- 支持漏检、误检、低置信度样本整理；
- 支持生成故障分析、维修单和派工记录。

### 数据与实验

- 支持 COCO、VOC XML、LabelMe 等标注格式转换；
- 支持高分辨率图片切片；
- 支持 DTU 1024 切片数据接入；
- 包含多个 YOLO 基线和 MSF-YOLO 方向的实验脚本；
- 支持小目标、困难样本、切片检测和形态感知等迭代方向；
- 使用 pytest 覆盖数据处理、报告、维修单和检测闭环等模块。

## 系统流程

```text
图片/视频输入
      │
      ▼
上传与任务创建
      │
      ▼
YOLO 模型推理
      │
      ├── 检测框、类别、置信度
      ├── 结果图片或结果视频
      ├── 关键帧
      └── 原始推理明细
      │
      ▼
报告生成与风险判断
      │
      ├── JSON
      ├── Markdown
      ├── HTML 页面
      └── 历史记录/维修建议
      │
      ▼
困难样本整理与下一轮训练
```

## 技术栈

- Python 3.10+
- Ultralytics YOLO
- FastAPI + Uvicorn
- OpenCV
- Pillow、NumPy、Pandas
- PyYAML
- pytest

## 工程结构

```text
wind-fault-detection-platform/
├── configs/
│   ├── default.yaml                  # 默认训练、推理和切片配置
│   └── *.yaml                        # 实验配置
├── datasets/
│   ├── README.md                     # 数据准备说明
│   └── */data.yaml                   # 各数据版本的数据配置
├── hard_cases/                       # 困难样本、漏检和误检样本
├── models/                           # 模型说明；大模型权重不提交
├── paper_assets_msf_yolo/            # 研究材料和论文文档
├── paper_latex/                      # 论文草稿及相关资料
├── release/v0.1_baseline/            # 可复现实验/发布版本说明
├── scripts/
│   ├── train_yolo.py                 # YOLO 训练入口
│   ├── predict_yolo.py               # 图片/批量图片推理
│   ├── predict_video.py              # 视频推理
│   ├── camera_realtime.py            # 摄像头/视频流实时检测
│   ├── prepare_*.py                  # 数据准备与转换
│   ├── evaluate_*.py                 # 模型评估
│   └── analyze_*.py                  # 错误和漏检分析
├── src/wind_fault/
│   ├── api/app.py                    # FastAPI 服务和报告页面
│   ├── analysis/                     # 故障分析和风险判断
│   ├── data/                         # 数据集转换和图片切片
│   ├── maintenance/                 # 维修单和派工记录
│   ├── model/yolo_runner.py          # YOLO 推理封装
│   ├── reports/                      # 检测报告和错误挖掘
│   └── video/                        # 视频检测和关键帧处理
├── tests/                            # 自动化测试
├── pyproject.toml
├── requirements.txt
└── README_EN.md
```

## 快速开始

### 1. 创建环境

```powershell
cd wind-fault-detection-platform
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 CUDA，请安装与本机显卡和驱动匹配的 PyTorch 版本。模型权重和数据集需要单独准备，仓库中的 `.gitignore` 已默认忽略 `*.pt`、图片数据和运行产物。

### 2. 准备数据集

YOLO 数据集目录建议整理为：

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

`data.yaml` 需要包含训练/验证路径和类别名称。例如：

```yaml
path: datasets/wind_blade_defect
train: images/train
val: images/val
test: images/test
names:
  0: crack
  1: hole
  2: spalling
  3: corrosion
```

### 3. 训练模型

```powershell
python scripts/train_yolo.py `
  --data datasets/wind_blade_defect/data.yaml `
  --model yolov8n.pt `
  --epochs 80 `
  --imgsz 640 `
  --batch 16
```

使用 GPU 时可以增加 `--device 0`：

```powershell
python scripts/train_yolo.py `
  --data datasets/wind_blade_defect/data.yaml `
  --model yolov8n.pt `
  --epochs 80 `
  --imgsz 640 `
  --batch 8 `
  --device 0
```

也可以使用项目中的长时间训练脚本。脚本会检查可用设备，并在显存不足时使用更保守的输入尺寸和 batch size：

```powershell
.\scripts\start_overnight_training.ps1
```

### 4. 执行图片推理

```powershell
python scripts/predict_yolo.py `
  --model runs/wind_fault/baseline/weights/best.pt `
  --source test.jpg `
  --conf 0.3
```

推理结果、渲染图片和报告通常会保存到 `runs/` 目录。

### 5. 启动 Web 工作台

```powershell
$env:WIND_MODEL_PATH="runs/wind_fault/baseline/weights/best.pt"
uvicorn wind_fault.api.app:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/
```

模型路径也可以指向 `release/v0.1_baseline/model/best.pt` 或其他本地权重文件。

## API 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/detect` | 单张图片检测，表单字段为 `file` |
| `POST` | `/detect/batch` | 批量图片检测，表单字段为 `files` |
| `POST` | `/detect/video` | 视频检测，表单字段为 `file` |
| `GET` | `/history` | 查询历史检测任务 |
| `GET` | `/evaluation` | 查看评估页面 |
| `GET` | `/report/{request_id}` | 查看 HTML 检测报告 |
| `GET` | `/report/{request_id}/json` | 获取结构化检测报告 |
| `POST` | `/analyze/{request_id}` | 对检测结果执行故障分析 |

单图 API 调用示例：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect?conf=0.45" `
  -F "file=@test.jpg"
```

接口返回内容包含请求 ID、检测类别、置信度、边框、风险等级、报告地址和结果图片地址。

## 漏检分析

模型完成一轮评估后，可以将检测结果与标注进行对比，整理困难样本：

```powershell
python scripts/analyze_missed_samples.py `
  --data datasets/wind_blade_defect/data.yaml `
  --images datasets/wind_blade_defect/images/test `
  --labels datasets/wind_blade_defect/labels/test `
  --report runs/wind_fault/test_batch_80e/reports/detection_report.json `
  --output runs/wind_fault/test_batch_80e/reports/missed_samples
```

分析结果可以进一步复制到 `hard_cases/`，用于下一轮数据清洗、阈值调整和模型训练。

## 测试

```powershell
pytest
```

测试覆盖数据集工具、检测报告、维修单、分析逻辑和困难样本处理等模块。测试运行不需要提交模型权重，但部分真实推理功能需要安装完整依赖并准备本地模型。

## 输出结果

每次检测任务都会生成独立的任务目录，便于复核、归档和二次分析。典型输出包括：

```text
runs/<task>/
├── inputs/                 # 原始图片或视频
├── renders/                # 绘制检测框后的结果
├── keyframes/              # 视频关键帧
├── reports/
│   ├── detection_report.json
│   ├── detection_report.md
│   └── detection_report.html
└── metadata.json           # 任务参数和运行信息
```

JSON 适合程序继续处理，Markdown 适合提交实验记录，HTML 适合在浏览器中查看。报告会记录检测类别、置信度、边界框、风险等级、输入文件和模型信息。

## 配置与环境变量

默认配置位于 `configs/default.yaml`，可通过命令行参数或环境变量覆盖。常用变量如下：

| 变量 | 说明 |
| --- | --- |
| `WIND_MODEL_PATH` | Web 服务使用的模型权重路径 |
| `WIND_RUNS_DIR` | 检测任务和报告的输出目录 |
| `WIND_DEVICE` | 推理设备，例如 `cpu` 或 `0` |
| `WIND_CONF` | 默认置信度阈值 |

建议将本地数据、权重和运行产物放在被 `.gitignore` 忽略的目录中，避免把大文件或敏感巡检数据提交到公开仓库。

## 开发说明

- 新增数据处理逻辑时，同时补充对应的单元测试；
- 新增 API 时，保持报告 JSON 字段向后兼容；
- 实验结果建议保存配置、数据版本、权重来源和评估命令；
- 对外发布前请确认数据集、预训练权重和论文材料的授权范围。

## 当前限制

- 仓库不包含完整训练图片、标签图片和模型权重；
- 不同数据版本的类别定义和训练参数需要以对应目录说明为准；
- 视频编码预览依赖本机 OpenCV 或 FFmpeg 环境；
- 当前主要面向本地原型和研究迭代，尚未提供生产环境鉴权、任务队列和多用户部署方案；
- 检测结果仍需要人工复核，不能直接替代专业巡检结论。

## 推荐迭代路线

1. 统一数据集版本、类别映射和数据质量检查；
2. 使用基础 YOLO 模型建立可比较的基线；
3. 接入 DTU 1024 高分辨率切片，重点提升细小裂纹检出率；
4. 使用漏检、误检和低置信度样本进行闭环训练；
5. 比较切片、小目标检测头、困难样本重加权和形态感知模块；
6. 增加实验指标看板、任务队列、用户鉴权和可部署化配置；
7. 在真实巡检数据上进行跨风场、跨设备和跨天气条件验证。

## License

当前仓库尚未单独声明开源许可证。用于面试展示时，建议先将仓库保持为个人作品展示仓库，并在确认依赖、数据集和研究材料的授权范围后再补充正式 License。
