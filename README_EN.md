# Wind Turbine Blade Defect Detection Platform

[中文 README](README.md)

A computer vision engineering project for wind turbine blade inspection. Built on top of Ultralytics YOLO, the project connects dataset preparation, model training, image/video inference, result reporting, and a FastAPI-based local web platform into one extensible workflow.

The goal is not only to provide a one-off training script. The platform keeps the complete inspection loop in mind: a user uploads an image or video, the system runs detection, stores the original input and inference artifacts, generates JSON/Markdown/HTML reports, and preserves structured information for missed-sample analysis, maintenance recommendations, and future model iterations.

## Project Overview

Wind turbine blades may develop cracks, holes, spalling, and corrosion during long-term operation. Manual inspection is expensive, while small defects, distant targets, and complex backgrounds can lead to missed detections. This project is intended to assist inspectors with an automated first-pass screening workflow and provide evidence for later review.

The default defect categories are:

| Class | Meaning | Typical examples |
| --- | --- | --- |
| `crack` | Crack | Fine or elongated surface cracks |
| `hole` | Hole | Local perforation or visible holes |
| `spalling` | Spalling | Surface material loss or damage |
| `corrosion` | Corrosion | Corroded areas or corrosion spots |

The repository is currently positioned as a local inspection and demonstration prototype. Training images, model weights, and runtime artifacts are intentionally excluded from Git to keep the repository manageable and avoid committing large files. See `datasets/README.md` for data preparation notes.

## Key Capabilities

### Detection and Inference

- Single-image detection;
- Batch image detection;
- Video detection with result videos and keyframes;
- Real-time camera/video-stream detection script;
- Configurable confidence thresholds;
- CPU and CUDA device support.

### Web API and Workbench

- FastAPI HTTP backend;
- Built-in browser-based inspection workbench;
- Automatic storage of uploads, rendered results, reports, and history;
- Separate workflows for images, batches, and videos;
- Health checks, history queries, report pages, and evaluation pages;
- Browser-accessible detection results.

### Reporting and Maintenance Workflow

- Structured JSON reports;
- Human-readable Markdown reports;
- HTML report pages;
- Defect class, confidence, bounding-box, and risk summaries;
- Keyframes and rendered detection images;
- Missed, false-positive, and low-confidence sample organization;
- Fault analysis, maintenance orders, and dispatch records.

### Data and Experimentation

- COCO, VOC XML, and LabelMe annotation conversion;
- High-resolution image slicing;
- DTU 1024 sliced-data integration;
- YOLO baseline and MSF-YOLO experiment scripts;
- Iteration directions for small objects, sliced inference, hard cases, and shape-aware features;
- Pytest coverage for data utilities, reports, maintenance orders, and the detection loop.

## System Workflow

```text
Image / video input
        │
        ▼
Upload and task creation
        │
        ▼
YOLO inference
        │
        ├── Classes, boxes, and confidence scores
        ├── Rendered images or result video
        ├── Keyframes
        └── Raw inference details
        │
        ▼
Report generation and risk assessment
        │
        ├── JSON
        ├── Markdown
        ├── HTML page
        └── History / maintenance recommendations
        │
        ▼
Hard-case mining and next training iteration
```

## Technology Stack

- Python 3.10+
- Ultralytics YOLO
- FastAPI + Uvicorn
- OpenCV
- Pillow, NumPy, and Pandas
- PyYAML
- pytest

## Repository Structure

```text
wind-fault-detection-platform/
├── configs/
│   ├── default.yaml                  # Default training, inference, and slicing config
│   └── *.yaml                        # Experiment configurations
├── datasets/
│   ├── README.md                     # Dataset preparation notes
│   └── */data.yaml                   # Dataset-version configurations
├── hard_cases/                       # Missed, false-positive, and hard samples
├── models/                           # Model notes; large weights are excluded
├── paper_assets_msf_yolo/            # Research materials and paper documents
├── paper_latex/                      # Paper drafts and related materials
├── release/v0.1_baseline/            # Baseline release notes and artifacts
├── scripts/
│   ├── train_yolo.py                 # YOLO training entry point
│   ├── predict_yolo.py               # Image/batch inference
│   ├── predict_video.py              # Video inference
│   ├── camera_realtime.py            # Camera/video-stream inference
│   ├── prepare_*.py                  # Dataset preparation and conversion
│   ├── evaluate_*.py                 # Model evaluation
│   └── analyze_*.py                  # Error and missed-sample analysis
├── src/wind_fault/
│   ├── api/app.py                    # FastAPI service and report pages
│   ├── analysis/                     # Fault analysis and risk assessment
│   ├── data/                         # Dataset conversion and image slicing
│   ├── maintenance/                 # Maintenance orders and dispatch records
│   ├── model/yolo_runner.py          # YOLO inference wrapper
│   ├── reports/                      # Reports and error mining
│   └── video/                        # Video detection and keyframe handling
├── tests/                            # Automated tests
├── pyproject.toml
├── requirements.txt
└── README_EN.md
```

## Quick Start

### 1. Create an environment

```powershell
cd wind-fault-detection-platform
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA inference or training, install a PyTorch build compatible with the local GPU and driver. Dataset images and model weights must be prepared separately. The repository `.gitignore` excludes `*.pt`, image datasets, and runtime outputs by default.

### 2. Prepare a YOLO dataset

The recommended layout is:

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

The `data.yaml` file should define the train/validation paths and class names. For example:

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

### 3. Train a model

```powershell
python scripts/train_yolo.py `
  --data datasets/wind_blade_defect/data.yaml `
  --model yolov8n.pt `
  --epochs 80 `
  --imgsz 640 `
  --batch 16
```

Use `--device 0` to select the first CUDA device:

```powershell
python scripts/train_yolo.py `
  --data datasets/wind_blade_defect/data.yaml `
  --model yolov8n.pt `
  --epochs 80 `
  --imgsz 640 `
  --batch 8 `
  --device 0
```

For longer training runs, the repository also includes a PowerShell launcher. It checks the available device and falls back to a more conservative image size and batch size when necessary:

```powershell
.\scripts\start_overnight_training.ps1
```

### 4. Run image inference

```powershell
python scripts/predict_yolo.py `
  --model runs/wind_fault/baseline/weights/best.pt `
  --source test.jpg `
  --conf 0.3
```

Inference results, rendered images, and reports are normally written under `runs/`.

### 5. Start the web workbench

```powershell
$env:WIND_MODEL_PATH="runs/wind_fault/baseline/weights/best.pt"
uvicorn wind_fault.api.app:app --reload
```

Open the workbench at:

```text
http://127.0.0.1:8000/
```

The model path can also point to `release/v0.1_baseline/model/best.pt` or another local checkpoint.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/detect` | Single-image detection, form field: `file` |
| `POST` | `/detect/batch` | Batch image detection, form field: `files` |
| `POST` | `/detect/video` | Video detection, form field: `file` |
| `GET` | `/history` | Query previous detection tasks |
| `GET` | `/evaluation` | Open the evaluation page |
| `GET` | `/report/{request_id}` | Open an HTML detection report |
| `GET` | `/report/{request_id}/json` | Return a structured detection report |
| `POST` | `/analyze/{request_id}` | Run fault analysis on a detection result |

Example request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect?conf=0.45" `
  -F "file=@test.jpg"
```

The response includes a request ID, detected classes, confidence scores, bounding boxes, risk level, report URLs, and rendered-result URLs.

## Missed-Sample Analysis

After an evaluation run, compare predictions with annotations and collect difficult samples:

```powershell
python scripts/analyze_missed_samples.py `
  --data datasets/wind_blade_defect/data.yaml `
  --images datasets/wind_blade_defect/images/test `
  --labels datasets/wind_blade_defect/labels/test `
  --report runs/wind_fault/test_batch_80e/reports/detection_report.json `
  --output runs/wind_fault/test_batch_80e/reports/missed_samples
```

The resulting reports can be used to populate `hard_cases/` and guide data cleaning, threshold tuning, and the next training round.

## Testing

```powershell
pytest
```

The tests cover dataset utilities, detection reports, maintenance orders, analysis logic, and hard-case processing. Full model inference additionally requires the relevant runtime dependencies and a local model checkpoint.

## Output Artifacts

Each detection task is stored in its own directory so that results can be reviewed, archived, and analyzed later. A typical task looks like this:

```text
runs/<task>/
├── inputs/                 # Original image or video
├── renders/                # Rendered detections
├── keyframes/              # Video keyframes
├── reports/
│   ├── detection_report.json
│   ├── detection_report.md
│   └── detection_report.html
└── metadata.json           # Task parameters and runtime metadata
```

JSON is intended for downstream processing, Markdown for experiment records, and HTML for browser review. Reports include detected classes, confidence scores, bounding boxes, risk levels, input metadata, and model information.

## Configuration and Environment Variables

The default configuration is stored in `configs/default.yaml` and can be overridden by command-line arguments or environment variables.

| Variable | Description |
| --- | --- |
| `WIND_MODEL_PATH` | Model checkpoint used by the web service |
| `WIND_RUNS_DIR` | Directory for task outputs and reports |
| `WIND_DEVICE` | Inference device, such as `cpu` or `0` |
| `WIND_CONF` | Default confidence threshold |

Keep local datasets, checkpoints, and runtime outputs in directories ignored by `.gitignore`. This avoids committing large files or sensitive inspection data to a public repository.

## Development Notes

- Add unit tests when introducing new data-processing logic;
- Keep report JSON fields backward-compatible when adding API features;
- Record the configuration, dataset version, checkpoint source, and evaluation command for each experiment;
- Confirm the licenses of datasets, pretrained weights, and research materials before redistribution.

## Current Limitations

- The repository does not include the complete training images, labels, or model weights;
- Class definitions and training parameters may differ between dataset versions;
- Video preview generation depends on the local OpenCV or FFmpeg environment;
- The current implementation targets local prototyping and research iteration, not a production multi-user deployment;
- Detection results still require human review and must not be treated as a standalone professional inspection conclusion.

## Suggested Roadmap

1. Standardize dataset versions, class mappings, and data-quality checks;
2. Establish a reproducible YOLO baseline;
3. Integrate DTU 1024 sliced data to improve small-crack recall;
4. Use missed, false-positive, and low-confidence samples for closed-loop training;
5. Compare slicing, small-object heads, hard-case reweighting, and shape-aware modules;
6. Add experiment dashboards, task queues, authentication, and deployment configuration;
7. Validate across different wind farms, devices, and weather conditions.

## License

This repository does not currently declare a separate open-source license. For interview presentation, it is reasonable to keep it as a personal project showcase and add a formal license after confirming the licensing scope of dependencies, datasets, and research materials.
