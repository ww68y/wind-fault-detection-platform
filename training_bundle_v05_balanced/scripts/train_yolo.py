from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.model.yolo_runner import train_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO for wind turbine blade defect detection.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml path.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--project", default="runs/wind_fault")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default=None, help="Training device, for example 0 for GPU or cpu.")
    parser.add_argument("--workers", default=8, type=int, help="Data loader workers. Use 0 on low-memory Windows PCs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_yolo(
        data=args.data,
        model=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
