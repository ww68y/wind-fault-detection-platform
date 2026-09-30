from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.model.yolo_runner import predict_yolo
from wind_fault.reports.prediction_report import write_prediction_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO prediction and write a detection report.")
    parser.add_argument("--model", required=True, type=Path, help="Trained YOLO model path.")
    parser.add_argument("--source", required=True, type=Path, help="Image, video, or directory.")
    parser.add_argument("--conf", default=0.45, type=float)
    parser.add_argument("--project", default="runs/wind_fault")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--device", default=None, help="Inference device, for example 0 for GPU or cpu.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = predict_yolo(
        model=args.model,
        source=args.source,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=True,
        device=args.device,
    )
    report = write_prediction_report(results, Path(args.project) / args.name / "reports")
    print(f"Detection report: {report['report_json']}")


if __name__ == "__main__":
    main()
