from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.reports.missed_samples import (
    analyze_misses,
    load_class_names,
    load_ground_truth,
    load_predictions,
    write_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze missed detection samples by comparing GT and predictions.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml path.")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing images for one split.")
    parser.add_argument("--labels", required=True, type=Path, help="Directory containing YOLO labels for one split.")
    parser.add_argument("--report", required=True, type=Path, help="Prediction detection_report.json path.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for missed sample reports.")
    parser.add_argument("--iou", default=0.5, type=float, help="IoU threshold for matching GT and predictions.")
    parser.add_argument("--top-n", default=50, type=int, help="Top N missed images in markdown report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    class_names = load_class_names(args.data)
    ground_truth = load_ground_truth(args.labels, args.images, class_names)
    predictions = load_predictions(args.report)
    rows, summary = analyze_misses(ground_truth, predictions, iou_threshold=args.iou)
    outputs = write_reports(args.output, rows, summary, top_n=args.top_n)
    print(f"Missed sample count: {summary['image_count_with_miss']}")
    print(f"Complete miss images: {summary['complete_miss_images']}")
    print(f"CSV report: {outputs['csv']}")
    print(f"Markdown report: {outputs['markdown']}")
    print(f"JSON report: {outputs['json']}")


if __name__ == "__main__":
    main()

