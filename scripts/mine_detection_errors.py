from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.reports.error_mining import mine_detection_errors, write_error_mining_reports
from wind_fault.reports.missed_samples import load_class_names, load_ground_truth, load_predictions


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
ERROR_DIRS = {
    "missed": "missed_samples",
    "low_confidence": "low_confidence_samples",
    "false_positive": "false_positive_samples",
}


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _copy_review_images(rows: list[dict], images_dir: Path, output_dir: Path) -> dict[str, int]:
    copied: dict[str, int] = {name: 0 for name in ERROR_DIRS}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        error_type = str(row["error_type"])
        image_stem = str(row["image_stem"])
        key = (error_type, image_stem)
        if key in seen or error_type not in ERROR_DIRS:
            continue
        seen.add(key)
        image_path = _find_image(images_dir, image_stem)
        if image_path is None:
            continue
        target_dir = output_dir / ERROR_DIRS[error_type]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_dir / image_path.name)
        copied[error_type] += 1
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine missed, low-confidence, and false-positive samples.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml path.")
    parser.add_argument("--images", required=True, type=Path, help="Image split directory.")
    parser.add_argument("--labels", required=True, type=Path, help="YOLO label split directory.")
    parser.add_argument("--report", required=True, type=Path, help="Prediction detection_report.json path.")
    parser.add_argument("--output", required=True, type=Path, help="Output report/review directory.")
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument("--low-conf", default=0.45, type=float)
    parser.add_argument("--copy-review-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    class_names = load_class_names(args.data)
    ground_truth = load_ground_truth(args.labels, args.images, class_names)
    predictions = load_predictions(args.report)
    rows, summary = mine_detection_errors(
        ground_truth=ground_truth,
        predictions=predictions,
        iou_threshold=args.iou,
        low_confidence_threshold=args.low_conf,
    )
    outputs = write_error_mining_reports(args.output, rows, summary)
    print(f"Recall: {summary['recall']}")
    print(f"Precision: {summary['precision']}")
    print(f"Missed boxes: {summary['missed_count']}")
    print(f"Low-confidence boxes: {summary['low_confidence_count']}")
    print(f"False-positive boxes: {summary['false_positive_count']}")
    print(f"JSON report: {outputs['json']}")
    print(f"CSV report: {outputs['csv']}")
    print(f"Markdown report: {outputs['markdown']}")
    if args.copy_review_images:
        copied = _copy_review_images(rows, args.images, args.output)
        for error_type, count in sorted(copied.items()):
            print(f"Copied {error_type} review images: {count}")


if __name__ == "__main__":
    main()
