from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.reports.error_mining import mine_detection_errors, write_error_mining_reports
from wind_fault.reports.missed_samples import Box, box_iou, load_class_names, load_ground_truth

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _register_msf_checkpoint_classes() -> None:
    """Allow torch to load checkpoints saved by train_msf_yolo_full.py as __main__."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import train_msf_yolo_full as msf_train
    except Exception:
        return

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    for name in (
        "ShapeAwareBboxLoss",
        "MSFShapeAwareDetectionLoss",
        "MSFDetectionModel",
        "MSFFullInnovationTrainer",
    ):
        if hasattr(msf_train, name):
            setattr(main_module, name, getattr(msf_train, name))


_register_msf_checkpoint_classes()


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run this script with the training venv.") from exc
    return YOLO


def _iter_images(images_dir: Path, max_images: int | None = None) -> list[Path]:
    images = [path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if max_images is not None:
        images = images[:max_images]
    return images


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _clip_xyxy(xyxy: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _result_to_detections(
    result: Any,
    class_names: dict[int, str],
    x_offset: int = 0,
    y_offset: int = 0,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    result_boxes = getattr(result, "boxes", None)
    if result_boxes is None:
        return detections

    for box in result_boxes:
        class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
        confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
        raw_xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else [0, 0, 0, 0]
        xyxy = (
            float(raw_xyxy[0]) + x_offset,
            float(raw_xyxy[1]) + y_offset,
            float(raw_xyxy[2]) + x_offset,
            float(raw_xyxy[3]) + y_offset,
        )
        if image_width is not None and image_height is not None:
            xyxy = _clip_xyxy(xyxy, image_width, image_height)
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        class_name = class_names.get(class_id, str(class_id))
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "xyxy": xyxy,
            }
        )
    return detections


def _nms(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: float(item["confidence"]), reverse=True):
        duplicate = False
        for existing in kept:
            if int(existing["class_id"]) != int(detection["class_id"]):
                continue
            if box_iou(tuple(existing["xyxy"]), tuple(detection["xyxy"])) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def _predict_full(
    model: Any,
    image: Image.Image,
    class_names: dict[int, str],
    imgsz: int,
    conf: float,
    device: str | None,
) -> list[dict[str, Any]]:
    array = np.array(image.convert("RGB"))
    results = model.predict(array, imgsz=imgsz, conf=conf, device=device, verbose=False)
    return _result_to_detections(results[0], class_names, image_width=image.width, image_height=image.height)


def _predict_sliced(
    model: Any,
    image: Image.Image,
    class_names: dict[int, str],
    imgsz: int,
    conf: float,
    device: str | None,
    tile_size: int,
    overlap: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    x_starts = _tile_starts(image.width, tile_size, overlap)
    y_starts = _tile_starts(image.height, tile_size, overlap)

    for y_start in y_starts:
        for x_start in x_starts:
            crop = image.crop(
                (
                    x_start,
                    y_start,
                    min(image.width, x_start + tile_size),
                    min(image.height, y_start + tile_size),
                )
            )
            array = np.array(crop.convert("RGB"))
            results = model.predict(array, imgsz=imgsz, conf=conf, device=device, verbose=False)
            detections.extend(
                _result_to_detections(
                    results[0],
                    class_names,
                    x_offset=x_start,
                    y_offset=y_start,
                    image_width=image.width,
                    image_height=image.height,
                )
            )

    return _nms(detections, nms_iou)


def _to_boxes(predictions: dict[str, list[dict[str, Any]]]) -> dict[str, list[Box]]:
    boxes_by_stem: dict[str, list[Box]] = {}
    for stem, detections in predictions.items():
        boxes_by_stem[stem] = [
            Box(
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                xyxy=tuple(map(float, item["xyxy"])),
                confidence=float(item["confidence"]),
            )
            for item in detections
        ]
    return boxes_by_stem


def _write_detection_report(output_dir: Path, image_paths: list[Path], predictions: dict[str, list[dict[str, Any]]]) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for image_path in image_paths:
        detections = []
        for item in predictions.get(image_path.stem, []):
            detections.append(
                {
                    "class_id": int(item["class_id"]),
                    "class_name": str(item["class_name"]),
                    "confidence": round(float(item["confidence"]), 6),
                    "xyxy": [round(float(value), 2) for value in item["xyxy"]],
                }
            )
        items.append(
            {
                "image": str(image_path),
                "defect_count": len(detections),
                "detections": detections,
            }
        )

    payload = {
        "image_count": len(items),
        "total_defects": sum(item["defect_count"] for item in items),
        "items": items,
    }
    report_path = reports_dir / "detection_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _summarize_predictions(predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_class: Counter[str] = Counter()
    images_with_predictions = 0
    for detections in predictions.values():
        if detections:
            images_with_predictions += 1
        for detection in detections:
            by_class[str(detection["class_name"])] += 1
    return {
        "images_with_predictions": images_with_predictions,
        "total_predictions": sum(by_class.values()),
        "prediction_by_class": dict(sorted(by_class.items())),
    }


def _evaluate_mode(
    mode_name: str,
    split_dir: Path,
    image_paths: list[Path],
    ground_truth: dict[str, list[Box]],
    model: Any,
    class_names: dict[int, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for index, image_path in enumerate(image_paths, start=1):
        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {split_dir.name}/{mode_name}: {index}/{len(image_paths)}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            full_predictions = _predict_full(model, image, class_names, args.imgsz, args.conf, args.device)
            if mode_name == "normal":
                mode_predictions = _nms(full_predictions, args.nms_iou)
            else:
                sliced_predictions = _predict_sliced(
                    model,
                    image,
                    class_names,
                    args.imgsz,
                    args.conf,
                    args.device,
                    args.tile_size,
                    args.overlap,
                    args.nms_iou,
                )
                if mode_name == "blade_slice":
                    mode_predictions = sliced_predictions
                elif mode_name == "blade_slice_fused":
                    mode_predictions = _nms(full_predictions + sliced_predictions, args.nms_iou)
                else:
                    raise ValueError(f"Unknown mode: {mode_name}")
        predictions[image_path.stem] = mode_predictions

    mode_dir = output_dir / mode_name
    detection_report = _write_detection_report(mode_dir, image_paths, predictions)
    rows, summary = mine_detection_errors(
        ground_truth=ground_truth,
        predictions=_to_boxes(predictions),
        iou_threshold=args.eval_iou,
        low_confidence_threshold=args.low_conf,
    )
    error_reports = write_error_mining_reports(mode_dir / "error_mining", rows, summary)
    return {
        "mode": mode_name,
        "detection_report": str(detection_report),
        "error_report_json": error_reports["json"],
        "error_report_csv": error_reports["csv"],
        "error_report_markdown": error_reports["markdown"],
        "prediction_summary": _summarize_predictions(predictions),
        "error_summary": summary,
    }


def _write_summary(output_dir: Path, config: dict[str, Any], results: list[dict[str, Any]]) -> None:
    payload = {"config": config, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V4 Blade-Slice evaluation",
        "",
        "## Config",
        "",
        f"- model: `{config['model']}`",
        f"- dataset: `{config['dataset']}`",
        f"- conf: `{config['conf']}`",
        f"- imgsz: `{config['imgsz']}`",
        f"- tile_size: `{config['tile_size']}`",
        f"- overlap: `{config['overlap']}`",
        f"- eval_iou: `{config['eval_iou']}`",
        "",
        "## Results",
        "",
        "| Split | Mode | Recall | Precision | Missed | False positives | Predictions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        summary = result["error_summary"]
        pred_summary = result["prediction_summary"]
        lines.append(
            f"| {result['split']} | {result['mode']} | "
            f"{summary.get('recall')} | {summary.get('precision')} | "
            f"{summary.get('missed_count')} | {summary.get('false_positive_count')} | "
            f"{pred_summary.get('total_predictions')} |"
        )

    lines.extend(["", "## Recall by class", ""])
    for result in results:
        lines.append(f"### {result['split']} / {result['mode']}")
        lines.append("")
        lines.append("| Class | GT | Matched | Missed | Recall |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        summary = result["error_summary"]
        for class_name, total_count in summary["total_gt_by_class"].items():
            matched = summary["matched_gt_by_class"].get(class_name, 0)
            missed = summary["missed_by_class"].get(class_name, 0)
            recall = summary["recall_by_class"].get(class_name, "")
            lines.append(f"| {class_name} | {total_count} | {matched} | {missed} | {recall} |")
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate normal YOLO inference against Blade-Slice inference.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root containing images/<split> and labels/<split>.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml used only for class names.")
    parser.add_argument("--splits", nargs="+", default=["val", "test_clean"])
    parser.add_argument("--output", default="runs/wind_fault/msf_blade_slice_v4_eval", type=Path)
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--tile-size", default=640, type=int)
    parser.add_argument("--overlap", default=0.25, type=float)
    parser.add_argument("--conf", default=0.45, type=float)
    parser.add_argument("--low-conf", default=0.45, type=float)
    parser.add_argument("--eval-iou", default=0.5, type=float)
    parser.add_argument("--nms-iou", default=0.5, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["normal", "blade_slice_fused"],
        choices=["normal", "blade_slice", "blade_slice_fused"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    class_names = load_class_names(args.data)
    YOLO = _load_yolo()
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for split in args.splits:
        split_images = args.dataset / "images" / split
        split_labels = args.dataset / "labels" / split
        image_paths = _iter_images(split_images, args.max_images)
        ground_truth = load_ground_truth(split_labels, split_images, class_names)
        if args.max_images is not None:
            ground_truth = {image_path.stem: ground_truth.get(image_path.stem, []) for image_path in image_paths}

        split_output = args.output / split
        split_output.mkdir(parents=True, exist_ok=True)
        for mode in args.modes:
            result = _evaluate_mode(
                mode_name=mode,
                split_dir=split_images,
                image_paths=image_paths,
                ground_truth=ground_truth,
                model=model,
                class_names=class_names,
                output_dir=split_output,
                args=args,
            )
            result["split"] = split
            results.append(result)

    config = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "data": str(args.data),
        "splits": args.splits,
        "modes": args.modes,
        "imgsz": args.imgsz,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "conf": args.conf,
        "low_conf": args.low_conf,
        "eval_iou": args.eval_iou,
        "nms_iou": args.nms_iou,
        "device": args.device,
        "max_images": args.max_images,
    }
    _write_summary(args.output, config, results)
    print(f"Summary: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
