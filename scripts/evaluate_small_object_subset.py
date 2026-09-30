from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * idx, 2) for idx in range(10))


def _register_msf_checkpoint_classes() -> None:
    """Allow torch to load MSF checkpoints saved with custom training classes."""
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


def _parse_data_yaml(data_yaml: Path) -> dict[str, Any]:
    content = data_yaml.read_text(encoding="utf-8").splitlines()
    payload: dict[str, Any] = {"names": {}}
    in_names = False
    for raw_line in content:
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        if raw_line.strip() == "names:":
            in_names = True
            continue
        if in_names and raw_line.startswith("  "):
            idx_text, name = raw_line.strip().split(":", 1)
            payload["names"][int(idx_text)] = name.strip()
            continue
        in_names = False
        if ":" in raw_line:
            key, value = raw_line.split(":", 1)
            payload[key.strip()] = value.strip()
    dataset_root = Path(payload["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    payload["root"] = dataset_root
    return payload


def _split_paths(data_info: dict[str, Any], split_key: str) -> tuple[Path, Path, str]:
    split_rel = str(data_info[split_key])
    images_dir = data_info["root"] / split_rel
    if "images/" in split_rel.replace("\\", "/"):
        labels_rel = split_rel.replace("\\", "/").replace("images/", "labels/", 1)
    else:
        labels_rel = split_rel
    labels_dir = data_info["root"] / labels_rel
    return images_dir, labels_dir, Path(split_rel).name


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        image_path = images_dir / f"{stem}{suffix}"
        if image_path.exists():
            return image_path
    return None


def _xywh_to_xyxy_pixels(x: float, y: float, w: float, h: float, width: int, height: int) -> list[float]:
    return [
        (x - w / 2.0) * width,
        (y - h / 2.0) * height,
        (x + w / 2.0) * width,
        (y + h / 2.0) * height,
    ]


def _parse_label(label_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not label_path.exists():
        return rows
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        x, y, w, h = (float(value) for value in parts[1:5])
        rows.append({"class_id": class_id, "xywh": [x, y, w, h], "area": w * h})
    return rows


def _load_split_items(images_dir: Path, labels_dir: Path, max_area: float) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        labels = _parse_label(label_path)
        if not any(label["area"] <= max_area for label in labels):
            continue
        image_path = _find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        items.append({"image": image_path, "label": label_path, "labels": labels})
    return items


def _box_iou(box: list[float], other: list[float]) -> float:
    ix1 = max(box[0], other[0])
    iy1 = max(box[1], other[1])
    ix2 = min(box[2], other[2])
    iy2 = min(box[3], other[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    union = box_area + other_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _ap_from_pr(tp_flags: list[int], fp_flags: list[int], npos: int) -> float:
    if npos <= 0:
        return 0.0
    cum_tp = 0
    cum_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(tp_flags, fp_flags):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / npos)
        denom = cum_tp + cum_fp
        precisions.append(cum_tp / denom if denom else 0.0)
    ap = 0.0
    for idx in range(101):
        recall_level = idx / 100.0
        best_precision = 0.0
        for recall, precision in zip(recalls, precisions):
            if recall >= recall_level and precision > best_precision:
                best_precision = precision
        ap += best_precision / 101.0
    return ap


def _match_detections(
    detections: list[dict[str, Any]],
    small_gts_by_image: dict[str, list[list[float]]],
    ignore_gts_by_image: dict[str, list[list[float]]],
    iou_threshold: float,
    ignore_iou: float,
) -> dict[str, Any]:
    matched: dict[str, list[bool]] = {
        image_key: [False] * len(boxes) for image_key, boxes in small_gts_by_image.items()
    }
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    ignored = 0
    for det in sorted(detections, key=lambda row: row["confidence"], reverse=True):
        image_key = det["image"]
        box = det["box"]
        small_gts = small_gts_by_image.get(image_key, [])
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt_box in enumerate(small_gts):
            if matched.get(image_key, [])[gt_idx]:
                continue
            iou = _box_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched[image_key][best_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
            continue

        ignore_hit = any(_box_iou(box, ignore_box) >= ignore_iou for ignore_box in ignore_gts_by_image.get(image_key, []))
        if ignore_hit:
            ignored += 1
            continue
        tp_flags.append(0)
        fp_flags.append(1)
    return {"tp_flags": tp_flags, "fp_flags": fp_flags, "ignored": ignored}


def _count_gt_by_class(
    labels_by_image: dict[str, list[dict[str, Any]]],
    shapes: dict[str, tuple[int, int]],
    area_threshold: float,
) -> dict[int, dict[str, Any]]:
    class_payload: dict[int, dict[str, Any]] = defaultdict(lambda: {"small": defaultdict(list), "ignore": defaultdict(list), "npos": 0})
    for image_key, labels in labels_by_image.items():
        height, width = shapes[image_key]
        for label in labels:
            class_id = int(label["class_id"])
            x, y, w, h = label["xywh"]
            box = _xywh_to_xyxy_pixels(x, y, w, h, width, height)
            if label["area"] <= area_threshold:
                class_payload[class_id]["small"][image_key].append(box)
                class_payload[class_id]["npos"] += 1
            else:
                class_payload[class_id]["ignore"][image_key].append(box)
    return class_payload


def _evaluate_model_on_threshold(
    labels_by_image: dict[str, list[dict[str, Any]]],
    predictions_by_image: dict[str, list[dict[str, Any]]],
    shapes: dict[str, tuple[int, int]],
    class_names: dict[int, str],
    area_threshold: float,
    metric_conf: float,
    ignore_iou: float,
) -> dict[str, Any]:
    class_gt = _count_gt_by_class(labels_by_image, shapes, area_threshold)
    class_metrics: dict[str, Any] = {}
    ap50_values: list[float] = []
    ap_all_values: list[float] = []
    micro_tp = 0
    micro_fp = 0
    micro_ignored = 0
    micro_npos = 0

    for class_id, class_name in class_names.items():
        gt_payload = class_gt.get(class_id, {"small": {}, "ignore": {}, "npos": 0})
        npos = int(gt_payload["npos"])
        detections = [
            {**prediction, "image": image_key}
            for image_key, predictions in predictions_by_image.items()
            for prediction in predictions
            if int(prediction["class_id"]) == int(class_id)
        ]
        aps: list[float] = []
        for iou_threshold in IOU_THRESHOLDS:
            match_payload = _match_detections(
                detections,
                gt_payload["small"],
                gt_payload["ignore"],
                iou_threshold,
                ignore_iou,
            )
            aps.append(_ap_from_pr(match_payload["tp_flags"], match_payload["fp_flags"], npos))
        conf_detections = [row for row in detections if row["confidence"] >= metric_conf]
        conf_match = _match_detections(conf_detections, gt_payload["small"], gt_payload["ignore"], 0.5, ignore_iou)
        tp = sum(conf_match["tp_flags"])
        fp = sum(conf_match["fp_flags"])
        ignored = int(conf_match["ignored"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / npos if npos else 0.0

        if npos > 0:
            ap50_values.append(aps[0])
            ap_all_values.extend(aps)
        micro_tp += tp
        micro_fp += fp
        micro_ignored += ignored
        micro_npos += npos
        class_metrics[class_name] = {
            "small_gt": npos,
            "precision_iou50_conf": precision,
            "recall_iou50_conf": recall,
            "ap50": aps[0] if npos else None,
            "map50_95": sum(aps) / len(aps) if npos else None,
            "tp_iou50_conf": tp,
            "fp_iou50_conf": fp,
            "ignored_detections_iou50_conf": ignored,
            "detections": len(detections),
        }

    return {
        "small_gt": micro_npos,
        "precision_iou50_conf": micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0,
        "recall_iou50_conf": micro_tp / micro_npos if micro_npos else 0.0,
        "ap50": sum(ap50_values) / len(ap50_values) if ap50_values else None,
        "map50_95": sum(ap_all_values) / len(ap_all_values) if ap_all_values else None,
        "tp_iou50_conf": micro_tp,
        "fp_iou50_conf": micro_fp,
        "ignored_detections_iou50_conf": micro_ignored,
        "class_metrics": class_metrics,
    }


def _predict_model(
    model_path: Path,
    image_paths: list[Path],
    imgsz: int,
    batch: int,
    ap_conf: float,
    device: str | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[int, int]]]:
    YOLO = _load_yolo()
    model = YOLO(str(model_path))
    predictions: dict[str, list[dict[str, Any]]] = {}
    shapes: dict[str, tuple[int, int]] = {}
    if not image_paths:
        return predictions, shapes
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=ap_conf,
        batch=batch,
        device=device,
        save=False,
        verbose=False,
    )
    for image_path, result in zip(image_paths, results):
        image_key = str(image_path)
        height, width = tuple(int(value) for value in result.orig_shape)
        shapes[image_key] = (height, width)
        rows: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                rows.append(
                    {
                        "class_id": int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0,
                        "confidence": float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0,
                        "box": xyxy,
                    }
                )
        predictions[image_key] = rows
    return predictions, shapes


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _threshold_name(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _write_subset_csv(output_dir: Path, threshold: float, split_name: str, rows: list[dict[str, Any]]) -> None:
    out_path = output_dir / f"subset_{split_name}_area_le_{_threshold_name(threshold)}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "label", "small_gt", "total_gt", "class_counts"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "small_object_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    lines = [
        "# Small-object subset evaluation",
        "",
        "Small GT boxes are selected by normalized bbox area. Larger GT boxes in the same images are treated as ignore regions, so detections on large objects are not counted as false positives.",
        "",
        "## Models",
        "",
    ]
    for model_name, model_info in payload["models"].items():
        lines.append(f"- {model_name}: `{model_info['path']}` @ imgsz={model_info['imgsz']}")
    lines.extend(
        [
            "",
            f"- data: `{payload['data']}`",
            f"- metric_conf: `{payload['metric_conf']}`",
            f"- ap_conf: `{payload['ap_conf']}`",
            f"- ignore_iou: `{payload['ignore_iou']}`",
            "",
        ]
    )

    reference = payload.get("reference")
    for threshold_key, threshold_payload in payload["thresholds"].items():
        lines.extend([f"## Area <= {threshold_payload['area_threshold']}", ""])
        for split_name, split_payload in threshold_payload["splits"].items():
            lines.extend(
                [
                    f"### {split_name}",
                    "",
                    f"- images: {split_payload['images']}",
                    f"- small_gt: {split_payload['small_gt']}",
                    f"- class_counts: `{json.dumps(split_payload['class_counts'], ensure_ascii=False)}`",
                    "",
                    "| Model | Precision@0.5/conf | Recall@0.5/conf | AP50 | mAP50-95 | TP | FP | Ignored det |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for model_name, metrics in split_payload["models"].items():
                lines.append(
                    f"| {model_name} | "
                    f"{_format_metric(metrics['precision_iou50_conf'])} | "
                    f"{_format_metric(metrics['recall_iou50_conf'])} | "
                    f"{_format_metric(metrics['ap50'])} | "
                    f"{_format_metric(metrics['map50_95'])} | "
                    f"{metrics['tp_iou50_conf']} | "
                    f"{metrics['fp_iou50_conf']} | "
                    f"{metrics['ignored_detections_iou50_conf']} |"
                )
                csv_rows.append(
                    {
                        "threshold": threshold_payload["area_threshold"],
                        "split": split_name,
                        "model": model_name,
                        "images": split_payload["images"],
                        "small_gt": split_payload["small_gt"],
                        "precision_iou50_conf": metrics["precision_iou50_conf"],
                        "recall_iou50_conf": metrics["recall_iou50_conf"],
                        "ap50": metrics["ap50"],
                        "map50_95": metrics["map50_95"],
                        "tp_iou50_conf": metrics["tp_iou50_conf"],
                        "fp_iou50_conf": metrics["fp_iou50_conf"],
                        "ignored_detections_iou50_conf": metrics["ignored_detections_iou50_conf"],
                    }
                )
            if reference and reference in split_payload["models"]:
                ref_metrics = split_payload["models"][reference]
                lines.extend(["", f"#### {reference} - baseline deltas", ""])
                lines.extend(
                    [
                        "| Baseline | ΔPrecision | ΔRecall | ΔAP50 | ΔmAP50-95 |",
                        "| --- | ---: | ---: | ---: | ---: |",
                    ]
                )
                for model_name, metrics in split_payload["models"].items():
                    if model_name == reference:
                        continue
                    lines.append(
                        f"| {model_name} | "
                        f"{_format_metric(ref_metrics['precision_iou50_conf'] - metrics['precision_iou50_conf'])} | "
                        f"{_format_metric(ref_metrics['recall_iou50_conf'] - metrics['recall_iou50_conf'])} | "
                        f"{_format_metric(None if ref_metrics['ap50'] is None or metrics['ap50'] is None else ref_metrics['ap50'] - metrics['ap50'])} | "
                        f"{_format_metric(None if ref_metrics['map50_95'] is None or metrics['map50_95'] is None else ref_metrics['map50_95'] - metrics['map50_95'])} |"
                    )
            lines.append("")

    with (output_dir / "small_object_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "threshold",
            "split",
            "model",
            "images",
            "small_gt",
            "precision_iou50_conf",
            "recall_iou50_conf",
            "ap50",
            "map50_95",
            "tp_iou50_conf",
            "fp_iou50_conf",
            "ignored_detections_iou50_conf",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_dir / "small_object_eval.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_model_args(raw_models: list[list[str]]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for name, path_text, imgsz_text in raw_models:
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        model_path = Path(path_text)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {name}: {model_path}")
        models[name] = {"path": model_path, "imgsz": int(imgsz_text)}
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate models on small-object GT boxes only.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", nargs=3, action="append", metavar=("NAME", "PATH", "IMGSZ"), required=True)
    parser.add_argument("--reference", default="V6")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.005])
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold for reported precision/recall.")
    parser.add_argument("--ap-conf", type=float, default=0.001, help="Low confidence threshold used for AP integration.")
    parser.add_argument("--ignore-iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--summary-only", action="store_true", help="Only summarize small-object subsets; skip model inference.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    data_info = _parse_data_yaml(args.data)
    class_names = {int(key): str(value) for key, value in data_info["names"].items()}
    models = _parse_model_args(args.model)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(set(args.thresholds), reverse=True)
    max_area = max(thresholds)

    payload: dict[str, Any] = {
        "data": str(args.data),
        "models": {name: {"path": str(info["path"]), "imgsz": info["imgsz"]} for name, info in models.items()},
        "reference": args.reference,
        "thresholds": {},
        "metric_conf": args.conf,
        "ap_conf": args.ap_conf,
        "ignore_iou": args.ignore_iou,
    }

    for threshold in thresholds:
        payload["thresholds"][_threshold_name(threshold)] = {"area_threshold": threshold, "splits": {}}

    for split_key in args.splits:
        if split_key not in data_info:
            continue
        images_dir, labels_dir, split_name = _split_paths(data_info, split_key)
        max_items = _load_split_items(images_dir, labels_dir, max_area)
        max_image_paths = [item["image"] for item in max_items]
        labels_by_image_all = {str(item["image"]): item["labels"] for item in max_items}
        print(f"Split {split_name}: {len(max_items)} images with area <= {max_area}")

        predictions_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
        shapes_by_model: dict[str, dict[str, tuple[int, int]]] = {}
        if not args.summary_only:
            for model_name, model_info in models.items():
                print(f"Predicting {model_name} on {split_name} ({len(max_image_paths)} images, imgsz={model_info['imgsz']})")
                predictions, shapes = _predict_model(
                    model_info["path"],
                    max_image_paths,
                    model_info["imgsz"],
                    args.batch,
                    args.ap_conf,
                    args.device,
                )
                predictions_by_model[model_name] = predictions
                shapes_by_model[model_name] = shapes

        for threshold in thresholds:
            threshold_key = _threshold_name(threshold)
            threshold_items = [
                item for item in max_items if any(label["area"] <= threshold for label in item["labels"])
            ]
            labels_by_image = {str(item["image"]): item["labels"] for item in threshold_items}
            image_keys = set(labels_by_image)
            class_counter: Counter[str] = Counter()
            subset_rows: list[dict[str, Any]] = []
            small_gt_total = 0
            for item in threshold_items:
                counts: Counter[str] = Counter()
                for label in item["labels"]:
                    if label["area"] <= threshold:
                        class_name = class_names.get(int(label["class_id"]), str(label["class_id"]))
                        counts[class_name] += 1
                        class_counter[class_name] += 1
                        small_gt_total += 1
                subset_rows.append(
                    {
                        "image": str(item["image"]),
                        "label": str(item["label"]),
                        "small_gt": sum(counts.values()),
                        "total_gt": len(item["labels"]),
                        "class_counts": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
                    }
                )
            _write_subset_csv(output_dir, threshold, split_name, subset_rows)
            split_payload: dict[str, Any] = {
                "images": len(threshold_items),
                "small_gt": small_gt_total,
                "class_counts": dict(sorted(class_counter.items())),
                "models": {},
            }
            if not args.summary_only:
                for model_name in models:
                    predictions_for_threshold = {
                        image_key: rows
                        for image_key, rows in predictions_by_model[model_name].items()
                        if image_key in image_keys
                    }
                    shapes_for_threshold = {
                        image_key: shape
                        for image_key, shape in shapes_by_model[model_name].items()
                        if image_key in image_keys
                    }
                    split_payload["models"][model_name] = _evaluate_model_on_threshold(
                        labels_by_image,
                        predictions_for_threshold,
                        shapes_for_threshold,
                        class_names,
                        threshold,
                        args.conf,
                        args.ignore_iou,
                    )
            payload["thresholds"][threshold_key]["splits"][split_name] = split_payload

    payload["elapsed_seconds"] = round(time.time() - start, 2)
    _write_report(output_dir, payload)
    print(f"Small-object report: {output_dir / 'small_object_eval.md'}")


if __name__ == "__main__":
    main()
