from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from wind_fault.reports.missed_samples import Box, box_iou


def _box_to_list(box: Box | None) -> list[float]:
    if box is None:
        return []
    return [round(float(value), 3) for value in box.xyxy]


def _match_predictions(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
) -> tuple[dict[int, tuple[int, float]], set[int]]:
    matches: dict[int, tuple[int, float]] = {}
    used_predictions: set[int] = set()
    for gt_index, gt_box in enumerate(gt_boxes):
        best_pred_index = None
        best_iou = 0.0
        for pred_index, pred_box in enumerate(pred_boxes):
            if pred_index in used_predictions:
                continue
            if pred_box.class_id != gt_box.class_id:
                continue
            iou = box_iou(gt_box.xyxy, pred_box.xyxy)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_pred_index = pred_index
        if best_pred_index is not None:
            matches[gt_index] = (best_pred_index, best_iou)
            used_predictions.add(best_pred_index)
    return matches, used_predictions


def mine_detection_errors(
    ground_truth: dict[str, list[Box]],
    predictions: dict[str, list[Box]],
    iou_threshold: float = 0.5,
    low_confidence_threshold: float = 0.45,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_gt_by_class: Counter[str] = Counter()
    matched_gt_by_class: Counter[str] = Counter()
    missed_by_class: Counter[str] = Counter()
    low_confidence_by_class: Counter[str] = Counter()
    false_positive_by_class: Counter[str] = Counter()
    prediction_by_class: Counter[str] = Counter()
    images_with_miss: set[str] = set()
    images_with_low_confidence: set[str] = set()
    images_with_false_positive: set[str] = set()

    for stem in sorted(set(ground_truth) | set(predictions)):
        gt_boxes = ground_truth.get(stem, [])
        pred_boxes = predictions.get(stem, [])
        matches, used_predictions = _match_predictions(gt_boxes, pred_boxes, iou_threshold)

        for gt_box in gt_boxes:
            total_gt_by_class[gt_box.class_name] += 1
        for pred_box in pred_boxes:
            prediction_by_class[pred_box.class_name] += 1

        for gt_index, gt_box in enumerate(gt_boxes):
            match = matches.get(gt_index)
            if match is None:
                missed_by_class[gt_box.class_name] += 1
                images_with_miss.add(stem)
                rows.append(
                    {
                        "error_type": "missed",
                        "image_stem": stem,
                        "class_id": gt_box.class_id,
                        "class_name": gt_box.class_name,
                        "confidence": "",
                        "iou": "",
                        "gt_xyxy": _box_to_list(gt_box),
                        "pred_xyxy": [],
                    }
                )
                continue

            pred_index, iou = match
            pred_box = pred_boxes[pred_index]
            matched_gt_by_class[gt_box.class_name] += 1
            confidence = pred_box.confidence if pred_box.confidence is not None else 0.0
            if confidence < low_confidence_threshold:
                low_confidence_by_class[gt_box.class_name] += 1
                images_with_low_confidence.add(stem)
                rows.append(
                    {
                        "error_type": "low_confidence",
                        "image_stem": stem,
                        "class_id": gt_box.class_id,
                        "class_name": gt_box.class_name,
                        "confidence": round(confidence, 6),
                        "iou": round(iou, 6),
                        "gt_xyxy": _box_to_list(gt_box),
                        "pred_xyxy": _box_to_list(pred_box),
                    }
                )

        for pred_index, pred_box in enumerate(pred_boxes):
            if pred_index in used_predictions:
                continue
            false_positive_by_class[pred_box.class_name] += 1
            images_with_false_positive.add(stem)
            rows.append(
                {
                    "error_type": "false_positive",
                    "image_stem": stem,
                    "class_id": pred_box.class_id,
                    "class_name": pred_box.class_name,
                    "confidence": round(float(pred_box.confidence or 0.0), 6),
                    "iou": "",
                    "gt_xyxy": [],
                    "pred_xyxy": _box_to_list(pred_box),
                }
            )

    rows.sort(key=lambda item: (item["error_type"], item["image_stem"], item["class_name"]))
    total_gt = sum(total_gt_by_class.values())
    matched_gt = sum(matched_gt_by_class.values())
    total_predictions = sum(prediction_by_class.values())
    false_positive_count = sum(false_positive_by_class.values())
    summary = {
        "iou_threshold": iou_threshold,
        "low_confidence_threshold": low_confidence_threshold,
        "total_gt": total_gt,
        "matched_gt": matched_gt,
        "total_predictions": total_predictions,
        "missed_count": sum(missed_by_class.values()),
        "low_confidence_count": sum(low_confidence_by_class.values()),
        "false_positive_count": false_positive_count,
        "image_count_with_miss": len(images_with_miss),
        "image_count_with_low_confidence": len(images_with_low_confidence),
        "image_count_with_false_positive": len(images_with_false_positive),
        "recall": round(matched_gt / total_gt, 6) if total_gt else None,
        "precision": round((total_predictions - false_positive_count) / total_predictions, 6)
        if total_predictions
        else None,
        "total_gt_by_class": dict(sorted(total_gt_by_class.items())),
        "matched_gt_by_class": dict(sorted(matched_gt_by_class.items())),
        "missed_by_class": dict(sorted(missed_by_class.items())),
        "low_confidence_by_class": dict(sorted(low_confidence_by_class.items())),
        "false_positive_by_class": dict(sorted(false_positive_by_class.items())),
        "prediction_by_class": dict(sorted(prediction_by_class.items())),
        "recall_by_class": {
            class_name: round(matched_gt_by_class[class_name] / total_gt_by_class[class_name], 6)
            for class_name in sorted(total_gt_by_class)
            if total_gt_by_class[class_name] > 0
        },
    }
    return rows, summary


def write_error_mining_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "error_mining_report.json"
    csv_path = output_dir / "error_mining_report.csv"
    md_path = output_dir / "error_mining_report.md"

    json_path.write_text(
        json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "error_type",
                "image_stem",
                "class_id",
                "class_name",
                "confidence",
                "iou",
                "gt_xyxy",
                "pred_xyxy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "gt_xyxy": json.dumps(row["gt_xyxy"], ensure_ascii=False),
                    "pred_xyxy": json.dumps(row["pred_xyxy"], ensure_ascii=False),
                }
            )

    md_lines = [
        "# Detection error mining report",
        "",
        f"- Recall: {summary['recall']}",
        f"- Precision: {summary['precision']}",
        f"- Missed boxes: {summary['missed_count']}",
        f"- Low-confidence matched boxes: {summary['low_confidence_count']}",
        f"- False-positive boxes: {summary['false_positive_count']}",
        "",
        "## Recall by class",
        "",
        "| Class | GT | Matched | Missed | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for class_name, total_count in summary["total_gt_by_class"].items():
        matched_count = summary["matched_gt_by_class"].get(class_name, 0)
        missed_count = summary["missed_by_class"].get(class_name, 0)
        recall = summary["recall_by_class"].get(class_name)
        recall_text = "" if recall is None else f"{recall:.4f}"
        md_lines.append(f"| {class_name} | {total_count} | {matched_count} | {missed_count} | {recall_text} |")

    md_lines.extend(
        [
            "",
            "## False positives by class",
            "",
            "| Class | Count |",
            "| --- | ---: |",
        ]
    )
    for class_name, count in summary["false_positive_by_class"].items():
        md_lines.append(f"| {class_name} | {count} |")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
