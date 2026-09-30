from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    xmin = x_center - width / 2.0
    ymin = y_center - height / 2.0
    xmax = x_center + width / 2.0
    ymax = y_center + height / 2.0
    return xmin, ymin, xmax, ymax


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_class_names(data_yaml: Path) -> dict[int, str]:
    content = data_yaml.read_text(encoding="utf-8").splitlines()
    names: dict[int, str] = {}
    capture = False
    for raw_line in content:
        line = raw_line.rstrip()
        if line.strip() == "names:":
            capture = True
            continue
        if not capture:
            continue
        if not line.startswith("  "):
            break
        idx_str, name = line.strip().split(":", 1)
        names[int(idx_str.strip())] = name.strip()
    return names


def load_ground_truth(labels_dir: Path, images_dir: Path, class_names: dict[int, str]) -> dict[str, list[Box]]:
    ground_truth: dict[str, list[Box]] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = None
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            candidate = images_dir / f"{label_path.stem}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes: list[Box] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            xyxy = yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height)
            boxes.append(Box(class_id=class_id, class_name=class_names.get(class_id, str(class_id)), xyxy=xyxy))

        ground_truth[label_path.stem] = boxes
    return ground_truth


def load_predictions(report_path: Path) -> dict[str, list[Box]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    predictions: dict[str, list[Box]] = {}
    for item in payload.get("items", []):
        stem = Path(item["image"]).stem
        boxes = [
            Box(
                class_id=int(detection["class_id"]),
                class_name=str(detection["class_name"]),
                xyxy=tuple(map(float, detection["xyxy"])),
                confidence=float(detection["confidence"]),
            )
            for detection in item.get("detections", [])
        ]
        predictions[stem] = boxes
    return predictions


def analyze_misses(
    ground_truth: dict[str, list[Box]],
    predictions: dict[str, list[Box]],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missed_by_class: Counter[str] = Counter()
    total_gt_by_class: Counter[str] = Counter()
    complete_miss_images = 0

    for stem, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(stem, [])
        for gt_box in gt_boxes:
            total_gt_by_class[gt_box.class_name] += 1

        used_predictions: set[int] = set()
        missed_boxes: list[Box] = []

        for gt_box in gt_boxes:
            best_index = None
            best_iou = 0.0
            for idx, pred_box in enumerate(pred_boxes):
                if idx in used_predictions:
                    continue
                if pred_box.class_id != gt_box.class_id:
                    continue
                iou = box_iou(gt_box.xyxy, pred_box.xyxy)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_index = idx
            if best_index is None:
                missed_boxes.append(gt_box)
                missed_by_class[gt_box.class_name] += 1
            else:
                used_predictions.add(best_index)

        if not missed_boxes:
            continue

        gt_counter = Counter(box.class_name for box in gt_boxes)
        pred_counter = Counter(box.class_name for box in pred_boxes)
        missed_counter = Counter(box.class_name for box in missed_boxes)
        if pred_boxes == [] and gt_boxes:
            complete_miss_images += 1

        rows.append(
            {
                "image_stem": stem,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "missed_count": len(missed_boxes),
                "gt_classes": dict(sorted(gt_counter.items())),
                "pred_classes": dict(sorted(pred_counter.items())),
                "missed_classes": dict(sorted(missed_counter.items())),
                "complete_miss": int(not pred_boxes and bool(gt_boxes)),
            }
        )

    rows.sort(key=lambda item: (-item["missed_count"], -item["complete_miss"], item["image_stem"]))

    summary = {
        "image_count_with_miss": len(rows),
        "complete_miss_images": complete_miss_images,
        "missed_by_class": dict(sorted(missed_by_class.items())),
        "total_gt_by_class": dict(sorted(total_gt_by_class.items())),
        "miss_rate_by_class": {
            class_name: round(missed_by_class[class_name] / total_gt_by_class[class_name], 4)
            for class_name in sorted(total_gt_by_class)
            if total_gt_by_class[class_name] > 0
        },
    }
    return rows, summary


def write_reports(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], top_n: int = 50) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "missed_samples_report.json"
    csv_path = output_dir / "missed_samples_report.csv"
    md_path = output_dir / "missed_samples_report.md"

    json_payload = {"summary": summary, "items": rows}
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_stem",
                "gt_count",
                "pred_count",
                "missed_count",
                "complete_miss",
                "gt_classes",
                "pred_classes",
                "missed_classes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_stem": row["image_stem"],
                    "gt_count": row["gt_count"],
                    "pred_count": row["pred_count"],
                    "missed_count": row["missed_count"],
                    "complete_miss": row["complete_miss"],
                    "gt_classes": json.dumps(row["gt_classes"], ensure_ascii=False),
                    "pred_classes": json.dumps(row["pred_classes"], ensure_ascii=False),
                    "missed_classes": json.dumps(row["missed_classes"], ensure_ascii=False),
                }
            )

    md_lines = [
        "# 漏检样本清单",
        "",
        f"- 存在漏检的图片数量：{summary['image_count_with_miss']}",
        f"- 完全漏检图片数量：{summary['complete_miss_images']}",
        "",
        "## 类别漏检统计",
        "",
        "| 类别 | 真值数 | 漏检数 | 漏检率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for class_name, total_count in summary["total_gt_by_class"].items():
        missed_count = summary["missed_by_class"].get(class_name, 0)
        miss_rate = summary["miss_rate_by_class"].get(class_name, 0.0)
        md_lines.append(f"| {class_name} | {total_count} | {missed_count} | {miss_rate:.2%} |")

    md_lines.extend(
        [
            "",
            f"## Top {min(top_n, len(rows))} 漏检图片",
            "",
            "| 图片 | 真值框 | 预测框 | 漏检框 | 完全漏检 | 漏检类别 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows[:top_n]:
        md_lines.append(
            f"| {row['image_stem']} | {row['gt_count']} | {row['pred_count']} | {row['missed_count']} | "
            f"{'是' if row['complete_miss'] else '否'} | {json.dumps(row['missed_classes'], ensure_ascii=False)} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }

