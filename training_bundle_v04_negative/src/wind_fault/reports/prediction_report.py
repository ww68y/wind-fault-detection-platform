from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}


def _serialize_one_result(result: Any) -> dict[str, Any]:
    boxes = []
    names = getattr(result, "names", {}) or {}
    result_path = str(getattr(result, "path", ""))
    result_boxes = getattr(result, "boxes", None)
    if result_boxes is not None:
        for box in result_boxes:
            class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else []
            class_name = str(names.get(class_id, str(class_id)))
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_name_cn": CLASS_LABELS.get(class_name, class_name),
                    "confidence": round(confidence, 6),
                    "xyxy": [round(float(value), 2) for value in xyxy],
                }
            )

    return {
        "image": result_path,
        "defect_count": len(boxes),
        "detections": boxes,
    }


def serialize_results(results: Any) -> dict[str, Any]:
    items = [_serialize_one_result(result) for result in list(results)]
    return {
        "image_count": len(items),
        "total_defects": sum(item["defect_count"] for item in items),
        "items": items,
    }


def write_prediction_report(results: Any, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = serialize_results(results)
    json_path = output_dir / "detection_report.json"
    md_path = output_dir / "detection_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(md_path)
    return payload


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 风机叶片缺陷检测报告",
        "",
        f"- 图片数量：{payload['image_count']}",
        f"- 缺陷总数：{payload['total_defects']}",
        "",
    ]
    for item in payload["items"]:
        lines.append(f"## {Path(item['image']).name}")
        lines.append("")
        lines.append(f"- 缺陷数量：{item['defect_count']}")
        if not item["detections"]:
            lines.append("- 未检测到缺陷")
        for detection in item["detections"]:
            class_name = detection["class_name"]
            class_name_cn = CLASS_LABELS.get(class_name, class_name)
            lines.append(
                f"- {class_name_cn}（{class_name}） | "
                f"置信度 {detection['confidence']:.3f} | "
                f"边框 {detection['xyxy']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
