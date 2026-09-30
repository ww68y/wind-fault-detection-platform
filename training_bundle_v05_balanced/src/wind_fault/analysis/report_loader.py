from __future__ import annotations

from collections import Counter
from typing import Any


CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}


def flatten_detections(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize image and video detections into one list."""
    detections: list[dict[str, Any]] = []
    mode = str(report.get("mode") or "single")

    if mode == "video":
        filename = str(report.get("uploaded_filename") or "-")
        for frame in report.get("frames", []) or []:
            for detection in frame.get("detections", []) or []:
                detections.append(
                    {
                        **detection,
                        "source_type": "video_frame",
                        "source_name": filename,
                        "frame_index": frame.get("frame_index"),
                        "time_seconds": frame.get("time_seconds"),
                    }
                )
        return detections

    for index, item in enumerate(report.get("items", []) or []):
        source_name = str(item.get("uploaded_filename") or item.get("image") or f"image_{index + 1}")
        for detection in item.get("detections", []) or []:
            detections.append(
                {
                    **detection,
                    "source_type": "image",
                    "source_name": source_name,
                    "image_index": index,
                }
            )
    return detections


def class_counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(item.get("class_name") or "unknown") for item in detections)
    return dict(sorted(counter.items()))


def class_counts_cn(counts: dict[str, int]) -> dict[str, int]:
    return {CLASS_LABELS.get(name, name): count for name, count in counts.items()}


def confidence_values(detections: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for item in detections:
        try:
            values.append(float(item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return values
