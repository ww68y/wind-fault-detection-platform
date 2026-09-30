from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    return YOLO


def _video_meta(source: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return {"total_frames": 0, "fps": 0.0, "duration_seconds": 0.0}

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    duration = round(total_frames / fps, 3) if fps > 0 and total_frames > 0 else 0.0
    return {
        "total_frames": total_frames,
        "fps": round(fps, 3),
        "duration_seconds": duration,
    }


def _serialize_box(box: Any, names: dict[int, str]) -> dict[str, Any]:
    class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
    class_name = str(names.get(class_id, str(class_id)))
    confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
    xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else []
    return {
        "class_id": class_id,
        "class_name": class_name,
        "class_name_cn": CLASS_LABELS.get(class_name, class_name),
        "confidence": round(confidence, 6),
        "xyxy": [round(float(value), 2) for value in xyxy],
    }


def _risk_level(class_counts: Counter[str], defect_frame_count: int, max_confidence: float) -> tuple[str, str, list[str]]:
    if not class_counts:
        return (
            "正常",
            "当前视频在设定阈值下未检测到缺陷。",
            [
                "可保留该视频作为正常巡检记录。",
                "如果视频画面模糊或光照较差，建议补拍关键区域。",
            ],
        )

    if "crack" in class_counts or defect_frame_count >= 5 or max_confidence >= 0.9:
        return (
            "高",
            "视频中检测到高风险或多帧连续缺陷，建议人工复核关键帧并形成巡检记录。",
            [
                "优先查看关键帧截图，确认缺陷类型、位置和范围。",
                "裂纹、腐蚀或连续出现的缺陷建议形成维修工单。",
                "必要时补拍局部高清图片再进行二次检测。",
            ],
        )

    return (
        "中",
        "视频中检测到疑似缺陷，建议结合关键帧截图进行人工复核。",
        [
            "检查检测框是否稳定出现在同一位置。",
            "如果只在单帧出现，需排除运动模糊、反光或背景干扰。",
        ],
    )


def _find_output_video(run_dir: Path, source: Path) -> str | None:
    candidates = []
    if source.suffix.lower() in VIDEO_SUFFIXES:
        candidates.append(run_dir / source.name)
        candidates.extend(run_dir.glob(f"{source.stem}.*"))
    candidates.extend(path for path in run_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# 风机叶片视频缺陷检测报告",
        "",
        f"- 视频文件：{payload['video_filename']}",
        f"- 检测时间：{payload['created_at']}",
        f"- 置信度阈值：{payload['confidence_threshold']}",
        f"- 总帧数：{payload['total_frames']}",
        f"- 实际处理帧数：{payload['processed_frames']}",
        f"- 存在缺陷的帧数：{payload['defect_frame_count']}",
        f"- 缺陷总数：{payload['total_defects']}",
        f"- 风险等级：{payload['risk_level']}",
        "",
        "## 处理建议",
        "",
        payload["advice"],
        "",
    ]
    for action in payload["recommended_actions"]:
        lines.append(f"- {action}")

    lines.extend(["", "## 缺陷统计", ""])
    if payload["class_summary"]:
        for name, count in payload["class_summary"].items():
            lines.append(f"- {CLASS_LABELS.get(name, name)}（{name}）：{count}")
    else:
        lines.append("- 无")

    lines.extend(["", "## 关键帧", ""])
    if payload["keyframes"]:
        for keyframe in payload["keyframes"]:
            lines.append(
                f"- 第 {keyframe['frame_index']} 帧，"
                f"{keyframe['time_seconds']} 秒，"
                f"缺陷数量 {keyframe['defect_count']}，"
                f"截图：{keyframe['image_path']}"
            )
    else:
        lines.append("- 未生成关键帧")

    lines.extend(["", "## 帧级检测明细", ""])
    if payload["frames"]:
        for frame in payload["frames"]:
            lines.append(f"### 第 {frame['frame_index']} 帧")
            for detection in frame["detections"]:
                lines.append(
                    f"- {detection['class_name_cn']}（{detection['class_name']}） | "
                    f"置信度 {detection['confidence']:.3f} | "
                    f"边框 {detection['xyxy']}"
                )
            lines.append("")
    else:
        lines.append("未检测到缺陷。")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _keyframe_candidate_score(frame_payload: dict[str, Any]) -> tuple[int, float]:
    detections = frame_payload.get("detections", []) or []
    max_confidence = max((float(item.get("confidence", 0.0)) for item in detections), default=0.0)
    return (int(frame_payload.get("defect_count", 0)), max_confidence)


def _should_save_keyframe(
    frame_payload: dict[str, Any],
    keyframes: list[dict[str, Any]],
    max_keyframes: int,
    fps: float,
) -> bool:
    if len(keyframes) >= max_keyframes:
        return False
    if not keyframes:
        return True

    current_classes = {
        str(item.get("class_name", "unknown"))
        for item in frame_payload.get("detections", []) or []
    }
    seen_classes = {
        str(item.get("class_name", "unknown"))
        for keyframe in keyframes
        for item in keyframe.get("detections", []) or []
    }
    if current_classes - seen_classes:
        return True

    gap = max(5, int(fps // 2) if fps > 0 else 5)
    last_frame_index = int(keyframes[-1].get("frame_index", 0) or 0)
    frame_index = int(frame_payload.get("frame_index", 0) or 0)
    if frame_index - last_frame_index >= gap:
        return True

    best_count, best_confidence = max((_keyframe_candidate_score(item) for item in keyframes), default=(0, 0.0))
    defect_count, max_confidence = _keyframe_candidate_score(frame_payload)
    return defect_count > best_count or max_confidence > best_confidence + 0.1


def detect_video(
    model: Path,
    source: Path,
    output_root: Path = Path("runs/video_detect"),
    name: str | None = None,
    conf: float = 0.3,
    device: str | None = None,
    vid_stride: int = 1,
    max_keyframes: int = 12,
) -> dict[str, Any]:
    """Run YOLO video detection and write video/keyframe/report artifacts."""
    model = model.resolve()
    source = source.resolve()
    output_root = output_root.resolve()
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    YOLO = _load_yolo()
    detector = YOLO(str(model))
    name = name or f"{source.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / name
    keyframe_dir = run_dir / "keyframes"
    report_dir = run_dir / "reports"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = detector.predict(
        source=str(source),
        conf=conf,
        project=str(output_root),
        name=name,
        save=True,
        device=device,
        vid_stride=vid_stride,
        exist_ok=True,
        stream=True,
    )

    meta = _video_meta(source)
    class_counts: Counter[str] = Counter()
    max_confidence_by_class: defaultdict[str, float] = defaultdict(float)
    frames: list[dict[str, Any]] = []
    keyframes: list[dict[str, Any]] = []
    max_confidence = 0.0
    fps = float(meta.get("fps", 0.0) or 0.0)
    processed_frames = 0

    for result_index, result in enumerate(results):
        processed_frames += 1
        frame_index = result_index * max(1, vid_stride)
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        detections = []
        if boxes is not None:
            for box in boxes:
                detection = _serialize_box(box, names)
                detections.append(detection)
                class_name = detection["class_name"]
                class_counts[class_name] += 1
                max_confidence = max(max_confidence, float(detection["confidence"]))
                max_confidence_by_class[class_name] = max(
                    max_confidence_by_class[class_name],
                    float(detection["confidence"]),
                )

        if not detections:
            continue

        time_seconds = round(frame_index / fps, 3) if fps > 0 else None
        frame_payload = {
            "frame_index": frame_index,
            "time_seconds": time_seconds,
            "defect_count": len(detections),
            "detections": detections,
        }
        frames.append(frame_payload)

        if _should_save_keyframe(frame_payload, keyframes, max_keyframes, fps):
            keyframe_path = keyframe_dir / f"frame_{frame_index:06d}.jpg"
            plotted = result.plot()
            cv2.imwrite(str(keyframe_path), plotted)
            max_frame_confidence = max(
                (float(item.get("confidence", 0.0)) for item in detections),
                default=0.0,
            )
            frame_classes = Counter(str(item.get("class_name", "unknown")) for item in detections)
            keyframes.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": time_seconds,
                    "defect_count": len(detections),
                    "class_summary": dict(sorted(frame_classes.items())),
                    "max_confidence": round(max_frame_confidence, 6),
                    "detections": detections,
                    "image_path": str(keyframe_path),
                }
            )

    risk_level, advice, recommended_actions = _risk_level(
        class_counts=class_counts,
        defect_frame_count=len(frames),
        max_confidence=max_confidence,
    )

    output_video = _find_output_video(run_dir, source)
    payload: dict[str, Any] = {
        "video_filename": source.name,
        "video_path": str(source),
        "output_video": output_video,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "confidence_threshold": conf,
        "device": device,
        "vid_stride": vid_stride,
        "total_frames": meta["total_frames"],
        "fps": meta["fps"],
        "duration_seconds": meta["duration_seconds"],
        "processed_frames": processed_frames,
        "defect_frame_count": len(frames),
        "total_defects": sum(class_counts.values()),
        "class_summary": dict(sorted(class_counts.items())),
        "class_summary_cn": {
            CLASS_LABELS.get(name, name): count
            for name, count in sorted(class_counts.items())
        },
        "max_confidence_by_class": {
            name: round(value, 6)
            for name, value in sorted(max_confidence_by_class.items())
        },
        "risk_level": risk_level,
        "advice": advice,
        "recommended_actions": recommended_actions,
        "keyframes": keyframes,
        "frames": frames,
    }

    json_path = report_dir / "video_detection_report.json"
    md_path = report_dir / "video_detection_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(md_path)
    return payload
