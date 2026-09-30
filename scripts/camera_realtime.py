from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    return YOLO


def _serialize_detections(result: Any) -> list[dict[str, Any]]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    detections: list[dict[str, Any]] = []
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
        class_name = str(names.get(class_id, str(class_id)))
        confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
        xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else []
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "class_name_cn": CLASS_LABELS.get(class_name, class_name),
                "confidence": round(confidence, 6),
                "xyxy": [round(float(value), 2) for value in xyxy],
            }
        )
    return detections


def _risk_level(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return "NORMAL"
    classes = {item["class_name"] for item in detections}
    max_confidence = max(float(item["confidence"]) for item in detections)
    if "crack" in classes or len(detections) >= 3 or max_confidence >= 0.9:
        return "HIGH"
    return "MEDIUM"


def _save_snapshot(
    frame: Any,
    detections: list[dict[str, Any]],
    save_dir: Path,
    frame_index: int,
    risk: str,
) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = save_dir / f"snapshot_{timestamp}_frame_{frame_index:06d}.jpg"
    json_path = save_dir / f"snapshot_{timestamp}_frame_{frame_index:06d}.json"
    cv2.imwrite(str(image_path), frame)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_index": frame_index,
        "risk_level": risk,
        "defect_count": len(detections),
        "detections": detections,
        "image_path": str(image_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime wind blade defect detection from camera.")
    parser.add_argument("--model", required=True, type=Path, help="Trained YOLO model path.")
    parser.add_argument("--source", default="0", help="Camera index or video stream URL. Default: 0.")
    parser.add_argument("--conf", default=0.3, type=float, help="Confidence threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example 0 for GPU or cpu.")
    parser.add_argument("--save-dir", default="runs/camera_detect", type=Path, help="Snapshot output directory.")
    parser.add_argument("--snapshot-cooldown", default=3.0, type=float, help="Seconds between automatic snapshots.")
    parser.add_argument("--window-name", default="Wind Blade Realtime Detection")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    source: int | str
    source = int(args.source) if str(args.source).isdigit() else args.source

    YOLO = _load_yolo()
    detector = YOLO(str(args.model))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera or stream: {args.source}")

    run_dir = args.save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    previous_time = time.time()
    last_snapshot_time = 0.0
    class_counts: Counter[str] = Counter()

    print("Realtime detection started. Press q in the video window to exit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            results = detector.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            detections = _serialize_detections(result)
            for detection in detections:
                class_counts[detection["class_name"]] += 1

            risk = _risk_level(detections)
            annotated = result.plot()

            now = time.time()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            status = "DEFECT" if detections else "NORMAL"
            overlay = f"Status: {status} | Risk: {risk} | FPS: {fps:.1f} | Defects: {len(detections)}"
            cv2.putText(
                annotated,
                overlay,
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255) if detections else (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

            if risk == "HIGH" and now - last_snapshot_time >= args.snapshot_cooldown:
                image_path, json_path = _save_snapshot(annotated, detections, run_dir, frame_index, risk)
                last_snapshot_time = now
                print(f"Saved high-risk snapshot: {image_path}")
                print(f"Snapshot report: {json_path}")

            cv2.imshow(args.window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    summary_path = run_dir / "camera_session_summary.json"
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.source,
        "frames": frame_index,
        "class_summary": dict(sorted(class_counts.items())),
        "class_summary_cn": {
            CLASS_LABELS.get(name, name): count
            for name, count in sorted(class_counts.items())
        },
        "save_dir": str(run_dir),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Camera session summary: {summary_path}")


if __name__ == "__main__":
    main()
