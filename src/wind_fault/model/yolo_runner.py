from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    return YOLO


def train_yolo(
    data: Path,
    model: str = "yolov8n.pt",
    imgsz: int = 640,
    epochs: int = 80,
    batch: int = 16,
    project: str = "runs/wind_fault",
    name: str = "baseline",
    device: str | None = None,
    workers: int = 8,
) -> Any:
    YOLO = _load_yolo()
    detector = YOLO(model)
    return detector.train(
        data=str(data),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=project,
        name=name,
        device=device,
        workers=workers,
    )


def predict_yolo(
    model: Path,
    source: Path | str,
    conf: float = 0.45,
    project: str = "runs/wind_fault",
    name: str = "predict",
    save: bool = True,
    device: str | None = None,
) -> Any:
    YOLO = _load_yolo()
    detector = YOLO(str(model))
    return detector.predict(
        source=str(source),
        conf=conf,
        project=project,
        name=name,
        save=save,
        device=device,
    )
