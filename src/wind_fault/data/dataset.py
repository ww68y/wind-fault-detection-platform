from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return tuple(clamp01(v) for v in (x_center, y_center, width, height))


def coco_box_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    return voc_box_to_yolo(x, y, x + width, y + height, image_width, image_height)


def ensure_yolo_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def split_items(
    items: Sequence[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_image(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def write_label_file(path: Path, labels: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for class_id, x, y, w, h in labels
        if w > 0 and h > 0
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(root: Path, class_names: Sequence[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    content = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names}\n"
    )
    (root / "data.yaml").write_text(content, encoding="utf-8")


def parse_classes(value: str | None) -> list[str]:
    if not value:
        return []
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]

