from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image

from wind_fault.data.dataset import IMAGE_SUFFIXES, ensure_yolo_dirs, write_data_yaml, write_label_file


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def read_yolo_labels(path: Path, image_width: int, image_height: int) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        xmin = (x_center - width / 2.0) * image_width
        ymin = (y_center - height / 2.0) * image_height
        xmax = (x_center + width / 2.0) * image_width
        ymax = (y_center + height / 2.0) * image_height
        boxes.append((class_id, xmin, ymin, xmax, ymax))
    return boxes


def clip_box_to_tile(
    box: tuple[int, float, float, float, float],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float,
) -> tuple[int, float, float, float, float] | None:
    class_id, xmin, ymin, xmax, ymax = box
    original_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    if original_area <= 0:
        return None

    clipped_xmin = max(xmin, tile_x)
    clipped_ymin = max(ymin, tile_y)
    clipped_xmax = min(xmax, tile_x + tile_size)
    clipped_ymax = min(ymax, tile_y + tile_size)
    clipped_area = max(0.0, clipped_xmax - clipped_xmin) * max(0.0, clipped_ymax - clipped_ymin)
    if clipped_area / original_area < min_visibility:
        return None

    rel_xmin = clipped_xmin - tile_x
    rel_ymin = clipped_ymin - tile_y
    rel_xmax = clipped_xmax - tile_x
    rel_ymax = clipped_ymax - tile_y
    x_center = ((rel_xmin + rel_xmax) / 2.0) / tile_size
    y_center = ((rel_ymin + rel_ymax) / 2.0) / tile_size
    width = (rel_xmax - rel_xmin) / tile_size
    height = (rel_ymax - rel_ymin) / tile_size
    return class_id, x_center, y_center, width, height


def slice_dataset(
    images_dir: Path,
    output_dir: Path,
    split: str,
    labels_dir: Path | None = None,
    tile_size: int = 1024,
    overlap: float = 0.2,
    min_visibility: float = 0.25,
    class_names: list[str] | None = None,
) -> None:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 and 1")

    ensure_yolo_dirs(output_dir)
    stride = max(1, math.floor(tile_size * (1 - overlap)))
    image_paths = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            source_labels = []
            if labels_dir is not None:
                source_labels = read_yolo_labels(labels_dir / f"{image_path.stem}.txt", image_width, image_height)

            for y in tile_positions(image_height, tile_size, stride):
                for x in tile_positions(image_width, tile_size, stride):
                    tile = Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))
                    crop = image.crop((x, y, min(x + tile_size, image_width), min(y + tile_size, image_height)))
                    tile.paste(crop, (0, 0))

                    tile_stem = f"{image_path.stem}_x{x}_y{y}"
                    tile_path = output_dir / "images" / split / f"{tile_stem}.jpg"
                    tile.save(tile_path, quality=95)

                    tile_labels = [
                        clipped
                        for box in source_labels
                        if (clipped := clip_box_to_tile(box, x, y, tile_size, min_visibility)) is not None
                    ]
                    write_label_file(output_dir / "labels" / split / f"{tile_stem}.txt", tile_labels)

    write_data_yaml(output_dir, class_names or ["defect"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slice high-resolution images into fixed-size YOLO tiles.")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--labels", default=None, type=Path)
    parser.add_argument("--tile-size", default=1024, type=int)
    parser.add_argument("--overlap", default=0.2, type=float)
    parser.add_argument("--min-visibility", default=0.25, type=float)
    parser.add_argument("--classes", default="defect")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    slice_dataset(
        images_dir=args.images,
        output_dir=args.output,
        split=args.split,
        labels_dir=args.labels,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        class_names=[item.strip() for item in args.classes.split(",") if item.strip()],
    )


if __name__ == "__main__":
    main()

