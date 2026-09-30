from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_fault.data.dataset import (
    copy_image,
    ensure_yolo_dirs,
    image_files,
    parse_classes,
    split_items,
    voc_box_to_yolo,
    write_data_yaml,
    write_label_file,
)


def _shape_box(points: list[list[float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _find_image(images_dir: Path, image_path_value: str | None, annotation_path: Path) -> Path:
    candidates: list[str] = []
    if image_path_value:
        candidates.append(Path(image_path_value).name)
    candidates.extend(
        [
            annotation_path.name.replace("_labelme.json", ".jpg"),
            f"{annotation_path.stem}.jpg",
            f"{annotation_path.stem}.png",
        ]
    )

    for candidate in candidates:
        direct = images_dir / candidate
        if direct.exists():
            return direct

    stems = {Path(candidate).stem for candidate in candidates}
    for path in image_files(images_dir):
        if path.stem in stems:
            return path

    raise FileNotFoundError(f"Image for {annotation_path.name} was not found under {images_dir}")


def _collect_classes(annotation_paths: list[Path], ignored: set[str]) -> list[str]:
    labels: set[str] = set()
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for shape in payload.get("shapes", []):
            label = str(shape.get("label", "defect")).strip()
            if label and label not in ignored:
                labels.add(label)
    return sorted(labels) or ["defect"]


def convert_labelme_dataset(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    classes: list[str] | None = None,
    ignore_classes: list[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    annotation_paths = sorted(annotations_dir.rglob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No LabelMe JSON annotations found in {annotations_dir}")

    ignored = set(ignore_classes or [])
    class_names = classes or _collect_classes(annotation_paths, ignored)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    ensure_yolo_dirs(output_dir)
    split_map = split_items(annotation_paths, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    for split, paths in split_map.items():
        for annotation_path in paths:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_width = float(payload.get("imageWidth", 0))
            image_height = float(payload.get("imageHeight", 0))
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"{annotation_path.name} is missing imageWidth or imageHeight")

            image_path = _find_image(images_dir, payload.get("imagePath"), annotation_path)
            labels: list[tuple[int, float, float, float, float]] = []

            for shape in payload.get("shapes", []):
                label = str(shape.get("label", "defect")).strip()
                if label in ignored or label not in class_to_id:
                    continue
                box = _shape_box(shape.get("points", []))
                if box is None:
                    continue
                xmin, ymin, xmax, ymax = box
                labels.append((class_to_id[label], *voc_box_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height)))

            copied = copy_image(image_path, output_dir / "images" / split)
            write_label_file(output_dir / "labels" / split / f"{copied.stem}.txt", labels)

    write_data_yaml(output_dir, class_names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations to YOLO labels.")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing images.")
    parser.add_argument("--annotations", required=True, type=Path, help="Directory containing LabelMe JSON files.")
    parser.add_argument("--output", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--classes", default=None, help="Comma-separated class names or a class file.")
    parser.add_argument("--ignore-classes", default="", help="Comma-separated labels to ignore, for example blade,ff.")
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_labelme_dataset(
        images_dir=args.images,
        annotations_dir=args.annotations,
        output_dir=args.output,
        classes=parse_classes(args.classes),
        ignore_classes=parse_classes(args.ignore_classes),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

