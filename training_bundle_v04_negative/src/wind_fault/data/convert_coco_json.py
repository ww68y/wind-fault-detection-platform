from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wind_fault.data.dataset import (
    copy_image,
    coco_box_to_yolo,
    ensure_yolo_dirs,
    parse_classes,
    write_data_yaml,
    write_label_file,
)


def _category_maps(payload: dict, classes: list[str] | None) -> tuple[list[str], dict[int, int]]:
    categories = payload.get("categories") or []
    if classes:
        class_names = classes
    elif categories:
        class_names = [str(item.get("name", f"class_{idx}")) for idx, item in enumerate(categories)]
    else:
        class_names = ["defect"]

    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    category_to_class: dict[int, int] = {}
    for category in categories:
        category_id = int(category.get("id", len(category_to_class)))
        name = str(category.get("name", "defect"))
        category_to_class[category_id] = name_to_id.get(name, 0)

    if not category_to_class:
        category_to_class[0] = 0
        category_to_class[1] = 0

    return class_names, category_to_class


def _find_image(images_dir: Path, file_name: str) -> Path:
    direct = images_dir / file_name
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image {file_name} was not found under {images_dir}")


def convert_coco_json(
    annotation_path: Path,
    images_dir: Path,
    output_dir: Path,
    split: str,
    classes: list[str] | None = None,
) -> None:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    if not images:
        raise ValueError(f"No images found in {annotation_path}")

    class_names, category_to_class = _category_maps(payload, classes)
    ensure_yolo_dirs(output_dir)

    for image_id, image_info in images.items():
        file_name = image_info.get("file_name")
        if not file_name:
            continue
        image_width = float(image_info.get("width", 0))
        image_height = float(image_info.get("height", 0))
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Image {file_name} is missing width or height")

        labels: list[tuple[int, float, float, float, float]] = []
        for annotation in annotations_by_image.get(image_id, []):
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            category_id = int(annotation.get("category_id", 0))
            labels.append(
                (
                    category_to_class.get(category_id, 0),
                    *coco_box_to_yolo(*map(float, bbox), image_width, image_height),
                )
            )

        image_path = _find_image(images_dir, file_name)
        copied = copy_image(image_path, output_dir / "images" / split)
        write_label_file(output_dir / "labels" / split / f"{copied.stem}.txt", labels)

    write_data_yaml(output_dir, class_names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert COCO-style JSON annotations to YOLO labels.")
    parser.add_argument("--annotation", required=True, type=Path, help="COCO/DTU annotation JSON.")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing images.")
    parser.add_argument("--output", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--classes", default=None, help="Comma-separated class names or a class file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_coco_json(
        annotation_path=args.annotation,
        images_dir=args.images,
        output_dir=args.output,
        split=args.split,
        classes=parse_classes(args.classes),
    )


if __name__ == "__main__":
    main()

