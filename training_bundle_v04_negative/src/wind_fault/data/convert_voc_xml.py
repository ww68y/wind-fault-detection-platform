from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

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


def _image_size_from_xml(root: ET.Element, image_path: Path) -> tuple[int, int]:
    size = root.find("size")
    if size is not None:
        width = size.findtext("width")
        height = size.findtext("height")
        if width and height:
            return int(float(width)), int(float(height))

    with Image.open(image_path) as image:
        return image.size


def _find_image(images_dir: Path, stem: str) -> Path | None:
    matches = [path for path in image_files(images_dir) if path.stem == stem]
    return matches[0] if matches else None


def _collect_classes(annotation_paths: list[Path]) -> list[str]:
    names: set[str] = set()
    for xml_path in annotation_paths:
        tree = ET.parse(xml_path)
        for obj in tree.findall(".//object"):
            name = obj.findtext("name")
            if name:
                names.add(name.strip())
    return sorted(names) or ["defect"]


def convert_voc_dataset(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    classes: list[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    annotation_paths = sorted(annotations_dir.rglob("*.xml"))
    if not annotation_paths:
        raise FileNotFoundError(f"No VOC XML annotations found in {annotations_dir}")

    class_names = classes or _collect_classes(annotation_paths)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    ensure_yolo_dirs(output_dir)
    split_map = split_items(annotation_paths, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    for split, xml_paths in split_map.items():
        for xml_path in xml_paths:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_name = root.findtext("filename")
            image_path = images_dir / image_name if image_name else _find_image(images_dir, xml_path.stem)
            if image_path is None or not image_path.exists():
                raise FileNotFoundError(f"Image for {xml_path.name} was not found")

            width, height = _image_size_from_xml(root, image_path)
            labels: list[tuple[int, float, float, float, float]] = []

            for obj in root.findall(".//object"):
                name = (obj.findtext("name") or "defect").strip()
                if name not in class_to_id:
                    continue
                box = obj.find("bndbox")
                if box is None:
                    continue
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                labels.append((class_to_id[name], *voc_box_to_yolo(xmin, ymin, xmax, ymax, width, height)))

            copied = copy_image(image_path, output_dir / "images" / split)
            write_label_file(output_dir / "labels" / split / f"{copied.stem}.txt", labels)

    write_data_yaml(output_dir, class_names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to YOLO labels.")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing images.")
    parser.add_argument("--annotations", required=True, type=Path, help="Directory containing VOC XML files.")
    parser.add_argument("--output", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--classes", default=None, help="Comma-separated class names or a class file.")
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_voc_dataset(
        images_dir=args.images,
        annotations_dir=args.annotations,
        output_dir=args.output,
        classes=parse_classes(args.classes),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

