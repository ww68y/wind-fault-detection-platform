from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGET_NAMES = {
    0: "crack",
    1: "hole",
    2: "spalling",
    3: "corrosion",
}
TARGET_IDS = {name: idx for idx, name in TARGET_NAMES.items()}

DEFECT_MAP = {
    "crack": "crack",
    "craze": "crack",
    "hide_craze": "crack",
    "hidecraze": "crack",
    "scratch": "crack",
    "corrosion": "corrosion",
    "erosion": "corrosion",
    "damage": "spalling",
    "mechanicaldamage": "spalling",
    "mechanical_damage": "spalling",
    "paintoff": "spalling",
    "paint_off": "spalling",
    "paint_peel": "spalling",
    "surface_injure": "spalling",
    "surfaceinjure": "spalling",
    "chipping": "spalling",
    "spalling": "spalling",
    "puncture": "hole",
    "hole": "hole",
}

NEGATIVE_NAMES = {"good", "normal", "dirt"}
UNSURE_NAMES = {"object", "thunderstrike"}


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def safe_stem(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in {"_", "-"} else "_")
    return "".join(keep).strip("_") or "sample"


def unique_destination(images_dir: Path, prefix: str, src: Path) -> tuple[Path, Path]:
    base = safe_stem(f"{prefix}_{src.stem}")
    suffix = src.suffix.lower()
    candidate = images_dir / f"{base}{suffix}"
    index = 1
    while candidate.exists():
        candidate = images_dir / f"{base}_{index}{suffix}"
        index += 1
    label = images_dir.parent.parent / "labels" / images_dir.name / f"{candidate.stem}.txt"
    return candidate, label


def write_yolo_label(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_image_and_label(src_image: Path, dst_image: Path, dst_label: Path, boxes: list[YoloBox]) -> None:
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)
    write_yolo_label(dst_label, boxes)


def load_yolo_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    raise ValueError(f"Unsupported names in {data_yaml}")


def parse_yolo_label(label_path: Path, class_names: dict[int, str]) -> tuple[list[YoloBox], list[str], list[str], bool]:
    boxes: list[YoloBox] = []
    original_names: list[str] = []
    ignored_names: list[str] = []
    saw_negative_only_name = False

    if not label_path.exists():
        return boxes, original_names, ignored_names, False

    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:5]]
        except ValueError:
            continue

        class_name = class_names.get(class_id, str(class_id))
        original_names.append(class_name)
        norm = normalize_name(class_name)

        if norm in NEGATIVE_NAMES:
            saw_negative_only_name = True
            continue
        if norm in UNSURE_NAMES or norm not in DEFECT_MAP:
            ignored_names.append(class_name)
            continue

        if any(value < 0 or value > 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            ignored_names.append(class_name)
            continue

        target_name = DEFECT_MAP[norm]
        boxes.append(YoloBox(TARGET_IDS[target_name], *values))

    return boxes, original_names, ignored_names, saw_negative_only_name


def parse_t6_class_names(labels_dir: Path, root: Path) -> dict[int, str]:
    candidates = [
        labels_dir / "labels.txt",
        root / "labels.txt",
        root / "classes.txt",
        root / "obj.names",
    ]
    for candidate in candidates:
        if candidate.exists():
            names = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            return {idx: name for idx, name in enumerate(names)}
    return {0: "dirt", 1: "damage"}


def find_t6_yolo_root(t6_dir: Path) -> tuple[Path, Path, Path]:
    for images_dir in sorted(t6_dir.rglob("images")):
        labels_dir = images_dir.parent / "labels"
        if labels_dir.exists():
            return images_dir.parent, images_dir, labels_dir
    raise FileNotFoundError(f"Could not find images/labels under {t6_dir}")


def voc_box_to_yolo(bndbox: ET.Element, width: int, height: int) -> YoloBox | None:
    try:
        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))
    except ValueError:
        return None

    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))
    if xmax <= xmin or ymax <= ymin or width <= 0 or height <= 0:
        return None

    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    return YoloBox(-1, x_center, y_center, box_width, box_height)


def parse_voc_xml(xml_path: Path) -> tuple[list[YoloBox], list[str], list[str]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0

    boxes: list[YoloBox] = []
    original_names: list[str] = []
    ignored_names: list[str] = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip()
        if not class_name:
            continue
        original_names.append(class_name)
        norm = normalize_name(class_name)
        if norm in UNSURE_NAMES or norm not in DEFECT_MAP:
            ignored_names.append(class_name)
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            ignored_names.append(class_name)
            continue
        yolo_box = voc_box_to_yolo(bndbox, width, height)
        if yolo_box is None:
            ignored_names.append(class_name)
            continue
        yolo_box.class_id = TARGET_IDS[DEFECT_MAP[norm]]
        boxes.append(yolo_box)

    return boxes, original_names, ignored_names


def append_manifest(
    rows: list[dict[str, str]],
    *,
    source_dataset: str,
    source_image: Path,
    source_annotation: Path | None,
    source_format: str,
    dest_split: str,
    dest_image: Path | None,
    dest_label: Path | None,
    boxes: list[YoloBox],
    original_names: list[str],
    reason: str,
) -> None:
    rows.append(
        {
            "source_dataset": source_dataset,
            "source_image_path": str(source_image),
            "source_annotation_path": str(source_annotation) if source_annotation else "",
            "source_format": source_format,
            "dest_split": dest_split,
            "dest_image_path": str(dest_image) if dest_image else "",
            "dest_label_path": str(dest_label) if dest_label else "",
            "mapped_classes": "|".join(sorted({TARGET_NAMES[b.class_id] for b in boxes})),
            "original_classes": "|".join(original_names),
            "reason": reason,
        }
    )


def copy_base_dataset(base_dir: Path, output_dir: Path, rows: list[dict[str, str]], stats: Counter) -> None:
    for split in ["train", "val", "test_clean"]:
        src_images = base_dir / "images" / split
        src_labels = base_dir / "labels" / split
        dst_images = output_dir / "images" / split
        dst_labels = output_dir / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for src_image in image_files(src_images):
            rel = src_image.relative_to(src_images)
            dst_image = dst_images / rel
            dst_label = dst_labels / f"{rel.with_suffix('').as_posix()}.txt"
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)
            src_label = src_labels / f"{rel.with_suffix('').as_posix()}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                dst_label.write_text("", encoding="utf-8")
            stats[f"base_{split}_images"] += 1
            append_manifest(
                rows,
                source_dataset="v04_negative_base",
                source_image=src_image,
                source_annotation=src_label if src_label.exists() else None,
                source_format="yolo",
                dest_split=split,
                dest_image=dst_image,
                dest_label=dst_label,
                boxes=[],
                original_names=[],
                reason="copied_from_v04_base",
            )


def add_yolo_dataset(
    *,
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    stats: Counter,
    skip_negative_only: bool,
) -> None:
    class_names = load_yolo_names(dataset_dir / "data.yaml")
    for split in ["train", "valid", "test"]:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for src_image in image_files(images_dir):
            stats[f"{dataset_name}_seen_images"] += 1
            label_path = labels_dir / f"{src_image.stem}.txt"
            boxes, original_names, ignored_names, saw_negative_name = parse_yolo_label(label_path, class_names)
            for name in original_names:
                stats[f"original_class:{name}"] += 1

            if not label_path.exists():
                stats[f"{dataset_name}_skipped_missing_label"] += 1
                append_manifest(
                    rows,
                    source_dataset=dataset_name,
                    source_image=src_image,
                    source_annotation=None,
                    source_format="yolo",
                    dest_split="",
                    dest_image=None,
                    dest_label=None,
                    boxes=[],
                    original_names=original_names,
                    reason="skipped_missing_label",
                )
                continue

            if not boxes and saw_negative_name and not ignored_names:
                if skip_negative_only:
                    stats[f"{dataset_name}_skipped_negative_only"] += 1
                    append_manifest(
                        rows,
                        source_dataset=dataset_name,
                        source_image=src_image,
                        source_annotation=label_path,
                        source_format="yolo",
                        dest_split="",
                        dest_image=None,
                        dest_label=None,
                        boxes=[],
                        original_names=original_names,
                        reason="skipped_negative_only_already_covered",
                    )
                    continue
                reason = "added_negative_only"
            elif boxes:
                reason = "added_mapped_defect"
            else:
                stats[f"{dataset_name}_skipped_unmapped"] += 1
                append_manifest(
                    rows,
                    source_dataset=dataset_name,
                    source_image=src_image,
                    source_annotation=label_path,
                    source_format="yolo",
                    dest_split="",
                    dest_image=None,
                    dest_label=None,
                    boxes=[],
                    original_names=original_names,
                    reason="skipped_unmapped_or_unsure",
                )
                continue

            dst_image, dst_label = unique_destination(output_dir / "images" / "train", f"{dataset_name}_{split}", src_image)
            copy_image_and_label(src_image, dst_image, dst_label, boxes)
            stats[f"{dataset_name}_added_images"] += 1
            stats[f"added_reason:{reason}"] += 1
            for box in boxes:
                stats[f"target_object:{TARGET_NAMES[box.class_id]}"] += 1
            append_manifest(
                rows,
                source_dataset=dataset_name,
                source_image=src_image,
                source_annotation=label_path,
                source_format="yolo",
                dest_split="train",
                dest_image=dst_image,
                dest_label=dst_label,
                boxes=boxes,
                original_names=original_names,
                reason=reason,
            )


def stable_image_order(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8", errors="ignore")).hexdigest()


def add_t6_dataset(
    t6_dir: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    stats: Counter,
    max_damage_images: int = 0,
    max_negative_images: int = 0,
) -> None:
    root, images_dir, labels_dir = find_t6_yolo_root(t6_dir)
    class_names = parse_t6_class_names(labels_dir, root)
    added_damage_images = 0
    added_negative_images = 0
    for src_image in sorted(image_files(images_dir), key=stable_image_order):
        stats["t6_seen_images"] += 1
        label_path = labels_dir / f"{src_image.stem}.txt"
        boxes, original_names, ignored_names, saw_negative_name = parse_yolo_label(label_path, class_names)
        for name in original_names:
            stats[f"original_class:{name}"] += 1

        if not label_path.exists():
            stats["t6_skipped_missing_label"] += 1
            append_manifest(
                rows,
                source_dataset="t6fwpc735s",
                source_image=src_image,
                source_annotation=None,
                source_format="yolo",
                dest_split="",
                dest_image=None,
                dest_label=None,
                boxes=[],
                original_names=[],
                reason="skipped_unannotated_conservative",
            )
            continue

        if boxes:
            reason = "added_mapped_damage"
        elif saw_negative_name and not ignored_names:
            reason = "added_dirt_negative"
        else:
            stats["t6_skipped_unmapped"] += 1
            append_manifest(
                rows,
                source_dataset="t6fwpc735s",
                source_image=src_image,
                source_annotation=label_path,
                source_format="yolo",
                dest_split="",
                dest_image=None,
                dest_label=None,
                boxes=[],
                original_names=original_names,
                reason="skipped_unmapped_or_empty_label",
            )
            continue

        if reason == "added_mapped_damage" and max_damage_images and added_damage_images >= max_damage_images:
            stats["t6_skipped_damage_cap"] += 1
            append_manifest(
                rows,
                source_dataset="t6fwpc735s",
                source_image=src_image,
                source_annotation=label_path,
                source_format="yolo",
                dest_split="",
                dest_image=None,
                dest_label=None,
                boxes=[],
                original_names=original_names,
                reason="skipped_damage_cap",
            )
            continue
        if reason == "added_dirt_negative" and max_negative_images and added_negative_images >= max_negative_images:
            stats["t6_skipped_negative_cap"] += 1
            append_manifest(
                rows,
                source_dataset="t6fwpc735s",
                source_image=src_image,
                source_annotation=label_path,
                source_format="yolo",
                dest_split="",
                dest_image=None,
                dest_label=None,
                boxes=[],
                original_names=original_names,
                reason="skipped_negative_cap",
            )
            continue

        dst_image, dst_label = unique_destination(output_dir / "images" / "train", "t6fwpc735s", src_image)
        copy_image_and_label(src_image, dst_image, dst_label, boxes)
        stats["t6_added_images"] += 1
        if reason == "added_mapped_damage":
            added_damage_images += 1
            stats["t6_added_damage_images"] += 1
        elif reason == "added_dirt_negative":
            added_negative_images += 1
            stats["t6_added_negative_images"] += 1
        stats[f"added_reason:{reason}"] += 1
        for box in boxes:
            stats[f"target_object:{TARGET_NAMES[box.class_id]}"] += 1
        append_manifest(
            rows,
            source_dataset="t6fwpc735s",
            source_image=src_image,
            source_annotation=label_path,
            source_format="yolo",
            dest_split="train",
            dest_image=dst_image,
            dest_label=dst_label,
            boxes=boxes,
            original_names=original_names,
            reason=reason,
        )


def add_wt_voc_dataset(wt_dir: Path, output_dir: Path, rows: list[dict[str, str]], stats: Counter) -> None:
    images_dir = wt_dir / "JPEGImages"
    xml_dir = wt_dir / "Annotations"
    image_by_stem = {p.stem: p for p in image_files(images_dir)}

    for xml_path in sorted(xml_dir.glob("*.xml")):
        stats["wt_seen_xml"] += 1
        boxes, original_names, ignored_names = parse_voc_xml(xml_path)
        for name in original_names:
            stats[f"original_class:{name}"] += 1
        src_image = image_by_stem.get(xml_path.stem)
        if src_image is None:
            stats["wt_skipped_missing_image"] += 1
            append_manifest(
                rows,
                source_dataset="wt_blade_defect",
                source_image=xml_path,
                source_annotation=xml_path,
                source_format="voc_xml",
                dest_split="",
                dest_image=None,
                dest_label=None,
                boxes=[],
                original_names=original_names,
                reason="skipped_missing_image",
            )
            continue
        if not boxes:
            stats["wt_skipped_unmapped"] += 1
            append_manifest(
                rows,
                source_dataset="wt_blade_defect",
                source_image=src_image,
                source_annotation=xml_path,
                source_format="voc_xml",
                dest_split="",
                dest_image=None,
                dest_label=None,
                boxes=[],
                original_names=original_names,
                reason="skipped_unmapped_or_thunderstrike",
            )
            continue

        dst_image, dst_label = unique_destination(output_dir / "images" / "train", "wt_blade_defect", src_image)
        copy_image_and_label(src_image, dst_image, dst_label, boxes)
        stats["wt_added_images"] += 1
        stats["added_reason:added_wt_voc"] += 1
        for box in boxes:
            stats[f"target_object:{TARGET_NAMES[box.class_id]}"] += 1
        append_manifest(
            rows,
            source_dataset="wt_blade_defect",
            source_image=src_image,
            source_annotation=xml_path,
            source_format="voc_xml",
            dest_split="train",
            dest_image=dst_image,
            dest_label=dst_label,
            boxes=boxes,
            original_names=original_names,
            reason="added_wt_voc",
        )


def write_data_yaml(output_dir: Path) -> None:
    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test_clean\n\n"
        "names:\n"
        "  0: crack\n"
        "  1: hole\n"
        "  2: spalling\n"
        "  3: corrosion\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def count_output_dataset(output_dir: Path, stats: Counter) -> None:
    for split in ["train", "val", "test_clean"]:
        images = list(image_files(output_dir / "images" / split))
        labels = list((output_dir / "labels" / split).glob("*.txt"))
        stats[f"output_{split}_images"] = len(images)
        stats[f"output_{split}_labels"] = len(labels)
        empty_labels = 0
        for label in labels:
            if not label.read_text(encoding="utf-8", errors="ignore").strip():
                empty_labels += 1
        stats[f"output_{split}_empty_labels"] = empty_labels


def write_reports(output_dir: Path, rows: list[dict[str, str]], stats: Counter) -> None:
    count_output_dataset(output_dir, stats)

    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "source_dataset",
        "source_image_path",
        "source_annotation_path",
        "source_format",
        "dest_split",
        "dest_image_path",
        "dest_label_path",
        "mapped_classes",
        "original_classes",
        "reason",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"{output_dir.name} dataset summary", ""]
    for key in sorted(stats):
        lines.append(f"{key}: {stats[key]}")
    lines.extend(
        [
            "",
            "Class mapping:",
            "Crack/Scratch/Craze/Hide_craze -> crack",
            "Erosion/Corrosion -> corrosion",
            "Damage/MechanicalDamage/Paintoff/Surface_injure/Chipping -> spalling",
            "Puncture/Hole -> hole",
            "Good/Dirt-only -> negative empty label, except Roboflow Good-only is skipped because v04 already includes curated negatives",
            "object/thunderstrike/unannotated -> skipped in this conservative version",
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build conservative v05 YOLO dataset from new wind blade datasets.")
    parser.add_argument("--base_dataset", type=Path, default=Path("datasets/wind_blade_defect_v04_negative"))
    parser.add_argument("--roboflow_dir", type=Path, default=Path(r"C:\Wind Turbine blades.v2-1111.yolov8"))
    parser.add_argument("--t6_dir", type=Path, default=Path(r"C:\t6fwpc735s-1"))
    parser.add_argument("--wt_dir", type=Path, default=Path(r"C:\WT blade defect dataset\WT blade defect dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/wind_blade_defect_v05_conservative"))
    parser.add_argument("--max_t6_damage_images", type=int, default=0, help="0 means use all labeled t6 damage images.")
    parser.add_argument("--max_t6_negative_images", type=int, default=0, help="0 means use all labeled t6 dirt-only images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Move or delete it before rebuilding.")

    stats: Counter = Counter()
    rows: list[dict[str, str]] = []

    copy_base_dataset(args.base_dataset, output_dir, rows, stats)
    if args.roboflow_dir.exists():
        add_yolo_dataset(
            dataset_name="roboflow_wind_turbine_blades",
            dataset_dir=args.roboflow_dir,
            output_dir=output_dir,
            rows=rows,
            stats=stats,
            skip_negative_only=True,
        )
    else:
        stats["roboflow_missing"] += 1

    if args.t6_dir.exists():
        add_t6_dataset(
            args.t6_dir,
            output_dir,
            rows,
            stats,
            max_damage_images=args.max_t6_damage_images,
            max_negative_images=args.max_t6_negative_images,
        )
    else:
        stats["t6_missing"] += 1

    if args.wt_dir.exists():
        add_wt_voc_dataset(args.wt_dir, output_dir, rows, stats)
    else:
        stats["wt_missing"] += 1

    write_data_yaml(output_dir)
    write_reports(output_dir, rows, stats)

    print("Done.")
    print(f"output_dir: {output_dir.resolve()}")
    print(f"train images: {stats['output_train_images']}")
    print(f"val images: {stats['output_val_images']}")
    print(f"test_clean images: {stats['output_test_clean_images']}")
    print(f"summary: {(output_dir / 'summary.txt').resolve()}")
    print(f"manifest: {(output_dir / 'manifest.csv').resolve()}")


if __name__ == "__main__":
    main()
