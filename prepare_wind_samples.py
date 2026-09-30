from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

NEGATIVE_CLASSES = {"good"}
CORROSION_CLASSES = {"erosion"}
SMALL_DEFECT_CLASSES = {"crack", "scratch", "mechanicaldamage", "paintoff"}


def load_class_names(data_yaml: Path) -> dict[int, str]:
    """Read Roboflow/YOLO data.yaml and return class_id -> class_name."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    names: Any = data.get("names")
    if names is None:
        raise ValueError(f"'names' not found in data.yaml: {data_yaml}")

    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}

    if isinstance(names, dict):
        class_names: dict[int, str] = {}
        for key, value in names.items():
            class_names[int(key)] = str(value)
        return dict(sorted(class_names.items()))

    raise ValueError(f"Unsupported names format in data.yaml: {type(names).__name__}")


def read_label_classes(label_path: Path, class_names: dict[int, str]) -> tuple[list[str], Counter[str]]:
    """Read one YOLO label file and return class names plus per-label counts."""
    image_classes: list[str] = []
    counts: Counter[str] = Counter()

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"Warning: malformed label line skipped: {label_path}:{line_number}")
            continue

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            print(f"Warning: invalid class id skipped: {label_path}:{line_number}")
            continue

        class_name = class_names.get(class_id, f"unknown_{class_id}")
        image_classes.append(class_name)
        counts[class_name] += 1

    return image_classes, counts


def classify_image(class_names: list[str]) -> list[str]:
    """Return target folder names for one image based on its label classes."""
    normalized = {name.strip().lower().replace(" ", "") for name in class_names}
    targets: list[str] = []

    if normalized and normalized <= NEGATIVE_CLASSES:
        targets.append("negative_samples")

    if normalized & CORROSION_CLASSES:
        targets.append("new_corrosion_samples")

    if normalized & SMALL_DEFECT_CLASSES:
        targets.append("new_small_defect_samples")

    return targets


def unique_target_path(target_dir: Path, desired_name: str) -> Path:
    """Avoid overwrite when two source images produce the same copied name."""
    target_path = target_dir / desired_name
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    index = 2
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def prepare_samples(dataset_dir: Path, output_dir: Path) -> dict[str, int | str]:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    data_yaml = dataset_dir / "data.yaml"
    class_id_to_name = load_class_names(data_yaml)

    target_dirs = {
        "negative_samples": output_dir / "negative_samples",
        "new_corrosion_samples": output_dir / "new_corrosion_samples",
        "new_small_defect_samples": output_dir / "new_small_defect_samples",
    }
    for target_dir in target_dirs.values():
        target_dir.mkdir(parents=True, exist_ok=True)

    sample_index_path = output_dir / "sample_index.csv"
    summary_path = output_dir / "summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    skipped_images = 0
    copied_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    index_rows: list[dict[str, str]] = []

    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"

        if not images_dir.exists():
            print(f"Warning: split images dir not found, skipped: {images_dir}")
            continue
        if not labels_dir.exists():
            print(f"Warning: split labels dir not found, labels will be treated as missing: {labels_dir}")

        for image_path in iter_images(images_dir):
            total_images += 1
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                skipped_images += 1
                index_rows.append(
                    {
                        "original_image_path": str(image_path),
                        "original_label_path": str(label_path),
                        "copied_to": "SKIPPED_MISSING_LABEL",
                        "class_names": "",
                    }
                )
                continue

            image_class_names, image_class_counts = read_label_classes(label_path, class_id_to_name)
            class_counts.update(image_class_counts)

            if not image_class_names:
                skipped_images += 1
                index_rows.append(
                    {
                        "original_image_path": str(image_path),
                        "original_label_path": str(label_path),
                        "copied_to": "SKIPPED_EMPTY_OR_INVALID_LABEL",
                        "class_names": "",
                    }
                )
                continue

            targets = classify_image(image_class_names)
            if not targets:
                skipped_images += 1
                index_rows.append(
                    {
                        "original_image_path": str(image_path),
                        "original_label_path": str(label_path),
                        "copied_to": "SKIPPED_NO_TARGET_CLASS",
                        "class_names": ";".join(sorted(set(image_class_names))),
                    }
                )
                continue

            copied_name = f"{split}_{image_path.name}"
            for target in targets:
                target_path = unique_target_path(target_dirs[target], copied_name)
                shutil.copy2(image_path, target_path)
                copied_counts[target] += 1
                index_rows.append(
                    {
                        "original_image_path": str(image_path),
                        "original_label_path": str(label_path),
                        "copied_to": str(target_path),
                        "class_names": ";".join(sorted(set(image_class_names))),
                    }
                )

    with sample_index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_image_path", "original_label_path", "copied_to", "class_names"],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    summary_lines = [
        "Roboflow YOLOv8 wind sample preparation summary",
        "",
        f"dataset_dir: {dataset_dir}",
        f"output_dir: {output_dir}",
        "",
        f"total_images: {total_images}",
        f"negative_samples: {copied_counts['negative_samples']}",
        f"new_corrosion_samples: {copied_counts['new_corrosion_samples']}",
        f"new_small_defect_samples: {copied_counts['new_small_defect_samples']}",
        f"skipped_images: {skipped_images}",
        "",
        "class_counts:",
    ]
    for class_id, class_name in class_id_to_name.items():
        summary_lines.append(f"- {class_id}: {class_name}: {class_counts[class_name]}")
    for class_name in sorted(set(class_counts) - set(class_id_to_name.values())):
        summary_lines.append(f"- {class_name}: {class_counts[class_name]}")

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "total_images": total_images,
        "negative_samples": copied_counts["negative_samples"],
        "new_corrosion_samples": copied_counts["new_corrosion_samples"],
        "new_small_defect_samples": copied_counts["new_small_defect_samples"],
        "skipped_images": skipped_images,
        "summary_path": str(summary_path),
        "sample_index_path": str(sample_index_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organize Roboflow YOLOv8 wind blade samples into review folders."
    )
    parser.add_argument("--dataset_dir", required=True, type=Path, help="Roboflow YOLOv8 dataset directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output sample directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = prepare_samples(args.dataset_dir, args.output_dir)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print("Done.")
    print(f"total_images: {result['total_images']}")
    print(f"negative_samples: {result['negative_samples']}")
    print(f"new_corrosion_samples: {result['new_corrosion_samples']}")
    print(f"new_small_defect_samples: {result['new_small_defect_samples']}")
    print(f"skipped_images: {result['skipped_images']}")
    print(f"summary saved to: {result['summary_path']}")
    print(f"sample index saved to: {result['sample_index_path']}")


if __name__ == "__main__":
    main()
