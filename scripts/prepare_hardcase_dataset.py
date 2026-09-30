from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["crack", "hole", "spalling", "corrosion"]


def _copy_files(source: Path, target: Path, pattern: str = "*") -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob(pattern)):
        if path.is_file():
            shutil.copy2(path, target / path.name)
            count += 1
    return count


def _find_label(original_dataset: Path, stem: str) -> Path | None:
    matches = sorted((original_dataset / "labels").rglob(f"{stem}.txt"))
    return matches[0] if matches else None


def _safe_clear_output(output: Path) -> None:
    resolved = output.resolve()
    if output.exists():
        if "wind_blade_defect_hardcase" not in resolved.name:
            raise RuntimeError(f"Refuse to overwrite unexpected output directory: {resolved}")
        shutil.rmtree(output)


def build_dataset(original_dataset: Path, hardcase_dir: Path, output: Path, overwrite: bool = False) -> dict:
    original_dataset = original_dataset.resolve()
    hardcase_dir = hardcase_dir.resolve()
    output = output.resolve()

    if overwrite:
        _safe_clear_output(output)
    elif output.exists():
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to rebuild.")

    for split in ["train", "val", "test"]:
        if not (original_dataset / "images" / split).exists():
            raise FileNotFoundError(f"Missing original image split: {split}")
        if not (original_dataset / "labels" / split).exists():
            raise FileNotFoundError(f"Missing original label split: {split}")

    output.mkdir(parents=True, exist_ok=True)

    counts = {
        "train_images_original": _copy_files(original_dataset / "images" / "train", output / "images" / "train"),
        "train_labels_original": _copy_files(original_dataset / "labels" / "train", output / "labels" / "train", "*.txt"),
        "val_images": _copy_files(original_dataset / "images" / "val", output / "images" / "val"),
        "val_labels": _copy_files(original_dataset / "labels" / "val", output / "labels" / "val", "*.txt"),
    }

    hardcase_images = [
        path for path in sorted(hardcase_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    hardcase_stems = {path.stem for path in hardcase_images}
    manifest_rows: list[dict[str, str]] = []

    added = 0
    skipped_missing_label: list[str] = []
    for image_path in hardcase_images:
        label_path = _find_label(original_dataset, image_path.stem)
        if not label_path:
            skipped_missing_label.append(image_path.name)
            continue

        new_stem = f"hardcase_missed_{image_path.stem}"
        new_image = output / "images" / "train" / f"{new_stem}{image_path.suffix.lower()}"
        new_label = output / "labels" / "train" / f"{new_stem}.txt"
        shutil.copy2(image_path, new_image)
        shutil.copy2(label_path, new_label)
        manifest_rows.append(
            {
                "source_image": str(image_path),
                "source_label": str(label_path),
                "target_image": str(new_image),
                "target_label": str(new_label),
            }
        )
        added += 1

    test_clean_images = 0
    test_clean_labels = 0
    test_image_dir = original_dataset / "images" / "test"
    test_label_dir = original_dataset / "labels" / "test"
    (output / "images" / "test_clean").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "test_clean").mkdir(parents=True, exist_ok=True)
    for image_path in sorted(test_image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem in hardcase_stems:
            continue
        shutil.copy2(image_path, output / "images" / "test_clean" / image_path.name)
        test_clean_images += 1

    for label_path in sorted(test_label_dir.glob("*.txt")):
        if label_path.stem in hardcase_stems:
            continue
        shutil.copy2(label_path, output / "labels" / "test_clean" / label_path.name)
        test_clean_labels += 1

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test_clean",
                "",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = output / "hardcase_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source_image", "source_label", "target_image", "target_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts.update(
        {
            "hardcase_added_to_train": added,
            "hardcase_missing_label": len(skipped_missing_label),
            "test_clean_images": test_clean_images,
            "test_clean_labels": test_clean_labels,
            "data_yaml": str(data_yaml),
            "manifest": str(manifest_path),
        }
    )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a YOLO dataset augmented with hard-case samples.")
    parser.add_argument("--original", type=Path, default=Path("datasets/wind_blade_defect"))
    parser.add_argument("--hardcases", type=Path, default=Path("hard_cases/missed_samples"))
    parser.add_argument("--output", type=Path, default=Path("datasets/wind_blade_defect_hardcase_v02"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = build_dataset(args.original, args.hardcases, args.output, args.overwrite)
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
