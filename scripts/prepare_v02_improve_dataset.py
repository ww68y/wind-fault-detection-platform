from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["crack", "hole", "spalling", "corrosion"]
CORROSION_CLASS_ID = 3

DEFAULT_FOCUS_STEMS = [
    "corrosion_166",
    "corrosion_66",
    "crack_68",
]


def _copy_files(source: Path, target: Path, pattern: str = "*") -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob(pattern)):
        if path.is_file():
            shutil.copy2(path, target / path.name)
            count += 1
    return count


def _image_for_stem(image_root: Path, stem: str) -> Path | None:
    for image_path in sorted(image_root.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES and image_path.stem == stem:
            return image_path
    return None


def _find_label(original_dataset: Path, stem: str) -> Path | None:
    matches = sorted((original_dataset / "labels").rglob(f"{stem}.txt"))
    return matches[0] if matches else None


def _label_has_class(label_path: Path, class_id: int) -> bool:
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            if int(parts[0]) == class_id:
                return True
        except ValueError:
            continue
    return False


def _safe_clear_output(output: Path) -> None:
    resolved = output.resolve()
    if output.exists():
        if "wind_blade_defect_v02_improve" not in resolved.name:
            raise RuntimeError(f"Refuse to overwrite unexpected output directory: {resolved}")
        shutil.rmtree(output)


def _copy_pair(image_path: Path, label_path: Path, target_image: Path, target_label: Path) -> None:
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)


def _add_training_copy(
    image_path: Path,
    label_path: Path,
    output: Path,
    new_stem: str,
    manifest_rows: list[dict[str, str]],
    reason: str,
) -> None:
    suffix = image_path.suffix.lower()
    target_image = output / "images" / "train" / f"{new_stem}{suffix}"
    target_label = output / "labels" / "train" / f"{new_stem}.txt"
    _copy_pair(image_path, label_path, target_image, target_label)
    manifest_rows.append(
        {
            "reason": reason,
            "source_image": str(image_path),
            "source_label": str(label_path),
            "target_image": str(target_image),
            "target_label": str(target_label),
        }
    )


def build_dataset(
    original_dataset: Path,
    hardcase_dir: Path,
    output: Path,
    focus_stems: list[str],
    focus_repeats: int,
    oversample_corrosion: bool,
    overwrite: bool = False,
) -> dict[str, int | str]:
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

    counts: dict[str, int | str] = {
        "train_images_original": _copy_files(original_dataset / "images" / "train", output / "images" / "train"),
        "train_labels_original": _copy_files(original_dataset / "labels" / "train", output / "labels" / "train", "*.txt"),
        "val_images": _copy_files(original_dataset / "images" / "val", output / "images" / "val"),
        "val_labels": _copy_files(original_dataset / "labels" / "val", output / "labels" / "val", "*.txt"),
    }

    manifest_rows: list[dict[str, str]] = []
    image_root = original_dataset / "images"

    hardcase_images = [
        path for path in sorted(hardcase_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    hardcase_stems = {path.stem for path in hardcase_images}
    hardcase_added = 0
    skipped_missing_label: list[str] = []
    for image_path in hardcase_images:
        label_path = _find_label(original_dataset, image_path.stem)
        if not label_path:
            skipped_missing_label.append(image_path.name)
            continue
        _add_training_copy(
            image_path=image_path,
            label_path=label_path,
            output=output,
            new_stem=f"hardcase_missed_{image_path.stem}",
            manifest_rows=manifest_rows,
            reason="hardcase_missed_once",
        )
        hardcase_added += 1

    corrosion_oversampled = 0
    if oversample_corrosion:
        train_label_dir = original_dataset / "labels" / "train"
        for label_path in sorted(train_label_dir.glob("*.txt")):
            if not _label_has_class(label_path, CORROSION_CLASS_ID):
                continue
            image_path = _image_for_stem(original_dataset / "images" / "train", label_path.stem)
            if not image_path:
                continue
            _add_training_copy(
                image_path=image_path,
                label_path=label_path,
                output=output,
                new_stem=f"oversample_corrosion_{label_path.stem}",
                manifest_rows=manifest_rows,
                reason="corrosion_train_oversample",
            )
            corrosion_oversampled += 1

    focus_added = 0
    for stem in focus_stems:
        image_path = _image_for_stem(image_root, stem)
        label_path = _find_label(original_dataset, stem)
        if not image_path or not label_path:
            skipped_missing_label.append(f"{stem}.*")
            continue
        for repeat_index in range(1, focus_repeats + 1):
            _add_training_copy(
                image_path=image_path,
                label_path=label_path,
                output=output,
                new_stem=f"focus_v02_{repeat_index}_{stem}",
                manifest_rows=manifest_rows,
                reason="v02_still_missed_or_regressed",
            )
            focus_added += 1

    test_clean_images = 0
    test_clean_labels = 0
    (output / "images" / "test_clean").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "test_clean").mkdir(parents=True, exist_ok=True)
    for image_path in sorted((original_dataset / "images" / "test").iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem in hardcase_stems:
            continue
        shutil.copy2(image_path, output / "images" / "test_clean" / image_path.name)
        test_clean_images += 1
    for label_path in sorted((original_dataset / "labels" / "test").glob("*.txt")):
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

    manifest_path = output / "v02_improve_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["reason", "source_image", "source_label", "target_image", "target_label"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts.update(
        {
            "hardcase_added_to_train": hardcase_added,
            "corrosion_oversampled_to_train": corrosion_oversampled,
            "focus_samples_added_to_train": focus_added,
            "missing_or_skipped": len(skipped_missing_label),
            "test_clean_images": test_clean_images,
            "test_clean_labels": test_clean_labels,
            "data_yaml": str(data_yaml),
            "manifest": str(manifest_path),
        }
    )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a v02 improvement dataset focused on corrosion and hard cases.")
    parser.add_argument("--original", type=Path, default=Path("datasets/wind_blade_defect"))
    parser.add_argument("--hardcases", type=Path, default=Path("hard_cases/missed_samples"))
    parser.add_argument("--output", type=Path, default=Path("datasets/wind_blade_defect_v02_improve"))
    parser.add_argument("--focus-stems", nargs="*", default=DEFAULT_FOCUS_STEMS)
    parser.add_argument("--focus-repeats", type=int, default=3)
    parser.add_argument("--no-corrosion-oversample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = build_dataset(
        original_dataset=args.original,
        hardcase_dir=args.hardcases,
        output=args.output,
        focus_stems=args.focus_stems,
        focus_repeats=args.focus_repeats,
        oversample_corrosion=not args.no_corrosion_oversample,
        overwrite=args.overwrite,
    )
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
