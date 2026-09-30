from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ["crack", "hole", "spalling", "corrosion"]
CORROSION_CLASS_ID = 3
DEFAULT_SMALL_TARGET_CLASS_IDS = [0, 1, 2]


def _iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def _copy_files(source: Path, target: Path, pattern: str = "*") -> int:
    target.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return 0
    count = 0
    for path in sorted(source.glob(pattern)):
        if path.is_file():
            shutil.copy2(path, target / path.name)
            count += 1
    return count


def _safe_clear_output(output: Path) -> None:
    resolved = output.resolve()
    if output.exists():
        if "wind_blade_defect_v03" not in resolved.name:
            raise RuntimeError(f"Refuse to overwrite unexpected output directory: {resolved}")
        shutil.rmtree(output)


def _find_image(dataset: Path, stem: str, split: str | None = None) -> Path | None:
    roots = [dataset / "images" / split] if split else [dataset / "images"]
    for root in roots:
        for image_path in _iter_images(root):
            if image_path.stem == stem:
                return image_path
    return None


def _find_label(label_sources: list[Path], stem: str) -> Path | None:
    for dataset in label_sources:
        matches = sorted((dataset / "labels").rglob(f"{stem}.txt"))
        if matches:
            return matches[0]
    return None


def _read_label_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return rows
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            continue
        rows.append((class_id, x_center, y_center, width, height))
    return rows


def _label_has_class(label_path: Path, class_id: int) -> bool:
    return any(row[0] == class_id for row in _read_label_rows(label_path))


def _label_has_small_target(label_path: Path, class_ids: set[int], max_area: float) -> bool:
    for class_id, _x, _y, width, height in _read_label_rows(label_path):
        if class_id in class_ids and width * height <= max_area:
            return True
    return False


def _target_paths(output: Path, stem: str, suffix: str) -> tuple[Path, Path]:
    image_path = output / "images" / "train" / f"{stem}{suffix.lower()}"
    label_path = output / "labels" / "train" / f"{stem}.txt"
    return image_path, label_path


def _unique_stem(output: Path, desired_stem: str, suffix: str) -> str:
    stem = desired_stem
    index = 2
    while True:
        image_path, label_path = _target_paths(output, stem, suffix)
        if not image_path.exists() and not label_path.exists():
            return stem
        stem = f"{desired_stem}_{index}"
        index += 1


def _add_labeled_training_copy(
    image_path: Path,
    label_path: Path,
    output: Path,
    desired_stem: str,
    manifest_rows: list[dict[str, str]],
    reason: str,
) -> None:
    target_stem = _unique_stem(output, desired_stem, image_path.suffix)
    target_image, target_label = _target_paths(output, target_stem, image_path.suffix)
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)
    manifest_rows.append(
        {
            "reason": reason,
            "source_image": str(image_path),
            "source_label": str(label_path),
            "target_image": str(target_image),
            "target_label": str(target_label),
        }
    )


def _add_negative_training_copy(
    image_path: Path,
    output: Path,
    desired_stem: str,
    manifest_rows: list[dict[str, str]],
    reason: str,
) -> None:
    target_stem = _unique_stem(output, desired_stem, image_path.suffix)
    target_image, target_label = _target_paths(output, target_stem, image_path.suffix)
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_image)
    target_label.write_text("", encoding="utf-8")
    manifest_rows.append(
        {
            "reason": reason,
            "source_image": str(image_path),
            "source_label": "",
            "target_image": str(target_image),
            "target_label": str(target_label),
        }
    )


def _copy_base_dataset(base_dataset: Path, output: Path) -> dict[str, int]:
    counts = {
        "base_train_images": _copy_files(base_dataset / "images" / "train", output / "images" / "train"),
        "base_train_labels": _copy_files(base_dataset / "labels" / "train", output / "labels" / "train", "*.txt"),
        "base_val_images": _copy_files(base_dataset / "images" / "val", output / "images" / "val"),
        "base_val_labels": _copy_files(base_dataset / "labels" / "val", output / "labels" / "val", "*.txt"),
    }
    test_split = "test_clean" if (base_dataset / "images" / "test_clean").exists() else "test"
    counts["base_test_images"] = _copy_files(
        base_dataset / "images" / test_split,
        output / "images" / "test_clean",
    )
    counts["base_test_labels"] = _copy_files(
        base_dataset / "labels" / test_split,
        output / "labels" / "test_clean",
        "*.txt",
    )
    return counts


def _write_data_yaml(output: Path) -> Path:
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
    return data_yaml


def _write_manifest(output: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = output / "v03_recall_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["reason", "source_image", "source_label", "target_image", "target_label"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_summary(output: Path, counts: dict[str, int | str]) -> Path:
    summary_path = output / "v03_recall_summary.md"
    lines = [
        "# v03 recall dataset summary",
        "",
        "Targets:",
        "",
        "- Recall >= 0.95",
        "- mAP50-95 >= 0.85",
        "- corrosion should not regress below the old model",
        "- hard case missed samples should keep dropping",
        "- false positives should drop after curated negative samples are added",
        "",
        "Counts:",
        "",
        "| Item | Value |",
        "| --- | ---: |",
    ]
    for key in sorted(counts):
        lines.append(f"| {key} | {counts[key]} |")
    lines.extend(
        [
            "",
            "Negative-sample note:",
            "",
            "Put reviewed normal blade/background images in `hard_cases/negative_samples` or "
            "`hard_cases/false_positive_samples`, then rebuild with `--overwrite`. The script will add them with "
            "empty YOLO label files.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_dataset(
    base_dataset: Path,
    original_dataset: Path,
    output: Path,
    missed_dir: Path,
    low_confidence_dir: Path,
    false_positive_dir: Path,
    negative_dirs: list[Path],
    corrosion_repeats: int,
    small_target_repeats: int,
    small_target_max_area: float,
    small_target_class_ids: list[int],
    overwrite: bool = False,
) -> dict[str, int | str]:
    base_dataset = base_dataset.resolve()
    original_dataset = original_dataset.resolve()
    output = output.resolve()
    missed_dir = missed_dir.resolve()
    low_confidence_dir = low_confidence_dir.resolve()
    false_positive_dir = false_positive_dir.resolve()
    negative_dirs = [path.resolve() for path in negative_dirs]

    if overwrite:
        _safe_clear_output(output)
    elif output.exists():
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to rebuild.")

    for split in ["train", "val"]:
        if not (base_dataset / "images" / split).exists():
            raise FileNotFoundError(f"Missing base image split: {split}")
        if not (base_dataset / "labels" / split).exists():
            raise FileNotFoundError(f"Missing base label split: {split}")

    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int | str] = _copy_base_dataset(base_dataset, output)
    manifest_rows: list[dict[str, str]] = []
    label_sources = [base_dataset, original_dataset]
    skipped_missing_label: list[str] = []

    error_dirs = [
        ("missed_sample", missed_dir),
        ("low_confidence_sample", low_confidence_dir),
    ]
    for reason, source_dir in error_dirs:
        added = 0
        for image_path in _iter_images(source_dir):
            label_path = _find_label(label_sources, image_path.stem)
            if label_path is None:
                skipped_missing_label.append(str(image_path))
                continue
            _add_labeled_training_copy(
                image_path=image_path,
                label_path=label_path,
                output=output,
                desired_stem=f"v03_{reason}_{image_path.stem}",
                manifest_rows=manifest_rows,
                reason=reason,
            )
            added += 1
        counts[f"{reason}_added"] = added

    corrosion_added = 0
    for repeat_index in range(1, corrosion_repeats + 1):
        for label_path in sorted((base_dataset / "labels" / "train").glob("*.txt")):
            if not _label_has_class(label_path, CORROSION_CLASS_ID):
                continue
            image_path = _find_image(base_dataset, label_path.stem, split="train")
            if image_path is None:
                continue
            _add_labeled_training_copy(
                image_path=image_path,
                label_path=label_path,
                output=output,
                desired_stem=f"v03_corrosion_r{repeat_index}_{label_path.stem}",
                manifest_rows=manifest_rows,
                reason="corrosion_recall_oversample",
            )
            corrosion_added += 1
    counts["corrosion_oversampled"] = corrosion_added

    small_class_ids = set(small_target_class_ids)
    small_target_added = 0
    for repeat_index in range(1, small_target_repeats + 1):
        for label_path in sorted((base_dataset / "labels" / "train").glob("*.txt")):
            if not _label_has_small_target(label_path, small_class_ids, small_target_max_area):
                continue
            image_path = _find_image(base_dataset, label_path.stem, split="train")
            if image_path is None:
                continue
            _add_labeled_training_copy(
                image_path=image_path,
                label_path=label_path,
                output=output,
                desired_stem=f"v03_small_r{repeat_index}_{label_path.stem}",
                manifest_rows=manifest_rows,
                reason="small_target_recall_oversample",
            )
            small_target_added += 1
    counts["small_target_oversampled"] = small_target_added

    negative_added_by_reason: Counter[str] = Counter()
    negative_sources = [("false_positive_as_negative", false_positive_dir)]
    negative_sources.extend(("curated_negative", path) for path in negative_dirs)
    for reason, source_dir in negative_sources:
        for image_path in _iter_images(source_dir):
            _add_negative_training_copy(
                image_path=image_path,
                output=output,
                desired_stem=f"v03_{reason}_{image_path.stem}",
                manifest_rows=manifest_rows,
                reason=reason,
            )
            negative_added_by_reason[reason] += 1
    for reason, added in sorted(negative_added_by_reason.items()):
        counts[f"{reason}_added"] = added
    counts["negative_samples_added"] = sum(negative_added_by_reason.values())

    data_yaml = _write_data_yaml(output)
    manifest_path = _write_manifest(output, manifest_rows)
    counts["missing_labels_skipped"] = len(skipped_missing_label)
    counts["manifest_rows"] = len(manifest_rows)
    counts["data_yaml"] = str(data_yaml)
    counts["manifest"] = str(manifest_path)
    counts["summary"] = str(_write_summary(output, counts))
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a v03 recall-focused YOLO dataset.")
    parser.add_argument("--base", type=Path, default=Path("datasets/wind_blade_defect_v02_improve"))
    parser.add_argument("--original", type=Path, default=Path("datasets/wind_blade_defect"))
    parser.add_argument("--output", type=Path, default=Path("datasets/wind_blade_defect_v03_recall"))
    parser.add_argument("--missed-dir", type=Path, default=Path("hard_cases/missed_samples"))
    parser.add_argument("--low-confidence-dir", type=Path, default=Path("hard_cases/low_confidence_samples"))
    parser.add_argument("--false-positive-dir", type=Path, default=Path("hard_cases/false_positive_samples"))
    parser.add_argument(
        "--negative-dir",
        type=Path,
        action="append",
        default=[Path("hard_cases/negative_samples")],
        help="Reviewed normal/background image directory. Can be passed more than once.",
    )
    parser.add_argument("--corrosion-repeats", type=int, default=1)
    parser.add_argument("--small-target-repeats", type=int, default=1)
    parser.add_argument("--small-target-max-area", type=float, default=0.012)
    parser.add_argument("--small-target-class-ids", type=int, nargs="*", default=DEFAULT_SMALL_TARGET_CLASS_IDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = build_dataset(
        base_dataset=args.base,
        original_dataset=args.original,
        output=args.output,
        missed_dir=args.missed_dir,
        low_confidence_dir=args.low_confidence_dir,
        false_positive_dir=args.false_positive_dir,
        negative_dirs=args.negative_dir,
        corrosion_repeats=args.corrosion_repeats,
        small_target_repeats=args.small_target_repeats,
        small_target_max_area=args.small_target_max_area,
        small_target_class_ids=args.small_target_class_ids,
        overwrite=args.overwrite,
    )
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
