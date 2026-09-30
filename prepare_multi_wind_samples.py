from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_SPLITS = ("train", "valid", "val", "test")

NEGATIVE_CLASSES = {"good", "normal"}
DIRT_CLASSES = {"dirt"}
CORROSION_CLASSES = {"corrosion", "erosion"}
SMALL_DEFECT_CLASSES = {
    "crack",
    "craze",
    "hidecraze",
    "hide_craze",
    "scratch",
    "damage",
    "surfaceinjure",
    "surface_injure",
    "mechanicaldamage",
    "mechanical_damage",
    "paintoff",
    "paint_off",
    "chipping",
    "puncture",
    "hole",
    "paintpeel",
    "paint_peel",
}

PATH_NEGATIVE_HINTS = {"normal", "good", "dirt"}


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def compact_class_name(name: str) -> str:
    return normalize_class_name(name).replace("_", "")


def sanitize_name(name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in name.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe.lower() or "dataset"


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def read_yaml_names(data_yaml: Path) -> dict[int, str]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    names: Any = data.get("names")
    if names is None:
        raise ValueError(f"'names' not found in {data_yaml}")

    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}

    if isinstance(names, dict):
        return {int(key): str(value) for key, value in sorted(names.items(), key=lambda item: int(item[0]))}

    raise ValueError(f"Unsupported names format in {data_yaml}: {type(names).__name__}")


def read_line_class_names(path: Path) -> dict[int, str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"class-name file is empty: {path}")
    return {index: name for index, name in enumerate(names)}


def find_class_name_file(yolo_root: Path) -> Path | None:
    candidates = [
        yolo_root / "data.yaml",
        yolo_root / "data.yml",
        yolo_root / "classes.txt",
        yolo_root / "obj.names",
        yolo_root / "labels.txt",
        yolo_root / "labels" / "classes.txt",
        yolo_root / "labels" / "obj.names",
        yolo_root / "labels" / "labels.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_yolo_class_names(yolo_root: Path) -> dict[int, str]:
    class_file = find_class_name_file(yolo_root)
    if class_file is None:
        raise FileNotFoundError(f"No class-name file found for YOLO root: {yolo_root}")
    if class_file.suffix.lower() in {".yaml", ".yml"}:
        return read_yaml_names(class_file)
    return read_line_class_names(class_file)


def read_yolo_label(label_path: Path, class_id_to_name: dict[int, str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    class_names: list[str] = []

    if not label_path.exists():
        return class_names, warnings

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            warnings.append(f"Malformed YOLO row skipped: {label_path}:{line_number}")
            continue

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            warnings.append(f"Invalid class id skipped: {label_path}:{line_number}")
            continue

        class_names.append(class_id_to_name.get(class_id, f"unknown_{class_id}"))

    return class_names, warnings


def read_voc_classes(xml_path: Path) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    class_names: list[str] = []

    try:
        root = ElementTree.parse(xml_path).getroot()
    except Exception as exc:  # Keep a bad XML from stopping the batch.
        return [], [f"Failed to parse XML {xml_path}: {exc}"]

    for obj in root.findall(".//object"):
        name_node = obj.find("name")
        if name_node is not None and name_node.text:
            class_names.append(name_node.text.strip())

    return class_names, warnings


def classify_sample(class_names: list[str], has_annotation: bool, path_negative_hint: bool) -> tuple[list[str], str]:
    if not has_annotation:
        if path_negative_hint:
            return ["negative_samples"], "path_hint_negative_no_annotation"
        return ["unsure_samples"], "pure_image_no_annotation"

    if not class_names:
        return ["negative_samples"], "unannotated_negative_candidate"

    normalized = {normalize_class_name(name) for name in class_names}
    compact = {compact_class_name(name) for name in class_names}
    targets: list[str] = []
    reasons: list[str] = []

    if normalized <= NEGATIVE_CLASSES:
        targets.append("negative_samples")
        reasons.append("only_good_or_normal")

    if normalized & DIRT_CLASSES:
        targets.append("negative_samples")
        reasons.append("dirt_negative_candidate")

    if normalized & CORROSION_CLASSES:
        targets.append("new_corrosion_samples")
        reasons.append("corrosion_or_erosion")

    if (normalized & SMALL_DEFECT_CLASSES) or (compact & {compact_class_name(name) for name in SMALL_DEFECT_CLASSES}):
        targets.append("new_small_defect_samples")
        reasons.append("small_defect_class")

    known_norm = NEGATIVE_CLASSES | DIRT_CLASSES | CORROSION_CLASSES | SMALL_DEFECT_CLASSES
    known_compact = {compact_class_name(name) for name in known_norm}
    unknown = [name for name in class_names if normalize_class_name(name) not in known_norm and compact_class_name(name) not in known_compact]
    if unknown:
        targets.append("unsure_samples")
        reasons.append("unknown_class:" + "|".join(sorted(set(unknown))))

    if not targets:
        targets.append("unsure_samples")
        reasons.append("no_matching_rule")

    return list(dict.fromkeys(targets)), ";".join(reasons)


def unique_target_path(target_dir: Path, desired_name: str) -> Path:
    target = target_dir / desired_name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    index = 2
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def split_from_path(image_path: Path) -> str:
    parts = {part.lower() for part in image_path.parts}
    if "train" in parts:
        return "train"
    if "valid" in parts:
        return "valid"
    if "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "all"


def path_has_negative_hint(path: Path) -> bool:
    haystack = " ".join(path.parts).lower()
    return any(hint in haystack for hint in PATH_NEGATIVE_HINTS)


def copy_to_targets(
    image_path: Path,
    output_dirs: dict[str, Path],
    targets: list[str],
    dataset_name: str,
    split: str,
) -> list[Path]:
    copied_paths: list[Path] = []
    desired_name = f"{dataset_name}_{split}_{image_path.name}"
    for target in targets:
        target_path = unique_target_path(output_dirs[target], desired_name)
        shutil.copy2(image_path, target_path)
        copied_paths.append(target_path)
    return copied_paths


def find_yolo_roots(dataset_root: Path) -> list[Path]:
    roots: list[Path] = []
    for data_yaml in sorted(dataset_root.rglob("data.yaml")):
        candidate = data_yaml.parent
        if any((candidate / split / "images").exists() for split in YOLO_SPLITS):
            roots.append(candidate)

    for labels_dir in sorted(dataset_root.rglob("labels")):
        candidate = labels_dir.parent
        if (candidate / "images").exists() and find_class_name_file(candidate) is not None:
            if candidate not in roots:
                roots.append(candidate)
    return roots


def process_yolo_dataset(
    dataset_root: Path,
    dataset_name: str,
    output_dirs: dict[str, Path],
    index_rows: list[dict[str, str]],
    class_counter: Counter[str],
    dataset_counter: Counter[str],
    target_counter: Counter[str],
    warning_rows: list[str],
    processed_images: set[Path],
) -> tuple[int, int]:
    scanned = 0
    skipped = 0

    for yolo_root in find_yolo_roots(dataset_root):
        try:
            class_id_to_name = read_yolo_class_names(yolo_root)
        except Exception as exc:
            warning_rows.append(f"YOLO root skipped {yolo_root}: {exc}")
            continue

        split_dirs = [
            (split, yolo_root / split / "images", yolo_root / split / "labels")
            for split in YOLO_SPLITS
        ]
        if (yolo_root / "images").exists() and (yolo_root / "labels").exists():
            split_dirs.append(("all", yolo_root / "images", yolo_root / "labels"))

        for split, images_dir, labels_dir in split_dirs:
            if not images_dir.exists():
                continue

            for image_path in iter_images(images_dir):
                scanned += 1
                dataset_counter[dataset_name] += 1
                processed_images.add(image_path.resolve())
                label_path = labels_dir / f"{image_path.stem}.txt"
                class_names, warnings = read_yolo_label(label_path, class_id_to_name)
                warning_rows.extend(warnings)
                class_counter.update(class_names)

                if label_path.exists():
                    targets, reason = classify_sample(
                        class_names=class_names,
                        has_annotation=True,
                        path_negative_hint=path_has_negative_hint(image_path),
                    )
                else:
                    targets = ["negative_samples"]
                    reason = "unannotated_negative_candidate_missing_label"

                try:
                    copied_paths = copy_to_targets(image_path, output_dirs, targets, dataset_name, split)
                except Exception as exc:
                    skipped += 1
                    warning_rows.append(f"Copy failed for {image_path}: {exc}")
                    continue

                for target, copied_path in zip(targets, copied_paths):
                    target_counter[target] += 1
                    index_rows.append(
                        {
                            "source_dataset": dataset_name,
                            "source_image_path": str(image_path),
                            "source_annotation_path": str(label_path) if label_path.exists() else "",
                            "detected_format": "yolo",
                            "class_names": ";".join(sorted(set(class_names))),
                            "copied_to": str(copied_path),
                            "reason": reason,
                        }
                    )

    return scanned, skipped


def find_voc_xmls(dataset_root: Path) -> list[Path]:
    primary = dataset_root / "Annotations"
    if primary.exists():
        return sorted(primary.glob("*.xml"))
    return sorted(dataset_root.rglob("*.xml"))


def build_image_stem_index(dataset_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for image_path in iter_images(dataset_root):
        index[image_path.stem].append(image_path)
    return index


def process_voc_dataset(
    dataset_root: Path,
    dataset_name: str,
    output_dirs: dict[str, Path],
    index_rows: list[dict[str, str]],
    class_counter: Counter[str],
    dataset_counter: Counter[str],
    target_counter: Counter[str],
    warning_rows: list[str],
    processed_images: set[Path],
) -> tuple[int, int]:
    xml_paths = find_voc_xmls(dataset_root)
    if not xml_paths:
        return 0, 0

    image_index = build_image_stem_index(dataset_root)
    scanned = 0
    skipped = 0

    for xml_path in xml_paths:
        candidates = image_index.get(xml_path.stem, [])
        if not candidates:
            skipped += 1
            warning_rows.append(f"VOC XML has no matching image: {xml_path}")
            continue

        image_path = candidates[0]
        scanned += 1
        dataset_counter[dataset_name] += 1
        processed_images.add(image_path.resolve())
        class_names, warnings = read_voc_classes(xml_path)
        warning_rows.extend(warnings)
        class_counter.update(class_names)
        targets, reason = classify_sample(
            class_names=class_names,
            has_annotation=True,
            path_negative_hint=path_has_negative_hint(image_path),
        )

        try:
            copied_paths = copy_to_targets(image_path, output_dirs, targets, dataset_name, split_from_path(image_path))
        except Exception as exc:
            skipped += 1
            warning_rows.append(f"Copy failed for {image_path}: {exc}")
            continue

        for target, copied_path in zip(targets, copied_paths):
            target_counter[target] += 1
            index_rows.append(
                {
                    "source_dataset": dataset_name,
                    "source_image_path": str(image_path),
                    "source_annotation_path": str(xml_path),
                    "detected_format": "voc_xml",
                    "class_names": ";".join(sorted(set(class_names))),
                    "copied_to": str(copied_path),
                    "reason": reason,
                }
            )

    return scanned, skipped


def process_pure_images(
    dataset_root: Path,
    dataset_name: str,
    output_dirs: dict[str, Path],
    index_rows: list[dict[str, str]],
    dataset_counter: Counter[str],
    target_counter: Counter[str],
    warning_rows: list[str],
    processed_images: set[Path],
) -> tuple[int, int]:
    scanned = 0
    skipped = 0

    for image_path in iter_images(dataset_root):
        if image_path.resolve() in processed_images:
            continue

        scanned += 1
        dataset_counter[dataset_name] += 1
        target = "negative_samples" if path_has_negative_hint(image_path) else "unsure_samples"
        reason = "path_hint_negative_no_annotation" if target == "negative_samples" else "pure_image_no_annotation"

        try:
            copied_paths = copy_to_targets(image_path, output_dirs, [target], dataset_name, split_from_path(image_path))
        except Exception as exc:
            skipped += 1
            warning_rows.append(f"Copy failed for {image_path}: {exc}")
            continue

        target_counter[target] += 1
        index_rows.append(
            {
                "source_dataset": dataset_name,
                "source_image_path": str(image_path),
                "source_annotation_path": "",
                "detected_format": "pure_image",
                "class_names": "",
                "copied_to": str(copied_paths[0]),
                "reason": reason,
            }
        )

    return scanned, skipped


def extract_zip(zip_path: Path, extracted_root: Path) -> Path:
    target_dir = extracted_root / sanitize_name(zip_path.stem)
    marker = target_dir / ".extract_complete"
    if marker.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return target_dir


def looks_like_dataset_root(path: Path) -> bool:
    return (
        (path / "data.yaml").exists()
        or ((path / "images").exists() and (path / "labels").exists())
        or any((path / split / "images").exists() for split in YOLO_SPLITS)
    )


def discover_sources(input_dir: Path) -> list[tuple[str, Path]]:
    extracted_root = input_dir / "_extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]] = []
    if looks_like_dataset_root(input_dir):
        sources.append((sanitize_name(input_dir.name), input_dir))

    for zip_path in sorted(input_dir.glob("*.zip")):
        try:
            extracted = extract_zip(zip_path, extracted_root)
        except Exception as exc:
            print(f"Warning: failed to extract {zip_path}: {exc}")
            continue
        sources.append((sanitize_name(zip_path.stem), extracted))

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir() or child.name == "_extracted":
            continue
        if child.resolve() == input_dir.resolve():
            continue
        sources.append((sanitize_name(child.name), child))

    return sources


def write_summary(
    output_dir: Path,
    total_scanned: int,
    total_copied: int,
    skipped: int,
    target_counter: Counter[str],
    class_counter: Counter[str],
    dataset_counter: Counter[str],
    warnings: list[str],
) -> Path:
    summary_path = output_dir / "summary.txt"
    lines = [
        "Multi wind blade sample preparation summary",
        "",
        f"total_images_scanned: {total_scanned}",
        f"total_copied_images: {total_copied}",
        f"negative_samples: {target_counter['negative_samples']}",
        f"new_corrosion_samples: {target_counter['new_corrosion_samples']}",
        f"new_small_defect_samples: {target_counter['new_small_defect_samples']}",
        f"unsure_samples: {target_counter['unsure_samples']}",
        f"skipped_images: {skipped}",
        "",
        "class_counts:",
    ]
    if class_counter:
        for class_name, count in sorted(class_counter.items(), key=lambda item: item[0].lower()):
            lines.append(f"- {class_name}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "dataset_counts:"])
    if dataset_counter:
        for dataset_name, count in sorted(dataset_counter.items()):
            lines.append(f"- {dataset_name}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "warnings:"])
    if warnings:
        for warning in warnings[:500]:
            lines.append(f"- {warning}")
        if len(warnings) > 500:
            lines.append(f"- ... {len(warnings) - 500} more warnings omitted")
    else:
        lines.append("- none")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def prepare_multi_samples(input_dir: Path, output_dir: Path) -> dict[str, int | str]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    output_dirs = {
        "negative_samples": output_dir / "negative_samples",
        "new_corrosion_samples": output_dir / "new_corrosion_samples",
        "new_small_defect_samples": output_dir / "new_small_defect_samples",
        "unsure_samples": output_dir / "unsure_samples",
    }
    for target_dir in output_dirs.values():
        target_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    class_counter: Counter[str] = Counter()
    dataset_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    warnings: list[str] = []
    total_scanned = 0
    skipped = 0

    sources = discover_sources(input_dir)
    if not sources:
        warnings.append(f"No zip files or dataset folders found in {input_dir}")

    for dataset_name, dataset_root in sources:
        processed_images: set[Path] = set()
        yolo_scanned, yolo_skipped = process_yolo_dataset(
            dataset_root,
            dataset_name,
            output_dirs,
            index_rows,
            class_counter,
            dataset_counter,
            target_counter,
            warnings,
            processed_images,
        )
        voc_scanned, voc_skipped = process_voc_dataset(
            dataset_root,
            dataset_name,
            output_dirs,
            index_rows,
            class_counter,
            dataset_counter,
            target_counter,
            warnings,
            processed_images,
        )
        pure_scanned, pure_skipped = process_pure_images(
            dataset_root,
            dataset_name,
            output_dirs,
            index_rows,
            dataset_counter,
            target_counter,
            warnings,
            processed_images,
        )
        total_scanned += yolo_scanned + voc_scanned + pure_scanned
        skipped += yolo_skipped + voc_skipped + pure_skipped

    index_path = output_dir / "sample_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_dataset",
                "source_image_path",
                "source_annotation_path",
                "detected_format",
                "class_names",
                "copied_to",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    total_copied = sum(target_counter.values())
    summary_path = write_summary(
        output_dir=output_dir,
        total_scanned=total_scanned,
        total_copied=total_copied,
        skipped=skipped,
        target_counter=target_counter,
        class_counter=class_counter,
        dataset_counter=dataset_counter,
        warnings=warnings,
    )

    return {
        "total_scanned": total_scanned,
        "total_copied": total_copied,
        "negative_samples": target_counter["negative_samples"],
        "new_corrosion_samples": target_counter["new_corrosion_samples"],
        "new_small_defect_samples": target_counter["new_small_defect_samples"],
        "unsure_samples": target_counter["unsure_samples"],
        "skipped": skipped,
        "summary_path": str(summary_path),
        "index_path": str(index_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare candidate wind blade samples from multiple datasets.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing zip files and/or dataset folders.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output sample directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = prepare_multi_samples(args.input_dir, args.output_dir)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print("Done.")
    print(f"Total images scanned: {result['total_scanned']}")
    print(f"Total copied images: {result['total_copied']}")
    print(f"negative_samples: {result['negative_samples']}")
    print(f"new_corrosion_samples: {result['new_corrosion_samples']}")
    print(f"new_small_defect_samples: {result['new_small_defect_samples']}")
    print(f"unsure_samples: {result['unsure_samples']}")
    print(f"skipped_images: {result['skipped']}")
    print(f"summary saved to: {result['summary_path']}")
    print(f"index saved to: {result['index_path']}")


if __name__ == "__main__":
    main()
