from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_DATA_YAML = Path("datasets/wind_blade_defect_v05_conservative/data.yaml")
DEFAULT_OUTPUT = Path("datasets/msf_shape_mask_candidates")
PRIORITY_CLASS_NAMES = {"crack", "corrosion"}


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    names: dict[int, str]


@dataclass(frozen=True)
class YoloObject:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    line_index: int

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        short_side = max(min(self.width, self.height), 1e-9)
        long_side = max(self.width, self.height)
        return long_side / short_side


@dataclass(frozen=True)
class Candidate:
    split: str
    image_path: Path
    label_path: Path
    image_width: int
    image_height: int
    obj: YoloObject
    score: float
    reasons: tuple[str, ...]


def _safe_stem(text: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    return "".join(safe).strip("_") or "sample"


def _read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def _read_dataset(data_yaml: Path) -> DatasetInfo:
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    names = data.get("names", {})
    if isinstance(names, list):
        class_names = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        class_names = {int(index): str(name) for index, name in names.items()}
    else:
        raise ValueError(f"Unsupported names format in {data_yaml}")

    return DatasetInfo(root=root, names=class_names)


def _iter_label_files(dataset_root: Path, split: str) -> Iterable[Path]:
    labels_dir = dataset_root / "labels" / split
    if not labels_dir.exists():
        return []
    return sorted(path for path in labels_dir.rglob("*.txt") if path.is_file())


def _find_image(dataset_root: Path, split: str, stem: str) -> Path | None:
    images_dir = dataset_root / "images" / split
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(path for path in images_dir.rglob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return matches[0] if matches else None


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    image = _read_image(image_path)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def _parse_label(label_path: Path, names: dict[int, str]) -> list[YoloObject]:
    objects: list[YoloObject] = []
    for line_index, raw_line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if class_id not in names or width <= 0 or height <= 0:
            continue
        if any(value < 0 or value > 1 for value in (x_center, y_center, width, height)):
            continue
        objects.append(
            YoloObject(
                class_id=class_id,
                class_name=names[class_id],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                line_index=line_index,
            )
        )
    return objects


def _score_candidate(obj: YoloObject, image_path: Path, small_area_threshold: float) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    score = 1.0

    class_name = obj.class_name.lower()
    if class_name in PRIORITY_CLASS_NAMES:
        score += 4.0
        reasons.append("priority_class")
    elif class_name == "hole":
        score += 2.0
        reasons.append("compact_small_defect")

    if obj.area_ratio <= small_area_threshold:
        score += 4.0
        reasons.append("small_target")
    elif obj.area_ratio <= small_area_threshold * 2.5:
        score += 2.0
        reasons.append("medium_small_target")

    if class_name == "crack" and obj.aspect_ratio >= 3.0:
        score += 3.0
        reasons.append("thin_or_long_crack")
    elif class_name == "corrosion" and obj.aspect_ratio >= 1.5:
        score += 1.0
        reasons.append("spread_corrosion_shape")

    stem = image_path.stem.lower()
    for token in ("missed", "hardcase", "low_conf", "lowconfidence", "scratch", "craze", "hide_craze", "erosion"):
        if token in stem:
            score += 2.0
            reasons.append(f"source_{token}")
            break

    return score, tuple(reasons or ["regular_labeled_defect"])


def collect_candidates(
    data_yaml: Path,
    splits: list[str],
    class_names: set[str],
    small_area_threshold: float,
) -> list[Candidate]:
    dataset = _read_dataset(data_yaml)
    candidates: list[Candidate] = []

    for split in splits:
        for label_path in _iter_label_files(dataset.root, split):
            image_path = _find_image(dataset.root, split, label_path.stem)
            if image_path is None:
                continue
            image_size = _read_image_size(image_path)
            if image_size is None:
                continue
            image_width, image_height = image_size
            for obj in _parse_label(label_path, dataset.names):
                if class_names and obj.class_name.lower() not in class_names:
                    continue
                score, reasons = _score_candidate(obj, image_path, small_area_threshold)
                candidates.append(
                    Candidate(
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        image_width=image_width,
                        image_height=image_height,
                        obj=obj,
                        score=score,
                        reasons=reasons,
                    )
                )

    return sorted(
        candidates,
        key=lambda item: (
            item.obj.class_name,
            -item.score,
            item.image_path.name,
            item.obj.line_index,
        ),
    )


def _box_xyxy(candidate: Candidate) -> tuple[int, int, int, int]:
    obj = candidate.obj
    width = candidate.image_width
    height = candidate.image_height
    x1 = int(round((obj.x_center - obj.width / 2.0) * width))
    y1 = int(round((obj.y_center - obj.height / 2.0) * height))
    x2 = int(round((obj.x_center + obj.width / 2.0) * width))
    y2 = int(round((obj.y_center + obj.height / 2.0) * height))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _crop_xyxy(candidate: Candidate, padding: float, min_crop_size: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _box_xyxy(candidate)
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(int(round(box_width * padding)), max(0, min_crop_size - box_width) // 2)
    pad_y = max(int(round(box_height * padding)), max(0, min_crop_size - box_height) // 2)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(candidate.image_width, x2 + pad_x),
        min(candidate.image_height, y2 + pad_y),
    )


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = 255
    return mask


def _grabcut_mask(crop: np.ndarray, rect: tuple[int, int, int, int], fallback: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = rect
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if crop.shape[0] < 8 or crop.shape[1] < 8 or width < 3 or height < 3:
        return fallback

    grabcut_rect = (
        max(0, x1),
        max(0, y1),
        min(width, crop.shape[1] - x1),
        min(height, crop.shape[0] - y1),
    )
    try:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, grabcut_rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return fallback

    foreground_pixels = int((foreground > 0).sum())
    fallback_pixels = int((fallback > 0).sum())
    if foreground_pixels < max(8, int(fallback_pixels * 0.03)):
        return fallback
    return foreground


def _overlay(crop: np.ndarray, mask: np.ndarray, rect: tuple[int, int, int, int], class_name: str) -> np.ndarray:
    overlay = crop.copy()
    color_layer = np.zeros_like(crop)
    color_layer[:, :, 1] = mask
    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0.0)
    x1, y1, x2, y2 = rect
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 1)
    cv2.putText(overlay, class_name, (max(2, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return overlay


def _safe_clear_output(output: Path) -> None:
    resolved = output.resolve()
    if output.exists():
        if "shape_mask" not in resolved.name and "mask_candidates" not in resolved.name:
            raise RuntimeError(f"Refuse to overwrite unexpected output directory: {resolved}")
        shutil.rmtree(output)


def _limit_per_class(candidates: list[Candidate], max_per_class: int) -> list[Candidate]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.obj.class_name].append(candidate)

    selected: list[Candidate] = []
    for class_name in sorted(grouped):
        selected.extend(grouped[class_name][:max_per_class])
    return sorted(selected, key=lambda item: (-item.score, item.obj.class_name, item.image_path.name, item.obj.line_index))


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "split",
        "class_id",
        "class_name",
        "score",
        "selection_reason",
        "source_image",
        "source_label",
        "label_line",
        "image_width",
        "image_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "area_ratio",
        "aspect_ratio",
        "crop_path",
        "rect_mask_path",
        "pseudo_mask_path",
        "overlay_path",
        "review_status",
        "review_notes",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output: Path, rows: list[dict[str, str]], data_yaml: Path, max_per_class: int) -> None:
    by_class = Counter(row["class_name"] for row in rows)
    lines = [
        "# MSF-YOLO shape mask candidate set",
        "",
        "Purpose:",
        "",
        "- Prepare high-value defect crops for later pixel-level morphology supervision.",
        "- Keep original YOLO boxes as rectangular masks and optional GrabCut pseudo masks.",
        "- Use overlays for fast manual review before converting reviewed masks to training labels.",
        "",
        "Source:",
        "",
        f"- data_yaml: `{data_yaml}`",
        f"- max_per_class: `{max_per_class}`",
        "",
        "Counts:",
        "",
        "| Class | Candidates |",
        "| --- | ---: |",
    ]
    for class_name in sorted(by_class):
        lines.append(f"| {class_name} | {by_class[class_name]} |")
    lines.extend(
        [
            f"| total | {len(rows)} |",
            "",
            "Review workflow:",
            "",
            "1. Open `overlays/<class>/` and inspect each candidate.",
            "2. Edit masks in `masks_grabcut/<class>/` or start from `masks_rect/<class>/` when GrabCut is noisy.",
            "3. Mark `review_status` in `manifest.csv` as `accepted`, `edited`, or `rejected`.",
            "4. Use reviewed masks later to build YOLO-seg/COCO masks or edge/centerline targets for MSF-YOLO v2.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_shape_mask_candidates(
    data_yaml: Path,
    output: Path,
    splits: list[str],
    class_names: set[str],
    max_per_class: int,
    small_area_threshold: float,
    padding: float,
    min_crop_size: int,
    generate_grabcut_masks: bool,
    overwrite: bool = False,
) -> dict[str, int]:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists. Use --overwrite to replace: {output}")
        _safe_clear_output(output)

    output.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(
        data_yaml=data_yaml,
        splits=splits,
        class_names={name.lower() for name in class_names},
        small_area_threshold=small_area_threshold,
    )
    selected = _limit_per_class(candidates, max_per_class=max_per_class)

    rows: list[dict[str, str]] = []
    for index, candidate in enumerate(selected, start=1):
        image = _read_image(candidate.image_path)
        if image is None:
            continue
        x1, y1, x2, y2 = _box_xyxy(candidate)
        cx1, cy1, cx2, cy2 = _crop_xyxy(candidate, padding=padding, min_crop_size=min_crop_size)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        local_rect = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        rect_mask = _rect_mask(crop.shape[:2], local_rect)
        pseudo_mask = _grabcut_mask(crop, local_rect, rect_mask) if generate_grabcut_masks else rect_mask

        class_name = candidate.obj.class_name
        candidate_id = _safe_stem(f"{index:05d}_{class_name}_{candidate.image_path.stem}_l{candidate.obj.line_index}")
        crop_path = output / "crops" / class_name / f"{candidate_id}.jpg"
        rect_mask_path = output / "masks_rect" / class_name / f"{candidate_id}.png"
        pseudo_mask_path = output / "masks_grabcut" / class_name / f"{candidate_id}.png"
        overlay_path = output / "overlays" / class_name / f"{candidate_id}.jpg"

        crop_path.parent.mkdir(parents=True, exist_ok=True)
        rect_mask_path.parent.mkdir(parents=True, exist_ok=True)
        pseudo_mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

        _write_image(crop_path, crop)
        _write_image(rect_mask_path, rect_mask)
        _write_image(pseudo_mask_path, pseudo_mask)
        _write_image(overlay_path, _overlay(crop, pseudo_mask, local_rect, class_name))

        rows.append(
            {
                "candidate_id": candidate_id,
                "split": candidate.split,
                "class_id": str(candidate.obj.class_id),
                "class_name": class_name,
                "score": f"{candidate.score:.2f}",
                "selection_reason": ";".join(candidate.reasons),
                "source_image": str(candidate.image_path),
                "source_label": str(candidate.label_path),
                "label_line": str(candidate.obj.line_index),
                "image_width": str(candidate.image_width),
                "image_height": str(candidate.image_height),
                "bbox_x1": str(x1),
                "bbox_y1": str(y1),
                "bbox_x2": str(x2),
                "bbox_y2": str(y2),
                "crop_x1": str(cx1),
                "crop_y1": str(cy1),
                "crop_x2": str(cx2),
                "crop_y2": str(cy2),
                "area_ratio": f"{candidate.obj.area_ratio:.8f}",
                "aspect_ratio": f"{candidate.obj.aspect_ratio:.4f}",
                "crop_path": str(crop_path),
                "rect_mask_path": str(rect_mask_path),
                "pseudo_mask_path": str(pseudo_mask_path),
                "overlay_path": str(overlay_path),
                "review_status": "pending",
                "review_notes": "",
            }
        )

    _write_manifest(output / "manifest.csv", rows)
    _write_summary(output, rows, data_yaml=data_yaml, max_per_class=max_per_class)

    counts = Counter(row["class_name"] for row in rows)
    counts["total"] = len(rows)
    return dict(counts)


def _parse_csv_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare high-value crop and pseudo-mask candidates for future MSF-YOLO shape-aware training."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, type=Path, help="YOLO data.yaml path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Candidate output directory.")
    parser.add_argument("--splits", default="train", help="Comma-separated dataset splits, for example train,val.")
    parser.add_argument(
        "--classes",
        default="crack,hole,spalling,corrosion",
        help="Comma-separated class names to include.",
    )
    parser.add_argument("--max-per-class", default=80, type=int, help="Maximum candidates to keep per class.")
    parser.add_argument(
        "--small-area-threshold",
        default=0.012,
        type=float,
        help="Normalized bbox area threshold used to prioritize small defects.",
    )
    parser.add_argument("--padding", default=0.35, type=float, help="Crop padding ratio around each YOLO bbox.")
    parser.add_argument("--min-crop-size", default=96, type=int, help="Minimum crop side target in pixels.")
    parser.add_argument("--no-grabcut", action="store_true", help="Only write rectangular masks.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing candidate output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = build_shape_mask_candidates(
        data_yaml=args.data,
        output=args.output,
        splits=[item.strip() for item in args.splits.split(",") if item.strip()],
        class_names=_parse_csv_set(args.classes),
        max_per_class=args.max_per_class,
        small_area_threshold=args.small_area_threshold,
        padding=args.padding,
        min_crop_size=args.min_crop_size,
        generate_grabcut_masks=not args.no_grabcut,
        overwrite=args.overwrite,
    )
    print("Prepared MSF-YOLO shape-mask candidates:")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
