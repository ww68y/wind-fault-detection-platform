from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SLICE_PATTERN = re.compile(r"(.+?)_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
DEFAULT_ANNOTATIONS = Path("datasets/raw/dtu_annotations/annotations")
DEFAULT_OUTPUT = Path("datasets/raw/dtu/images")


def build_source_index(source_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in source_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem.lower()
        if stem not in index:
            index[stem] = path
    return index


def required_slices(annotation_dir: Path) -> list[str]:
    names: set[str] = set()
    for file_name in ("train1024-s.json", "val1024-s.json", "test1024-s.json"):
        annotation_path = annotation_dir / file_name
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for image in payload.get("images", []):
            slice_name = str(image.get("file_name", ""))
            if SLICE_PATTERN.match(slice_name):
                names.add(slice_name)
    return sorted(names)


def save_slice(source: Path, output: Path, row: int, col: int, tile_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        left = col * tile_size
        top = row * tile_size
        right = left + tile_size
        bottom = top + tile_size

        tile = Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))
        crop = image.crop((left, top, min(right, image.width), min(bottom, image.height)))
        tile.paste(crop, (0, 0))
        tile.save(output, quality=95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DTU 1024 slice images from the high-resolution Mendeley images."
    )
    parser.add_argument("--source", required=True, type=Path, help="Directory containing HR DTU images.")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--tile-size", default=1024, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source image directory not found: {args.source}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotation directory not found: {args.annotations}")

    source_index = build_source_index(args.source)
    slices = required_slices(args.annotations)
    missing_sources: set[str] = set()
    created = 0
    skipped = 0

    for slice_name in slices:
        match = SLICE_PATTERN.match(slice_name)
        if not match:
            continue
        base_name = match.group(1).lower()
        row = int(match.group(2))
        col = int(match.group(3))
        source = source_index.get(base_name)
        if source is None:
            missing_sources.add(base_name)
            continue

        output = args.output / slice_name
        if output.exists() and output.stat().st_size > 0:
            skipped += 1
            continue
        save_slice(source, output, row=row, col=col, tile_size=args.tile_size)
        created += 1

    if missing_sources:
        examples = ", ".join(sorted(missing_sources)[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_sources)} source HR images. Examples: {examples}"
        )

    print(f"Required slices: {len(slices)}")
    print(f"Created slices: {created}")
    print(f"Skipped existing slices: {skipped}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
