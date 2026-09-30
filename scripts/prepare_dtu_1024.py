from __future__ import annotations

import argparse
from pathlib import Path

from wind_fault.data.convert_coco_json import convert_coco_json


DTU_CLASSES = ["LE;ER", "SF;PO", "VG;MT", "LR;DA", "LE;CR"]
SPLIT_FILES = {
    "train": "train1024-s.json",
    "val": "val1024-s.json",
    "test": "test1024-s.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare DTU 1024 sliced annotations as a YOLO dataset."
    )
    parser.add_argument(
        "--annotations",
        default=Path("datasets/raw/dtu_annotations/annotations"),
        type=Path,
        help="Directory containing DTU train1024-s/val1024-s/test1024-s JSON files.",
    )
    parser.add_argument(
        "--images",
        default=Path("datasets/raw/dtu/images"),
        type=Path,
        help="Directory containing DTU image files from the Mendeley dataset.",
    )
    parser.add_argument(
        "--output",
        default=Path("datasets/dtu_1024_yolo"),
        type=Path,
        help="Output YOLO dataset directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.annotations.exists():
        raise FileNotFoundError(f"DTU annotation directory not found: {args.annotations}")
    if not args.images.exists():
        raise FileNotFoundError(
            "DTU images were not found. Download the image dataset and place the files under "
            f"{args.images}"
        )

    for split, file_name in SPLIT_FILES.items():
        annotation = args.annotations / file_name
        if not annotation.exists():
            raise FileNotFoundError(f"DTU annotation file not found: {annotation}")
        convert_coco_json(
            annotation_path=annotation,
            images_dir=args.images,
            output_dir=args.output,
            split=split,
            classes=DTU_CLASSES,
        )
        print(f"Converted DTU {split}: {annotation}")

    print(f"DTU 1024 YOLO dataset is ready: {args.output}")


if __name__ == "__main__":
    main()
