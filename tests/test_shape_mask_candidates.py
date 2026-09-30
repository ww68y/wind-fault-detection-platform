from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_shape_mask_candidates import build_shape_mask_candidates


def _write_image(path: Path, color: tuple[int, int, int] = (210, 210, 210)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((160, 200, 3), color, dtype=np.uint8)
    cv2.line(image, (70, 80), (130, 83), (20, 20, 20), 2)
    cv2.imwrite(str(path), image)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_shape_mask_candidates_writes_manifest_crops_and_masks(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "msf_shape_mask_candidates"
    _write_image(dataset / "images" / "train" / "crack_a.jpg")
    _write_text(dataset / "labels" / "train" / "crack_a.txt", "0 0.500000 0.500000 0.300000 0.040000\n")
    _write_image(dataset / "images" / "train" / "corrosion_a.jpg")
    _write_text(dataset / "labels" / "train" / "corrosion_a.txt", "3 0.500000 0.500000 0.200000 0.160000\n")
    _write_text(
        dataset / "data.yaml",
        "\n".join(
            [
                f"path: {dataset.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: crack",
                "  1: hole",
                "  2: spalling",
                "  3: corrosion",
                "",
            ]
        ),
    )

    counts = build_shape_mask_candidates(
        data_yaml=dataset / "data.yaml",
        output=output,
        splits=["train"],
        class_names={"crack", "corrosion"},
        max_per_class=5,
        small_area_threshold=0.02,
        padding=0.25,
        min_crop_size=48,
        generate_grabcut_masks=False,
        overwrite=False,
    )

    assert counts["total"] == 2
    assert counts["crack"] == 1
    assert (output / "README.md").exists()
    assert (output / "crops" / "crack").exists()
    assert (output / "masks_rect" / "crack").exists()
    assert (output / "masks_grabcut" / "crack").exists()
    assert (output / "overlays" / "crack").exists()

    with (output / "manifest.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert {row["class_name"] for row in rows} == {"crack", "corrosion"}
    assert all(row["review_status"] == "pending" for row in rows)
    assert all(Path(row["crop_path"]).exists() for row in rows)
    assert all(Path(row["rect_mask_path"]).exists() for row in rows)
    assert all(Path(row["pseudo_mask_path"]).exists() for row in rows)
