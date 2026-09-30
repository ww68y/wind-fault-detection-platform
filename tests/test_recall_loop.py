from __future__ import annotations

import sys
from pathlib import Path

from wind_fault.reports.error_mining import mine_detection_errors
from wind_fault.reports.missed_samples import Box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_v03_recall_dataset import build_dataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def _write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_prepare_v03_dataset_adds_recall_and_negative_samples(tmp_path: Path) -> None:
    base = tmp_path / "base"
    original = tmp_path / "original"
    missed = tmp_path / "missed"
    low_confidence = tmp_path / "low_confidence"
    false_positive = tmp_path / "false_positive"
    negative = tmp_path / "negative"
    output = tmp_path / "wind_blade_defect_v03_recall"

    _touch(base / "images" / "train" / "corrosion_a.jpg")
    _write_label(base / "labels" / "train" / "corrosion_a.txt", "3 0.5 0.5 0.2 0.2\n")
    _touch(base / "images" / "train" / "small_crack.jpg")
    _write_label(base / "labels" / "train" / "small_crack.txt", "0 0.5 0.5 0.04 0.04\n")
    _touch(base / "images" / "val" / "val_a.jpg")
    _write_label(base / "labels" / "val" / "val_a.txt", "1 0.5 0.5 0.2 0.2\n")
    _touch(base / "images" / "test_clean" / "test_a.jpg")
    _write_label(base / "labels" / "test_clean" / "test_a.txt", "2 0.5 0.5 0.2 0.2\n")

    _touch(original / "images" / "test" / "miss_a.jpg")
    _write_label(original / "labels" / "test" / "miss_a.txt", "0 0.5 0.5 0.1 0.1\n")
    _touch(missed / "miss_a.jpg")
    _touch(false_positive / "normal_fp.jpg")
    _touch(negative / "normal_manual.jpg")

    counts = build_dataset(
        base_dataset=base,
        original_dataset=original,
        output=output,
        missed_dir=missed,
        low_confidence_dir=low_confidence,
        false_positive_dir=false_positive,
        negative_dirs=[negative],
        corrosion_repeats=1,
        small_target_repeats=1,
        small_target_max_area=0.012,
        small_target_class_ids=[0, 1, 2],
        overwrite=False,
    )

    assert counts["missed_sample_added"] == 1
    assert counts["corrosion_oversampled"] == 1
    assert counts["small_target_oversampled"] == 1
    assert counts["negative_samples_added"] == 2
    assert (output / "data.yaml").exists()
    assert (output / "labels" / "train" / "v03_false_positive_as_negative_normal_fp.txt").read_text(
        encoding="utf-8"
    ) == ""
    assert (output / "labels" / "train" / "v03_curated_negative_normal_manual.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_error_mining_reports_misses_low_confidence_and_false_positive() -> None:
    ground_truth = {
        "matched_low": [Box(0, "crack", (0, 0, 10, 10))],
        "missed": [Box(1, "hole", (0, 0, 10, 10))],
        "negative": [],
    }
    predictions = {
        "matched_low": [Box(0, "crack", (0, 0, 10, 10), confidence=0.31)],
        "negative": [Box(3, "corrosion", (20, 20, 30, 30), confidence=0.8)],
    }

    rows, summary = mine_detection_errors(
        ground_truth=ground_truth,
        predictions=predictions,
        iou_threshold=0.5,
        low_confidence_threshold=0.45,
    )

    assert summary["recall"] == 0.5
    assert summary["missed_by_class"] == {"hole": 1}
    assert summary["low_confidence_by_class"] == {"crack": 1}
    assert summary["false_positive_by_class"] == {"corrosion": 1}
    assert {row["error_type"] for row in rows} == {"missed", "low_confidence", "false_positive"}
