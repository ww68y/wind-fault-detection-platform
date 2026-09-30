from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.data.dataset import coco_box_to_yolo, split_items, voc_box_to_yolo
from wind_fault.data.slice_images import tile_positions


def main() -> None:
    assert voc_box_to_yolo(10, 20, 30, 60, 100, 100) == (0.2, 0.4, 0.2, 0.4)
    assert coco_box_to_yolo(10, 20, 20, 40, 100, 100) == (0.2, 0.4, 0.2, 0.4)
    split = split_items(list(range(10)), train_ratio=0.7, val_ratio=0.2, seed=1)
    assert sum(len(value) for value in split.values()) == 10
    assert tile_positions(2500, 1024, 819) == [0, 819, 1476]
    print("Smoke check passed.")


if __name__ == "__main__":
    main()

