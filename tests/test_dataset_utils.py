from wind_fault.data.dataset import coco_box_to_yolo, split_items, voc_box_to_yolo
from wind_fault.data.slice_images import tile_positions


def test_voc_box_to_yolo() -> None:
    assert voc_box_to_yolo(10, 20, 30, 60, 100, 100) == (0.2, 0.4, 0.2, 0.4)


def test_coco_box_to_yolo() -> None:
    assert coco_box_to_yolo(10, 20, 20, 40, 100, 100) == (0.2, 0.4, 0.2, 0.4)


def test_split_items_keeps_all_items() -> None:
    items = list(range(10))
    split = split_items(items, train_ratio=0.7, val_ratio=0.2, seed=1)
    assert sum(len(value) for value in split.values()) == 10


def test_tile_positions_include_last_edge() -> None:
    assert tile_positions(2500, 1024, 819) == [0, 819, 1476]

