from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
ROOT = Path(__file__).resolve().parents[1]


def _register_msf_checkpoint_classes() -> None:
    """Allow torch to load checkpoints saved by train_msf_yolo_full.py as __main__."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import train_msf_yolo_full as msf_train
    except Exception:
        return

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    for name in (
        "ShapeAwareBboxLoss",
        "MSFShapeAwareDetectionLoss",
        "MSFDetectionModel",
        "MSFFullInnovationTrainer",
    ):
        if hasattr(msf_train, name):
            setattr(main_module, name, getattr(msf_train, name))


_register_msf_checkpoint_classes()


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run this script with the training venv.") from exc
    return YOLO


def _parse_data_yaml(data_yaml: Path) -> dict[str, Any]:
    content = data_yaml.read_text(encoding="utf-8").splitlines()
    payload: dict[str, Any] = {"names": {}}
    in_names = False
    for raw_line in content:
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        if raw_line.strip() == "names:":
            in_names = True
            continue
        if in_names and raw_line.startswith("  "):
            idx_text, name = raw_line.strip().split(":", 1)
            payload["names"][int(idx_text)] = name.strip()
            continue
        in_names = False
        if ":" in raw_line:
            key, value = raw_line.split(":", 1)
            payload[key.strip()] = value.strip()
    dataset_root = Path(payload["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    payload["root"] = dataset_root
    return payload


def _split_paths(data_info: dict[str, Any], split_key: str) -> tuple[Path, Path, str]:
    split_rel = str(data_info[split_key])
    images_dir = data_info["root"] / split_rel
    if "images/" in split_rel.replace("\\", "/"):
        labels_rel = split_rel.replace("\\", "/").replace("images/", "labels/", 1)
    else:
        labels_rel = split_rel
    labels_dir = data_info["root"] / labels_rel
    return images_dir, labels_dir, Path(split_rel).name


def _empty_label_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    images: list[Path] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.read_text(encoding="utf-8").strip():
            continue
        for suffix in IMAGE_SUFFIXES:
            image_path = images_dir / f"{label_path.stem}{suffix}"
            if image_path.exists():
                images.append(image_path)
                break
    return images


def _extract_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    maps_raw = getattr(box, "maps", []) if box is not None else []
    maps = list(maps_raw) if maps_raw is not None else []
    return {
        "precision": float(getattr(box, "mp", 0.0)) if box is not None else None,
        "recall": float(getattr(box, "mr", 0.0)) if box is not None else None,
        "map50": float(getattr(box, "map50", 0.0)) if box is not None else None,
        "map50_95": float(getattr(box, "map", 0.0)) if box is not None else None,
        "class_map50_95": [float(value) for value in maps],
    }


def _run_val(
    model_path: Path,
    data_yaml: Path,
    split_key: str,
    imgsz: int,
    batch: int,
    device: str | None,
    project: Path,
    name: str,
) -> dict[str, Any]:
    YOLO = _load_yolo()
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split_key,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        plots=False,
        verbose=False,
    )
    return _extract_metrics(metrics)


def _count_empty_false_positives(
    model_path: Path,
    image_paths: list[Path],
    class_names: dict[int, str],
    imgsz: int,
    conf: float,
    device: str | None,
) -> dict[str, Any]:
    YOLO = _load_yolo()
    model = YOLO(str(model_path))
    by_class: Counter[str] = Counter()
    image_rows: list[dict[str, Any]] = []
    total = 0
    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            save=False,
            verbose=False,
        )
        detections = []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
                class_name = class_names.get(class_id, str(class_id))
                confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
                by_class[class_name] += 1
                total += 1
                detections.append({"class": class_name, "confidence": round(confidence, 6)})
        if detections:
            image_rows.append({"image": str(image_path), "detections": detections})
    return {
        "empty_images": len(image_paths),
        "false_positive_count": total,
        "false_positive_by_class": dict(sorted(by_class.items())),
        "images_with_false_positive": len(image_rows),
        "items": image_rows,
    }


def _diff(newer: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = ("precision", "recall", "map50", "map50_95")
    return {
        key: None
        if newer.get(key) is None or baseline.get(key) is None
        else round(float(newer[key]) - float(baseline[key]), 6)
        for key in keys
    }


def _metric_text(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# YOLO-P2 model analysis",
        "",
        "## Models",
        "",
        f"- baseline: `{payload['baseline_model']}`",
        f"- candidate: `{payload['candidate_model']}`",
        f"- data: `{payload['data']}`",
        f"- baseline_imgsz: `{payload.get('baseline_imgsz', payload.get('imgsz'))}`",
        f"- candidate_imgsz: `{payload.get('candidate_imgsz', payload.get('imgsz'))}`",
        "",
        "## Validation metrics",
        "",
        "| Split | Model | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split_name, split_payload in payload["splits"].items():
        for model_name in ("baseline", "candidate"):
            metrics = split_payload[model_name]["metrics"]
            lines.append(
                f"| {split_name} | {model_name} | "
                f"{_metric_text(metrics.get('precision'))} | "
                f"{_metric_text(metrics.get('recall'))} | "
                f"{_metric_text(metrics.get('map50'))} | "
                f"{_metric_text(metrics.get('map50_95'))} |"
            )
        diff = split_payload["diff"]
        lines.append(
            f"| {split_name} | candidate - baseline | "
            f"{_metric_text(diff.get('precision'))} | "
            f"{_metric_text(diff.get('recall'))} | "
            f"{_metric_text(diff.get('map50'))} | "
            f"{_metric_text(diff.get('map50_95'))} |"
        )

    lines.extend(["", "## Background false positives", ""])
    lines.extend(
        [
            "| Split | Model | Empty images | FP count | Images with FP | FP by class |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for split_name, split_payload in payload["splits"].items():
        for model_name in ("baseline", "candidate"):
            fp = split_payload[model_name]["empty_false_positive"]
            lines.append(
                f"| {split_name} | {model_name} | "
                f"{fp['empty_images']} | {fp['false_positive_count']} | "
                f"{fp['images_with_false_positive']} | "
                f"{json.dumps(fp['false_positive_by_class'], ensure_ascii=False)} |"
            )

    lines.extend(["", "## Initial conclusion", ""])
    lines.append(payload["initial_conclusion"])
    lines.append("")
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _build_initial_conclusion(payload: dict[str, Any]) -> str:
    test_payload = payload["splits"].get("test_clean") or payload["splits"].get("test")
    if test_payload is None:
        return "No test split was available; use val metrics only."

    diff = test_payload["diff"]
    baseline_fp = test_payload["baseline"]["empty_false_positive"]["false_positive_count"]
    candidate_fp = test_payload["candidate"]["empty_false_positive"]["false_positive_count"]
    map_delta = diff.get("map50_95") or 0.0
    recall_delta = diff.get("recall") or 0.0
    fp_delta = candidate_fp - baseline_fp

    if map_delta > 0 and recall_delta >= 0 and fp_delta <= 0:
        return "Candidate is promising: test mAP50-95 improved without recall loss or background FP increase."
    if recall_delta > 0 and fp_delta > 0:
        return "Candidate may improve recall, but background false positives increased; do not replace V4 before manual review."
    if map_delta <= 0:
        return "Candidate does not beat V4 on test mAP50-95; keep V4 as the stable baseline."
    return "Candidate needs manual review before any model-chain change."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a finished YOLO-P2 model against the V4 baseline.")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--baseline-imgsz", default=None, type=int)
    parser.add_argument("--candidate-imgsz", default=None, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--conf", default=0.45, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    data_info = _parse_data_yaml(args.data)
    class_names = data_info["names"]
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_imgsz = args.baseline_imgsz or args.imgsz
    candidate_imgsz = args.candidate_imgsz or args.imgsz

    payload: dict[str, Any] = {
        "candidate_model": str(args.candidate),
        "baseline_model": str(args.baseline),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "baseline_imgsz": baseline_imgsz,
        "candidate_imgsz": candidate_imgsz,
        "conf": args.conf,
        "splits": {},
    }

    val_project = output_dir / "ultralytics_val"
    for split_key in args.splits:
        if split_key not in data_info:
            continue
        images_dir, labels_dir, split_name = _split_paths(data_info, split_key)
        empty_images = _empty_label_images(images_dir, labels_dir)
        print(f"Evaluating split={split_name}, empty_images={len(empty_images)}")
        baseline_metrics = _run_val(
            args.baseline,
            args.data,
            split_key,
            baseline_imgsz,
            args.batch,
            args.device,
            val_project,
            f"baseline_{split_name}",
        )
        candidate_metrics = _run_val(
            args.candidate,
            args.data,
            split_key,
            candidate_imgsz,
            args.batch,
            args.device,
            val_project,
            f"candidate_{split_name}",
        )
        baseline_fp = _count_empty_false_positives(
            args.baseline,
            empty_images,
            class_names,
            baseline_imgsz,
            args.conf,
            args.device,
        )
        candidate_fp = _count_empty_false_positives(
            args.candidate,
            empty_images,
            class_names,
            candidate_imgsz,
            args.conf,
            args.device,
        )
        payload["splits"][split_name] = {
            "baseline": {"metrics": baseline_metrics, "empty_false_positive": baseline_fp},
            "candidate": {"metrics": candidate_metrics, "empty_false_positive": candidate_fp},
            "diff": _diff(candidate_metrics, baseline_metrics),
        }

    payload["elapsed_seconds"] = round(time.time() - start, 2)
    payload["initial_conclusion"] = _build_initial_conclusion(payload)
    _write_report(output_dir, payload)
    print(f"Analysis report: {output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
