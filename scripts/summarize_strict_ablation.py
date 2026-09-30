# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs" / "detect" / "runs" / "wind_fault" / "msf_yolo_strict_ablation"


VARIANT_ORDER = [
    "p2_plain_control",
    "p2_shape_only",
    "p2_hard_only",
    "p2_shape_hard",
]


def _latest_analysis(run_root: Path, variant: str) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted(
        run_root.glob(f"{variant}_finetune_*_from_warmup_*/analysis/analysis.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], split: str, model: str, key: str) -> Any:
    return payload.get("splits", {}).get(split, {}).get(model, {}).get("metrics", {}).get(key)


def _fp(payload: dict[str, Any], split: str, model: str) -> Any:
    return payload.get("splits", {}).get(split, {}).get(model, {}).get("empty_false_positive", {}).get("false_positive_count")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _read_blade_scan(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_root.glob("blade_slice_scan_*/tile_*_overlap*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        for result in payload.get("results", []):
            err = result.get("error_summary", {})
            pred = result.get("prediction_summary", {})
            rows.append(
                {
                    "tile_size": cfg.get("tile_size"),
                    "overlap": cfg.get("overlap"),
                    "split": result.get("split"),
                    "mode": result.get("mode"),
                    "recall": err.get("recall"),
                    "precision": err.get("precision"),
                    "missed": err.get("missed_count"),
                    "false_positives": err.get("false_positive_count"),
                    "predictions": pred.get("total_predictions"),
                    "summary": str(summary_path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MSF-YOLO strict ablation queue outputs.")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    run_root = args.run_root
    output = args.output or (run_root / "strict_ablation_summary")
    output.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        item = _latest_analysis(run_root, variant)
        if item is None:
            metric_rows.append({"variant": variant, "status": "missing"})
            continue
        analysis_path, payload = item
        metric_rows.append(
            {
                "variant": variant,
                "status": "done",
                "analysis": str(analysis_path),
                "val_precision": _metric(payload, "val", "candidate", "precision"),
                "val_recall": _metric(payload, "val", "candidate", "recall"),
                "val_map50": _metric(payload, "val", "candidate", "map50"),
                "val_map50_95": _metric(payload, "val", "candidate", "map50_95"),
                "test_precision": _metric(payload, "test_clean", "candidate", "precision"),
                "test_recall": _metric(payload, "test_clean", "candidate", "recall"),
                "test_map50": _metric(payload, "test_clean", "candidate", "map50"),
                "test_map50_95": _metric(payload, "test_clean", "candidate", "map50_95"),
                "val_fp": _fp(payload, "val", "candidate"),
                "test_fp": _fp(payload, "test_clean", "candidate"),
                "conclusion": payload.get("initial_conclusion", ""),
            }
        )

    metrics_csv = output / "strict_ablation_metrics.csv"
    metric_headers = [
        "variant",
        "status",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "test_precision",
        "test_recall",
        "test_map50",
        "test_map50_95",
        "val_fp",
        "test_fp",
        "conclusion",
        "analysis",
    ]
    with metrics_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_headers)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({key: row.get(key, "") for key in metric_headers})

    blade_rows = _read_blade_scan(run_root)
    blade_csv = output / "blade_slice_scan_metrics.csv"
    blade_headers = ["tile_size", "overlap", "split", "mode", "recall", "precision", "missed", "false_positives", "predictions", "summary"]
    with blade_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=blade_headers)
        writer.writeheader()
        for row in blade_rows:
            writer.writerow({key: row.get(key, "") for key in blade_headers})

    lines = [
        "# MSF-YOLO 严格消融队列汇总",
        "",
        f"- run_root: `{run_root}`",
        f"- metrics_csv: `{metrics_csv}`",
        f"- blade_scan_csv: `{blade_csv}`",
        "",
        "## 训练消融指标",
        "",
        "| Variant | Status | val mAP50-95 | test_clean Recall | test_clean mAP50-95 | val FP | test FP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("variant", "")),
                    str(row.get("status", "")),
                    _fmt(row.get("val_map50_95")),
                    _fmt(row.get("test_recall")),
                    _fmt(row.get("test_map50_95")),
                    _fmt(row.get("val_fp")),
                    _fmt(row.get("test_fp")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Blade-Slice 参数扫描", ""])
    if blade_rows:
        lines.extend(
            [
                "| tile | split | Recall | Precision | Missed | FP | Predictions |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in blade_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row.get("tile_size")),
                        str(row.get("split", "")),
                        _fmt(row.get("recall")),
                        _fmt(row.get("precision")),
                        _fmt(row.get("missed")),
                        _fmt(row.get("false_positives")),
                        _fmt(row.get("predictions")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("尚未找到 Blade-Slice 参数扫描结果。")
    lines.append("")
    summary_md = output / "strict_ablation_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
