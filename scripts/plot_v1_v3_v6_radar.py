from __future__ import annotations

import csv
import json
from pathlib import Path
from math import pi

import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "wind_blade_defect_v05_conservative" / "data.yaml"
OUT_DIR = ROOT / "paper_assets_msf_yolo" / "figures"
PROBE_DIR = ROOT / "runs" / "detect" / "runs" / "wind_fault" / "metric_probe" / "v1_v3_v6_radar"
V6_CSV = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "wind_fault"
    / "msf_yolo_full_innovation"
    / "msf_full_p2_shape_hard_finetune_60e_amp_b12_from_warmup_20260606_182744"
    / "comparison_report"
    / "metrics_summary.csv"
)

MODELS = [
    {
        "name": "V1 Initial",
        "path": ROOT / "runs" / "detect" / "runs" / "wind_fault" / "gpu_640_b4_80e" / "weights" / "best.pt",
        "imgsz": 640,
    },
    {
        "name": "V3 Recall",
        "path": ROOT
        / "runs"
        / "detect"
        / "runs"
        / "wind_fault"
        / "overnight"
        / "v03_recall_832_b2_80e_20260604_224650"
        / "weights"
        / "best.pt",
        "imgsz": 832,
    },
]

METRICS = [
    ("val_recall", "Val Recall"),
    ("val_map50_95", "Val mAP50-95"),
    ("test_precision", "Test Precision"),
    ("test_recall", "Test Recall"),
    ("test_map50", "Test mAP50"),
    ("test_map50_95", "Test mAP50-95"),
]

COLORS = {
    "V1 Initial": "#2c7fb8",
    "V3 Recall": "#31a354",
    "MSF-YOLO V6": "#c0392b",
}


def extract_metrics(model_path: Path, imgsz: int, model_name: str) -> dict[str, float | str | int]:
    model = YOLO(str(model_path))
    row: dict[str, float | str | int] = {"model": model_name, "imgsz": imgsz}
    for split, prefix in (("val", "val"), ("test", "test")):
        metrics = model.val(
            data=str(DATA_YAML),
            split=split,
            imgsz=imgsz,
            batch=16,
            device="0",
            workers=0,
            project=str(PROBE_DIR),
            name=f"{model_name.lower().replace(' ', '_')}_{prefix}",
            plots=False,
            verbose=False,
        )
        box = metrics.box
        row[f"{prefix}_precision"] = float(box.mp)
        row[f"{prefix}_recall"] = float(box.mr)
        row[f"{prefix}_map50"] = float(box.map50)
        row[f"{prefix}_map50_95"] = float(box.map)
    return row


def read_v6_metrics() -> dict[str, float | str | int]:
    by_split: dict[str, dict[str, float]] = {}
    with V6_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {str(k).strip(): v for k, v in raw_row.items()}
            if row["model"] != "MSF-YOLO V6":
                continue
            split = "test" if row["split"] == "test_clean" else row["split"]
            by_split[split] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
            }
    return {
        "model": "MSF-YOLO V6",
        "imgsz": 1024,
        "val_precision": by_split["val"]["precision"],
        "val_recall": by_split["val"]["recall"],
        "val_map50": by_split["val"]["map50"],
        "val_map50_95": by_split["val"]["map50_95"],
        "test_precision": by_split["test"]["precision"],
        "test_recall": by_split["test"]["recall"],
        "test_map50": by_split["test"]["map50"],
        "test_map50_95": by_split["test"]["map50_95"],
    }


def write_rows(rows: list[dict[str, float | str | int]], out_path: Path) -> None:
    fieldnames = [
        "model",
        "imgsz",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "test_precision",
        "test_recall",
        "test_map50",
        "test_map50_95",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str | int]], out_stem: Path) -> None:
    labels = [label for _, label in METRICS]
    keys = [key for key, _ in METRICS]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(8.0, 7.1), dpi=220)
    ax = plt.subplot(111, polar=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.70, 1.00)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(["0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"], fontsize=8)
    ax.grid(color="#cfd8dc", linewidth=0.8)
    ax.spines["polar"].set_color("#78909c")

    for row in rows:
        name = str(row["model"])
        values = [float(row[key]) for key in keys]
        values += values[:1]
        color = COLORS.get(name, "#455a64")
        linewidth = 2.6 if name == "MSF-YOLO V6" else 2.0
        marker = "o" if name == "MSF-YOLO V6" else "s"
        ax.plot(angles, values, color=color, linewidth=linewidth, marker=marker, markersize=3.2, label=f"{name} ({row['imgsz']})")
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_title("V1 / V3 / MSF-YOLO V6 Radar Comparison", pad=22, fontsize=13, weight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.027,
        "All models are evaluated on wind_blade_defect_v05_conservative val/test_clean; V1/V3 use their native input sizes.",
        ha="center",
        fontsize=8,
        color="#607d8b",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.96])
    for suffix in (".svg", ".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [extract_metrics(item["path"], int(item["imgsz"]), str(item["name"])) for item in MODELS]
    rows.append(read_v6_metrics())
    (OUT_DIR / "fig_v1_v3_v6_radar_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows(rows, OUT_DIR / "fig_v1_v3_v6_radar_data.csv")
    plot(rows, OUT_DIR / "fig_v1_v3_v6_radar")


if __name__ == "__main__":
    main()
