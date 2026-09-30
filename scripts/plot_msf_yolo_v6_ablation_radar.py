from __future__ import annotations

import csv
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STRICT_CSV = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "wind_fault"
    / "msf_yolo_strict_ablation"
    / "strict_ablation_summary_20260607_141729"
    / "strict_ablation_metrics.csv"
)
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
OUT_DIR = ROOT / "paper_assets_msf_yolo" / "figures"


METRICS = [
    ("val_recall", "Val Recall"),
    ("val_map50_95", "Val mAP50-95"),
    ("test_precision", "Test Precision"),
    ("test_recall", "Test Recall"),
    ("test_map50", "Test mAP50"),
    ("test_map50_95", "Test mAP50-95"),
]

DISPLAY_NAMES = {
    "MSF-YOLO V6": "MSF-YOLO V6",
    "p2_plain_control": "P2",
    "p2_shape_only": "P2+Shape",
    "p2_hard_only": "P2+Hard",
    "p2_shape_hard": "P2+Shape+Hard",
}

COLORS = {
    "MSF-YOLO V6": "#c0392b",
    "P2": "#2c7fb8",
    "P2+Shape": "#31a354",
    "P2+Hard": "#756bb1",
    "P2+Shape+Hard": "#fd8d3c",
}


def read_strict_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with STRICT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = DISPLAY_NAMES.get(row["variant"], row["variant"])
            rows.append(
                {
                    "model": name,
                    "val_recall": float(row["val_recall"]),
                    "val_map50_95": float(row["val_map50_95"]),
                    "test_precision": float(row["test_precision"]),
                    "test_recall": float(row["test_recall"]),
                    "test_map50": float(row["test_map50"]),
                    "test_map50_95": float(row["test_map50_95"]),
                }
            )
    return rows


def read_v6_row() -> dict[str, float | str]:
    by_split: dict[str, dict[str, float]] = {}
    with V6_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {str(key).strip(): value for key, value in raw_row.items()}
            if row["model"] != "MSF-YOLO V6":
                continue
            by_split[row["split"]] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
            }
    return {
        "model": "MSF-YOLO V6",
        "val_recall": by_split["val"]["recall"],
        "val_map50_95": by_split["val"]["map50_95"],
        "test_precision": by_split["test_clean"]["precision"],
        "test_recall": by_split["test_clean"]["recall"],
        "test_map50": by_split["test_clean"]["map50"],
        "test_map50_95": by_split["test_clean"]["map50_95"],
    }


def write_data_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *[key for key, _ in METRICS]])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_radar(rows: list[dict[str, float | str]], out_stem: Path) -> None:
    labels = [label for _, label in METRICS]
    metric_keys = [key for key, _ in METRICS]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#546e7a",
        }
    )

    fig = plt.figure(figsize=(8.2, 7.2), dpi=220)
    ax = plt.subplot(111, polar=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.75, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(["0.75", "0.80", "0.85", "0.90", "0.95", "1.00"], fontsize=8)
    ax.grid(color="#cfd8dc", linewidth=0.8)
    ax.spines["polar"].set_color("#78909c")
    ax.spines["polar"].set_linewidth(0.9)

    # Draw non-V6 ablations first, then V6 on top.
    ordered = [row for row in rows if row["model"] != "MSF-YOLO V6"]
    ordered += [row for row in rows if row["model"] == "MSF-YOLO V6"]
    for row in ordered:
        model_name = str(row["model"])
        values = [float(row[key]) for key in metric_keys]
        values += values[:1]
        color = COLORS.get(model_name, "#455a64")
        linewidth = 2.6 if model_name == "MSF-YOLO V6" else 1.6
        alpha = 0.13 if model_name == "MSF-YOLO V6" else 0.06
        marker = "o" if model_name == "MSF-YOLO V6" else None
        ax.plot(angles, values, color=color, linewidth=linewidth, marker=marker, markersize=3.2, label=model_name)
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.set_title("MSF-YOLO V6 vs Ablation Models (No V4 Baseline)", pad=22, fontsize=13, weight="bold")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
    )
    fig.text(
        0.5,
        0.025,
        "Metrics use raw validation/test values; radial range is 0.75-1.00 for visual separation.",
        ha="center",
        va="center",
        fontsize=8,
        color="#607d8b",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.96])

    for suffix in (".svg", ".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [read_v6_row(), *read_strict_rows()]
    write_data_csv(rows, OUT_DIR / "fig_ablation_radar_v6_no_v4_data.csv")
    plot_radar(rows, OUT_DIR / "fig_ablation_radar_v6_no_v4")


if __name__ == "__main__":
    main()
