# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP50",
    "map50_95": "mAP50-95",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_chinese_font() -> tuple[FontProperties, str | None]:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = [prop.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return prop, str(font_path)
    plt.rcParams["axes.unicode_minus"] = False
    return FontProperties(), None


FONT, FONT_PATH = _find_chinese_font()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_data_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"names": {}}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names and line.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            payload["names"][int(key)] = value.strip()
            continue
        in_names = False
        if ":" in line:
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip()
    return payload


def _read_args_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line or raw_line.lstrip().startswith("#"):
            continue
        key, value = raw_line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _read_results_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                clean_key = key.strip()
                clean_value = (value or "").strip()
                if not clean_value:
                    continue
                try:
                    row[clean_key] = float(clean_value)
                except ValueError:
                    pass
            if row:
                rows.append(row)
    return rows


def _best_training_row(rows: list[dict[str, float]]) -> dict[str, float] | None:
    key = "metrics/mAP50-95(B)"
    valid = [row for row in rows if key in row and not math.isnan(row[key])]
    if not valid:
        return None
    return max(valid, key=lambda row: row[key])


def _metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.{digits}f}"


def _pct_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:+.2f} pp"


def _short_path(path_text: str, max_len: int = 88) -> str:
    text = str(path_text)
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3) :]


def _split_payload(analysis: dict[str, Any], split: str) -> dict[str, Any] | None:
    return analysis.get("splits", {}).get(split)


def _primary_split_names(analysis: dict[str, Any]) -> list[str]:
    ordered = [name for name in ("val", "test_clean", "test") if name in analysis.get("splits", {})]
    rest = [name for name in analysis.get("splits", {}) if name not in ordered]
    return ordered + rest


def _build_conclusion(analysis: dict[str, Any]) -> str:
    split_name = "test_clean" if "test_clean" in analysis.get("splits", {}) else _primary_split_names(analysis)[-1]
    payload = analysis["splits"][split_name]
    diff = payload.get("diff", {})
    candidate_fp = payload["candidate"]["empty_false_positive"]["false_positive_count"]
    baseline_fp = payload["baseline"]["empty_false_positive"]["false_positive_count"]
    fp_delta = candidate_fp - baseline_fp
    map_delta = float(diff.get("map50_95", 0.0))
    recall_delta = float(diff.get("recall", 0.0))
    if map_delta > 0 and recall_delta >= 0 and fp_delta <= 0:
        return "当前 best checkpoint 对 V4 有正向信号：test_clean mAP50-95 与 Recall 同时不低于 V4，且背景误检未增加。建议继续做人工样例复核。"
    if map_delta > 0 and fp_delta <= 0:
        return "当前 best checkpoint 的定位质量有改善，但 Recall 仍需关注。建议保留为候选，不直接替换 V4。"
    return "当前 best checkpoint 尚未超过 V4：建议作为“全创新集合模型”的阶段性实验结果保留，V4 继续作为主基线与部署候选。"


def _parse_markdown_table(lines: list[str], header_startswith: str) -> list[dict[str, str]]:
    start = None
    for index, line in enumerate(lines):
        if line.strip().startswith(header_startswith):
            start = index
            break
    if start is None or start + 2 >= len(lines):
        return []
    headers = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[start + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _read_blade_slice_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"results": []}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "results": _parse_markdown_table(lines, "| Split | Mode |"),
    }


def _table(ax: plt.Axes, data: list[list[str]], columns: list[str], bbox: list[float], font_size: int = 8) -> None:
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", colLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.get_text().set_fontproperties(FONT)
        if row == 0:
            cell.set_facecolor("#263238")
            cell.get_text().set_color("white")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F7F9")


def _new_page(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.94, title, fontsize=20, fontproperties=FONT, weight="bold", color="#18242A")
    ax.plot([0.055, 0.945], [0.915, 0.915], color="#D0D7DE", linewidth=1.0)
    return fig, ax


def _footer(fig: plt.Figure, page: int, total: int) -> None:
    fig.text(0.055, 0.035, "MSF-YOLO 实验报告", fontsize=8, fontproperties=FONT, color="#6A737D")
    fig.text(0.92, 0.035, f"{page}/{total}", fontsize=8, fontproperties=FONT, color="#6A737D")


def _add_wrapped_text(ax: plt.Axes, x: float, y: float, text: str, width: int = 58, size: int = 11, color: str = "#263238") -> None:
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        replace_whitespace=False,
        drop_whitespace=True,
    )
    ax.text(x, y, "\n".join(lines), fontsize=size, fontproperties=FONT, color=color, va="top", linespacing=1.55)


def _metric_matrix(analysis: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for split in _primary_split_names(analysis):
        payload = analysis["splits"][split]
        for model_key, model_name in (("baseline", "V4"), ("candidate", "MSF-YOLO V6")):
            metrics = payload[model_key]["metrics"]
            rows.append(
                [
                    split,
                    model_name,
                    _metric(metrics.get("precision")),
                    _metric(metrics.get("recall")),
                    _metric(metrics.get("map50")),
                    _metric(metrics.get("map50_95")),
                ]
            )
        diff = payload.get("diff", {})
        rows.append(
            [
                split,
                "Delta",
                _signed(diff.get("precision")),
                _signed(diff.get("recall")),
                _signed(diff.get("map50")),
                _signed(diff.get("map50_95")),
            ]
        )
    return rows


def _page_cover(
    pdf: PdfPages,
    analysis: dict[str, Any],
    run_dir: Path,
    data_info: dict[str, Any],
    rows: list[dict[str, float]],
    best_row: dict[str, float] | None,
) -> None:
    fig, ax = _new_page("MSF-YOLO 全创新集合模型 vs V4 基线对比报告")
    conclusion = _build_conclusion(analysis)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_name = "test_clean" if "test_clean" in analysis.get("splits", {}) else _primary_split_names(analysis)[-1]
    test_payload = analysis["splits"][split_name]
    diff = test_payload.get("diff", {})
    fp_delta = (
        test_payload["candidate"]["empty_false_positive"]["false_positive_count"]
        - test_payload["baseline"]["empty_false_positive"]["false_positive_count"]
    )
    ax.text(0.055, 0.855, "核心结论", fontsize=15, fontproperties=FONT, weight="bold", color="#18242A")
    _add_wrapped_text(ax, 0.055, 0.815, conclusion, width=74, size=12)

    cards = [
        ("test_clean mAP50-95", _pct_delta(diff.get("map50_95")), "#0B7285"),
        ("test_clean Recall", _pct_delta(diff.get("recall")), "#5C2D91"),
        ("背景误检变化", f"{fp_delta:+d}", "#9A3412"),
        ("训练最佳 epoch", "-" if best_row is None else str(int(best_row.get("epoch", 0))), "#1F6FEB"),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 0.055 + i * 0.225
        ax.add_patch(plt.Rectangle((x, 0.63), 0.195, 0.105, facecolor="#F6F8FA", edgecolor="#D0D7DE", linewidth=0.8))
        ax.text(x + 0.018, 0.705, label, fontsize=9, fontproperties=FONT, color="#57606A")
        ax.text(x + 0.018, 0.662, value, fontsize=18, fontproperties=FONT, weight="bold", color=color)

    ax.text(0.055, 0.555, "实验对象", fontsize=13, fontproperties=FONT, weight="bold", color="#18242A")
    info_rows = [
        ["V4 基线", _short_path(analysis.get("baseline_model", ""))],
        ["MSF-YOLO V6 best", _short_path(analysis.get("candidate_model", ""))],
        ["Run", run_dir.name],
        ["数据配置", _short_path(analysis.get("data", ""))],
        ["类别", ", ".join(data_info.get("names", {}).values())],
        ["报告时间", generated_at],
    ]
    _table(ax, info_rows, ["项目", "内容"], [0.055, 0.235, 0.89, 0.285], font_size=8)

    if rows:
        ax.text(
            0.055,
            0.19,
            f"训练日志：已读取 {len(rows)} 个 epoch；曲线来自 results.csv，评估指标来自 analysis.json。",
            fontsize=9,
            fontproperties=FONT,
            color="#57606A",
        )
    if FONT_PATH:
        ax.text(0.055, 0.155, f"PDF 字体：{FONT_PATH}", fontsize=8, fontproperties=FONT, color="#8C959F")
    _footer(fig, 1, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_metrics(pdf: PdfPages, analysis: dict[str, Any]) -> None:
    fig, ax = _new_page("整体指标对比")
    rows = _metric_matrix(analysis)
    _table(ax, rows, ["Split", "Model", "Precision", "Recall", "mAP50", "mAP50-95"], [0.055, 0.50, 0.89, 0.36], font_size=8)

    chart_ax = fig.add_axes([0.09, 0.11, 0.82, 0.30])
    splits = _primary_split_names(analysis)
    x = list(range(len(splits)))
    width = 0.32
    baseline = [analysis["splits"][split]["baseline"]["metrics"].get("map50_95", 0) for split in splits]
    candidate = [analysis["splits"][split]["candidate"]["metrics"].get("map50_95", 0) for split in splits]
    chart_ax.bar([value - width / 2 for value in x], baseline, width=width, label="V4", color="#3B82F6")
    chart_ax.bar([value + width / 2 for value in x], candidate, width=width, label="MSF-YOLO V6", color="#F97316")
    chart_ax.set_title("mAP50-95 对比", fontproperties=FONT, fontsize=12)
    chart_ax.set_xticks(x, splits, fontproperties=FONT)
    chart_ax.set_ylim(0, max(max(baseline), max(candidate), 0.9) * 1.08)
    chart_ax.grid(axis="y", alpha=0.25)
    chart_ax.legend(prop=FONT)
    for index, split in enumerate(splits):
        delta = candidate[index] - baseline[index]
        chart_ax.text(index, max(baseline[index], candidate[index]) + 0.008, _signed(delta), ha="center", fontproperties=FONT, fontsize=8)
    _footer(fig, 2, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_training_curves(pdf: PdfPages, rows: list[dict[str, float]], best_row: dict[str, float] | None) -> None:
    fig, ax = _new_page("训练过程曲线")
    if not rows:
        ax.text(0.055, 0.82, "未找到 results.csv，无法绘制训练曲线。", fontproperties=FONT, fontsize=12)
        _footer(fig, 3, 5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    epochs = [row["epoch"] for row in rows if "epoch" in row]
    metrics_ax = fig.add_axes([0.075, 0.50, 0.86, 0.32])
    for key, label, color in (
        ("metrics/mAP50-95(B)", "mAP50-95", "#F97316"),
        ("metrics/mAP50(B)", "mAP50", "#2563EB"),
        ("metrics/recall(B)", "Recall", "#16A34A"),
        ("metrics/precision(B)", "Precision", "#7C3AED"),
    ):
        values = [row.get(key) for row in rows if "epoch" in row]
        metrics_ax.plot(epochs, values, label=label, linewidth=1.8, color=color)
    if best_row is not None:
        best_epoch = best_row.get("epoch")
        metrics_ax.axvline(best_epoch, color="#B45309", linestyle="--", linewidth=1)
        metrics_ax.text(best_epoch, best_row.get("metrics/mAP50-95(B)", 0), f"best e{int(best_epoch)}", fontproperties=FONT, fontsize=8)
    metrics_ax.set_title("验证集指标随 epoch 变化", fontproperties=FONT, fontsize=12)
    metrics_ax.set_xlabel("Epoch", fontproperties=FONT)
    metrics_ax.grid(alpha=0.25)
    metrics_ax.legend(prop=FONT, ncol=4, fontsize=8)

    loss_ax = fig.add_axes([0.075, 0.13, 0.86, 0.28])
    for key, label, color in (
        ("train/box_loss", "train box", "#2563EB"),
        ("train/cls_loss", "train cls", "#DC2626"),
        ("train/dfl_loss", "train dfl", "#7C3AED"),
        ("val/box_loss", "val box", "#60A5FA"),
        ("val/cls_loss", "val cls", "#F87171"),
        ("val/dfl_loss", "val dfl", "#A78BFA"),
    ):
        values = [row.get(key) for row in rows if "epoch" in row]
        if any(value is not None for value in values):
            loss_ax.plot(epochs, values, label=label, linewidth=1.3, color=color)
    loss_ax.set_title("训练/验证损失曲线", fontproperties=FONT, fontsize=12)
    loss_ax.set_xlabel("Epoch", fontproperties=FONT)
    loss_ax.grid(alpha=0.25)
    loss_ax.legend(prop=FONT, ncol=3, fontsize=7)
    _footer(fig, 3, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_class_metrics(pdf: PdfPages, analysis: dict[str, Any], class_names: dict[int, str]) -> None:
    fig, ax = _new_page("分类别 mAP50-95 对比")
    splits = _primary_split_names(analysis)[:2]
    if not splits:
        ax.text(0.055, 0.82, "analysis.json 未包含 split 指标。", fontproperties=FONT, fontsize=12)
        _footer(fig, 4, 5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [class_names[index] for index in sorted(class_names)]
    for chart_index, split in enumerate(splits):
        chart_ax = fig.add_axes([0.08, 0.54 - chart_index * 0.39, 0.84, 0.29])
        payload = analysis["splits"][split]
        baseline = payload["baseline"]["metrics"].get("class_map50_95", [])
        candidate = payload["candidate"]["metrics"].get("class_map50_95", [])
        x = list(range(len(labels)))
        width = 0.35
        chart_ax.bar([value - width / 2 for value in x], baseline, width=width, label="V4", color="#3B82F6")
        chart_ax.bar([value + width / 2 for value in x], candidate, width=width, label="MSF-YOLO V6", color="#F97316")
        chart_ax.set_title(f"{split} 分类别 mAP50-95", fontproperties=FONT, fontsize=12)
        chart_ax.set_xticks(x, labels, fontproperties=FONT)
        chart_ax.set_ylim(0, max(max(baseline or [0]), max(candidate or [0]), 1.0) * 1.05)
        chart_ax.grid(axis="y", alpha=0.25)
        chart_ax.legend(prop=FONT, fontsize=8)
        for idx, _label in enumerate(labels):
            if idx < len(baseline) and idx < len(candidate):
                chart_ax.text(idx, max(baseline[idx], candidate[idx]) + 0.015, _signed(candidate[idx] - baseline[idx], 3), ha="center", fontsize=7, fontproperties=FONT)
    _footer(fig, 4, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_fp_and_blade(pdf: PdfPages, analysis: dict[str, Any], blade_summary: dict[str, Any], args_info: dict[str, str]) -> None:
    fig, ax = _new_page("背景误检与 Blade-Slice 评估")
    fp_rows: list[list[str]] = []
    splits = _primary_split_names(analysis)
    for split in splits:
        payload = analysis["splits"][split]
        for model_key, model_name in (("baseline", "V4"), ("candidate", "MSF-YOLO V6")):
            fp = payload[model_key]["empty_false_positive"]
            fp_rows.append(
                [
                    split,
                    model_name,
                    str(fp["empty_images"]),
                    str(fp["false_positive_count"]),
                    str(fp["images_with_false_positive"]),
                    json.dumps(fp["false_positive_by_class"], ensure_ascii=False),
                ]
            )
    _table(ax, fp_rows, ["Split", "Model", "Empty", "FP", "Images FP", "Class"], [0.055, 0.57, 0.89, 0.27], font_size=7)

    blade_rows = blade_summary.get("results", [])
    ax.text(0.055, 0.505, "Blade-Slice 结果", fontsize=12, fontproperties=FONT, weight="bold", color="#18242A")
    if blade_rows:
        table_rows = [
            [
                row.get("Split", ""),
                row.get("Mode", ""),
                row.get("Recall", ""),
                row.get("Precision", ""),
                row.get("Missed", ""),
                row.get("False positives", ""),
                row.get("Predictions", ""),
            ]
            for row in blade_rows
        ]
        _table(ax, table_rows, ["Split", "Mode", "Recall", "Precision", "Missed", "FP", "Pred"], [0.055, 0.22, 0.89, 0.25], font_size=7)
    else:
        ax.text(0.055, 0.455, "尚未找到 Blade-Slice summary.md；本页只包含标准评估的背景误检统计。", fontproperties=FONT, fontsize=10)

    setup_text = (
        "训练集成：YOLO-P2 小目标检测头、困难样本重加权、类别权重、Shape-Aware Loss、小目标增强。"
        "Blade-Slice 属于推理/评估策略，不会写入 .pt 权重。"
    )
    if args_info:
        setup_text += f" 训练参数摘要：imgsz={args_info.get('imgsz', '-')}, batch={args_info.get('batch', '-')}, epochs={args_info.get('epochs', '-')}, optimizer={args_info.get('optimizer', '-')}。"
    _add_wrapped_text(ax, 0.055, 0.165, setup_text, width=105, size=9, color="#57606A")
    _footer(fig, 5, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(path: Path, analysis: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "model", "precision", "recall", "map50", "map50_95"])
        for split in _primary_split_names(analysis):
            payload = analysis["splits"][split]
            for model_key, model_name in (("baseline", "V4"), ("candidate", "MSF-YOLO V6")):
                metrics = payload[model_key]["metrics"]
                writer.writerow([split, model_name] + [metrics.get(key) for key in METRIC_KEYS])
            diff = payload.get("diff", {})
            writer.writerow([split, "delta"] + [diff.get(key) for key in METRIC_KEYS])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a PDF report comparing MSF-YOLO best checkpoint against V4.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--blade-summary", type=Path)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    analysis_json = args.analysis_json or run_dir / "analysis" / "analysis.json"
    blade_summary_path = args.blade_summary or run_dir / "blade_slice_eval" / "summary.md"
    data_yaml = args.data_yaml or _repo_root() / "datasets" / "wind_blade_defect_v05_conservative" / "data.yaml"
    output_dir = args.output_dir or run_dir / "comparison_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not analysis_json.exists():
        raise FileNotFoundError(f"analysis.json not found: {analysis_json}")

    analysis = _read_json(analysis_json)
    data_info = _read_data_yaml(data_yaml)
    args_info = _read_args_yaml(run_dir / "args.yaml")
    rows = _read_results_csv(run_dir / "results.csv")
    best_row = _best_training_row(rows)
    blade_summary = _read_blade_slice_summary(blade_summary_path)

    pdf_path = output_dir / "MSF-YOLO_V6_vs_V4_对比报告.pdf"
    with PdfPages(pdf_path) as pdf:
        _page_cover(pdf, analysis, run_dir, data_info, rows, best_row)
        _page_metrics(pdf, analysis)
        _page_training_curves(pdf, rows, best_row)
        _page_class_metrics(pdf, analysis, data_info.get("names", {}))
        _page_fp_and_blade(pdf, analysis, blade_summary, args_info)

    _write_summary_csv(output_dir / "metrics_summary.csv", analysis)
    manifest = {
        "pdf": str(pdf_path),
        "analysis_json": str(analysis_json),
        "blade_summary": str(blade_summary_path) if blade_summary_path.exists() else None,
        "results_csv": str(run_dir / "results.csv"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "font": FONT_PATH,
        "conclusion": _build_conclusion(analysis),
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pdf_path)


if __name__ == "__main__":
    main()
