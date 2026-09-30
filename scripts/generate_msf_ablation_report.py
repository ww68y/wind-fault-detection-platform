# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "wind_fault"
    / "msf_yolo_full_innovation"
    / "msf_full_p2_shape_hard_finetune_60e_amp_b12_from_warmup_20260606_182744"
)
OUT_DIR = RUN_DIR / "ablation_report"


def find_font() -> FontProperties:
    for font_path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = [prop.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    plt.rcParams["axes.unicode_minus"] = False
    return FontProperties()


FONT = find_font()


STANDARD_ROWS = [
    {
        "id": "A0",
        "experiment": "V4 baseline",
        "p2": "否",
        "shape_loss": "否",
        "hard_reweight": "否",
        "small_aug": "常规1024",
        "blade_slice": "否",
        "val_precision": 0.958748,
        "val_recall": 0.941098,
        "val_map50": 0.958317,
        "val_map50_95": 0.809276,
        "test_precision": 0.958652,
        "test_recall": 0.979213,
        "test_map50": 0.979939,
        "test_map50_95": 0.840088,
        "background_fp": "val 0 / test_clean 1",
        "conclusion": "当前稳定基线",
    },
    {
        "id": "A1",
        "experiment": "V5.1 hard reweight c3_crack2",
        "p2": "否",
        "shape_loss": "否",
        "hard_reweight": "是",
        "small_aug": "常规1024",
        "blade_slice": "否",
        "val_precision": 0.953004,
        "val_recall": 0.906663,
        "val_map50": 0.951125,
        "val_map50_95": 0.810248,
        "test_precision": 0.944713,
        "test_recall": 0.961271,
        "test_map50": 0.976619,
        "test_map50_95": 0.840086,
        "background_fp": "val 1 / test_clean 1",
        "conclusion": "mAP接近V4，但Recall下降",
    },
    {
        "id": "A2",
        "experiment": "YOLO-P2 staged finetune",
        "p2": "是",
        "shape_loss": "否",
        "hard_reweight": "否",
        "small_aug": "常规1024",
        "blade_slice": "否",
        "val_precision": 0.965401,
        "val_recall": 0.906403,
        "val_map50": 0.950276,
        "val_map50_95": 0.792186,
        "test_precision": 0.966490,
        "test_recall": 0.958282,
        "test_map50": 0.974561,
        "test_map50_95": 0.819964,
        "background_fp": "val 0 / test_clean 1",
        "conclusion": "Precision上升但Recall和mAP下降",
    },
    {
        "id": "A3",
        "experiment": "MSF-YOLO V6 full innovation",
        "p2": "是",
        "shape_loss": "是",
        "hard_reweight": "是",
        "small_aug": "增强1024",
        "blade_slice": "训练后评估",
        "val_precision": 0.951690,
        "val_recall": 0.907099,
        "val_map50": 0.939834,
        "val_map50_95": 0.792203,
        "test_precision": 0.946405,
        "test_recall": 0.954102,
        "test_map50": 0.972500,
        "test_map50_95": 0.830064,
        "background_fp": "val 2 / test_clean 1",
        "conclusion": "完整链路成立，但综合指标未超V4",
    },
]

HARD_ROWS = [
    ["V4 negative", 0.957717, 0.979439, 0.979917, 0.840210, 0.8613, 0.8945, 0.9720, 0.6330, "val 0 / test 1"],
    ["V5.1 c4_crack2", 0.928644, 0.969080, 0.959200, 0.830396, 0.8853, 0.8946, 0.9814, 0.5603, "val 2 / test 1"],
    ["V5.1 c3_crack2", 0.944713, 0.961271, 0.976619, 0.840086, 0.8749, 0.8925, 0.9795, 0.6134, "val 1 / test 1"],
    ["V5.1 c4_crack1", 0.950160, 0.951758, 0.977855, 0.838515, 0.8702, 0.8870, 0.9826, 0.6142, "val 0 / test 1"],
]

BLADE_ROWS = [
    ["val", "normal", 0.868624, 0.899200, 85, 63, 625],
    ["val", "blade_slice_fused", 0.867079, 0.880691, 86, 76, 637],
    ["test_clean", "normal", 0.888325, 0.903614, 66, 56, 581],
    ["test_clean", "blade_slice_fused", 0.890017, 0.872305, 65, 77, 603],
]

PENDING_ROWS = [
    ["B1", "P2 + Shape-Aware only", "隔离形态损失贡献，其他采样权重保持普通配置", "待补跑"],
    ["B2", "P2 + hard reweight only", "隔离困难样本重加权对P2结构的影响", "待补跑"],
    ["B3", "P2 + Shape-Aware + hard reweight", "不启用Blade-Slice，仅看训练链路组合贡献", "待补跑"],
    ["B4", "V4 + Blade-Slice 参数扫描", "tile=512/768/896，切片分支单独置信度门控", "待补跑"],
]


def fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_csv(path: Path, headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def build_markdown() -> str:
    lines: list[str] = []
    lines.append("# MSF-YOLO 消融实验补充报告（初版）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append("为支撑论文中“创新模块有效性分析”部分，本报告将项目中已经完成的实验整理为消融实验初版。当前报告只使用已有训练与评估结果，不编造未单独完成的实验数据。")
    lines.append("")
    lines.append("需要特别说明：Blade-Slice 是推理/评估策略，不写入 `.pt` 权重；Shape-Aware Loss 当前只在 V6 全创新组合中出现，尚缺少单独 Shape-only 的严格隔离实验。")
    lines.append("")
    lines.append("## 2. 消融实验矩阵")
    lines.append("")
    headers = ["编号", "实验项", "P2", "Shape Loss", "困难样本", "小目标增强", "Blade-Slice", "结论"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in STANDARD_ROWS:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["experiment"],
                    row["p2"],
                    row["shape_loss"],
                    row["hard_reweight"],
                    row["small_aug"],
                    row["blade_slice"],
                    row["conclusion"],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 3. 标准 val/test_clean 指标")
    lines.append("")
    headers = ["编号", "模型", "val P", "val R", "val mAP50", "val mAP50-95", "test P", "test R", "test mAP50", "test mAP50-95", "背景误检"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in STANDARD_ROWS:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["experiment"],
                    fmt(row["val_precision"]),
                    fmt(row["val_recall"]),
                    fmt(row["val_map50"]),
                    fmt(row["val_map50_95"]),
                    fmt(row["test_precision"]),
                    fmt(row["test_recall"]),
                    fmt(row["test_map50"]),
                    fmt(row["test_map50_95"]),
                    row["background_fp"],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("主要观察：V5.1 困难样本重加权的 `c3_crack2` 在 test_clean mAP50-95 上最接近 V4，但 Recall 下降；P2 分支在当前训练策略下没有带来小目标收益；V6 完整链路虽然包含全部创新点，但综合 mAP 与 Recall 仍低于 V4。")
    lines.append("")
    lines.append("## 4. 困难样本重加权专项消融")
    lines.append("")
    headers = ["模型", "P", "R", "mAP50", "mAP50-95", "crack", "hole", "spalling", "corrosion", "背景误检"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in HARD_ROWS:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    lines.append("")
    lines.append("结论：困难样本重加权可以改善部分 crack 指标，但 corrosion 指标没有稳定提升。过强的 corrosion 权重还会带来 Precision 与背景误检风险，因此不能直接作为正式模型替换策略。")
    lines.append("")
    lines.append("## 5. Blade-Slice 推理消融")
    lines.append("")
    lines.append("该表采用固定阈值检测口径：conf=0.45、eval_iou=0.5。它不是 mAP 指标，不能与上表的 Ultralytics mAP 直接混算。")
    lines.append("")
    headers = ["Split", "Mode", "Recall", "Precision", "Missed", "False positives", "Predictions"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in BLADE_ROWS:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    lines.append("")
    lines.append("结论：V4 + Blade-Slice 在 test_clean 上只多召回 1 个样本，但 false positives 从 56 增加到 77，主要新增在 crack 类，默认参数下不适合写成有效提升。")
    lines.append("")
    lines.append("## 6. 论文中建议写法")
    lines.append("")
    lines.append("可以写：本文围绕 P2 小目标检测头、Shape-Aware Loss、困难样本重加权和 Blade-Slice 推理策略构建了 MSF-YOLO 消融实验。实验表明，各模块均已完成工程实现与评估闭环，但在当前数据划分和训练策略下，完整组合模型尚未超过 V4 稳定基线。")
    lines.append("")
    lines.append("不建议写：MSF-YOLO V6 相比 V4 显著提升检测精度。当前数据不支持这个结论。")
    lines.append("")
    lines.append("## 7. 严格论文版仍需补跑的消融")
    lines.append("")
    headers = ["编号", "补跑项", "目的", "状态"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in PENDING_ROWS:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("当前这版可作为论文初稿中的“消融实验记录/阶段性结果”，但如果论文需要强结论，应继续补跑 Shape-only 与 P2+Hard/Shape 的严格隔离实验。")
    return "\n".join(lines) + "\n"


def make_charts() -> list[Path]:
    chart_paths: list[Path] = []
    names = [r["experiment"].replace("V5.1 hard reweight ", "Hard ").replace("YOLO-P2 staged finetune", "P2").replace("MSF-YOLO V6 full innovation", "V6") for r in STANDARD_ROWS]
    x = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = [r["test_map50_95"] for r in STANDARD_ROWS]
    bars = ax.bar(x, vals, color=["#27343B", "#2F80ED", "#8E5CF7", "#D97706"])
    ax.set_xticks(x, names, rotation=12, ha="right", fontproperties=FONT)
    ax.set_ylabel("test_clean mAP50-95", fontproperties=FONT)
    ax.set_ylim(0.78, 0.85)
    ax.grid(axis="y", color="#E5E7EB")
    ax.set_title("标准消融：test_clean mAP50-95", fontproperties=FONT, fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    p = OUT_DIR / "standard_ablation_map50_95.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    chart_paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = [r["test_recall"] for r in STANDARD_ROWS]
    bars = ax.bar(x, vals, color=["#27343B", "#2F80ED", "#8E5CF7", "#D97706"])
    ax.set_xticks(x, names, rotation=12, ha="right", fontproperties=FONT)
    ax.set_ylabel("test_clean Recall", fontproperties=FONT)
    ax.set_ylim(0.93, 0.99)
    ax.grid(axis="y", color="#E5E7EB")
    ax.set_title("标准消融：test_clean Recall", fontproperties=FONT, fontsize=13)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    p = OUT_DIR / "standard_ablation_recall.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    chart_paths.append(p)

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    labels = ["val normal", "val fused", "test normal", "test fused"]
    recall = [r[2] for r in BLADE_ROWS]
    fp = [r[5] for r in BLADE_ROWS]
    x = list(range(len(labels)))
    ax1.bar([i - 0.18 for i in x], recall, width=0.36, color="#2563EB", label="Recall")
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], fp, width=0.36, color="#DC2626", alpha=0.78, label="False positives")
    ax1.set_xticks(x, labels, rotation=12, ha="right", fontproperties=FONT)
    ax1.set_ylim(0.84, 0.91)
    ax2.set_ylim(0, 90)
    ax1.set_ylabel("Recall", fontproperties=FONT)
    ax2.set_ylabel("False positives", fontproperties=FONT)
    ax1.set_title("Blade-Slice 推理消融：召回与误检", fontproperties=FONT, fontsize=13)
    ax1.grid(axis="y", color="#E5E7EB")
    fig.tight_layout()
    p = OUT_DIR / "blade_slice_recall_fp.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    chart_paths.append(p)
    return chart_paths


class FlowPdf:
    def __init__(self, path: Path):
        self.path = path
        self.pdf = PdfPages(path)
        self.fig = None
        self.ax = None
        self.y = 0.92
        self.page = 0
        self.left = 0.07
        self.right = 0.93
        self.bottom = 0.08
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page += 1
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.text(self.left, 0.965, "MSF-YOLO 消融实验补充报告（初版）", fontproperties=FONT, fontsize=14, weight="bold")
        self.ax.plot([self.left, self.right], [0.945, 0.945], color="#D0D7DE", linewidth=0.8)
        self.y = 0.92

    def footer(self):
        self.fig.text(self.left, 0.035, "MSF-YOLO ablation supplement", fontsize=7.5, color="#6A737D")
        self.fig.text(0.90, 0.035, str(self.page), fontsize=7.5, color="#6A737D")

    def ensure(self, h: float):
        if self.y - h < self.bottom:
            self.new_page()

    def heading(self, text: str):
        self.ensure(0.05)
        self.ax.text(self.left, self.y, text, fontproperties=FONT, fontsize=12, weight="bold", va="top")
        self.y -= 0.04

    def paragraph(self, text: str):
        lines = []
        line = ""
        for ch in text:
            line += ch
            if len(line) >= 45:
                lines.append(line)
                line = ""
        if line:
            lines.append(line)
        h = 0.022 * len(lines) + 0.01
        self.ensure(h)
        self.ax.text(self.left, self.y, "\n".join(lines), fontproperties=FONT, fontsize=8.8, va="top", linespacing=1.25, color="#25313B")
        self.y -= h

    def table(self, headers: list[str], rows: list[list[object]], widths: list[float], size: float = 7.2):
        total_h = 0.034 * (len(rows) + 1) + 0.016
        self.ensure(total_h)
        total_w = self.right - self.left
        col_w = [w * total_w for w in widths]
        y = self.y
        all_rows = [headers] + [[fmt(c) for c in r] for r in rows]
        for ridx, row in enumerate(all_rows):
            x = self.left
            h = 0.034
            for cidx, cell in enumerate(row):
                face = "#263238" if ridx == 0 else ("#F6F8FA" if ridx % 2 else "white")
                color = "white" if ridx == 0 else "#111827"
                self.ax.add_patch(Rectangle((x, y - h), col_w[cidx], h, facecolor=face, edgecolor="#667085", linewidth=0.4))
                self.ax.text(x + col_w[cidx] / 2, y - h / 2, str(cell), fontproperties=FONT, fontsize=size, ha="center", va="center", color=color)
                x += col_w[cidx]
            y -= h
        self.y -= total_h

    def image(self, path: Path, height: float = 0.30):
        self.ensure(height + 0.02)
        img = plt.imread(path)
        ax_img = self.fig.add_axes([self.left, self.y - height, self.right - self.left, height])
        ax_img.imshow(img)
        ax_img.axis("off")
        self.y -= height + 0.02

    def close(self):
        self.footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.pdf.close()


def write_pdf(path: Path, chart_paths: list[Path]) -> None:
    pdf = FlowPdf(path)
    pdf.heading("1. 实验目的与边界")
    pdf.paragraph("本报告整理项目中已经完成的 MSF-YOLO 相关实验，作为论文消融实验初版。报告只使用已有实验数据；尚未单独完成的 Shape-only 等严格隔离实验列入待补跑项。")
    pdf.paragraph("Blade-Slice 是推理/评估策略，不写入 .pt 权重，因此其结果采用固定阈值检测口径单独统计，不与 mAP 表直接混算。")
    pdf.heading("2. 标准指标消融")
    pdf.table(
        ["ID", "实验项", "P2", "Shape", "Hard", "test R", "test mAP50-95"],
        [[r["id"], r["experiment"], r["p2"], r["shape_loss"], r["hard_reweight"], r["test_recall"], r["test_map50_95"]] for r in STANDARD_ROWS],
        [0.06, 0.30, 0.08, 0.09, 0.10, 0.15, 0.22],
        size=6.5,
    )
    pdf.paragraph("结果显示：V5.1 c3_crack2 的 test_clean mAP50-95 最接近 V4，但 Recall 下降；P2 分支与 V6 完整组合均未超过 V4。")
    pdf.image(chart_paths[0], 0.29)
    pdf.image(chart_paths[1], 0.29)
    pdf.heading("3. 困难样本重加权专项")
    pdf.table(
        ["模型", "R", "mAP50-95", "crack", "corrosion", "背景误检"],
        [[r[0], r[2], r[4], r[5], r[8], r[9]] for r in HARD_ROWS],
        [0.28, 0.12, 0.15, 0.13, 0.15, 0.17],
        size=6.8,
    )
    pdf.paragraph("困难样本重加权能改善部分 crack 指标，但 corrosion 未稳定提升，过强权重还会带来 Precision 和误检风险。")
    pdf.heading("4. Blade-Slice 推理消融")
    pdf.table(
        ["Split", "Mode", "Recall", "Precision", "Missed", "FP"],
        [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in BLADE_ROWS],
        [0.16, 0.25, 0.16, 0.16, 0.13, 0.12],
        size=6.8,
    )
    pdf.image(chart_paths[2], 0.30)
    pdf.paragraph("V4 + Blade-Slice 在 test_clean 上只多召回 1 个样本，但 false positives 从 56 增加到 77，默认参数下不适合作为有效提升结论。")
    pdf.heading("5. 论文结论建议")
    pdf.paragraph("可写为：本文构建了包含 P2、Shape-Aware Loss、困难样本重加权与 Blade-Slice 的 MSF-YOLO 消融实验体系，并完成工程验证。当前实验说明完整链路可运行，但综合指标尚未超过 V4 稳定基线。")
    pdf.paragraph("不建议写为：MSF-YOLO V6 相比 V4 显著提升检测精度。当前数据不支持该结论。")
    pdf.heading("6. 待补跑项")
    pdf.table(["编号", "补跑项", "状态"], [[r[0], r[1], r[3]] for r in PENDING_ROWS], [0.12, 0.68, 0.20], size=7.0)
    pdf.close()


def docx_xml_paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{html.escape(text)}</w:t></w:r></w:p>"


def write_docx(path: Path, markdown: str) -> None:
    paragraphs = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            paragraphs.append(docx_xml_paragraph(line[2:], "Title"))
        elif line.startswith("## "):
            paragraphs.append(docx_xml_paragraph(line[3:], "Heading1"))
        elif line.startswith("|"):
            paragraphs.append(docx_xml_paragraph(line))
        elif line.strip():
            paragraphs.append(docx_xml_paragraph(line))
        else:
            paragraphs.append("<w:p/>")
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(paragraphs)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown()
    md_path = OUT_DIR / "MSF-YOLO_消融实验补充报告_初版.md"
    md_path.write_text(markdown, encoding="utf-8")

    write_csv(
        OUT_DIR / "standard_ablation_metrics.csv",
        [
            "id",
            "experiment",
            "p2",
            "shape_loss",
            "hard_reweight",
            "small_aug",
            "blade_slice",
            "val_precision",
            "val_recall",
            "val_map50",
            "val_map50_95",
            "test_precision",
            "test_recall",
            "test_map50",
            "test_map50_95",
            "background_fp",
            "conclusion",
        ],
        [[row[k] for k in ["id", "experiment", "p2", "shape_loss", "hard_reweight", "small_aug", "blade_slice", "val_precision", "val_recall", "val_map50", "val_map50_95", "test_precision", "test_recall", "test_map50", "test_map50_95", "background_fp", "conclusion"]] for row in STANDARD_ROWS],
    )
    write_csv(OUT_DIR / "hard_reweight_ablation.csv", ["model", "precision", "recall", "map50", "map50_95", "crack", "hole", "spalling", "corrosion", "background_fp"], HARD_ROWS)
    write_csv(OUT_DIR / "blade_slice_ablation.csv", ["split", "mode", "recall", "precision", "missed", "false_positives", "predictions"], BLADE_ROWS)
    write_csv(OUT_DIR / "pending_strict_ablation.csv", ["id", "experiment", "purpose", "status"], PENDING_ROWS)

    chart_paths = make_charts()
    pdf_path = OUT_DIR / "MSF-YOLO_消融实验补充报告_初版.pdf"
    write_pdf(pdf_path, chart_paths)
    docx_path = OUT_DIR / "MSF-YOLO_消融实验补充报告_初版.docx"
    write_docx(docx_path, markdown)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "markdown": str(md_path),
        "pdf": str(pdf_path),
        "docx": str(docx_path),
        "csv": [
            str(OUT_DIR / "standard_ablation_metrics.csv"),
            str(OUT_DIR / "hard_reweight_ablation.csv"),
            str(OUT_DIR / "blade_slice_ablation.csv"),
            str(OUT_DIR / "pending_strict_ablation.csv"),
        ],
        "charts": [str(p) for p in chart_paths],
    }
    (OUT_DIR / "ablation_report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md_path)
    print(pdf_path)
    print(docx_path)


if __name__ == "__main__":
    main()
