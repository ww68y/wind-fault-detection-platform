# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "wind_fault"
    / "msf_yolo_full_innovation"
    / "msf_full_p2_shape_hard_finetune_60e_amp_b12_from_warmup_20260606_182744"
)


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


def _load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        replace_whitespace=False,
        drop_whitespace=True,
    ) or [""]


class FlowPdf:
    """A simple flow-layout PDF writer to avoid overlapping floating objects."""

    def __init__(self, pdf: PdfPages, title: str) -> None:
        self.pdf = pdf
        self.title = title
        self.page_no = 0
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.y = 0.0
        self.left = 0.07
        self.right = 0.93
        self.bottom = 0.075
        self.top = 0.93
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_no += 1
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.text(self.left, 0.965, self.title, fontsize=14, fontproperties=FONT, weight="bold", color="#17212B")
        self.ax.plot([self.left, self.right], [0.945, 0.945], color="#D0D7DE", linewidth=0.8)
        self.y = self.top

    def close(self) -> None:
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _footer(self) -> None:
        assert self.fig is not None
        self.fig.text(self.left, 0.035, "MSF-YOLO V6 数学原理论文可读版", fontsize=7.5, fontproperties=FONT, color="#6A737D")
        self.fig.text(0.90, 0.035, str(self.page_no), fontsize=7.5, fontproperties=FONT, color="#6A737D")

    def ensure(self, height: float) -> None:
        if self.y - height < self.bottom:
            self.new_page()

    def space(self, height: float = 0.012) -> None:
        self.y -= height

    def heading(self, text: str, level: int = 1) -> None:
        size = 12 if level == 1 else 10.5
        height = 0.04 if level == 1 else 0.034
        self.ensure(height + 0.01)
        assert self.ax is not None
        self.ax.text(self.left, self.y, text, fontsize=size, fontproperties=FONT, weight="bold", color="#17212B", va="top")
        self.y -= height

    def paragraph(self, text: str, width: int = 52, size: float = 8.5, color: str = "#25313B") -> None:
        lines = _wrap(text, width)
        height = 0.020 * len(lines) + 0.012
        self.ensure(height)
        assert self.ax is not None
        self.ax.text(
            self.left,
            self.y,
            "\n".join(lines),
            fontsize=size,
            fontproperties=FONT,
            color=color,
            va="top",
            linespacing=1.28,
        )
        self.y -= height

    def formula(self, text: str, width: int = 78, size: float = 7.8) -> None:
        lines = _wrap(text, width)
        height = 0.024 * len(lines) + 0.033
        self.ensure(height + 0.006)
        assert self.ax is not None
        x = self.left
        y0 = self.y - height + 0.004
        self.ax.add_patch(Rectangle((x, y0), self.right - self.left, height, facecolor="#F6F8FA", edgecolor="#D0D7DE", linewidth=0.55))
        self.ax.text(
            x + 0.012,
            self.y - 0.012,
            "\n".join(lines),
            fontsize=size,
            color="#111827",
            va="top",
            family="DejaVu Sans Mono",
            linespacing=1.25,
        )
        self.y -= height + 0.012

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float] | None = None, size: float = 7.4) -> None:
        n = len(headers)
        if widths is None:
            widths = [1 / n] * n
        total_w = self.right - self.left
        col_w = [w * total_w for w in widths]

        wrapped_rows: list[list[list[str]]] = []
        for row in [headers] + rows:
            wrapped = []
            for idx, cell in enumerate(row):
                chars = max(6, int(28 * widths[idx]))
                wrapped.append(_wrap(str(cell), chars))
            wrapped_rows.append(wrapped)

        heights = [0.032 + 0.020 * (max(len(c) for c in row) - 1) for row in wrapped_rows]
        total_h = sum(heights)
        self.ensure(total_h + 0.02)
        assert self.ax is not None

        y = self.y
        for r, row in enumerate(wrapped_rows):
            h = heights[r]
            x = self.left
            for c, cell_lines in enumerate(row):
                face = "#263238" if r == 0 else ("#F4F7F9" if r % 2 == 0 else "white")
                text_color = "white" if r == 0 else "#111827"
                self.ax.add_patch(Rectangle((x, y - h), col_w[c], h, facecolor=face, edgecolor="#6A737D", linewidth=0.45))
                self.ax.text(
                    x + col_w[c] / 2,
                    y - h / 2,
                    "\n".join(cell_lines),
                    ha="center",
                    va="center",
                    fontsize=size,
                    fontproperties=FONT,
                    color=text_color,
                    linespacing=1.12,
                )
                x += col_w[c]
            y -= h
        self.y -= total_h + 0.016


def _content(config: dict[str, Any], run_dir: Path) -> list[tuple[str, list[Any]]]:
    shape = config.get("shape_config", {})
    sampler = config.get("sampler_config", {})
    args = config.get("train_args", {})
    return [
        (
            "0. 使用说明与边界",
            [
                ("p", f"实验名：{run_dir.name}"),
                ("p", "本文件采用文档流排版，避免公式框、表格和标题重叠。内容用于论文方法章节参考；当前 MSF-YOLO V6 已实现完整创新链路，但实验指标尚未超过 V4，论文中应写为方法探索与实现验证。"),
                ("f", "theta* = argmin_theta E_{i~p}[ L_det(I_i, Y_i; theta, q) ]"),
                ("t", ["模块", "阶段", "说明"], [
                    ["P2 检测头", "结构/训练", "Detect(P2,P3,P4,P5)，进入 .pt 权重"],
                    ["Shape-Aware Loss", "训练损失", "小面积/细长目标加权"],
                    ["困难样本重采样", "训练采样", "corrosion x3, crack x2, 小目标 x1.5"],
                    ["Blade-Slice", "推理/评估", "tile=640, overlap=0.25，不写入 .pt"],
                ], [0.25, 0.22, 0.53]),
            ],
        ),
        (
            "1. P2 浅层高分辨率检测分支",
            [
                ("p", "P2 的数学作用是减小最小检测 stride。stride 越小，同一小缺陷在特征图上覆盖的网格越多，越有利于保留浅层纹理、边缘和局部结构。"),
                ("f", "F_l in R^{C_l x ceil(H/s_l) x ceil(W/s_l)}"),
                ("f", "s_P2=4, s_P3=8, s_P4=16, s_P5=32"),
                ("f", "n_cell(b,l)=sqrt(w_b h_b)/s_l"),
                ("f", "Y_hat = Detect(F_P2, F_P3, F_P4, F_P5)"),
                ("t", ["层", "stride", "1024 输入特征图", "偏关注对象"], [
                    ["P2", "4", "256 x 256", "小裂纹、小腐蚀、小孔洞"],
                    ["P3", "8", "128 x 128", "小/中目标"],
                    ["P4", "16", "64 x 64", "中目标"],
                    ["P5", "32", "32 x 32", "大目标/语义"],
                ], [0.16, 0.16, 0.26, 0.42]),
                ("p", "实现位置：configs/yolov8n-p2-wind.yaml。该模块进入模型结构和 .pt 权重。"),
            ],
        ),
        (
            "2. Shape-Aware Loss：形态权重",
            [
                ("p", "Shape-Aware Loss 是框级形态先验，不是像素级边界监督。它在正样本 bbox/DFL 损失上乘形态权重 q_i，使小面积目标和细长目标获得更大梯度贡献。"),
                ("f", "A_i = w_i h_i / (W H),    R_i = max(w_i/h_i, h_i/w_i)"),
                ("f", f"S_i = clip(({shape.get('area_threshold', 0.012):.3f} - A_i) / {shape.get('area_threshold', 0.012):.3f}, 0, 1)"),
                ("f", f"E_i = clip((R_i - {shape.get('slender_ratio', 3.0):.1f}) / {shape.get('slender_ratio', 3.0):.1f}, 0, 1)"),
                ("f", f"q_i = clip(1 + {shape.get('small_gain', 0.35):.2f} S_i + {shape.get('slender_gain', 0.25):.2f} E_i, 1, {shape.get('max_shape_weight', 1.7):.1f})"),
                ("f", "L_box^shape = sum[(1-CIoU_i) t_i q_i] / sum[t_i]"),
                ("f", "L_dfl^shape = sum[DFL_i t_i q_i] / sum[t_i]"),
                ("f", "grad L_box^shape = (1/T) sum[t_i q_i grad(1-CIoU_i)]"),
                ("t", ["目标形态", "几何条件", "优化含义"], [
                    ["小腐蚀/小孔洞", "A_i 小", "增强小面积目标定位约束"],
                    ["细长裂纹", "R_i 大", "增强细长框边界和尺度约束"],
                    ["极端小且细长", "A_i 小且 R_i 大", "使用 q_max 截断，避免梯度过大"],
                ], [0.28, 0.24, 0.48]),
            ],
        ),
        (
            "3. 困难样本重采样与类别权重",
            [
                ("p", "困难样本机制改变图像被抽到的概率；类别权重改变分类 BCE 中每个类别的损失比例。两者分别作用于采样分布和分类损失。"),
                ("f", f"w_i = clip(1.0 * {sampler.get('corrosion_weight', 3.0):.1f}^I[corrosion] * {sampler.get('crack_weight', 2.0):.1f}^I[crack] * {sampler.get('small_target_weight', 1.5):.1f}^I[small], 0.05, {sampler.get('max_sample_weight', 6.0):.1f})"),
                ("f", "I[small]=1 if c in {crack, corrosion} and A_i <= 0.012"),
                ("f", "p_i = w_i / sum_j w_j,    E[m_i] = N p_i"),
                ("f", f"alpha_tilde_c = (1/n_c)^{args.get('cls_pw', 0.35)}"),
                ("f", "alpha_c = alpha_tilde_c / mean(alpha_tilde)"),
                ("f", "L_cls^cw = sum_{j,c} alpha_c BCE(s_hat_{j,c}, s_{j,c}) / sum_j t_j"),
                ("t", ["样本类型", "触发条件", "采样权重"], [
                    ["普通", "无触发", "1.0"],
                    ["裂纹", "crack", str(sampler.get("crack_weight", 2.0))],
                    ["腐蚀", "corrosion", str(sampler.get("corrosion_weight", 3.0))],
                    ["小目标", "small", str(sampler.get("small_target_weight", 1.5))],
                    ["上限", "max cap", str(sampler.get("max_sample_weight", 6.0))],
                ], [0.30, 0.38, 0.32]),
            ],
        ),
        (
            "4. 小目标增强与 Blade-Slice",
            [
                ("p", "小目标增强扩展训练分布；Blade-Slice 在推理阶段用局部窗口缓解整图缩放造成的小缺陷信息损失。二者分别作用于训练分布和推理尺度。"),
                ("f", "I' = T_phi(I),    B'_k = T_phi(B_k),    phi ~ P_aug"),
                ("t", ["增强项", "当前值", "作用"], [
                    ["imgsz", str(args.get("imgsz", 1024)), "提高输入分辨率，减少缩放损失"],
                    ["multi_scale", str(args.get("multi_scale", 0.10)), "增强尺度鲁棒性"],
                    ["mosaic", str(args.get("mosaic", 1.0)), "增加组合背景"],
                    ["copy_paste", str(args.get("copy_paste", 0.05)), "增加少数类/小目标出现概率"],
                    ["scale", str(args.get("scale", 0.60)), "改变目标相对面积"],
                ], [0.24, 0.20, 0.56]),
                ("f", "d = floor(L(1-rho)),    L=640, rho=0.25, d=480"),
                ("f", "S_x={0,d,2d,...,W-L},    S_y={0,d,2d,...,H-L}"),
                ("f", "C_{u,v}=I[u:u+L, v:v+L]"),
                ("f", "b_global=(x1+u, y1+v, x2+u, y2+v, c, p)"),
                ("f", "B_fused = NMS(B_full union B_slice, tau)"),
                ("p", "边界：Blade-Slice 是推理/评估策略，不写入 .pt；其余结构、损失和采样策略影响训练链路。"),
            ],
        ),
        (
            "5. NMS、评价指标与当前实验边界",
            [
                ("f", "remove b_j if c_i=c_j, p_i>=p_j, IoU(b_i,b_j)>=tau"),
                ("f", "Precision=TP/(TP+FP),    Recall=TP/(TP+FN)"),
                ("f", "mAP50-95 = (1/10) sum_{tau=0.50:0.05:0.95} mAP_tau"),
                ("f", "FP_bg = sum_{empty images} #{boxes with conf>=eta}"),
                ("t", ["指标", "V4", "MSF-YOLO V6", "变化"], [
                    ["val mAP50-95", "0.8093", "0.7922", "-0.0171"],
                    ["val Recall", "0.9411", "0.9071", "-0.0340"],
                    ["test_clean mAP50-95", "0.8401", "0.8301", "-0.0100"],
                    ["test_clean Recall", "0.9792", "0.9541", "-0.0251"],
                    ["test_clean 背景误检", "1", "1", "持平"],
                ], [0.34, 0.18, 0.28, 0.20]),
                ("p", "严谨写法：本文提出并实现 MSF-YOLO，用于探索 P2 小目标检测头、切片融合推理、形态感知损失和困难样本重加权在风机叶片缺陷检测中的作用。当前实验表明该链路具备完整实现基础，但在现有训练策略下尚未超过 V4 基线。"),
                ("p", "不建议写法：MSF-YOLO 显著提升召回率和工程可靠性。除非后续实验获得正向结果，否则这句话风险很高。"),
            ],
        ),
    ]


def _generate_readable_pdf(run_dir: Path, config: dict[str, Any], output_dir: Path) -> Path:
    pdf_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明_论文可读版.pdf"
    with PdfPages(pdf_path) as pdf:
        flow = FlowPdf(pdf, "MSF-YOLO V6 数学原理与公式说明")
        for title, items in _content(config, run_dir):
            flow.heading(title, level=1)
            for item in items:
                if item[0] == "p":
                    flow.paragraph(item[1])
                elif item[0] == "f":
                    flow.formula(item[1])
                elif item[0] == "t":
                    _, headers, rows, widths = item
                    flow.table(headers, rows, widths)
            flow.space(0.012)
        flow.close()
    return pdf_path


def _xml_text(text: str) -> str:
    return html.escape(text, quote=False)


def _r(text: str, *, bold: bool = False, size: int = 22, style: str = "Microsoft YaHei") -> str:
    b = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        f"{b}<w:rFonts w:ascii=\"{style}\" w:eastAsia=\"Microsoft YaHei\" w:hAnsi=\"{style}\"/>"
        f"<w:sz w:val=\"{size}\"/>"
        "</w:rPr>"
        f"<w:t xml:space=\"preserve\">{_xml_text(text)}</w:t></w:r>"
    )


def _p(text: str, *, bold: bool = False, size: int = 22, style: str = "Microsoft YaHei") -> str:
    return f"<w:p>{_r(text, bold=bold, size=size, style=style)}</w:p>"


def _heading(text: str, level: int = 1) -> str:
    size = 32 if level == 1 else 26
    return (
        "<w:p><w:pPr><w:spacing w:before=\"160\" w:after=\"80\"/></w:pPr>"
        + _r(text, bold=True, size=size)
        + "</w:p>"
    )


def _formula_p(text: str) -> str:
    return (
        "<w:p><w:pPr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F6F8FA\"/>"
        "<w:spacing w:before=\"60\" w:after=\"60\"/></w:pPr>"
        + _r(text, size=20, style="Consolas")
        + "</w:p>"
    )


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    def cell(text: str, header: bool = False) -> str:
        shade = "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"263238\"/>" if header else ""
        color = "<w:color w:val=\"FFFFFF\"/>" if header else ""
        bold = "<w:b/>" if header else ""
        return (
            "<w:tc><w:tcPr>"
            "<w:tcW w:w=\"2400\" w:type=\"dxa\"/>"
            f"{shade}"
            "</w:tcPr><w:p><w:r><w:rPr>"
            f"{bold}{color}<w:rFonts w:eastAsia=\"Microsoft YaHei\" w:ascii=\"Microsoft YaHei\"/>"
            "<w:sz w:val=\"18\"/></w:rPr>"
            f"<w:t>{_xml_text(text)}</w:t></w:r></w:p></w:tc>"
        )

    table = [
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders><w:top w:val=\"single\" w:sz=\"4\"/><w:left w:val=\"single\" w:sz=\"4\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\"/><w:right w:val=\"single\" w:sz=\"4\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\"/><w:insideV w:val=\"single\" w:sz=\"4\"/></w:tblBorders>"
        "</w:tblPr>"
    ]
    table.append("<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>")
    for row in rows:
        table.append("<w:tr>" + "".join(cell(x) for x in row) + "</w:tr>")
    table.append("</w:tbl>")
    return "".join(table)


def _generate_readable_docx(run_dir: Path, config: dict[str, Any], output_dir: Path) -> Path:
    docx_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明_论文可读版.docx"
    body: list[str] = []
    body.append(_heading("MSF-YOLO V6 数学原理与公式说明", 1))
    body.append(_p(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}"))
    body.append(_p(f"实验名：{run_dir.name}"))
    body.append(_p("说明：本文档为论文可读版，采用顺序排版，避免 PDF 中公式、表格和标题重叠。"))
    for title, items in _content(config, run_dir):
        body.append(_heading(title, 2))
        for item in items:
            if item[0] == "p":
                body.append(_p(item[1]))
            elif item[0] == "f":
                body.append(_formula_p(item[1]))
            elif item[0] == "t":
                _, headers, rows, _widths = item
                body.append(_tbl(headers, rows))

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
    return docx_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate readable MSF-YOLO principle PDF and Word docx.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "principle_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_json(run_dir / "msf_full_innovation_config.json")
    if not config:
        config = {"shape_config": {}, "sampler_config": {}, "train_args": {}}
    pdf_path = _generate_readable_pdf(run_dir, config, output_dir)
    docx_path = _generate_readable_docx(run_dir, config, output_dir)
    manifest = {
        "readable_pdf": str(pdf_path),
        "readable_docx": str(docx_path),
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "font": FONT_PATH,
        "note": "Flow-layout readable outputs. No floating table/formula layout is used.",
    }
    (output_dir / "readable_principle_outputs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pdf_path)
    print(docx_path)


if __name__ == "__main__":
    main()
