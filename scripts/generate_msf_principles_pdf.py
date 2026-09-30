# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle


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


def _wrap(text: str, width: int = 58) -> str:
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=True,
        )
    )


def _new_page(title: str, page: int, total: int) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.94, title, fontsize=19, fontproperties=FONT, weight="bold", color="#17212B")
    ax.plot([0.055, 0.945], [0.915, 0.915], color="#D0D7DE", linewidth=1.0)
    fig.text(0.055, 0.035, "MSF-YOLO 数学原理说明", fontsize=8, fontproperties=FONT, color="#6A737D")
    fig.text(0.92, 0.035, f"{page}/{total}", fontsize=8, fontproperties=FONT, color="#6A737D")
    return fig, ax


def _txt(ax: plt.Axes, x: float, y: float, text: str, width: int = 74, size: int = 10.5, color: str = "#25313B") -> None:
    ax.text(x, y, _wrap(text, width), fontsize=size, fontproperties=FONT, color=color, va="top", linespacing=1.45)


def _formula(ax: plt.Axes, x: float, y: float, formula: str, size: int = 13) -> None:
    ax.text(x, y, formula, fontsize=size, color="#111827", va="top")


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


def _box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str = "#F6F8FA") -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#8C959F", linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, _wrap(text, 12), ha="center", va="center", fontsize=9, fontproperties=FONT)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#57606A"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cover(pdf: PdfPages, run_dir: Path, config: dict[str, Any], page: int, total: int) -> None:
    fig, ax = _new_page("MSF-YOLO V6 数学原理与公式说明", page, total)
    _txt(
        ax,
        0.055,
        0.86,
        "本文档面向论文写作，解释当前项目中已经实现的 MSF-YOLO V6 创新链路：P2 浅层高分辨率检测分支、高分辨率重叠切片融合推理、Shape-Aware 形态感知损失、困难样本重加权训练机制、类别权重以及小目标增强。文档重点是数学原理与代码实现对应关系，不把尚未验证的性能提升写成结论。",
        width=88,
        size=11,
    )
    rows = [
        ["模型简称", "MSF-YOLO V6 / 全创新集合模型"],
        ["实验名", run_dir.name],
        ["训练输入", f"imgsz={config.get('train_args', {}).get('imgsz', 1024)}, batch={config.get('train_args', {}).get('batch', 12)}"],
        ["核心结构", "YOLO-P2: Detect(P2, P3, P4, P5)"],
        ["推理策略", "Blade-Slice: tile=640, overlap=0.25, NMS IoU=0.5"],
        ["重要提醒", "当前 V6 指标未超过 V4，论文中应表述为已实现并完成消融探索"],
    ]
    _table(ax, rows, ["项目", "说明"], [0.075, 0.43, 0.85, 0.30], font_size=9)
    _txt(
        ax,
        0.075,
        0.36,
        "建议论文写法：本文构建一种形态引导小目标融合检测算法，并在现有数据集上完成实验验证；若按当前结果写作，应说明该链路具备实现基础但尚未稳定超过 V4 基线。",
        width=92,
        size=10,
        color="#57606A",
    )
    ax.text(0.075, 0.20, f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", fontsize=9, fontproperties=FONT, color="#57606A")
    if FONT_PATH:
        ax.text(0.075, 0.165, f"PDF 字体：{FONT_PATH}", fontsize=8, fontproperties=FONT, color="#8C959F")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_notation(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("1. 统一符号系统", page, total)
    _txt(ax, 0.055, 0.86, "后续公式统一使用下列符号。写论文时先给出符号表，可以避免 P2、切片、损失和采样权重之间的定义混乱。", width=88)
    rows = [
        [r"$I$", "输入高分辨率图像，尺寸为 W x H"],
        [r"$B_i=(x_1,y_1,x_2,y_2,c)$", "第 i 个目标框及类别"],
        [r"$s_l$", "第 l 个检测层 stride，例如 P2/P3/P4/P5 对应 4/8/16/32"],
        [r"$F_l$", "第 l 个尺度的特征图"],
        [r"$A_i$", "目标框面积占整图面积的比例"],
        [r"$R_i$", "目标框长宽比，取 max(w/h, h/w)"],
        [r"$q_i$", "Shape-Aware 形态权重"],
        [r"$w_i$", "第 i 张训练图像的困难样本采样权重"],
        [r"$p_i$", "WeightedRandomSampler 中第 i 张图像被采样的概率"],
        [r"$\tau$", "NMS 或匹配时使用的 IoU 阈值"],
    ]
    _table(ax, rows, ["符号", "含义"], [0.065, 0.18, 0.86, 0.60], font_size=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_pipeline(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("2. MSF-YOLO V6 总体流程", page, total)
    _txt(ax, 0.055, 0.86, "MSF-YOLO V6 的创新点不是单一模块，而是训练、结构、损失和推理四个层面的组合。训练阶段影响权重学习；Blade-Slice 只发生在推理/评估阶段，不写入 .pt 权重。", width=90)
    xs = [0.07, 0.24, 0.42, 0.60, 0.78]
    labels = ["训练图像与标注", "困难样本重加权\n小目标增强", "YOLO-P2\n多尺度特征", "Shape-Aware\n检测损失", "best.pt\n候选权重"]
    for x, label in zip(xs, labels):
        _box(ax, (x, 0.58), (0.13, 0.12), label)
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + 0.13, 0.64), (xs[i + 1], 0.64))
    _box(ax, (0.28, 0.28), (0.17, 0.12), "整图推理")
    _box(ax, (0.54, 0.28), (0.17, 0.12), "Blade-Slice\n重叠切片融合")
    _box(ax, (0.78, 0.28), (0.13, 0.12), "检测结果\n评估指标")
    _arrow(ax, (0.84, 0.58), (0.625, 0.40))
    _arrow(ax, (0.45, 0.34), (0.54, 0.34))
    _arrow(ax, (0.71, 0.34), (0.78, 0.34))
    _txt(
        ax,
        0.07,
        0.18,
        "数学上可以把训练阶段写成最小化加权经验风险：在采样分布 p_i 改变的训练集上，优化带形态权重 q_i 的检测损失。推理阶段再用切片集合上的预测框并集做坐标回映射和 NMS 融合。",
        width=92,
        size=10,
    )
    _formula(ax, 0.20, 0.105, r"$\theta^*=\arg\min_{\theta}\ \mathbb{E}_{i\sim p}\left[\mathcal{L}_{det}(I_i,Y_i;\theta,q)\right]$")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_p2_resolution(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("3. P2 浅层高分辨率检测分支：尺度原理", page, total)
    _txt(ax, 0.055, 0.86, "普通 YOLO 检测头通常使用 P3/P4/P5 等尺度。P2 分支的核心价值在于 stride 更小，同一小缺陷会覆盖更多特征网格单元，局部纹理和边缘信息在下采样前保留得更多。", width=92)
    _formula(ax, 0.08, 0.765, r"$F_l \in \mathbb{R}^{C_l\times \lceil H/s_l\rceil \times \lceil W/s_l\rceil},\quad s_{P2}=4,\ s_{P3}=8,\ s_{P4}=16,\ s_{P5}=32$")
    _formula(ax, 0.08, 0.705, r"$n_{cell}(b,l)=\frac{\sqrt{w_bh_b}}{s_l}\quad\Rightarrow\quad s_l\downarrow \Longrightarrow n_{cell}\uparrow$")
    rows = [
        ["P2", "4", "256 x 256", "小裂纹、小腐蚀、小孔洞"],
        ["P3", "8", "128 x 128", "小/中等缺陷"],
        ["P4", "16", "64 x 64", "中等缺陷"],
        ["P5", "32", "32 x 32", "大目标/全局语义"],
    ]
    _table(ax, rows, ["检测层", "stride", "1024 输入下特征图", "偏关注对象"], [0.07, 0.42, 0.86, 0.22], font_size=8)
    chart = fig.add_axes([0.11, 0.12, 0.78, 0.22])
    sizes = np.array([4, 8, 12, 16, 24, 32, 48])
    for stride, color in [(4, "#F97316"), (8, "#2563EB"), (16, "#16A34A"), (32, "#7C3AED")]:
        chart.plot(sizes, sizes / stride, marker="o", label=f"P stride={stride}", color=color)
    chart.set_title("缺陷像素尺度映射到特征网格后的覆盖单元数", fontproperties=FONT, fontsize=11)
    chart.set_xlabel("缺陷近似边长 / pixel", fontproperties=FONT)
    chart.set_ylabel("覆盖网格单元数", fontproperties=FONT)
    chart.grid(alpha=0.25)
    chart.legend(prop=FONT, ncol=4, fontsize=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_p2_fusion(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("4. P2 分支的特征融合公式", page, total)
    _txt(ax, 0.055, 0.86, "项目中的 P2 配置使用 Detect(P2,P3,P4,P5)。其思想是把深层强语义特征逐级上采样，与浅层高分辨率特征拼接，再通过 C2f 模块融合，形成既有纹理细节又有语义上下文的检测特征。", width=92)
    _formula(ax, 0.08, 0.765, r"$\tilde{F}_{P4}=C2f\left(Concat(Up(F_{P5}),F_{P4}^{backbone})\right)$")
    _formula(ax, 0.08, 0.705, r"$\tilde{F}_{P3}=C2f\left(Concat(Up(\tilde{F}_{P4}),F_{P3}^{backbone})\right)$")
    _formula(ax, 0.08, 0.645, r"$\tilde{F}_{P2}=C2f\left(Concat(Up(\tilde{F}_{P3}),F_{P2}^{backbone})\right)$")
    _formula(ax, 0.08, 0.585, r"$\hat{Y}=Detect(\tilde{F}_{P2},\tilde{F}_{P3},\tilde{F}_{P4},\tilde{F}_{P5})$")
    levels = [("P5/32", 0.16, 0.40, "#E0E7FF"), ("P4/16", 0.34, 0.46, "#DBEAFE"), ("P3/8", 0.52, 0.52, "#DCFCE7"), ("P2/4", 0.70, 0.58, "#FFEDD5")]
    for text, x, y, color in levels:
        _box(ax, (x, y), (0.13, 0.09), text, fc=color)
    _arrow(ax, (0.29, 0.445), (0.34, 0.50))
    _arrow(ax, (0.47, 0.505), (0.52, 0.56))
    _arrow(ax, (0.65, 0.565), (0.70, 0.62))
    _box(ax, (0.41, 0.22), (0.22, 0.10), "四尺度 Detect 输出\n分类 + 边框分布")
    for _, x, y, _ in levels:
        _arrow(ax, (x + 0.065, y), (0.52, 0.32))
    _txt(ax, 0.08, 0.14, "论文表述重点：P2 不是简单增加参数，而是改变最小检测尺度。它把 stride=4 的浅层细节纳入检测头，使小缺陷在输出层拥有更密集的候选位置。", width=90, size=9.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_loss_base(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("5. YOLO 检测损失的基础形式", page, total)
    _txt(ax, 0.055, 0.86, "MSF-YOLO V6 的 Shape-Aware Loss 是在 YOLOv8 检测损失中的 bbox/DFL 部分加入形态权重。理解它之前，需要先明确基线检测损失由边框回归、分类 BCE 和 DFL 三部分组成。", width=92)
    _formula(ax, 0.08, 0.765, r"$\mathcal{L}_{det}=\lambda_{box}\mathcal{L}_{box}+\lambda_{cls}\mathcal{L}_{cls}+\lambda_{dfl}\mathcal{L}_{dfl}$")
    _formula(ax, 0.08, 0.705, r"$\mathcal{L}_{box}=\frac{\sum_{j\in\Omega^+}(1-CIoU(\hat{b}_j,b_j))\cdot t_j}{\sum_{j\in\Omega^+}t_j}$")
    _formula(ax, 0.08, 0.645, r"$\mathcal{L}_{cls}=\frac{\sum_j BCEWithLogits(\hat{s}_j,s_j)}{\sum_{j\in\Omega^+}t_j}$")
    _formula(ax, 0.08, 0.585, r"$\mathcal{L}_{dfl}=\frac{\sum_{j\in\Omega^+}DFL(\hat{d}_j,d_j)\cdot t_j}{\sum_{j\in\Omega^+}t_j}$")
    rows = [
        ["Ω+", "Task-Aligned Assigner 分配到的正样本锚点集合"],
        ["t_j", "target_scores 的类别置信权重之和"],
        ["CIoU", "同时考虑重叠面积、中心距离和长宽比一致性的 IoU 变体"],
        ["DFL", "Distribution Focal Loss，把边框距离建模为离散分布"],
    ]
    _table(ax, rows, ["符号", "解释"], [0.08, 0.20, 0.84, 0.25], font_size=8)
    _txt(ax, 0.08, 0.13, "项目代码对应：本地 venv 的 ultralytics/utils/loss.py 中 v8DetectionLoss 计算 BCE、BboxLoss 和 DFL；train_msf_yolo_full.py 用 ShapeAwareBboxLoss 替换原 BboxLoss。", width=92, size=9.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_shape_loss(pdf: PdfPages, config: dict[str, Any], page: int, total: int) -> None:
    fig, ax = _new_page("6. Shape-Aware Loss：形态权重定义", page, total)
    shape = config.get("shape_config", {})
    alpha = float(shape.get("small_gain", 0.35))
    beta = float(shape.get("slender_gain", 0.25))
    a0 = float(shape.get("area_threshold", 0.012))
    r0 = float(shape.get("slender_ratio", 3.0))
    qmax = float(shape.get("max_shape_weight", 1.7))
    _txt(ax, 0.055, 0.86, "风机叶片缺陷中，裂纹常表现为细长框，腐蚀可能表现为小面积弱边界区域。Shape-Aware Loss 不改变标签，而是在正样本边框损失上加入由面积和长宽比决定的连续权重。", width=92)
    _formula(ax, 0.08, 0.765, r"$A_i=\frac{w_i h_i}{W H},\qquad R_i=\max\left(\frac{w_i}{h_i},\frac{h_i}{w_i}\right)$")
    _formula(ax, 0.08, 0.705, rf"$S_i=clip\left(\frac{{{a0:.3f}-A_i}}{{{a0:.3f}}},0,1\right),\quad E_i=clip\left(\frac{{R_i-{r0:.1f}}}{{{r0:.1f}}},0,1\right)$")
    _formula(ax, 0.08, 0.645, rf"$q_i=clip\left(1+{alpha:.2f}S_i+{beta:.2f}E_i,\ 1,\ {qmax:.1f}\right)$")
    _formula(ax, 0.08, 0.585, r"$\mathcal{L}_{box}^{shape}=\frac{\sum_{i\in\Omega^+}(1-CIoU_i)\cdot t_i\cdot q_i}{\sum_{i\in\Omega^+}t_i}$")
    _formula(ax, 0.08, 0.525, r"$\mathcal{L}_{dfl}^{shape}=\frac{\sum_{i\in\Omega^+}DFL_i\cdot t_i\cdot q_i}{\sum_{i\in\Omega^+}t_i}$")
    curve_ax = fig.add_axes([0.10, 0.13, 0.36, 0.25])
    a = np.linspace(0, 0.025, 200)
    s = np.clip((a0 - a) / a0, 0, 1)
    curve_ax.plot(a, s, color="#F97316")
    curve_ax.axvline(a0, linestyle="--", color="#8C959F", linewidth=1)
    curve_ax.set_title("小面积项 S_i", fontproperties=FONT, fontsize=10)
    curve_ax.set_xlabel("面积占比 A_i", fontproperties=FONT)
    curve_ax.set_ylim(-0.05, 1.05)
    curve_ax.grid(alpha=0.25)
    ratio_ax = fig.add_axes([0.55, 0.13, 0.36, 0.25])
    r = np.linspace(1, 8, 200)
    e = np.clip((r - r0) / r0, 0, 1)
    ratio_ax.plot(r, e, color="#2563EB")
    ratio_ax.axvline(r0, linestyle="--", color="#8C959F", linewidth=1)
    ratio_ax.set_title("细长项 E_i", fontproperties=FONT, fontsize=10)
    ratio_ax.set_xlabel("长宽比 R_i", fontproperties=FONT)
    ratio_ax.set_ylim(-0.05, 1.05)
    ratio_ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_shape_interpretation(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("7. Shape-Aware Loss 的梯度含义", page, total)
    _txt(ax, 0.055, 0.86, "形态权重 q_i 的本质是改变正样本边框误差对总损失的贡献比例。对模型参数 θ 求梯度时，小目标或细长目标的边框误差会被放大，从而使优化器更倾向于修正这类目标的定位误差。", width=92)
    _formula(ax, 0.08, 0.765, r"$\nabla_{\theta}\mathcal{L}_{box}^{shape}=\frac{1}{T}\sum_{i\in\Omega^+}t_iq_i\nabla_{\theta}\left(1-CIoU(\hat{b}_i,b_i)\right)$")
    _formula(ax, 0.08, 0.705, r"$T=\sum_{i\in\Omega^+}t_i,\qquad q_i>1\Rightarrow |\nabla_{\theta}\ell_i|\uparrow$")
    rows = [
        ["普通大目标", "A_i 大、R_i 接近 1", "q_i≈1", "保持原 YOLO 损失"],
        ["小腐蚀/小孔洞", "A_i 小", "q_i>1", "增强小面积目标定位约束"],
        ["细长裂纹", "R_i 大", "q_i>1", "增强细长框边界和方向尺度约束"],
        ["极端小且细长", "A_i 小且 R_i 大", "q_i≤q_max", "避免梯度过度放大"],
    ]
    _table(ax, rows, ["目标形态", "几何条件", "权重", "优化含义"], [0.07, 0.32, 0.86, 0.27], font_size=8)
    _txt(ax, 0.08, 0.23, "需要注意：这里的 Shape-Aware Loss 是框级形态先验，不是分割边界损失。它不直接学习腐蚀像素边缘，而是通过目标框面积和长宽比调整检测损失。论文中不应把它描述成像素级边缘监督。", width=92, size=9.5, color="#9A3412")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_sampler(pdf: PdfPages, config: dict[str, Any], page: int, total: int) -> None:
    fig, ax = _new_page("8. 困难样本重加权：采样概率公式", page, total)
    sc = config.get("sampler_config", {})
    cw = float(sc.get("corrosion_weight", 3.0))
    kw = float(sc.get("crack_weight", 2.0))
    sw = float(sc.get("small_target_weight", 1.5))
    nw = float(sc.get("negative_weight", 1.0))
    wmax = float(sc.get("max_sample_weight", 6.0))
    amax = float(sc.get("small_target_max_area", 0.012))
    _txt(ax, 0.055, 0.86, "困难样本重加权不是改标签，也不是改损失，而是改变训练图像被抽到的概率。当前项目用 WeightedRandomSampler，以 replacement=True 重复抽样，使腐蚀、裂纹和小目标样本在每个 epoch 内出现得更频繁。", width=92)
    _formula(ax, 0.08, 0.765, rf"$w_i=clip\left({nw:.1f}\cdot {cw:.1f}^{{\mathbb{{1}}[corrosion]}}\cdot {kw:.1f}^{{\mathbb{{1}}[crack]}}\cdot {sw:.1f}^{{\mathbb{{1}}[small]}},\ 0.05,\ {wmax:.1f}\right)$")
    _formula(
        ax,
        0.08,
        0.705,
        r"$\mathbb{1}[small]=1\quad \mathrm{if}\quad c\in\{\mathrm{crack},\mathrm{corrosion}\},\quad A_i\leq"
        + f"{amax:.3f}"
        + r"$",
    )
    _formula(ax, 0.08, 0.645, r"$p_i=\frac{w_i}{\sum_{j=1}^{N}w_j},\qquad \mathbb{E}[m_i]=N\cdot p_i$")
    rows = [
        ["普通样本", "无 crack/corrosion/小目标", "1.0"],
        ["裂纹样本", "crack", f"{kw:.1f}"],
        ["腐蚀样本", "corrosion", f"{cw:.1f}"],
        ["小裂纹", "crack + small", f"{kw * sw:.1f}"],
        ["腐蚀 + 裂纹", "corrosion + crack", f"{min(cw * kw, wmax):.1f}"],
        ["腐蚀 + 裂纹 + 小目标", "三者同时满足", f"{wmax:.1f} 上限截断"],
    ]
    _table(ax, rows, ["样本类型", "触发条件", "采样权重"], [0.08, 0.23, 0.84, 0.30], font_size=8)
    bars = fig.add_axes([0.12, 0.08, 0.76, 0.10])
    values = [1, kw, cw, kw * sw, min(cw * kw, wmax), wmax]
    bars.bar(range(len(values)), values, color=["#9CA3AF", "#2563EB", "#F97316", "#16A34A", "#7C3AED", "#DC2626"])
    bars.set_xticks(range(len(values)), ["normal", "crack", "corrosion", "small crack", "c+c", "max"], rotation=0, fontproperties=FONT)
    bars.set_ylabel("w_i", fontproperties=FONT)
    bars.grid(axis="y", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_class_weight(pdf: PdfPages, config: dict[str, Any], page: int, total: int) -> None:
    fig, ax = _new_page("9. 类别权重 cls_pw：分类 BCE 加权", page, total)
    gamma = float(config.get("train_args", {}).get("cls_pw", 0.35))
    _txt(ax, 0.055, 0.86, "除了图像级采样权重，项目本地 Ultralytics 还加入了类别级分类损失权重。它根据训练集类别频次 n_c 计算反频率权重，再用 cls_pw 控制强度，最后归一化到均值为 1。", width=92)
    class_weight_formula = (
        r"$\tilde{\alpha}_c=\left(\frac{1}{n_c}\right)^{"
        + f"{gamma:.2f}"
        + r"},\qquad \alpha_c=\frac{\tilde{\alpha}_c}{\frac{1}{C}\sum_{k=1}^C\tilde{\alpha}_k}$"
    )
    _formula(ax, 0.08, 0.765, class_weight_formula)
    _formula(ax, 0.08, 0.705, r"$BCE(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))$")
    _formula(ax, 0.08, 0.645, r"$\mathcal{L}_{cls}^{cw}=\frac{\sum_{j,c}\alpha_c\cdot BCE(\hat{s}_{j,c},s_{j,c})}{\sum_{j\in\Omega^+}t_j}$")
    _txt(ax, 0.08, 0.555, "解释：如果某类样本数量少，则 n_c 小，反频率权重更大；但 cls_pw=0.35 会把权重差异压平，避免少数类权重过大导致训练不稳定。", width=90)
    sample_counts = np.array([2200, 1800, 1500, 500], dtype=float)
    weights = (1.0 / sample_counts) ** gamma
    weights = weights / weights.mean()
    chart = fig.add_axes([0.14, 0.17, 0.70, 0.25])
    chart.bar(["crack", "hole", "spalling", "corrosion"], weights, color=["#2563EB", "#16A34A", "#7C3AED", "#F97316"])
    chart.set_title("示意：样本越少，类别权重越高", fontproperties=FONT, fontsize=11)
    chart.set_ylabel("normalized alpha_c", fontproperties=FONT)
    chart.grid(axis="y", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_augment(pdf: PdfPages, config: dict[str, Any], page: int, total: int) -> None:
    fig, ax = _new_page("10. 小目标增强：尺度扰动与组合增强", page, total)
    args = config.get("train_args", {})
    _txt(ax, 0.055, 0.86, "小目标增强的数学目标是提高小缺陷在不同尺度、不同上下文和不同局部组合下出现的概率。它不等于单独复制小目标，而是通过输入尺度、多尺度训练、mosaic、copy-paste 和随机缩放共同增加样本分布覆盖。", width=92)
    _formula(ax, 0.08, 0.765, r"$I' = T_{\phi}(I),\qquad B'_k=T_{\phi}(B_k)$")
    _formula(ax, 0.08, 0.705, r"$A'_k=\frac{w'_kh'_k}{W'H'},\qquad \phi\sim \mathcal{P}_{aug}$")
    _formula(ax, 0.08, 0.645, r"$\mathcal{D}_{aug}=\{(T_{\phi}(I_i),T_{\phi}(Y_i))\ |\ (I_i,Y_i)\in\mathcal{D}\}$")
    rows = [
        ["imgsz", str(args.get("imgsz", 1024)), "提高输入分辨率，减少整图缩放造成的小缺陷信息损失"],
        ["multi_scale", str(args.get("multi_scale", 0.10)), "训练时改变输入尺度，提高尺度鲁棒性"],
        ["mosaic", str(args.get("mosaic", 1.0)), "四图拼接，增加背景和目标组合多样性"],
        ["copy_paste", str(args.get("copy_paste", 0.05)), "复制目标区域，缓解少数类和小目标出现频率不足"],
        ["scale", str(args.get("scale", 0.60)), "随机缩放，改变缺陷在图像中的相对面积"],
        ["fliplr", str(args.get("fliplr", 0.5)), "水平翻转，增加视角变化"],
    ]
    _table(ax, rows, ["参数", "当前值", "数学/工程含义"], [0.07, 0.25, 0.86, 0.30], font_size=8)
    _txt(ax, 0.08, 0.17, "论文中可将该部分写成数据分布增强：通过随机变换族 T_phi 扩展训练分布，使模型看到更多小尺度、低对比度和复杂背景下的缺陷实例。", width=92, size=9.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_blade_slice(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("11. Blade-Slice：重叠切片推理公式", page, total)
    _txt(ax, 0.055, 0.86, "Blade-Slice 的目标是避免整图缩放把小缺陷压缩到过少像素。它把高分辨率图像切成重叠局部窗口，在每个窗口中独立检测，再把局部坐标映射回原图并进行跨切片 NMS 融合。", width=92)
    _formula(ax, 0.08, 0.765, r"$d=\lfloor L(1-\rho)\rfloor,\quad L=640,\ \rho=0.25,\ d=480$")
    _formula(ax, 0.08, 0.705, r"$S_x=\{0,d,2d,\ldots,W-L\},\qquad S_y=\{0,d,2d,\ldots,H-L\}$")
    _formula(ax, 0.08, 0.645, r"$C_{u,v}=I[u:u+L,\ v:v+L],\quad u\in S_x,\ v\in S_y$")
    _formula(ax, 0.08, 0.585, r"$b^{global}=(x_1+u,y_1+v,x_2+u,y_2+v,c,p)$")
    tile_ax = fig.add_axes([0.13, 0.14, 0.74, 0.34])
    tile_ax.set_xlim(0, 10)
    tile_ax.set_ylim(0, 6)
    tile_ax.axis("off")
    tile_ax.add_patch(Rectangle((0.5, 0.5), 9, 5, facecolor="#F8FAFC", edgecolor="#111827", linewidth=1.2))
    colors = ["#DBEAFE", "#FFEDD5", "#DCFCE7", "#F3E8FF"]
    starts = [(0.7, 0.7), (3.3, 0.7), (0.7, 2.6), (3.3, 2.6)]
    for idx, (x, y) in enumerate(starts):
        tile_ax.add_patch(Rectangle((x, y), 4.2, 2.6, facecolor=colors[idx], alpha=0.72, edgecolor="#57606A", linewidth=1.0))
    tile_ax.text(5, 5.72, "原始高分辨率图像 I", ha="center", fontproperties=FONT, fontsize=10)
    tile_ax.text(6.05, 2.15, "重叠区域", fontproperties=FONT, fontsize=9, color="#9A3412")
    tile_ax.arrow(5.55, 2.2, -0.8, 0.35, head_width=0.08, color="#9A3412")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_fusion_nms(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("12. 切片框融合与 NMS", page, total)
    _txt(ax, 0.055, 0.86, "当前 Blade-Slice 实现包含 full image prediction 与 sliced prediction 两部分。在 blade_slice_fused 模式下，先取两类预测框并集，再按类别和置信度做 NMS。", width=92)
    _formula(ax, 0.08, 0.765, r"$\mathcal{B}_{full}=f_{\theta}(I),\qquad \mathcal{B}_{slice}=\bigcup_{u,v}R_{u,v}(f_{\theta}(C_{u,v}))$")
    _formula(ax, 0.08, 0.705, r"$\mathcal{B}_{fused}=NMS(\mathcal{B}_{full}\cup\mathcal{B}_{slice},\tau)$")
    _formula(ax, 0.08, 0.645, r"$IoU(b_i,b_j)=\frac{|b_i\cap b_j|}{|b_i\cup b_j|}$")
    _formula(ax, 0.08, 0.585, r"$b_j\ \mathrm{removed\ if}\ c_i=c_j,\ p_i\geq p_j,\ IoU(b_i,b_j)\geq\tau$")
    rows = [
        ["normal", "只用整图预测", "速度快，可能损失小缺陷像素信息"],
        ["blade_slice", "只用切片预测", "小目标更清晰，但上下文减少"],
        ["blade_slice_fused", "整图 + 切片并集后 NMS", "兼顾全局上下文和局部细节，但误检可能增加"],
    ]
    _table(ax, rows, ["模式", "数学定义", "主要影响"], [0.08, 0.25, 0.84, 0.23], font_size=8)
    _txt(ax, 0.08, 0.17, "当前实验结果提示：切片融合并不必然提升 Recall。如果模型本身对小目标置信度不足，或者切片带来背景局部纹理误检，NMS 融合可能增加 FP。", width=92, size=9.5, color="#9A3412")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_metrics(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("13. 评价指标公式", page, total)
    _txt(ax, 0.055, 0.86, "模型好坏不能只看训练损失，检测任务应同时看 Precision、Recall、AP/mAP 和背景误检。本文项目里 V4 与 V6 的比较主要依据 val/test_clean 上的 mAP50-95、mAP50、Recall 以及空标签图误检数。", width=92)
    _formula(ax, 0.08, 0.765, r"$Precision=\frac{TP}{TP+FP},\qquad Recall=\frac{TP}{TP+FN}$")
    _formula(ax, 0.08, 0.705, r"$AP_c=\int_0^1 P_c(R)\ dR,\qquad mAP=\frac{1}{C}\sum_{c=1}^{C}AP_c$")
    _formula(ax, 0.08, 0.645, r"$mAP50{-}95=\frac{1}{10}\sum_{\tau\in\{0.50,0.55,\ldots,0.95\}}mAP_{\tau}$")
    _formula(ax, 0.08, 0.585, r"$FP_{bg}=\sum_{I_i\in\mathcal{D}_{empty}}|\{b\in f_\theta(I_i):conf(b)\geq\eta\}|$")
    rows = [
        ["mAP50", "IoU=0.50 时的平均精度", "反映宽松定位下的检测能力"],
        ["mAP50-95", "IoU 0.50 到 0.95 的平均值", "更严格，能反映定位质量"],
        ["Recall", "TP/(TP+FN)", "漏检风险，风机缺陷场景很关键"],
        ["背景误检", "空标签图上的预测框数量", "反映复杂背景鲁棒性"],
    ]
    _table(ax, rows, ["指标", "公式/定义", "论文解释"], [0.08, 0.25, 0.84, 0.25], font_size=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_code_mapping(pdf: PdfPages, config: dict[str, Any], page: int, total: int) -> None:
    fig, ax = _new_page("14. 当前代码实现与公式对应", page, total)
    _txt(ax, 0.055, 0.86, "下面表格把论文中的数学模块映射到项目中的实际文件和参数。写论文或答辩时，最好能把这些模块说清楚：哪些进入训练权重，哪些只是推理策略。", width=92)
    rows = [
        ["P2 检测头", "configs/yolov8n-p2-wind.yaml", "Detect(P2,P3,P4,P5)", "进入 .pt 权重"],
        ["Shape-Aware Loss", "scripts/train_msf_yolo_full.py", "ShapeAwareBboxLoss", "进入训练优化"],
        ["困难样本采样", "scripts/train_msf_yolo_full.py", "WeightedRandomSampler", "影响训练分布"],
        ["类别权重", "本地 ultralytics detect/train.py + loss.py", "cls_pw=0.35", "影响分类损失"],
        ["小目标增强", "train args", "imgsz/mosaic/copy_paste/scale", "影响训练样本"],
        ["Blade-Slice", "scripts/evaluate_blade_slice.py", "tile/overlap/remap/NMS", "只在推理/评估阶段"],
    ]
    _table(ax, rows, ["模块", "代码位置", "实现点", "是否写入权重"], [0.055, 0.42, 0.89, 0.33], font_size=7.5)
    _txt(ax, 0.08, 0.34, "当前关键参数：Shape-Aware 使用 area_threshold=0.012、small_gain=0.35、slender_gain=0.25、slender_ratio=3.0、max_shape_weight=1.7；困难样本采样使用 corrosion x3、crack x2、小目标 x1.5、max_sample_weight=6。", width=92, size=9.5)
    _txt(ax, 0.08, 0.22, "这页可直接支撑论文方法章节中的“实现细节”或“实验设置”部分。", width=92, size=9.5, color="#57606A")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_results_warning(pdf: PdfPages, page: int, total: int) -> None:
    fig, ax = _new_page("15. 论文表述边界：已实现不等于已提升", page, total)
    _txt(ax, 0.055, 0.86, "从论文诚信和工程判断上，需要区分“方法链路已经实现”和“方法效果已经超过基线”。当前 MSF-YOLO V6 的实验结果没有超过 V4，因此不能写成已经提升召回和鲁棒性。", width=92)
    rows = [
        ["val mAP50-95", "V4 0.8093", "V6 0.7922", "-0.0171"],
        ["val Recall", "V4 0.9411", "V6 0.9071", "-0.0340"],
        ["test_clean mAP50-95", "V4 0.8401", "V6 0.8301", "-0.0100"],
        ["test_clean Recall", "V4 0.9792", "V6 0.9541", "-0.0251"],
        ["test_clean 背景误检", "V4 1", "V6 1", "持平"],
    ]
    _table(ax, rows, ["指标", "V4", "MSF-YOLO V6", "变化"], [0.08, 0.47, 0.84, 0.25], font_size=8)
    _txt(ax, 0.08, 0.38, "严谨写法：本文提出并实现 MSF-YOLO，用于探索 P2 小目标检测头、切片融合推理、形态感知损失和困难样本重加权在风机叶片缺陷检测中的作用。当前实验表明该链路具备完整实现基础，但在现有训练策略下尚未超过 V4 基线。", width=92, size=10)
    _txt(ax, 0.08, 0.24, "不建议写法：MSF-YOLO 显著提升了召回率和工程可靠性。除非后续训练或消融实验真的获得正向指标，否则这句话风险很高。", width=92, size=10, color="#9A3412")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _markdown_text(config: dict[str, Any], run_dir: Path) -> str:
    shape = config.get("shape_config", {})
    sampler = config.get("sampler_config", {})
    args = config.get("train_args", {})
    return f"""# MSF-YOLO V6 数学原理与公式说明

生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}

实验名：`{run_dir.name}`

> 注意：本文档用于论文方法章节参考。当前 MSF-YOLO V6 已实现完整创新链路，但实验指标尚未超过 V4，因此建议写作时区分“提出并实现”与“效果提升”。

## 1. 总体优化目标

MSF-YOLO V6 可以抽象为在困难样本采样分布下，最小化带形态权重的检测损失：

```math
\\theta^*=\\arg\\min_\\theta \\mathbb{{E}}_{{i\\sim p}}[\\mathcal{{L}}_{{det}}(I_i,Y_i;\\theta,q)]
```

其中 `p_i` 来自 WeightedRandomSampler，`q_i` 来自 Shape-Aware Loss。

## 2. P2 浅层高分辨率检测分支

输入图像尺寸为 `W x H`，第 `l` 个检测层 stride 为 `s_l`，则特征图尺寸为：

```math
F_l\\in\\mathbb{{R}}^{{C_l\\times \\lceil H/s_l\\rceil\\times \\lceil W/s_l\\rceil}}
```

当前项目采用：

```math
s_{{P2}}=4,\\quad s_{{P3}}=8,\\quad s_{{P4}}=16,\\quad s_{{P5}}=32
```

小目标在特征图上的覆盖单元数近似为：

```math
n_{{cell}}(b,l)=\\frac{{\\sqrt{{w_bh_b}}}}{{s_l}}
```

因此 `s_l` 越小，同样像素尺度的小缺陷覆盖的特征单元越多。P2 分支的价值在于让小裂纹、小腐蚀、小孔洞不至于在深层下采样后退化成极少数特征点。

项目配置文件：`configs/yolov8n-p2-wind.yaml`。检测输出为：

```math
\\hat{{Y}}=Detect(\\tilde{{F}}_{{P2}},\\tilde{{F}}_{{P3}},\\tilde{{F}}_{{P4}},\\tilde{{F}}_{{P5}})
```

## 3. Shape-Aware 形态感知损失

对第 `i` 个正样本框，定义面积占比和长宽比：

```math
A_i=\\frac{{w_ih_i}}{{WH}},\\qquad R_i=\\max\\left(\\frac{{w_i}}{{h_i}},\\frac{{h_i}}{{w_i}}\\right)
```

当前参数：

- `area_threshold={shape.get("area_threshold", 0.012)}`
- `small_gain={shape.get("small_gain", 0.35)}`
- `slender_gain={shape.get("slender_gain", 0.25)}`
- `slender_ratio={shape.get("slender_ratio", 3.0)}`
- `max_shape_weight={shape.get("max_shape_weight", 1.7)}`

小目标项：

```math
S_i=clip\\left(\\frac{{a_0-A_i}}{{a_0}},0,1\\right)
```

细长项：

```math
E_i=clip\\left(\\frac{{R_i-r_0}}{{r_0}},0,1\\right)
```

形态权重：

```math
q_i=clip(1+\\alpha S_i+\\beta E_i,1,q_{{max}})
```

加权边框损失：

```math
\\mathcal{{L}}_{{box}}^{{shape}}=
\\frac{{\\sum_{{i\\in\\Omega^+}}(1-CIoU_i)t_iq_i}}{{\\sum_{{i\\in\\Omega^+}}t_i}}
```

加权 DFL：

```math
\\mathcal{{L}}_{{dfl}}^{{shape}}=
\\frac{{\\sum_{{i\\in\\Omega^+}}DFL_i t_iq_i}}{{\\sum_{{i\\in\\Omega^+}}t_i}}
```

梯度角度解释：

```math
\\nabla_\\theta \\mathcal{{L}}_{{box}}^{{shape}}
=\\frac{{1}}{{T}}\\sum_{{i\\in\\Omega^+}}t_iq_i\\nabla_\\theta(1-CIoU_i)
```

当 `q_i>1` 时，该目标的定位误差对总梯度贡献增大。

## 4. 困难样本重加权训练机制

当前图像级采样权重：

```math
w_i=clip(1.0\\cdot 3.0^{{\\mathbb{{1}}[corrosion]}}\\cdot 2.0^{{\\mathbb{{1}}[crack]}}\\cdot 1.5^{{\\mathbb{{1}}[small]}},0.05,6.0)
```

小目标条件：

```math
\\mathbb{{1}}[small]=1\\quad\\text{{if}}\\quad c\\in\\{{crack,corrosion\\}},\\ A_i\\le {sampler.get("small_target_max_area", 0.012)}
```

采样概率：

```math
p_i=\\frac{{w_i}}{{\\sum_{{j=1}}^Nw_j}}
```

期望出现次数：

```math
\\mathbb{{E}}[m_i]=N\\cdot p_i
```

## 5. 类别权重 cls_pw

本地 Ultralytics 中 `cls_pw={args.get("cls_pw", 0.35)}` 被用于类别频次反比权重：

```math
\\tilde{{\\alpha}}_c=\\left(\\frac{{1}}{{n_c}}\\right)^{{cls\\_pw}},
\\qquad
\\alpha_c=\\frac{{\\tilde{{\\alpha}}_c}}{{\\frac{{1}}{{C}}\\sum_{{k=1}}^C\\tilde{{\\alpha}}_k}}
```

加权 BCE：

```math
\\mathcal{{L}}_{{cls}}^{{cw}}=
\\frac{{\\sum_{{j,c}}\\alpha_c BCE(\\hat{{s}}_{{j,c}},s_{{j,c}})}}{{\\sum_{{j\\in\\Omega^+}}t_j}}
```

## 6. 小目标增强

随机增强可写为：

```math
I'=T_\\phi(I),\\qquad B'_k=T_\\phi(B_k),\\qquad \\phi\\sim\\mathcal{{P}}_{{aug}}
```

当前参数：

- `imgsz={args.get("imgsz", 1024)}`
- `multi_scale={args.get("multi_scale", 0.10)}`
- `mosaic={args.get("mosaic", 1.0)}`
- `copy_paste={args.get("copy_paste", 0.05)}`
- `scale={args.get("scale", 0.60)}`
- `fliplr={args.get("fliplr", 0.5)}`

## 7. Blade-Slice 重叠切片融合推理

切片边长 `L=640`，重叠率 `rho=0.25`，步长：

```math
d=\\lfloor L(1-\\rho)\\rfloor=480
```

切片起点集合：

```math
S_x=\\{{0,d,2d,\\ldots,W-L\\}},\\qquad S_y=\\{{0,d,2d,\\ldots,H-L\\}}
```

切片：

```math
C_{{u,v}}=I[u:u+L,v:v+L]
```

局部框回映射：

```math
b^{{global}}=(x_1+u,y_1+v,x_2+u,y_2+v,c,p)
```

融合：

```math
\\mathcal{{B}}_{{fused}}=NMS(\\mathcal{{B}}_{{full}}\\cup\\mathcal{{B}}_{{slice}},\\tau)
```

## 8. 评价指标

```math
Precision=\\frac{{TP}}{{TP+FP}},\\qquad Recall=\\frac{{TP}}{{TP+FN}}
```

```math
mAP50\\text{{-}}95=\\frac{{1}}{{10}}\\sum_{{\\tau\\in\\{{0.50,0.55,\\ldots,0.95\\}}}}mAP_\\tau
```

背景误检：

```math
FP_{{bg}}=\\sum_{{I_i\\in\\mathcal{{D}}_{{empty}}}}|\\{{b\\in f_\\theta(I_i):conf(b)\\ge\\eta\\}}|
```

## 9. 论文表述边界

当前 V6 指标未超过 V4：

- val mAP50-95: V4 0.8093, V6 0.7922
- val Recall: V4 0.9411, V6 0.9071
- test_clean mAP50-95: V4 0.8401, V6 0.8301
- test_clean Recall: V4 0.9792, V6 0.9541

建议写作：本文提出并实现 MSF-YOLO，用于探索 P2 小目标检测头、切片融合推理、形态感知损失和困难样本重加权在风机叶片缺陷检测中的作用。当前实验表明该链路具备完整实现基础，但在现有训练策略下尚未超过 V4 基线。
"""


def _write_outputs(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_json(run_dir / "msf_full_innovation_config.json")
    if not config:
        config = {
            "shape_config": {
                "area_threshold": 0.012,
                "small_gain": 0.35,
                "slender_gain": 0.25,
                "slender_ratio": 3.0,
                "max_shape_weight": 1.7,
            },
            "sampler_config": {
                "corrosion_weight": 3.0,
                "crack_weight": 2.0,
                "small_target_weight": 1.5,
                "small_target_max_area": 0.012,
            },
            "train_args": {
                "imgsz": 1024,
                "batch": 12,
                "cls_pw": 0.35,
                "multi_scale": 0.10,
                "mosaic": 1.0,
                "copy_paste": 0.05,
                "scale": 0.60,
                "fliplr": 0.5,
            },
        }

    pdf_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明.pdf"
    md_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明.md"
    total = 16
    with PdfPages(pdf_path) as pdf:
        _cover(pdf, run_dir, config, 1, total)
        _page_notation(pdf, 2, total)
        _page_pipeline(pdf, 3, total)
        _page_p2_resolution(pdf, 4, total)
        _page_p2_fusion(pdf, 5, total)
        _page_loss_base(pdf, 6, total)
        _page_shape_loss(pdf, config, 7, total)
        _page_shape_interpretation(pdf, 8, total)
        _page_sampler(pdf, config, 9, total)
        _page_class_weight(pdf, config, 10, total)
        _page_augment(pdf, config, 11, total)
        _page_blade_slice(pdf, 12, total)
        _page_fusion_nms(pdf, 13, total)
        _page_metrics(pdf, 14, total)
        _page_code_mapping(pdf, config, 15, total)
        _page_results_warning(pdf, 16, total)

    md_path.write_text(_markdown_text(config, run_dir), encoding="utf-8")
    manifest = {
        "pdf": str(pdf_path),
        "markdown": str(md_path),
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "font": FONT_PATH,
        "note": "This is a mathematical-principle reference for MSF-YOLO V6. It does not claim V6 outperforms V4.",
    }
    (output_dir / "principle_report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pdf_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MSF-YOLO V6 principle PDF and Markdown.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "principle_report"
    pdf_path, md_path = _write_outputs(run_dir, output_dir)
    print(pdf_path)
    print(md_path)


if __name__ == "__main__":
    main()
