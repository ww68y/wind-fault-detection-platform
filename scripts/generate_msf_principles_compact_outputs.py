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


def _load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _wrap(text: str, width: int = 38) -> str:
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=True,
        )
    )


def _page(title: str, page: int, total: int) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.045, 0.955, title, fontsize=16, fontproperties=FONT, weight="bold", color="#17212B")
    ax.plot([0.045, 0.955], [0.932, 0.932], color="#D0D7DE", linewidth=0.9)
    fig.text(0.045, 0.025, "MSF-YOLO V6 数学原理紧凑版", fontsize=7.5, fontproperties=FONT, color="#6A737D")
    fig.text(0.925, 0.025, f"{page}/{total}", fontsize=7.5, fontproperties=FONT, color="#6A737D")
    return fig, ax


def _text(ax: plt.Axes, x: float, y: float, text: str, w: int = 47, size: float = 8.5, color: str = "#25313B") -> None:
    ax.text(x, y, _wrap(text, w), va="top", fontsize=size, fontproperties=FONT, color=color, linespacing=1.28)


def _formula(ax: plt.Axes, x: float, y: float, text: str, w: float = 0.42, h: float = 0.045, size: float = 8.2) -> None:
    ax.add_patch(Rectangle((x, y - h + 0.006), w, h, facecolor="#F6F8FA", edgecolor="#D0D7DE", linewidth=0.5))
    ax.text(x + 0.008, y - 0.006, text, va="top", fontsize=size, color="#111827", family="DejaVu Sans Mono")


def _table(ax: plt.Axes, rows: list[list[str]], cols: list[str], bbox: list[float], size: float = 7.2) -> None:
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", colLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.get_text().set_fontproperties(FONT)
        if r == 0:
            cell.set_facecolor("#263238")
            cell.get_text().set_color("white")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7F9")


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, fc: str = "#F6F8FA") -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#8C959F", linewidth=0.85))
    ax.text(x + w / 2, y + h / 2, _wrap(text, 11), ha="center", va="center", fontsize=7.4, fontproperties=FONT)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color="#57606A"))


def _compact_pdf(run_dir: Path, config: dict[str, Any], output_dir: Path) -> Path:
    pdf_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明_紧凑版.pdf"
    shape = config.get("shape_config", {})
    sampler = config.get("sampler_config", {})
    args = config.get("train_args", {})
    total = 8

    with PdfPages(pdf_path) as pdf:
        fig, ax = _page("MSF-YOLO V6 数学原理紧凑说明", 1, total)
        _text(
            ax,
            0.05,
            0.88,
            "本版压缩排版，保留论文写作需要的核心数学公式、图表和代码实现对应关系。注意：V6 已实现完整创新链路，但当前实验指标尚未超过 V4，论文中应写为方法探索与实现验证。",
            w=88,
            size=9.2,
        )
        labels = ["训练图像", "困难样本重采样\n小目标增强", "YOLO-P2\n四尺度检测", "Shape-Aware\n检测损失", "best.pt"]
        xs = [0.06, 0.245, 0.43, 0.615, 0.80]
        for x, label in zip(xs, labels):
            _box(ax, x, 0.66, 0.13, 0.09, label)
        for i in range(len(xs) - 1):
            _arrow(ax, (xs[i] + 0.13, 0.705), (xs[i + 1], 0.705))
        _formula(ax, 0.07, 0.57, "theta* = argmin_theta E_{i~p}[ L_det(I_i,Y_i; theta, q) ]", w=0.84)
        _box(ax, 0.18, 0.38, 0.18, 0.09, "整图推理")
        _box(ax, 0.43, 0.38, 0.20, 0.09, "Blade-Slice\n重叠切片融合")
        _box(ax, 0.70, 0.38, 0.18, 0.09, "检测结果\n指标评估")
        _arrow(ax, (0.80, 0.66), (0.55, 0.47))
        _arrow(ax, (0.36, 0.425), (0.43, 0.425))
        _arrow(ax, (0.63, 0.425), (0.70, 0.425))
        rows = [
            ["P2", "进入权重", "stride=4 检测头"],
            ["Shape-Aware Loss", "进入训练", "小目标/细长目标加权"],
            ["Hard Reweight", "进入训练", "corrosion x3, crack x2"],
            ["Blade-Slice", "推理评估", "tile=640, overlap=0.25"],
        ]
        _table(ax, rows, ["模块", "阶段", "实现摘要"], [0.08, 0.13, 0.84, 0.20])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("1. P2 浅层高分辨率检测分支", 2, total)
        _text(ax, 0.05, 0.87, "P2 的数学作用是减小最小检测 stride，使小缺陷在特征图上覆盖更多网格单元，保留浅层纹理、边缘和局部结构。", w=52)
        _formula(ax, 0.05, 0.78, "F_l in R^{C_l x ceil(H/s_l) x ceil(W/s_l)}", w=0.48)
        _formula(ax, 0.05, 0.71, "s_P2=4, s_P3=8, s_P4=16, s_P5=32", w=0.48)
        _formula(ax, 0.05, 0.64, "n_cell(b,l)=sqrt(w_b h_b)/s_l", w=0.48)
        rows = [
            ["P2", "4", "256x256", "小裂纹/小腐蚀/小孔洞"],
            ["P3", "8", "128x128", "小/中目标"],
            ["P4", "16", "64x64", "中目标"],
            ["P5", "32", "32x32", "大目标/语义"],
        ]
        _table(ax, rows, ["层", "stride", "1024输入特征图", "偏关注对象"], [0.05, 0.36, 0.48, 0.20])
        chart = fig.add_axes([0.60, 0.56, 0.33, 0.30])
        sizes = np.array([4, 8, 12, 16, 24, 32, 48])
        for stride, color in [(4, "#F97316"), (8, "#2563EB"), (16, "#16A34A"), (32, "#7C3AED")]:
            chart.plot(sizes, sizes / stride, marker="o", label=f"s={stride}", color=color, linewidth=1.2)
        chart.set_title("小缺陷尺度到网格覆盖数", fontproperties=FONT, fontsize=8.5)
        chart.set_xlabel("缺陷边长/pixel", fontproperties=FONT, fontsize=7)
        chart.set_ylabel("网格数", fontproperties=FONT, fontsize=7)
        chart.grid(alpha=0.25)
        chart.legend(prop=FONT, fontsize=6, ncol=2)
        _formula(ax, 0.59, 0.43, "Y_hat = Detect(F_P2, F_P3, F_P4, F_P5)", w=0.34)
        _text(ax, 0.59, 0.34, "实现位置：configs/yolov8n-p2-wind.yaml，最终 Detect(P2,P3,P4,P5)。", w=36, size=8.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("2. Shape-Aware Loss：面积与细长形态权重", 3, total)
        a0 = float(shape.get("area_threshold", 0.012))
        alpha = float(shape.get("small_gain", 0.35))
        beta = float(shape.get("slender_gain", 0.25))
        r0 = float(shape.get("slender_ratio", 3.0))
        qmax = float(shape.get("max_shape_weight", 1.7))
        _text(ax, 0.05, 0.87, "该损失不做像素级边界监督，而是在正样本 bbox/DFL 损失上乘形态权重 q_i。小面积和细长形态的目标会获得更大梯度贡献。", w=58)
        _formula(ax, 0.05, 0.78, "A_i = w_i h_i / (W H),   R_i = max(w_i/h_i, h_i/w_i)", w=0.54)
        _formula(ax, 0.05, 0.71, f"S_i=clip(({a0:.3f}-A_i)/{a0:.3f},0,1),  E_i=clip((R_i-{r0:.1f})/{r0:.1f},0,1)", w=0.54)
        _formula(ax, 0.05, 0.64, f"q_i=clip(1+{alpha:.2f}S_i+{beta:.2f}E_i, 1, {qmax:.1f})", w=0.54)
        _formula(ax, 0.05, 0.57, "L_box^shape = sum[(1-CIoU_i)t_iq_i] / sum[t_i]", w=0.54)
        _formula(ax, 0.05, 0.50, "L_dfl^shape = sum[DFL_i t_iq_i] / sum[t_i]", w=0.54)
        curve = fig.add_axes([0.64, 0.60, 0.28, 0.25])
        a = np.linspace(0, 0.025, 200)
        s = np.clip((a0 - a) / a0, 0, 1)
        curve.plot(a, s, color="#F97316")
        curve.axvline(a0, linestyle="--", color="#8C959F", linewidth=0.8)
        curve.set_title("小面积项 S_i", fontproperties=FONT, fontsize=8)
        curve.grid(alpha=0.25)
        ratio = fig.add_axes([0.64, 0.25, 0.28, 0.25])
        r = np.linspace(1, 8, 200)
        e = np.clip((r - r0) / r0, 0, 1)
        ratio.plot(r, e, color="#2563EB")
        ratio.axvline(r0, linestyle="--", color="#8C959F", linewidth=0.8)
        ratio.set_title("细长项 E_i", fontproperties=FONT, fontsize=8)
        ratio.grid(alpha=0.25)
        _text(ax, 0.05, 0.34, "梯度解释：grad L_box^shape = (1/T) sum[t_i q_i grad(1-CIoU_i)]。当 q_i>1 时，该目标定位误差的反向传播贡献被放大。", w=56, size=8.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("3. 困难样本采样与类别权重", 4, total)
        cw = float(sampler.get("corrosion_weight", 3.0))
        kw = float(sampler.get("crack_weight", 2.0))
        sw = float(sampler.get("small_target_weight", 1.5))
        wmax = float(sampler.get("max_sample_weight", 6.0))
        gamma = float(args.get("cls_pw", 0.35))
        _text(ax, 0.05, 0.87, "困难样本重加权通过改变图像采样概率 p_i，让腐蚀、裂纹、小目标在训练中更频繁出现；类别权重则改变分类 BCE 中每个类别的损失比例。", w=92)
        _formula(ax, 0.05, 0.77, f"w_i=clip(1.0 * {cw:.1f}^I[corrosion] * {kw:.1f}^I[crack] * {sw:.1f}^I[small], 0.05, {wmax:.1f})", w=0.86)
        _formula(ax, 0.05, 0.70, "p_i = w_i / sum_j w_j,     E[m_i] = N p_i", w=0.48)
        _formula(ax, 0.05, 0.63, f"alpha_tilde_c=(1/n_c)^{gamma:.2f},   alpha_c=alpha_tilde_c / mean(alpha_tilde)", w=0.86)
        _formula(ax, 0.05, 0.56, "L_cls^cw = sum_{j,c} alpha_c BCE(s_hat_{j,c},s_{j,c}) / sum_j t_j", w=0.86)
        rows = [
            ["普通样本", "无触发", "1.0"],
            ["裂纹", "crack", f"{kw:.1f}"],
            ["腐蚀", "corrosion", f"{cw:.1f}"],
            ["小裂纹", "crack + small", f"{kw * sw:.1f}"],
            ["腐蚀+裂纹", "corrosion + crack", f"{min(cw * kw, wmax):.1f}"],
            ["三者同时满足", "max cap", f"{wmax:.1f}"],
        ]
        _table(ax, rows, ["样本类型", "触发条件", "采样权重"], [0.07, 0.18, 0.45, 0.32])
        chart = fig.add_axes([0.61, 0.20, 0.30, 0.30])
        vals = [1, kw, cw, kw * sw, min(cw * kw, wmax), wmax]
        chart.bar(range(len(vals)), vals, color=["#9CA3AF", "#2563EB", "#F97316", "#16A34A", "#7C3AED", "#DC2626"])
        chart.set_xticks(range(len(vals)), ["N", "Ck", "Co", "S", "C+C", "Max"], fontproperties=FONT)
        chart.set_ylabel("w_i", fontproperties=FONT, fontsize=7)
        chart.set_title("采样权重示意", fontproperties=FONT, fontsize=8)
        chart.grid(axis="y", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("4. 小目标增强与 Blade-Slice 切片推理", 5, total)
        _text(ax, 0.05, 0.87, "小目标增强扩展训练分布；Blade-Slice 在推理阶段用局部窗口缓解整图缩放带来的小缺陷信息损失。二者分别作用于训练分布和推理尺度。", w=92)
        _formula(ax, 0.05, 0.77, "I' = T_phi(I),   B'_k = T_phi(B_k),   phi ~ P_aug", w=0.58)
        rows = [
            ["imgsz", str(args.get("imgsz", 1024)), "减少缩放损失"],
            ["multi_scale", str(args.get("multi_scale", 0.10)), "尺度鲁棒性"],
            ["mosaic", str(args.get("mosaic", 1.0)), "组合背景"],
            ["copy_paste", str(args.get("copy_paste", 0.05)), "增加少数类/小目标"],
            ["scale", str(args.get("scale", 0.60)), "改变相对面积"],
        ]
        _table(ax, rows, ["增强项", "值", "作用"], [0.05, 0.42, 0.40, 0.23])
        _formula(ax, 0.52, 0.77, "d = floor(L(1-rho)),  L=640, rho=0.25, d=480", w=0.42)
        _formula(ax, 0.52, 0.70, "C_{u,v}=I[u:u+L, v:v+L]", w=0.42)
        _formula(ax, 0.52, 0.63, "b_global=(x1+u,y1+v,x2+u,y2+v,c,p)", w=0.42)
        _formula(ax, 0.52, 0.56, "B_fused=NMS(B_full union B_slice, tau)", w=0.42)
        tile = fig.add_axes([0.55, 0.18, 0.35, 0.24])
        tile.set_xlim(0, 10)
        tile.set_ylim(0, 6)
        tile.axis("off")
        tile.add_patch(Rectangle((0.5, 0.5), 9, 5, facecolor="#F8FAFC", edgecolor="#111827", linewidth=1.0))
        for i, (x, y) in enumerate([(0.8, 0.8), (3.2, 0.8), (0.8, 2.5), (3.2, 2.5)]):
            tile.add_patch(Rectangle((x, y), 4.1, 2.4, facecolor=["#DBEAFE", "#FFEDD5", "#DCFCE7", "#F3E8FF"][i], alpha=0.75, edgecolor="#57606A"))
        tile.text(5, 5.65, "重叠切片示意", ha="center", fontsize=7.5, fontproperties=FONT)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("5. NMS 融合与评价指标", 6, total)
        _text(ax, 0.05, 0.87, "切片融合后的检测框需要按类别和置信度做 NMS。最终评价仍以 Precision、Recall、mAP50-95 和背景误检为主。", w=90)
        _formula(ax, 0.05, 0.78, "IoU(b_i,b_j)=area(intersection)/area(union)", w=0.55)
        _formula(ax, 0.05, 0.71, "remove b_j if c_i=c_j, p_i>=p_j, IoU(b_i,b_j)>=tau", w=0.76)
        _formula(ax, 0.05, 0.64, "Precision=TP/(TP+FP),   Recall=TP/(TP+FN)", w=0.55)
        _formula(ax, 0.05, 0.57, "mAP50-95=(1/10) sum_{tau=0.50:0.05:0.95} mAP_tau", w=0.76)
        _formula(ax, 0.05, 0.50, "FP_bg=sum_{empty images} #{boxes with conf>=eta}", w=0.55)
        rows = [
            ["mAP50", "IoU=0.50 的 AP 均值", "宽松定位能力"],
            ["mAP50-95", "0.50 到 0.95 的均值", "更严格定位质量"],
            ["Recall", "TP/(TP+FN)", "漏检风险"],
            ["FP_bg", "空标签图预测框数量", "复杂背景鲁棒性"],
        ]
        _table(ax, rows, ["指标", "定义", "解释"], [0.08, 0.18, 0.84, 0.28])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("6. 代码映射与实现边界", 7, total)
        rows = [
            ["P2检测头", "configs/yolov8n-p2-wind.yaml", "进入.pt"],
            ["Shape-Aware Loss", "scripts/train_msf_yolo_full.py", "训练损失"],
            ["困难样本采样", "scripts/train_msf_yolo_full.py", "训练分布"],
            ["类别权重", "本地 ultralytics detect/train.py + loss.py", "分类损失"],
            ["小目标增强", "train args", "训练样本"],
            ["Blade-Slice", "scripts/evaluate_blade_slice.py", "推理/评估"],
        ]
        _table(ax, rows, ["模块", "代码位置", "作用阶段"], [0.05, 0.58, 0.90, 0.30], size=7.0)
        _text(ax, 0.06, 0.48, "关键参数：area_threshold=0.012, small_gain=0.35, slender_gain=0.25, slender_ratio=3.0, max_shape_weight=1.7；采样为 corrosion x3、crack x2、小目标 x1.5、max_sample_weight=6。", w=90)
        _text(ax, 0.06, 0.34, "论文边界：P2、Shape-Aware、困难样本重加权、类别权重、小目标增强属于训练/结构链路；Blade-Slice 是推理/评估策略，不写进权重。", w=90, color="#9A3412")
        _text(ax, 0.06, 0.22, f"实验名：{run_dir.name}", w=90, size=8.0, color="#57606A")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("7. 当前实验结果与论文写法建议", 8, total)
        rows = [
            ["val mAP50-95", "V4 0.8093", "V6 0.7922", "-0.0171"],
            ["val Recall", "V4 0.9411", "V6 0.9071", "-0.0340"],
            ["test_clean mAP50-95", "V4 0.8401", "V6 0.8301", "-0.0100"],
            ["test_clean Recall", "V4 0.9792", "V6 0.9541", "-0.0251"],
            ["test_clean 背景误检", "V4 1", "V6 1", "持平"],
        ]
        _table(ax, rows, ["指标", "V4", "MSF-YOLO V6", "变化"], [0.08, 0.58, 0.84, 0.25], size=7.5)
        _text(ax, 0.08, 0.46, "严谨写法：本文提出并实现 MSF-YOLO，用于探索 P2 小目标检测头、切片融合推理、形态感知损失和困难样本重加权在风机叶片缺陷检测中的作用。当前实验表明该链路具备完整实现基础，但在现有训练策略下尚未超过 V4 基线。", w=92, size=9)
        _text(ax, 0.08, 0.28, "不建议写法：MSF-YOLO 显著提升召回率和工程可靠性。除非后续实验获得正向结果，否则这句话风险很高。", w=92, size=9, color="#9A3412")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return pdf_path


def _compact_pdf_clean(run_dir: Path, config: dict[str, Any], output_dir: Path) -> Path:
    pdf_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明_紧凑清晰版.pdf"
    shape = config.get("shape_config", {})
    sampler = config.get("sampler_config", {})
    args = config.get("train_args", {})
    total = 7

    with PdfPages(pdf_path) as pdf:
        fig, ax = _page("MSF-YOLO V6 数学原理紧凑清晰版", 1, total)
        _text(
            ax,
            0.05,
            0.885,
            "本版重新排版：训练链路、推理链路、公式和表格分区显示，避免箭头穿过表格或模块框重叠。内容用于论文方法章节参考；当前 V6 已实现完整创新链路，但实验指标尚未超过 V4。",
            w=96,
            size=9,
        )
        ax.text(0.07, 0.775, "训练阶段：写入模型权重", fontsize=9.5, fontproperties=FONT, weight="bold", color="#17212B")
        ax.text(0.58, 0.775, "推理/评估阶段：不写入权重", fontsize=9.5, fontproperties=FONT, weight="bold", color="#17212B")
        train_items = [
            ("训练图像与标注", "#F6F8FA"),
            ("困难样本重采样\n小目标增强", "#E0F2FE"),
            ("YOLO-P2\n四尺度检测头", "#DCFCE7"),
            ("Shape-Aware Loss\n类别权重", "#FFEDD5"),
            ("best.pt\n候选权重", "#F3E8FF"),
        ]
        y = 0.665
        for idx, (label, color) in enumerate(train_items):
            _box(ax, 0.08, y - idx * 0.095, 0.30, 0.065, label, fc=color)
            if idx < len(train_items) - 1:
                _arrow(ax, (0.23, y - idx * 0.095), (0.23, y - (idx + 1) * 0.095 + 0.065))

        infer_items = [
            ("整图推理", "#F6F8FA"),
            ("重叠切片\n坐标回映射", "#E0F2FE"),
            ("整图框 + 切片框\nNMS 融合", "#DCFCE7"),
            ("检测结果\n指标评估", "#FFEDD5"),
        ]
        y = 0.62
        for idx, (label, color) in enumerate(infer_items):
            _box(ax, 0.61, y - idx * 0.105, 0.27, 0.07, label, fc=color)
            if idx < len(infer_items) - 1:
                _arrow(ax, (0.745, y - idx * 0.105), (0.745, y - (idx + 1) * 0.105 + 0.07))

        _formula(ax, 0.055, 0.205, "theta* = argmin_theta E_{i~p}[ L_det(I_i,Y_i; theta, q) ]", w=0.52)
        rows = [
            ["P2", "结构/训练", "Detect(P2,P3,P4,P5)"],
            ["Shape-Aware", "训练损失", "小面积/细长目标加权"],
            ["Hard Reweight", "训练采样", "corrosion x3, crack x2"],
            ["Blade-Slice", "推理评估", "tile=640, overlap=0.25"],
        ]
        _table(ax, rows, ["模块", "阶段", "核心作用"], [0.60, 0.105, 0.34, 0.18], size=6.8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("1. P2 浅层高分辨率检测分支", 2, total)
        _text(ax, 0.05, 0.88, "P2 的数学作用是减小最小检测 stride。stride 越小，同一小缺陷在特征图上覆盖的网格越多，越有利于保留浅层纹理和边缘结构。", w=58, size=8.8)
        _formula(ax, 0.05, 0.78, "F_l in R^{C_l x ceil(H/s_l) x ceil(W/s_l)}", w=0.46)
        _formula(ax, 0.05, 0.71, "s_P2=4, s_P3=8, s_P4=16, s_P5=32", w=0.46)
        _formula(ax, 0.05, 0.64, "n_cell(b,l)=sqrt(w_b h_b)/s_l", w=0.46)
        _formula(ax, 0.05, 0.57, "Y_hat=Detect(F_P2,F_P3,F_P4,F_P5)", w=0.46)
        rows = [
            ["P2", "4", "256x256", "小裂纹/小腐蚀/小孔洞"],
            ["P3", "8", "128x128", "小/中目标"],
            ["P4", "16", "64x64", "中目标"],
            ["P5", "32", "32x32", "大目标/语义"],
        ]
        _table(ax, rows, ["层", "stride", "1024输入特征图", "偏关注对象"], [0.05, 0.28, 0.48, 0.22], size=6.8)
        chart = fig.add_axes([0.61, 0.36, 0.31, 0.36])
        sizes = np.array([4, 8, 12, 16, 24, 32, 48])
        for stride, color in [(4, "#F97316"), (8, "#2563EB"), (16, "#16A34A"), (32, "#7C3AED")]:
            chart.plot(sizes, sizes / stride, marker="o", label=f"s={stride}", color=color, linewidth=1.2)
        chart.set_title("缺陷像素尺度到网格覆盖数", fontproperties=FONT, fontsize=8)
        chart.set_xlabel("缺陷边长 / pixel", fontproperties=FONT, fontsize=7)
        chart.set_ylabel("网格数", fontproperties=FONT, fontsize=7)
        chart.grid(alpha=0.25)
        chart.legend(prop=FONT, fontsize=6, ncol=2)
        _text(ax, 0.61, 0.25, "实现位置：configs/yolov8n-p2-wind.yaml。该模块进入 .pt 权重。", w=38, size=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("2. Shape-Aware Loss：形态权重", 3, total)
        a0 = float(shape.get("area_threshold", 0.012))
        alpha = float(shape.get("small_gain", 0.35))
        beta = float(shape.get("slender_gain", 0.25))
        r0 = float(shape.get("slender_ratio", 3.0))
        qmax = float(shape.get("max_shape_weight", 1.7))
        _text(ax, 0.05, 0.88, "Shape-Aware Loss 是框级形态先验：小面积目标和细长目标获得更大 bbox/DFL 损失权重，从而放大定位误差的梯度贡献。", w=65, size=8.8)
        _formula(ax, 0.05, 0.78, "A_i=w_i h_i/(W H),   R_i=max(w_i/h_i,h_i/w_i)", w=0.52)
        _formula(ax, 0.05, 0.71, f"S_i=clip(({a0:.3f}-A_i)/{a0:.3f},0,1)", w=0.38)
        _formula(ax, 0.05, 0.64, f"E_i=clip((R_i-{r0:.1f})/{r0:.1f},0,1)", w=0.38)
        _formula(ax, 0.05, 0.57, f"q_i=clip(1+{alpha:.2f}S_i+{beta:.2f}E_i,1,{qmax:.1f})", w=0.46)
        _formula(ax, 0.05, 0.50, "L_box^shape=sum[(1-CIoU_i)t_iq_i]/sum[t_i]", w=0.52)
        _formula(ax, 0.05, 0.43, "L_dfl^shape=sum[DFL_i t_iq_i]/sum[t_i]", w=0.52)
        curve = fig.add_axes([0.62, 0.58, 0.30, 0.25])
        a = np.linspace(0, 0.025, 200)
        curve.plot(a, np.clip((a0 - a) / a0, 0, 1), color="#F97316")
        curve.axvline(a0, linestyle="--", color="#8C959F", linewidth=0.8)
        curve.set_title("小面积项 S_i", fontproperties=FONT, fontsize=8)
        curve.grid(alpha=0.25)
        ratio = fig.add_axes([0.62, 0.25, 0.30, 0.25])
        r = np.linspace(1, 8, 200)
        ratio.plot(r, np.clip((r - r0) / r0, 0, 1), color="#2563EB")
        ratio.axvline(r0, linestyle="--", color="#8C959F", linewidth=0.8)
        ratio.set_title("细长项 E_i", fontproperties=FONT, fontsize=8)
        ratio.grid(alpha=0.25)
        _text(ax, 0.05, 0.31, "梯度解释：grad L=(1/T) sum[t_i q_i grad(loss_i)]。q_i>1 时，该类目标对反向传播贡献更大。", w=58, size=8.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("3. 困难样本重采样与类别权重", 4, total)
        cw = float(sampler.get("corrosion_weight", 3.0))
        kw = float(sampler.get("crack_weight", 2.0))
        sw = float(sampler.get("small_target_weight", 1.5))
        wmax = float(sampler.get("max_sample_weight", 6.0))
        gamma = float(args.get("cls_pw", 0.35))
        _text(ax, 0.05, 0.88, "困难样本机制改变图像被抽到的概率；类别权重改变分类 BCE 中各类别损失比例。两者分别作用于采样分布和分类损失。", w=92, size=8.8)
        _formula(ax, 0.05, 0.78, f"w_i=clip(1.0*{cw:.1f}^I[corrosion]*{kw:.1f}^I[crack]*{sw:.1f}^I[small],0.05,{wmax:.1f})", w=0.86)
        _formula(ax, 0.05, 0.71, "p_i=w_i/sum_j w_j,   E[m_i]=N p_i", w=0.42)
        _formula(ax, 0.05, 0.64, f"alpha_tilde_c=(1/n_c)^{gamma:.2f},   alpha_c=alpha_tilde_c/mean(alpha_tilde)", w=0.78)
        _formula(ax, 0.05, 0.57, "L_cls^cw=sum alpha_c BCE(s_hat,s)/sum t_j", w=0.58)
        rows = [
            ["普通", "无触发", "1.0"],
            ["裂纹", "crack", f"{kw:.1f}"],
            ["腐蚀", "corrosion", f"{cw:.1f}"],
            ["小裂纹", "crack+small", f"{kw * sw:.1f}"],
            ["腐蚀+裂纹", "corrosion+crack", f"{min(cw * kw, wmax):.1f}"],
            ["上限", "max cap", f"{wmax:.1f}"],
        ]
        _table(ax, rows, ["样本类型", "触发条件", "采样权重"], [0.07, 0.18, 0.44, 0.32], size=6.8)
        chart = fig.add_axes([0.61, 0.20, 0.30, 0.30])
        vals = [1, kw, cw, kw * sw, min(cw * kw, wmax), wmax]
        chart.bar(range(len(vals)), vals, color=["#9CA3AF", "#2563EB", "#F97316", "#16A34A", "#7C3AED", "#DC2626"])
        chart.set_xticks(range(len(vals)), ["N", "Cr", "Co", "S", "C+C", "Max"], fontproperties=FONT)
        chart.set_ylabel("w_i", fontproperties=FONT, fontsize=7)
        chart.set_title("采样权重示意", fontproperties=FONT, fontsize=8)
        chart.grid(axis="y", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("4. 小目标增强与 Blade-Slice", 5, total)
        _text(ax, 0.05, 0.88, "小目标增强扩展训练分布；Blade-Slice 在推理阶段用局部窗口缓解整图缩放造成的小缺陷信息损失。", w=90, size=8.8)
        _formula(ax, 0.05, 0.78, "I'=T_phi(I),   B'_k=T_phi(B_k),   phi~P_aug", w=0.52)
        rows = [
            ["imgsz", str(args.get("imgsz", 1024)), "减少缩放损失"],
            ["multi_scale", str(args.get("multi_scale", 0.10)), "尺度鲁棒性"],
            ["mosaic", str(args.get("mosaic", 1.0)), "组合背景"],
            ["copy_paste", str(args.get("copy_paste", 0.05)), "增加少数类/小目标"],
            ["scale", str(args.get("scale", 0.60)), "改变相对面积"],
        ]
        _table(ax, rows, ["增强项", "值", "作用"], [0.05, 0.42, 0.41, 0.23], size=6.8)
        _formula(ax, 0.54, 0.78, "d=floor(L(1-rho)), L=640, rho=0.25, d=480", w=0.40)
        _formula(ax, 0.54, 0.70, "C_{u,v}=I[u:u+L, v:v+L]", w=0.36)
        _formula(ax, 0.54, 0.62, "b_global=(x1+u,y1+v,x2+u,y2+v,c,p)", w=0.40)
        _formula(ax, 0.54, 0.54, "B_fused=NMS(B_full union B_slice,tau)", w=0.40)
        tile = fig.add_axes([0.56, 0.19, 0.33, 0.25])
        tile.set_xlim(0, 10)
        tile.set_ylim(0, 6)
        tile.axis("off")
        tile.add_patch(Rectangle((0.5, 0.5), 9, 5, facecolor="#F8FAFC", edgecolor="#111827", linewidth=1.0))
        for i, (x, y0) in enumerate([(0.8, 0.8), (3.2, 0.8), (0.8, 2.5), (3.2, 2.5)]):
            tile.add_patch(Rectangle((x, y0), 4.1, 2.4, facecolor=["#DBEAFE", "#FFEDD5", "#DCFCE7", "#F3E8FF"][i], alpha=0.75, edgecolor="#57606A"))
        tile.text(5, 5.65, "重叠切片示意", ha="center", fontsize=7.5, fontproperties=FONT)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("5. NMS、评价指标与代码映射", 6, total)
        _formula(ax, 0.05, 0.82, "remove b_j if c_i=c_j, p_i>=p_j, IoU(b_i,b_j)>=tau", w=0.70)
        _formula(ax, 0.05, 0.75, "Precision=TP/(TP+FP),   Recall=TP/(TP+FN)", w=0.52)
        _formula(ax, 0.05, 0.68, "mAP50-95=(1/10) sum_{0.50:0.05:0.95} mAP_tau", w=0.58)
        _formula(ax, 0.05, 0.61, "FP_bg=sum_{empty images} #{boxes with conf>=eta}", w=0.52)
        rows = [
            ["P2检测头", "configs/yolov8n-p2-wind.yaml", "进入.pt"],
            ["Shape-Aware Loss", "scripts/train_msf_yolo_full.py", "训练损失"],
            ["困难样本采样", "scripts/train_msf_yolo_full.py", "训练分布"],
            ["类别权重", "本地 ultralytics", "分类损失"],
            ["Blade-Slice", "scripts/evaluate_blade_slice.py", "推理/评估"],
        ]
        _table(ax, rows, ["模块", "代码位置", "作用阶段"], [0.05, 0.20, 0.90, 0.30], size=6.8)
        _text(ax, 0.05, 0.12, "边界：Blade-Slice 是推理/评估策略，不写入 .pt；其余结构、损失和采样策略影响训练链路。", w=92, size=8.2, color="#9A3412")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _page("6. 当前结果与论文写法建议", 7, total)
        rows = [
            ["val mAP50-95", "0.8093", "0.7922", "-0.0171"],
            ["val Recall", "0.9411", "0.9071", "-0.0340"],
            ["test_clean mAP50-95", "0.8401", "0.8301", "-0.0100"],
            ["test_clean Recall", "0.9792", "0.9541", "-0.0251"],
            ["test_clean 背景误检", "1", "1", "持平"],
        ]
        _table(ax, rows, ["指标", "V4", "MSF-YOLO V6", "变化"], [0.08, 0.60, 0.84, 0.24], size=7.2)
        _text(ax, 0.08, 0.47, "严谨写法：本文提出并实现 MSF-YOLO，用于探索 P2 小目标检测头、切片融合推理、形态感知损失和困难样本重加权在风机叶片缺陷检测中的作用。当前实验表明该链路具备完整实现基础，但在现有训练策略下尚未超过 V4 基线。", w=92, size=8.8)
        _text(ax, 0.08, 0.27, "不建议写法：MSF-YOLO 显著提升召回率和工程可靠性。除非后续实验获得正向结果，否则这句话风险很高。", w=92, size=8.8, color="#9A3412")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

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


def _compact_docx(run_dir: Path, config: dict[str, Any], output_dir: Path) -> Path:
    docx_path = output_dir / "MSF-YOLO_V6_数学原理与公式说明.docx"
    shape = config.get("shape_config", {})
    sampler = config.get("sampler_config", {})
    args = config.get("train_args", {})

    body: list[str] = []
    body.append(_heading("MSF-YOLO V6 数学原理与公式说明", 1))
    body.append(_p(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}"))
    body.append(_p(f"实验名：{run_dir.name}"))
    body.append(_p("说明：本文档为 Word 版，可用于论文方法章节参考。当前 MSF-YOLO V6 已实现完整创新链路，但实验指标尚未超过 V4，因此论文中应区分“提出并实现”和“效果提升”。"))

    body.append(_heading("1. 总体优化目标", 2))
    body.append(_p("MSF-YOLO V6 可以抽象为在困难样本采样分布下，最小化带形态权重的检测损失。"))
    body.append(_formula_p("theta* = argmin_theta E_{i~p}[ L_det(I_i, Y_i; theta, q) ]"))
    body.append(_p("其中 p_i 来自 WeightedRandomSampler，q_i 来自 Shape-Aware Loss。"))

    body.append(_heading("2. P2 浅层高分辨率检测分支", 2))
    body.append(_formula_p("F_l in R^{C_l x ceil(H/s_l) x ceil(W/s_l)}"))
    body.append(_formula_p("s_P2=4, s_P3=8, s_P4=16, s_P5=32"))
    body.append(_formula_p("n_cell(b,l)=sqrt(w_b h_b)/s_l"))
    body.append(_tbl(["检测层", "stride", "1024 输入下特征图", "作用"], [
        ["P2", "4", "256 x 256", "小裂纹、小腐蚀、小孔洞"],
        ["P3", "8", "128 x 128", "小/中目标"],
        ["P4", "16", "64 x 64", "中目标"],
        ["P5", "32", "32 x 32", "大目标/全局语义"],
    ]))
    body.append(_formula_p("Y_hat = Detect(F_P2, F_P3, F_P4, F_P5)"))

    body.append(_heading("3. Shape-Aware Loss", 2))
    body.append(_p("Shape-Aware Loss 在正样本 bbox/DFL 损失上乘形态权重，增强小面积和细长目标的定位梯度。"))
    body.append(_formula_p("A_i = w_i h_i / (W H),   R_i = max(w_i/h_i, h_i/w_i)"))
    body.append(_formula_p(f"S_i = clip(({shape.get('area_threshold', 0.012):.3f} - A_i) / {shape.get('area_threshold', 0.012):.3f}, 0, 1)"))
    body.append(_formula_p(f"E_i = clip((R_i - {shape.get('slender_ratio', 3.0):.1f}) / {shape.get('slender_ratio', 3.0):.1f}, 0, 1)"))
    body.append(_formula_p(f"q_i = clip(1 + {shape.get('small_gain', 0.35):.2f} S_i + {shape.get('slender_gain', 0.25):.2f} E_i, 1, {shape.get('max_shape_weight', 1.7):.1f})"))
    body.append(_formula_p("L_box^shape = sum[(1-CIoU_i) t_i q_i] / sum[t_i]"))
    body.append(_formula_p("L_dfl^shape = sum[DFL_i t_i q_i] / sum[t_i]"))
    body.append(_formula_p("grad L_box^shape = (1/T) sum[t_i q_i grad(1-CIoU_i)]"))

    body.append(_heading("4. 困难样本重加权训练机制", 2))
    body.append(_formula_p(
        f"w_i = clip(1.0 * {sampler.get('corrosion_weight', 3.0):.1f}^I[corrosion] * "
        f"{sampler.get('crack_weight', 2.0):.1f}^I[crack] * {sampler.get('small_target_weight', 1.5):.1f}^I[small], "
        f"0.05, {sampler.get('max_sample_weight', 6.0):.1f})"
    ))
    body.append(_formula_p("I[small]=1 if c in {crack, corrosion} and A_i <= 0.012"))
    body.append(_formula_p("p_i = w_i / sum_j w_j,    E[m_i] = N p_i"))
    body.append(_tbl(["样本类型", "触发条件", "采样权重"], [
        ["普通样本", "无触发", "1.0"],
        ["裂纹", "crack", str(sampler.get("crack_weight", 2.0))],
        ["腐蚀", "corrosion", str(sampler.get("corrosion_weight", 3.0))],
        ["小目标", "small", str(sampler.get("small_target_weight", 1.5))],
        ["上限", "max cap", str(sampler.get("max_sample_weight", 6.0))],
    ]))

    body.append(_heading("5. 类别权重 cls_pw", 2))
    body.append(_formula_p(f"alpha_tilde_c = (1 / n_c)^{args.get('cls_pw', 0.35)}"))
    body.append(_formula_p("alpha_c = alpha_tilde_c / mean(alpha_tilde)"))
    body.append(_formula_p("L_cls^cw = sum_{j,c} alpha_c BCE(s_hat_{j,c}, s_{j,c}) / sum_j t_j"))

    body.append(_heading("6. 小目标增强", 2))
    body.append(_formula_p("I' = T_phi(I),    B'_k = T_phi(B_k),    phi ~ P_aug"))
    body.append(_tbl(["参数", "当前值", "作用"], [
        ["imgsz", str(args.get("imgsz", 1024)), "提高输入分辨率"],
        ["multi_scale", str(args.get("multi_scale", 0.10)), "尺度鲁棒性"],
        ["mosaic", str(args.get("mosaic", 1.0)), "组合背景"],
        ["copy_paste", str(args.get("copy_paste", 0.05)), "增加少数类/小目标"],
        ["scale", str(args.get("scale", 0.60)), "改变目标相对面积"],
    ]))

    body.append(_heading("7. Blade-Slice 重叠切片融合推理", 2))
    body.append(_formula_p("d = floor(L(1-rho)),   L=640, rho=0.25, d=480"))
    body.append(_formula_p("S_x={0,d,2d,...,W-L},   S_y={0,d,2d,...,H-L}"))
    body.append(_formula_p("C_{u,v}=I[u:u+L, v:v+L]"))
    body.append(_formula_p("b_global=(x1+u, y1+v, x2+u, y2+v, c, p)"))
    body.append(_formula_p("B_fused = NMS(B_full union B_slice, tau)"))
    body.append(_p("注意：Blade-Slice 是推理/评估策略，不写入 .pt 权重。"))

    body.append(_heading("8. NMS 与评价指标", 2))
    body.append(_formula_p("IoU(b_i,b_j)=area(intersection)/area(union)"))
    body.append(_formula_p("remove b_j if c_i=c_j, p_i>=p_j, IoU(b_i,b_j)>=tau"))
    body.append(_formula_p("Precision=TP/(TP+FP),    Recall=TP/(TP+FN)"))
    body.append(_formula_p("mAP50-95=(1/10) sum_{tau=0.50:0.05:0.95} mAP_tau"))
    body.append(_formula_p("FP_bg=sum_{empty images} #{boxes with conf>=eta}"))

    body.append(_heading("9. 当前实验结果与论文表述边界", 2))
    body.append(_tbl(["指标", "V4", "MSF-YOLO V6", "变化"], [
        ["val mAP50-95", "0.8093", "0.7922", "-0.0171"],
        ["val Recall", "0.9411", "0.9071", "-0.0340"],
        ["test_clean mAP50-95", "0.8401", "0.8301", "-0.0100"],
        ["test_clean Recall", "0.9792", "0.9541", "-0.0251"],
        ["test_clean 背景误检", "1", "1", "持平"],
    ]))
    body.append(_p("严谨写法：本文提出并实现 MSF-YOLO，用于探索 P2 小目标检测头、切片融合推理、形态感知损失和困难样本重加权在风机叶片缺陷检测中的作用。当前实验表明该链路具备完整实现基础，但在现有训练策略下尚未超过 V4 基线。"))

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
    parser = argparse.ArgumentParser(description="Generate compact MSF-YOLO principle PDF and Word docx.")
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
    pdf_path = _compact_pdf_clean(run_dir, config, output_dir)
    docx_path = _compact_docx(run_dir, config, output_dir)
    manifest = {
        "compact_pdf": str(pdf_path),
        "word_docx": str(docx_path),
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "font": FONT_PATH,
        "note": "Compact PDF and Word docx for MSF-YOLO V6 mathematical principles.",
    }
    (output_dir / "compact_principle_outputs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pdf_path)
    print(docx_path)


if __name__ == "__main__":
    main()
