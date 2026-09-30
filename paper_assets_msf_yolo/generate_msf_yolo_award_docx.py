from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT_DIR = Path("paper_assets_msf_yolo")
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = OUT_DIR / "MSF-YOLO_论文方法与实验设计_获奖论文风格初稿.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def text_run(text: object, bold: bool = False, italic: bool = False) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    chunks = str(text).split("\n")
    body = []
    for idx, chunk in enumerate(chunks):
        if idx:
            body.append("<w:br/>")
        body.append(f'<w:t xml:space="preserve">{esc(chunk)}</w:t>')
    return f"<w:r>{rpr}{''.join(body)}</w:r>"


def para(
    text: object = "",
    style: str | None = None,
    align: str | None = None,
    bold: bool = False,
    keep_next: bool = False,
) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if keep_next:
        ppr.append("<w:keepNext/>")
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr_xml}{text_run(text, bold=bold)}</w:p>"


def mixed_para(parts: list[tuple[str, bool]], style: str | None = None, align: str | None = None) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    runs = "".join(text_run(text, bold=bold) for text, bold in parts)
    return f"<w:p>{ppr_xml}{runs}</w:p>"


def formula(text: object) -> str:
    rpr = (
        '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" '
        'w:eastAsia="Cambria Math"/><w:sz w:val="21"/></w:rPr>'
    )
    ppr = '<w:pPr><w:pStyle w:val="Formula"/><w:jc w:val="center"/></w:pPr>'
    body = []
    for idx, line in enumerate(str(text).split("\n")):
        if idx:
            body.append("<w:br/>")
        body.append(f'<w:t xml:space="preserve">{esc(line)}</w:t>')
    return f"<w:p>{ppr}<w:r>{rpr}{''.join(body)}</w:r></w:p>"


def cell(content: object, header: bool = False, width: int | None = None) -> str:
    tcpr = []
    if width is not None:
        tcpr.append(f'<w:tcW w:w="{width}" w:type="dxa"/>')
    if header:
        tcpr.append('<w:shd w:fill="D9EAF7"/>')
    tcpr.append('<w:vAlign w:val="center"/>')
    tcpr_xml = f"<w:tcPr>{''.join(tcpr)}</w:tcPr>"
    if isinstance(content, list):
        inner = "".join(content)
    else:
        inner = para(content, bold=header)
    return f"<w:tc>{tcpr_xml}{inner}</w:tc>"


def table(rows: list[list[object]], widths: list[int] | None = None) -> str:
    tbl_pr = """
    <w:tblPr>
      <w:tblStyle w:val="TableGrid"/>
      <w:tblW w:w="5000" w:type="pct"/>
      <w:tblLook w:val="04A0"/>
      <w:tblBorders>
        <w:top w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:left w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:bottom w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:right w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:insideH w:val="single" w:sz="6" w:space="0" w:color="BFC9CA"/>
        <w:insideV w:val="single" w:sz="6" w:space="0" w:color="BFC9CA"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="80" w:type="dxa"/>
        <w:left w:w="100" w:type="dxa"/>
        <w:bottom w:w="80" w:type="dxa"/>
        <w:right w:w="100" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>
    """
    tr_xml = []
    for ridx, row in enumerate(rows):
        cells = []
        for cidx, value in enumerate(row):
            width = widths[cidx] if widths and cidx < len(widths) else None
            cells.append(cell(value, header=(ridx == 0), width=width))
        tr_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return f"<w:tbl>{tbl_pr}{''.join(tr_xml)}</w:tbl>"


def image(rid: str, docpr_id: int, name: str, cx: int, cy: int) -> str:
    return f"""
    <w:p>
      <w:pPr><w:jc w:val="center"/></w:pPr>
      <w:r>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:effectExtent l="0" t="0" r="0" b="0"/>
            <wp:docPr id="{docpr_id}" name="{esc(name)}"/>
            <wp:cNvGraphicFramePr>
              <a:graphicFrameLocks noChangeAspect="1"/>
            </wp:cNvGraphicFramePr>
            <a:graphic>
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="{docpr_id}" name="{esc(name)}"/>
                    <pic:cNvPicPr/>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{rid}"/>
                    <a:stretch><a:fillRect/></a:stretch>
                  </pic:blipFill>
                  <pic:spPr>
                    <a:xfrm>
                      <a:off x="0" y="0"/>
                      <a:ext cx="{cx}" cy="{cy}"/>
                    </a:xfrm>
                    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                  </pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
    """


def caption(text: str) -> str:
    return para(text, style="Caption", align="center")


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS['w']}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
        <w:sz w:val="21"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="360" w:lineRule="auto" w:before="0" w:after="120"/>
        <w:jc w:val="both"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="240" w:after="240"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:after="180"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="300" w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="220" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="160" w:after="80"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="19"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Formula">
    <w:name w:val="Formula"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="80"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="Cambria Math"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr>
      <w:tblBorders>
        <w:top w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:left w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:bottom w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:right w:val="single" w:sz="8" w:space="0" w:color="6C7A89"/>
        <w:insideH w:val="single" w:sz="6" w:space="0" w:color="BFC9CA"/>
        <w:insideV w:val="single" w:sz="6" w:space="0" w:color="BFC9CA"/>
      </w:tblBorders>
    </w:tblPr>
  </w:style>
</w:styles>
"""


def document_xml() -> str:
    body: list[str] = []

    body.append(para("基于 MSF-YOLO 的风机叶片缺陷检测方法研究", style="Title"))
    body.append(para("论文方法章节与实验设计初稿", style="Subtitle"))
    body.append(para("摘要", style="Heading1"))
    body.append(
        para(
            "风机叶片在长期服役过程中易受到疲劳载荷、雨蚀、沙尘和雷击等因素影响，"
            "产生裂纹、孔洞、腐蚀和表面剥落等缺陷。由于巡检图像分辨率高、背景纹理复杂，"
            "缺陷目标常呈现尺度小、形态细长和类别分布不均衡等特点，传统检测模型在小目标召回和困难类别定位方面仍存在不足。"
            "针对上述问题，本文构建一种多尺度形态感知检测方法 MSF-YOLO。该方法在 YOLO 检测框架基础上引入 P2 浅层高分辨率检测分支，"
            "提高小尺度缺陷在特征图上的表达能力；设计 Shape-Aware 形态感知损失，将目标面积占比和长宽比转化为框级损失权重，"
            "增强模型对小面积和细长缺陷的定位约束；同时结合困难样本重加权训练与 Blade-Slice 重叠切片融合推理，"
            "分别从训练样本分布和推理尺度补偿两个方面改善复杂场景下的检测稳定性。本文进一步设计了逐项消融实验、分类别指标统计和背景误检分析，"
            "用于验证各模块对风机叶片缺陷检测任务的实际贡献。"
        )
    )
    body.append(mixed_para([("关键词：", True), ("风机叶片；缺陷检测；YOLO；小目标检测；形态感知损失；消融实验", False)]))

    body.append(para("一、研究背景与难点创新", style="Heading1"))
    body.append(para("1. 研究背景及意义", style="Heading2"))
    body.append(
        para(
            "风机叶片是风力发电机组中直接承受气动载荷的关键部件，其运行状态对机组安全性和发电效率具有重要影响。"
            "在实际运维中，叶片表面缺陷通常通过无人机或巡检设备采集图像后进行人工复核。该方式虽然具有较强的可解释性，"
            "但面对大规模风场时存在效率低、主观性强和漏检风险高等问题。利用深度学习目标检测方法实现叶片缺陷自动识别，"
            "能够为后续状态评估、检修决策和故障预警提供更稳定的数据基础。"
        )
    )
    body.append(
        para(
            "与通用目标检测相比，风机叶片缺陷检测对模型提出了更严格的要求。一方面，裂纹和腐蚀等缺陷在整幅巡检图像中的面积占比较小，"
            "经过输入缩放和深层下采样后容易丢失局部纹理；另一方面，不同缺陷类别之间外观边界并不总是清晰，"
            "复杂背景中的脏污、阴影和叶片纹理也可能诱发误检。因此，模型不仅需要获得较高的 mAP，还需要在召回率、困难类别和背景误检方面保持平衡。"
        )
    )
    body.append(para("2. 研究难点", style="Heading2"))
    body.append(table([
        ["序号", "难点", "实际表现", "对模型的影响"],
        ["D1", "小尺度缺陷特征易退化", "远距离裂纹、小腐蚀点和小孔洞在整图缩放后仅占少量像素", "深层下采样后目标纹理和边缘信息不足，容易造成漏检"],
        ["D2", "缺陷形态差异大", "裂纹呈细长结构，腐蚀和剥落呈不规则区域", "统一权重的框回归损失难以突出小面积和细长目标"],
        ["D3", "类别分布与学习难度不均衡", "crack、corrosion 等类别更易出现低置信度或漏检", "训练过程可能被易学类别主导，困难类别学习不足"],
        ["D4", "高分辨率图像推理存在尺度损失", "整图推理时小缺陷被压缩，局部细节弱化", "普通全图推理难以兼顾全局语义和局部细节"],
    ], widths=[900, 2100, 4300, 4300]))
    body.append(caption("表 1  风机叶片缺陷检测任务难点分析"))

    body.append(para("3. 本文创新点", style="Heading2"))
    body.append(
        para(
            "针对上述难点，本文并非单独增加某一检测模块，而是围绕小尺度缺陷检测构建由结构、损失、采样和推理组成的完整方法链路。"
            "主要创新点如下。"
        )
    )
    body.append(table([
        ["序号", "创新点", "技术内容", "对应难点"],
        ["1", "P2 高分辨率检测分支", "在检测头中引入 stride=4 的 P2 分支，使小缺陷在浅层特征图上获得更多响应单元", "D1"],
        ["2", "Shape-Aware 形态感知损失", "根据目标面积占比和长宽比构造框级权重，增强小面积和细长缺陷的定位约束", "D2"],
        ["3", "困难样本重加权训练", "提高 crack、corrosion 和小目标样本在训练过程中的采样概率，并结合类别权重调整分类损失", "D3"],
        ["4", "Blade-Slice 切片融合推理", "通过局部切片检测、坐标回映射和 NMS 融合缓解整图缩放造成的小目标细节损失", "D4"],
    ], widths=[900, 2300, 5900, 1200]))
    body.append(caption("表 2  MSF-YOLO 主要创新点"))

    body.append(para("二、MSF-YOLO 算法设计结构", style="Heading1"))
    body.append(para("1. 总体结构", style="Heading2"))
    body.append(
        para(
            "MSF-YOLO 的总体结构如图 1 所示。训练阶段，模型在 YOLO 检测框架上增加 P2 高分辨率检测头，"
            "并在损失函数中引入形态感知权重，同时通过困难样本重加权改变训练样本分布。推理阶段，"
            "在普通全图检测结果之外增加 Blade-Slice 局部切片检测结果，并将局部框回映射到原图坐标后进行融合。"
        )
    )
    body.append(image("rId10", 1, "fig_1_msf_yolo_framework.svg", 5_580_000, 3_310_000))
    body.append(caption("图 1  MSF-YOLO 总体结构框图"))
    body.append(formula("θ* = arg min_θ  E_{i~p}[ L_det(I_i, Y_i; θ, q) ]"))
    body.append(
        para(
            "式中，I_i 表示输入图像，Y_i 表示标注信息，θ 为模型参数，p_i 为困难样本重加权后得到的采样概率，"
            "q_i 为 Shape-Aware Loss 中由目标形态计算得到的框级权重。"
        )
    )
    body.append(para("2. P2 浅层高分辨率检测分支", style="Heading2"))
    body.append(
        para(
            "常规 YOLO 检测头多从 P3 层开始预测，其最小检测 stride 为 8。对于远距离小裂纹、小孔洞或小腐蚀区域，"
            "目标经过多次下采样后容易退化为极少数特征点。本文引入 P2 检测分支，将最小检测 stride 扩展至 4，"
            "使浅层纹理、边缘和局部结构能够直接参与检测预测。"
        )
    )
    body.append(formula("F_l ∈ R^{C_l × ceil(H/s_l) × ceil(W/s_l)}"))
    body.append(formula("s_P2=4,  s_P3=8,  s_P4=16,  s_P5=32"))
    body.append(formula("n_cell(b,l)=sqrt(w_b h_b)/s_l"))
    body.append(
        para(
            "由上式可知，在目标像素面积相同的情况下，检测层 stride 越小，目标在特征图上覆盖的单元数越多。"
            "因此，P2 分支能够从结构上缓解小缺陷在深层特征中被过度压缩的问题。"
        )
    )
    body.append(table([
        ["检测层", "stride", "1024 输入下特征图尺寸", "主要关注对象"],
        ["P2", "4", "256 × 256", "小裂纹、小腐蚀、小孔洞"],
        ["P3", "8", "128 × 128", "小目标和中等目标"],
        ["P4", "16", "64 × 64", "中等目标"],
        ["P5", "32", "32 × 32", "大目标和强语义目标"],
    ], widths=[1400, 1200, 2600, 4300]))
    body.append(caption("表 3  P2-P5 多尺度检测层说明"))

    body.append(para("3. Shape-Aware 形态感知损失", style="Heading2"))
    body.append(
        para(
            "风机叶片缺陷并非规则目标。裂纹通常呈细长结构，腐蚀和孔洞可能面积较小且边界不规则。"
            "若所有正样本框采用相同损失权重，小面积和细长目标对总体损失的贡献容易被大面积目标稀释。"
            "为此，本文根据目标面积占比和长宽比构造形态权重，使模型在框回归阶段更加关注小面积和细长缺陷。"
        )
    )
    body.append(image("rId11", 2, "fig_2_shape_aware_loss.svg", 5_580_000, 2_834_000))
    body.append(caption("图 2  Shape-Aware Loss 权重计算流程"))
    body.append(formula("A_i = w_i h_i / (W H),     R_i = max(w_i/h_i, h_i/w_i)"))
    body.append(formula("S_i = clip((a_0 - A_i)/a_0, 0, 1)"))
    body.append(formula("E_i = clip((R_i - r_0)/r_0, 0, 1)"))
    body.append(formula("q_i = clip(1 + αS_i + βE_i, 1, q_max)"))
    body.append(table([
        ["参数", "数值", "含义"],
        ["a_0", "0.012", "小目标面积阈值"],
        ["r_0", "3.0", "细长目标长宽比阈值"],
        ["α", "0.35", "小面积目标增益"],
        ["β", "0.25", "细长目标增益"],
        ["q_max", "1.7", "最大形态权重，避免梯度过大"],
    ], widths=[1800, 1600, 6000]))
    body.append(caption("表 4  Shape-Aware Loss 参数设置"))
    body.append(formula("L_box^shape = Σ_{i∈Ω+} (1-CIoU_i)t_iq_i / Σ_{i∈Ω+} t_i"))
    body.append(formula("L_dfl^shape = Σ_{i∈Ω+} DFL_i t_iq_i / Σ_{i∈Ω+} t_i"))
    body.append(
        para(
            "当 q_i 大于 1 时，该目标的定位误差将在反向传播中获得更高权重。需要说明的是，"
            "Shape-Aware Loss 属于检测框级别的形态先验，并不引入像素级分割标注，因此能够在保持检测任务标注形式的前提下增强形态约束。"
        )
    )

    body.append(para("4. 困难样本重加权训练机制", style="Heading2"))
    body.append(
        para(
            "在训练集中，部分类别虽然样本数量不少，但受尺度、背景和外观边界影响，实际学习难度更高。"
            "本文对包含 crack、corrosion 以及小目标的图像赋予更高采样权重，使困难样本在训练过程中获得更多更新机会。"
        )
    )
    body.append(formula("w_i = clip(1.0 × 3.0^{I[corrosion]} × 2.0^{I[crack]} × 1.5^{I[small]}, 0.05, 6.0)"))
    body.append(formula("p_i = w_i / Σ_{j=1}^{N} w_j,       E[m_i] = Np_i"))
    body.append(table([
        ["样本类型", "触发条件", "权重系数"],
        ["普通样本", "未触发困难条件", "1.0"],
        ["裂纹样本", "包含 crack", "2.0"],
        ["腐蚀样本", "包含 corrosion", "3.0"],
        ["小目标样本", "crack/corrosion 且面积占比不高于 0.012", "1.5"],
        ["上限截断", "多个条件叠加后超过上限", "6.0"],
    ], widths=[2200, 5200, 1600]))
    body.append(caption("表 5  困难样本采样权重设置"))
    body.append(
        para(
            "与采样权重不同，类别权重作用于分类损失项，用于调整不同类别 BCE 损失的相对贡献。"
            "因此，困难样本重加权改变的是样本被抽取的概率，类别权重改变的是分类误差的损失比例，两者分别从数据分布和损失函数两个层面发挥作用。"
        )
    )

    body.append(para("5. Blade-Slice 重叠切片融合推理", style="Heading2"))
    body.append(
        para(
            "Blade-Slice 是推理阶段的尺度补偿策略，不改变模型结构，也不写入权重。"
            "该方法在全图检测之外增加局部窗口检测，将切片检测框回映射至原图坐标后，与全图检测结果进行 NMS 融合。"
            "其作用在于使小缺陷在局部窗口中获得更大的相对尺度。"
        )
    )
    body.append(image("rId12", 3, "fig_3_blade_slice_pipeline.svg", 5_580_000, 2_834_000))
    body.append(caption("图 3  Blade-Slice 重叠切片融合推理流程"))
    body.append(formula("d = floor(L(1-ρ)),     L=640,  ρ=0.25,  d=480"))
    body.append(formula("C_{u,v}=I[u:u+L, v:v+L]"))
    body.append(formula("b_global=(x_1+u, y_1+v, x_2+u, y_2+v, c, p)"))
    body.append(formula("B_fused = NMS(B_full ∪ B_slice, τ)"))

    body.append(para("三、实验方案与评价指标", style="Heading1"))
    body.append(para("1. 消融实验设计", style="Heading2"))
    body.append(
        para(
            "为验证各模块的独立贡献，本文采用逐项加入的方式设计消融实验。"
            "其中，P2、Shape-Aware Loss 和困难样本重加权属于训练链路，Blade-Slice 属于推理链路，"
            "因此将 Blade-Slice 单独置于最后一组实验中进行比较。"
        )
    )
    body.append(table([
        ["实验编号", "模型设置", "P2", "Shape Loss", "Hard Sampling", "Blade-Slice", "实验目的"],
        ["A0", "YOLO baseline", "—", "—", "—", "—", "建立普通 YOLO 检测基线"],
        ["A1", "YOLO + P2", "√", "—", "—", "—", "验证小目标检测头的结构贡献"],
        ["A2", "YOLO + P2 + Shape Loss", "√", "√", "—", "—", "验证形态感知损失对小面积和细长目标的贡献"],
        ["A3", "YOLO + P2 + Shape Loss + Hard Sampling", "√", "√", "√", "—", "验证困难样本重加权对 crack/corrosion 的作用"],
        ["A4", "YOLO + P2 + Shape Loss + Hard Sampling + Blade-Slice", "√", "√", "√", "√", "验证推理阶段尺度补偿收益"],
    ], widths=[1200, 3000, 700, 1200, 1400, 1300, 3000]))
    body.append(caption("表 6  MSF-YOLO 消融实验设计"))

    body.append(para("2. 评价指标", style="Heading2"))
    body.append(formula("Precision = TP/(TP+FP),       Recall = TP/(TP+FN)"))
    body.append(formula("mAP50-95 = (1/10) Σ_{τ∈{0.50,0.55,...,0.95}} mAP_τ"))
    body.append(formula("FP_bg = Σ_{I_i∈D_empty} |{b∈f_θ(I_i): conf(b)≥η}|"))
    body.append(table([
        ["指标", "含义", "论文中关注原因"],
        ["Precision", "预测为缺陷的目标中真正缺陷所占比例", "反映误检控制能力"],
        ["Recall", "真实缺陷中被模型检出的比例", "反映漏检风险"],
        ["mAP50", "IoU=0.50 下的平均精度", "反映较宽松定位条件下的检测能力"],
        ["mAP50-95", "多 IoU 阈值下的平均 mAP", "反映更严格定位条件下的综合性能"],
        ["FP_bg", "空标注背景图中的误检框数量", "反映工程部署中的背景误报风险"],
    ], widths=[1800, 4200, 4200]))
    body.append(caption("表 7  评价指标说明"))

    body.append(para("3. 结果记录表", style="Heading2"))
    body.append(
        para(
            "完整消融实验结束后，应按照表 8 和表 9 统一填写结果。所有模型应使用相同数据划分、输入尺寸、置信度阈值、IoU 阈值和评估脚本，"
            "避免由于评估口径不同造成结论偏差。"
        )
    )
    body.append(table([
        ["实验编号", "Precision", "Recall", "mAP50", "mAP50-95", "FP_bg", "参数量/M", "FLOPs/G"],
        ["A0", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A1", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A2", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A3", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A4", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[1300, 1400, 1400, 1400, 1500, 1200, 1300, 1300]))
    body.append(caption("表 8  整体消融实验结果记录表"))
    body.append(table([
        ["实验编号", "类别", "Precision", "Recall", "mAP50", "mAP50-95", "Missed", "FP"],
        ["A0-A4", "crack", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A0-A4", "hole", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A0-A4", "spalling", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["A0-A4", "corrosion", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[1400, 1500, 1400, 1400, 1300, 1500, 1200, 1100]))
    body.append(caption("表 9  分类别检测结果记录表"))

    body.append(para("四、阶段性实验记录与分析边界", style="Heading1"))
    body.append(
        para(
            "在完整消融实验完成前，已有阶段性实验可用于说明方法链路的探索过程，但不宜直接作为最终性能结论。"
            "从现有结果看，P2、困难样本重加权和 Blade-Slice 均已完成独立验证，但部分配置尚未超过 V4 稳定基线，"
            "说明后续需要通过统一消融实验进一步区分模块贡献与训练策略影响。"
        )
    )
    body.append(table([
        ["实验方向", "主要观察", "论文写作边界"],
        ["YOLO-P2 初始实验", "test_clean mAP50-95 和 Recall 低于 V4，背景误检增加", "可写为结构方向已验证链路可行，不可写为性能已提升"],
        ["困难样本重加权", "部分方案接近 V4，但 corrosion 未形成稳定提升", "可写为困难样本策略需要控制误检风险"],
        ["Blade-Slice 默认参数", "小幅增加召回但误检明显增加", "可写为推理尺度补偿方向需进一步门控和分场景验证"],
        ["V6 全创新集合", "完整链路已实现，但当前指标尚未超过 V4", "可写为方法探索与实现验证，最终结论等待消融实验"],
    ], widths=[2600, 4300, 4300]))
    body.append(caption("表 10  阶段性实验观察与论文表述边界"))

    body.append(para("五、总结与后续工作", style="Heading1"))
    body.append(
        para(
            "本文围绕风机叶片缺陷检测中的小尺度、细长形态、困难类别和高分辨率推理问题，"
            "构建了 MSF-YOLO 方法框架。该框架通过 P2 高分辨率检测头增强小目标特征表达，"
            "通过 Shape-Aware Loss 提高小面积和细长缺陷的定位学习强度，通过困难样本重加权改善训练样本分布，"
            "并通过 Blade-Slice 在推理阶段补偿局部尺度信息。"
        )
    )
    body.append(
        para(
            "后续工作将围绕统一消融实验展开，重点比较 A0 至 A4 各组模型在整体指标、分类别指标、背景误检和典型可视化样本上的变化。"
            "只有当实验结果同时支持精度、召回和误检控制时，才能进一步将 MSF-YOLO 写成相对于基线模型的有效提升方法；"
            "若部分模块仅在特定类别或特定场景中有效，则应在论文中如实限定其适用范围。"
        )
    )

    sect = """
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
      <w:cols w:space="720"/>
      <w:docGrid w:linePitch="312"/>
    </w:sectPr>
    """
    body.append(sect)

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS['w']}" xmlns:r="{NS['r']}" xmlns:wp="{NS['wp']}" xmlns:a="{NS['a']}" xmlns:pic="{NS['pic']}">
  <w:body>
    {''.join(body)}
  </w:body>
</w:document>
"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="svg" ContentType="image/svg+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
</Types>
"""


def package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  <Relationship Id="rId10" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig_1_msf_yolo_framework.svg"/>
  <Relationship Id="rId11" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig_2_shape_aware_loss.svg"/>
  <Relationship Id="rId12" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig_3_blade_slice_pipeline.svg"/>
</Relationships>
"""


def settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{NS['w']}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="420"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
</w:settings>
"""


def build_docx() -> None:
    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    required = [
        FIG_DIR / "fig_1_msf_yolo_framework.svg",
        FIG_DIR / "fig_2_shape_aware_loss.svg",
        FIG_DIR / "fig_3_blade_slice_pipeline.svg",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing figure assets: " + ", ".join(missing))

    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml())
        z.writestr("_rels/.rels", package_rels_xml())
        z.writestr("word/document.xml", document_xml())
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/settings.xml", settings_xml())
        z.writestr("word/_rels/document.xml.rels", document_rels_xml())
        for fig in required:
            z.write(fig, f"word/media/{fig.name}")

    print(DOCX_PATH)


if __name__ == "__main__":
    build_docx()
