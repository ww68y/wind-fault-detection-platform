from __future__ import annotations

import zipfile
from pathlib import Path

from generate_msf_yolo_award_docx import (
    NS,
    caption,
    content_types_xml,
    esc,
    formula,
    image,
    package_rels_xml,
    para,
    settings_xml,
    text_run,
)


OUT_DIR = Path("paper_assets_msf_yolo")
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = OUT_DIR / "MSF-YOLO_论文方法与实验设计_FEB参考风格改进版.docx"


def table_gray(rows: list[list[object]], widths: list[int] | None = None) -> str:
    def cell(value: object, header: bool = False, width: int | None = None) -> str:
        tcpr = []
        if width is not None:
            tcpr.append(f'<w:tcW w:w="{width}" w:type="dxa"/>')
        if header:
            tcpr.append('<w:shd w:fill="EDEDED"/>')
        tcpr.append('<w:vAlign w:val="center"/>')
        ppr = '<w:pPr><w:jc w:val="center"/></w:pPr>' if header else ""
        run = text_run(value, bold=header)
        return f"<w:tc><w:tcPr>{''.join(tcpr)}</w:tcPr><w:p>{ppr}{run}</w:p></w:tc>"

    border = """
    <w:tblPr>
      <w:tblW w:w="5000" w:type="pct"/>
      <w:tblBorders>
        <w:top w:val="single" w:sz="10" w:space="0" w:color="000000"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>
        <w:bottom w:val="single" w:sz="10" w:space="0" w:color="000000"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="70" w:type="dxa"/>
        <w:left w:w="90" w:type="dxa"/>
        <w:bottom w:w="70" w:type="dxa"/>
        <w:right w:w="90" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>
    """
    body = []
    for ridx, row in enumerate(rows):
        body.append(
            "<w:tr>"
            + "".join(
                cell(value, header=(ridx == 0), width=(widths[cidx] if widths and cidx < len(widths) else None))
                for cidx, value in enumerate(row)
            )
            + "</w:tr>"
        )
    return f"<w:tbl>{border}{''.join(body)}</w:tbl>"


def styles_xml_feb() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS['w']}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
        <w:sz w:val="20"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="330" w:lineRule="auto" w:before="0" w:after="80"/>
        <w:jc w:val="both"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="160" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:after="100"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="220" w:after="100"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="23"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="150" w:after="70"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:b/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="100"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Formula">
    <w:name w:val="Formula"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="70" w:after="70"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="Cambria Math"/><w:sz w:val="20"/></w:rPr>
  </w:style>
</w:styles>
"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  <Relationship Id="rId20" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig_feb_1_msf_yolo_network.svg"/>
  <Relationship Id="rId21" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig_feb_2_msf_yolo_modules.svg"/>
  <Relationship Id="rId22" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig_feb_3_experiment_flow.svg"/>
</Relationships>
"""


def document_xml() -> str:
    body: list[str] = []
    body.append(para("MSF-YOLO：基于改进 YOLO 的风机叶片缺陷检测算法", style="Title"))
    body.append(para("MSF-YOLO: A Wind Turbine Blade Defect Detection Algorithm Based on Improved YOLO", style="Subtitle"))

    body.append(para("摘要", style="Heading1"))
    body.append(
        para(
            "为解决风机叶片巡检图像中小尺度缺陷漏检、细长缺陷定位不稳定以及困难类别样本学习不足等问题，"
            "本文提出一种基于改进 YOLO 的风机叶片缺陷检测算法 MSF-YOLO。首先，在检测头中引入 P2 浅层高分辨率分支，"
            "使小裂纹、小孔洞和小腐蚀区域在特征图上获得更多有效响应。其次，设计 Shape-Aware 形态感知损失，"
            "根据缺陷框面积占比和长宽比对边界框回归损失进行加权，以增强小面积和细长目标的定位约束。最后，"
            "结合困难样本重加权训练和 Blade-Slice 重叠切片融合推理，从训练分布和推理尺度两个方面改善复杂场景下的检测稳定性。"
            "后续实验将通过消融实验、分类别指标和可视化结果验证各改进模块的有效性。"
        )
    )
    body.append(para("关键词：风机叶片；缺陷检测；YOLO；小目标检测；形态感知损失；切片融合"))

    body.append(para("I. 引言", style="Heading1"))
    body.append(
        para(
            "风机叶片长期暴露在复杂自然环境中，容易受到疲劳载荷、雨蚀、沙尘和外部冲击等因素影响，"
            "产生裂纹、孔洞、腐蚀和表面剥落等缺陷。这些缺陷若不能及时发现，可能进一步扩展并影响机组安全运行。"
            "传统巡检方式主要依赖人工查看无人机或巡检设备采集的图像，存在工作量大、主观性强和漏检风险高等问题。"
        )
    )
    body.append(
        para(
            "近年来，YOLO 系列目标检测算法因速度快、端到端部署方便，被广泛用于工业缺陷检测任务。"
            "然而，风机叶片缺陷与常规目标存在明显差异：裂纹通常呈细长形态，腐蚀和孔洞面积较小，"
            "高分辨率巡检图像在缩放输入后会造成局部细节损失。针对上述问题，本文在 YOLO 检测框架基础上进行结构、损失和推理策略改进，"
            "形成 MSF-YOLO 检测方法。"
        )
    )

    body.append(para("II. MSF-YOLO 网络模型", style="Heading1"))
    body.append(
        para(
            "考虑到检测精度、模型复杂度和工程部署需求，本文以轻量级 YOLO 检测框架为基础构建风机叶片缺陷检测模型。"
            "MSF-YOLO 的网络结构如图 1 所示。与普通检测头相比，本文增加 P2 检测分支，并将形态感知损失和困难样本采样机制引入训练过程。"
        )
    )
    body.append(image("rId20", 20, "fig_feb_1_msf_yolo_network.svg", 5_850_000, 2_377_000))
    body.append(caption("图 1  MSF-YOLO 网络结构"))
    body.append(formula("θ* = arg min_θ E_{i~p}[L_det(I_i, Y_i; θ, q)]"))
    body.append(
        para(
            "其中，I_i 为输入图像，Y_i 为标注信息，p_i 为困难样本重加权后的采样概率，q_i 为由目标形态计算得到的损失权重。"
        )
    )

    body.append(para("III. 算法改进", style="Heading1"))
    body.append(para("A. P2 小目标检测头", style="Heading2"))
    body.append(
        para(
            "普通 YOLO 检测头通常从 P3 层开始预测，最小检测步长为 8。对于远距离小缺陷，深层下采样容易造成纹理和边缘信息损失。"
            "本文引入 stride=4 的 P2 检测头，使小目标在特征图上覆盖更多网格单元。"
        )
    )
    body.append(formula("F_l ∈ R^{C_l × ceil(H/s_l) × ceil(W/s_l)}"))
    body.append(formula("n_cell(b,l)=sqrt(w_b h_b)/s_l"))
    body.append(caption("表 1  MSF-YOLO 多尺度检测层设置"))
    body.append(table_gray([
        ["检测层", "Stride", "1024 输入下特征图", "主要检测对象"],
        ["P2", "4", "256 × 256", "小裂纹、小腐蚀、小孔洞"],
        ["P3", "8", "128 × 128", "小目标和中等目标"],
        ["P4", "16", "64 × 64", "中等目标"],
        ["P5", "32", "32 × 32", "大目标和强语义目标"],
    ], widths=[1400, 1200, 2600, 4300]))

    body.append(para("B. Shape-Aware 形态感知损失", style="Heading2"))
    body.append(
        para(
            "裂纹、腐蚀和孔洞在几何形态上差异明显。为了避免小面积和细长缺陷在训练中被易定位目标稀释，"
            "本文根据目标框面积占比和长宽比构造形态权重。图 2 给出了各改进模块的关系。"
        )
    )
    body.append(image("rId21", 21, "fig_feb_2_msf_yolo_modules.svg", 5_850_000, 2_560_000))
    body.append(caption("图 2  MSF-YOLO 主要改进模块"))
    body.append(formula("A_i = w_i h_i/(WH),       R_i=max(w_i/h_i, h_i/w_i)"))
    body.append(formula("S_i=clip((a_0-A_i)/a_0,0,1),       E_i=clip((R_i-r_0)/r_0,0,1)"))
    body.append(formula("q_i=clip(1+αS_i+βE_i,1,q_max)"))
    body.append(formula("L_box^shape = Σ(1-CIoU_i)t_iq_i / Σt_i"))
    body.append(caption("表 2  Shape-Aware Loss 参数设置"))
    body.append(table_gray([
        ["参数", "数值", "说明"],
        ["a_0", "0.012", "小目标面积阈值"],
        ["r_0", "3.0", "细长目标长宽比阈值"],
        ["α", "0.35", "小面积目标增益"],
        ["β", "0.25", "细长目标增益"],
        ["q_max", "1.7", "最大权重截断"],
    ], widths=[1800, 1600, 6000]))

    body.append(para("C. 困难样本重加权训练", style="Heading2"))
    body.append(
        para(
            "在训练集中，crack 和 corrosion 更容易受到尺度、背景和外观边界影响。本文通过样本权重改变训练样本出现概率，"
            "使困难类别和小目标样本在训练过程中获得更多更新机会。"
        )
    )
    body.append(formula("w_i=clip(1.0×3.0^{I[corrosion]}×2.0^{I[crack]}×1.5^{I[small]},0.05,6.0)"))
    body.append(formula("p_i=w_i/Σw_j,       E[m_i]=Np_i"))
    body.append(caption("表 3  困难样本采样权重"))
    body.append(table_gray([
        ["样本类型", "触发条件", "权重系数"],
        ["普通样本", "无困难条件", "1.0"],
        ["裂纹样本", "包含 crack", "2.0"],
        ["腐蚀样本", "包含 corrosion", "3.0"],
        ["小目标样本", "crack/corrosion 且面积占比较小", "1.5"],
        ["上限截断", "多条件叠加超过上限", "6.0"],
    ], widths=[2300, 5000, 1700]))

    body.append(para("D. Blade-Slice 切片融合推理", style="Heading2"))
    body.append(
        para(
            "Blade-Slice 不改变模型权重，仅在推理阶段使用。该策略将原图划分为重叠局部窗口，对局部检测结果进行坐标回映射，"
            "再与全图预测框进行 NMS 融合，以缓解高分辨率图像缩放造成的小目标信息损失。"
        )
    )
    body.append(formula("d=floor(L(1-ρ)),       L=640, ρ=0.25, d=480"))
    body.append(formula("B_fused=NMS(B_full ∪ B_slice, τ)"))

    body.append(para("IV. 实验与结果设计", style="Heading1"))
    body.append(para("A. 实验环境", style="Heading2"))
    body.append(caption("表 4  实验环境与训练设置"))
    body.append(table_gray([
        ["项目", "设置"],
        ["操作系统", "Windows 平台"],
        ["深度学习框架", "PyTorch / Ultralytics YOLO"],
        ["输入尺寸", "1024 × 1024"],
        ["优化器", "AdamW"],
        ["主要评价集", "val、test_clean、背景空标注样本"],
        ["说明", "GPU 型号、CUDA 和 PyTorch 版本在最终实验完成后统一补充"],
    ], widths=[3000, 6500]))

    body.append(para("B. 数据集", style="Heading2"))
    body.append(
        para(
            "本文实验使用风机叶片缺陷数据集 wind_blade_defect_v05_conservative。数据集包含 crack、hole、spalling 和 corrosion 四类目标，"
            "并保留一定数量空标注图像用于背景误检评估。"
        )
    )
    body.append(caption("表 5  数据集划分"))
    body.append(table_gray([
        ["Split", "图像数量", "空标注图像", "用途"],
        ["train", "9136", "786", "模型训练"],
        ["val", "571", "17", "验证集评估"],
        ["test_clean", "543", "18", "干净测试集评估"],
    ], widths=[1800, 1800, 1800, 4500]))
    body.append(caption("表 6  类别映射关系"))
    body.append(table_gray([
        ["目标类别", "来源类别映射"],
        ["crack", "Crack / Scratch / Craze / Hide_craze"],
        ["corrosion", "Erosion / Corrosion"],
        ["spalling", "Damage / MechanicalDamage / Paintoff / Surface_injure / Chipping"],
        ["hole", "Puncture / Hole"],
    ], widths=[2200, 7200]))

    body.append(para("C. 评价指标", style="Heading2"))
    body.append(formula("Precision=TP/(TP+FP),       Recall=TP/(TP+FN)"))
    body.append(formula("mAP50-95=(1/10)Σ_{τ=0.50:0.05:0.95}mAP_τ"))
    body.append(
        para(
            "除常规检测指标外，本文单独统计空标注图像上的背景误检数 FP_bg，用于判断召回提升是否以误检增加为代价。"
        )
    )

    body.append(para("D. 消融实验", style="Heading2"))
    body.append(
        para(
            "为了验证各改进模块的有效性，本文设计五组消融实验。A0 为基础模型，A1 至 A4 按照 P2、Shape-Aware Loss、"
            "困难样本重加权和 Blade-Slice 的顺序逐步加入。实验流程如图 3 所示。"
        )
    )
    body.append(image("rId22", 22, "fig_feb_3_experiment_flow.svg", 5_850_000, 1_920_000))
    body.append(caption("图 3  消融实验与评价流程"))
    body.append(caption("表 7  MSF-YOLO 消融实验设计"))
    body.append(table_gray([
        ["模型", "P2", "Shape Loss", "Hard Sampling", "Blade-Slice", "Precision", "Recall", "mAP50-95", "FP_bg"],
        ["A0 baseline", "—", "—", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A1 +P2", "√", "—", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A2 +Shape", "√", "√", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A3 +Hard", "√", "√", "√", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A4 +Slice", "√", "√", "√", "√", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[1700, 800, 1300, 1500, 1300, 1300, 1300, 1500, 1100]))

    body.append(para("E. 对比实验与可视化分析", style="Heading2"))
    body.append(
        para(
            "在完整消融实验完成后，将 MSF-YOLO 与主流检测模型或项目已有稳定基线进行对比。"
            "对比指标包括 Precision、Recall、mAP50、mAP50-95、参数量、FLOPs 和背景误检数。"
            "同时选取典型小裂纹、小腐蚀、孔洞和剥落样本进行可视化比较，分析模型在小目标和困难场景中的检测差异。"
        )
    )
    body.append(caption("表 8  不同算法对比实验记录表"))
    body.append(table_gray([
        ["模型", "Precision", "Recall", "mAP50", "mAP50-95", "Params(M)", "FLOPs(G)", "FP_bg"],
        ["YOLO baseline", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["V4 stable baseline", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["MSF-YOLO", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[2200, 1300, 1300, 1300, 1500, 1400, 1400, 1100]))

    body.append(para("V. 结论", style="Heading1"))
    body.append(
        para(
            "本文针对风机叶片缺陷检测中的小目标漏检、细长缺陷定位不稳定和困难类别学习不足等问题，提出 MSF-YOLO 检测方法。"
            "该方法通过 P2 检测头增强浅层小目标特征表达，通过 Shape-Aware Loss 提高小面积和细长目标的定位权重，"
            "通过困难样本重加权改善训练样本分布，并通过 Blade-Slice 在推理阶段补偿局部尺度信息。"
            "后续将根据完整消融实验和对比实验结果，进一步分析各模块的实际贡献及其适用边界。"
        )
    )

    body.append(
        """
        <w:sectPr>
          <w:pgSz w:w="11906" w:h="16838"/>
          <w:pgMar w:top="1150" w:right="1150" w:bottom="1150" w:left="1150" w:header="720" w:footer="720" w:gutter="0"/>
          <w:cols w:space="720"/>
          <w:docGrid w:linePitch="312"/>
        </w:sectPr>
        """
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS['w']}" xmlns:r="{NS['r']}" xmlns:wp="{NS['wp']}" xmlns:a="{NS['a']}" xmlns:pic="{NS['pic']}">
  <w:body>{''.join(body)}</w:body>
</w:document>
"""


def build_docx() -> None:
    required = [
        FIG_DIR / "fig_feb_1_msf_yolo_network.svg",
        FIG_DIR / "fig_feb_2_msf_yolo_modules.svg",
        FIG_DIR / "fig_feb_3_experiment_flow.svg",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing figure assets: " + ", ".join(missing))

    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml())
        z.writestr("_rels/.rels", package_rels_xml())
        z.writestr("word/document.xml", document_xml())
        z.writestr("word/styles.xml", styles_xml_feb())
        z.writestr("word/settings.xml", settings_xml())
        z.writestr("word/_rels/document.xml.rels", document_rels_xml())
        for fig in required:
            z.write(fig, f"word/media/{fig.name}")
    print(DOCX_PATH)


if __name__ == "__main__":
    build_docx()
