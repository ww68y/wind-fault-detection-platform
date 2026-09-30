from __future__ import annotations

import zipfile
from pathlib import Path

from generate_msf_yolo_award_docx import (
    NS,
    caption,
    content_types_xml,
    formula,
    image,
    package_rels_xml,
    para,
    settings_xml,
    styles_xml,
    table,
)


OUT_DIR = Path("paper_assets_msf_yolo")
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = OUT_DIR / "MSF-YOLO_论文方法与实验设计_彩色结构图增强版.docx"


FIGURES = [
    ("rId30", "fig_1_msf_yolo_framework.svg"),
    ("rId31", "fig_color_1_msf_yolo_network_detail.svg"),
    ("rId32", "fig_color_2_p2_head_structure.svg"),
    ("rId33", "fig_color_3_shape_aware_formula.svg"),
    ("rId34", "fig_2_shape_aware_loss.svg"),
    ("rId35", "fig_color_4_blade_slice_formula.svg"),
    ("rId36", "fig_3_blade_slice_pipeline.svg"),
]


def document_rels_xml() -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
    ]
    for rid, name in FIGURES:
        rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def document_xml() -> str:
    body: list[str] = []
    body.append(para("MSF-YOLO：面向风机叶片缺陷检测的多尺度形态感知算法", style="Title"))
    body.append(para("彩色结构图增强版", style="Subtitle"))

    body.append(para("摘要", style="Heading1"))
    body.append(
        para(
            "为解决风机叶片巡检图像中小尺度缺陷漏检、细长缺陷定位不稳定以及困难类别学习不足等问题，"
            "本文提出一种多尺度形态感知检测算法 MSF-YOLO。首先，在 YOLO 检测框架中引入 P2 高分辨率检测分支，"
            "增强小裂纹、小腐蚀和小孔洞在浅层特征图上的表达能力；其次，设计 Shape-Aware 形态感知损失，"
            "根据目标框面积占比和长宽比动态调整定位损失权重；最后，结合困难样本重加权训练和 Blade-Slice 重叠切片融合推理，"
            "从训练分布和推理尺度两个层面提升模型对复杂叶片缺陷的检测稳定性。"
        )
    )
    body.append(para("关键词：风机叶片；缺陷检测；YOLO；小目标检测；Shape-Aware Loss；Blade-Slice"))

    body.append(para("一、研究背景与难点创新", style="Heading1"))
    body.append(
        para(
            "风机叶片长期运行于复杂自然环境中，表面容易出现裂纹、孔洞、腐蚀和剥落等缺陷。"
            "这些缺陷在无人机巡检图像中通常面积较小、形态不规则，并且容易受到背景纹理、光照和叶片表面污渍干扰。"
            "因此，风机叶片缺陷检测不仅要求模型具有较高检测精度，也要求模型在小目标召回、困难类别识别和背景误检控制之间保持平衡。"
        )
    )
    body.append(table([
        ["序号", "难点", "具体表现", "对应改进"],
        ["1", "小尺度缺陷易退化", "远距离裂纹、小孔洞和小腐蚀在整图缩放后仅占少量像素", "P2 高分辨率检测头"],
        ["2", "缺陷形态差异大", "裂纹细长，腐蚀和剥落边界不规则", "Shape-Aware 形态感知损失"],
        ["3", "困难类别学习不足", "crack、corrosion 更容易低置信度或漏检", "困难样本重加权与类别权重"],
        ["4", "整图推理细节损失", "高分辨率图像缩放后局部细节弱化", "Blade-Slice 切片融合推理"],
    ], widths=[900, 2200, 4800, 3000]))
    body.append(caption("表 1  风机叶片缺陷检测难点与对应改进"))

    body.append(para("二、MSF-YOLO 网络结构", style="Heading1"))
    body.append(
        para(
            "MSF-YOLO 的总体技术路线如图 1 所示。训练阶段，P2 检测头、Shape-Aware Loss、困难样本重加权和类别权重共同作用于模型学习过程；"
            "推理阶段，Blade-Slice 作为独立尺度补偿策略参与检测结果融合。"
        )
    )
    body.append(image("rId30", 30, "fig_1_msf_yolo_framework.svg", 5_760_000, 3_420_000))
    body.append(caption("图 1  MSF-YOLO 总体框架"))
    body.append(
        para(
            "为了更清晰地说明本文对 YOLO 检测结构的改进，图 2 给出了 MSF-YOLO 的网络结构模型。"
            "图中蓝色模块表示常规卷积结构，黄色模块表示本文强调的小目标检测相关结构，绿色和青色模块表示特征聚合与融合，"
            "橙色模块表示检测头，紫色模块表示形态感知损失。"
        )
    )
    body.append(image("rId31", 31, "fig_color_1_msf_yolo_network_detail.svg", 5_900_000, 3_870_000))
    body.append(caption("图 2  MSF-YOLO 改进网络结构模型"))
    body.append(formula("θ* = arg min_θ E_{i~p}[L_det(I_i,Y_i;θ,q)]"))
    body.append(
        para(
            "其中，I_i 表示输入图像，Y_i 表示标注信息，p_i 表示困难样本重加权后的采样概率，"
            "q_i 表示由目标面积和长宽比计算得到的形态权重。"
        )
    )

    body.append(para("三、算法改进模块", style="Heading1"))
    body.append(para("1. P2 小目标检测头", style="Heading2"))
    body.append(
        para(
            "常规 YOLO 检测头通常以 P3、P4、P5 为主要预测层，最小检测 stride 为 8。"
            "对于风机叶片中的远距离裂纹和小面积腐蚀，目标在深层特征图中容易被压缩为极少数响应单元。"
            "因此，本文增加 P2 检测头，使模型在 stride=4 的浅层高分辨率特征图上直接进行小目标预测。"
        )
    )
    body.append(image("rId32", 32, "fig_color_2_p2_head_structure.svg", 5_760_000, 2_300_000))
    body.append(caption("图 3  P2 小目标检测头结构"))
    body.append(formula("F_l ∈ R^{C_l × ceil(H/s_l) × ceil(W/s_l)}"))
    body.append(formula("s_P2=4, s_P3=8, s_P4=16, s_P5=32"))
    body.append(formula("n_cell(b,l)=sqrt(w_bh_b)/s_l"))
    body.append(table([
        ["检测层", "stride", "1024 输入下特征图", "主要关注对象"],
        ["P2", "4", "256 × 256", "小裂纹、小腐蚀、小孔洞"],
        ["P3", "8", "128 × 128", "小目标和中等目标"],
        ["P4", "16", "64 × 64", "中等目标"],
        ["P5", "32", "32 × 32", "大目标和强语义目标"],
    ], widths=[1500, 1200, 3000, 4600]))
    body.append(caption("表 2  MSF-YOLO 多尺度检测层设置"))

    body.append(para("2. Shape-Aware 形态感知损失", style="Heading2"))
    body.append(
        para(
            "风机叶片缺陷的几何形态差异明显。裂纹通常具有较大的长宽比，而小腐蚀和小孔洞具有较低的面积占比。"
            "为使这些目标在训练中获得更高定位优先级，本文将目标框面积占比和长宽比转化为损失权重。"
        )
    )
    body.append(image("rId33", 33, "fig_color_3_shape_aware_formula.svg", 5_760_000, 2_770_000))
    body.append(caption("图 4  Shape-Aware Loss 公式结构"))
    body.append(image("rId34", 34, "fig_2_shape_aware_loss.svg", 5_760_000, 2_930_000))
    body.append(caption("图 5  Shape-Aware Loss 权重计算流程"))
    body.append(formula("A_i = w_ih_i/(WH),   R_i=max(w_i/h_i,h_i/w_i)"))
    body.append(formula("S_i=clip((a_0-A_i)/a_0,0,1),   E_i=clip((R_i-r_0)/r_0,0,1)"))
    body.append(formula("q_i=clip(1+αS_i+βE_i,1,q_max)"))
    body.append(formula("L_box^shape = Σ(1-CIoU_i)t_iq_i / Σt_i"))
    body.append(formula("L_dfl^shape = ΣDFL_it_iq_i / Σt_i"))
    body.append(table([
        ["参数", "数值", "含义"],
        ["a_0", "0.012", "小目标面积阈值"],
        ["r_0", "3.0", "细长目标长宽比阈值"],
        ["α", "0.35", "小面积目标增益"],
        ["β", "0.25", "细长目标增益"],
        ["q_max", "1.7", "最大形态权重"],
    ], widths=[1800, 1600, 6200]))
    body.append(caption("表 3  Shape-Aware Loss 参数设置"))

    body.append(para("3. 困难样本重加权训练机制", style="Heading2"))
    body.append(
        para(
            "在训练过程中，本文对包含 crack、corrosion 和小目标的图像提高采样权重，使困难样本在同等训练轮数内获得更多优化机会。"
            "该机制改变的是样本被抽取的概率，而类别权重则改变分类损失中不同类别的贡献，两者分别作用于训练分布和损失函数。"
        )
    )
    body.append(formula("w_i=clip(1.0·3.0^{I[corrosion]}·2.0^{I[crack]}·1.5^{I[small]},0.05,6.0)"))
    body.append(formula("p_i=w_i/Σw_j,   E[m_i]=Np_i"))
    body.append(table([
        ["样本类型", "触发条件", "权重系数"],
        ["普通样本", "无困难条件", "1.0"],
        ["裂纹样本", "包含 crack", "2.0"],
        ["腐蚀样本", "包含 corrosion", "3.0"],
        ["小目标样本", "crack/corrosion 且面积占比较小", "1.5"],
        ["上限截断", "多条件叠加超过上限", "6.0"],
    ], widths=[2300, 5200, 1800]))
    body.append(caption("表 4  困难样本采样权重设置"))

    body.append(para("4. Blade-Slice 切片融合推理", style="Heading2"))
    body.append(
        para(
            "Blade-Slice 是推理阶段的尺度补偿策略，不改变训练权重。其思想是在全图推理之外增加局部切片推理，"
            "将局部检测框回映射至原图坐标后，与全图检测结果进行 NMS 融合。"
        )
    )
    body.append(image("rId35", 35, "fig_color_4_blade_slice_formula.svg", 5_760_000, 2_650_000))
    body.append(caption("图 6  Blade-Slice 融合推理公式结构"))
    body.append(image("rId36", 36, "fig_3_blade_slice_pipeline.svg", 5_760_000, 2_930_000))
    body.append(caption("图 7  Blade-Slice 重叠切片融合推理流程"))
    body.append(formula("d=floor(L(1-ρ)), L=640, ρ=0.25, d=480"))
    body.append(formula("C_{u,v}=I[u:u+L,v:v+L]"))
    body.append(formula("b_global=(x_1+u,y_1+v,x_2+u,y_2+v,c,p)"))
    body.append(formula("B_fused=NMS(B_full∪B_slice,τ)"))

    body.append(para("四、实验设计", style="Heading1"))
    body.append(
        para(
            "为验证各模块有效性，本文设计逐项消融实验。P2、Shape-Aware Loss 和困难样本重加权属于训练链路，"
            "Blade-Slice 属于推理链路，因此在实验中作为最后一项单独加入。"
        )
    )
    body.append(table([
        ["编号", "模型设置", "P2", "Shape Loss", "Hard Sampling", "Blade-Slice", "Precision", "Recall", "mAP50-95", "FP_bg"],
        ["A0", "YOLO baseline", "—", "—", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A1", "YOLO + P2", "√", "—", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A2", "YOLO + P2 + Shape", "√", "√", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A3", "YOLO + P2 + Shape + Hard", "√", "√", "√", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A4", "YOLO + P2 + Shape + Hard + Slice", "√", "√", "√", "√", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[800, 3100, 700, 1300, 1500, 1300, 1200, 1200, 1400, 1000]))
    body.append(caption("表 5  MSF-YOLO 消融实验设计"))
    body.append(formula("Precision=TP/(TP+FP),   Recall=TP/(TP+FN)"))
    body.append(formula("mAP50-95=(1/10)Σ_{τ=0.50:0.05:0.95}mAP_τ"))
    body.append(formula("FP_bg=Σ_{I_i∈D_empty}|{b∈f_θ(I_i):conf(b)≥η}|"))

    body.append(para("五、总结", style="Heading1"))
    body.append(
        para(
            "本文围绕风机叶片缺陷检测中的小尺度、细长形态、困难类别和高分辨率推理问题，构建了 MSF-YOLO 算法。"
            "与普通 YOLO 检测框架相比，MSF-YOLO 在结构上增加 P2 小目标检测头，在损失函数上引入 Shape-Aware 形态权重，"
            "在训练过程中使用困难样本重加权，并在推理阶段采用 Blade-Slice 切片融合。"
            "后续完整消融实验完成后，将进一步根据整体指标、分类别指标、背景误检和典型可视化结果确定最终结论强度。"
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


def document_rels_xml() -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
    ]
    for rid, name in FIGURES:
        rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def build_docx() -> None:
    required = [FIG_DIR / name for _, name in FIGURES]
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
