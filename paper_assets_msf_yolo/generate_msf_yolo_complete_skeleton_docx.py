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
DOCX_PATH = OUT_DIR / "MSF-YOLO_完整论文骨架版_含研究现状数据集实验参考文献.docx"

FIGURES = [
    ("rId40", "fig_color_5_msf_yolo_system_pipeline.svg"),
    ("rId41", "fig_1_msf_yolo_framework.svg"),
    ("rId42", "fig_color_0_msf_yolo_precise_network.svg"),
    ("rId43", "fig_color_2_p2_head_structure.svg"),
    ("rId44", "fig_color_3_shape_aware_formula.svg"),
    ("rId45", "fig_color_4_blade_slice_formula.svg"),
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
    body.append(para("MSF-YOLO：面向风机叶片缺陷检测的多尺度形态感知算法研究", style="Title"))
    body.append(para("完整论文骨架版", style="Subtitle"))

    body.append(para("摘  要", style="Heading1"))
    body.append(
        para(
            "风机叶片在长期运行过程中易产生裂纹、孔洞、腐蚀和表面剥落等缺陷。针对巡检图像中缺陷目标尺度小、"
            "形态差异大、类别分布不均衡以及高分辨率图像整图推理细节损失等问题，本文提出一种多尺度形态感知检测算法 MSF-YOLO。"
            "首先，基于 YOLO 检测框架构建 P2-P5 四尺度检测头，引入 P2 浅层高分辨率分支以增强小目标特征表达；"
            "其次，设计 Shape-Aware 形态感知损失函数，根据目标框面积占比和长宽比动态调整框回归损失权重；"
            "再次，采用困难样本重加权和类别权重策略，提高 crack、corrosion 和小目标样本在训练过程中的学习强度；"
            "最后，使用 Blade-Slice 重叠切片融合推理缓解整图缩放造成的小缺陷信息损失。"
            "本文进一步构建了包含数据集构建、消融实验、对比实验、背景误检统计和可视化分析的完整实验方案，为后续形成风机叶片缺陷检测论文提供系统化技术框架。"
        )
    )
    body.append(para("关键词：风机叶片；缺陷检测；YOLO；小目标检测；形态感知损失；切片融合"))

    body.append(para("Abstract", style="Heading1"))
    body.append(
        para(
            "Wind turbine blades are prone to cracks, holes, corrosion and surface spalling during long-term service. "
            "To address small-scale defects, slender shapes, imbalanced categories and detail loss in high-resolution inference, "
            "this paper proposes a multi-scale shape-aware detector, named MSF-YOLO. The method introduces a P2 high-resolution detection branch, "
            "designs a Shape-Aware localization loss, adopts hard-sample reweighting, and performs Blade-Slice fusion inference. "
            "A complete experimental framework including ablation study, comparison experiments, category-wise evaluation, background false-positive analysis and visualization is designed for further validation."
        )
    )
    body.append(para("Keywords: wind turbine blade; defect detection; YOLO; small object detection; Shape-Aware Loss; Blade-Slice"))

    body.append(para("一、研究背景与难点创新", style="Heading1"))
    body.append(para("1. 研究背景与意义", style="Heading2"))
    body.append(
        para(
            "风力发电机组长期处于交变载荷和复杂自然环境中，叶片作为关键受力部件，其表面状态直接影响机组运行安全。"
            "裂纹、腐蚀、孔洞和剥落等缺陷在早期往往尺寸较小，如果不能及时发现，可能进一步扩展并带来更高维修成本。"
            "因此，构建稳定、自动化的风机叶片缺陷检测方法，对巡检效率提升和运维风险控制具有重要意义。"
        )
    )
    body.append(
        para(
            "传统人工巡检依赖经验判断，面对大规模风场时存在效率低、主观性强和漏检风险高等问题。"
            "随着无人机巡检和深度学习目标检测技术的发展，基于图像的自动缺陷检测逐渐成为风机叶片智能运维的重要方向。"
            "然而，风机叶片缺陷在视觉上具有目标小、背景干扰强和形态不规则等特点，直接使用通用目标检测模型仍难以满足工程应用要求。"
        )
    )
    body.append(para("2. 国内外研究现状", style="Heading2"))
    body.append(
        para(
            "工业表面缺陷检测通常经历了人工特征、传统机器学习和深度学习三个阶段。早期方法多依赖灰度、纹理、边缘和形态学特征，"
            "在规则背景下具有一定效果，但对光照变化、复杂纹理和跨场景泛化能力有限。深度学习方法通过卷积神经网络自动学习特征，"
            "在钢材、织物、电子元件和风机叶片等缺陷检测任务中得到广泛应用。"
        )
    )
    body.append(
        para(
            "两阶段检测器如 Faster R-CNN 具有较强定位能力，但模型复杂度和推理速度不利于实时检测。"
            "单阶段检测器如 YOLO 系列在速度和部署方面具有优势，适合工业巡检场景。近年来，研究者通常从轻量化网络、注意力机制、"
            "多尺度特征融合和损失函数改进等方面提升 YOLO 在工业缺陷检测中的表现。FEB-YOLOv8 通过改进 C2f、引入 EMA 和 BiFPN-light，"
            "以结构图、消融表和对比表形成了清晰的改进论证链路。本文借鉴此类论文的结构化表达方式，但围绕风机叶片小尺度异形缺陷，"
            "重点从 P2 小目标检测头、形态感知损失、困难样本训练和切片推理四个方面进行改进。"
        )
    )
    body.append(para("3. 研究难点与创新点", style="Heading2"))
    body.append(table([
        ["序号", "检测难点", "具体表现", "本文对应创新"],
        ["1", "小尺度缺陷易退化", "远距离裂纹、小孔洞和小腐蚀在整图缩放后仅占少量像素", "引入 P2 高分辨率检测分支"],
        ["2", "缺陷形态差异大", "裂纹细长，腐蚀和剥落边界不规则", "设计 Shape-Aware 形态感知损失"],
        ["3", "困难类别学习不足", "crack、corrosion 更容易低置信度或漏检", "构建困难样本重加权与类别权重机制"],
        ["4", "高分辨率推理细节损失", "普通整图推理难以兼顾全局语义与局部小缺陷细节", "采用 Blade-Slice 重叠切片融合推理"],
    ], widths=[800, 2200, 5100, 3400]))
    body.append(caption("表 1  风机叶片缺陷检测难点与本文创新点"))
    body.append(image("rId40", 40, "fig_color_5_msf_yolo_system_pipeline.svg", 5_760_000, 2_381_000))
    body.append(caption("图 1  MSF-YOLO 研究整体流程"))

    body.append(para("二、MSF-YOLO 算法设计结构", style="Heading1"))
    body.append(para("1. 总体结构", style="Heading2"))
    body.append(
        para(
            "MSF-YOLO 的总体框架如图 2 所示。训练阶段，模型在 YOLO 检测框架基础上引入 P2 高分辨率检测分支，"
            "并通过 Shape-Aware Loss、困难样本重加权和类别权重强化小尺度、细长形态和困难类别目标的学习。"
            "推理阶段，Blade-Slice 作为独立增强模块参与检测结果融合。"
        )
    )
    body.append(image("rId41", 41, "fig_1_msf_yolo_framework.svg", 5_760_000, 3_420_000))
    body.append(caption("图 2  MSF-YOLO 总体框架"))
    body.append(
        para(
            "为明确网络内部连接关系，本文根据 configs/yolov8n-p2-wind.yaml 绘制精确网络结构，如图 3 所示。"
            "Backbone 第 0 至 9 层提取 P2-P5 多尺度特征，Neck 第 10 至 27 层通过上采样、拼接和 C2f 模块完成自顶向下与自底向上的特征融合，"
            "最终 Detect 接收第 18、21、24 和 27 层输出，对应 P2、P3、P4 和 P5 四个检测尺度。"
        )
    )
    body.append(image("rId42", 42, "fig_color_0_msf_yolo_precise_network.svg", 5_820_000, 4_040_000))
    body.append(caption("图 3  MSF-YOLO 精确网络结构图"))
    body.append(formula("θ* = arg min_θ E_{i~p}[L_det(I_i,Y_i;θ,q)]"))

    body.append(para("2. P2 小目标检测头", style="Heading2"))
    body.append(
        para(
            "常规 YOLO 检测头通常使用 P3、P4 和 P5 三个输出层，最小检测步长为 8。"
            "本文增加 P2/4 输出层，使小尺度缺陷在浅层高分辨率特征图上获得更多响应单元。"
        )
    )
    body.append(image("rId43", 43, "fig_color_2_p2_head_structure.svg", 5_600_000, 2_236_000))
    body.append(caption("图 4  P2 小目标检测头结构"))
    body.append(formula("F_l ∈ R^{C_l × ceil(H/s_l) × ceil(W/s_l)}"))
    body.append(formula("n_cell(b,l)=sqrt(w_bh_b)/s_l"))
    body.append(table([
        ["检测层", "stride", "1024 输入下特征图", "主要关注对象"],
        ["P2", "4", "256 × 256", "小裂纹、小腐蚀、小孔洞"],
        ["P3", "8", "128 × 128", "小目标和中等目标"],
        ["P4", "16", "64 × 64", "中等目标"],
        ["P5", "32", "32 × 32", "大目标和强语义目标"],
    ], widths=[1500, 1200, 3000, 4600]))
    body.append(caption("表 2  多尺度检测层设置"))

    body.append(para("3. Shape-Aware 形态感知损失", style="Heading2"))
    body.append(
        para(
            "为提高模型对小面积和细长缺陷的定位能力，本文根据目标框面积占比和长宽比构造形态权重 q_i。"
            "当目标面积较小或长宽比较大时，其框回归误差在损失中获得更高权重。"
        )
    )
    body.append(image("rId44", 44, "fig_color_3_shape_aware_formula.svg", 5_600_000, 2_696_000))
    body.append(caption("图 5  Shape-Aware Loss 公式结构"))
    body.append(formula("A_i = w_ih_i/(WH),   R_i=max(w_i/h_i,h_i/w_i)"))
    body.append(formula("S_i=clip((a_0-A_i)/a_0,0,1),   E_i=clip((R_i-r_0)/r_0,0,1)"))
    body.append(formula("q_i=clip(1+αS_i+βE_i,1,q_max)"))
    body.append(formula("L_box^shape=Σ(1-CIoU_i)t_iq_i/Σt_i,   L_dfl^shape=ΣDFL_it_iq_i/Σt_i"))
    body.append(table([
        ["参数", "数值", "含义"],
        ["a_0", "0.012", "小目标面积阈值"],
        ["r_0", "3.0", "细长目标长宽比阈值"],
        ["α", "0.35", "小面积目标增益"],
        ["β", "0.25", "细长目标增益"],
        ["q_max", "1.7", "最大形态权重"],
    ], widths=[1800, 1600, 6200]))
    body.append(caption("表 3  Shape-Aware Loss 参数设置"))

    body.append(para("4. 困难样本重加权训练机制", style="Heading2"))
    body.append(
        para(
            "困难样本重加权用于改变训练样本的抽样概率。对包含 crack、corrosion 和小目标的图像赋予更高权重，"
            "使模型在训练过程中更频繁地学习易漏检样本。"
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

    body.append(para("5. Blade-Slice 重叠切片融合推理", style="Heading2"))
    body.append(
        para(
            "Blade-Slice 属于推理阶段策略，不写入模型权重。该方法将大图划分为重叠切片，对切片检测框进行坐标回映射后与全图检测结果融合，"
            "用于缓解整图缩放导致的小缺陷细节损失。"
        )
    )
    body.append(image("rId45", 45, "fig_color_4_blade_slice_formula.svg", 5_600_000, 2_572_000))
    body.append(caption("图 6  Blade-Slice 融合推理公式结构"))
    body.append(formula("d=floor(L(1-ρ)), L=640, ρ=0.25, d=480"))
    body.append(formula("B_fused=NMS(B_full∪B_slice,τ)"))

    body.append(para("三、数据集构建与预处理", style="Heading1"))
    body.append(para("1. 数据来源与类别映射", style="Heading2"))
    body.append(
        para(
            "本文使用 wind_blade_defect_v05_conservative 数据集。该数据集由多来源风机叶片缺陷图像整理得到，"
            "并将原始类别统一映射为 crack、hole、spalling 和 corrosion 四类目标。"
        )
    )
    body.append(table([
        ["目标类别", "来源类别映射"],
        ["crack", "Crack / Scratch / Craze / Hide_craze"],
        ["corrosion", "Erosion / Corrosion"],
        ["spalling", "Damage / MechanicalDamage / Paintoff / Surface_injure / Chipping"],
        ["hole", "Puncture / Hole"],
        ["negative", "Good / Dirt-only 等空标注背景图"],
    ], widths=[2200, 7800]))
    body.append(caption("表 5  缺陷类别映射关系"))
    body.append(para("2. 数据划分", style="Heading2"))
    body.append(table([
        ["Split", "图像数量", "空标注图像数量", "用途"],
        ["train", "9136", "786", "模型训练"],
        ["val", "571", "17", "验证集评估"],
        ["test_clean", "543", "18", "干净测试集评估"],
    ], widths=[1800, 1800, 2200, 4600]))
    body.append(caption("表 6  数据集划分统计"))
    body.append(
        para(
            "训练阶段使用 1024 输入尺寸，并结合 mosaic、multi-scale、copy-paste、scale 和水平翻转等增强方式提高模型对尺度变化和复杂背景的适应能力。"
            "空标注背景图用于评估模型在无缺陷场景中的误检风险。"
        )
    )

    body.append(para("四、实验结果与分析设计", style="Heading1"))
    body.append(para("1. 实验环境与训练参数", style="Heading2"))
    body.append(table([
        ["项目", "设置"],
        ["操作系统", "Windows 平台"],
        ["深度学习框架", "PyTorch / Ultralytics YOLO"],
        ["Python 版本", ">=3.10"],
        ["输入尺寸", "1024 × 1024"],
        ["优化器", "AdamW"],
        ["基础学习率", "根据实验组设置，典型值 0.00018-0.0005"],
        ["评价数据", "val、test_clean 和空标注背景图"],
        ["待补充项", "GPU 型号、CUDA 版本、PyTorch 版本、最终训练 epoch"],
    ], widths=[2800, 7200]))
    body.append(caption("表 7  实验环境与训练参数"))
    body.append(para("2. 评价指标", style="Heading2"))
    body.append(formula("Precision=TP/(TP+FP),   Recall=TP/(TP+FN)"))
    body.append(formula("mAP50-95=(1/10)Σ_{τ=0.50:0.05:0.95}mAP_τ"))
    body.append(formula("FP_bg=Σ_{I_i∈D_empty}|{b∈f_θ(I_i):conf(b)≥η}|"))
    body.append(
        para(
            "除 Precision、Recall、mAP50 和 mAP50-95 外，本文单独统计背景空标注图的误检数 FP_bg，"
            "用于判断召回提升是否以误检增加为代价。"
        )
    )
    body.append(para("3. 消融实验设计", style="Heading2"))
    body.append(table([
        ["编号", "模型设置", "P2", "Shape Loss", "Hard Sampling", "Blade-Slice", "Precision", "Recall", "mAP50-95", "FP_bg"],
        ["A0", "YOLO baseline", "—", "—", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A1", "YOLO + P2", "√", "—", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A2", "YOLO + P2 + Shape", "√", "√", "—", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A3", "YOLO + P2 + Shape + Hard", "√", "√", "√", "—", "待补充", "待补充", "待补充", "待补充"],
        ["A4", "YOLO + P2 + Shape + Hard + Slice", "√", "√", "√", "√", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[800, 3100, 700, 1300, 1500, 1300, 1200, 1200, 1400, 1000]))
    body.append(caption("表 8  MSF-YOLO 消融实验设计表"))
    body.append(para("4. 对比实验设计", style="Heading2"))
    body.append(table([
        ["模型", "Precision", "Recall", "mAP50", "mAP50-95", "Params(M)", "FLOPs(G)", "FP_bg"],
        ["YOLO baseline", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["V4 stable baseline", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
        ["MSF-YOLO", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充", "待补充"],
    ], widths=[2200, 1300, 1300, 1300, 1500, 1400, 1400, 1100]))
    body.append(caption("表 9  不同算法对比实验表"))
    body.append(para("5. 可视化与错误分析", style="Heading2"))
    body.append(
        para(
            "实验完成后，应选取典型裂纹、孔洞、剥落、腐蚀样本以及背景误检样本，分别展示 baseline 与 MSF-YOLO 的检测结果。"
            "可视化分析应同时包含成功案例和失败案例，重点说明小目标召回、低置信度样本和背景误检变化。"
        )
    )

    body.append(para("五、总结与展望", style="Heading1"))
    body.append(
        para(
            "本文构建了面向风机叶片缺陷检测的 MSF-YOLO 完整技术框架。该方法从网络结构、损失函数、训练采样和推理策略四个层面对小尺度异形缺陷检测进行改进。"
            "后续将根据完整消融结果，进一步判断各模块对 Precision、Recall、mAP50-95、分类别指标和背景误检的实际贡献。"
            "若实验表明部分模块只在特定类别或特定场景有效，论文结论将明确限定其适用范围。"
        )
    )

    body.append(para("参考文献框架", style="Heading1"))
    refs = [
        "[1] Redmon J, Divvala S, Girshick R, Farhadi A. You Only Look Once: Unified, Real-Time Object Detection.",
        "[2] Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection.",
        "[3] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8 documentation and implementation.",
        "[4] Lin T Y, Dollár P, Girshick R, et al. Feature Pyramid Networks for Object Detection.",
        "[5] Liu S, Qi L, Qin H, et al. Path Aggregation Network for Instance Segmentation.",
        "[6] Tan M, Pang R, Le Q V. EfficientDet: Scalable and Efficient Object Detection.",
        "[7] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection.",
        "[8] Zheng Z, Wang P, Liu W, et al. Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.",
        "[9] Rezatofighi H, Tsoi N, Gwak J, et al. Generalized Intersection over Union.",
        "[10] Yuan J, Lei G, Wan F, Xu L. FEB-YOLOv8: A Steel Surface Defect Detection Algorithm Based on Improved YOLOv8s.",
        "[11] 工业表面缺陷检测综述类文献：用于补充钢材、织物、电子元件等工业缺陷检测背景。",
        "[12] 风机叶片无人机巡检与缺陷识别相关文献：用于支撑应用背景。",
        "[13] 小目标检测综述或改进 YOLO 小目标检测论文：用于支撑 P2 检测头设计。",
        "[14] 类别不均衡与困难样本采样相关文献：用于支撑 Hard Sampling。",
        "[15] 切片推理、SAHI 或高分辨率图像小目标检测相关文献：用于支撑 Blade-Slice。",
        "[16] 注意力机制与多尺度融合相关论文：用于对比其他常见改进路线。",
        "[17] 风机叶片裂纹、腐蚀、剥落图像检测数据集或应用论文。",
        "[18] 目标检测评价指标与 mAP 相关标准文献或工具说明。",
    ]
    for ref in refs:
        body.append(para(ref))

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
