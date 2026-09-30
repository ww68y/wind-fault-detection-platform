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
DOCX_PATH = OUT_DIR / "MSF-YOLO_正式论文预稿_消融实验结果待填.docx"

FIGURES = [
    ("rId60", "fig_color_5_msf_yolo_system_pipeline.svg"),
    ("rId61", "fig_1_msf_yolo_framework.svg"),
    ("rId62", "fig_color_0_msf_yolo_precise_network.svg"),
    ("rId63", "fig_color_2_p2_head_structure.svg"),
    ("rId64", "fig_color_3_shape_aware_formula.svg"),
    ("rId65", "fig_2_shape_aware_loss.svg"),
    ("rId66", "fig_color_4_blade_slice_formula.svg"),
    ("rId67", "fig_3_blade_slice_pipeline.svg"),
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


def result_row(code: str, model: str, p2: str, shape: str, hard: str, slice_: str) -> list[str]:
    return [
        code,
        model,
        p2,
        shape,
        hard,
        slice_,
        "实验填入",
        "实验填入",
        "实验填入",
        "实验填入",
        "统一脚本计算",
    ]


def document_xml() -> str:
    body: list[str] = []

    body.append(para("MSF-YOLO：面向风机叶片缺陷检测的多尺度形态感知算法研究", style="Title"))
    body.append(para("正式论文预稿", style="Subtitle"))

    body.append(para("摘要", style="Heading1"))
    body.append(
        para(
            "风机叶片长期暴露在风沙、雨蚀和交变载荷环境中，表面裂纹、孔洞、腐蚀和剥落等缺陷具有较强的隐蔽性。"
            "无人机巡检能够获得大范围叶片图像，但缺陷在整幅图中占比很小，且常伴随细长形态、不规则边界、类别分布不均和背景纹理干扰。"
            "通用 YOLO 检测器在该场景下容易漏掉远距离小缺陷，也可能把污渍、阴影或叶片纹理误判为缺陷。"
            "围绕这一问题，本文构建了多尺度形态感知检测算法 MSF-YOLO。"
            "算法在原 YOLO 检测头中加入 P2 高分辨率分支，形成 P2-P5 四尺度预测结构；"
            "在定位损失中引入由目标面积占比和长宽比计算得到的 Shape-Aware 权重，使小面积和细长目标在训练中获得更高的定位权重；"
            "训练采样阶段提高 crack、corrosion 等困难样本的出现频率；推理阶段采用 Blade-Slice 重叠切片融合，补偿整图缩放带来的局部细节损失。"
            "实验部分按训练模块、推理策略、参数敏感性、分类别指标、背景误检和可视化结果组织，便于逐项判断各改进的实际作用。"
        )
    )
    body.append(para("关键词：风机叶片；缺陷检测；YOLO；小目标检测；Shape-Aware Loss；切片推理"))

    body.append(para("Abstract", style="Heading1"))
    body.append(
        para(
            "Wind turbine blade defects are usually small in area, irregular in shape, and easily disturbed by blade texture, stains and illumination changes. "
            "For this inspection scenario, MSF-YOLO is developed as a multi-scale shape-aware detector. "
            "The detector adds a P2 high-resolution branch to the YOLO head, applies a Shape-Aware localization weight based on object area and aspect ratio, "
            "uses hard-sample reweighting for difficult categories, and performs Blade-Slice fusion inference on high-resolution images. "
            "The evaluation is arranged around training-module ablation, inference ablation, parameter sensitivity, category-wise metrics, background false positives and visual comparison."
        )
    )
    body.append(para("Keywords: wind turbine blade; defect detection; YOLO; small object detection; Shape-Aware Loss; sliced inference"))

    body.append(para("一、研究背景与问题定义", style="Heading1"))
    body.append(para("1.1 研究背景", style="Heading2"))
    body.append(
        para(
            "风力发电机组的叶片尺寸大、服役周期长，长期受到风沙、雨蚀、盐雾、温湿度变化以及交变载荷影响。"
            "叶片表面早期缺陷往往并不显著，但裂纹扩展、腐蚀加深或涂层剥落进一步发展后，会增加结构风险和运维成本。"
            "因此，如何在无人机巡检图像中及时、稳定地识别叶片表面缺陷，是风机智能运维中的关键问题。"
        )
    )
    body.append(
        para(
            "传统人工巡检依赖经验判断，存在效率低、主观性强和漏检风险高等不足。"
            "深度学习检测方法能够从图像中学习缺陷纹理和边界特征，已经成为工业视觉检测的重要技术路线。"
            "但风机叶片图像不同于常规自然图像：缺陷区域占比小，裂纹和腐蚀边缘形态不稳定，叶片纹理、光照和污渍都会引入干扰。"
            "模型设计需要同时考虑小目标召回、困难类别识别和背景误检控制，而不能只追求单一精度指标。"
        )
    )

    body.append(para("1.2 国内外研究现状", style="Heading2"))
    body.append(
        para(
            "工业表面缺陷检测经历了由人工特征到深度特征的演进。早期方法多依赖灰度、纹理、边缘、形态学和统计特征，"
            "在规则背景和稳定光照下能够取得一定效果，但面对多源巡检图像中的尺度变化、表面纹理干扰和复杂光照时鲁棒性不足。"
            "深度卷积神经网络通过端到端特征学习减少了人工特征设计依赖，逐渐成为工业缺陷检测的主流路线。"
        )
    )
    body.append(
        para(
            "目标检测算法通常可分为两阶段检测器和单阶段检测器。Faster R-CNN 等两阶段方法具有较强定位能力，但推理速度和部署成本较高；"
            "YOLO 系列等单阶段方法在速度与精度之间取得较好平衡，更适合巡检场景中的工程应用。"
            "近年来，改进 YOLO 的研究主要集中于轻量化骨干网络、多尺度特征融合、注意力机制、小目标检测头和损失函数优化等方向。"
            "FEB-YOLOv8 等工业缺陷检测研究通常将网络结构、模块改进、消融实验和可视化结果放在同一论证链条中展开。"
            "与钢材表面等相对规则的检测对象相比，风机叶片巡检图像中的缺陷尺度更小、形态更分散，整图推理时还存在明显的细节压缩问题。"
            "本文的改进重点因此放在浅层小目标检测、形态感知定位和切片推理补偿三个方面。"
        )
    )

    body.append(para("1.3 检测难点与创新点", style="Heading2"))
    body.append(
        table(
            [
                ["序号", "检测难点", "具体表现", "本文对应设计"],
                ["1", "小尺度缺陷易退化", "远距离裂纹、小腐蚀和小孔洞在整图缩放后仅占少量像素", "引入 P2 高分辨率检测分支"],
                ["2", "缺陷形态差异明显", "裂纹细长，腐蚀和剥落边界不规则，普通损失难以体现形态差异", "设计 Shape-Aware Loss"],
                ["3", "困难类别学习不足", "crack、corrosion 等样本更容易低置信度或漏检", "构建困难样本重加权训练机制"],
                ["4", "整图推理细节损失", "高分辨率图像缩放后局部小缺陷信息被压缩", "采用 Blade-Slice 重叠切片融合推理"],
            ],
            widths=[800, 2200, 5200, 3300],
        )
    )
    body.append(caption("表 1  风机叶片缺陷检测难点与本文改进设计"))
    body.append(image("rId60", 60, "fig_color_5_msf_yolo_system_pipeline.svg", 5_850_000, 2_420_000))
    body.append(caption("图 1  MSF-YOLO 研究总体流程"))

    body.append(para("二、MSF-YOLO 算法设计", style="Heading1"))
    body.append(para("2.1 总体框架", style="Heading2"))
    body.append(
        para(
            "MSF-YOLO 的整体框架如图 2 所示。训练阶段，模型在 YOLO 检测框架基础上引入 P2 检测头、Shape-Aware Loss 和困难样本重加权；"
            "推理阶段，Blade-Slice 对全图检测与切片检测结果进行融合。"
            "两类模块的作用边界不同：P2、Shape-Aware Loss 和困难样本重加权参与训练，Blade-Slice 只改变推理流程，不写入模型权重。"
        )
    )
    body.append(image("rId61", 61, "fig_1_msf_yolo_framework.svg", 5_850_000, 3_470_000))
    body.append(caption("图 2  MSF-YOLO 总体框架"))

    body.append(para("2.2 精确网络结构", style="Heading2"))
    body.append(
        para(
            "图 3 按实际训练配置给出了 MSF-YOLO 的网络连接关系。"
            "主干网络由第 0-9 层组成，逐级得到 P2、P3、P4 和 P5 特征；第 10-27 层构成特征融合部分，"
            "通过上采样、拼接和 C2f 模块把深层语义信息回传到浅层尺度。"
            "最终送入 Detect 的是第 18、21、24 和 27 层，分别对应 P2/4、P3/8、P4/16 和 P5/32。"
            "其中 P2 分支保留了步长为 4 的浅层细节，主要用于远距离裂纹、小孔洞和局部腐蚀等小缺陷的预测。"
        )
    )
    body.append(image("rId62", 62, "fig_color_0_msf_yolo_precise_network.svg", 5_900_000, 4_100_000))
    body.append(caption("图 3  MSF-YOLO 精确网络结构"))
    body.append(
        table(
            [
                ["项目", "设置"],
                ["检测类别数", "4 类：crack、hole、spalling、corrosion"],
                ["输入尺寸", "1024 × 1024"],
                ["检测尺度", "P2/4、P3/8、P4/16、P5/32"],
                ["P2 结构统计", "161 layers，2.927M parameters，12.4 GFLOPs"],
                ["训练目标", "YOLO 检测损失 + Shape-Aware 定位权重 + 困难样本重加权"],
            ],
            widths=[2600, 7600],
        )
    )
    body.append(caption("表 2  MSF-YOLO 网络结构设置"))

    body.append(para("2.3 P2 小目标检测头", style="Heading2"))
    body.append(
        para(
            "常规 YOLO 检测头通常使用 P3、P4 和 P5 三个预测尺度，最小检测步长为 8。"
            "当远距离裂纹或小面积腐蚀区域在 1024 输入图像中仅占很少像素时，若最浅预测层为 P3，目标在特征图上的响应单元数量有限，容易在下采样过程中退化。"
            "本文将 P2/4 高分辨率特征纳入检测头，保留更多浅层边缘和纹理信息，使小缺陷在特征图上对应到更多响应单元。"
        )
    )
    body.append(image("rId63", 63, "fig_color_2_p2_head_structure.svg", 5_650_000, 2_260_000))
    body.append(caption("图 4  P2 小目标检测头结构"))
    body.append(formula("F_l ∈ R^{C_l × ceil(H/s_l) × ceil(W/s_l)},     s_l ∈ {4, 8, 16, 32}"))
    body.append(formula("N_cell(b,l)=ceil(w_b/s_l) · ceil(h_b/s_l)"))
    body.append(
        para(
            "其中，F_l 表示第 l 个尺度的特征图，s_l 表示对应步长，w_b 和 h_b 为目标框在输入图像中的像素宽高。"
            "N_cell(b,l) 用于描述目标在特征图上覆盖的响应单元数量。与用 sqrt(w_bh_b)/s_l 粗略估计单边尺度相比，上式能够同时反映宽度和高度方向的覆盖情况，"
            "更适合解释裂纹等细长目标在浅层特征图上的可检测性。"
        )
    )
    body.append(
        table(
            [
                ["检测层", "stride", "1024 输入下特征图", "主要关注对象"],
                ["P2", "4", "256 × 256", "小裂纹、小腐蚀、小孔洞"],
                ["P3", "8", "128 × 128", "小目标与中等目标"],
                ["P4", "16", "64 × 64", "中等目标"],
                ["P5", "32", "32 × 32", "较大缺陷与强语义目标"],
            ],
            widths=[1500, 1200, 3000, 4600],
        )
    )
    body.append(caption("表 3  多尺度检测层设置"))

    body.append(para("2.4 Shape-Aware 形态感知损失", style="Heading2"))
    body.append(
        para(
            "风机叶片缺陷具有明显几何差异：裂纹通常细长，腐蚀和剥落边界不规则，小孔洞面积较小。"
            "若所有正样本框在定位损失中使用相同权重，模型容易优先学习面积较大、形态较规则的缺陷。"
            "Shape-Aware Loss 将目标框面积占比和长宽比转化为形态权重 q_i，并作用于框回归损失和 DFL 损失。"
        )
    )
    body.append(image("rId64", 64, "fig_color_3_shape_aware_formula.svg", 5_650_000, 2_720_000))
    body.append(caption("图 5  Shape-Aware Loss 计算结构"))
    body.append(image("rId65", 65, "fig_2_shape_aware_loss.svg", 5_650_000, 2_900_000))
    body.append(caption("图 6  Shape-Aware Loss 权重计算流程"))
    body.append(formula("A_i = w_i h_i / (W H),     R_i = max(w_i/h_i, h_i/w_i)"))
    body.append(formula("S_i = clip((a_0 - A_i)/a_0, 0, 1),     E_i = clip((R_i - r_0)/r_0, 0, 1)"))
    body.append(formula("q_i = min(1 + αS_i + βE_i, q_max)"))
    body.append(formula("L_box^shape = Σ_i (1-CIoU_i)t_iq_i / Σ_i t_i"))
    body.append(formula("L_dfl^shape = Σ_i DFL_i t_iq_i / Σ_i t_i"))
    body.append(
        para(
            "其中，A_i 为目标面积占比，R_i 为目标长宽比，S_i 和 E_i 分别表示小面积项与细长项，t_i 为目标分配得到的正样本权重。"
            "当前实现采用 α=0.35、β=0.25、a_0=0.012、r_0=3.0、q_max=1.7。"
            "S_i 和 E_i 均限制在 [0,1] 内，因此在当前参数下 q_i 的有效范围为 [1,1.60]；q_max=1.7 保留为参数搜索时的上限。"
            "损失分母沿用 Σ_i t_i，而不是 Σ_i t_iq_i。这样在一个 batch 中出现小面积或细长目标时，定位损失会被适度放大，训练过程对困难形态目标更敏感。"
        )
    )
    body.append(
        table(
            [
                ["参数", "取值", "含义"],
                ["a_0", "0.012", "小面积目标阈值"],
                ["r_0", "3.0", "细长目标长宽比阈值"],
                ["α", "0.35", "小面积项增益"],
                ["β", "0.25", "细长项增益"],
                ["q_max", "1.7", "形态权重上限；当前参数下有效最大值为 1.60"],
            ],
            widths=[1800, 1600, 6600],
        )
    )
    body.append(caption("表 4  Shape-Aware Loss 参数设置"))

    body.append(para("2.5 困难样本重加权训练", style="Heading2"))
    body.append(
        para(
            "数据集中不同缺陷类别的样本分布和识别难度并不一致。crack 和 corrosion 在巡检图像中通常面积更小、纹理更接近背景，"
            "若完全按照原始样本分布训练，模型可能更偏向于学习数量更多或形态更明显的类别。"
            "训练采样时对含有困难类别和小目标的图像提高抽样概率，使这些样本在相同训练轮数下被模型更充分地看到。"
        )
    )
    body.append(formula("w_i = clip(1.0 × 3.0^{I_i[corrosion]} × 2.0^{I_i[crack]} × 1.5^{I_i[small]}, 0.05, 6.0)"))
    body.append(formula("p_i = w_i / Σ_j w_j"))
    body.append(
        table(
            [
                ["样本类型", "触发条件", "权重系数"],
                ["普通样本", "不含困难类别或小目标条件", "1.0"],
                ["裂纹样本", "包含 crack", "2.0"],
                ["腐蚀样本", "包含 corrosion", "3.0"],
                ["小目标样本", "困难类别且面积占比较小", "1.5"],
                ["上限截断", "多条件叠加超过上限", "6.0"],
            ],
            widths=[2300, 5200, 1800],
        )
    )
    body.append(caption("表 5  困难样本采样权重设置"))

    body.append(para("2.6 Blade-Slice 重叠切片融合推理", style="Heading2"))
    body.append(
        para(
            "Blade-Slice 用于推理阶段的尺度补偿。其处理流程与高分辨率切片推理相近：先保留一条全图检测分支，再对原始图像进行重叠切片，"
            "在局部高分辨率视野中检测小缺陷，随后把切片检测框映射回原图坐标并与全图结果融合。"
            "该模块不改变模型结构和训练权重，实验中单独统计其带来的召回变化、误检变化和耗时变化。"
        )
    )
    body.append(image("rId66", 66, "fig_color_4_blade_slice_formula.svg", 5_650_000, 2_600_000))
    body.append(caption("图 7  Blade-Slice 切片融合推理结构"))
    body.append(image("rId67", 67, "fig_3_blade_slice_pipeline.svg", 5_650_000, 2_900_000))
    body.append(caption("图 8  Blade-Slice 重叠切片融合推理流程"))
    body.append(formula("d = floor(L(1-ρ)),     L=640, ρ=0.25, d=480"))
    body.append(formula("C_{u,v}=I[u:u+L, v:v+L]"))
    body.append(formula("B_fused = NMS(B_full ∪ B_slice, τ)"))

    body.append(para("三、数据集构建与预处理", style="Heading1"))
    body.append(para("3.1 数据集来源与类别映射", style="Heading2"))
    body.append(
        para(
            "实验数据采用 wind_blade_defect_v05_conservative 版本。"
            "该数据集由多来源风机叶片缺陷图像整理而成，原始标签被统一映射为 crack、hole、spalling 和 corrosion 四类。"
            "对于语义接近但命名不同的标签，按照缺陷形态和工程含义进行合并；Good、Dirt-only 等无缺陷或弱干扰图像保留为空标注背景样本。"
        )
    )
    body.append(
        table(
            [
                ["目标类别", "原始类别映射"],
                ["crack", "Crack / Scratch / Craze / Hide_craze"],
                ["corrosion", "Erosion / Corrosion"],
                ["spalling", "Damage / MechanicalDamage / Paintoff / Surface_injure / Chipping"],
                ["hole", "Puncture / Hole"],
                ["negative", "Good / Dirt-only 等空标注背景图"],
            ],
            widths=[2200, 7800],
        )
    )
    body.append(caption("表 6  缺陷类别映射关系"))

    body.append(para("3.2 数据划分与背景样本", style="Heading2"))
    body.append(
        table(
            [
                ["数据集划分", "图像数量", "空标注图像数量", "用途"],
                ["train", "9136", "786", "模型训练"],
                ["val", "571", "17", "训练过程验证与阈值选择"],
                ["test_clean", "543", "18", "干净测试集评估"],
            ],
            widths=[1800, 1800, 2200, 4600],
        )
    )
    body.append(caption("表 7  数据集划分统计"))
    body.append(
        para(
            "空标注背景图并非无意义样本，而是用于约束模型在无缺陷场景下的误检风险。"
            "实验记录中将背景误检数 FP_bg 与 Precision、Recall 和 mAP 同时列出，以避免召回率提升掩盖误报增加。"
            "各组实验使用同一训练、验证和测试划分；数据集若再次去重或清洗，则重新记录数据版本号和样本数量。"
        )
    )

    body.append(para("3.3 数据增强与训练输入", style="Heading2"))
    body.append(
        para(
            "训练阶段采用 1024 × 1024 输入尺寸，以保留叶片小缺陷的像素细节。"
            "数据增强包括 mosaic、尺度缩放、水平翻转和颜色扰动等常规策略；copy-paste、multi-scale 等增强在消融组之间保持一致，"
            "使实验差异主要来自模型模块本身。"
        )
    )
    body.append(
        table(
            [
                ["处理环节", "设置原则"],
                ["标签格式", "统一为 YOLO 检测格式"],
                ["输入尺寸", "1024 × 1024"],
                ["负样本", "保留空标注图像，用于评估背景误检"],
                ["增强策略", "所有消融实验保持一致；仅在专门增强实验中改变"],
                ["数据版本", "使用 wind_blade_defect_v05_conservative 并记录最终 hash/目录名"],
            ],
            widths=[2400, 7600],
        )
    )
    body.append(caption("表 8  数据预处理与增强设置"))

    body.append(para("四、实验设计与评价指标", style="Heading1"))
    body.append(para("4.1 实验环境", style="Heading2"))
    body.append(
        table(
            [
                ["项目", "设置"],
                ["操作系统", "Windows 平台"],
                ["GPU", "NVIDIA GeForce RTX 4080 SUPER，16376 MiB"],
                ["NVIDIA 驱动 / CUDA", "Driver 560.94 / CUDA 12.6"],
                ["训练日志环境", "Ultralytics 8.4.60，Python 3.12.7，torch 2.6.0+cu124，CUDA:0"],
                ["输入尺寸", "1024 × 1024"],
                ["优化器", "AdamW 或实验组指定优化器"],
                ["训练轮数", "按各消融组统一设置；最终以实验记录为准"],
                ["评价数据", "val、test_clean、空标注背景图"],
            ],
            widths=[3000, 7200],
        )
    )
    body.append(caption("表 9  实验环境与训练设置"))

    body.append(para("4.2 评价指标", style="Heading2"))
    body.append(formula("Precision = TP/(TP+FP),     Recall = TP/(TP+FN)"))
    body.append(formula("mAP50-95 = (1/10) Σ_{τ∈{0.50,0.55,...,0.95}} mAP_τ"))
    body.append(formula("FP_bg = Σ_{I_i∈D_empty} |{b∈f_θ(I_i): conf(b) ≥ γ}|"))
    body.append(
        para(
            "其中，Precision 反映误检控制能力，Recall 反映漏检风险，mAP50-95 反映多 IoU 阈值下的综合定位能力。"
            "FP_bg 表示空标注背景图中的误检框数量，用于衡量无缺陷场景下的误报风险。"
            "除总体指标外，实验还统计各类别 Precision、Recall、mAP50 和 mAP50-95，以观察总体性能变化是否来自少数类别。"
        )
    )
    body.append(
        table(
            [
                ["指标", "含义", "关注原因"],
                ["Precision", "预测为缺陷的框中真实缺陷所占比例", "控制背景误检和误报"],
                ["Recall", "真实缺陷中被模型检出的比例", "控制漏检风险"],
                ["mAP50", "IoU=0.50 下的平均精度", "反映较宽松定位条件下的检测能力"],
                ["mAP50-95", "多个 IoU 阈值下的平均 mAP", "反映更严格定位条件下的综合性能"],
                ["FP_bg", "空标注背景图中的误检框数量", "衡量工程部署误报风险"],
            ],
            widths=[1800, 4200, 4200],
        )
    )
    body.append(caption("表 10  评价指标说明"))

    body.append(para("4.3 训练模块消融实验", style="Heading2"))
    body.append(
        para(
            "训练模块消融用于验证 P2、Shape-Aware Loss 和困难样本重加权的独立贡献。"
            "各训练组保持相同的数据划分、输入尺寸、训练轮数、优化器和评价脚本。"
            "Blade-Slice 不放入该组消融，避免把推理流程带来的变化计入训练模块贡献。"
        )
    )
    body.append(
        table(
            [
                ["编号", "模型设置", "P2", "Shape", "Hard", "Slice", "Precision", "Recall", "mAP50-95", "FP_bg", "备注"],
                result_row("A0", "YOLO baseline", "否", "否", "否", "否"),
                result_row("A1", "YOLO + P2", "是", "否", "否", "否"),
                result_row("A2", "YOLO + P2 + Shape", "是", "是", "否", "否"),
                result_row("A3", "YOLO + P2 + Shape + Hard", "是", "是", "是", "否"),
            ],
            widths=[800, 3000, 700, 900, 900, 900, 1200, 1200, 1400, 1000, 1800],
        )
    )
    body.append(caption("表 11  训练模块消融实验记录表"))

    body.append(para("4.4 推理策略消融实验", style="Heading2"))
    body.append(
        para(
            "推理策略消融用于单独验证 Blade-Slice 的收益与代价。"
            "该组实验固定同一个训练完成模型，分别测试整图推理、仅切片推理和整图加切片融合推理，重点记录 Recall、FP_bg 和推理耗时的同步变化。"
        )
    )
    body.append(
        table(
            [
                ["编号", "推理方式", "tile", "overlap", "融合方式", "Precision", "Recall", "mAP50-95", "FP_bg", "FPS/耗时"],
                ["S0", "Full image", "-", "-", "NMS", "实验填入", "实验填入", "实验填入", "实验填入", "实验填入"],
                ["S1", "Slice only", "640", "0.25", "NMS", "实验填入", "实验填入", "实验填入", "实验填入", "实验填入"],
                ["S2", "Full + Slice", "640", "0.25", "NMS", "实验填入", "实验填入", "实验填入", "实验填入", "实验填入"],
            ],
            widths=[800, 2000, 1000, 1000, 1400, 1200, 1200, 1400, 1000, 1200],
        )
    )
    body.append(caption("表 12  Blade-Slice 推理消融实验记录表"))

    body.append(para("4.5 分类别指标与错误分析", style="Heading2"))
    body.append(
        table(
            [
                ["模型", "类别", "Precision", "Recall", "mAP50", "mAP50-95", "Missed", "FP"],
                ["A0-A3/S2", "crack", "实验填入", "实验填入", "实验填入", "实验填入", "统一脚本计算", "统一脚本计算"],
                ["A0-A3/S2", "hole", "实验填入", "实验填入", "实验填入", "实验填入", "统一脚本计算", "统一脚本计算"],
                ["A0-A3/S2", "spalling", "实验填入", "实验填入", "实验填入", "实验填入", "统一脚本计算", "统一脚本计算"],
                ["A0-A3/S2", "corrosion", "实验填入", "实验填入", "实验填入", "实验填入", "统一脚本计算", "统一脚本计算"],
            ],
            widths=[1500, 1400, 1400, 1400, 1300, 1500, 1200, 1100],
        )
    )
    body.append(caption("表 13  分类别检测结果记录表"))
    body.append(
        para(
            "分类别结果主要观察 crack 和 corrosion 的召回变化，同时检查 spalling 等样本较多类别的 Precision 是否下降。"
            "若某一模块带来召回提升但同时增加背景误检，则在结论中限定其适用场景。"
        )
    )

    body.append(para("4.6 参数敏感性实验", style="Heading2"))
    body.append(
        table(
            [
                ["实验项", "参数设置", "观察指标", "目的"],
                ["Shape 面积阈值", "a_0 ∈ {0.006, 0.012, 0.018}", "mAP50-95、crack/corrosion Recall、FP_bg", "判断小目标权重阈值是否稳定"],
                ["细长阈值", "r_0 ∈ {2.5, 3.0, 3.5}", "crack Recall、定位质量", "验证细长形态建模是否有效"],
                ["形态增益", "α/β 分组变化", "总体指标与分类别指标", "避免权重过大导致误检增加"],
                ["切片重叠率", "ρ ∈ {0.15, 0.25, 0.35}", "Recall、FP_bg、推理耗时", "平衡切片边界漏检与计算成本"],
            ],
            widths=[1800, 3100, 3600, 3000],
        )
    )
    body.append(caption("表 14  参数敏感性实验设计"))

    body.append(para("4.7 结果曲线与可视化图", style="Heading2"))
    body.append(
        para(
            "PR 曲线、F1-Confidence 曲线、Precision-Confidence 曲线、Recall-Confidence 曲线和混淆矩阵用于分析训练后的置信度分布、阈值选择和类别识别趋势。"
            "相关图表放在“模型训练过程与阈值选择”小节；"
            "模型优劣比较仍以同一评价脚本下的消融表为准。"
        )
    )
    body.append(
        table(
            [
                ["图表类型", "建议位置", "使用要求"],
                ["F1-Confidence 曲线", "阈值选择分析", "标出最佳 F1 对应置信度，不单独作为模型有效性结论"],
                ["PR 曲线", "分类别性能分析", "与各类别 AP 表对应"],
                ["混淆矩阵", "错误分析", "重点解释易混类别与背景误检"],
                ["典型检测图", "可视化结果", "同一图片展示 baseline 与 MSF-YOLO 对比"],
                ["失败案例图", "局限性分析", "保留漏检、误检或定位偏差样例"],
            ],
            widths=[2200, 2800, 5200],
        )
    )
    body.append(caption("表 15  结果曲线与可视化图使用规范"))

    body.append(para("五、阶段性实验记录", style="Heading1"))
    body.append(
        para(
            "当前版本先固定研究背景、方法结构、公式表达、数据集划分和实验记录格式。"
            "定量提升幅度不提前写入摘要和结论，待消融实验完成后再依据统一评价结果补入。"
            "若某一模块只在特定类别或特定推理条件下有效，结果分析中按实际适用范围表述，不作无条件推广。"
        )
    )
    body.append(
        table(
            [
                ["内容", "当前状态", "记录方式"],
                ["研究现状", "已成稿", "围绕工业缺陷检测、YOLO 改进、小目标检测和切片推理组织"],
                ["方法设计", "已成稿", "公式与代码实现保持一致，训练模块和推理模块分开描述"],
                ["数据集构建", "主体完成", "补入来源授权、去重规则和完整类别数量统计"],
                ["消融实验", "结果待填", "所有模型使用统一数据、阈值和评价脚本"],
                ["可视化结果", "版式已定", "最终图像来自同一测试集，同时包含成功和失败案例"],
                ["结论", "结果后修订", "依据最终表格和图像确定结论强度"],
            ],
            widths=[2100, 2500, 5600],
        )
    )
    body.append(caption("表 16  阶段性实验记录与结果写入方式"))

    body.append(para("六、结论", style="Heading1"))
    body.append(
        para(
            "本文面向风机叶片巡检图像中的小尺度、细长形态、困难类别和高分辨率推理细节损失问题，设计了 MSF-YOLO 检测框架。"
            "P2 高分辨率检测头用于保留浅层小缺陷信息，Shape-Aware Loss 用于加强小面积和细长框的定位学习，"
            "困难样本重加权用于调整训练样本分布，Blade-Slice 则在推理阶段补偿局部尺度信息。"
            "性能结论以 YOLO baseline、P2、Shape-Aware Loss、Hard Sampling 和 Blade-Slice 等消融实验的统一评价结果为准。"
        )
    )

    body.append(para("参考文献", style="Heading1"))
    refs = [
        "[1] Redmon J, Divvala S, Girshick R, Farhadi A. You Only Look Once: Unified, Real-Time Object Detection[C]//CVPR, 2016.",
        "[2] Ren S, He K, Girshick R, Sun J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE TPAMI, 2017.",
        "[3] Lin T Y, Dollár P, Girshick R, He K, Hariharan B, Belongie S. Feature Pyramid Networks for Object Detection[C]//CVPR, 2017.",
        "[4] Liu S, Qi L, Qin H, Shi J, Jia J. Path Aggregation Network for Instance Segmentation[C]//CVPR, 2018.",
        "[5] Tan M, Pang R, Le Q V. EfficientDet: Scalable and Efficient Object Detection[C]//CVPR, 2020.",
        "[6] Lin T Y, Goyal P, Girshick R, He K, Dollár P. Focal Loss for Dense Object Detection[C]//ICCV, 2017.",
        "[7] Rezatofighi H, Tsoi N, Gwak J, Sadeghian A, Reid I, Savarese S. Generalized Intersection over Union[C]//CVPR, 2019.",
        "[8] Zheng Z, Wang P, Liu W, Li J, Ye R, Ren D. Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression[C]//AAAI, 2020.",
        "[9] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8: Documentation and Implementation[EB/OL].",
        "[10] Akyon F C, Altinuc S O, Temizel A. Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection[C]//ICIP, 2022.",
        "[11] Chen J, Kao S H, He H, Zhuo W, Wen S, Lee C H, Chan S H G. Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks[C]//CVPR, 2023.",
        "[12] Yuan J, Lei G, Wan F, Xu L. FEB-YOLOv8: A Steel Surface Defect Detection Algorithm Based on Improved YOLOv8s[J].",
        "[13] Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[EB/OL]. arXiv:2004.10934, 2020.",
        "[14] Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors[C]//CVPR, 2023.",
        "[15] Ge Z, Liu S, Wang F, Li Z, Sun J. YOLOX: Exceeding YOLO Series in 2021[EB/OL]. arXiv:2107.08430, 2021.",
        "[16] Neubeck A, Van Gool L. Efficient Non-Maximum Suppression[C]//ICPR, 2006.",
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


def build_docx() -> None:
    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
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
