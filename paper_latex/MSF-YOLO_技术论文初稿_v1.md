# MSF-YOLO：面向风机叶片缺陷检测的多尺度形态感知与多模态推理系统

第二十一届中国研究生电子设计竞赛技术论文初稿

## 摘要

风机叶片长期工作在风沙、雨蚀、盐雾、温差变化和交变载荷环境中，表面容易出现裂纹、孔洞、剥落和腐蚀等缺陷。随着无人机巡检和现场图像采集方式逐渐普及，巡检人员能够获得大量叶片图片和视频文件，但人工逐张筛查效率较低，且对早期细小裂纹、低对比度腐蚀和复杂背景下的疑似损伤容易产生漏检或误判。针对上述问题，本文设计并实现了一套基于 YOLO 的风机叶片缺陷检测与多模态故障分析系统。系统当前采用图片、批量图片和视频文件上传的方式完成本地检测，不涉及摄像头实时采集。

在算法层面，本文围绕风机叶片缺陷小尺度、细长形态和类别不均衡等特点，构建 MSF-YOLO 技术路线，包括 P2 小目标检测头、Shape-Aware Loss 形态感知定位约束、Difficulty-Aware 困难样本重加权以及 Blade-Slice 高分辨率切片融合推理。平台层面，系统基于 FastAPI、Ultralytics YOLO 和本地 Web 工作台实现，支持单图检测、批量图片检测、视频文件检测、结果图展示、检测报告保存、历史记录、多模态分析和维修工单生成。多模态推理模块不替代目标检测模型，而是读取检测报告、原图、结果图、缺陷类别、置信度和检测框等信息，结合风险规则生成故障解释、风险影响、复检要求和运维建议。

实验结果表明，当前 V4 稳定基线在 test_clean 测试集上取得 0.8402 的 mAP50-95；严格消融实验中，P2+Shape 候选模型在 test_clean 上取得 0.8309 的 mAP50-95，尚未超过 V4 稳定基线。小目标子集评估显示，MSF-YOLO V6 在面积小于 1% 的 test_clean 子集上取得 0.7720 的 mAP50-95，高于 A0、B0 和 C0 基线，但在面积小于 0.5% 的极小目标子集上仍低于 A0。本文据实给出实验边界，重点展示从缺陷识别到多模态推理、风险评估和维修工单生成的完整工程链路。

关键词：风机叶片；缺陷检测；YOLO；小目标检测；多模态推理

## Abstract

Wind turbine blades are exposed to wind erosion, rain erosion, salt fog, temperature variation and alternating loads for long periods, which may lead to cracks, holes, spalling and corrosion on blade surfaces. UAV inspection and file-based image acquisition can provide a large number of blade images and videos, but manual checking is inefficient and may miss small cracks or low-contrast defects. This paper develops a YOLO-based wind turbine blade defect detection and multimodal fault analysis system. The current system uses uploaded images, batch images and video files as inputs, and does not include real-time camera acquisition.

At the algorithm level, the proposed MSF-YOLO route introduces a P2 small-object detection head, Shape-Aware Loss, Difficulty-Aware hard-sample reweighting and Blade-Slice tiled inference. At the platform level, FastAPI, Ultralytics YOLO and a local Web workstation are used to support image detection, batch detection, video file detection, result visualization, report saving, history management, multimodal analysis and maintenance work-order generation. The multimodal reasoning module does not replace the detector. Instead, it uses detection reports, original images, annotated images, categories, confidences and bounding boxes to generate fault explanations, risk impacts, re-inspection requirements and maintenance suggestions.

Experiments show that the V4 stable baseline achieves 0.8402 mAP50-95 on test_clean. In strict ablation, the P2+Shape candidate obtains 0.8309 mAP50-95 and does not outperform the V4 baseline. Small-object subset evaluation shows that MSF-YOLO V6 achieves the highest mAP50-95 on the test_clean subset with object area below 1%, while it is still lower than A0 on the extremely small subset below 0.5%. Therefore, this paper reports the performance boundary honestly and focuses on the complete workflow from defect detection to multimodal reasoning, risk assessment and maintenance work-order generation.

Keywords: wind turbine blade; defect detection; YOLO; small object detection; multimodal reasoning

## 目录

一、研究背景与难点创新

二、系统总体方案与软硬件设计

三、数据集构建与算法原理

四、软件实现与平台功能

五、实验结果与系统测试分析

六、总结与展望

参考文献

## 一、研究背景与难点创新

### 1. 问题背景与研究现状

风力发电机组叶片是风机捕获风能和完成能量转换的关键部件，其结构尺寸大、运行周期长、维护成本高。叶片在长期服役过程中会受到雨蚀、沙尘冲刷、盐雾侵蚀、雷击、外物撞击和交变载荷等因素影响，表面可能出现裂纹、孔洞、剥落、腐蚀等缺陷。早期缺陷若不能及时发现，可能进一步扩展为结构损伤，影响气动效率、叶片寿命和机组安全。

传统叶片巡检依赖人工经验，巡检效率和一致性受人员水平影响较大。无人机巡检和长焦相机拍摄可以显著提高图像采集效率，但也带来了新的问题：一次巡检会产生大量图像和视频文件，人工筛查成本高；远距离拍摄时小裂纹在整幅图像中占比很低；叶片表面纹理、阴影、水渍、污迹和反光容易与缺陷混淆。因此，将深度学习目标检测方法引入风机叶片缺陷识别，是提高巡检效率和降低人工复核压力的重要方向。

近年来，YOLO 系列目标检测算法由于推理速度快、工程部署方便，被广泛用于工业缺陷检测。对于风机叶片场景，单纯使用通用检测器仍存在局限：一方面，通用模型更关注常规尺度目标，对小裂纹和细长腐蚀区域的定位能力不足；另一方面，检测模型输出的是类别、置信度和检测框，并不能直接告诉运维人员缺陷风险、复检要求和维修优先级。因此，本项目不仅关注缺陷检测模型本身，也将检测结果进一步组织成多模态故障分析和维修决策信息。

### 2. 研究难点

风机叶片缺陷检测的第一类难点是小目标和细长目标检测。裂纹类缺陷常呈长条状或弯曲状，宽度很小，经过整图缩放后容易只保留少量像素。腐蚀和剥落类缺陷边界不规则，颜色和叶片背景接近，检测框标注也更容易受到人工主观影响。普通 YOLO 在低分辨率输入下虽然速度较快，但浅层细节损失可能导致小缺陷漏检。

第二类难点是复杂背景干扰。巡检图像中经常出现天空、塔筒、叶片阴影、表面污渍、反光区域和拍摄角度变化。部分背景纹理与裂纹或腐蚀外观相似，容易引发误检。对于实际工程系统而言，单纯追求召回率并不充分，误检过多会增加人工复核成本，甚至降低使用人员对系统的信任。

第三类难点是检测结果到运维决策的转化。YOLO 输出的检测框只能说明“疑似缺陷位于哪里”，但巡检人员还需要了解“该缺陷可能是什么原因造成”“风险等级如何”“是否需要停机”“应由哪个检修组处理”“复检时应拍摄什么内容”。这部分信息既包含视觉证据，也包含检测置信度、缺陷类别、数量和维修规则，适合通过规则约束下的多模态推理来完成。

### 3. 研究创新点

本文的第一项创新是提出面向风机叶片缺陷的多尺度形态感知检测思路。系统在 YOLO 检测框架基础上，引入 P2 小目标检测头，使网络在 stride=4 的高分辨率特征层上输出预测结果，以增强小裂纹、小腐蚀和小孔洞的定位能力。同时，设计 Shape-Aware Loss，根据目标框面积和长宽比对定位损失进行加权，使小面积和细长形态缺陷在训练时获得更高关注。

本文的第二项创新是构建困难样本重加权和 Blade-Slice 切片推理机制。困难样本重加权主要针对 crack、corrosion 和小目标样本，提高其训练采样概率；Blade-Slice 则在推理阶段对高分辨率图像进行重叠切片，将局部检测结果回映射到原图并进行融合，用于补偿整图缩放造成的细节损失。实验也表明，Blade-Slice 需要结合置信度门控使用，不能简单作为默认推理模式。

本文的第三项创新是把多模态推理作为系统的重要组成部分。与只展示检测框的普通 Demo 不同，本系统将原图、检测结果图、视频关键帧、缺陷类别、置信度、检测框、风险规则和维修约束组织起来，生成故障摘要、可能原因、风险影响、复检要求和维修工单。多模态推理模块不重新检测缺陷，而是在 YOLO 检测结果基础上完成解释和决策辅助，使系统从“识别缺陷”进一步扩展到“理解缺陷和辅助运维”。

![MSF-YOLO 研究总体流程](figures_from_docx/image1.png)

图 1-1 MSF-YOLO 研究总体流程

## 二、系统总体方案与软硬件设计

### 1. 系统总体设计方案

本系统定位为风机叶片缺陷检测与多模态故障分析平台，当前采用本地图片、批量图片和视频文件上传方式，不涉及摄像头实时采集。系统整体由 Web 工作台、FastAPI 后端、YOLO 推理模块、报告生成模块、多模态分析模块和维修工单模块组成。用户在浏览器中上传风机叶片图片或视频文件，后端保存文件并调用 YOLO 模型进行推理，随后生成检测结果图和检测报告。用户可以进一步触发多模态分析接口，系统读取检测报告并生成故障解释和维修建议。

系统总体流程可以概括为：图片或视频文件输入，YOLO 检测输出缺陷类别、置信度和检测框，报告模块保存结构化结果，多模态分析模块完成风险判断和故障解释，维修模块生成工单和复检要求。该设计保留了目标检测模型的实时性和工程可用性，同时补足了检测结果难以直接服务运维决策的短板。

![MSF-YOLO 总体框架](figures_from_docx/image3.png)

图 2-1 MSF-YOLO 总体框架

### 2. 硬件环境与部署说明

本项目主要运行在本地计算机环境中，后端服务由 FastAPI 提供，模型推理由 Ultralytics YOLO 完成。训练阶段使用 NVIDIA GeForce GTX 1650 GPU 进行验证，1024 输入尺寸下由于显存限制需要控制 batch size。项目已经完成 GPU 训练环境检查，torch 能够识别 CUDA，YOLO 可以调用 device 0 完成训练和推理。

需要明确的是，当前项目没有接入摄像头，也没有实现树莓派 3B+ 实时采集。树莓派 3B+ 可以作为后续扩展方案：其作用是现场采集图像、缓存文件并上传到本地推理服务，而不是当前系统已经完成的核心输入设备。为避免论文表述与项目实际不一致，本文将硬件部分写为本地计算与部署环境说明，并将树莓派定位为后续边缘采集节点的扩展方向。

### 3. 软件系统设计

项目采用 Python 工程结构，主要源码位于 src/wind_fault。api 模块负责 FastAPI 接口和网页静态资源；model 模块封装 YOLO 训练与推理；video 模块负责视频文件抽帧检测、关键帧保存和结果视频生成；reports 模块负责检测报告、错误挖掘和综合报告；analysis 模块负责风险判断、多模态提示词构建和故障分析；maintenance 模块负责维修工单和派工记录生成。

本地 Web 工作台支持图片上传、批量图片上传、视频文件上传、置信度阈值设置、检测状态展示、原图和结果图展示、视频关键帧展示、历史记录查看和报告页面跳转。系统每次检测均会生成唯一 request_id，并将上传文件、渲染图片、报告 JSON、报告 Markdown 和维修单等文件保存到对应运行目录，便于后续复查和论文实验复现。

### 4. 系统整体流程

系统整体流程分为检测流程和分析流程两个阶段。检测流程中，用户上传文件后，FastAPI 服务将文件保存到本地目录，调用 YOLO 模型推理，渲染检测框，并写入 detection_report.json 和 detection_report.md。视频文件检测时，系统使用 OpenCV 和 YOLO 对视频进行抽帧推理，并保存包含缺陷的关键帧。

分析流程中，系统读取已有检测报告，将缺陷类别、置信度、检测框、原图路径、结果图路径和视频关键帧路径整理为多模态输入。风险引擎先根据规则给出风险等级，然后故障分析模块生成可能原因、风险影响、运维建议和复检要求，最后维修工单模块根据风险等级和缺陷类别生成优先级、派工团队、工具材料和验收标准。该流程保证了分析结果能够追溯到检测证据，而不是脱离图像结果进行自由生成。

## 三、数据集构建与算法原理

### 1. 数据集构建与预处理

实验主要采用 wind_blade_defect_v05_conservative 数据集，类别统一为 crack、hole、spalling 和 corrosion。数据集划分包括 train、val 和 test_clean，其中 train 用于模型训练，val 用于训练过程验证，test_clean 用于最终评估。当前统计中，train 包含 9136 张图像，val 包含 571 张图像，test_clean 包含 543 张图像。部分图像为空标注背景图，用于检查模型在正常背景下的误检情况。

数据预处理阶段主要完成类别统一、数据划分、YOLO 标注格式整理、困难样本整理和 DTU 1024 切片数据接入。DTU 数据集属于另一套缺陷体系，目前已经完成切片、标注转换和训练流程测试，但未直接合并到当前四类模型中。对于主论文实验，本文以四类风机叶片缺陷数据集为主，保证训练、验证和测试口径一致。

表 3-1 数据集划分统计

| 数据划分 | 图像数量 | 空标注图像数量 | 用途 |
| --- | ---: | ---: | --- |
| train | 9136 | 786 | 模型训练 |
| val | 571 | 17 | 训练过程验证 |
| test_clean | 543 | 18 | 干净测试集评估 |

### 2. MSF-YOLO 模型构建

MSF-YOLO 的模型构建围绕小目标、细长形态和困难类别三个问题展开。首先，在普通 YOLOv8n 基础上加入 P2 高分辨率检测分支，形成 P2、P3、P4、P5 四尺度预测结构。P2 分支对应更小的特征步长，有助于保留小裂纹和小孔洞的局部纹理。实验配置中，YOLOv8n-p2-wind 模型包含 161 层、约 292.7 万参数和 12.4 GFLOPs。

![MSF-YOLO 精确网络结构](figures_from_docx/image5.png)

图 3-1 MSF-YOLO 精确网络结构

其次，设计 Shape-Aware Loss。对于第 i 个目标框，设其宽高为 w_i、h_i，输入图像尺寸为 W、H，则面积占比和长宽比可表示为：

Ai = wi hi / WH

Ri = max(wi / hi, hi / wi)

根据面积项和细长项构建形态权重 qi，使小面积和细长目标在框回归损失中获得更高权重。该设计符合风机叶片裂纹“细长”、小孔洞“面积小”、腐蚀“形态不规则”的任务特点。

![Shape-Aware Loss 计算结构](figures_from_docx/image9.png)

图 3-2 Shape-Aware Loss 计算结构

再次，采用 Difficulty-Aware 困难样本重加权。重加权策略主要针对 crack、corrosion 和小目标样本，提高其训练出现概率。实验表明，困难样本重加权可以在部分 crack 指标上带来改善，但过度提高 corrosion 权重会增加误检风险，因此该模块需要与数据质量分析和失败样本复核结合使用。

### 3. Blade-Slice 切片融合推理

Blade-Slice 是推理阶段的高分辨率补偿策略。对于大尺寸叶片图像，若直接缩放到模型输入尺寸，小裂纹和局部腐蚀可能被压缩。Blade-Slice 保留全图推理结果，同时对原图进行重叠切片，在局部视野中检测缺陷，再将切片结果回映射到原图坐标系，并通过 NMS 进行融合。

![Blade-Slice 切片融合结构](figures_from_docx/image13.png)

图 3-3 Blade-Slice 切片融合结构

该方法的优点是能够在不重新训练模型的情况下提高局部细节可见性；不足是切片会增加预测次数，并可能放大局部背景纹理造成的误检。实验中，默认参数下 Blade-Slice 在 test_clean 上仅多召回 1 个 crack，但 false positives 从 56 增加到 77。因此，本文将 Blade-Slice 定位为高分辨率小目标复核模式，而不是默认部署推理方式。

### 4. 多模态推理方法

多模态推理方法以 detection_report.json 为核心输入。报告中保存了请求 ID、检测模式、文件名、缺陷数量、类别统计、检测框坐标和置信度等信息。对于图片任务，系统同时保留原图和检测结果图；对于视频任务，系统保留关键帧截图、帧号和时间戳。这些内容共同构成多模态分析的输入证据。

风险评估由规则引擎完成。系统根据缺陷类别设置基础风险：corrosion 为低风险，spalling 为中风险，hole 和 crack 为中高风险；缺陷数量达到 3 个及以上时风险提升一级；同一任务中出现多类缺陷时风险提升一级；多个高置信度缺陷同时出现时按高风险处理；低置信度结果会被标记为需要人工复核。

故障分析模块在规则风险等级基础上生成解释文本。对于 crack，系统会提示长期疲劳载荷、局部应力集中、材料老化或外物冲击等可能原因，并建议人工复核裂纹长度、方向和扩展趋势；对于 corrosion，系统会提示潮湿或盐雾环境、防护层失效和长期侵蚀等原因，并建议检查腐蚀范围和材料强度影响。维修工单模块进一步生成优先级、派工团队、材料工具、安全控制和复检验收要求。

## 四、软件实现与平台功能

### 1. 图片检测功能实现

单张图片检测由 POST /detect 接口完成。用户在网页端选择图片并设置置信度阈值后，后端创建运行目录，将上传图片保存到 uploads 子目录，调用 YOLO 模型完成推理，并将结果图渲染到 rendered 子目录。接口返回 request_id、检测状态、图片数量、缺陷数量、文件名、原图 URL、结果图 URL 和报告链接。

该功能已经完成本地运行验证。例如，上传 crack_388.jpg 后，系统检测到 1 个 crack 缺陷，置信度约为 0.968，检测框覆盖叶片表面的细长裂纹区域。平台能够在网页中同时显示原图和检测结果图，并在报告中记录检测框坐标和类别信息。

![平台裂纹检测案例](figures_cases/case_crack_388_pred.jpg)

图 4-1 平台裂纹检测结果图

### 2. 批量图片检测功能实现

批量图片检测由 POST /detect/batch 接口完成。用户可以一次选择多张叶片图片，系统将文件保存到同一请求目录下，再逐张调用 YOLO 模型推理。批量检测报告会统计总图片数量、总缺陷数量、类别分布和每张图片的检测明细。该功能适合巡检人员对某次任务采集的图片目录进行快速筛查。

批量检测与单图检测共用报告格式，这保证了后续多模态分析模块可以复用同一套读取逻辑。对于无缺陷图片，系统会在报告中记录未检出缺陷，并保留原始文件路径，便于后续人工复核。

### 3. 视频文件检测功能实现

视频文件检测由 POST /detect/video 接口完成。当前系统不是摄像头实时检测，而是对用户上传的视频文件进行离线检测。video 模块使用 OpenCV 读取视频帧，调用 YOLO 进行抽帧推理，保存缺陷帧信息、关键帧截图、检测结果视频和视频检测报告。系统还接入 FFmpeg，将结果视频转码为浏览器可播放的 H.264 MP4 预览格式。

视频检测主要用于模拟无人机巡检视频场景，观察模型在连续帧中的稳定性。如果同一缺陷在多个相邻帧中连续出现，说明该缺陷可信度较高；如果只在单帧出现，则需要排除运动模糊、反光或背景干扰。因此，视频检测报告会重点保存缺陷关键帧、帧号、时间戳和最高置信度，而不是简单输出一段视频。

### 4. 检测报告与历史记录

系统每次检测都会保存 detection_report.json 和 detection_report.md。JSON 报告面向程序读取，包含检测框、类别、置信度、路径和风险字段；Markdown 报告面向人工查看，便于导出和归档。历史记录接口会扫描 runs/api 目录下的检测任务，读取报告摘要，并在网页端展示最近任务，用户可点击进入报告页面。

报告机制是本项目从“检测 Demo”走向“巡检平台”的关键。由于所有中间结果都保存在运行目录中，后续可以从维修建议回溯到分析报告、检测报告、结果图和原始图片。这种可追溯性有助于人工复核、失败样本整理和模型再训练。

### 5. 多模态分析与维修工单生成

多模态分析由 POST /analyze/{request_id} 接口触发。系统根据 request_id 找到对应检测报告，读取缺陷信息并计算风险等级，再生成 analysis_report.json 和 analysis_report.md。当前默认使用规则型 mock 分析，保证无外部模型接口时系统仍能稳定运行；当配置视觉语言模型接口后，vlm_client 模块可以将原图、结果图和关键帧作为图像输入传给 OpenAI-compatible 接口。

维修工单生成由 maintenance 模块完成。系统根据风险等级和主要缺陷类别自动设置维修优先级。例如 crack 且高置信度时，维修单可能进入 P0 紧急，分配到叶片结构检修组，并要求复核裂纹长度、方向、端点和是否跨越结构受力区域；corrosion 通常优先级较低，但仍要求检查腐蚀面积、深度和防护层失效范围。该功能使平台输出从检测框扩展为可执行的维修任务。

表 4-1 软件模块划分

| 模块 | 主要功能 |
| --- | --- |
| api | FastAPI 接口、网页工作台、报告页面与历史记录 |
| model | YOLO 训练与推理封装 |
| video | 视频文件检测、关键帧和结果视频生成 |
| reports | 检测报告、错误挖掘和综合报告生成 |
| analysis | 风险评估、提示词构建和多模态故障分析 |
| maintenance | 维修工单、派工记录和验收要求生成 |

## 五、实验结果与系统测试分析

### 1. 评价指标

本文采用 Precision、Recall、mAP50 和 mAP50-95 作为主要评价指标。Precision 衡量预测结果中真实缺陷所占比例，Recall 衡量真实缺陷被检出的比例，mAP50 表示 IoU 阈值为 0.5 时的平均精度，mAP50-95 则统计 IoU 从 0.5 到 0.95 多个阈值下的平均精度，更能反映定位质量。对于工程部署，还统计空标注背景图上的误检数量 FP_bg，用于评估模型在正常背景下的稳定性。

### 2. 消融实验结果

严格消融实验中，V4 baseline 在 test_clean 上取得 0.9577 Precision、0.9795 Recall 和 0.8402 mAP50-95，是当前最稳定的部署基线。候选模型中，YOLO+P2+Shape 在 test_clean 上取得 0.8309 mAP50-95，为候选组最优，但仍未超过 V4 baseline。该结果说明，新增模块已完成实现链路，但当前训练策略下还不能直接替代稳定基线。

表 5-1 训练侧消融总体指标

| 模型 | val P | val R | val mAP50-95 | test P | test R | test mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 baseline | 0.9599 | 0.9411 | 0.8094 | 0.9577 | 0.9795 | 0.8402 |
| YOLO + P2 | 0.9358 | 0.9097 | 0.7941 | 0.9663 | 0.9493 | 0.8287 |
| YOLO + P2 + Shape | 0.9137 | 0.9371 | 0.7990 | 0.9670 | 0.9581 | 0.8309 |
| YOLO + P2 + Hard | 0.9423 | 0.9163 | 0.7897 | 0.9612 | 0.9492 | 0.8290 |
| YOLO + P2 + Shape + Hard | 0.9125 | 0.9392 | 0.7940 | 0.9516 | 0.9488 | 0.8257 |

![严格消融候选组雷达对比](figures_from_docx/image17.png)

图 5-1 严格消融候选组雷达对比

### 3. 分类别检测结果分析

从 test_clean 分类别 mAP50-95 看，V4 baseline 在 crack、hole 和 corrosion 上仍具有优势，spalling 类指标较高且各组差异较小。corrosion 指标明显低于其他类别，是后续优化重点。V5.1 困难样本重加权实验显示，简单提高 corrosion 采样权重并不能稳定提升 corrosion 指标，反而可能增加背景误检，说明 corrosion 问题更可能来自外观边界模糊、标注一致性和场景差异。

表 5-2 test_clean 分类别 mAP50-95

| 模型 | crack | hole | spalling | corrosion |
| --- | ---: | ---: | ---: | ---: |
| V4 baseline | 0.8613 | 0.8945 | 0.9720 | 0.6331 |
| YOLO + P2 | 0.8524 | 0.8913 | 0.9739 | 0.5974 |
| YOLO + P2 + Shape | 0.8408 | 0.8893 | 0.9719 | 0.6215 |
| YOLO + P2 + Hard | 0.8452 | 0.8878 | 0.9730 | 0.6102 |
| YOLO + P2 + Shape + Hard | 0.8538 | 0.8767 | 0.9697 | 0.6024 |

### 4. 小目标检测结果分析

为避免整体测试集中中大尺度缺陷掩盖小目标表现，本文补充构建小目标子集评估，仅统计归一化面积小于 1% 和 0.5% 的缺陷框。结果显示，在 test_clean 的 area<=0.01 子集上，MSF-YOLO V6 取得 0.7720 mAP50-95，高于 A0 的 0.7691、B0 的 0.7536 和 C0 的 0.7496。但在 area<=0.005 的极小目标子集上，A0 的 mAP50-95 为 0.7607，高于 V6 的 0.7443。

该结果支持一个保守结论：MSF-YOLO 的多尺度和形态感知设计在部分小目标场景中具有收益，尤其相对低分辨率普通 YOLO 基线更稳定；但普通 YOLOv8n-640 仍是强竞争基线，不能写成 MSF-YOLO 在所有小目标指标上全面领先。

表 5-3 小目标子集 mAP50-95 对比

| 阈值 | Split | V6 | A0-640 | B0-416 | C0-320 |
| --- | --- | ---: | ---: | ---: | ---: |
| area <= 0.01 | val | 0.5249 | 0.5283 | 0.5327 | 0.5181 |
| area <= 0.01 | test_clean | 0.7720 | 0.7691 | 0.7536 | 0.7496 |
| area <= 0.005 | val | 0.4768 | 0.5131 | 0.4865 | 0.4611 |
| area <= 0.005 | test_clean | 0.7443 | 0.7607 | 0.7329 | 0.6973 |

### 5. Blade-Slice 推理测试

Blade-Slice 在 V4 稳定基线上进行普通推理与切片融合推理对比。测试结果显示，test_clean 上 normal 模式 Recall 为 0.8883，Precision 为 0.9036；tile 640 模式 Recall 为 0.8900，Precision 降至 0.8723。切片融合只多召回 1 个 crack，但 false positives 从 56 增加到 77，主要新增在 crack 类误检。因此，该模块更适合作为高分辨率疑难图像的复核模式，而不是默认推理策略。

表 5-4 Blade-Slice 参数扫描结果

| 推理模式 | Recall | Precision | Missed | FP |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.8883 | 0.9036 | 66 | 56 |
| tile 512 | 0.8917 | 0.8682 | 64 | 80 |
| tile 640 | 0.8900 | 0.8723 | 65 | 77 |
| tile 768 | 0.8917 | 0.8798 | 64 | 72 |
| tile 896 | 0.8917 | 0.8872 | 64 | 67 |

### 6. 平台功能测试

平台功能测试验证了图片检测、批量图片检测、视频文件检测、报告保存和历史记录查看等功能。图片检测中，系统能够在网页显示原图和结果图；批量检测中，系统能够统计本次任务图片数量和缺陷总数；视频检测中，系统能够生成原视频预览、检测结果视频和关键帧截图。测试说明系统已经从单纯模型训练扩展为可运行、可演示、可保存结果的本地检测平台。

腐蚀样例测试中，上传 corrosion_112.jpg 后，系统检测到 1 个 corrosion 缺陷，置信度约为 0.895。由于 corrosion 类外观变化较大、与背景纹理容易混淆，该样例可作为后续优化 corrosion 数据质量和标注一致性的参考。

![平台腐蚀检测案例](figures_cases/case_corrosion_112_pred.jpg)

图 5-2 平台腐蚀检测结果图

### 7. 多模态推理结果分析

多模态推理测试表明，系统能够将检测结果进一步转化为风险和维修建议。对于 crack_388.jpg，系统根据裂纹类别和高置信度检测结果给出中高风险判断，并建议复核裂纹长度、方向和扩展趋势，必要时评估降载或停机检查。对于 corrosion_112.jpg，系统给出低风险判断，但建议确认腐蚀面积、深度和防护层失效范围，并进行防腐处理和归档跟踪。

表 5-5 多模态推理样例结果

| 样例 | 检测类别 | 置信度 | 风险/优先级 | 推理输出重点 |
| --- | --- | ---: | --- | --- |
| crack_388.jpg | crack | 0.968 | 中高风险/P0 | 复核裂纹长度、方向和端点，必要时补拍高清图并评估降载或停机 |
| corrosion_112.jpg | corrosion | 0.895 | 低风险/P3 | 确认腐蚀面积和深度，检查防护层失效范围，建议防腐处理并跟踪 |

该结果体现了多模态推理模块的工程价值：检测模型负责给出视觉证据，多模态推理负责将视觉证据转化为巡检语言和维修任务。由于风险等级先由规则引擎约束，系统不会完全依赖语言模型自由判断，降低了“凭空编造”风险。后续若接入真实视觉语言模型，也应保持检测证据约束和人工复核机制。

## 六、总结与展望

### 1. 工作总结

本文围绕风机叶片智能巡检任务，设计并实现了一套 MSF-YOLO 多尺度形态感知检测与多模态推理系统。算法方面，系统实现了 P2 小目标检测头、Shape-Aware Loss、困难样本重加权和 Blade-Slice 切片推理等模块；软件方面，平台实现了图片检测、批量图片检测、视频文件检测、报告保存、历史记录、多模态分析和维修工单生成；工程方面，系统已经具备本地运行、网页演示和实验复现能力。

实验结果表明，当前 V4 稳定基线仍是总体指标最优的部署模型，MSF-YOLO 候选模型已经完成完整方法链路，并在部分小目标评估中体现出一定收益，但尚未形成全面超过强基线的结论。本文没有夸大实验结果，而是将检测算法、平台功能和多模态推理闭环作为整体贡献进行呈现。

### 2. 不足分析

当前系统仍存在四方面不足。第一，corrosion 类别检测稳定性不足，简单重加权没有带来稳定提升，需要进一步分析标注质量、外观边界和场景差异。第二，Blade-Slice 默认参数会增加误检，需要增加置信度门控、类别门控和触发条件。第三，当前系统没有摄像头实时采集功能，输入方式仍是图片和视频文件上传。第四，多模态分析默认仍以规则和 mock 模板为主，真实视觉语言模型和运维知识库接入还需要进一步完善。

### 3. 后续展望

后续工作可从四个方向推进。首先，继续整理 hard_cases 和 corrosion 失败样本，提升数据质量和标注一致性。其次，优化 P2 结构训练策略和 Shape-Aware Loss 参数，重点验证其在小目标子集上的收益。第三，为 Blade-Slice 增加更严格的门控机制，使其成为高分辨率复核工具。第四，接入真实视觉语言模型和运维知识库，同时保留规则引擎的安全边界，使系统能够输出更贴近实际巡检业务的故障解释和维修建议。

## 参考文献

[1] Zhao Y., Lv W., Xu S., et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR, 2024.

[2] Wang C.-Y., Yeh I.-H., Liao H.-Y. M. YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. ECCV, 2024.

[3] Wang A., Chen H., Liu L., et al. YOLOv10: Real-Time End-to-End Object Detection. NeurIPS, 2024.

[4] Chen Z., Wang W., Cao Y., et al. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. CVPR, 2024.

[5] Yue X., Ni Y., Zhang K., et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. CVPR, 2024.

[6] Jia X., Chen X. Unsupervised Wind Turbine Blade Damage Detection With Memory-Aided Denoising Reconstruction. IEEE Transactions on Industrial Informatics, 2025.

[7] Zhang Y., Wang L., Huang C., Luo X. Wind Turbine Blade Defect Detection Based on the Genetic Algorithm-Enhanced YOLOv5 Algorithm Using Synthetic Data. IEEE Transactions on Industry Applications, 2025.

[8] Li W., Zhao W., Du Y. Large-scale Wind Turbine Blade Operational Condition Monitoring Based on UAV and Improved YOLOv5 Deep Learning Model. Mechanical Systems and Signal Processing, 2025.

[9] Ye X., Wang L., Huang C., Luo X. Wind Turbine Blade Defect Detection With a Semi-Supervised Deep Learning Framework. Engineering Applications of Artificial Intelligence, 2024.

