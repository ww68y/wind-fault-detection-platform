from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT_DIR = Path(__file__).resolve().parent / "figures_system"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw, text, fnt, max_width):
    lines = []
    for raw in text.split("\n"):
        line = ""
        for ch in raw:
            candidate = line + ch
            if text_size(draw, candidate, fnt)[0] <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = ch
        lines.append(line)
    return lines


def rounded(draw, box, fill, outline, width=3, radius=12):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, box, text, fnt, fill="#111827", max_width=None, line_gap=7):
    if max_width is None:
        max_width = box[2] - box[0] - 30
    lines = wrap(draw, text, fnt, max_width)
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total = sum(heights) + line_gap * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total) / 2
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        x = box[0] + (box[2] - box[0] - w) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + line_gap


def draw_box(draw, box, title, body, fill, outline, title_size=28, body_size=22):
    rounded(draw, box, fill, outline)
    tf = font(title_size, True)
    bf = font(body_size, False)
    title_area = (box[0], box[1] + 14, box[2], box[1] + 58)
    centered(draw, title_area, title, tf, max_width=box[2] - box[0] - 28)
    draw.line((box[0] + 24, box[1] + 70, box[2] - 24, box[1] + 70), fill=outline, width=2)
    body_area = (box[0] + 12, box[1] + 82, box[2] - 12, box[3] - 14)
    centered(draw, body_area, body, bf, fill="#1f2937", max_width=box[2] - box[0] - 36, line_gap=6)


def arrow(draw, start, end, color="#334155", width=4):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)


PALETTE = {
    "blue": ("#e8f3ff", "#2563eb"),
    "green": ("#eef7ed", "#2f7d32"),
    "orange": ("#fff4dc", "#b7791f"),
    "purple": ("#f4ecff", "#6b46a5"),
    "red": ("#fff1f2", "#b91c1c"),
    "gray": ("#f8fafc", "#475569"),
    "line": "#334155",
    "soft": "#94a3b8",
}


def save(img, name):
    path = OUT_DIR / name
    img.save(path, quality=95)
    print(path)


def option1_vertical_flow():
    w, h = 1500, 1780
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    rounded(d, (55, 45, w - 55, h - 45), "#ffffff", "#cbd5e1", width=2)

    boxes = [
        ((170, 90, 930, 250), "数据输入", "风机叶片图片 / 视频 / 摄像头画面\n无人机多模态采集：可见光、红外、激光雷达\n1 mm × 3 mm 细节拍摄；±5 cm 动态避障", *PALETTE["blue"]),
        ((170, 350, 930, 510), "YOLO 缺陷检测模型", "基于 YOLOv8 改进模型\n支持裂纹、孔洞、剥落、腐蚀多尺度定位\n热力梯度约束提升微裂纹检出", *PALETTE["orange"]),
        ((170, 610, 930, 830), "结构化检测结果", "缺陷类别：crack / hole / spalling / corrosion\n置信度、检测框坐标、缺陷数量\n风险等级、检测结果图、结构化检测报告", *PALETTE["gray"]),
        ((170, 960, 930, 1140), "多模态推理模块", "输入：原始图像、YOLO 结果图、检测报告\n缺陷类别、置信度、位置、风险等级", *PALETTE["purple"]),
        ((170, 1260, 930, 1480), "推理输出", "故障现象描述、可能原因分析\n风险影响判断、运维处理建议\n复检要求、是否建议停机", *PALETTE["red"]),
        ((170, 1580, 930, 1700), "本地网页展示", "检测结果 + 多模态分析报告", *PALETTE["green"]),
    ]
    for box, title, body, fill, outline in boxes:
        draw_box(d, box, title, body, fill, outline)
    for i in range(len(boxes) - 1):
        b1 = boxes[i][0]
        b2 = boxes[i + 1][0]
        arrow(d, ((b1[0] + b1[2]) // 2, b1[3]), ((b2[0] + b2[2]) // 2, b2[1]), PALETTE["line"])

    kb = (1005, 920, 1360, 1165)
    draw_box(d, kb, "知识推理支撑", "多模态知识库\n缺陷描述 / 叶片结构\n维护日志 / 参考图像\nRAG + Qwen-VL-Max", *PALETTE["purple"], title_size=26, body_size=21)
    arrow(d, (1005, 1045), (930, 1045), PALETTE["soft"], width=3)
    save(img, "framework_option1_vertical_flow.png")


def option2_layered_architecture():
    w, h = 1900, 1080
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    rounded(d, (45, 45, w - 45, h - 45), "#ffffff", "#cbd5e1", width=2)

    top = [
        ((85, 130, 360, 345), "数据采集层", "图片 / 视频 / 摄像头画面\n可见光相机、红外热像仪、激光雷达\n高精度拍摄与动态避障", *PALETTE["blue"]),
        ((435, 130, 710, 345), "缺陷检测层", "YOLOv8 改进模型\n多尺度缺陷定位\n热力梯度约束提升微裂纹检出", *PALETTE["orange"]),
        ((785, 130, 1060, 345), "结果组织层", "缺陷类别、置信度\n检测框坐标、缺陷数量\n风险等级、结果图", *PALETTE["gray"]),
        ((1135, 130, 1410, 345), "知识推理层", "多模态知识库\nRAG 检索增强\nQwen-VL-Max 推理", *PALETTE["purple"]),
        ((1485, 130, 1760, 345), "决策输出层", "故障描述、原因分析\n风险影响、处理建议\n复检要求、停机建议", *PALETTE["red"]),
    ]
    for box, title, body, fill, outline in top:
        draw_box(d, box, title, body, fill, outline)
    for i in range(len(top) - 1):
        b1, b2 = top[i][0], top[i + 1][0]
        arrow(d, (b1[2], (b1[1] + b1[3]) // 2), (b2[0], (b2[1] + b2[3]) // 2))

    bottom = [
        ((85, 585, 360, 790), "当前平台输入", "上传图片\n批量图片\n上传视频文件", *PALETTE["blue"]),
        ((435, 585, 710, 790), "后端服务", "文件接收\n任务管理\n模型调用", *PALETTE["green"]),
        ((785, 585, 1060, 790), "结构化报告", "检测结果图\n检测报告 JSON\n历史记录", *PALETTE["gray"]),
        ((1135, 585, 1410, 790), "分析依据", "原始图像、结果图\n类别、置信度、位置\n风险等级", *PALETTE["purple"]),
        ((1485, 585, 1760, 790), "网页端展示", "检测结果\n多模态分析报告\n维修工单", *PALETTE["green"]),
    ]
    for box, title, body, fill, outline in bottom:
        draw_box(d, box, title, body, fill, outline)
    for i in range(len(bottom) - 1):
        b1, b2 = bottom[i][0], bottom[i + 1][0]
        arrow(d, (b1[2], (b1[1] + b1[3]) // 2), (b2[0], (b2[1] + b2[3]) // 2), PALETTE["soft"], width=3)

    for top_box, bottom_box in zip(top, bottom):
        arrow(d, ((bottom_box[0][0] + bottom_box[0][2]) // 2, bottom_box[0][1]),
              ((top_box[0][0] + top_box[0][2]) // 2, top_box[0][3]), PALETTE["soft"], width=3)

    # bottom process line
    steps = [("采集", 220), ("检测", 575), ("报告", 925), ("推理", 1275), ("展示", 1625)]
    y = 925
    for i, (label, x) in enumerate(steps):
        d.ellipse((x - 13, y - 13, x + 13, y + 13), outline=PALETTE["line"], width=3, fill="white")
        tw, _ = text_size(d, label, font(20))
        d.text((x - tw / 2, y + 28), label, font=font(20), fill=PALETTE["line"])
        if i < len(steps) - 1:
            arrow(d, (x + 16, y), (steps[i + 1][1] - 16, y), PALETTE["soft"], width=3)
    save(img, "framework_option2_layered_architecture.png")


def option3_closed_loop():
    w, h = 1700, 1300
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    rounded(d, (45, 45, w - 45, h - 45), "#ffffff", "#cbd5e1", width=2)

    center = (590, 500, 1110, 760)
    draw_box(d, center, "多模态推理与运维决策中心", "融合视觉证据、结构化检测结果、领域知识和历史维护信息\n输出故障解释、风险判断和处置建议", *PALETTE["purple"], title_size=27, body_size=22)

    nodes = [
        ((100, 130, 440, 330), "数据采集", "图片 / 视频 / 摄像头画面\n多模态传感器采集", *PALETTE["blue"]),
        ((680, 110, 1020, 310), "YOLO 检测", "缺陷类别、置信度\n检测框与数量", *PALETTE["orange"]),
        ((1260, 130, 1600, 330), "结构化结果", "风险等级\n检测结果图\n检测报告 JSON", *PALETTE["gray"]),
        ((1260, 915, 1600, 1115), "网页展示", "检测结果\n多模态分析报告\n维修工单", *PALETTE["green"]),
        ((680, 970, 1020, 1170), "维护反馈", "复检结果\n维修记录\n历史日志更新", *PALETTE["red"]),
        ((100, 915, 440, 1115), "知识库更新", "缺陷描述\n叶片结构\n参考图像", *PALETTE["purple"]),
    ]
    for box, title, body, fill, outline in nodes:
        draw_box(d, box, title, body, fill, outline)

    # clean outer-loop arrows
    arrow(d, (440, 230), (680, 210), PALETTE["soft"], width=3)
    arrow(d, (1020, 210), (1260, 230), PALETTE["soft"], width=3)
    arrow(d, (1430, 330), (1430, 915), PALETTE["soft"], width=3)
    arrow(d, (1260, 1015), (1020, 1070), PALETTE["soft"], width=3)
    arrow(d, (680, 1070), (440, 1015), PALETTE["soft"], width=3)
    arrow(d, (270, 915), (270, 330), PALETTE["soft"], width=3)

    # evidence paths into the center, drawn short to avoid crossing text.
    arrow(d, (850, 310), (850, 500), "#64748b", width=3)
    arrow(d, (1260, 260), (1110, 565), "#64748b", width=3)
    arrow(d, (440, 1015), (590, 695), "#64748b", width=3)
    arrow(d, (1260, 1015), (1110, 695), "#64748b", width=3)

    save(img, "framework_option3_closed_loop.png")


def option4_dual_channel_fusion():
    w, h = 1900, 1080
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    rounded(d, (45, 45, w - 45, h - 45), "#ffffff", "#cbd5e1", width=2)

    # visual branch
    visual = [
        ((90, 135, 375, 325), "视觉输入", "叶片图片 / 视频\n摄像头画面\n无人机多模态采集", *PALETTE["blue"]),
        ((455, 135, 740, 325), "YOLOv8 改进检测", "多尺度缺陷定位\ncrack / hole / spalling / corrosion\n热力梯度约束", *PALETTE["orange"]),
        ((820, 135, 1105, 325), "结构化检测结果", "类别、置信度、位置\n检测框坐标、缺陷数量\n风险等级、结果图", *PALETTE["gray"]),
    ]
    for box, title, body, fill, outline in visual:
        draw_box(d, box, title, body, fill, outline)
    for i in range(len(visual) - 1):
        b1, b2 = visual[i][0], visual[i + 1][0]
        arrow(d, (b1[2], (b1[1] + b1[3]) // 2), (b2[0], (b2[1] + b2[3]) // 2))

    # knowledge branch
    knowledge = [
        ((90, 595, 375, 785), "知识来源", "缺陷描述\n叶片结构\n维护日志 / 参考图像", *PALETTE["purple"]),
        ((455, 595, 740, 785), "RAG 检索增强", "领域知识检索\n检测证据对齐\n提示词组织", *PALETTE["purple"]),
        ((820, 595, 1105, 785), "视觉语言模型", "Qwen-VL-Max 等模型\n图文联合推理\n知识动态注入", *PALETTE["purple"]),
    ]
    for box, title, body, fill, outline in knowledge:
        draw_box(d, box, title, body, fill, outline)
    for i in range(len(knowledge) - 1):
        b1, b2 = knowledge[i][0], knowledge[i + 1][0]
        arrow(d, (b1[2], (b1[1] + b1[3]) // 2), (b2[0], (b2[1] + b2[3]) // 2), PALETTE["soft"], width=3)

    fusion = (1245, 345, 1535, 575)
    draw_box(d, fusion, "多模态推理模块", "原始图像、结果图\n检测报告 JSON\n缺陷类别、置信度、位置、风险等级", *PALETTE["green"], body_size=21)
    arrow(d, (1105, 230), (1245, 420), PALETTE["line"])
    arrow(d, (1105, 690), (1245, 500), PALETTE["line"])

    output = (1600, 365, 1830, 555)
    draw_box(d, output, "本地网页展示", "检测结果\n多模态分析报告\n运维处理建议", *PALETTE["red"])
    arrow(d, (1535, 460), (1600, 460), PALETTE["line"])

    decision = (1245, 705, 1830, 900)
    draw_box(d, decision, "决策输出", "故障现象描述、可能原因分析、风险影响判断\n运维处理建议、复检要求、是否建议停机\n缺陷类型、位置、严重程度及维修优先级", *PALETTE["red"], body_size=21)
    arrow(d, ((fusion[0] + fusion[2]) // 2, fusion[3]), ((decision[0] + decision[2]) // 2, decision[1]), PALETTE["line"])
    arrow(d, ((decision[0] + decision[2]) // 2, decision[1]), ((output[0] + output[2]) // 2, output[3]), PALETTE["soft"], width=3)

    # branch labels
    d.text((90, 80), "视觉检测通道", font=font(28, True), fill="#1d4ed8")
    d.text((90, 540), "知识推理通道", font=font(28, True), fill="#6b46a5")
    save(img, "framework_option4_dual_channel_fusion.png")


if __name__ == "__main__":
    option1_vertical_flow()
    option2_layered_architecture()
    option3_closed_loop()
    option4_dual_channel_fusion()
