from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT_DIR = Path(__file__).resolve().parent / "figures_system"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "system_overall_framework_v5_extended.png"

W, H = 1900, 920
img = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(img)

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
font_head = ImageFont.truetype(FONT_BOLD, 27)
font_body = ImageFont.truetype(FONT_REG, 22)
font_small = ImageFont.truetype(FONT_REG, 19)

COLORS = {
    "line": "#334155",
    "soft": "#94a3b8",
    "collect": ("#e8f3ff", "#2563eb"),
    "backend": ("#eef7ed", "#2f7d32"),
    "detect": ("#fff4dc", "#b7791f"),
    "reason": ("#f4ecff", "#6b46a5"),
    "decision": ("#fff1f2", "#b91c1c"),
    "output": ("#f8fafc", "#475569"),
}


def rounded_rect(box, fill, outline, width=3, radius=8):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_size(text, font):
    bbox = d.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def centered_text(box, lines, font, fill="#111827", line_gap=7):
    if isinstance(lines, str):
        lines = lines.split("\n")
    heights = [text_size(line, font)[1] for line in lines]
    total = sum(heights) + line_gap * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total) / 2
    for line, h in zip(lines, heights):
        w, _ = text_size(line, font)
        x = box[0] + (box[2] - box[0] - w) / 2
        d.text((x, y), line, font=font, fill=fill)
        y += h + line_gap


def arrow(start, end, color="#334155", width=4):
    x1, y1 = start
    x2, y2 = end
    d.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 15
    p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    d.polygon([end, p1, p2], fill=color)


rounded_rect((45, 35, W - 45, H - 35), "#ffffff", "#cbd5e1", width=2, radius=8)

main_boxes = [
    ("数据采集层", "无人机平台\n可见光 / 红外 / 激光雷达\n1 mm × 3 mm 细节拍摄", (80, 130, 345, 335), *COLORS["collect"]),
    ("后端服务层", "文件接收\n任务管理\n数据缓存与模型调度", (410, 130, 675, 335), *COLORS["backend"]),
    ("缺陷检测层", "YOLOv8 改进模型\n多尺度缺陷定位\n热力梯度约束", (740, 130, 1005, 335), *COLORS["detect"]),
    ("知识推理层", "多模态知识库\nRAG 检索增强\n视觉语言模型推理", (1070, 130, 1335, 335), *COLORS["reason"]),
    ("决策输出层", "缺陷类型 / 位置\n严重程度\n维修优先级", (1400, 130, 1665, 335), *COLORS["decision"]),
]

for title, body, box, fill, outline in main_boxes:
    rounded_rect(box, fill, outline)
    centered_text((box[0], box[1] + 18, box[2], box[1] + 58), title, font_head)
    d.line((box[0] + 24, box[1] + 72, box[2] - 24, box[1] + 72), fill=outline, width=2)
    centered_text((box[0], box[1] + 82, box[2], box[3] - 18), body, font_body, fill="#1f2937")

for i in range(len(main_boxes) - 1):
    b1 = main_boxes[i][2]
    b2 = main_boxes[i + 1][2]
    arrow((b1[2], (b1[1] + b1[3]) // 2), (b2[0], (b2[1] + b2[3]) // 2))

support_boxes = [
    ("当前平台输入", "图片检测\n批量检测\n视频文件检测", (80, 535, 345, 710), *COLORS["output"]),
    ("网页端展示", "检测状态 / 缺陷数量\n原图 / 结果图\n报告查看 / 历史记录", (410, 535, 675, 710), *COLORS["output"]),
    ("检测证据", "结果图、关键帧\n类别、置信度、检测框\n缺陷数量", (740, 535, 1005, 710), *COLORS["detect"]),
    ("知识来源", "缺陷描述\n叶片结构\n维护日志 / 参考图像", (1070, 535, 1335, 710), *COLORS["reason"]),
    ("运维建议", "风险等级\n复检要求\n维修工单", (1400, 535, 1665, 710), *COLORS["decision"]),
]

for title, body, box, fill, outline in support_boxes:
    rounded_rect(box, fill, outline, width=2)
    centered_text((box[0], box[1] + 16, box[2], box[1] + 54), title, font_head)
    d.line((box[0] + 24, box[1] + 66, box[2] - 24, box[1] + 66), fill=outline, width=2)
    centered_text((box[0], box[1] + 78, box[2], box[3] - 15), body, font_body, fill="#1f2937")

arrow((212, 535), (212, 335), color=COLORS["soft"], width=3)
arrow((542, 335), (542, 535), color=COLORS["soft"], width=3)
arrow((872, 335), (872, 535), color=COLORS["soft"], width=3)
arrow((1202, 535), (1202, 335), color=COLORS["soft"], width=3)
arrow((1532, 335), (1532, 535), color=COLORS["soft"], width=3)

# Evidence and knowledge feed the reasoning layer.
arrow((1005, 622), (1070, 250), color=COLORS["soft"], width=3)
arrow((1335, 622), (1202, 335), color=COLORS["soft"], width=3)

steps = [
    ("多模态采集", 210),
    ("任务接收", 540),
    ("模型检测", 870),
    ("证据检索", 1200),
    ("决策生成", 1530),
]
y = 810
for idx, (label, x) in enumerate(steps):
    d.ellipse((x - 13, y - 13, x + 13, y + 13), fill="#ffffff", outline="#334155", width=3)
    w, _ = text_size(label, font_small)
    d.text((x - w / 2, y + 26), label, font=font_small, fill="#334155")
    if idx < len(steps) - 1:
        arrow((x + 16, y), (steps[idx + 1][1] - 16, y), color="#94a3b8", width=3)

img.save(OUT, quality=95)
print(OUT)
