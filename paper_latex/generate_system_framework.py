from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT_DIR = Path(__file__).resolve().parent / "figures_system"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "system_overall_framework_v4.png"

W, H = 1800, 860
img = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(img)


def load_font(path, size):
    return ImageFont.truetype(path, size)


FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
font_head = load_font(FONT_BOLD, 28)
font_body = load_font(FONT_REG, 23)
font_small = load_font(FONT_REG, 20)

COLORS = {
    "line": "#334155",
    "soft_line": "#94a3b8",
    "input": "#e8f3ff",
    "input_border": "#2f6f9f",
    "service": "#eef7ed",
    "service_border": "#397a42",
    "model": "#fff4dc",
    "model_border": "#b7791f",
    "result": "#f1f5f9",
    "result_border": "#475569",
    "analysis": "#f4ecff",
    "analysis_border": "#6b46a5",
    "order": "#fff1f2",
    "order_border": "#b91c1c",
}


def rounded_rect(box, fill, outline, width=3, radius=8):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_size(text, font):
    bbox = d.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(box, lines, font, fill="#111827", line_gap=8):
    if isinstance(lines, str):
        lines = lines.split("\n")
    heights = [text_size(line, font)[1] for line in lines]
    total_height = sum(heights) + line_gap * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total_height) / 2
    for line, height in zip(lines, heights):
        width, _ = text_size(line, font)
        x = box[0] + (box[2] - box[0] - width) / 2
        d.text((x, y), line, font=font, fill=fill)
        y += height + line_gap


def arrow(start, end, color="#334155", width=4):
    x1, y1 = start
    x2, y2 = end
    d.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    d.polygon([end, p1, p2], fill=color)


rounded_rect((45, 30, W - 45, H - 35), "#ffffff", "#cbd5e1", width=2, radius=8)

boxes = [
    ("输入层", "巡检图片\n批量图片\n视频文件", (90, 115, 300, 295), COLORS["input"], COLORS["input_border"]),
    ("前端交互层", "浏览器工作台\n文件上传\n结果查看", (360, 115, 570, 295), COLORS["input"], COLORS["input_border"]),
    ("后端服务层", "任务管理\n文件缓存\n模型调度", (630, 115, 840, 295), COLORS["service"], COLORS["service_border"]),
    ("缺陷检测层", "YOLO 推理\n裂纹 / 孔洞\n剥落 / 腐蚀", (900, 115, 1110, 295), COLORS["model"], COLORS["model_border"]),
    ("结果管理层", "结果图生成\n结构化报告\n历史记录", (1170, 115, 1380, 295), COLORS["result"], COLORS["result_border"]),
    ("多模态推理层", "视觉证据\n风险规则\n故障解释", (1440, 115, 1650, 295), COLORS["analysis"], COLORS["analysis_border"]),
]

for label, body, box, fill, outline in boxes:
    rounded_rect(box, fill, outline)
    draw_centered_text((box[0], box[1] + 18, box[2], box[1] + 58), label, font_head)
    d.line((box[0] + 22, box[1] + 72, box[2] - 22, box[1] + 72), fill=outline, width=2)
    draw_centered_text((box[0], box[1] + 82, box[2], box[3] - 18), body, font_body, fill="#1f2937")

for i in range(len(boxes) - 1):
    b1 = boxes[i][2]
    b2 = boxes[i + 1][2]
    arrow((b1[2], (b1[1] + b1[3]) // 2), (b2[0], (b2[1] + b2[3]) // 2))

order_box = (1440, 480, 1650, 660)
rounded_rect(order_box, COLORS["order"], COLORS["order_border"])
draw_centered_text((order_box[0], order_box[1] + 18, order_box[2], order_box[1] + 58), "运维处置层", font_head)
d.line((order_box[0] + 22, order_box[1] + 72, order_box[2] - 22, order_box[1] + 72), fill=COLORS["order_border"], width=2)
draw_centered_text((order_box[0], order_box[1] + 82, order_box[2], order_box[3] - 18), "风险等级\n复检建议\n维修工单", font_body, fill="#1f2937")
arrow(((boxes[-1][2][0] + boxes[-1][2][2]) // 2, boxes[-1][2][3]), ((order_box[0] + order_box[2]) // 2, order_box[1]))

output_box = (360, 480, 840, 660)
rounded_rect(output_box, "#f8fafc", "#64748b", width=2)
draw_centered_text((output_box[0], output_box[1] + 18, output_box[2], output_box[1] + 58), "网页端输出", font_head)
d.line((output_box[0] + 28, output_box[1] + 72, output_box[2] - 28, output_box[1] + 72), fill="#64748b", width=2)
draw_centered_text(
    (output_box[0], output_box[1] + 82, output_box[2], output_box[3] - 20),
    "检测状态 / 缺陷数量 / 文件信息\n原始图像 / 检测结果图 / 报告查看",
    font_body,
    fill="#1f2937",
)
arrow((1275, 295), (760, 480), color="#64748b", width=3)
arrow((1545, 480), (840, 570), color="#64748b", width=3)

basis_box = (900, 480, 1380, 660)
rounded_rect(basis_box, "#fffbeb", "#b7791f", width=2)
draw_centered_text((basis_box[0], basis_box[1] + 18, basis_box[2], basis_box[1] + 58), "分析依据", font_head)
d.line((basis_box[0] + 28, basis_box[1] + 72, basis_box[2] - 28, basis_box[1] + 72), fill="#b7791f", width=2)
draw_centered_text(
    (basis_box[0], basis_box[1] + 82, basis_box[2], basis_box[3] - 20),
    "原图、结果图、视频关键帧\n缺陷类别、置信度、检测框、数量",
    font_body,
    fill="#1f2937",
)
arrow((1275, 295), (1130, 480), color="#64748b", width=3)
arrow((1130, 480), (1440, 205), color="#64748b", width=3)

entry_box = (90, 480, 300, 660)
rounded_rect(entry_box, "#f8fafc", "#94a3b8", width=2)
draw_centered_text((entry_box[0], entry_box[1] + 18, entry_box[2], entry_box[1] + 58), "数据入口", font_head)
d.line((entry_box[0] + 24, entry_box[1] + 72, entry_box[2] - 24, entry_box[1] + 72), fill="#94a3b8", width=2)
draw_centered_text((entry_box[0], entry_box[1] + 82, entry_box[2], entry_box[3] - 20), "图片检测\n批量检测\n视频检测", font_body, fill="#1f2937")
arrow((195, 480), (195, 295), color="#64748b", width=3)

steps = [
    ("文件接收", 190),
    ("任务创建", 465),
    ("模型推理", 735),
    ("结果保存", 1005),
    ("证据组织", 1275),
    ("风险推理", 1545),
]
y = 755
for idx, (text, x) in enumerate(steps):
    d.ellipse((x - 13, y - 13, x + 13, y + 13), fill="#ffffff", outline="#334155", width=3)
    text_width, _ = text_size(text, font_small)
    d.text((x - text_width / 2, y + 26), text, font=font_small, fill="#334155")
    if idx < len(steps) - 1:
        arrow((x + 16, y), (steps[idx + 1][1] - 16, y), color="#94a3b8", width=3)

img.save(OUT, quality=95)
print(OUT)
