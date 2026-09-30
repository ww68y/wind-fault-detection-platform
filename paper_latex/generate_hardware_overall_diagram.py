from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT_DIR = Path(__file__).resolve().parent / "figures_hardware"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "hardware_overall_camera_pi_display.png"

W, H = 1400, 520
img = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(img)

font_path = r"C:\Windows\Fonts\msyhbd.ttc"
font = ImageFont.truetype(font_path, 42)


def grid():
    for x in range(0, W, 25):
        d.line((x, 0, x, H), fill="#eef2f7", width=1)
    for y in range(0, H, 25):
        d.line((0, y, W, y), fill="#eef2f7", width=1)


def text_size(text):
    box = d.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def box(rect, text):
    d.rectangle(rect, fill="#ffffff", outline="#111827", width=3)
    tw, th = text_size(text)
    x = rect[0] + (rect[2] - rect[0] - tw) / 2
    y = rect[1] + (rect[3] - rect[1] - th) / 2 - 4
    d.text((x, y), text, font=font, fill="#111827")


def arrow(start, end):
    x1, y1 = start
    x2, y2 = end
    d.line((x1, y1, x2, y2), fill="#111827", width=4)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 20
    p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    d.polygon([end, p1, p2], fill="#111827")


grid()

camera = (170, 185, 410, 305)
pi = (580, 185, 820, 305)
display = (990, 185, 1230, 305)

box(camera, "摄像头")
box(pi, "树莓派")
box(display, "显示器")
arrow((410, 245), (580, 245))
arrow((820, 245), (990, 245))

img.save(OUT, quality=95)
print(OUT)
