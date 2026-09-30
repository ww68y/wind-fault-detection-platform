from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import win32com.client


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
PNG_DIR = ROOT / "figures_png"

FIGURES = [
    "fig_color_5_msf_yolo_system_pipeline.svg",
    "fig_1_msf_yolo_framework.svg",
    "fig_color_0_msf_yolo_precise_network.svg",
    "fig_color_2_p2_head_structure.svg",
    "fig_color_3_shape_aware_formula.svg",
    "fig_2_shape_aware_loss.svg",
    "fig_color_4_blade_slice_formula.svg",
    "fig_3_blade_slice_pipeline.svg",
]


def svg_size(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    width = root.attrib.get("width", "")
    height = root.attrib.get("height", "")

    def parse_px(value: str) -> int | None:
        match = re.match(r"^\s*([0-9.]+)", value)
        return int(float(match.group(1))) if match else None

    w = parse_px(width)
    h = parse_px(height)
    if w and h:
        return w, h

    view_box = root.attrib.get("viewBox", "")
    parts = [float(item) for item in view_box.replace(",", " ").split() if item]
    if len(parts) == 4:
        return int(parts[2]), int(parts[3])
    raise ValueError(f"Cannot determine SVG size: {path}")


def convert() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    presentation = app.Presentations.Add()
    try:
        for name in FIGURES:
            svg_path = FIG_DIR / name
            png_path = PNG_DIR / f"{Path(name).stem}.png"
            width_px, height_px = svg_size(svg_path)
            width_pt = width_px * 0.75
            height_pt = height_px * 0.75
            presentation.PageSetup.SlideWidth = width_pt
            presentation.PageSetup.SlideHeight = height_pt
            slide = presentation.Slides.Add(1, 12)
            slide.FollowMasterBackground = False
            slide.Background.Fill.ForeColor.RGB = 0xFFFFFF
            slide.Shapes.AddPicture(
                FileName=str(svg_path),
                LinkToFile=False,
                SaveWithDocument=True,
                Left=0,
                Top=0,
                Width=width_pt,
                Height=height_pt,
            )
            slide.Export(str(png_path), "PNG", width_px * 2, height_px * 2)
            slide.Delete()
            print(png_path)
    finally:
        presentation.Close()
        app.Quit()


if __name__ == "__main__":
    convert()
