from __future__ import annotations

import zipfile
from pathlib import Path

import generate_msf_yolo_formal_paper_docx as formal
from generate_msf_yolo_award_docx import (
    content_types_xml,
    package_rels_xml,
    settings_xml,
    styles_xml,
)


OUT_DIR = Path("paper_assets_msf_yolo")
PNG_DIR = OUT_DIR / "figures_png"
DOCX_PATH = OUT_DIR / "MSF-YOLO_正式论文预稿_全文降模板润色版_手机兼容PNG版.docx"

FIGURES_PNG = [
    ("rId60", "fig_color_5_msf_yolo_system_pipeline.png"),
    ("rId61", "fig_1_msf_yolo_framework.png"),
    ("rId62", "fig_color_0_msf_yolo_precise_network.png"),
    ("rId63", "fig_color_2_p2_head_structure.png"),
    ("rId64", "fig_color_3_shape_aware_formula.png"),
    ("rId65", "fig_2_shape_aware_loss.png"),
    ("rId66", "fig_color_4_blade_slice_formula.png"),
    ("rId67", "fig_3_blade_slice_pipeline.png"),
]


def content_types_png_xml() -> str:
    xml = content_types_xml()
    if '<Default Extension="png" ContentType="image/png"/>' in xml:
        return xml
    return xml.replace(
        '<Default Extension="svg" ContentType="image/svg+xml"/>',
        '<Default Extension="svg" ContentType="image/svg+xml"/>\n'
        '  <Default Extension="png" ContentType="image/png"/>',
    )


def document_rels_png_xml() -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
    ]
    for rid, name in FIGURES_PNG:
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
    required = [PNG_DIR / name for _, name in FIGURES_PNG]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PNG assets: " + ", ".join(missing))

    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_png_xml())
        z.writestr("_rels/.rels", package_rels_xml())
        z.writestr("word/document.xml", formal.document_xml())
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/settings.xml", settings_xml())
        z.writestr("word/_rels/document.xml.rels", document_rels_png_xml())
        for png in required:
            z.write(png, f"word/media/{png.name}")
    print(DOCX_PATH)


if __name__ == "__main__":
    build_docx()
