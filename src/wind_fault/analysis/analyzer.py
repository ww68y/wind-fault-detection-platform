from __future__ import annotations

from pathlib import Path
from typing import Any

from .analysis_report import save_analysis_report
from .mock_vlm_analyzer import generate_mock_analysis
from .prompt_builder import build_prompt
from .report_loader import flatten_detections
from .risk_engine import assess_risk
from .vlm_client import generate_real_vlm_analysis


def _collect_vlm_images(report: dict[str, Any], work_dir: Path, max_images: int = 4) -> list[Path]:
    images: list[Path] = []
    mode = str(report.get("mode") or "single")

    if mode == "video":
        keyframes = sorted(
            report.get("keyframes", []) or [],
            key=lambda item: (
                -int(item.get("defect_count", 0) or 0),
                -float(item.get("max_confidence", 0.0) or 0.0),
                int(item.get("frame_index", 0) or 0),
            ),
        )
        selected = sorted(keyframes[:max_images], key=lambda item: int(item.get("frame_index", 0) or 0))
        for item in selected:
            path = Path(str(item.get("image_path") or ""))
            if path.exists():
                images.append(path)
        return images

    rendered_dir = work_dir / "rendered"
    for item in report.get("items", []) or []:
        image_path = Path(str(item.get("image") or ""))
        if image_path.exists():
            images.append(image_path)
        rendered_path = rendered_dir / f"{image_path.stem}_pred.jpg"
        if rendered_path.exists():
            images.append(rendered_path)
        if len(images) >= max_images:
            break
    return images[:max_images]


def analyze_detection_report(report: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    from wind_fault.maintenance import build_maintenance_order

    detections = flatten_detections(report)
    risk = assess_risk(detections).to_dict()
    prompt = build_prompt(report, risk)
    analysis = generate_mock_analysis(
        report=report,
        detections=detections,
        risk_result=risk,
        prompt=prompt,
    )
    analysis = generate_real_vlm_analysis(
        prompt=prompt,
        image_paths=_collect_vlm_images(report, work_dir),
        fallback=analysis,
    )
    analysis["maintenance_order"] = build_maintenance_order(report, analysis)
    return save_analysis_report(analysis, work_dir / "reports")
