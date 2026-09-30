from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}


def _class_label(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name)


def _class_summary_lines(class_summary: dict[str, int]) -> list[str]:
    if not class_summary:
        return ["- 无"]
    return [
        f"- {_class_label(str(name))}（{name}）：{count}"
        for name, count in class_summary.items()
    ]


def _analysis_source(analysis: dict[str, Any]) -> str:
    if not analysis:
        return "未生成故障分析"
    if analysis.get("vlm_used") is True:
        model = analysis.get("vlm_model") or "VLM"
        return f"视觉语言模型分析（{model}）"
    return "规则兜底分析"


def _image_detection_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in report.get("items", []) or []:
        image_name = item.get("uploaded_filename") or Path(str(item.get("image", "-"))).name
        lines.append(f"### {image_name}")
        lines.append(f"- 缺陷数量：{item.get('defect_count', 0)}")
        detections = item.get("detections", []) or []
        if not detections:
            lines.append("- 未检测到缺陷")
        for detection in detections:
            class_name = str(detection.get("class_name", "unknown"))
            confidence = float(detection.get("confidence", 0.0))
            lines.append(
                f"- {_class_label(class_name)}（{class_name}） | "
                f"置信度 {confidence:.3f} | 边框 {detection.get('xyxy', [])}"
            )
        lines.append("")
    return lines or ["- 无图片明细", ""]


def _video_detection_lines(report: dict[str, Any]) -> list[str]:
    frames = report.get("frames", []) or []
    if not frames:
        return ["- 未检测到缺陷帧", ""]

    lines: list[str] = []
    for frame in frames[:80]:
        lines.append(f"### 第 {frame.get('frame_index')} 帧")
        lines.append(f"- 时间：{frame.get('time_seconds', '-')} 秒")
        lines.append(f"- 缺陷数量：{frame.get('defect_count', 0)}")
        for detection in frame.get("detections", []) or []:
            class_name = str(detection.get("class_name", "unknown"))
            confidence = float(detection.get("confidence", 0.0))
            lines.append(
                f"- {_class_label(class_name)}（{class_name}） | "
                f"置信度 {confidence:.3f} | 边框 {detection.get('xyxy', [])}"
            )
        lines.append("")
    if len(frames) > 80:
        lines.append(f"- 仅展示前 80 个缺陷帧，完整帧级明细请查看 complete_report.json。")
        lines.append("")
    return lines


def _keyframe_lines(report: dict[str, Any]) -> list[str]:
    keyframes = report.get("keyframes", []) or []
    if not keyframes:
        return ["- 未生成关键帧", ""]

    lines: list[str] = []
    for keyframe in keyframes:
        lines.append(
            f"- 第 {keyframe.get('frame_index')} 帧，"
            f"{keyframe.get('time_seconds', '-')} 秒，"
            f"{keyframe.get('defect_count', 0)} 个缺陷，"
            f"最高置信度 {float(keyframe.get('max_confidence', 0.0)):.3f}，"
            f"截图：{keyframe.get('image_path', '-')}"
        )
    lines.append("")
    return lines


def to_markdown(report: dict[str, Any]) -> str:
    mode = str(report.get("mode") or "single")
    is_video = mode == "video"
    analysis = report.get("analysis_report") or {}
    title = "风机叶片视频缺陷检测与故障分析综合报告" if is_video else "风机叶片缺陷检测与故障分析综合报告"
    media_label = "视频名称" if is_video else "图片名称"
    media_name = report.get("uploaded_filename", "-")
    media_count_label = "处理帧数" if is_video else "图片数量"
    media_count = report.get("processed_frames", report.get("image_count", 0)) if is_video else report.get("image_count", 0)

    lines = [
        f"# {title}",
        "",
        "## 基本信息",
        "",
        f"- 请求 ID：{report.get('request_id', '-')}",
        f"- 检测类型：{mode}",
        f"- {media_label}：{media_name}",
        f"- 检测时间：{report.get('created_at', '-')}",
        f"- 置信度阈值：{report.get('confidence_threshold', '-')}",
        f"- {media_count_label}：{media_count}",
        f"- 缺陷总数：{report.get('total_defects', 0)}",
        f"- 风险等级：{analysis.get('risk_level') or report.get('risk_level', '-')}",
        f"- 分析来源：{_analysis_source(analysis)}",
        "",
        "## 一、检测结果概览",
        "",
        str(report.get("advice", "-")),
        "",
        "### 缺陷类别统计",
        "",
        *_class_summary_lines(report.get("class_summary", {}) or {}),
        "",
    ]

    if is_video:
        lines.extend(
            [
                "### 视频检测统计",
                "",
                f"- 原始总帧数：{report.get('total_frames', 0)}",
                f"- 实际处理帧数：{report.get('processed_frames', 0)}",
                f"- 缺陷帧数：{report.get('defect_frame_count', 0)}",
                f"- 视频时长：{report.get('duration_seconds', 0)} 秒",
                f"- FPS：{report.get('fps', 0)}",
                "",
                "## 二、视频关键帧",
                "",
                *_keyframe_lines(report),
                "## 三、帧级缺陷明细",
                "",
                *_video_detection_lines(report),
            ]
        )
    else:
        lines.extend(
            [
                "## 二、图片缺陷明细",
                "",
                *_image_detection_lines(report),
            ]
        )

    lines.extend(
        [
            "## 四、故障分析与运维建议",
            "",
        ]
    )
    if not analysis:
        lines.extend(["尚未生成故障分析。", ""])
    else:
        lines.extend(
            [
                f"- 风险等级：{analysis.get('risk_level', '-')}",
                f"- 置信度判断：{analysis.get('confidence_level', '-')}",
                "",
                "### 故障摘要",
                "",
                str(analysis.get("fault_summary", "-")),
                "",
                "### 故障现象",
                "",
                str(analysis.get("fault_description", "-")),
                "",
                "### 可能原因",
                "",
            ]
        )
        for item in analysis.get("possible_causes", []) or ["-"]:
            lines.append(f"- {item}")
        lines.extend(["", "### 风险影响", "", str(analysis.get("risk_impact", "-")), "", "### 运维建议", ""])
        for item in analysis.get("maintenance_advice", []) or ["-"]:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "### 复检要求",
                "",
                str(analysis.get("recheck_requirement", "-")),
                "",
                "### 是否建议停机",
                "",
                str(analysis.get("shutdown_suggestion", "-")),
                "",
                "### 人工复核提示",
                "",
                str(analysis.get("manual_review_notice", "-")),
                "",
                "### 风险判断依据",
                "",
            ]
        )
        for item in analysis.get("risk_reasons", []) or ["-"]:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 五、报告文件",
            "",
            f"- detection_report.json：{report.get('report_json', '-')}",
            f"- detection_report.md：{report.get('report_markdown', '-')}",
            f"- analysis_report.json：{analysis.get('analysis_report_json', '-') if analysis else '-'}",
            f"- analysis_report.md：{analysis.get('analysis_report_markdown', '-') if analysis else '-'}",
            "",
        ]
    )
    return "\n".join(lines)


def to_json(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": report.get("request_id"),
        "mode": report.get("mode"),
        "created_at": report.get("created_at"),
        "uploaded_filename": report.get("uploaded_filename"),
        "confidence_threshold": report.get("confidence_threshold"),
        "summary": {
            "image_count": report.get("image_count"),
            "total_frames": report.get("total_frames"),
            "processed_frames": report.get("processed_frames"),
            "defect_frame_count": report.get("defect_frame_count"),
            "total_defects": report.get("total_defects"),
            "risk_level": report.get("risk_level"),
            "class_summary": report.get("class_summary", {}),
            "class_summary_cn": report.get("class_summary_cn", {}),
        },
        "detection_report": report,
        "analysis_report": report.get("analysis_report"),
    }


def write_combined_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "complete_report.json"
    md_path = output_dir / "complete_report.md"
    json_path.write_text(json.dumps(to_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {
        "complete_report_json": str(json_path),
        "complete_report_markdown": str(md_path),
    }
