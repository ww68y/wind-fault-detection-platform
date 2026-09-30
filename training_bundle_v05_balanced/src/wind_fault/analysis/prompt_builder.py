from __future__ import annotations

from typing import Any


CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}


def _class_label(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name)


def _image_detection_lines(report: dict[str, Any], limit: int = 30) -> list[str]:
    lines: list[str] = []
    count = 0
    for item in report.get("items", []) or []:
        source = item.get("uploaded_filename") or item.get("image") or "-"
        for detection in item.get("detections", []) or []:
            class_name = str(detection.get("class_name", "unknown"))
            confidence = float(detection.get("confidence", 0.0))
            lines.append(
                f"- 图片 {source}：{_class_label(class_name)}（{class_name}），"
                f"置信度 {confidence:.3f}，边框 {detection.get('xyxy', [])}"
            )
            count += 1
            if count >= limit:
                lines.append(f"- 其余检测结果省略，完整明细见 detection_report.json。")
                return lines
    return lines or ["- 当前任务未检测到明确缺陷。"]


def _video_detection_lines(report: dict[str, Any], limit: int = 40) -> list[str]:
    lines = [
        f"- 视频名称：{report.get('uploaded_filename') or report.get('video_filename') or '-'}",
        f"- 原始总帧数：{report.get('total_frames', 0)}",
        f"- 实际处理帧数：{report.get('processed_frames', 0)}",
        f"- 缺陷帧数：{report.get('defect_frame_count', 0)}",
        f"- 视频时长：{report.get('duration_seconds', 0)} 秒",
    ]
    frames = report.get("frames", []) or []
    if not frames:
        lines.append("- 当前视频未检测到明确缺陷帧。")
        return lines

    lines.append("- 关键缺陷帧明细：")
    count = 0
    for frame in frames:
        frame_index = frame.get("frame_index")
        time_seconds = frame.get("time_seconds")
        for detection in frame.get("detections", []) or []:
            class_name = str(detection.get("class_name", "unknown"))
            confidence = float(detection.get("confidence", 0.0))
            lines.append(
                f"  - 第 {frame_index} 帧 / {time_seconds} 秒："
                f"{_class_label(class_name)}（{class_name}），"
                f"置信度 {confidence:.3f}，边框 {detection.get('xyxy', [])}"
            )
            count += 1
            if count >= limit:
                lines.append("  - 其余视频帧检测结果省略，完整明细见 detection_report.json。")
                return lines
    return lines


def _keyframe_lines(report: dict[str, Any]) -> list[str]:
    keyframes = report.get("keyframes", []) or []
    if not keyframes:
        return ["- 未生成视频关键帧。"]
    return [
        f"- 第 {item.get('frame_index')} 帧，{item.get('time_seconds')} 秒，"
        f"{item.get('defect_count', 0)} 个缺陷，"
        f"最高置信度 {float(item.get('max_confidence', 0.0)):.3f}"
        for item in keyframes[:8]
    ]


def build_prompt(report: dict[str, Any], risk_result: dict[str, Any]) -> str:
    """Build the VLM prompt from detection report data."""
    mode = str(report.get("mode") or "single")
    if mode == "video":
        mode_note = [
            "这是视频检测任务。你会看到若干检测关键帧截图，文字中包含关键帧帧号、时间、类别、置信度和检测框。",
            "请优先判断缺陷是否在多个相邻帧中连续出现，避免把单帧运动模糊或反光过度解释为严重故障。",
            "",
            "视频关键帧：",
            *_keyframe_lines(report),
            "",
            "视频检测明细：",
            *_video_detection_lines(report),
        ]
    else:
        mode_note = [
            "这是图片检测任务。你会看到原图和检测结果图，文字中包含类别、置信度和检测框。",
            "",
            "图片检测明细：",
            *_image_detection_lines(report),
        ]

    return "\n".join(
        [
            "你是一名风机叶片巡检与运维工程师。",
            "请根据 YOLO 检测结果、原图/关键帧和检测结果图生成故障分析与运维建议。",
            "注意：YOLO 检测结果是主要依据，不要编造未检测到的缺陷。",
            "如果证据不足，请明确说明需要人工复核。",
            "如果检测框落在文字、水印、边缘阴影、反光或背景区域，应提醒人工复核，不要直接下严重结论。",
            "输出必须是 JSON 对象，不要输出 Markdown。",
            "JSON 字段必须包含：fault_summary, fault_description, possible_causes, risk_impact, maintenance_advice, recheck_requirement, shutdown_suggestion, manual_review_notice。",
            "possible_causes 和 maintenance_advice 必须是字符串数组。",
            "",
            f"请求 ID：{report.get('request_id', '-')}",
            f"检测模式：{report.get('mode', '-')}",
            f"缺陷总数：{risk_result.get('total_defects', 0)}",
            f"风险等级：{risk_result.get('risk_level', '-')}",
            f"类别统计：{risk_result.get('class_summary_cn', {})}",
            f"风险判断依据：{risk_result.get('risk_reasons', [])}",
            "",
            *mode_note,
        ]
    )
