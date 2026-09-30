from __future__ import annotations

from typing import Any

from .report_loader import CLASS_LABELS


CAUSES = {
    "crack": ["长期疲劳载荷", "局部应力集中", "材料老化或外物冲击损伤"],
    "hole": ["外物冲击", "材料局部破损", "制造缺陷或长期磨损"],
    "spalling": ["表面涂层脱落", "局部结构损伤", "长期运行导致表面材料剥离"],
    "corrosion": ["潮湿或盐雾环境影响", "防护层失效", "长期环境侵蚀"],
}

ADVICE = {
    "crack": ["人工复核裂纹长度、方向和扩展趋势", "必要时降低负载或停机检查", "维修后上传复检图片"],
    "hole": ["检查孔洞深度和边缘扩展情况", "判断是否影响叶片结构强度", "必要时进行修补或更换"],
    "spalling": ["检查剥落面积和范围", "判断是否存在继续扩展", "进行局部修补和表面防护处理"],
    "corrosion": ["检查腐蚀范围", "判断是否已经影响材料强度", "进行除锈、防腐和涂层修复"],
}

RISK_IMPACT = {
    "正常": "当前未发现明显缺陷，可作为本次巡检的正常记录保存。",
    "待复核": "检测结果存在不确定性，建议人工复核后再形成正式运维结论。",
    "低风险": "当前缺陷风险较低，但仍建议记录位置并在后续巡检中跟踪变化。",
    "中风险": "缺陷可能继续扩展，建议安排人工复核并结合现场情况制定维修计划。",
    "中高风险": "缺陷可能影响叶片局部结构或气动性能，建议尽快复核并评估是否降载运行。",
    "高风险": "缺陷数量、类型或置信度显示风险较高，建议优先人工复核并评估停机检修。"
}


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _format_class_summary(class_summary_cn: dict[str, int]) -> str:
    if not class_summary_cn:
        return "未检测到明确缺陷"
    return "、".join(f"{name} {count} 个" for name, count in class_summary_cn.items())


def _shutdown_suggestion(risk_level: str) -> str:
    if risk_level == "高风险":
        return "建议尽快安排人工复核，并结合现场情况评估是否停机检修。"
    if risk_level == "中高风险":
        return "建议结合缺陷位置、范围和运行工况判断是否需要降载或停机检查。"
    if risk_level in {"中风险", "待复核"}:
        return "暂不直接建议停机，但需要人工复核后再决定后续处置。"
    return "当前不建议停机，按常规巡检和记录跟踪处理。"


def generate_mock_analysis(
    report: dict[str, Any],
    detections: list[dict[str, Any]],
    risk_result: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    class_summary = risk_result.get("class_summary", {}) or {}
    class_summary_cn = risk_result.get("class_summary_cn", {}) or {}
    risk_level = str(risk_result.get("risk_level") or "正常")
    mode = str(report.get("mode") or "single")
    total = int(risk_result.get("total_defects") or 0)

    causes: list[str] = []
    advice: list[str] = []
    for class_name in class_summary:
        causes.extend(CAUSES.get(class_name, [f"{CLASS_LABELS.get(class_name, class_name)} 需要人工确认具体成因"]))
        advice.extend(ADVICE.get(class_name, [f"人工复核 {CLASS_LABELS.get(class_name, class_name)} 的实际位置和范围"]))

    if not detections:
        causes = ["当前图像证据中未检测到明确缺陷"]
        advice = ["保留本次检测记录", "如画面模糊、反光或角度受限，建议补拍关键区域"]

    source_note = "视频关键帧" if mode == "video" else "图片"
    summary = (
        f"本次{source_note}检测共识别到 {total} 个缺陷，"
        f"类别统计为：{_format_class_summary(class_summary_cn)}。"
    )

    return {
        "request_id": report.get("request_id", "-"),
        "mode": mode,
        "risk_level": risk_level,
        "confidence_level": risk_result.get("confidence_level", "-"),
        "fault_summary": summary,
        "fault_description": (
            f"根据 YOLO 检测结果，系统在当前{source_note}中发现叶片表面异常。"
            if detections
            else f"根据 YOLO 检测结果，当前{source_note}未发现明确缺陷。"
        ),
        "possible_causes": _unique(causes),
        "risk_impact": RISK_IMPACT.get(risk_level, RISK_IMPACT["待复核"]),
        "maintenance_advice": _unique(advice),
        "recheck_requirement": "维修或现场确认后，建议上传复检图片或关键帧再次检测，确认缺陷是否消除或扩展。",
        "shutdown_suggestion": _shutdown_suggestion(risk_level),
        "manual_review_notice": "本分析基于图像检测结果自动生成，最终结论需结合现场人工检查。",
        "risk_reasons": risk_result.get("risk_reasons", []),
        "class_summary": class_summary,
        "class_summary_cn": class_summary_cn,
        "total_defects": total,
        "detections": detections,
        "prompt": prompt,
    }
