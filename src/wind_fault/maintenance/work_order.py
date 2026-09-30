from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from wind_fault.analysis.report_loader import CLASS_LABELS, confidence_values, flatten_detections


CLASS_WEIGHT = {
    "crack": 5,
    "hole": 4,
    "spalling": 3,
    "corrosion": 2,
}

CLASS_TEAM = {
    "crack": "叶片结构检修组",
    "hole": "复合材料修补组",
    "spalling": "叶片表面修复组",
    "corrosion": "防腐涂层处理组",
}

CLASS_TASKS = {
    "crack": [
        "复核裂纹长度、方向、端点和是否跨越结构受力区域。",
        "对裂纹区域做近距离高清拍摄，必要时安排无损检测。",
        "评估是否需要降载、停机或进行复合材料结构修补。",
    ],
    "hole": [
        "确认孔洞深度、边缘扩展和是否贯穿表层材料。",
        "清理破损区域并评估局部强度影响。",
        "按复合材料修补工艺进行补强或提交更换评估。",
    ],
    "spalling": [
        "复核剥落面积、边缘松动和周边涂层附着情况。",
        "清洁并打磨剥落区域，处理松动涂层。",
        "完成局部修补、封闭和表面防护处理。",
    ],
    "corrosion": [
        "确认腐蚀面积、深度和是否影响材料强度。",
        "清理腐蚀区域，检查防护层失效范围。",
        "进行除锈、防腐底涂和表面涂层恢复。",
    ],
}

CLASS_MATERIALS = {
    "crack": ["裂纹测量标尺", "无损检测工具", "复合材料修补包", "高清复检相机"],
    "hole": ["复合材料补片", "树脂/胶黏剂", "打磨工具", "边缘探针"],
    "spalling": ["表面清洁剂", "打磨工具", "涂层修补材料", "封闭涂料"],
    "corrosion": ["除锈工具", "防腐底涂", "表面涂层材料", "腐蚀深度检查工具"],
}

PRIORITY_SLA = {
    "P0 紧急": 6,
    "P1 高": 24,
    "P2 中": 72,
    "P3 复核": 120,
    "P4 观察": 720,
}


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _class_counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(item.get("class_name") or "unknown") for item in detections)
    return dict(sorted(counter.items()))


def _risk_rank(risk_level: str) -> int:
    normalized = (risk_level or "").strip()
    aliases = {
        "正常": 0,
        "待复核": 1,
        "低": 2,
        "低风险": 2,
        "中": 3,
        "中风险": 3,
        "中高": 4,
        "中高风险": 4,
        "高": 5,
        "高风险": 5,
    }
    return aliases.get(normalized, 1 if normalized else 0)


def _primary_class(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(
        counts,
        key=lambda name: (CLASS_WEIGHT.get(name, 1), counts[name], name),
    )


def _priority(
    *,
    risk_level: str,
    counts: dict[str, int],
    total_defects: int,
    min_confidence: float,
    max_confidence: float,
) -> str:
    if total_defects == 0:
        return "P4 观察"

    rank = _risk_rank(risk_level)
    if "crack" in counts and (rank >= 4 or max_confidence >= 0.9):
        return "P0 紧急"
    if rank >= 5 or total_defects >= 5:
        return "P0 紧急"
    if rank >= 4 or "hole" in counts:
        return "P1 高"
    if rank >= 3 or len(counts) >= 2:
        return "P2 中"
    if min_confidence and min_confidence < 0.6:
        return "P3 复核"
    return "P3 复核"


def _status(priority: str, total_defects: int, min_confidence: float) -> str:
    if total_defects == 0:
        return "无需派单"
    if priority == "P0 紧急":
        return "待立即派单"
    if min_confidence and min_confidence < 0.6:
        return "待人工复核"
    return "待派单"


def _due_at(created_at: Any, hours: int) -> str | None:
    if not created_at:
        return None
    try:
        base = datetime.fromisoformat(str(created_at))
    except ValueError:
        return None
    return (base + timedelta(hours=hours)).isoformat(timespec="seconds")


def _assigned_team(primary_class: str | None, total_defects: int, min_confidence: float) -> str:
    if total_defects == 0:
        return "巡检值班组"
    if min_confidence and min_confidence < 0.6:
        return "巡检复核组"
    return CLASS_TEAM.get(primary_class or "", "巡检复核组")


def _required_roles(counts: dict[str, int], priority: str, total_defects: int) -> list[str]:
    if total_defects == 0:
        return ["巡检记录员"]

    roles = ["巡检复核工程师"]
    if any(name in counts for name in ("crack", "hole")):
        roles.extend(["叶片结构工程师", "高空作业人员"])
    if any(name in counts for name in ("spalling", "corrosion")):
        roles.append("表面修复技师")
    if priority in {"P0 紧急", "P1 高"}:
        roles.append("安全负责人")
    return _unique(roles)


def _material_list(counts: dict[str, int], priority: str, total_defects: int) -> list[str]:
    if total_defects == 0:
        return ["巡检记录表", "复检图片归档目录"]

    items = ["高清复检相机", "缺陷位置标记贴", "个人防护装备"]
    for class_name in counts:
        items.extend(CLASS_MATERIALS.get(class_name, []))
    if priority in {"P0 紧急", "P1 高"}:
        items.extend(["叶片锁定确认单", "吊篮/绳索作业安全检查表"])
    return _unique(items)


def _safety_controls(priority: str, counts: dict[str, int], total_defects: int) -> list[str]:
    if total_defects == 0:
        return ["按常规巡检记录归档，无需现场维修安全措施。"]

    controls = [
        "现场作业前确认天气、风速和叶轮锁定状态。",
        "高空作业人员必须完成个人防护和工具防坠检查。",
        "维修前后保留同角度、同区域复检图片。",
    ]
    if priority in {"P0 紧急", "P1 高"}:
        controls.insert(0, "派工前由安全负责人确认是否需要停机或降载。")
    if "crack" in counts:
        controls.append("裂纹区域未完成结构复核前，不建议扩大运行负载。")
    return _unique(controls)


def _acceptance_criteria(total_defects: int, counts: dict[str, int]) -> list[str]:
    if total_defects == 0:
        return ["本次记录归档后，按既定巡检周期继续观察。"]

    criteria = [
        "维修后上传同区域复检图片或关键帧，并重新运行检测。",
        "复检报告中同类缺陷置信度低于当前派单阈值，或人工确认已修复。",
        "维修记录包含缺陷位置、处理方式、材料批次和复检结论。",
    ]
    if "crack" in counts:
        criteria.append("裂纹类缺陷需记录长度变化，并由结构工程师确认闭环。")
    return criteria


def _source_summary(report: dict[str, Any], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not detections:
        return []

    mode = str(report.get("mode") or "single")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        source = str(detection.get("source_name") or "-")
        grouped.setdefault(source, []).append(detection)

    summary: list[dict[str, Any]] = []
    for source, items in sorted(grouped.items()):
        counts = _class_counts(items)
        payload: dict[str, Any] = {
            "source_name": source,
            "defect_count": len(items),
            "class_summary": counts,
            "class_summary_cn": {CLASS_LABELS.get(name, name): count for name, count in counts.items()},
            "max_confidence": round(max(confidence_values(items)), 6),
        }
        if mode == "video":
            frames = sorted(
                {
                    int(item.get("frame_index"))
                    for item in items
                    if item.get("frame_index") is not None
                }
            )
            payload["frames"] = frames[:20]
            payload["frame_count"] = len(frames)
        summary.append(payload)
    return summary


def _defect_map(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detection in detections[:80]:
        class_name = str(detection.get("class_name") or "unknown")
        row: dict[str, Any] = {
            "source_name": detection.get("source_name", "-"),
            "source_type": detection.get("source_type", "-"),
            "class_name": class_name,
            "class_name_cn": CLASS_LABELS.get(class_name, class_name),
            "confidence": round(float(detection.get("confidence", 0.0) or 0.0), 6),
            "xyxy": detection.get("xyxy", []),
        }
        if "frame_index" in detection:
            row["frame_index"] = detection.get("frame_index")
            row["time_seconds"] = detection.get("time_seconds")
        rows.append(row)
    return rows


def _tasks(
    *,
    counts: dict[str, int],
    priority: str,
    total_defects: int,
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    if total_defects == 0:
        return [
            {
                "task_id": "T01",
                "category": "归档",
                "title": "保存本次正常巡检记录",
                "detail": "将原始图片/视频、检测报告和阈值配置归档，作为后续趋势对比基线。",
                "source": "rule",
            },
            {
                "task_id": "T02",
                "category": "复查",
                "title": "按巡检周期复查",
                "detail": "如画面模糊、反光或角度受限，补拍关键区域后重新检测。",
                "source": "rule",
            },
        ]

    task_details: list[tuple[str, str, str]] = [
        ("复核", "人工确认检测框", "核对检测框是否落在真实缺陷区域，排除反光、阴影、文字或背景干扰。"),
    ]
    if priority in {"P0 紧急", "P1 高"}:
        task_details.append(("调度", "确认运行处置", "派工前确认是否需要停机、降载或设置现场安全隔离。"))

    for class_name in sorted(counts, key=lambda name: (-CLASS_WEIGHT.get(name, 1), name)):
        label = CLASS_LABELS.get(class_name, class_name)
        for detail in CLASS_TASKS.get(class_name, [f"人工复核 {label} 的真实位置、范围和影响。"]):
            task_details.append((label, f"处理{label}", detail))

    for advice in analysis.get("maintenance_advice", []) or []:
        task_details.append(("AI建议", "执行故障分析建议", str(advice)))

    tasks = []
    for index, (category, title, detail) in enumerate(_unique_task_details(task_details), start=1):
        tasks.append(
            {
                "task_id": f"T{index:02d}",
                "category": category,
                "title": title,
                "detail": detail,
                "source": "analysis" if category == "AI建议" else "rule",
            }
        )
    return tasks


def _unique_task_details(items: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str, str]] = []
    for item in items:
        key = item[2]
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _summary_text(
    *,
    priority: str,
    counts_cn: dict[str, int],
    total_defects: int,
    analysis: dict[str, Any],
) -> str:
    if total_defects == 0:
        return "未检测到明确缺陷，系统建议归档并按巡检周期继续观察。"

    class_text = "、".join(f"{name} {count} 个" for name, count in counts_cn.items()) or "疑似缺陷"
    summary = analysis.get("fault_summary")
    if summary:
        return f"{priority}：{summary}"
    return f"{priority}：检测到 {class_text}，建议生成维修单并完成人工复核。"


def build_maintenance_order(
    report: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic maintenance work order from detection and analysis data."""
    analysis = analysis or {}
    detections = flatten_detections(report)
    counts = _class_counts(detections)
    counts_cn = {CLASS_LABELS.get(name, name): count for name, count in counts.items()}
    total_defects = len(detections)
    confidences = confidence_values(detections)
    max_confidence = max(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0
    risk_level = str(analysis.get("risk_level") or report.get("risk_level") or "正常")
    priority = _priority(
        risk_level=risk_level,
        counts=counts,
        total_defects=total_defects,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    sla_hours = PRIORITY_SLA[priority]
    request_id = str(report.get("request_id") or "-")
    primary = _primary_class(counts)
    media_name = str(report.get("uploaded_filename") or report.get("video_filename") or "-")
    status = _status(priority, total_defects, min_confidence)
    created_at = report.get("created_at") or analysis.get("created_at")

    return {
        "order_id": f"WO-{request_id[:8].upper()}" if request_id and request_id != "-" else "WO-PENDING",
        "request_id": request_id,
        "title": (
            f"{media_name} 缺陷维修单"
            if total_defects
            else f"{media_name} 巡检归档单"
        ),
        "status": status,
        "dispatch_required": total_defects > 0,
        "auto_generated": True,
        "priority": priority,
        "risk_level": risk_level,
        "sla_hours": sla_hours,
        "due_at": _due_at(created_at, sla_hours),
        "suggested_schedule": _schedule_text(priority),
        "assigned_team": _assigned_team(primary, total_defects, min_confidence),
        "required_roles": _required_roles(counts, priority, total_defects),
        "primary_defect": primary,
        "primary_defect_cn": CLASS_LABELS.get(primary, primary) if primary else None,
        "defect_summary": {
            "total_defects": total_defects,
            "class_summary": counts,
            "class_summary_cn": counts_cn,
            "max_confidence": round(max_confidence, 6),
            "min_confidence": round(min_confidence, 6),
            "source_summary": _source_summary(report, detections),
        },
        "summary": _summary_text(
            priority=priority,
            counts_cn=counts_cn,
            total_defects=total_defects,
            analysis=analysis,
        ),
        "tasks": _tasks(
            counts=counts,
            priority=priority,
            total_defects=total_defects,
            analysis=analysis,
        ),
        "materials_and_tools": _material_list(counts, priority, total_defects),
        "safety_controls": _safety_controls(priority, counts, total_defects),
        "recheck_plan": analysis.get("recheck_requirement")
        or "处理完成后上传同区域复检图片或关键帧，并重新运行检测形成闭环。",
        "shutdown_suggestion": analysis.get("shutdown_suggestion")
        or _shutdown_text(priority, total_defects),
        "acceptance_criteria": _acceptance_criteria(total_defects, counts),
        "manual_review_notice": analysis.get("manual_review_notice")
        or "维修单由系统根据检测结果自动生成，最终处置需人工确认。",
        "defect_map": _defect_map(detections),
        "analysis_used": bool(analysis),
        "vlm_used": bool(analysis.get("vlm_used")),
    }


def _schedule_text(priority: str) -> str:
    return {
        "P0 紧急": "6 小时内完成复核和运行处置决策。",
        "P1 高": "24 小时内安排现场复核或维修准备。",
        "P2 中": "3 天内完成人工复核并排入维修计划。",
        "P3 复核": "5 天内完成图片/关键帧复核。",
        "P4 观察": "按巡检周期归档观察。",
    }.get(priority, "按人工调度结果执行。")


def _shutdown_text(priority: str, total_defects: int) -> str:
    if total_defects == 0:
        return "当前不建议停机。"
    if priority == "P0 紧急":
        return "建议立即人工复核，并结合现场情况评估停机或降载。"
    if priority == "P1 高":
        return "建议尽快复核，并判断是否需要降载或短暂停机检查。"
    return "暂不直接建议停机，需人工复核后决定。"


def maintenance_order_to_markdown(order: dict[str, Any]) -> str:
    lines = [
        "# 风机叶片智能维修单",
        "",
        "## 基本信息",
        "",
        f"- 维修单号：{order.get('order_id', '-')}",
        f"- 请求 ID：{order.get('request_id', '-')}",
        f"- 标题：{order.get('title', '-')}",
        f"- 状态：{order.get('status', '-')}",
        f"- 优先级：{order.get('priority', '-')}",
        f"- 风险等级：{order.get('risk_level', '-')}",
        f"- 派工团队：{order.get('assigned_team', '-')}",
        f"- 建议时限：{order.get('suggested_schedule', '-')}",
        f"- 到期时间：{order.get('due_at') or '-'}",
        "",
        "## 派单摘要",
        "",
        str(order.get("summary", "-")),
        "",
        "## 缺陷概览",
        "",
    ]
    defect_summary = order.get("defect_summary", {}) or {}
    class_summary_cn = defect_summary.get("class_summary_cn", {}) or {}
    if class_summary_cn:
        for name, count in class_summary_cn.items():
            lines.append(f"- {name}：{count}")
    else:
        lines.append("- 未检测到明确缺陷")

    lines.extend(["", "## 处理任务", ""])
    for task in order.get("tasks", []) or []:
        lines.append(f"- {task.get('task_id')} | {task.get('category')} | {task.get('title')}：{task.get('detail')}")

    lines.extend(["", "## 人员与物料", ""])
    lines.append("- 角色：" + "、".join(order.get("required_roles", []) or ["-"]))
    lines.append("- 工具/物料：" + "、".join(order.get("materials_and_tools", []) or ["-"]))

    lines.extend(["", "## 安全控制", ""])
    for item in order.get("safety_controls", []) or ["-"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 复检与验收",
            "",
            f"- 复检计划：{order.get('recheck_plan', '-')}",
            f"- 停机建议：{order.get('shutdown_suggestion', '-')}",
        ]
    )
    for item in order.get("acceptance_criteria", []) or ["-"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def save_maintenance_order(order: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "maintenance_order.json"
    md_path = output_dir / "maintenance_order.md"
    order_with_paths = dict(order)
    order_with_paths["maintenance_order_json"] = str(json_path)
    order_with_paths["maintenance_order_markdown"] = str(md_path)
    json_path.write_text(json.dumps(order_with_paths, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(maintenance_order_to_markdown(order_with_paths), encoding="utf-8")
    return {
        "maintenance_order_json": str(json_path),
        "maintenance_order_markdown": str(md_path),
    }
