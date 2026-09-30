from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 风机叶片故障分析报告",
        "",
        f"- 请求 ID：{report.get('request_id', '-')}",
        f"- 风险等级：{report.get('risk_level', '-')}",
        f"- 置信度判断：{report.get('confidence_level', '-')}",
        f"- 缺陷总数：{report.get('total_defects', 0)}",
        "",
        "## 故障摘要",
        "",
        str(report.get("fault_summary", "-")),
        "",
        "## 故障现象描述",
        "",
        str(report.get("fault_description", "-")),
        "",
        "## 可能原因",
        "",
    ]
    for item in report.get("possible_causes", []) or ["-"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 风险影响", "", str(report.get("risk_impact", "-")), "", "## 运维建议", ""])
    for item in report.get("maintenance_advice", []) or ["-"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 复检要求",
            "",
            str(report.get("recheck_requirement", "-")),
            "",
            "## 是否建议停机",
            "",
            str(report.get("shutdown_suggestion", "-")),
            "",
            "## 人工复核提示",
            "",
            str(report.get("manual_review_notice", "-")),
            "",
            "## 风险判断依据",
            "",
        ]
    )
    for item in report.get("risk_reasons", []) or ["-"]:
        lines.append(f"- {item}")

    order = report.get("maintenance_order") or {}
    if order:
        dispatch = report.get("maintenance_dispatch") or {}
        lines.extend(
            [
                "",
                "## 智能维修单",
                "",
                f"- 维修单号：{order.get('order_id', '-')}",
                f"- 状态：{order.get('status', '-')}",
                f"- 优先级：{order.get('priority', '-')}",
                f"- 派工团队：{order.get('assigned_team', '-')}",
                f"- 建议时限：{order.get('suggested_schedule', '-')}",
                f"- 机器人派发状态：{dispatch.get('status') or order.get('dispatch_status') or '-'}",
                f"- 机器人渠道：{dispatch.get('provider') or order.get('dispatch_provider') or '-'}",
                f"- 派单摘要：{order.get('summary', '-')}",
                "",
            ]
        )
        for task in order.get("tasks", [])[:8]:
            lines.append(f"- {task.get('task_id', '-')}：{task.get('detail', '-')}")

    return "\n".join(lines) + "\n"


def save_analysis_report(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_report.json"
    md_path = output_dir / "analysis_report.md"
    report = dict(report)
    report["analysis_report_json"] = str(json_path)
    report["analysis_report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return report
