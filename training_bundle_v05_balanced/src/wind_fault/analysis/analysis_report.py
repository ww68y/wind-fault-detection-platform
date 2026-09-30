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
