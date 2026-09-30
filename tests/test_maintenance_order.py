from pathlib import Path

from wind_fault.maintenance import build_maintenance_order, maintenance_order_to_markdown, save_maintenance_order
from wind_fault.reports.combined_report import write_combined_report


def test_maintenance_order_escalates_high_confidence_crack() -> None:
    report = {
        "request_id": "a" * 32,
        "mode": "single",
        "uploaded_filename": "blade.jpg",
        "created_at": "2026-06-06T10:00:00",
        "risk_level": "高风险",
        "items": [
            {
                "uploaded_filename": "blade.jpg",
                "detections": [
                    {
                        "class_name": "crack",
                        "class_name_cn": "裂纹",
                        "confidence": 0.94,
                        "xyxy": [1, 2, 30, 40],
                    }
                ],
            }
        ],
    }
    analysis = {
        "risk_level": "高风险",
        "fault_summary": "检测到高置信度裂纹。",
        "maintenance_advice": ["优先复核裂纹长度。"],
        "recheck_requirement": "修复后上传复检图。",
    }

    order = build_maintenance_order(report, analysis)

    assert order["order_id"] == "WO-AAAAAAAA"
    assert order["priority"] == "P0 紧急"
    assert order["status"] == "待立即派单"
    assert order["assigned_team"] == "叶片结构检修组"
    assert order["due_at"] == "2026-06-06T16:00:00"
    assert any(task["source"] == "analysis" for task in order["tasks"])


def test_maintenance_order_archives_normal_detection() -> None:
    report = {
        "request_id": "b" * 32,
        "mode": "single",
        "uploaded_filename": "normal.jpg",
        "risk_level": "正常",
        "items": [{"uploaded_filename": "normal.jpg", "detections": []}],
    }

    order = build_maintenance_order(report)

    assert order["dispatch_required"] is False
    assert order["priority"] == "P4 观察"
    assert order["status"] == "无需派单"
    assert order["assigned_team"] == "巡检值班组"


def test_save_maintenance_order_writes_markdown_and_json(tmp_path: Path) -> None:
    order = build_maintenance_order(
        {
            "request_id": "c" * 32,
            "mode": "single",
            "uploaded_filename": "blade.jpg",
            "risk_level": "中风险",
            "items": [
                {
                    "uploaded_filename": "blade.jpg",
                    "detections": [
                        {"class_name": "spalling", "confidence": 0.82, "xyxy": [1, 2, 3, 4]}
                    ],
                }
            ],
        }
    )

    paths = save_maintenance_order(order, tmp_path)
    markdown = maintenance_order_to_markdown(order)

    assert Path(paths["maintenance_order_json"]).exists()
    assert Path(paths["maintenance_order_markdown"]).exists()
    assert "智能维修单" in markdown
    assert "叶片表面修复组" in markdown


def test_combined_report_includes_maintenance_order(tmp_path: Path) -> None:
    report = {
        "request_id": "d" * 32,
        "mode": "single",
        "uploaded_filename": "blade.jpg",
        "created_at": "2026-06-06T10:00:00",
        "confidence_threshold": 0.45,
        "image_count": 1,
        "total_defects": 1,
        "risk_level": "中风险",
        "class_summary": {"corrosion": 1},
        "class_summary_cn": {"腐蚀": 1},
        "advice": "建议复核。",
        "items": [
            {
                "uploaded_filename": "blade.jpg",
                "defect_count": 1,
                "detections": [{"class_name": "corrosion", "confidence": 0.88, "xyxy": [1, 2, 3, 4]}],
            }
        ],
    }
    report["maintenance_order"] = build_maintenance_order(report)

    paths = write_combined_report(report, tmp_path)
    markdown = Path(paths["complete_report_markdown"]).read_text(encoding="utf-8")

    assert "智能维修单" in markdown
    assert "防腐涂层处理组" in markdown
