from pathlib import Path

from wind_fault.analysis.analyzer import analyze_detection_report
from wind_fault.analysis.prompt_builder import build_prompt
from wind_fault.analysis.report_loader import flatten_detections
from wind_fault.analysis.risk_engine import assess_risk
from wind_fault.analysis.vlm_client import generate_real_vlm_analysis
from wind_fault.reports.combined_report import write_combined_report


def test_risk_engine_promotes_multiple_defects() -> None:
    detections = [
        {"class_name": "crack", "confidence": 0.94},
        {"class_name": "hole", "confidence": 0.92},
        {"class_name": "spalling", "confidence": 0.88},
    ]

    result = assess_risk(detections)

    assert result.risk_level == "高风险"
    assert result.total_defects == 3
    assert result.needs_manual_review is False


def test_analyze_detection_report_writes_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    report = {
        "request_id": "0" * 32,
        "mode": "single",
        "image_count": 1,
        "total_defects": 1,
        "items": [
            {
                "uploaded_filename": "blade.jpg",
                "defect_count": 1,
                "detections": [
                    {
                        "class_name": "crack",
                        "class_name_cn": "裂纹",
                        "confidence": 0.91,
                        "xyxy": [1, 2, 3, 4],
                    }
                ],
            }
        ],
    }

    detections = flatten_detections(report)
    analysis = analyze_detection_report(report, tmp_path)

    assert len(detections) == 1
    assert analysis["risk_level"] in {"中高风险", "高风险"}
    assert "裂纹" in analysis["fault_summary"]
    assert analysis["maintenance_order"]["priority"] in {"P0 紧急", "P1 高"}
    assert analysis["maintenance_order"]["assigned_team"] == "叶片结构检修组"
    assert analysis["vlm_used"] is False
    assert analysis["vlm_provider"] in {"mock", "openai_compatible"}
    assert (tmp_path / "reports" / "analysis_report.json").exists()
    assert (tmp_path / "reports" / "analysis_report.md").exists()


def test_vlm_client_falls_back_without_api_key(tmp_path: Path) -> None:
    config = tmp_path / "vlm_config.yaml"
    config.write_text(
        "\n".join(
            [
                "provider: openai_compatible",
                "endpoint: https://example.invalid/v1/chat/completions",
                "model: demo-vlm",
                "api_key_env: MISSING_TEST_VLM_KEY",
            ]
        ),
        encoding="utf-8",
    )
    fallback = {"fault_summary": "fallback"}

    result = generate_real_vlm_analysis(
        prompt="test",
        image_paths=[],
        fallback=fallback,
        config_path=config,
    )

    assert result["fault_summary"] == "fallback"
    assert result["vlm_used"] is False
    assert "MISSING_TEST_VLM_KEY" in result["vlm_error"]


def test_video_prompt_includes_keyframe_context() -> None:
    report = {
        "request_id": "1" * 32,
        "mode": "video",
        "uploaded_filename": "blade_video.mp4",
        "total_frames": 100,
        "processed_frames": 10,
        "defect_frame_count": 2,
        "duration_seconds": 4.0,
        "keyframes": [
            {
                "frame_index": 20,
                "time_seconds": 0.8,
                "defect_count": 1,
                "max_confidence": 0.91,
            }
        ],
        "frames": [
            {
                "frame_index": 20,
                "time_seconds": 0.8,
                "defect_count": 1,
                "detections": [
                    {
                        "class_name": "spalling",
                        "class_name_cn": "剥落",
                        "confidence": 0.91,
                        "xyxy": [1, 2, 3, 4],
                    }
                ],
            }
        ],
    }
    risk = assess_risk(flatten_detections(report)).to_dict()

    prompt = build_prompt(report, risk)

    assert "视频检测任务" in prompt
    assert "第 20 帧" in prompt
    assert "剥落" in prompt


def test_combined_report_exports_detection_and_analysis(tmp_path: Path) -> None:
    report = {
        "request_id": "2" * 32,
        "mode": "video",
        "uploaded_filename": "blade_video.mp4",
        "created_at": "2026-06-04T20:00:00",
        "confidence_threshold": 0.3,
        "processed_frames": 10,
        "total_frames": 100,
        "defect_frame_count": 1,
        "total_defects": 1,
        "risk_level": "中风险",
        "class_summary": {"spalling": 1},
        "class_summary_cn": {"剥落": 1},
        "advice": "建议复核关键帧。",
        "keyframes": [
            {
                "frame_index": 20,
                "time_seconds": 0.8,
                "defect_count": 1,
                "max_confidence": 0.91,
                "image_path": "frame_000020.jpg",
            }
        ],
        "frames": [
            {
                "frame_index": 20,
                "time_seconds": 0.8,
                "defect_count": 1,
                "detections": [{"class_name": "spalling", "confidence": 0.91, "xyxy": [1, 2, 3, 4]}],
            }
        ],
        "analysis_report": {
            "risk_level": "中风险",
            "confidence_level": "检测可信度较高",
            "fault_summary": "检测到剥落缺陷。",
            "fault_description": "关键帧中存在叶片表面异常。",
            "possible_causes": ["表面涂层脱落"],
            "risk_impact": "缺陷可能继续扩展。",
            "maintenance_advice": ["检查剥落面积"],
            "recheck_requirement": "维修后复检。",
            "shutdown_suggestion": "暂不直接建议停机。",
            "manual_review_notice": "需要人工复核。",
            "risk_reasons": ["检测到剥落。"],
            "vlm_used": True,
            "vlm_model": "glm-5v-turbo",
        },
    }

    paths = write_combined_report(report, tmp_path)
    markdown = Path(paths["complete_report_markdown"]).read_text(encoding="utf-8")

    assert "视频关键帧" in markdown
    assert "故障分析与运维建议" in markdown
    assert "智能维修单" in markdown
    assert "glm-5v-turbo" in markdown
    assert Path(paths["complete_report_json"]).exists()
