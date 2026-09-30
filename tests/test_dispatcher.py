from wind_fault.maintenance.dispatcher import dispatch_maintenance_order, should_auto_dispatch


def _order(priority: str = "P0 紧急", dispatch_required: bool = True) -> dict:
    return {
        "order_id": "WO-TEST",
        "dispatch_required": dispatch_required,
        "priority": priority,
        "status": "待立即派单",
        "assigned_team": "叶片结构检修组",
        "suggested_schedule": "6 小时内完成复核。",
        "summary": "检测到高置信度裂纹。",
        "defect_summary": {"class_summary_cn": {"裂纹": 1}},
        "tasks": [{"task_id": "T01", "title": "人工确认检测框"}],
    }


def test_should_auto_dispatch_any_defect_order() -> None:
    assert should_auto_dispatch(_order("P0 紧急")) is True
    assert should_auto_dispatch(_order("P1 高")) is True
    assert should_auto_dispatch(_order("P2 中")) is True
    assert should_auto_dispatch(_order("P3 复核")) is True
    assert should_auto_dispatch(_order("P4 观察", dispatch_required=False)) is False


def test_dispatch_returns_not_configured_without_provider(monkeypatch) -> None:
    monkeypatch.delenv("WIND_DISPATCH_PROVIDER", raising=False)

    result = dispatch_maintenance_order(_order(), {"report_page_url": "/report/demo"})

    assert result["status"] == "not_configured"


def test_dispatch_skips_when_no_defect_requires_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("WIND_DISPATCH_PROVIDER", "wecom")
    monkeypatch.setenv("WIND_WECOM_WEBHOOK", "https://example.invalid/webhook")

    result = dispatch_maintenance_order(
        _order("P4 观察", dispatch_required=False),
        {"report_page_url": "/report/demo"},
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "dispatch_not_required"


def test_dispatch_wecom_posts_markdown(monkeypatch) -> None:
    sent = {}

    def fake_post(url: str, payload: dict, timeout_seconds: int) -> tuple[int, str]:
        sent["url"] = url
        sent["payload"] = payload
        sent["timeout_seconds"] = timeout_seconds
        return 200, '{"errcode":0}'

    monkeypatch.setenv("WIND_DISPATCH_PROVIDER", "wecom")
    monkeypatch.setenv("WIND_WECOM_WEBHOOK", "https://example.invalid/wecom")
    monkeypatch.setattr("wind_fault.maintenance.dispatcher._post_json", fake_post)

    result = dispatch_maintenance_order(_order("P3 复核"), {"report_page_url": "/report/demo"})

    assert result["status"] == "sent"
    assert sent["url"] == "https://example.invalid/wecom"
    assert sent["payload"]["msgtype"] == "markdown"
    assert "WO-TEST" in sent["payload"]["markdown"]["content"]


def test_dispatch_dingtalk_posts_markdown(monkeypatch) -> None:
    sent = {}

    def fake_post(url: str, payload: dict, timeout_seconds: int) -> tuple[int, str]:
        sent["url"] = url
        sent["payload"] = payload
        return 200, '{"errcode":0}'

    monkeypatch.setenv("WIND_DISPATCH_PROVIDER", "dingtalk")
    monkeypatch.setenv("WIND_DINGTALK_WEBHOOK", "https://example.invalid/dingtalk")
    monkeypatch.setattr("wind_fault.maintenance.dispatcher._post_json", fake_post)

    result = dispatch_maintenance_order(_order("P1 高"), {"report_page_url": "/report/demo"})

    assert result["status"] == "sent"
    assert sent["url"] == "https://example.invalid/dingtalk"
    assert sent["payload"]["msgtype"] == "markdown"
    assert sent["payload"]["markdown"]["title"] == "风机叶片维修单 WO-TEST"
