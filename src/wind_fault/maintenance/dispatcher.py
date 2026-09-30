from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import base64
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def should_auto_dispatch(order: dict[str, Any]) -> bool:
    return bool(order.get("dispatch_required"))


def _provider() -> str:
    return str(os.getenv("WIND_DISPATCH_PROVIDER", "")).strip().lower()


def _webhook_url(provider: str) -> str:
    if provider == "wecom":
        return os.getenv("WIND_WECOM_WEBHOOK", "").strip()
    if provider == "dingtalk":
        return os.getenv("WIND_DINGTALK_WEBHOOK", "").strip()
    return ""


def _dingtalk_signed_url(url: str) -> str:
    secret = os.getenv("WIND_DINGTALK_SECRET", "").strip()
    if not secret:
        return url
    timestamp = str(round(time.time() * 1000))
    message = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(digest).decode("utf-8"))
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}timestamp={timestamp}&sign={sign}"


def _markdown(order: dict[str, Any], report: dict[str, Any]) -> str:
    defect_summary = order.get("defect_summary") or {}
    class_summary_cn = defect_summary.get("class_summary_cn") or {}
    class_text = "、".join(f"{name}{count}处" for name, count in class_summary_cn.items()) or "未记录"
    report_url = _public_url(report.get("report_page_url") or "-")
    order_url = _public_url(order.get("maintenance_order_markdown_url") or report.get("maintenance_order_markdown_url") or "-")
    tasks = order.get("tasks") or []
    task_lines = "\n".join(
        f"> {task.get('task_id', '-')}. {task.get('title', '-')}"
        for task in tasks[:5]
        if isinstance(task, dict)
    )
    return "\n".join(
        [
            f"### 风机叶片维修单 {order.get('order_id', '-')}",
            f"> 优先级：{order.get('priority', '-')}",
            f"> 状态：{order.get('status', '-')}",
            f"> 派工团队：{order.get('assigned_team', '-')}",
            f"> 缺陷统计：{class_text}",
            f"> 建议时限：{order.get('suggested_schedule', '-')}",
            f"> 到期时间：{order.get('due_at') or '-'}",
            f"> 摘要：{order.get('summary', '-')}",
            "",
            "#### 处理任务",
            task_lines or "> -",
            "",
            f"[查看检测报告]({report_url})",
            f"[查看维修单]({order_url})",
        ]
    )


def _public_url(url: Any) -> str:
    text = str(url or "")
    if not text or text == "-" or text.startswith(("http://", "https://")):
        return text or "-"
    base_url = os.getenv("WIND_PUBLIC_BASE_URL", "").strip()
    if not base_url:
        return text
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", text.lstrip("/"))


def _payload(provider: str, markdown: str, order: dict[str, Any]) -> dict[str, Any]:
    title = f"风机叶片维修单 {order.get('order_id', '-')}"
    if provider == "wecom":
        return {"msgtype": "markdown", "markdown": {"content": markdown}}
    if provider == "dingtalk":
        return {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": markdown,
            },
        }
    raise ValueError(f"Unsupported dispatch provider: {provider}")


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return int(response.status), response.read().decode("utf-8", errors="replace")


def dispatch_maintenance_order(
    order: dict[str, Any],
    report: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    provider = _provider()
    if not force and not should_auto_dispatch(order):
        return {
            "status": "skipped",
            "reason": "dispatch_not_required",
            "provider": provider or None,
            "priority": order.get("priority"),
        }

    if provider not in {"wecom", "dingtalk"}:
        return {
            "status": "not_configured",
            "reason": "set WIND_DISPATCH_PROVIDER to wecom or dingtalk",
            "provider": provider or None,
            "priority": order.get("priority"),
        }

    url = _webhook_url(provider)
    if not url:
        env_name = "WIND_WECOM_WEBHOOK" if provider == "wecom" else "WIND_DINGTALK_WEBHOOK"
        return {
            "status": "not_configured",
            "reason": f"missing {env_name}",
            "provider": provider,
            "priority": order.get("priority"),
        }

    target_url = _dingtalk_signed_url(url) if provider == "dingtalk" else url
    markdown = _markdown(order, report)
    try:
        response_status, response_text = _post_json(
            target_url,
            _payload(provider, markdown, order),
            int(os.getenv("WIND_DISPATCH_TIMEOUT", "15") or 15),
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "provider": provider,
            "priority": order.get("priority"),
        }

    return {
        "status": "sent",
        "provider": provider,
        "priority": order.get("priority"),
        "http_status": response_status,
        "response": response_text[:1000],
    }


def save_dispatch_record(record: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "maintenance_dispatch.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"maintenance_dispatch_json": str(path)}
