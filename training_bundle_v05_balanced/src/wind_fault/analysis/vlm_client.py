from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vlm_config.yaml"


def load_vlm_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG
    if not path.exists():
        return {"provider": "mock"}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["provider"] = os.getenv("WIND_VLM_PROVIDER", data.get("provider", "mock"))
    data["endpoint"] = os.getenv("WIND_VLM_ENDPOINT", data.get("endpoint", ""))
    data["model"] = os.getenv("WIND_VLM_MODEL", data.get("model", ""))
    data["api_key_env"] = os.getenv("WIND_VLM_API_KEY_ENV", data.get("api_key_env", "VLM_API_KEY"))
    return data


def _image_data_url(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _request_openai_compatible(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    image_paths: list[Path],
    timeout_seconds: int,
    temperature: float,
) -> tuple[dict[str, Any] | None, str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        data_url = _image_data_url(path)
        if data_url:
            content.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    text = str(response_payload["choices"][0]["message"]["content"])
    return _extract_json_object(text), text


def generate_real_vlm_analysis(
    *,
    prompt: str,
    image_paths: list[Path],
    fallback: dict[str, Any],
    config_path: Path | None = None,
) -> dict[str, Any]:
    config = load_vlm_config(config_path)
    provider = str(config.get("provider", "mock")).lower()
    if provider in {"", "mock", "disabled", "none"}:
        result = dict(fallback)
        result["vlm_provider"] = "mock"
        result["vlm_used"] = False
        return result

    if provider not in {"openai_compatible", "openai"}:
        result = dict(fallback)
        result["vlm_provider"] = provider
        result["vlm_used"] = False
        result["vlm_error"] = f"Unsupported VLM provider: {provider}"
        return result

    endpoint = str(config.get("endpoint") or "https://api.openai.com/v1/chat/completions")
    model = str(config.get("model") or "")
    api_key_env = str(config.get("api_key_env") or "VLM_API_KEY")
    api_key = os.getenv(api_key_env, "")
    if not model or not api_key:
        result = dict(fallback)
        result["vlm_provider"] = provider
        result["vlm_used"] = False
        result["vlm_error"] = f"Missing VLM model or API key env: {api_key_env}"
        return result

    try:
        parsed, raw_text = _request_openai_compatible(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            prompt=prompt,
            image_paths=image_paths[: int(config.get("max_images", 4) or 4)],
            timeout_seconds=int(config.get("timeout_seconds", 60) or 60),
            temperature=float(config.get("temperature", 0.2) or 0.2),
        )
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as exc:
        result = dict(fallback)
        result["vlm_provider"] = provider
        result["vlm_used"] = False
        result["vlm_error"] = str(exc)
        return result

    result = dict(fallback)
    if parsed:
        for key in [
            "fault_summary",
            "fault_description",
            "possible_causes",
            "risk_impact",
            "maintenance_advice",
            "recheck_requirement",
            "shutdown_suggestion",
            "manual_review_notice",
        ]:
            if key in parsed:
                result[key] = parsed[key]
    result["vlm_provider"] = provider
    result["vlm_model"] = model
    result["vlm_used"] = True
    result["vlm_raw_text"] = raw_text
    return result
