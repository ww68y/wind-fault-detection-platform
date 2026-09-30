from __future__ import annotations

import html
import json
import os
import re
import csv
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from wind_fault.analysis import analyze_detection_report
from wind_fault.model.yolo_runner import predict_yolo
from wind_fault.reports.combined_report import write_combined_report
from wind_fault.reports.prediction_report import write_prediction_report
from wind_fault.video.video_detector import detect_video as run_video_detection

PROJECT_ROOT = Path.cwd().resolve()
RUNS_ROOT = PROJECT_ROOT / "runs"
API_RUNS_ROOT = RUNS_ROOT / "api"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
TRAIN_RUN_DIR = RUNS_ROOT / "detect" / "runs" / "wind_fault" / "gpu_640_b4_80e"
MISSED_REPORT = RUNS_ROOT / "wind_fault" / "test_batch_80e" / "reports" / "missed_samples" / "missed_samples_report.csv"
HARD_CASES_ROOT = PROJECT_ROOT / "hard_cases"
HARD_CASES_DIR = HARD_CASES_ROOT / "missed_samples"
HARD_CASE_CATEGORIES = {
    "missed": ("漏检", HARD_CASES_ROOT / "missed_samples"),
    "false_positive": ("误检", HARD_CASES_ROOT / "false_positive_samples"),
    "low_confidence": ("低置信度", HARD_CASES_ROOT / "low_confidence_samples"),
}

CLASS_LABELS = {
    "crack": "裂纹",
    "hole": "孔洞",
    "spalling": "剥落",
    "corrosion": "腐蚀",
}

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
API_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
HARD_CASES_ROOT.mkdir(parents=True, exist_ok=True)
for _, category_dir in HARD_CASE_CATEGORIES.values():
    category_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Wind Fault Detection API", version="0.3.0")
app.mount("/artifacts", StaticFiles(directory=str(RUNS_ROOT)), name="artifacts")
app.mount("/hard_cases", StaticFiles(directory=str(HARD_CASES_ROOT)), name="hard_cases")


def _safe_request_id(request_id: str) -> str:
    if not re.fullmatch(r"[a-fA-F0-9]{32}", request_id):
        raise HTTPException(status_code=404, detail="Detection record not found")
    return request_id


def _model_path() -> Path:
    configured_model = os.getenv("WIND_MODEL_PATH")
    candidates = [Path(configured_model)] if configured_model else [
        TRAIN_RUN_DIR / "weights" / "best.pt",
        PROJECT_ROOT / "release" / "v0.1_baseline" / "model" / "best.pt",
        RUNS_ROOT / "wind_fault" / "baseline" / "weights" / "best.pt",
    ]

    checked_paths = []
    for model_path in candidates:
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        checked_paths.append(str(model_path))
        if model_path.exists():
            return model_path

    raise HTTPException(status_code=503, detail=f"Model not found. Checked: {checked_paths}")


def _to_artifact_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    try:
        relative = target.resolve().relative_to(RUNS_ROOT.resolve())
    except ValueError:
        return None
    return f"/artifacts/{relative.as_posix()}"


def _to_hard_case_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_absolute():
        target = HARD_CASES_ROOT / target
    try:
        relative = target.resolve().relative_to(HARD_CASES_ROOT.resolve())
    except ValueError:
        return None
    return f"/hard_cases/{relative.as_posix()}"


def _find_ffmpeg() -> str | None:
    configured = os.getenv("WIND_FFMPEG_PATH")
    if configured and Path(configured).exists():
        return configured

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    winget_root = os.getenv("LOCALAPPDATA")
    if winget_root:
        package_root = Path(winget_root) / "Microsoft" / "WinGet" / "Packages"
        if package_root.exists():
            matches = sorted(package_root.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"), reverse=True)
            if matches:
                return str(matches[0])

    return None


def _make_h264_mp4(source: Path, output_path: Path) -> str | None:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    except (subprocess.SubprocessError, OSError):
        if output_path.exists():
            output_path.unlink()
        return None

    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)
    return None


def _make_video_preview(source_path: str | Path | None, output_path: Path) -> str | None:
    if not source_path:
        return None
    source = Path(source_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.exists():
        return None

    h264_preview = _make_h264_mp4(source, output_path)
    if h264_preview:
        return h264_preview
    if source.suffix.lower() == ".mp4":
        return str(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return None
        height, width = frame.shape[:2]
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for codec in ("avc1", "H264", "mp4v"):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if output_path.exists():
            output_path.unlink()
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            continue

        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frame_count += 1
        writer.release()
        if frame_count > 0 and output_path.exists() and output_path.stat().st_size > 0:
            capture.release()
            return str(output_path)

    capture.release()
    return None


def _safe_upload_name(filename: str | None, index: int) -> str:
    raw_name = Path(filename or f"image_{index}.jpg").name
    suffix = Path(raw_name).suffix.lower() or ".jpg"
    stem = Path(raw_name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or f"image_{index}"
    return f"{index:03d}_{stem}{suffix}"


async def _save_uploads(files: list[UploadFile], upload_dir: Path) -> list[dict]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for index, file in enumerate(files, start=1):
        saved_name = _safe_upload_name(file.filename, index)
        saved_path = upload_dir / saved_name
        saved_path.write_bytes(await file.read())
        saved.append(
            {
                "path": saved_path,
                "uploaded_filename": file.filename or saved_name,
                "saved_filename": saved_name,
            }
        )
    return saved


def _render_result_images(results: list, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, str] = {}
    used_names: set[str] = set()

    for index, result in enumerate(results, start=1):
        source_path = Path(str(getattr(result, "path", f"image_{index}.jpg")))
        suffix = source_path.suffix or ".jpg"
        base_name = f"{source_path.stem}_pred{suffix}"
        render_name = base_name
        if render_name in used_names:
            render_name = f"{source_path.stem}_{index}_pred{suffix}"
        used_names.add(render_name)

        render_path = output_dir / render_name
        if hasattr(result, "save"):
            result.save(filename=str(render_path))
        rendered[str(source_path.resolve())] = str(render_path)
    return rendered


def _detections(payload: dict) -> list[dict]:
    return [
        detection
        for item in payload.get("items", [])
        for detection in item.get("detections", [])
    ]


def _class_summary(payload: dict) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for detection in _detections(payload):
        counter[str(detection.get("class_name", "unknown"))] += 1
    return dict(sorted(counter.items()))


def _class_summary_cn(payload: dict) -> dict[str, int]:
    return {
        CLASS_LABELS.get(name, name): count
        for name, count in _class_summary(payload).items()
    }


def _risk_and_advice(payload: dict) -> tuple[str, str, list[str]]:
    detections = _detections(payload)
    total = len(detections)
    image_count = int(payload.get("image_count", 1) or 1)

    if total == 0:
        return (
            "正常",
            "当前置信度阈值下未发现缺陷，建议保留图片记录并按巡检计划复查。",
            [
                "本批图片未检出裂纹、孔洞、剥落或腐蚀。",
                "如果图片存在低对比度细裂纹，可降低阈值或补拍局部高清图复查。",
            ],
        )

    max_conf = max(float(detection.get("confidence", 0.0)) for detection in detections)
    classes = {str(detection.get("class_name", "")) for detection in detections}
    affected_images = sum(1 for item in payload.get("items", []) if item.get("defect_count", 0) > 0)
    if total >= 3 or affected_images >= 2 or "crack" in classes or max_conf >= 0.9:
        return (
            "高",
            f"本次共检测 {image_count} 张图片，发现 {total} 处缺陷，建议安排人工复核并形成巡检记录。",
            [
                "优先复核高置信度缺陷框，确认缺陷类型和范围。",
                "裂纹、腐蚀或多处缺陷建议形成维修工单。",
                "必要时使用更高分辨率或切片检测进行二次确认。",
            ],
        )

    return (
        "中",
        f"本次共检测 {image_count} 张图片，发现 {total} 处疑似缺陷，建议人工复核检测框。",
        [
            "检查检测框是否贴合真实缺陷。",
            "如果是污染、阴影或背景干扰，可记录为误报样本。",
        ],
    )


def _readable_markdown(payload: dict) -> str:
    risk, advice, actions = _risk_and_advice(payload)
    lines = [
        "# 风机叶片缺陷检测报告",
        "",
        f"- 请求 ID：{payload.get('request_id', '-')}",
        f"- 检测模式：{payload.get('mode', '-')}",
        f"- 图片数量：{payload.get('image_count', 0)}",
        f"- 图片名称：{payload.get('uploaded_filename', '-')}",
        f"- 检测时间：{payload.get('created_at', '-')}",
        f"- 置信度阈值：{payload.get('confidence_threshold', '-')}",
        f"- 缺陷总数：{payload.get('total_defects', 0)}",
        f"- 风险等级：{risk}",
        "",
        "## 处理建议",
        "",
        advice,
        "",
    ]
    for action in actions:
        lines.append(f"- {action}")

    lines.extend(["", "## 缺陷统计", ""])
    class_summary = payload.get("class_summary", {})
    if class_summary:
        for name, count in class_summary.items():
            lines.append(f"- {CLASS_LABELS.get(name, name)}（{name}）：{count}")
    else:
        lines.append("- 无")

    lines.extend(["", "## 图片明细", ""])
    for item in payload.get("items", []):
        image_name = item.get("uploaded_filename") or Path(str(item.get("image", "-"))).name
        lines.append(f"### {image_name}")
        lines.append(f"- 缺陷数量：{item.get('defect_count', 0)}")
        if not item.get("detections"):
            lines.append("- 未检测到缺陷")
        for detection in item.get("detections", []):
            class_name = str(detection.get("class_name", "unknown"))
            zh_name = CLASS_LABELS.get(class_name, class_name)
            confidence = float(detection.get("confidence", 0.0))
            lines.append(f"- {zh_name}（{class_name}） | 置信度 {confidence:.3f} | 边框 {detection.get('xyxy', [])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _enrich_items(
    payload: dict,
    saved_uploads: list[dict] | None = None,
    rendered_by_image: dict[str, str] | None = None,
) -> None:
    saved_by_path = {
        str(item["path"].resolve()): item
        for item in saved_uploads or []
    }
    rendered_by_image = rendered_by_image or {}

    for item in payload.get("items", []):
        image_path = Path(str(item.get("image", "")))
        image_key = str(image_path.resolve()) if str(image_path) else ""
        saved = saved_by_path.get(image_key)
        if saved:
            item["uploaded_filename"] = saved["uploaded_filename"]
            item["saved_filename"] = saved["saved_filename"]
        else:
            item.setdefault("uploaded_filename", image_path.name or "-")

        item.setdefault("class_summary", {})
        counter: Counter[str] = Counter()
        for detection in item.get("detections", []):
            class_name = str(detection.get("class_name", "unknown"))
            detection.setdefault("class_name_cn", CLASS_LABELS.get(class_name, class_name))
            counter[class_name] += 1
        item["class_summary"] = dict(sorted(counter.items()))
        item["class_summary_cn"] = {
            CLASS_LABELS.get(name, name): count
            for name, count in item["class_summary"].items()
        }
        item["original_image_url"] = _to_artifact_url(image_path)
        item["result_image_url"] = _to_artifact_url(rendered_by_image.get(image_key))


def _finalize_report(
    payload: dict,
    work_dir: Path,
    request_id: str,
    saved_uploads: list[dict],
    confidence_threshold: float,
    rendered_by_image: dict[str, str],
) -> dict:
    _enrich_items(payload, saved_uploads, rendered_by_image)
    image_count = int(payload.get("image_count", 0))
    uploaded_filenames = [item["uploaded_filename"] for item in saved_uploads]
    uploaded_filename = uploaded_filenames[0] if image_count == 1 else f"批量检测 {image_count} 张图片"

    payload["request_id"] = request_id
    payload["mode"] = "single" if image_count == 1 else "batch"
    payload["uploaded_filename"] = uploaded_filename
    payload["uploaded_filenames"] = uploaded_filenames
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["confidence_threshold"] = confidence_threshold
    payload["class_summary"] = _class_summary(payload)
    payload["class_summary_cn"] = _class_summary_cn(payload)

    risk, advice, actions = _risk_and_advice(payload)
    payload["risk_level"] = risk
    payload["advice"] = advice
    payload["recommended_actions"] = actions

    first_item = payload.get("items", [{}])[0] if payload.get("items") else {}
    payload["original_image_url"] = first_item.get("original_image_url")
    payload["result_image_url"] = first_item.get("result_image_url")
    payload["original_image_urls"] = [item.get("original_image_url") for item in payload.get("items", [])]
    payload["result_image_urls"] = [item.get("result_image_url") for item in payload.get("items", [])]
    payload["report_page_url"] = f"/report/{request_id}"

    report_json = Path(payload["report_json"])
    report_markdown = Path(payload["report_markdown"])
    if not report_json.is_absolute():
        report_json = PROJECT_ROOT / report_json
    if not report_markdown.is_absolute():
        report_markdown = PROJECT_ROOT / report_markdown

    payload["report_json_url"] = _to_artifact_url(report_json)
    payload["report_markdown_url"] = _to_artifact_url(report_markdown)
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_markdown.write_text(_readable_markdown(payload), encoding="utf-8")
    return payload


def _load_report_payload(request_id: str) -> dict:
    safe_id = _safe_request_id(request_id)
    work_dir = API_RUNS_ROOT / safe_id
    report_path = work_dir / "reports" / "detection_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Detection record not found")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.setdefault("request_id", safe_id)
    payload.setdefault(
        "created_at",
        datetime.fromtimestamp(report_path.stat().st_mtime).isoformat(timespec="seconds"),
    )

    upload_files = sorted((work_dir / "uploads").glob("*"))
    rendered_files = sorted((work_dir / "rendered").glob("*"))
    rendered_by_stem = {path.stem.replace("_pred", ""): path for path in rendered_files}
    for item in payload.get("items", []):
        image_path = Path(str(item.get("image", "")))
        item.setdefault("uploaded_filename", image_path.name or "-")
        item.setdefault("original_image_url", _to_artifact_url(image_path))
        result_path = rendered_by_stem.get(image_path.stem)
        if result_path:
            item.setdefault("result_image_url", _to_artifact_url(result_path))

    payload.setdefault("image_count", len(payload.get("items", [])))
    payload.setdefault("mode", "single" if int(payload.get("image_count", 0)) <= 1 else "batch")
    payload.setdefault("class_summary", _class_summary(payload))
    payload.setdefault("class_summary_cn", _class_summary_cn(payload))
    risk, advice, actions = _risk_and_advice(payload)
    payload.setdefault("risk_level", risk)
    payload.setdefault("advice", advice)
    payload.setdefault("recommended_actions", actions)

    if upload_files:
        payload.setdefault("original_image_url", _to_artifact_url(upload_files[0]))
    if rendered_files:
        payload.setdefault("result_image_url", _to_artifact_url(rendered_files[0]))
    payload.setdefault("report_json_url", _to_artifact_url(report_path))
    payload.setdefault("report_markdown_url", _to_artifact_url(work_dir / "reports" / "detection_report.md"))
    payload.setdefault("report_page_url", f"/report/{safe_id}")
    _attach_analysis_payload(payload, work_dir)
    return payload


def _attach_analysis_payload(payload: dict, work_dir: Path) -> None:
    report_dir = work_dir / "reports"
    analysis_json = report_dir / "analysis_report.json"
    analysis_markdown = report_dir / "analysis_report.md"
    if not analysis_json.exists():
        payload.setdefault("analysis_report", None)
        return

    try:
        analysis_payload = json.loads(analysis_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload.setdefault("analysis_report", None)
        return

    analysis_payload["analysis_report_json_url"] = _to_artifact_url(analysis_json)
    analysis_payload["analysis_report_markdown_url"] = _to_artifact_url(analysis_markdown)
    payload["analysis_report"] = analysis_payload
    payload["analysis_report_json_url"] = analysis_payload["analysis_report_json_url"]
    payload["analysis_report_markdown_url"] = analysis_payload["analysis_report_markdown_url"]
    _attach_complete_report_payload(payload, work_dir)


def _attach_complete_report_payload(payload: dict, work_dir: Path) -> None:
    report_dir = work_dir / "reports"
    report_paths = write_combined_report(payload, report_dir)
    complete_json = Path(report_paths["complete_report_json"])
    complete_markdown = Path(report_paths["complete_report_markdown"])
    payload["complete_report_json"] = str(complete_json)
    payload["complete_report_markdown"] = str(complete_markdown)
    payload["complete_report_json_url"] = _to_artifact_url(complete_json)
    payload["complete_report_markdown_url"] = _to_artifact_url(complete_markdown)
    if payload.get("analysis_report"):
        payload["analysis_report"]["complete_report_json_url"] = payload["complete_report_json_url"]
        payload["analysis_report"]["complete_report_markdown_url"] = payload["complete_report_markdown_url"]


def _history_item(payload: dict) -> dict:
    request_id = str(payload.get("request_id", ""))
    return {
        "request_id": request_id,
        "short_id": request_id[:8],
        "created_at": payload.get("created_at", "-"),
        "uploaded_filename": payload.get("uploaded_filename", "-"),
        "image_count": int(payload.get("image_count", 0)),
        "total_defects": int(payload.get("total_defects", 0)),
        "risk_level": payload.get("risk_level", "正常"),
        "class_summary": payload.get("class_summary", {}),
        "class_summary_cn": payload.get("class_summary_cn", {}),
        "result_image_url": payload.get("result_image_url"),
        "report_page_url": payload.get("report_page_url", f"/report/{request_id}"),
    }


def _read_latest_metrics() -> dict[str, float | int | str]:
    results_csv = TRAIN_RUN_DIR / "results.csv"
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}

    latest = rows[-1]

    def read_float(name: str) -> float:
        value = latest.get(name, "")
        return round(float(value), 4) if value not in ("", None) else 0.0

    return {
        "epoch": int(float(latest.get("epoch", 0) or 0)),
        "precision": read_float("metrics/precision(B)"),
        "recall": read_float("metrics/recall(B)"),
        "mAP50": read_float("metrics/mAP50(B)"),
        "mAP50_95": read_float("metrics/mAP50-95(B)"),
        "val_box_loss": read_float("val/box_loss"),
        "val_cls_loss": read_float("val/cls_loss"),
        "val_dfl_loss": read_float("val/dfl_loss"),
    }


def _evaluation_payload() -> dict:
    hard_cases = []
    if HARD_CASES_DIR.exists():
        hard_cases = [
            {
                "name": path.name,
                "url": _to_hard_case_url(path),
            }
            for path in sorted(HARD_CASES_DIR.glob("*.jpg"))[:30]
        ]

    artifacts = {
        "results_curve": _to_artifact_url(TRAIN_RUN_DIR / "results.png"),
        "confusion_matrix": _to_artifact_url(TRAIN_RUN_DIR / "confusion_matrix.png"),
        "confusion_matrix_normalized": _to_artifact_url(TRAIN_RUN_DIR / "confusion_matrix_normalized.png"),
        "pr_curve": _to_artifact_url(TRAIN_RUN_DIR / "BoxPR_curve.png"),
        "f1_curve": _to_artifact_url(TRAIN_RUN_DIR / "BoxF1_curve.png"),
        "val_labels": _to_artifact_url(TRAIN_RUN_DIR / "val_batch0_labels.jpg"),
        "val_pred": _to_artifact_url(TRAIN_RUN_DIR / "val_batch0_pred.jpg"),
        "results_csv": _to_artifact_url(TRAIN_RUN_DIR / "results.csv"),
        "missed_report": _to_artifact_url(MISSED_REPORT),
    }
    return {
        "model_version": "v0.1_baseline",
        "model_path": str(TRAIN_RUN_DIR / "weights" / "best.pt"),
        "dataset_config": str(PROJECT_ROOT / "datasets" / "wind_blade_defect" / "data.yaml"),
        "classes": CLASS_LABELS,
        "metrics": _read_latest_metrics(),
        "artifacts": artifacts,
        "hard_cases": hard_cases,
        "hard_cases_dir": str(HARD_CASES_DIR),
    }


def _evaluation_html(payload: dict) -> str:
    metrics = payload.get("metrics", {})
    artifacts = payload.get("artifacts", {})
    class_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(label)}</td></tr>"
        for name, label in payload.get("classes", {}).items()
    )
    hard_case_rows = "".join(
        f"<tr><td><img class=\"thumb\" src=\"{html.escape(str(item.get('url') or ''))}\" alt=\"{html.escape(item['name'])}\"></td><td>{html.escape(item['name'])}</td></tr>"
        for item in payload.get("hard_cases", [])
    ) or '<tr><td colspan="2">暂无困难样本。</td></tr>'

    def metric_card(label: str, key: str) -> str:
        value = metrics.get(key, "-")
        return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'

    def image_card(title: str, url: str | None) -> str:
        if not url:
            return f'<article class="panel"><h2>{title}</h2><p>文件不存在。</p></article>'
        return f'<article class="panel"><h2>{title}</h2><img src="{html.escape(url)}" alt="{title}"></article>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>模型评估 - 风机叶片缺陷检测</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #16202a; background: #f5f7fb; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px; display: grid; gap: 18px; }}
    .panel, .metric {{ background: #fff; border: 1px solid #d7dde7; border-radius: 8px; padding: 18px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.7; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
    .label {{ color: #637081; font-size: 12px; margin-bottom: 6px; }}
    .value {{ font-size: 24px; font-weight: 800; overflow-wrap: anywhere; }}
    .gallery {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    img {{ width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #d7dde7; border-radius: 6px; background: #eef2f7; }}
    .thumb {{ width: 96px; height: 72px; object-fit: contain; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #d7dde7; text-align: left; }}
    a {{ color: #2563eb; text-decoration: none; }}
    @media (max-width: 900px) {{ .metrics, .gallery {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <a href="/">返回检测页面</a>
    <section class="panel">
      <h1>模型评估与结果汇总</h1>
      <p>当前模型版本：{html.escape(str(payload.get('model_version', '-')))}</p>
      <p>模型文件：{html.escape(str(payload.get('model_path', '-')))}</p>
    </section>
    <section class="metrics">
      {metric_card("Epoch", "epoch")}
      {metric_card("Precision", "precision")}
      {metric_card("Recall", "recall")}
      {metric_card("mAP50", "mAP50")}
      {metric_card("mAP50-95", "mAP50_95")}
    </section>
    <section class="gallery">
      {image_card("训练曲线 results.png", artifacts.get("results_curve"))}
      {image_card("混淆矩阵", artifacts.get("confusion_matrix"))}
      {image_card("PR 曲线", artifacts.get("pr_curve"))}
      {image_card("验证集预测示例", artifacts.get("val_pred"))}
      {image_card("验证集真实标注示例", artifacts.get("val_labels"))}
      {image_card("F1 曲线", artifacts.get("f1_curve"))}
    </section>
    <section class="panel">
      <h2>检测类别</h2>
      <table><thead><tr><th>类别</th><th>中文名称</th></tr></thead><tbody>{class_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>困难样本与漏检分析</h2>
      <p>困难样本目录：{html.escape(str(payload.get('hard_cases_dir', '-')))}</p>
      <p><a href="{html.escape(str(artifacts.get('missed_report') or '#'))}" target="_blank" rel="noreferrer">打开漏检样本 CSV 报告</a></p>
      <table><thead><tr><th>预览</th><th>样本文件</th></tr></thead><tbody>{hard_case_rows}</tbody></table>
    </section>
  </main>
  <script>
    document.querySelectorAll(".keyframe img").forEach((img) => {{
      img.addEventListener("click", () => {{
        if (img.currentSrc || img.src) {{
          window.open(img.currentSrc || img.src, "_blank", "noopener");
        }}
      }});
    }});
  </script>
</body>
</html>"""


def _report_html(payload: dict) -> str:
    detections = _detections(payload)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(d.get('image_name', d.get('uploaded_filename', '-'))))}</td>"
        f"<td>{html.escape(str(d.get('class_name', '-')))}</td>"
        f"<td>{html.escape(str(CLASS_LABELS.get(str(d.get('class_name', '')), d.get('class_name', '-'))))}</td>"
        f"<td>{float(d.get('confidence', 0.0)):.3f}</td>"
        f"<td>{html.escape(str(d.get('xyxy', [])))}</td>"
        "</tr>"
        for d in [
            {**detection, "image_name": item.get("uploaded_filename", "-")}
            for item in payload.get("items", [])
            for detection in item.get("detections", [])
        ]
    ) or '<tr><td colspan="5">未检测到缺陷。</td></tr>'

    actions = "".join(f"<li>{html.escape(str(item))}</li>" for item in payload.get("recommended_actions", []))
    cards = "".join(
        f"""
        <article class="image-card">
          <h3>{html.escape(str(item.get('uploaded_filename', '-')))}</h3>
          <p>缺陷数量：{int(item.get('defect_count', 0))}</p>
          <div class="image-pair">
            <img src="{html.escape(str(item.get('original_image_url') or ''))}" alt="原图">
            <img src="{html.escape(str(item.get('result_image_url') or ''))}" alt="检测结果图">
          </div>
        </article>
        """
        for item in payload.get("items", [])
    )
    request_id = html.escape(str(payload.get("request_id", "-")))
    short_id = html.escape(request_id[:8])
    filename = html.escape(str(payload.get("uploaded_filename", "-")))
    risk = html.escape(str(payload.get("risk_level", "正常")))
    advice = html.escape(str(payload.get("advice", "")))
    export_links = "".join(
        f'<a href="{html.escape(str(url))}" target="_blank" rel="noreferrer">{html.escape(label)}</a>'
        for label, url in [
            ("完整报告", payload.get("complete_report_markdown_url")),
            ("完整JSON", payload.get("complete_report_json_url")),
            ("检测JSON", payload.get("report_json_url")),
            ("检测Markdown", payload.get("report_markdown_url")),
            ("分析报告", payload.get("analysis_report_markdown_url")),
        ]
        if url
    )
    analysis = payload.get("analysis_report") or {}
    analysis_section = ""
    if analysis:
        causes = "".join(f"<li>{html.escape(str(item))}</li>" for item in analysis.get("possible_causes", []))
        maintenance = "".join(f"<li>{html.escape(str(item))}</li>" for item in analysis.get("maintenance_advice", []))
        reasons = "".join(f"<li>{html.escape(str(item))}</li>" for item in analysis.get("risk_reasons", []))
        analysis_section = f"""
    <section class="panel">
      <h2>故障分析与运维建议</h2>
      <p><strong>风险等级：</strong>{html.escape(str(analysis.get('risk_level', '-')))}</p>
      <p><strong>故障摘要：</strong>{html.escape(str(analysis.get('fault_summary', '-')))}</p>
      <p><strong>故障现象：</strong>{html.escape(str(analysis.get('fault_description', '-')))}</p>
      <h3>可能原因</h3>
      <ul>{causes or '<li>-</li>'}</ul>
      <h3>风险影响</h3>
      <p>{html.escape(str(analysis.get('risk_impact', '-')))}</p>
      <h3>运维建议</h3>
      <ul>{maintenance or '<li>-</li>'}</ul>
      <h3>复检要求</h3>
      <p>{html.escape(str(analysis.get('recheck_requirement', '-')))}</p>
      <h3>是否建议停机</h3>
      <p>{html.escape(str(analysis.get('shutdown_suggestion', '-')))}</p>
      <h3>风险判断依据</h3>
      <ul>{reasons or '<li>-</li>'}</ul>
      <p>{html.escape(str(analysis.get('manual_review_notice', '-')))}</p>
    </section>
        """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>检测报告 {short_id}</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #16202a; background: #f5f7fb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; display: grid; gap: 18px; }}
    .panel, .metric, .image-card {{ background: #fff; border: 1px solid #d7dde7; border-radius: 8px; padding: 18px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.7; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
    .label {{ color: #637081; font-size: 12px; margin-bottom: 6px; }}
    .value {{ font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }}
    .cards {{ display: grid; gap: 16px; }}
    .image-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    img {{ width: 100%; max-height: 420px; object-fit: contain; border: 1px solid #d7dde7; border-radius: 6px; background: #eef2f7; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #d7dde7; text-align: left; vertical-align: top; }}
    th {{ color: #637081; }}
    a {{ color: #2563eb; text-decoration: none; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    @media (max-width: 900px) {{ .grid, .image-pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <a href="/">返回检测页面</a>
    <section class="grid">
      <div class="metric"><div class="label">请求 ID</div><div class="value">{short_id}</div></div>
      <div class="metric"><div class="label">图片名称</div><div class="value">{filename}</div></div>
      <div class="metric"><div class="label">图片数量</div><div class="value">{payload.get('image_count', 0)}</div></div>
      <div class="metric"><div class="label">缺陷总数</div><div class="value">{payload.get('total_defects', 0)}</div></div>
      <div class="metric"><div class="label">风险等级</div><div class="value">{risk}</div></div>
    </section>
    <section class="panel">
      <h1>风机叶片缺陷检测报告</h1>
      <p>{advice}</p>
      <p class="links">{export_links}</p>
      <ul>{actions}</ul>
    </section>
    {analysis_section}
    <section class="cards">{cards}</section>
    <section class="panel">
      <h2>缺陷明细</h2>
      <table>
        <thead><tr><th>图片</th><th>类别</th><th>中文类别</th><th>置信度</th><th>边框坐标</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def _predict_payload(saved_uploads: list[dict], work_dir: Path, confidence_threshold: float) -> dict:
    source = saved_uploads[0]["path"] if len(saved_uploads) == 1 else work_dir / "uploads"
    results = predict_yolo(
        model=_model_path(),
        source=source,
        conf=confidence_threshold,
        project=str(work_dir),
        name="predict",
        save=True,
        device=os.getenv("WIND_DEVICE") or None,
    )
    results_list = list(results)
    rendered_by_image = _render_result_images(results_list, work_dir / "rendered")
    report_payload = write_prediction_report(results_list, work_dir / "reports")
    return _finalize_report(
        payload=report_payload,
        work_dir=work_dir,
        request_id=work_dir.name,
        saved_uploads=saved_uploads,
        confidence_threshold=confidence_threshold,
        rendered_by_image=rendered_by_image,
    )


def _finalize_video_payload(
    payload: dict,
    work_dir: Path,
    request_id: str,
    saved_upload: dict,
    confidence_threshold: float,
) -> dict:
    preview_dir = work_dir / "video_preview"
    original_preview = _make_video_preview(saved_upload["path"], preview_dir / "original_preview.mp4")
    result_preview = _make_video_preview(payload.get("output_video"), preview_dir / "result_preview.mp4")

    payload["request_id"] = request_id
    payload["mode"] = "video"
    payload["uploaded_filename"] = saved_upload["uploaded_filename"]
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["confidence_threshold"] = confidence_threshold
    payload["raw_original_video_url"] = _to_artifact_url(saved_upload["path"])
    payload["raw_result_video_url"] = _to_artifact_url(payload.get("output_video"))
    payload["original_video_preview"] = original_preview
    payload["result_video_preview"] = result_preview
    payload["original_video_url"] = _to_artifact_url(original_preview or saved_upload["path"])
    payload["result_video_url"] = _to_artifact_url(result_preview or payload.get("output_video"))
    payload["report_page_url"] = f"/report/{request_id}"

    for keyframe in payload.get("keyframes", []):
        keyframe["image_url"] = _to_artifact_url(keyframe.get("image_path"))

    report_dir = work_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "detection_report.json"
    report_markdown = report_dir / "detection_report.md"
    payload["report_json"] = str(report_json)
    payload["report_markdown"] = str(report_markdown)
    payload["report_json_url"] = _to_artifact_url(report_json)
    payload["report_markdown_url"] = _to_artifact_url(report_markdown)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_markdown.write_text(_video_markdown(payload), encoding="utf-8")
    return payload


def _video_markdown(payload: dict) -> str:
    lines = [
        "# 风机叶片视频缺陷检测报告",
        "",
        f"- 请求 ID：{payload.get('request_id', '-')}",
        f"- 视频名称：{payload.get('uploaded_filename', '-')}",
        f"- 检测时间：{payload.get('created_at', '-')}",
        f"- 置信度阈值：{payload.get('confidence_threshold', '-')}",
        f"- 总帧数：{payload.get('total_frames', 0)}",
        f"- 实际处理帧数：{payload.get('processed_frames', 0)}",
        f"- 缺陷帧数：{payload.get('defect_frame_count', 0)}",
        f"- 缺陷总数：{payload.get('total_defects', 0)}",
        f"- 风险等级：{payload.get('risk_level', '-')}",
        "",
        "## 处理建议",
        "",
        str(payload.get("advice", "-")),
        "",
    ]
    for action in payload.get("recommended_actions", []):
        lines.append(f"- {action}")

    lines.extend(["", "## 缺陷统计", ""])
    if payload.get("class_summary"):
        for name, count in payload["class_summary"].items():
            lines.append(f"- {CLASS_LABELS.get(name, name)}（{name}）：{count}")
    else:
        lines.append("- 无")

    lines.extend(["", "## 关键帧", ""])
    if payload.get("keyframes"):
        for item in payload["keyframes"]:
            lines.append(
                f"- 第 {item.get('frame_index')} 帧，"
                f"{item.get('time_seconds')} 秒，"
                f"缺陷数量 {item.get('defect_count')}，"
                f"截图 {item.get('image_path')}"
            )
    else:
        lines.append("- 未生成关键帧")
    return "\n".join(lines) + "\n"


def _video_report_html(payload: dict) -> str:
    keyframes = "".join(
        f"""
        <article class="keyframe">
          <img src="{html.escape(str(item.get('image_url') or ''))}" alt="关键帧">
          <p>第 {item.get('frame_index')} 帧，{item.get('time_seconds')} 秒，{item.get('defect_count')} 个缺陷</p>
        </article>
        """
        for item in payload.get("keyframes", [])
    ) or "<p>未生成关键帧。</p>"
    if payload.get("keyframes"):
        keyframes += """
        <script>
          document.querySelectorAll(".keyframe img").forEach((img) => {
            img.addEventListener("click", () => {
              if (img.currentSrc || img.src) {
                window.open(img.currentSrc || img.src, "_blank", "noopener");
              }
            });
          });
        </script>
        """

    rows = "".join(
        f"<tr><td>{frame.get('frame_index')}</td><td>{frame.get('time_seconds')}</td><td>{frame.get('defect_count')}</td></tr>"
        for frame in payload.get("frames", [])
    ) or '<tr><td colspan="3">未检测到缺陷帧。</td></tr>'
    export_links = "".join(
        f'<a href="{html.escape(str(url))}" target="_blank" rel="noreferrer">{html.escape(label)}</a>'
        for label, url in [
            ("完整报告", payload.get("complete_report_markdown_url")),
            ("完整JSON", payload.get("complete_report_json_url")),
            ("检测JSON", payload.get("report_json_url")),
            ("检测Markdown", payload.get("report_markdown_url")),
            ("分析报告", payload.get("analysis_report_markdown_url")),
        ]
        if url
    )
    analysis = payload.get("analysis_report") or {}
    analysis_section = ""
    if analysis:
        causes = "".join(f"<li>{html.escape(str(item))}</li>" for item in analysis.get("possible_causes", []))
        maintenance = "".join(f"<li>{html.escape(str(item))}</li>" for item in analysis.get("maintenance_advice", []))
        analysis_section = f"""
    <section class="panel">
      <h2>故障分析与运维建议</h2>
      <p><strong>风险等级：</strong>{html.escape(str(analysis.get('risk_level', '-')))}</p>
      <p><strong>故障摘要：</strong>{html.escape(str(analysis.get('fault_summary', '-')))}</p>
      <p><strong>故障现象：</strong>{html.escape(str(analysis.get('fault_description', '-')))}</p>
      <h3>可能原因</h3>
      <ul>{causes or '<li>-</li>'}</ul>
      <h3>风险影响</h3>
      <p>{html.escape(str(analysis.get('risk_impact', '-')))}</p>
      <h3>运维建议</h3>
      <ul>{maintenance or '<li>-</li>'}</ul>
      <h3>复检要求</h3>
      <p>{html.escape(str(analysis.get('recheck_requirement', '-')))}</p>
      <h3>是否建议停机</h3>
      <p>{html.escape(str(analysis.get('shutdown_suggestion', '-')))}</p>
      <p>{html.escape(str(analysis.get('manual_review_notice', '-')))}</p>
    </section>
        """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>视频检测报告 {html.escape(str(payload.get('request_id', ''))[:8])}</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #16202a; background: #f5f7fb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; display: grid; gap: 18px; }}
    .panel, .metric, .keyframe {{ background: #fff; border: 1px solid #d7dde7; border-radius: 8px; padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
    .videos, .keyframes {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .label {{ color: #637081; font-size: 12px; margin-bottom: 6px; }}
    .value {{ font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }}
    video, img {{ width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #d7dde7; border-radius: 6px; background: #eef2f7; }}
    .keyframe img {{ cursor: pointer; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #d7dde7; text-align: left; }}
    a {{ color: #2563eb; text-decoration: none; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    @media (max-width: 900px) {{ .grid, .videos, .keyframes {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <a href="/">返回检测页面</a>
    <section class="grid">
      <div class="metric"><div class="label">请求 ID</div><div class="value">{html.escape(str(payload.get('request_id', ''))[:8])}</div></div>
      <div class="metric"><div class="label">视频名称</div><div class="value">{html.escape(str(payload.get('uploaded_filename', '-')))}</div></div>
      <div class="metric"><div class="label">处理帧数</div><div class="value">{payload.get('processed_frames', 0)}</div></div>
      <div class="metric"><div class="label">缺陷总数</div><div class="value">{payload.get('total_defects', 0)}</div></div>
      <div class="metric"><div class="label">风险等级</div><div class="value">{html.escape(str(payload.get('risk_level', '-')))}</div></div>
    </section>
    <section class="panel">
      <h1>风机叶片视频缺陷检测报告</h1>
      <p>{html.escape(str(payload.get('advice', '-')))}</p>
      <p class="links">{export_links}</p>
    </section>
    {analysis_section}
    <section class="videos">
      <div class="panel"><h2>原视频</h2><video src="{html.escape(str(payload.get('original_video_url') or ''))}" controls></video></div>
      <div class="panel"><h2>检测结果视频</h2><video src="{html.escape(str(payload.get('result_video_url') or ''))}" controls></video></div>
    </section>
    <section class="panel">
      <h2>关键帧</h2>
      <div class="keyframes">{keyframes}</div>
    </section>
    <section class="panel">
      <h2>缺陷帧明细</h2>
      <table><thead><tr><th>帧号</th><th>时间秒</th><th>缺陷数量</th></tr></thead><tbody>{rows}</tbody></table>
    </section>
  </main>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    index_path = STATIC_ROOT / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/history")
def history(limit: int = Query(default=30, ge=1, le=200)) -> dict:
    items: list[dict] = []
    for run_dir in sorted(API_RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        report_path = run_dir / "reports" / "detection_report.json"
        if not report_path.exists():
            continue
        try:
            items.append(_history_item(_load_report_payload(run_dir.name)))
        except Exception:
            continue
        if len(items) >= limit:
            break
    return {"items": items}


@app.get("/evaluation", response_class=HTMLResponse)
def evaluation_page() -> str:
    return _evaluation_html(_evaluation_payload())


@app.get("/evaluation/json")
def evaluation_json() -> dict:
    return _evaluation_payload()


@app.get("/report/{request_id}", response_class=HTMLResponse)
def report_page(request_id: str) -> str:
    payload = _load_report_payload(request_id)
    if payload.get("mode") == "video":
        return _video_report_html(payload)
    return _report_html(payload)


@app.get("/report/{request_id}/json")
def report_json(request_id: str) -> dict:
    return _load_report_payload(request_id)


@app.post("/analyze/{request_id}")
def analyze(request_id: str) -> dict:
    safe_id = _safe_request_id(request_id)
    work_dir = API_RUNS_ROOT / safe_id
    payload = _load_report_payload(safe_id)
    analysis_payload = analyze_detection_report(payload, work_dir)
    analysis_json = work_dir / "reports" / "analysis_report.json"
    analysis_markdown = work_dir / "reports" / "analysis_report.md"
    analysis_payload["analysis_report_json_url"] = _to_artifact_url(analysis_json)
    analysis_payload["analysis_report_markdown_url"] = _to_artifact_url(analysis_markdown)
    payload["analysis_report"] = analysis_payload
    _attach_complete_report_payload(payload, work_dir)
    analysis_payload["complete_report_json_url"] = payload.get("complete_report_json_url")
    analysis_payload["complete_report_markdown_url"] = payload.get("complete_report_markdown_url")
    return analysis_payload


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float | None = None) -> dict:
    confidence_threshold = conf if conf is not None else float(os.getenv("WIND_CONF", "0.45"))
    request_id = uuid4().hex
    work_dir = API_RUNS_ROOT / request_id
    saved_uploads = await _save_uploads([file], work_dir / "uploads")
    return _predict_payload(saved_uploads, work_dir, confidence_threshold)


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...), conf: float | None = None) -> dict:
    confidence_threshold = conf if conf is not None else float(os.getenv("WIND_CONF", "0.45"))
    request_id = uuid4().hex
    work_dir = API_RUNS_ROOT / request_id
    saved_uploads = await _save_uploads(files, work_dir / "uploads")
    return _predict_payload(saved_uploads, work_dir, confidence_threshold)


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float | None = None) -> dict:
    confidence_threshold = conf if conf is not None else float(os.getenv("WIND_CONF", "0.45"))
    request_id = uuid4().hex
    work_dir = API_RUNS_ROOT / request_id
    saved_upload = (await _save_uploads([file], work_dir / "uploads"))[0]
    payload = run_video_detection(
        model=_model_path(),
        source=saved_upload["path"],
        output_root=work_dir,
        name="video_predict",
        conf=confidence_threshold,
        device=os.getenv("WIND_DEVICE") or None,
        max_keyframes=12,
    )
    return _finalize_video_payload(
        payload=payload,
        work_dir=work_dir,
        request_id=request_id,
        saved_upload=saved_upload,
        confidence_threshold=confidence_threshold,
    )
