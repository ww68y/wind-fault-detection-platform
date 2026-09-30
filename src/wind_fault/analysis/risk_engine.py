from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .report_loader import CLASS_LABELS, class_counts, confidence_values


RISK_ORDER = ["正常", "待复核", "低风险", "中风险", "中高风险", "高风险"]
CLASS_BASE_RISK = {
    "corrosion": "低风险",
    "spalling": "中风险",
    "hole": "中高风险",
    "crack": "中高风险",
}


@dataclass(frozen=True)
class RiskResult:
    risk_level: str
    confidence_level: str
    reasons: list[str]
    class_summary: dict[str, int]
    class_summary_cn: dict[str, int]
    total_defects: int
    max_confidence: float
    min_confidence: float
    needs_manual_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "confidence_level": self.confidence_level,
            "risk_reasons": self.reasons,
            "class_summary": self.class_summary,
            "class_summary_cn": self.class_summary_cn,
            "total_defects": self.total_defects,
            "max_confidence": round(self.max_confidence, 6),
            "min_confidence": round(self.min_confidence, 6),
            "needs_manual_review": self.needs_manual_review,
        }


def _rank(level: str) -> int:
    return RISK_ORDER.index(level) if level in RISK_ORDER else 0


def _level(rank: int) -> str:
    rank = max(0, min(rank, len(RISK_ORDER) - 1))
    return RISK_ORDER[rank]


def _confidence_level(max_confidence: float, min_confidence: float, has_defects: bool) -> str:
    if not has_defects:
        return "无缺陷检测结果"
    if min_confidence < 0.6:
        return "存在低置信度结果，建议人工复核"
    if max_confidence >= 0.9:
        return "检测可信度较高"
    return "检测可信度中等"


def assess_risk(detections: list[dict[str, Any]]) -> RiskResult:
    counts = class_counts(detections)
    counts_cn = {CLASS_LABELS.get(name, name): count for name, count in counts.items()}
    total = len(detections)
    confidences = confidence_values(detections)
    max_confidence = max(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0

    if total == 0:
        return RiskResult(
            risk_level="正常",
            confidence_level="无缺陷检测结果",
            reasons=["当前检测任务在设定阈值下未发现缺陷。"],
            class_summary=counts,
            class_summary_cn=counts_cn,
            total_defects=0,
            max_confidence=0.0,
            min_confidence=0.0,
            needs_manual_review=False,
        )

    rank = 1
    reasons: list[str] = []
    for class_name, count in counts.items():
        base = CLASS_BASE_RISK.get(class_name, "待复核")
        rank = max(rank, _rank(base))
        reasons.append(f"检测到 {CLASS_LABELS.get(class_name, class_name)}（{class_name}）{count} 个。")

    if total >= 3:
        rank += 1
        reasons.append("缺陷数量达到 3 个及以上，风险等级提升一级。")

    if len(counts) >= 2:
        rank += 1
        reasons.append("同一任务中出现多种缺陷类型，风险等级提升一级。")

    high_conf_count = sum(1 for value in confidences if value >= 0.9)
    if high_conf_count >= 2:
        rank = max(rank, _rank("高风险"))
        reasons.append("多个高置信度缺陷同时出现，按高风险处理。")
    elif max_confidence >= 0.9:
        reasons.append("存在高置信度检测结果，检测可信度较高。")

    needs_manual_review = min_confidence < 0.6
    if needs_manual_review:
        reasons.append("存在置信度低于 0.60 的检测结果，必须人工复核。")
        if rank <= _rank("低风险"):
            rank = _rank("待复核")

    return RiskResult(
        risk_level=_level(rank),
        confidence_level=_confidence_level(max_confidence, min_confidence, True),
        reasons=reasons,
        class_summary=counts,
        class_summary_cn=counts_cn,
        total_defects=total,
        max_confidence=max_confidence,
        min_confidence=min_confidence,
        needs_manual_review=needs_manual_review,
    )
