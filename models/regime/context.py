from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.news import REGIME_CONTEXT_WEIGHTS, SUPPRESSION_WEIGHTS


@dataclass
class NewsRegimeContext:
    regime_bias: float
    confidence_delta: float
    suppression_score: float
    reason: str


def _weighted_sum(row: pd.Series, weights: dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        val = float(row.get(key, 0.0))
        if np.isfinite(val):
            total += weight * np.tanh(val)
    return float(total)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def score_news_regime_context(row: Optional[pd.Series]) -> NewsRegimeContext:
    if row is None or len(row) == 0:
        return NewsRegimeContext(0.0, 0.0, 0.0, "No news context")

    regime_bias = _weighted_sum(row, REGIME_CONTEXT_WEIGHTS)
    suppression_score = max(0.0, _weighted_sum(row, SUPPRESSION_WEIGHTS))
    reason_parts = []

    raw_dexter_status = row.get("dexter_dexter_status", "")
    dexter_status = "" if pd.isna(raw_dexter_status) else str(raw_dexter_status).lower()
    dexter_hallucination_risk = _safe_float(row.get("dexter_hallucination_risk_score", 1.0), 1.0)
    dexter_confidence = _safe_float(row.get("dexter_overall_confidence", 0.0), 0.0)
    dexter_is_usable = (
        dexter_status == "ok"
        and np.isfinite(dexter_hallucination_risk)
        and dexter_hallucination_risk < 0.65
        and np.isfinite(dexter_confidence)
        and dexter_confidence > 0.0
    )

    if dexter_is_usable:
        india_bias = _safe_float(row.get("dexter_india_research_bias_effective", row.get("dexter_india_research_bias", 0.0)))
        domestic_vulnerability = _safe_float(row.get("dexter_domestic_vulnerability_effective", 0.0))
        global_spillover = _safe_float(row.get("dexter_global_spillover_risk_effective", 0.0))
        combined_risk = _safe_float(row.get("dexter_combined_risk_alert_effective", 0.0))
        spillover_amplifier = _safe_float(row.get("dexter_spillover_amplifier_effective", 0.0))

        # India gets priority; global risk only affects the model through
        # explicit spillover and domestic-vulnerability channels.
        dexter_bias = (
            0.25 * np.tanh(india_bias)
            - 0.12 * np.tanh(combined_risk)
            - 0.10 * np.tanh(spillover_amplifier)
        ) * dexter_confidence
        dexter_suppression = (
            0.35 * combined_risk
            + 0.25 * domestic_vulnerability * global_spillover
            + 0.15 * spillover_amplifier
        ) * dexter_confidence

        regime_bias += float(dexter_bias)
        suppression_score += max(0.0, float(dexter_suppression))
        if combined_risk > 0.25:
            reason_parts.append("Dexter India/global risk")
    elif dexter_status:
        reason_parts.append("Dexter ignored: stale or weak evidence")

    confidence_delta = 0.0
    if row.get("news_stress_flag", 0) > 0:
        confidence_delta -= min(0.25, 0.10 + suppression_score * 0.10)
    else:
        confidence_delta += min(0.15, abs(regime_bias) * 0.08)

    if float(row.get("attention_shock_z", 0.0)) > 1.0:
        reason_parts.append("attention shock")
    if float(row.get("policy_share_z", 0.0)) > 1.0:
        reason_parts.append("policy heavy")
    if float(row.get("conflict_share_z", 0.0)) > 1.0:
        reason_parts.append("conflict stress")
    if float(row.get("negative_tone_z", 0.0)) > 1.0:
        reason_parts.append("negative tone")

    reason = ", ".join(reason_parts) if reason_parts else "Neutral news backdrop"
    return NewsRegimeContext(
        regime_bias=float(np.clip(regime_bias, -1.0, 1.0)),
        confidence_delta=float(np.clip(confidence_delta, -0.25, 0.15)),
        suppression_score=float(np.clip(suppression_score, 0.0, 1.0)),
        reason=reason,
    )
