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


def score_news_regime_context(row: Optional[pd.Series]) -> NewsRegimeContext:
    if row is None or len(row) == 0:
        return NewsRegimeContext(0.0, 0.0, 0.0, "No news context")

    regime_bias = _weighted_sum(row, REGIME_CONTEXT_WEIGHTS)
    suppression_score = max(0.0, _weighted_sum(row, SUPPRESSION_WEIGHTS))

    confidence_delta = 0.0
    if row.get("news_stress_flag", 0) > 0:
        confidence_delta -= min(0.25, 0.10 + suppression_score * 0.10)
    else:
        confidence_delta += min(0.15, abs(regime_bias) * 0.08)

    reason_parts = []
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

