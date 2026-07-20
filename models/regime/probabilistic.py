from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


REGIME_TEMPERATURE = 1.0


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=float) / temperature
    logits = logits - np.nanmax(logits)
    exp = np.exp(logits)
    denom = exp.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return exp / denom


@dataclass
class RegimeFeatures:
    above_sma: float
    trend_score: float
    breadth_score: float
    risk_score: float
    macro_score: float
    news_bias: float
    suppression_score: float


@dataclass
class RegimeProbabilities:
    bull: float
    chop: float
    bear: float
    confidence: float
    transition_risk: float
    instability: float
    regime_score: float
    selected_regime: str


def infer_regime_probabilities(
    features: RegimeFeatures,
    previous: Optional[RegimeProbabilities] = None,
) -> RegimeProbabilities:
    bull_logit = (
        1.35 * features.above_sma
        + 1.10 * features.trend_score
        + 0.65 * features.breadth_score
        + 0.45 * features.risk_score
        + 0.35 * np.tanh(features.macro_score * 8.0)
        + 0.30 * features.news_bias
        - 0.40 * features.suppression_score
    )
    bear_logit = (
        -1.35 * features.above_sma
        - 0.90 * features.trend_score
        - 0.55 * features.breadth_score
        - 0.70 * features.risk_score
        - 0.30 * np.tanh(features.macro_score * 8.0)
        - 0.20 * features.news_bias
        + 0.55 * features.suppression_score
    )
    chop_logit = (
        -0.55 * abs(features.above_sma)
        -0.45 * abs(features.trend_score)
        -0.30 * abs(features.breadth_score)
        +0.30 * (1.0 - abs(features.risk_score))
        +0.15 * (1.0 - abs(features.news_bias))
        +0.20 * features.suppression_score
    )

    temperature = REGIME_TEMPERATURE
    probs = _softmax(np.array([bull_logit, chop_logit, bear_logit], dtype=float), temperature)
    bull, chop, bear = map(float, probs)

    ordered = sorted([bull, chop, bear], reverse=True)
    top_prob = ordered[0]
    second_prob = ordered[1]
    confidence = _clip(top_prob - second_prob + 0.35 * top_prob, 0.0, 1.0)
    instability = _clip(1.0 - abs(bull - bear), 0.0, 1.0)

    if previous is None:
        transition_risk = _clip(0.35 * instability + 0.35 * features.suppression_score + 0.20 * (1.0 - confidence), 0.0, 1.0)
    else:
        prior = np.array([previous.bull, previous.chop, previous.bear], dtype=float)
        current = np.array([bull, chop, bear], dtype=float)
        shift = float(np.abs(current - prior).sum() / 2.0)
        transition_risk = _clip(
            0.45 * shift
            + 0.20 * instability
            + 0.20 * features.suppression_score
            + 0.15 * (1.0 - confidence),
            0.0,
            1.0,
        )

    selected_regime = ["BULL", "CHOP", "BEAR"][int(np.argmax(probs))]
    regime_score = bull - bear

    return RegimeProbabilities(
        bull=bull,
        chop=chop,
        bear=bear,
        confidence=confidence,
        transition_risk=transition_risk,
        instability=instability,
        regime_score=float(regime_score),
        selected_regime=selected_regime,
    )

