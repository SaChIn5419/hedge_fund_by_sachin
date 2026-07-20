"""
ML-based Regime Classifier with EMA Smoothing
Replaces the hand-tuned probabilistic model with XGBoost + sticky EMA smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import xgboost as xgb
import joblib


@dataclass
class RegimeFeatures:
    """Legacy compatibility."""
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


class MLRegimeClassifier:
    """
    XGBoost Regime Classifier with EMA + Sticky Hysteresis.
    
    Layer 1: XGBoost produces raw P(BULL), P(CHOP), P(BEAR) from 27 market features
    Layer 2: EMA smooths probabilities to avoid reacting to noise
    Layer 3: Sticky hysteresis requires a threshold gap to switch regimes
    """
    
    EMA_ALPHA = 0.35          # Smoothing factor (lower = smoother, stickier)
    SWITCH_THRESHOLD = 0.10   # Must exceed current regime by this much to switch
    
    def __init__(self, 
                 xgb_path: str = 'models/regime/xgb_regime.json',
                 feature_cols_path: str = 'models/regime/regime_feature_cols.pkl'):
        
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(xgb_path)
        self.feature_cols = joblib.load(feature_cols_path)
        
        # Running state
        self.smoothed_probs = np.array([0.33, 0.34, 0.33])  # [BULL, CHOP, BEAR]
        self.current_regime = 'CHOP'
        self.prev_probs: Optional[RegimeProbabilities] = None
        self.step_count = 0
        
    def predict(self, features: Dict[str, float]) -> RegimeProbabilities:
        """Predict regime with EMA smoothing and sticky hysteresis."""
        # Build feature vector
        feature_vec = np.array([[features.get(col, 0.0) for col in self.feature_cols]], dtype=np.float32)
        feature_vec = np.nan_to_num(feature_vec, nan=0.0)
        
        # XGBoost raw probabilities [P(BULL), P(CHOP), P(BEAR)]
        raw_probs = self.xgb_model.predict_proba(feature_vec)[0]
        
        # EMA Smoothing
        if self.step_count == 0:
            self.smoothed_probs = raw_probs.copy()
        else:
            self.smoothed_probs = (
                self.EMA_ALPHA * raw_probs + 
                (1 - self.EMA_ALPHA) * self.smoothed_probs
            )
        
        bull, chop, bear = float(self.smoothed_probs[0]), float(self.smoothed_probs[1]), float(self.smoothed_probs[2])
        
        # Sticky Hysteresis: only switch if the new best regime EXCEEDS current by threshold
        regime_map = {'BULL': 0, 'CHOP': 1, 'BEAR': 2}
        current_prob = self.smoothed_probs[regime_map[self.current_regime]]
        best_idx = np.argmax(self.smoothed_probs)
        best_prob = self.smoothed_probs[best_idx]
        
        if best_prob > current_prob + self.SWITCH_THRESHOLD:
            self.current_regime = ['BULL', 'CHOP', 'BEAR'][best_idx]
        
        # Confidence: gap between top and second
        ordered = sorted([bull, chop, bear], reverse=True)
        confidence = float(np.clip(ordered[0] - ordered[1] + 0.35 * ordered[0], 0.0, 1.0))
        
        # Instability
        instability = float(np.clip(1.0 - abs(bull - bear), 0.0, 1.0))
        
        # Transition risk
        if self.prev_probs is not None:
            prior = np.array([self.prev_probs.bull, self.prev_probs.chop, self.prev_probs.bear])
            current = np.array([bull, chop, bear])
            shift = float(np.abs(current - prior).sum() / 2.0)
            transition_risk = float(np.clip(
                0.45 * shift + 0.20 * instability + 0.15 * (1.0 - confidence),
                0.0, 1.0
            ))
        else:
            transition_risk = float(np.clip(0.35 * instability + 0.20 * (1.0 - confidence), 0.0, 1.0))
        
        self.step_count += 1
        
        result = RegimeProbabilities(
            bull=bull, chop=chop, bear=bear,
            confidence=confidence,
            transition_risk=transition_risk,
            instability=instability,
            regime_score=float(bull - bear),
            selected_regime=self.current_regime,
        )
        self.prev_probs = result
        return result


# Global singleton
_classifier: Optional[MLRegimeClassifier] = None

def get_ml_regime_classifier() -> MLRegimeClassifier:
    global _classifier
    if _classifier is None:
        _classifier = MLRegimeClassifier()
    return _classifier


def reset_ml_regime_classifier():
    """Reset the global classifier (call before each new simulation run)."""
    global _classifier
    _classifier = None


def infer_regime_probabilities(
    features: RegimeFeatures,
    previous: Optional[RegimeProbabilities] = None,
) -> RegimeProbabilities:
    """Legacy fallback — used when ML models are not loaded."""
    return _legacy_infer(features, previous)


def _legacy_infer(features: RegimeFeatures, previous: Optional[RegimeProbabilities] = None) -> RegimeProbabilities:
    """Original hand-tuned logistic regime model (fallback)."""
    bull_logit = (
        1.35 * features.above_sma + 1.10 * features.trend_score
        + 0.65 * features.breadth_score + 0.45 * features.risk_score
        + 0.35 * np.tanh(features.macro_score * 8.0) + 0.30 * features.news_bias
        - 0.40 * features.suppression_score
    )
    bear_logit = (
        -1.35 * features.above_sma - 0.90 * features.trend_score
        - 0.55 * features.breadth_score - 0.70 * features.risk_score
        - 0.30 * np.tanh(features.macro_score * 8.0) - 0.20 * features.news_bias
        + 0.55 * features.suppression_score
    )
    chop_logit = (
        -0.55 * abs(features.above_sma) - 0.45 * abs(features.trend_score)
        - 0.30 * abs(features.breadth_score) + 0.30 * (1.0 - abs(features.risk_score))
        + 0.15 * (1.0 - abs(features.news_bias)) + 0.20 * features.suppression_score
    )
    logits = np.array([bull_logit, chop_logit, bear_logit], dtype=float)
    logits = logits - np.nanmax(logits)
    exp = np.exp(logits)
    denom = exp.sum()
    probs = exp / denom if (np.isfinite(denom) and denom > 0) else np.array([1/3, 1/3, 1/3])
    bull, chop, bear = map(float, probs)
    ordered = sorted([bull, chop, bear], reverse=True)
    confidence = float(np.clip(ordered[0] - ordered[1] + 0.35 * ordered[0], 0.0, 1.0))
    instability = float(np.clip(1.0 - abs(bull - bear), 0.0, 1.0))
    if previous is None:
        transition_risk = float(np.clip(0.35 * instability + 0.35 * features.suppression_score + 0.20 * (1.0 - confidence), 0.0, 1.0))
    else:
        prior = np.array([previous.bull, previous.chop, previous.bear])
        current = np.array([bull, chop, bear])
        shift = float(np.abs(current - prior).sum() / 2.0)
        transition_risk = float(np.clip(0.45 * shift + 0.20 * instability + 0.20 * features.suppression_score + 0.15 * (1.0 - confidence), 0.0, 1.0))
    selected_regime = ["BULL", "CHOP", "BEAR"][int(np.argmax(probs))]
    return RegimeProbabilities(
        bull=bull, chop=chop, bear=bear, confidence=confidence,
        transition_risk=transition_risk, instability=instability,
        regime_score=float(bull - bear), selected_regime=selected_regime,
    )
