from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from agents.dexter.registry import ALL_SPECS, FEATURE_SPECS, canonical_columns


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns and "as_of_date" in out.columns:
        out["date"] = out["as_of_date"]
    if "date" not in out.columns:
        raise ValueError("Dexter feature frame requires date or as_of_date")

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if "as_of_date" not in out.columns:
        out["as_of_date"] = out["date"]
    out["as_of_date"] = pd.to_datetime(out["as_of_date"]).dt.normalize()

    for name, spec in ALL_SPECS.items():
        if name not in out.columns:
            out[name] = spec.neutral
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(spec.neutral).clip(spec.lower, spec.upper)

    if "freshest_source_age_days" not in out.columns:
        out["freshest_source_age_days"] = 999.0
    out["freshest_source_age_days"] = pd.to_numeric(out["freshest_source_age_days"], errors="coerce").fillna(999.0).clip(lower=0.0)

    defaults = {
        "generated_at": "",
        "dexter_version": "",
        "registry_version": "",
        "model_id": "",
        "prompt_hash": "",
        "source_hash": "",
        "source_window_start": "",
        "source_window_end": "",
        "freshest_source_at": "",
        "source_count": 0,
        "india_source_count": 0,
        "global_source_count": 0,
        "unique_domain_count": 0,
        "dexter_status": "unknown",
        "reason": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    return out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def apply_staleness_decay(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_feature_frame(df)
    age = pd.to_numeric(out["freshest_source_age_days"], errors="coerce").fillna(999.0).clip(lower=0.0)
    quality = pd.to_numeric(out["overall_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    for name, spec in FEATURE_SPECS.items():
        decay = np.exp(-age / max(spec.half_life_days, 1e-9))
        out[f"{name}_effective"] = spec.neutral + (out[name] - spec.neutral) * decay * quality
    return out


def add_rolling_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = apply_staleness_decay(df)
    for name, spec in FEATURE_SPECS.items():
        out[f"{name}_z60"] = rolling_z(out[f"{name}_effective"], spec.z_window).clip(-4.0, 4.0)
    return out


def canonicalize(df: pd.DataFrame, *, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    out = add_rolling_zscores(df)
    wanted = list(columns) if columns is not None else canonical_columns()
    for col in wanted:
        if col not in out.columns:
            out[col] = np.nan
    return out[wanted]


def prepare_for_calendar(df: pd.DataFrame, calendar_index: pd.DatetimeIndex, *, max_fresh_age_days: int = 3) -> pd.DataFrame:
    """Forward-fill Dexter rows onto a market calendar with fresh decay.

    The source row is carried forward for observability, but effective feature
    values are recomputed from the carried row's freshest source timestamp for
    each market date. This prevents stale research from retaining day-one weight.
    """
    out = coerce_feature_frame(df)
    calendar = pd.to_datetime(calendar_index).normalize()
    out = out.set_index("date").sort_index().reindex(calendar).ffill()
    out.index.name = "date"
    out["date"] = out.index
    out["as_of_date"] = out.index

    source_ts = pd.to_datetime(out["freshest_source_at"], errors="coerce", utc=True).dt.tz_convert(None)
    source_dates = source_ts.dt.normalize()
    source_dates = pd.Series(source_dates.to_numpy(), index=out.index)
    age = (out.index.to_series() - source_dates).dt.days
    out["freshest_source_age_days"] = age.where(age >= 0, 999).fillna(999).astype(float)
    freshness = np.exp(-out["freshest_source_age_days"] / max(max_fresh_age_days, 1))
    coverage = pd.to_numeric(out["source_coverage_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    evidence_confidence = np.clip(freshness * coverage, 0.0, 1.0)
    hallucination_risk = np.clip(1.0 - evidence_confidence, 0.0, 1.0)
    out["source_freshness_score"] = freshness
    out["evidence_confidence"] = evidence_confidence
    out["hallucination_risk_score"] = hallucination_risk
    out["overall_confidence"] = np.clip(evidence_confidence * (1.0 - 0.50 * hallucination_risk), 0.0, 1.0)
    stale_mask = (out["freshest_source_age_days"] > max_fresh_age_days) & (out["dexter_status"] == "ok")
    out.loc[stale_mask, "dexter_status"] = "stale_sources"

    return canonicalize(out.reset_index(drop=True))
