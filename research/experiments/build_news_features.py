from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config.paths import FINBERT_SCORES_PATH, NEWS_DAILY_FEATURES_PATH, NEWSAPI_ARTICLES_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily news features from cached articles and FinBERT scores.")
    return parser.parse_args()


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    std = series.rolling(window, min_periods=max(3, window // 2)).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def main() -> None:
    parse_args()
    if not NEWSAPI_ARTICLES_PATH.exists():
        raise SystemExit(f"No article cache found at {NEWSAPI_ARTICLES_PATH}")
    if not FINBERT_SCORES_PATH.exists():
        raise SystemExit(f"No FinBERT scores found at {FINBERT_SCORES_PATH}")

    articles = pd.read_parquet(NEWSAPI_ARTICLES_PATH)
    scores = pd.read_parquet(FINBERT_SCORES_PATH)
    df = articles.merge(scores, on="article_id", how="inner")
    if df.empty:
        raise SystemExit("No scored article overlap found.")

    df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()

    # ── Stage 1: Daily aggregation ──────────────────────────────────────
    daily = df.groupby("date", as_index=False).agg(
        article_count=("article_id", "count"),
        relevant_article_count=("article_id", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        negative_share=("finbert_label", lambda s: float((s == "negative").mean())),
        positive_share=("finbert_label", lambda s: float((s == "positive").mean())),
        policy_count=("topic_policy", "sum"),
        inflation_count=("topic_inflation", "sum"),
        growth_count=("topic_growth", "sum"),
        banking_count=("topic_banking", "sum"),
        oil_count=("topic_oil", "sum"),
        global_risk_count=("topic_global_risk", "sum"),
        india_count=("is_india", "sum"),
        source_dispersion=("domain", "nunique"),
    ).sort_values("date")

    # ── Stage 2: Derived features ───────────────────────────────────────
    safe_count = daily["article_count"].replace(0, np.nan)

    # Topic shares (fraction of articles mentioning each topic)
    daily["policy_share"] = daily["policy_count"] / safe_count
    daily["inflation_share"] = daily["inflation_count"] / safe_count
    daily["growth_share"] = daily["growth_count"] / safe_count
    daily["banking_share"] = daily["banking_count"] / safe_count
    daily["oil_share"] = daily["oil_count"] / safe_count
    daily["global_risk_share"] = daily["global_risk_count"] / safe_count
    daily["india_share"] = daily["india_count"] / safe_count

    # Composite shares for regime model
    # conflict ≈ global_risk (closest proxy from NewsAPI topic tags)
    daily["conflict_share"] = daily["global_risk_share"]
    # macro ≈ union of policy + inflation + growth
    daily["macro_share"] = (daily["policy_count"] + daily["inflation_count"] + daily["growth_count"]) / safe_count
    # liquidity ≈ banking (closest proxy)
    daily["liquidity_share"] = daily["banking_share"]

    # Negative tone (mirrors the old GDELT concept: net negative sentiment)
    daily["negative_tone"] = daily["negative_share"] - daily["positive_share"]

    # Novelty score
    daily["novelty_score"] = np.log1p(daily["source_dispersion"]) * (
        1.0 + daily["global_risk_count"] / safe_count
    ).fillna(1.0)

    # Smoothed sentiment
    daily["sentiment_3d"] = daily["avg_sentiment"].rolling(3, min_periods=1).mean()
    daily["sentiment_5d"] = daily["avg_sentiment"].rolling(5, min_periods=1).mean()
    daily["negative_share_3d"] = daily["negative_share"].rolling(3, min_periods=1).mean()

    # ── Stage 3: Z-scored features (what the regime model consumes) ─────
    Z_WINDOW = 20

    daily["attention_shock_z"] = _rolling_z(np.log1p(daily["article_count"]), Z_WINDOW)
    daily["tone_z"] = _rolling_z(daily["avg_sentiment"], Z_WINDOW)
    daily["negative_tone_z"] = _rolling_z(daily["negative_tone"], Z_WINDOW)
    daily["novelty_z"] = _rolling_z(daily["novelty_score"], Z_WINDOW)
    daily["policy_share_z"] = _rolling_z(daily["policy_share"].fillna(0.0), Z_WINDOW)
    daily["conflict_share_z"] = _rolling_z(daily["conflict_share"].fillna(0.0), Z_WINDOW)
    daily["macro_share_z"] = _rolling_z(daily["macro_share"].fillna(0.0), Z_WINDOW)
    daily["liquidity_share_z"] = _rolling_z(daily["liquidity_share"].fillna(0.0), Z_WINDOW)
    daily["growth_share_z"] = _rolling_z(daily["growth_share"].fillna(0.0), Z_WINDOW)
    daily["inflation_share_z"] = _rolling_z(daily["inflation_share"].fillna(0.0), Z_WINDOW)
    daily["banking_share_z"] = _rolling_z(daily["banking_share"].fillna(0.0), Z_WINDOW)
    daily["oil_share_z"] = _rolling_z(daily["oil_share"].fillna(0.0), Z_WINDOW)
    daily["global_risk_share_z"] = _rolling_z(daily["global_risk_share"].fillna(0.0), Z_WINDOW)

    # ── Stage 4: Stress flag ────────────────────────────────────────────
    daily["news_stress_flag"] = (
        (daily["negative_share_3d"] >= 0.45)
        | ((daily["global_risk_count"] > 0) & (daily["attention_shock_z"] > 1.0))
        | ((daily["policy_count"] > 0) & (daily["avg_sentiment"] < -0.15))
        | (daily["conflict_share_z"] > 1.5)
        | (daily["negative_tone_z"] > 1.5)
        | (daily["global_risk_share_z"] > 1.5)
        | (daily["banking_share_z"] > 1.5)
        | ((daily["attention_shock_z"] > 1.5) & (daily["novelty_z"] > 1.0))
    ).astype(int)

    # ── Stage 5: Clean up and save ──────────────────────────────────────
    numeric_cols = [c for c in daily.columns if c != "date"]
    daily[numeric_cols] = daily[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    NEWS_DAILY_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(NEWS_DAILY_FEATURES_PATH, index=False)
    print(f"Wrote daily news features: {NEWS_DAILY_FEATURES_PATH} ({len(daily)} rows)")
    print(f"Z-scored columns: {[c for c in daily.columns if c.endswith('_z')]}")
    print(f"Stress flags: {daily['news_stress_flag'].sum()} / {len(daily)} days")


if __name__ == "__main__":
    main()
