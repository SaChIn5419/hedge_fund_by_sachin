from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config.paths import FINBERT_SCORES_PATH, NEWS_DAILY_FEATURES_PATH, NEWSAPI_ARTICLES_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily news features from cached articles and FinBERT scores.")
    return parser.parse_args()


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

    daily["attention_shock"] = _rolling_z(np.log1p(daily["article_count"]), 20)
    daily["novelty_score"] = np.log1p(daily["source_dispersion"]) * (1.0 + daily["global_risk_count"] / daily["article_count"].replace(0, np.nan)).fillna(1.0)
    daily["sentiment_3d"] = daily["avg_sentiment"].rolling(3, min_periods=1).mean()
    daily["sentiment_5d"] = daily["avg_sentiment"].rolling(5, min_periods=1).mean()
    daily["negative_share_3d"] = daily["negative_share"].rolling(3, min_periods=1).mean()
    daily["macro_stress_flag"] = (
        (daily["negative_share_3d"] >= 0.45)
        | ((daily["global_risk_count"] > 0) & (daily["attention_shock"] > 1.0))
        | ((daily["policy_count"] > 0) & (daily["avg_sentiment"] < -0.15))
    ).astype(int)
    numeric_cols = [c for c in daily.columns if c != "date"]
    daily[numeric_cols] = daily[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    NEWS_DAILY_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(NEWS_DAILY_FEATURES_PATH, index=False)
    print(f"Wrote daily news features: {NEWS_DAILY_FEATURES_PATH} ({len(daily)} rows)")


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    std = series.rolling(window, min_periods=max(3, window // 2)).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


if __name__ == "__main__":
    main()
