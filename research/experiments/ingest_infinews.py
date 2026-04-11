"""
Ingest the IN-FINews dataset (Indian Financial News from MoneyControl)
into the Chimera article cache format so it can be scored by FinBERT
and used by the regime detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from config.news import (
    BLOCKLIST_KEYWORDS,
    INDIA_KEYWORDS,
    NEWS_REQUIRED_COLUMNS,
    RELEVANCE_WEIGHTS,
    TOPIC_KEYWORDS,
    TRUSTED_DOMAINS,
)
from config.paths import NEWSAPI_ARTICLES_PATH
from data.news.newsapi_ingest import merge_article_cache


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "IN-FINews  Dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest IN-FINews dataset into article cache.")
    parser.add_argument("--path", type=str, default=str(DATASET_PATH), help="Path to IN-FINews JSON")
    parser.add_argument("--min-relevance", type=float, default=1.5, help="Minimum relevance score to keep")
    return parser.parse_args()


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def _keyword_hit(text_series: pd.Series, keywords: set[str]) -> pd.Series:
    pattern = "|".join(sorted(re.escape(keyword) for keyword in keywords))
    return text_series.str.contains(pattern, regex=True, na=False)


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} articles from IN-FINews dataset")

    df = pd.DataFrame(raw)

    # Normalize to our schema
    out = pd.DataFrame()
    out["published_at"] = pd.to_datetime(df["Date"], errors="coerce")
    out["source"] = "MoneyControl"
    out["url"] = df["URL"].fillna("").astype(str)
    out["title"] = df["Title"].fillna("").astype(str)
    out["description"] = df["Description"].fillna("").astype(str)
    out["language"] = "en"

    # Extract domain
    from urllib.parse import urlparse
    out["domain"] = out["url"].map(lambda u: urlparse(u).netloc.lower() if u else "")
    out["url_hash"] = out["url"].map(_hash_text)
    out["article_id"] = out["url_hash"]

    # Topic tagging (on title + description + keywords if available)
    keywords_col = df["Keywords"].fillna("").astype(str)
    text = (out["title"].fillna("") + " " + out["description"].fillna("") + " " + keywords_col).str.lower()

    out["is_india"] = _keyword_hit(text, INDIA_KEYWORDS).astype(int)
    # MoneyControl is an Indian financial source — boost India flag
    out["is_india"] = out["is_india"].clip(lower=1)

    for topic, keywords in TOPIC_KEYWORDS.items():
        out[f"topic_{topic}"] = _keyword_hit(text, keywords).astype(int)

    # Relevance scoring
    out["relevance_score"] = (
        RELEVANCE_WEIGHTS["india"] * out["is_india"]
        + RELEVANCE_WEIGHTS["policy"] * out["topic_policy"]
        + RELEVANCE_WEIGHTS["inflation"] * out["topic_inflation"]
        + RELEVANCE_WEIGHTS["growth"] * out["topic_growth"]
        + RELEVANCE_WEIGHTS["banking"] * out["topic_banking"]
        + RELEVANCE_WEIGHTS["oil"] * out["topic_oil"]
        + RELEVANCE_WEIGHTS["global_risk"] * out["topic_global_risk"]
    )

    # Filter
    out = out.dropna(subset=["published_at"]).drop_duplicates(subset=["url_hash"], keep="last")

    # Apply relevance filter
    relevant = out[out["relevance_score"] >= args.min_relevance].copy()

    # Ensure all required columns
    for col in NEWS_REQUIRED_COLUMNS:
        if col not in relevant.columns:
            relevant[col] = 0 if col.startswith(("is_", "topic_", "relevance")) else ""

    relevant = relevant[NEWS_REQUIRED_COLUMNS].sort_values("published_at").reset_index(drop=True)

    print(f"Relevant articles: {len(relevant)} / {len(out)}")
    print(f"Date range: {relevant['published_at'].min()} -> {relevant['published_at'].max()}")
    print(f"Topic breakdown:")
    for topic in TOPIC_KEYWORDS:
        count = relevant[f"topic_{topic}"].sum()
        if count > 0:
            print(f"  {topic}: {count}")

    # Merge into existing article cache
    merged = merge_article_cache(NEWSAPI_ARTICLES_PATH, relevant)
    print(f"Merged article cache: {NEWSAPI_ARTICLES_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
