from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from config.news import (
    BLOCKLIST_KEYWORDS,
    DEFAULT_NEWS_QUERY,
    INDIA_KEYWORDS,
    NEWS_REQUIRED_COLUMNS,
    NEWSAPI_BASE_URL,
    RELEVANCE_WEIGHTS,
    TOPIC_KEYWORDS,
    TRUSTED_DOMAINS,
)


def build_newsapi_url(
    api_key: str,
    query: str = DEFAULT_NEWS_QUERY,
    from_date: str | None = None,
    to_date: str | None = None,
    page_size: int = 100,
    page: int = 1,
    language: str = "en",
    sort_by: str = "publishedAt",
) -> str:
    params = {
        "apiKey": api_key,
        "q": query,
        "pageSize": str(page_size),
        "page": str(page),
        "language": language,
        "sortBy": sort_by,
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    return f"{NEWSAPI_BASE_URL}?{urlencode(params)}"


def fetch_newsapi_articles(
    api_key: str,
    query: str = DEFAULT_NEWS_QUERY,
    from_date: str | None = None,
    to_date: str | None = None,
    page_size: int = 100,
    max_pages: int = 3,
) -> pd.DataFrame:
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        url = build_newsapi_url(api_key, query=query, from_date=from_date, to_date=to_date, page_size=page_size, page=page)
        req = Request(url, headers={"User-Agent": "chimera-news-pipeline/1.0"})
        with urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
        articles = payload.get("articles", [])
        if not articles:
            break
        rows.extend(articles)
        if len(articles) < page_size:
            break
    return normalize_newsapi_articles(pd.DataFrame(rows))


def normalize_newsapi_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=NEWS_REQUIRED_COLUMNS)

    out = pd.DataFrame()
    out["published_at"] = pd.to_datetime(df.get("publishedAt"), errors="coerce").dt.tz_localize(None)
    source_col = df.get("source")
    if source_col is not None:
        out["source"] = source_col.map(lambda x: x.get("name", "") if isinstance(x, dict) else "")
    else:
        out["source"] = ""
    out["url"] = df.get("url", "").fillna("").astype(str)
    out["title"] = df.get("title", "").fillna("").astype(str)
    out["description"] = df.get("description", "").fillna("").astype(str)
    out["language"] = df.get("language", "en")
    out["domain"] = out["url"].map(_extract_domain)
    out["url_hash"] = out["url"].map(_hash_text)
    out["article_id"] = out["url_hash"]

    text = (out["title"].fillna("") + " " + out["description"].fillna("")).str.lower()
    out["is_india"] = _keyword_hit(text, INDIA_KEYWORDS).astype(int)
    for topic, keywords in TOPIC_KEYWORDS.items():
        out[f"topic_{topic}"] = _keyword_hit(text, keywords).astype(int)
    out["relevance_score"] = (
        RELEVANCE_WEIGHTS["india"] * out["is_india"]
        + RELEVANCE_WEIGHTS["policy"] * out["topic_policy"]
        + RELEVANCE_WEIGHTS["inflation"] * out["topic_inflation"]
        + RELEVANCE_WEIGHTS["growth"] * out["topic_growth"]
        + RELEVANCE_WEIGHTS["banking"] * out["topic_banking"]
        + RELEVANCE_WEIGHTS["oil"] * out["topic_oil"]
        + RELEVANCE_WEIGHTS["global_risk"] * out["topic_global_risk"]
    )
    out = out.dropna(subset=["published_at"]).drop_duplicates(subset=["url_hash"], keep="last")
    for col in NEWS_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = 0 if col.startswith(("is_", "topic_", "relevance")) else ""
    return out[NEWS_REQUIRED_COLUMNS].sort_values("published_at").reset_index(drop=True)


def merge_article_cache(existing_path: Path | str, new_articles: pd.DataFrame) -> pd.DataFrame:
    existing_path = Path(existing_path)
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        merged = pd.concat([existing, new_articles], ignore_index=True)
        merged = merged.drop_duplicates(subset=["url_hash"], keep="last")
    else:
        merged = new_articles.copy()
    merged = merged.sort_values("published_at").reset_index(drop=True)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(existing_path, index=False)
    return merged


def filter_relevant_articles(df: pd.DataFrame, min_relevance_score: float = 2.0) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=NEWS_REQUIRED_COLUMNS)
    work = df.copy()
    text = (work["title"].fillna("") + " " + work["description"].fillna("")).str.lower()
    topic_sum = (
        work["topic_policy"].fillna(0)
        + work["topic_inflation"].fillna(0)
        + work["topic_growth"].fillna(0)
        + work["topic_banking"].fillna(0)
        + work["topic_oil"].fillna(0)
        + work["topic_global_risk"].fillna(0)
    )
    blocklisted = _keyword_hit(text, BLOCKLIST_KEYWORDS)
    trusted = work["domain"].isin(TRUSTED_DOMAINS)
    market_relevant = (work["is_india"].fillna(0) > 0) | (topic_sum > 0)
    strong_relevant = (work["is_india"].fillna(0) > 0) & (topic_sum > 0)
    keep = (
        (work["relevance_score"] >= min_relevance_score)
        & market_relevant
        & (~blocklisted)
        & (strong_relevant | trusted)
    )
    return work[keep].copy().reset_index(drop=True)


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def _keyword_hit(text_series: pd.Series, keywords: set[str]) -> pd.Series:
    pattern = "|".join(sorted(re.escape(keyword) for keyword in keywords))
    return text_series.str.contains(pattern, regex=True, na=False)
