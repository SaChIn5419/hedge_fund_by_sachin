from __future__ import annotations

import argparse
import os

import pandas as pd

from config.news import DEFAULT_NEWS_QUERY
from config.paths import NEWSAPI_ARTICLES_PATH, NEWS_RAW_DIR
from data.news.newsapi_ingest import fetch_newsapi_articles, filter_relevant_articles, merge_article_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NewsAPI articles and build the local article cache.")
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--query", default=DEFAULT_NEWS_QUERY, help="NewsAPI query")
    parser.add_argument("--page-size", type=int, default=100, help="NewsAPI page size")
    parser.add_argument("--max-pages", type=int, default=3, help="Max NewsAPI pages to fetch")
    parser.add_argument("--min-relevance", type=float, default=2.0, help="Minimum relevance score to keep")
    parser.add_argument(
        "--cutoff-time-ist",
        default="",
        help="Optional HH:MM publication cutoff on --to-date in Asia/Kolkata, e.g. 09:00 for pre-open signals.",
    )
    return parser.parse_args()


def _cutoff_utc(to_date: str, cutoff_time_ist: str) -> tuple[str, str]:
    if not cutoff_time_ist:
        return to_date, ""
    cutoff_ist = pd.Timestamp(f"{to_date} {cutoff_time_ist}", tz="Asia/Kolkata")
    cutoff_utc = cutoff_ist.tz_convert("UTC")
    return cutoff_utc.isoformat(), cutoff_utc.tz_convert(None).isoformat()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise SystemExit("Set NEWSAPI_KEY in the environment before running this script.")
    newsapi_to, publication_cutoff_utc = _cutoff_utc(args.to_date, args.cutoff_time_ist)
    articles = fetch_newsapi_articles(
        api_key=api_key,
        query=args.query,
        from_date=args.from_date,
        to_date=newsapi_to,
        page_size=args.page_size,
        max_pages=args.max_pages,
        ingestion_as_of_ist=args.to_date,
        publication_cutoff_utc=publication_cutoff_utc,
    )
    if publication_cutoff_utc and not articles.empty:
        cutoff_ts = pd.Timestamp(publication_cutoff_utc)
        articles = articles[pd.to_datetime(articles["published_at"]) <= cutoff_ts].copy()
    raw_path = NEWS_RAW_DIR / f"newsapi_{args.from_date}_{args.to_date}.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(raw_path, index=False)
    relevant = filter_relevant_articles(articles, min_relevance_score=args.min_relevance)
    merged = merge_article_cache(NEWSAPI_ARTICLES_PATH, relevant)
    print(f"Wrote raw NewsAPI articles: {raw_path}")
    if publication_cutoff_utc:
        print(f"Applied publication cutoff UTC: {publication_cutoff_utc}")
    print(f"Relevant articles this run: {len(relevant)}")
    print(f"Merged article cache: {NEWSAPI_ARTICLES_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
