from __future__ import annotations

import argparse
import os

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise SystemExit("Set NEWSAPI_KEY in the environment before running this script.")
    articles = fetch_newsapi_articles(
        api_key=api_key,
        query=args.query,
        from_date=args.from_date,
        to_date=args.to_date,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    raw_path = NEWS_RAW_DIR / f"newsapi_{args.from_date}_{args.to_date}.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(raw_path, index=False)
    relevant = filter_relevant_articles(articles, min_relevance_score=args.min_relevance)
    merged = merge_article_cache(NEWSAPI_ARTICLES_PATH, relevant)
    print(f"Wrote raw NewsAPI articles: {raw_path}")
    print(f"Relevant articles this run: {len(relevant)}")
    print(f"Merged article cache: {NEWSAPI_ARTICLES_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
