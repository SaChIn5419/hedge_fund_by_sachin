from __future__ import annotations

import argparse

import pandas as pd

from config.paths import FINBERT_SCORES_PATH, NEWSAPI_ARTICLES_PATH
from data.news.finbert import FinBERTConfig, score_articles_with_finbert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cached relevant news articles with FinBERT.")
    parser.add_argument("--batch-size", type=int, default=16, help="FinBERT batch size")
    parser.add_argument("--max-chars", type=int, default=512, help="Max chars per article text")
    parser.add_argument("--device", type=int, default=-1, help="Transformers device id; -1 for CPU")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on newest articles to score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not NEWSAPI_ARTICLES_PATH.exists():
        raise SystemExit(f"No article cache found at {NEWSAPI_ARTICLES_PATH}")
    articles = pd.read_parquet(NEWSAPI_ARTICLES_PATH).sort_values("published_at")
    if args.limit > 0:
        articles = articles.tail(args.limit).reset_index(drop=True)
    scores = score_articles_with_finbert(
        articles,
        existing_scores_path=str(FINBERT_SCORES_PATH),
        config=FinBERTConfig(batch_size=args.batch_size, max_chars=args.max_chars, device=args.device),
    )
    FINBERT_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(FINBERT_SCORES_PATH, index=False)
    print(f"Wrote FinBERT scores: {FINBERT_SCORES_PATH} ({len(scores)} rows)")


if __name__ == "__main__":
    main()
