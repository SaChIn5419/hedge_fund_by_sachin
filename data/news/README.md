# News Layer

NewsAPI + FinBERT pipeline for macro sentiment scoring.

## Pipeline

```bash
export NEWSAPI_KEY="your_api_key"
python -m research.experiments.build_newsapi_articles --from-date 2024-10-01 --to-date 2024-10-31
python -m research.experiments.score_finbert_news --batch-size 16 --device -1
python -m research.experiments.build_news_features
```

## Artifacts

- `data/news/raw/newsapi_*.parquet`: fetched article snapshots
- `data/news/processed/newsapi_articles.parquet`: merged relevant article cache
- `data/news/scored/finbert_scores.parquet`: one-time article sentiment cache
- `data/features/news_daily_features.parquet`: daily aggregated news features
