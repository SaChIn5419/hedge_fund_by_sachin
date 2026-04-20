# News Layer

NewsAPI + FinBERT pipeline for macro sentiment scoring.

## Pipeline

```bash
export NEWSAPI_KEY="your_api_key"
python -m research.experiments.build_newsapi_articles --from-date 2024-10-01 --to-date 2024-10-31
python -m research.experiments.score_finbert_news --batch-size 16 --device -1
python -m research.experiments.build_news_features
python -m research.experiments.build_dexter_features
```

For pre-open India signals, enforce a publication cutoff:

```bash
python -m research.experiments.build_newsapi_articles --from-date 2026-04-19 --to-date 2026-04-20 --cutoff-time-ist 09:00
```

## Artifacts

- `data/news/raw/newsapi_*.parquet`: fetched article snapshots
- `data/news/processed/newsapi_articles.parquet`: merged relevant article cache
- `data/news/scored/finbert_scores.parquet`: one-time article sentiment cache
- `data/features/news_daily_features.parquet`: daily aggregated news features
- `data/features/dexter_research_features.parquet`: India-first guarded research features with source freshness and hallucination-risk telemetry

Article rows include `fetched_at_utc`, `ingestion_as_of_ist`, and `publication_cutoff_utc` so backtests can enforce strict as-of timing.
