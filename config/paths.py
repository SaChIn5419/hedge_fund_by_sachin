from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('CHIMERA_DATA_DIR', REPO_ROOT / 'chimera_data'))

OUTPUT_DATA_DIR = REPO_ROOT / 'data'
MARKET_DATA_DIR = REPO_ROOT / 'data' / 'market'
NEWS_DATA_DIR = REPO_ROOT / 'data' / 'news'
FEATURES_DATA_DIR = REPO_ROOT / 'data' / 'features'
NEWS_RAW_DIR = NEWS_DATA_DIR / 'raw'
NEWS_PROCESSED_DIR = NEWS_DATA_DIR / 'processed'
NEWS_SCORED_DIR = NEWS_DATA_DIR / 'scored'

NEWSAPI_ARTICLES_PATH = NEWS_PROCESSED_DIR / 'newsapi_articles.parquet'
FINBERT_SCORES_PATH = NEWS_SCORED_DIR / 'finbert_scores.parquet'
NEWS_DAILY_FEATURES_PATH = FEATURES_DATA_DIR / 'news_daily_features.parquet'
DEXTER_RESEARCH_FEATURES_PATH = FEATURES_DATA_DIR / 'dexter_research_features.parquet'
DEXTER_AUDIT_DIR = NEWS_DATA_DIR / 'dexter'

PRIMARY_TRADELOG = OUTPUT_DATA_DIR / 'tradelog_chimera_fip.csv'
REGIME_TRACE_PATH = OUTPUT_DATA_DIR / 'regime_trace_chimera_fip.csv'
WEEKLY_TRACE_PATH = OUTPUT_DATA_DIR / 'weekly_returns_chimera_fip.csv'

