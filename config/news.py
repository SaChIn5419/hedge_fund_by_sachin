from __future__ import annotations

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

NEWS_REQUIRED_COLUMNS = [
    "article_id",
    "published_at",
    "fetched_at_utc",
    "ingestion_as_of_ist",
    "publication_cutoff_utc",
    "source",
    "domain",
    "url",
    "url_hash",
    "title",
    "description",
    "language",
    "relevance_score",
    "is_india",
    "topic_policy",
    "topic_inflation",
    "topic_growth",
    "topic_banking",
    "topic_oil",
    "topic_global_risk",
]

TOPIC_KEYWORDS = {
    "policy": {"rbi", "reserve bank", "policy", "repo", "regulation", "budget", "tax"},
    "inflation": {"inflation", "cpi", "wpi", "food prices", "core inflation"},
    "growth": {"growth", "gdp", "pmi", "iip", "industrial output", "capex", "demand"},
    "banking": {"bank", "nbfc", "credit", "default", "npa", "loan", "deposit"},
    "oil": {"oil", "crude", "brent", "opec", "energy"},
    "global_risk": {"war", "sanctions", "fed", "fomc", "yield spike", "recession", "risk-off", "geopolitical"},
}

INDIA_KEYWORDS = {"india", "indian", "rbi", "nifty", "sensex", "rupee", "mumbai", "new delhi"}

DEFAULT_NEWS_QUERY = "India OR Indian economy OR RBI OR Nifty OR Sensex OR rupee"

BLOCKLIST_KEYWORDS = {
    "flight",
    "airline",
    "hotel",
    "vacation",
    "travel",
    "tourism",
    "beach",
    "trip",
    "cruise",
    "restaurant",
    "movie",
    "celebrity",
    "fashion",
    "sports",
    "football",
}

TRUSTED_DOMAINS = {
    "economictimes.indiatimes.com",
    "www.economictimes.indiatimes.com",
    "business-standard.com",
    "www.business-standard.com",
    "livemint.com",
    "www.livemint.com",
    "moneycontrol.com",
    "www.moneycontrol.com",
    "reuters.com",
    "www.reuters.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "ft.com",
    "www.ft.com",
    "wsj.com",
    "www.wsj.com",
}

RELEVANCE_WEIGHTS = {
    "india": 3.0,
    "policy": 2.0,
    "inflation": 1.5,
    "growth": 1.5,
    "banking": 1.5,
    "oil": 1.0,
    "global_risk": 1.5,
}

FINBERT_MODEL_NAME = "ProsusAI/finbert"

# ── Regime context weights ─────────────────────────────────────────────
# These weights are intentionally conservative. The news layer should first
# help with regime confidence and trade suppression, not dominate the state call.
REGIME_CONTEXT_WEIGHTS = {
    "attention_shock_z": 0.20,
    "tone_z": 0.20,
    "novelty_z": 0.15,
    "policy_share_z": 0.15,
    "conflict_share_z": -0.25,
    "macro_share_z": 0.10,
    "liquidity_share_z": 0.12,
    "growth_share_z": 0.10,
    "inflation_share_z": -0.10,
    "banking_share_z": -0.08,
    "oil_share_z": -0.10,
    "global_risk_share_z": -0.15,
}

SUPPRESSION_WEIGHTS = {
    "conflict_share_z": 0.40,
    "negative_tone_z": 0.30,
    "attention_shock_z": 0.20,
    "novelty_z": 0.10,
    "global_risk_share_z": 0.20,
    "banking_share_z": 0.15,
    "oil_share_z": 0.15,
}
