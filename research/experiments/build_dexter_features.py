from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from agents.dexter.normalize import canonicalize
from agents.dexter.registry import (
    DEXTER_VERSION,
    FEATURE_SPECS,
    REGISTRY_VERSION,
    STATUS_INSUFFICIENT_SOURCES,
    STATUS_OK,
    STATUS_STALE,
)
from agents.dexter.schema import prompt_hash, stable_hash
from config.paths import DEXTER_RESEARCH_FEATURES_PATH, FINBERT_SCORES_PATH, NEWSAPI_ARTICLES_PATH


PROMPT_TEMPLATE = "dexter-india-first-source-grounded-v1"

FLOW_KEYWORDS = ("fii", "dii", "foreign institutional", "domestic institutional", "outflow", "inflow", "fpi")
USD_KEYWORDS = ("fed", "fomc", "dollar", "us yield", "treasury", "rupee", "usd", "rate hike")
POLICY_KEYWORDS = ("rbi", "repo", "policy", "gst", "budget", "tax", "regulation", "election", "monsoon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build guarded Dexter India-first research features.")
    parser.add_argument("--as-of-date", default="", help="YYYY-MM-DD. Defaults to current Asia/Kolkata date.")
    parser.add_argument("--lookback-days", type=int, default=3, help="Fresh source window.")
    parser.add_argument("--min-total-sources", type=int, default=5)
    parser.add_argument("--min-india-sources", type=int, default=3)
    parser.add_argument("--replace", action="store_true", help="Replace the existing Dexter feature store instead of appending.")
    return parser.parse_args()


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _clip11(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _keyword_share(series: pd.Series, keywords: Tuple[str, ...]) -> float:
    if series.empty:
        return 0.0
    text = series.fillna("").astype(str).str.lower()
    return float(text.apply(lambda x: any(k in x for k in keywords)).mean())


def _load_articles() -> pd.DataFrame:
    if not NEWSAPI_ARTICLES_PATH.exists():
        raise SystemExit(f"No article cache found at {NEWSAPI_ARTICLES_PATH}")
    articles = pd.read_parquet(NEWSAPI_ARTICLES_PATH).copy()
    articles["published_at"] = pd.to_datetime(articles["published_at"], utc=True)

    if FINBERT_SCORES_PATH.exists():
        scores = pd.read_parquet(FINBERT_SCORES_PATH)
        keep = ["article_id", "finbert_label", "sentiment_score"]
        articles = articles.merge(scores[keep], on="article_id", how="left")
    else:
        articles["finbert_label"] = "neutral"
        articles["sentiment_score"] = 0.0

    articles["sentiment_score"] = pd.to_numeric(articles["sentiment_score"], errors="coerce").fillna(0.0)
    for col in [
        "is_india",
        "topic_policy",
        "topic_inflation",
        "topic_growth",
        "topic_banking",
        "topic_oil",
        "topic_global_risk",
    ]:
        if col not in articles.columns:
            articles[col] = 0
        articles[col] = pd.to_numeric(articles[col], errors="coerce").fillna(0).astype(int)
    return articles


def _source_health(window: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int, min_total: int, min_india: int) -> Dict[str, float | int | str]:
    if window.empty:
        return {
            "source_freshness_score": 0.0,
            "source_coverage_score": 0.0,
            "evidence_confidence": 0.0,
            "hallucination_risk_score": 1.0,
            "overall_confidence": 0.0,
            "freshest_source_age_days": 999,
            "source_count": 0,
            "india_source_count": 0,
            "global_source_count": 0,
            "unique_domain_count": 0,
            "dexter_status": STATUS_INSUFFICIENT_SOURCES,
        }

    freshest = window["published_at"].max().normalize()
    freshest_age_days = int(max(0, (as_of - freshest).days))
    source_count = int(len(window))
    india_count = int(window["is_india"].sum())
    global_count = int(source_count - india_count)
    unique_domains = int(window["domain"].nunique())

    freshness_score = _clip01(np.exp(-freshest_age_days / max(lookback_days, 1)))
    total_score = min(1.0, source_count / max(min_total, 1))
    india_score = min(1.0, india_count / max(min_india, 1))
    domain_score = min(1.0, unique_domains / 4.0)
    global_score = min(1.0, global_count / 2.0)
    coverage_score = _clip01(0.40 * total_score + 0.30 * india_score + 0.20 * domain_score + 0.10 * global_score)
    evidence_confidence = _clip01(freshness_score * coverage_score)
    hallucination_risk = _clip01(1.0 - evidence_confidence)
    overall_confidence = _clip01(evidence_confidence * (1.0 - 0.50 * hallucination_risk))

    if freshest_age_days > lookback_days:
        status = STATUS_STALE
    elif source_count < min_total or india_count < min_india:
        status = STATUS_INSUFFICIENT_SOURCES
    else:
        status = STATUS_OK

    return {
        "source_freshness_score": freshness_score,
        "source_coverage_score": coverage_score,
        "evidence_confidence": evidence_confidence,
        "hallucination_risk_score": hallucination_risk,
        "overall_confidence": overall_confidence,
        "freshest_source_age_days": freshest_age_days,
        "source_count": source_count,
        "india_source_count": india_count,
        "global_source_count": global_count,
        "unique_domain_count": unique_domains,
        "dexter_status": status,
    }


def _score_window(window: pd.DataFrame, health: Dict[str, float | int | str]) -> Dict[str, float]:
    if window.empty:
        return {name: spec.neutral for name, spec in FEATURE_SPECS.items()}

    india = window[window["is_india"] == 1]
    global_rows = window[window["is_india"] != 1]
    india_base = india if not india.empty else window
    global_base = global_rows if not global_rows.empty else window

    india_sentiment = float(india_base["sentiment_score"].mean())
    global_negative = float((global_base["sentiment_score"] < -0.10).mean())
    india_negative = float((india_base["sentiment_score"] < -0.10).mean())

    policy_share = float(india_base["topic_policy"].mean())
    inflation_share = float(india_base["topic_inflation"].mean())
    growth_share = float(india_base["topic_growth"].mean())
    banking_share = float(india_base["topic_banking"].mean())
    oil_share = float(window["topic_oil"].mean())
    global_risk_share = float(window["topic_global_risk"].mean())

    title_desc = (window["title"].fillna("") + " " + window["description"].fillna(""))
    india_title_desc = (india_base["title"].fillna("") + " " + india_base["description"].fillna(""))
    flow_share = _keyword_share(india_title_desc, FLOW_KEYWORDS)
    usd_share = _keyword_share(title_desc, USD_KEYWORDS)
    policy_keyword_share = _keyword_share(india_title_desc, POLICY_KEYWORDS)

    india_research_bias = _clip11(0.75 * india_sentiment + 0.25 * growth_share - 0.30 * india_negative - 0.15 * inflation_share)
    india_macro_conviction = _clip01(0.35 * abs(india_sentiment) + 0.25 * growth_share + 0.20 * policy_share + 0.20 * inflation_share)
    india_liquidity_stress = _clip01(0.45 * banking_share + 0.30 * flow_share + 0.25 * india_negative)
    india_policy_stress = _clip01(0.45 * policy_share + 0.30 * policy_keyword_share + 0.25 * india_negative)
    india_banking_stress = _clip01(0.65 * banking_share + 0.35 * india_negative)
    fii_dii_flow_stress = _clip01(0.70 * flow_share + 0.30 * india_negative)

    global_spillover_risk = _clip01(0.55 * global_risk_share + 0.25 * global_negative + 0.20 * usd_share)
    usd_rate_pressure = _clip01(0.70 * usd_share + 0.30 * global_negative)
    commodity_import_stress = _clip01(0.65 * oil_share + 0.20 * global_negative + 0.15 * inflation_share)
    geopolitical_stress = _clip01(0.75 * global_risk_share + 0.25 * global_negative)

    domestic_vulnerability = _clip01(
        0.30 * india_liquidity_stress + 0.25 * india_policy_stress + 0.25 * india_banking_stress + 0.20 * fii_dii_flow_stress
    )
    spillover_amplifier = _clip01(global_spillover_risk * (0.50 + domestic_vulnerability))
    combined_risk_alert = _clip01(
        0.45 * domestic_vulnerability + 0.30 * global_spillover_risk + 0.25 * spillover_amplifier
    )

    scores = {
        "india_research_bias": india_research_bias,
        "india_macro_conviction": india_macro_conviction,
        "india_liquidity_stress": india_liquidity_stress,
        "india_policy_stress": india_policy_stress,
        "india_banking_stress": india_banking_stress,
        "fii_dii_flow_stress": fii_dii_flow_stress,
        "global_spillover_risk": global_spillover_risk,
        "usd_rate_pressure": usd_rate_pressure,
        "commodity_import_stress": commodity_import_stress,
        "geopolitical_stress": geopolitical_stress,
        "domestic_vulnerability": domestic_vulnerability,
        "spillover_amplifier": spillover_amplifier,
        "combined_risk_alert": combined_risk_alert,
    }

    if health["dexter_status"] != STATUS_OK or float(health["hallucination_risk_score"]) >= 0.65:
        return {name: spec.neutral for name, spec in FEATURE_SPECS.items()}
    return scores


def build_row(
    articles: pd.DataFrame,
    *,
    as_of_date: str,
    lookback_days: int,
    min_total_sources: int,
    min_india_sources: int,
) -> Dict[str, object]:
    as_of = pd.to_datetime(as_of_date, utc=True).normalize()
    start = as_of - pd.Timedelta(days=lookback_days)
    mask = (articles["published_at"].dt.normalize() >= start) & (articles["published_at"].dt.normalize() <= as_of)
    window = articles.loc[mask].sort_values("published_at").copy()
    health = _source_health(window, as_of, lookback_days, min_total_sources, min_india_sources)
    scores = _score_window(window, health)

    if window.empty:
        source_hash = stable_hash({"as_of_date": as_of_date, "sources": []})
        source_start = start.date().isoformat()
        source_end = as_of.date().isoformat()
        freshest = ""
    else:
        evidence_cols = ["article_id", "published_at", "domain", "url", "title", "is_india"]
        evidence = window[evidence_cols].tail(25).to_dict(orient="records")
        source_hash = stable_hash(evidence)
        source_start = window["published_at"].min().date().isoformat()
        source_end = window["published_at"].max().date().isoformat()
        freshest = window["published_at"].max().isoformat()

    status = str(health["dexter_status"])
    if status == STATUS_OK:
        reason = "India-first source-grounded research row built from fresh articles."
    elif status == STATUS_STALE:
        reason = "Neutralized because freshest source evidence is stale."
    else:
        reason = "Neutralized because fresh India-centric source coverage is insufficient."

    row: Dict[str, object] = {
        "date": as_of.date().isoformat(),
        "as_of_date": as_of.date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dexter_version": DEXTER_VERSION,
        "registry_version": REGISTRY_VERSION,
        "model_id": "deterministic-news-cache-v1",
        "prompt_hash": prompt_hash(PROMPT_TEMPLATE),
        "source_hash": source_hash,
        "source_window_start": source_start,
        "source_window_end": source_end,
        "freshest_source_at": freshest,
        "reason": reason,
    }
    row.update(health)
    row.update(scores)
    return row


def main() -> None:
    args = parse_args()
    articles = _load_articles()
    if args.as_of_date:
        as_of_date = args.as_of_date
    else:
        as_of_date = pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()

    row = build_row(
        articles,
        as_of_date=as_of_date,
        lookback_days=args.lookback_days,
        min_total_sources=args.min_total_sources,
        min_india_sources=args.min_india_sources,
    )
    new_df = pd.DataFrame([row])

    if DEXTER_RESEARCH_FEATURES_PATH.exists() and not args.replace:
        old = pd.read_parquet(DEXTER_RESEARCH_FEATURES_PATH)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df

    out = canonicalize(out)
    DEXTER_RESEARCH_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DEXTER_RESEARCH_FEATURES_PATH, index=False)
    print(f"Wrote Dexter features: {DEXTER_RESEARCH_FEATURES_PATH} ({len(out)} rows)")
    print(f"Status: {row['dexter_status']} | hallucination_risk={row['hallucination_risk_score']:.3f} | sources={row['source_count']}")


if __name__ == "__main__":
    main()
