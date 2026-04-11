from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.news import FINBERT_MODEL_NAME


@dataclass
class FinBERTConfig:
    model_name: str = FINBERT_MODEL_NAME
    batch_size: int = 16
    max_chars: int = 512
    device: int = -1


def load_finbert_pipeline(config: FinBERTConfig):
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers is required for FinBERT scoring. Install transformers and torch.") from exc
    return pipeline(
        "text-classification",
        model=config.model_name,
        tokenizer=config.model_name,
        device=config.device,
        truncation=True,
    )


def score_articles_with_finbert(
    articles: pd.DataFrame,
    existing_scores_path: str | None = None,
    config: FinBERTConfig | None = None,
) -> pd.DataFrame:
    config = config or FinBERTConfig()
    if articles is None or articles.empty:
        return pd.DataFrame(columns=_score_columns())

    existing = pd.read_parquet(existing_scores_path) if existing_scores_path and pd.io.common.file_exists(existing_scores_path) else pd.DataFrame(columns=_score_columns())
    unseen = articles[~articles["article_id"].isin(existing.get("article_id", pd.Series(dtype=str)))].copy()
    if unseen.empty:
        return existing.sort_values("scored_at").reset_index(drop=True)

    clf = load_finbert_pipeline(config)
    texts = unseen.apply(lambda row: _build_text(row, max_chars=config.max_chars), axis=1).tolist()
    results = clf(texts, batch_size=config.batch_size)

    scored = unseen[["article_id"]].copy()
    scored["finbert_label"] = [str(r.get("label", "")).lower() for r in results]
    scored["finbert_score_raw"] = [float(r.get("score", 0.0)) for r in results]
    scored["finbert_pos"] = scored.apply(lambda row: row["finbert_score_raw"] if row["finbert_label"] == "positive" else 0.0, axis=1)
    scored["finbert_neg"] = scored.apply(lambda row: row["finbert_score_raw"] if row["finbert_label"] == "negative" else 0.0, axis=1)
    scored["finbert_neu"] = scored.apply(lambda row: row["finbert_score_raw"] if row["finbert_label"] == "neutral" else 0.0, axis=1)
    scored["sentiment_score"] = scored["finbert_pos"] - scored["finbert_neg"]
    scored["scored_at"] = pd.Timestamp.utcnow().tz_localize(None)
    scored["model_version"] = config.model_name

    merged = pd.concat([existing, scored], ignore_index=True).drop_duplicates(subset=["article_id"], keep="last")
    return merged.sort_values("scored_at").reset_index(drop=True)


def _build_text(row: pd.Series, max_chars: int) -> str:
    text = f"{row.get('title', '')}. {row.get('description', '')}".strip()
    return text[:max_chars]


def _score_columns() -> list[str]:
    return [
        "article_id",
        "finbert_label",
        "finbert_score_raw",
        "finbert_pos",
        "finbert_neu",
        "finbert_neg",
        "sentiment_score",
        "scored_at",
        "model_version",
    ]
