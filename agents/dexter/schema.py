from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from agents.dexter.registry import ALL_SPECS, DEXTER_VERSION, FEATURE_SPECS, REGISTRY_VERSION


class DexterValidationError(ValueError):
    """Raised when a Dexter row is not safe to persist."""


@dataclass(frozen=True)
class SourceEvidence:
    title: str
    url: str
    domain: str
    published_at: pd.Timestamp
    geography: str


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception as exc:
        raise DexterValidationError(f"{field} is not a valid timestamp: {value!r}") from exc
    if pd.isna(ts):
        raise DexterValidationError(f"{field} is missing")
    return ts


def _date(value: Any, field: str) -> pd.Timestamp:
    ts = _timestamp(value, field)
    return ts.normalize()


def _bounded_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise DexterValidationError(f"{field} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise DexterValidationError(f"{field} must be finite, got {value!r}")
    spec = ALL_SPECS[field]
    if out < spec.lower or out > spec.upper:
        raise DexterValidationError(f"{field}={out} outside [{spec.lower}, {spec.upper}]")
    return out


def _field(payload: Mapping[str, Any], flat: str, path: Iterable[str]) -> Any:
    if flat in payload:
        return payload[flat]
    cur: Any = payload
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise DexterValidationError(f"missing required field {flat}")
        cur = cur[key]
    return cur


NESTED_FIELD_PATHS = {
    "india_research_bias": ("india", "research_bias"),
    "india_macro_conviction": ("india", "macro_conviction"),
    "india_liquidity_stress": ("india", "liquidity_stress"),
    "india_policy_stress": ("india", "policy_stress"),
    "india_banking_stress": ("india", "banking_stress"),
    "fii_dii_flow_stress": ("india", "fii_dii_flow_stress"),
    "global_spillover_risk": ("global", "global_spillover_risk"),
    "usd_rate_pressure": ("global", "usd_rate_pressure"),
    "commodity_import_stress": ("global", "commodity_import_stress"),
    "geopolitical_stress": ("global", "geopolitical_stress"),
    "domestic_vulnerability": ("interaction", "domestic_vulnerability"),
    "spillover_amplifier": ("interaction", "spillover_amplifier"),
    "combined_risk_alert": ("interaction", "combined_risk_alert"),
}


def validate_llm_payload(
    payload: Mapping[str, Any],
    *,
    as_of_date: str,
    model_id: str,
    prompt_text: str,
    max_source_age_days: int,
) -> Dict[str, Any]:
    """Validate a raw LLM-style Dexter payload and return a flat canonical row.

    The row is only accepted if every numeric feature is bounded and every
    non-neutral opinion is backed by fresh source evidence.
    """
    row: Dict[str, Any] = {}
    row["date"] = _date(as_of_date, "as_of_date").date().isoformat()
    row["as_of_date"] = row["date"]
    row["generated_at"] = datetime.now(timezone.utc).isoformat()
    row["dexter_version"] = str(payload.get("dexter_version", DEXTER_VERSION))
    row["registry_version"] = REGISTRY_VERSION
    row["model_id"] = model_id
    row["prompt_hash"] = prompt_hash(prompt_text)

    if row["dexter_version"] != DEXTER_VERSION:
        raise DexterValidationError(f"unsupported dexter_version {row['dexter_version']}")

    for field, path in NESTED_FIELD_PATHS.items():
        row[field] = _bounded_float(_field(payload, field, path), field)

    confidence = payload.get("confidence", payload.get("overall_confidence", 0.0))
    row["overall_confidence"] = _bounded_float(confidence, "overall_confidence")

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise DexterValidationError("reason is required")
    row["reason"] = reason[:500]

    evidence = payload.get("source_evidence", [])
    sources = validate_source_evidence(evidence, as_of_date=row["as_of_date"], max_source_age_days=max_source_age_days)
    source_health = score_source_health(sources, as_of_date=row["as_of_date"], max_source_age_days=max_source_age_days)
    row.update(source_health)
    row["source_hash"] = stable_hash([s.__dict__ for s in sources])
    row["source_window_start"] = min(s.published_at for s in sources).date().isoformat()
    row["source_window_end"] = max(s.published_at for s in sources).date().isoformat()
    row["freshest_source_at"] = max(s.published_at for s in sources).isoformat()
    row["dexter_status"] = source_health["dexter_status"]

    row["evidence_confidence"] = float(
        np.clip(row["source_freshness_score"] * row["source_coverage_score"], 0.0, 1.0)
    )
    row["hallucination_risk_score"] = float(
        np.clip(1.0 - row["evidence_confidence"], 0.0, 1.0)
    )
    row["overall_confidence"] = float(
        np.clip(row["overall_confidence"] * (1.0 - row["hallucination_risk_score"]), 0.0, 1.0)
    )
    if row["hallucination_risk_score"] >= 0.65:
        for name, spec in FEATURE_SPECS.items():
            row[name] = spec.neutral
        row["reason"] = f"Neutralized due to source risk: {row['reason']}"

    return row


def validate_source_evidence(
    raw_sources: Any,
    *,
    as_of_date: str,
    max_source_age_days: int,
) -> List[SourceEvidence]:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DexterValidationError("source_evidence must be a non-empty list")

    as_of = _date(as_of_date, "as_of_date")
    out: List[SourceEvidence] = []
    for idx, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise DexterValidationError(f"source_evidence[{idx}] must be an object")
        published_at = _timestamp(raw.get("published_at"), f"source_evidence[{idx}].published_at")
        if published_at.normalize() > as_of:
            raise DexterValidationError(f"source_evidence[{idx}] is after as_of_date")
        age_days = int((as_of - published_at.normalize()).days)
        if age_days > max_source_age_days:
            raise DexterValidationError(f"source_evidence[{idx}] is stale by {age_days} days")
        geography = str(raw.get("geography", "")).lower()
        if geography not in {"india", "global"}:
            raise DexterValidationError(f"source_evidence[{idx}].geography must be india or global")
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("url", "")).strip()
        domain = str(raw.get("domain", "")).strip()
        if not title or not url or not domain:
            raise DexterValidationError(f"source_evidence[{idx}] requires title, url, and domain")
        out.append(SourceEvidence(title=title, url=url, domain=domain, published_at=published_at, geography=geography))
    return out


def score_source_health(sources: List[SourceEvidence], *, as_of_date: str, max_source_age_days: int) -> Dict[str, Any]:
    as_of = _date(as_of_date, "as_of_date")
    freshest = max(s.published_at for s in sources)
    freshest_age_days = int(max(0, (as_of - freshest.normalize()).days))
    unique_domains = len({s.domain for s in sources})
    india_sources = sum(1 for s in sources if s.geography == "india")
    global_sources = sum(1 for s in sources if s.geography == "global")

    freshness_score = float(np.exp(-freshest_age_days / max(max_source_age_days, 1)))
    source_score = min(1.0, len(sources) / 8.0)
    domain_score = min(1.0, unique_domains / 4.0)
    india_score = min(1.0, india_sources / 4.0)
    global_score = min(1.0, global_sources / 2.0)
    coverage_score = float(np.clip(0.45 * source_score + 0.25 * domain_score + 0.20 * india_score + 0.10 * global_score, 0.0, 1.0))

    if india_sources < 2 or len(sources) < 3:
        status = "insufficient_fresh_sources"
    elif freshness_score <= 0.05:
        status = "stale_sources"
    else:
        status = "ok"

    return {
        "source_freshness_score": freshness_score,
        "source_coverage_score": coverage_score,
        "freshest_source_age_days": freshest_age_days,
        "source_count": len(sources),
        "india_source_count": india_sources,
        "global_source_count": global_sources,
        "unique_domain_count": unique_domains,
        "dexter_status": status,
    }
