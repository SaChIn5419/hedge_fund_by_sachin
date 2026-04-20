from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


DEXTER_VERSION = "1.0.0"
REGISTRY_VERSION = "2026-04-20"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    lower: float
    upper: float
    neutral: float
    half_life_days: float
    z_window: int = 60
    description: str = ""


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "india_research_bias": FeatureSpec(
        "india_research_bias", "india", -1.0, 1.0, 0.0, 3.0,
        description="India-first directional research bias.",
    ),
    "india_macro_conviction": FeatureSpec(
        "india_macro_conviction", "india", 0.0, 1.0, 0.0, 7.0,
        description="Strength of domestic macro evidence.",
    ),
    "india_liquidity_stress": FeatureSpec(
        "india_liquidity_stress", "india", 0.0, 1.0, 0.0, 2.0,
        description="Short-lived domestic liquidity and funding stress.",
    ),
    "india_policy_stress": FeatureSpec(
        "india_policy_stress", "india", 0.0, 1.0, 0.0, 10.0,
        description="RBI, fiscal, tax, regulatory, election, or policy stress.",
    ),
    "india_banking_stress": FeatureSpec(
        "india_banking_stress", "india", 0.0, 1.0, 0.0, 4.0,
        description="Domestic banking/NBFC/credit stress.",
    ),
    "fii_dii_flow_stress": FeatureSpec(
        "fii_dii_flow_stress", "india", 0.0, 1.0, 0.0, 2.0,
        description="Foreign/domestic institutional flow stress inferred from sourced news.",
    ),
    "global_spillover_risk": FeatureSpec(
        "global_spillover_risk", "global", 0.0, 1.0, 0.0, 2.0,
        description="Global risk that can spill into Indian equities.",
    ),
    "usd_rate_pressure": FeatureSpec(
        "usd_rate_pressure", "global", 0.0, 1.0, 0.0, 3.0,
        description="Fed, dollar, US yield, and rupee pressure.",
    ),
    "commodity_import_stress": FeatureSpec(
        "commodity_import_stress", "global", 0.0, 1.0, 0.0, 5.0,
        description="Crude, energy, gold, and other import-cost stress.",
    ),
    "geopolitical_stress": FeatureSpec(
        "geopolitical_stress", "global", 0.0, 1.0, 0.0, 4.0,
        description="War, sanctions, conflict, and supply-chain stress.",
    ),
    "domestic_vulnerability": FeatureSpec(
        "domestic_vulnerability", "interaction", 0.0, 1.0, 0.0, 4.0,
        description="Domestic fragility that amplifies external shocks.",
    ),
    "spillover_amplifier": FeatureSpec(
        "spillover_amplifier", "interaction", 0.0, 1.0, 0.0, 3.0,
        description="Conditional India/global stress interaction.",
    ),
    "combined_risk_alert": FeatureSpec(
        "combined_risk_alert", "interaction", 0.0, 1.0, 0.0, 2.0,
        description="Final guardrail risk score from domestic and global evidence.",
    ),
}

QUALITY_SPECS: Dict[str, FeatureSpec] = {
    "source_freshness_score": FeatureSpec(
        "source_freshness_score", "quality", 0.0, 1.0, 0.0, 1.0,
        description="Freshness of cited input evidence.",
    ),
    "source_coverage_score": FeatureSpec(
        "source_coverage_score", "quality", 0.0, 1.0, 0.0, 1.0,
        description="Diversity and count of fresh India/global sources.",
    ),
    "evidence_confidence": FeatureSpec(
        "evidence_confidence", "quality", 0.0, 1.0, 0.0, 1.0,
        description="Confidence after source coverage and freshness checks.",
    ),
    "hallucination_risk_score": FeatureSpec(
        "hallucination_risk_score", "quality", 0.0, 1.0, 1.0, 1.0,
        description="Risk that the row is unsupported, stale, or under-sourced.",
    ),
    "overall_confidence": FeatureSpec(
        "overall_confidence", "quality", 0.0, 1.0, 0.0, 1.0,
        description="Usable confidence after hallucination penalties.",
    ),
}

ALL_SPECS: Dict[str, FeatureSpec] = {**FEATURE_SPECS, **QUALITY_SPECS}

STATUS_OK = "ok"
STATUS_INSUFFICIENT_SOURCES = "insufficient_fresh_sources"
STATUS_STALE = "stale_sources"
STATUS_FAILED_VALIDATION = "failed_validation"

REQUIRED_METADATA = [
    "date",
    "as_of_date",
    "generated_at",
    "dexter_version",
    "registry_version",
    "model_id",
    "prompt_hash",
    "source_hash",
    "source_window_start",
    "source_window_end",
    "freshest_source_at",
    "freshest_source_age_days",
    "source_count",
    "india_source_count",
    "global_source_count",
    "unique_domain_count",
    "dexter_status",
    "reason",
]


def feature_names(include_quality: bool = False) -> List[str]:
    specs = ALL_SPECS if include_quality else FEATURE_SPECS
    return list(specs.keys())


def effective_feature_names() -> List[str]:
    return [f"{name}_effective" for name in FEATURE_SPECS]


def z_feature_names() -> List[str]:
    return [f"{name}_z60" for name in FEATURE_SPECS]


def canonical_columns() -> List[str]:
    return REQUIRED_METADATA + feature_names(include_quality=True) + effective_feature_names() + z_feature_names()


def specs_for(names: Iterable[str]) -> List[FeatureSpec]:
    return [ALL_SPECS[name] for name in names]
