from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedundancyGate:
    max_abs_spearman: float = 0.75
    max_rolling_median_abs_spearman: float = 0.70
    max_rolling_p75_abs_spearman: float = 0.80
    max_vif: float = 5.0
    hard_max_vif: float = 10.0
    stationarized_only: bool = True


@dataclass(frozen=True)
class IncrementalValueGate:
    min_weekly_return_oos_r2_gain: float = 0.0015
    min_classification_auc_gain: float = 0.01
    min_balanced_accuracy_gain: float = 0.02
    min_positive_rolling_ic_share: float = 0.55


@dataclass(frozen=True)
class SpilloverGate:
    min_dynamic_corr_beta: float = 0.0
    min_bucket_corr_lift: float = 0.10
    min_weekly_high_spillover_observations: int = 30
    bootstrap_p_value_warn: float = 0.10
    bootstrap_p_value_promote: float = 0.05
    prefer_dcc_garch: bool = True


@dataclass(frozen=True)
class RegimeUtilityGate:
    max_false_defensive_rate_increase: float = 0.10
    min_drawdown_warning_recall_gain: float = 0.05
    min_average_lead_rebalances: float = 1.0
    min_saved_capital_to_friction_multiple: float = 2.0


@dataclass(frozen=True)
class StabilityGate:
    min_same_sign_rolling_window_share: float = 0.60
    max_single_period_edge_contribution: float = 0.35


@dataclass(frozen=True)
class ProductionWeightGate:
    telemetry_only_weight: float = 0.0
    weak_pass_max_news_layer_weight: float = 0.15
    forward_validated_max_news_layer_weight: float = 0.30


@dataclass(frozen=True)
class DexterValidationContract:
    redundancy: RedundancyGate = RedundancyGate()
    incremental_value: IncrementalValueGate = IncrementalValueGate()
    spillover: SpilloverGate = SpilloverGate()
    regime_utility: RegimeUtilityGate = RegimeUtilityGate()
    stability: StabilityGate = StabilityGate()
    production_weight: ProductionWeightGate = ProductionWeightGate()


VALIDATION_CONTRACT = DexterValidationContract()

