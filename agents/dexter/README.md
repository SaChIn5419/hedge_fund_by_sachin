# Dexter Research Guardrails

Dexter is an India-first financial research sensor for Chimera. It is not allowed to act as an oracle.
Its output is accepted only when backed by fresh, dated source evidence.

## Production Rules

- Default generation uses the current Asia/Kolkata date, not the newest cached article date.
- If the source cache is stale or India coverage is weak, Dexter writes neutral scores with high `hallucination_risk_score`.
- Scores are split into India, global, and interaction features.
- Global stress affects Chimera only through explicit spillover fields.
- Effective scores are recomputed with feature-specific staleness decay before the regime model reads them.
- The feature store is append-first. Use `--replace` only for intentional rebuilds.

## Build

```bash
python -m research.experiments.build_dexter_features
```

For historical diagnostics:

```bash
python -m research.experiments.build_dexter_features --as-of-date 2026-04-12
```

## Key Outputs

- `data/features/dexter_research_features.parquet`
- `dexter_status`
- `hallucination_risk_score`
- `overall_confidence`
- `*_effective`
- `*_z60`

Rows with `dexter_status != "ok"` or `hallucination_risk_score >= 0.65` are ignored by the regime context scorer.

## Validation Contract

The production gates live in `agents/dexter/validation.py`.

- Redundancy checks use only stationarized features such as `*_effective` and `*_z60`, never raw LLM-style scores.
- A feature is redundant if absolute Spearman exceeds `0.75`, or rolling median absolute Spearman exceeds `0.70`, or VIF exceeds `5.0`.
- Weekly return targets require only `+0.15%` absolute OOS R-squared improvement after controls. Text features are too noisy for a higher default threshold.
- Classification targets keep the stricter requirement: `+1%` AUC or `+2%` balanced accuracy.
- `spillover_amplifier` should be validated against time-varying India/global correlation, preferably DCC-GARCH, not only hard percentile buckets.
- Defensive regime shifts must clear transaction costs: preserved capital must exceed `2x` estimated round-trip rebalance friction.
- Dexter remains a brake, not the driver. Even after forward validation, its influence is capped at `25-30%` of the news layer.
