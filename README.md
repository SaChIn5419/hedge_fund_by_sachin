# Chimera — Quantitative Equity Strategy (NSE India)

A production-grade systematic long/short equity strategy for the NSE universe (~500 stocks). Chimera combines XGBoost-based stock selection, Mean-Variance Optimization, a probabilistic regime classifier, and a sigmoidal gold hedge into a weekly rebalancing pipeline.

---

## Architecture

```
Data (parquet) ──► Feature Engineering ──► XGBoost Alpha Model
                                                    │
                                                    ▼
                  Regime Classifier ──────► MVO Portfolio Optimizer
                  (HMM + probabilistic)       (OAS covariance, sector caps)
                                                    │
                                                    ▼
                                          Sigmoidal Gold Hedge (GOLDBEES)
                                                    │
                                                    ▼
                                          Weekly Rebalance Output
```

**Key components:**

| Module | Description |
|--------|-------------|
| `engine/signal.py` | Core rebalancer — regime classification, portfolio construction, constraint enforcement |
| `engine/ml_engine.py` | XGBoost alpha model wrapper (`ChimeraEngineML`, `RollingChimeraEngineML`) |
| `models/regime/probabilistic.py` | Probabilistic regime classifier (BULL/CHOP/BEAR) |
| `models/regime/ml_regime.py` | XGBoost-based regime scoring |
| `run_forward_test.py` | Main production entry point — data update + engine run + reports |
| `scripts/train_xgb.py` | Retrain XGBoost alpha model |
| `scripts/train_regime.py` | Retrain regime classifier |
| `scripts/create_ml_dataset.py` | Build feature dataset from price parquets |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Price parquets (stocks, indices, macro) are not committed to this repo.
Place them in:
```
data/stocks/*.parquet
data/indices/*.parquet
data/macro/*.parquet
```
Each parquet must have columns: `Date, Open, High, Low, Close, Volume, Ticker`.

### 3. Run the full pipeline
```bash
# Full pipeline: update data → run engine → generate reports
python run_forward_test.py

# Skip data download (use existing parquets)
python run_forward_test.py --skip-data-update
```

### 4. Retrain models
```bash
# Build feature dataset
python scripts/create_ml_dataset.py

# Retrain XGBoost alpha model
python scripts/train_xgb.py

# Retrain regime classifier
python scripts/train_regime.py
```

---

## Configuration

All runtime parameters live in `engine/signal.py → CONFIG`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LONG_NAMES` | 20 | Max long positions |
| `FRICTION_BPS` | 15 | Round-trip transaction cost assumption |
| `LONG_GROSS_BULL` | 1.00 | Gross exposure in BULL regime |
| `LONG_GROSS_CHOP` | 0.30 | Gross exposure in CHOP regime |
| `LONG_GROSS_BEAR` | 0.40 | Gross exposure in BEAR regime |
| `USE_MVO` | False | Enable MVO optimizer (set True in production) |
| `GOLD_MODEL` | `'baseline'` | Gold allocation model (`sigmoid` = champion) |

Champion configuration (set in `run_forward_test.py`):
```python
CONFIG['USE_MVO'] = True
CONFIG['COV_ESTIMATOR'] = 'oas'
CONFIG['GOLD_MODEL'] = 'sigmoid'
```

---

## Validated Performance (Exp_R01 — git: f9cef41, 2026-07-20)

Champion config: `XGBoost Regr_Residual (rolling) + MVO(OAS) + Sigmoid Gold + REGIME_BANDS`

| Window | Sharpe | CAGR | Max DD | Weeks |
|--------|--------|------|--------|-------|
| Validation (Jan 2023 – Sep 2025) | **1.94** | **+37.2%** | -10.4% | 143 |
| Frozen Test (Oct 2025 – Jul 2026) | **0.56** | **+6.2%** | -10.6% | 40 |
| Full OOS (Jan 2023 – Jul 2026) | **1.70** | **+29.7%** | -10.6% | 183 |

**Constraints verified:** 0 sector cap breaches (>25%), 0 GOLDBEES policy cap breaches (>25%).

> **Capacity bound:** Square-root market impact modelling bounds viable AUM to ≈ ₹5–10 Crore (~$600K–$1.2M USD). Performance degrades sharply beyond ₹10 Crore.

---

## Constraints & Risk Controls

- **Single-stock cap:** 8% max position weight (enforced in MVO bounds)
- **Sector cap:** 25% max aggregate sector weight (enforced as SLSQP hard constraint, post-aggregation)
- **Gold sleeve cap:** 25% max GOLDBEES allocation (policy limit)
- **Max leverage:** 1.0× (long-only, no net leverage)
- **Regime gating:** CHOP/BEAR regimes reduce gross exposure to 30–40%

---

## Project Structure

```
hedgefund_chimera/
├── config/paths.py          — Path constants
├── engine/
│   ├── signal.py            — Core rebalancer (ChimeraEngineNormal)
│   ├── ml_engine.py         — ML engine (ChimeraEngineML, RollingChimeraEngineML)
│   └── analytics/           — Performance metrics
├── models/regime/
│   ├── probabilistic.py     — Regime classifier
│   ├── ml_regime.py         — XGBoost regime model
│   └── hmm_smoother.pkl     — Trained HMM (not committed)
├── scripts/
│   ├── train_xgb.py         — Retrain alpha model
│   ├── train_regime.py      — Retrain regime classifier
│   ├── create_ml_dataset.py — Feature pipeline
│   ├── validate_significance.py — Statistical audit
│   └── reconcile_champion.py    — Champion verification run
├── research/experiments/
│   ├── backtest_report.py   — Report generator
│   └── regime_validation.py — Regime performance reporter
├── data/
│   ├── evidence_registry.json        — Experiment log
│   └── negative_results_registry.json
├── run_forward_test.py      — Main entry point
└── requirements.txt
```

---

## Data Requirements

Price data is **not committed** to this repository (too large, proprietary). You need:
- NSE daily OHLCV parquets for ~500 stocks (2018–present)
- Nifty 50, BankNifty index parquets
- Macro parquets: India VIX, GOLDBEES, USD/INR, US10Y, Crude Oil, Silver

---

## Evidence Registry

Experiment results are tracked in `data/evidence_registry.json` and `data/negative_results_registry.json`. Key findings:

- `REGIME_BANDS` is the recommended live exposure mode
- `DYNAMIC_GEOMETRY` (continuous κ scaling) is deprecated to research-only (100% bootstrap drawdown penalty)
- MVO sector caps are applied post-aggregation via SLSQP — not affected by the pre-aggregation bug in the heuristic path
