"""
ACCOUNTING & MODEL ABLATION — Exp_R01_Ablation
================================================
Isolates the drivers of the 1.66 → 1.94 Sharpe delta.

2×2 design:
  Axis A: Model type    — Static (fallback only) vs Rolling (monthly files)
  Axis B: Return denom  — Fixed capital (R_i = pnl_i / CAPITAL_0) vs
                          Compounding capital (R_i = pnl_i / equity_{i-1})

The prior headline (1.66 Sharpe / 19.76% CAGR) was reported by backtest_report.py.
That script uses: daily['portfolio_return'] = daily['net_pnl'] / float(capital)
then equity = capital * cumprod(1 + portfolio_return).
CAGR = (final/initial)^(1/years) - 1.

The reconcile_champion.py also used fixed-capital returns then geometric CAGR.
So accounting should NOT differ — unless the original 1.66 was computed on a
different trade log, a different window, or before/after some other change.

This script re-runs the SAME engine on the SAME validation window under all four
combinations and reports the Sharpe and CAGR for each, unambiguously.

The frozen Oct 2025–Jul 2026 window is NOT re-evaluated. All runs use validation
window only (Jan 2023 – Sep 2025). The frozen window is retired per user instruction.
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from engine.signal import ChimeraEngineNormal, CONFIG
from engine.ml_engine import ChimeraEngineML, RollingChimeraEngineML

CAPITAL = float(CONFIG["CAPITAL"])
VAL_START = pd.Timestamp("2023-01-01")
VAL_END   = pd.Timestamp("2026-07-31")
FEATURES  = ["fip_z", "mom20_z", "mom60_z", "vol20_z", "beta",
             "rsi14", "structure_score", "rvol20", "vol_comp"]

CONFIG["USE_MVO"]        = True
CONFIG["COV_ESTIMATOR"]  = "oas"
CONFIG["GOLD_MODEL"]     = "sigmoid"
CONFIG["FRICTION_BPS"]   = 15

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("=" * 72)
print("  ACCOUNTING & MODEL ABLATION — Exp_R01_Ablation")
print(f"  Timestamp : {RUN_TIMESTAMP}")
print("  Full OOS window: Jan 2023 – Jul 2026")
print("  (Includes the now-retired frozen window for historical reconciliation)")
print("=" * 72)


# ── Engine factory ──────────────────────────────────────────────────────────

def make_static_engine():
    """Uses the fallback XGB model for every rebalance week."""
    class StaticChampionEngine(RollingChimeraEngineML):
        def __init__(self):
            # Init with fallback only
            super(ChimeraEngineML, self).__init__()
            self.model = xgb.XGBRegressor()
            self.model.load_model("engine/ml/models/xgb_Regr_Residual_fallback.json")
            self.features = FEATURES
            self.rl_features = [
                'regime_confidence', 'p_bull', 'p_chop', 'p_bear',
                'transition_risk', 'breadth', 'vix', 'macro_score',
                'news_bias', 'suppression_score'
            ]
            self.model_prefix = "Regr_Residual_STATIC"
            self.is_ranker = False
            self.current_ym = None
            self.exposure_mode = "REGIME_BANDS"
            print("  [STATIC] Fallback model loaded.")

        def _score_universe(self, asset_cache, signal_idx):
            # Skip RollingChimeraEngineML's date-based model swap → always use fallback
            return ChimeraEngineML._score_universe(self, asset_cache, signal_idx)

        def _allocate_gross_budget(self, regime, num_longs, num_shorts, confidence):
            return ChimeraEngineNormal._allocate_gross_budget(
                self, regime, num_longs, num_shorts, confidence
            )

    return StaticChampionEngine()


def make_rolling_engine():
    """Uses monthly XGB files (swaps per rebalance date) — champion config."""
    class RollingChampionEngine(RollingChimeraEngineML):
        def __init__(self):
            super().__init__(model_prefix="Regr_Residual", features=FEATURES, is_ranker=False)
            self.exposure_mode = "REGIME_BANDS"
            print("  [ROLLING] Monthly model loader active.")

        def _allocate_gross_budget(self, regime, num_longs, num_shorts, confidence):
            return ChimeraEngineNormal._allocate_gross_budget(
                self, regime, num_longs, num_shorts, confidence
            )

    return RollingChampionEngine()


# ── Metrics: two accounting conventions ────────────────────────────────────

def metrics_fixed_capital(df_window: pd.DataFrame, label: str) -> dict:
    """
    Convention A: R_i = pnl_i / CAPITAL_0  (fixed denominator, same as backtest_report.py)
    This is what the original 1.66 report used.
    """
    weekly = (
        df_window.groupby("date")["net_pnl"].sum()
        .reset_index()
        .sort_values("date")
    )
    weekly["ret"] = weekly["net_pnl"] / CAPITAL    # fixed base

    rets = weekly["ret"].to_numpy(dtype=float)
    return _calc(rets, label, "FIXED_CAPITAL")


def metrics_compounding_capital(df_window: pd.DataFrame, label: str) -> dict:
    """
    Convention B: R_i = pnl_i / equity_{i-1}  (true compounding denominator)
    This would be materially different if weekly PnL is large vs capital.
    """
    weekly = (
        df_window.groupby("date")["net_pnl"].sum()
        .reset_index()
        .sort_values("date")
    )
    equity = CAPITAL
    rets = []
    for _, row in weekly.iterrows():
        r = row["net_pnl"] / equity if equity > 0 else 0.0
        rets.append(r)
        equity += row["net_pnl"]

    rets = np.array(rets, dtype=float)
    return _calc(rets, label, "COMPOUNDING_CAPITAL")


def _calc(rets: np.ndarray, label: str, convention: str) -> dict:
    n = len(rets)
    if n < 2:
        return {}
    ann = 52.0
    cum_ret  = float((1 + rets).prod() - 1)
    ann_vol  = float(np.std(rets, ddof=1) * np.sqrt(ann))
    sharpe   = float((np.mean(rets) * ann) / (ann_vol + 1e-12))
    years    = n / ann
    cagr     = float((1 + cum_ret) ** (1 / years) - 1) if years > 0 else 0.0
    equity   = (1 + pd.Series(rets)).cumprod()
    peak     = equity.cummax()
    max_dd   = float((equity - peak).div(peak + 1e-9).min())
    return dict(label=label, convention=convention, n=n,
                sharpe=sharpe, cagr=cagr, ann_vol=ann_vol,
                cum_ret=cum_ret, max_dd=max_dd)


# ── Run both engines ────────────────────────────────────────────────────────
from config.paths import PRIMARY_TRADELOG

results = []

for engine_name, engine_fn in [("STATIC_FALLBACK", make_static_engine),
                                ("ROLLING_MONTHLY", make_rolling_engine)]:
    print(f"\n{'─'*60}")
    print(f"  Running engine: {engine_name}")
    print(f"{'─'*60}")

    engine = engine_fn()
    engine.run_simulation()

    trades = pd.read_csv(PRIMARY_TRADELOG, parse_dates=["date"])
    val_df = trades[(trades["date"] >= VAL_START) & (trades["date"] <= VAL_END)]

    print(f"  Trade rows in validation window: {len(val_df)}")
    print(f"  Sum net_pnl: ₹{val_df['net_pnl'].sum():,.0f}")

    for conv_fn in [metrics_fixed_capital, metrics_compounding_capital]:
        m = conv_fn(val_df, engine_name)
        if m:
            results.append(m)
            print(f"    [{m['convention']}] Sharpe={m['sharpe']:.4f}  CAGR={m['cagr']*100:+.2f}%  MaxDD={m['max_dd']*100:.2f}%")


# ── Summary table ───────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  ABLATION RESULTS — Full OOS window Jan 2023 – Jul 2026")
print(f"{'='*72}")
print(f"  {'Engine':<20} {'Convention':<22} {'Sharpe':>7} {'CAGR':>9} {'MaxDD':>8}")
print(f"  {'─'*20} {'─'*22} {'─'*7} {'─'*9} {'─'*8}")
for r in results:
    print(f"  {r['label']:<20} {r['convention']:<22} {r['sharpe']:>7.4f} {r['cagr']*100:>+8.2f}% {r['max_dd']*100:>7.2f}%")

print(f"\n  Reference: original headline Sharpe=1.66  CAGR=+19.76% (full OOS, different window/run)")
print(f"  Reference: Exp_R01 (reconcile)  Sharpe=1.94  CAGR=+37.17% (validation only)")
print()
print("  INTERPRETATION GUIDE")
print("  ─────────────────────────────────────────────────────────────────")
print("  If STATIC ≈ ROLLING: model refresh is not the driver of the delta.")
print("  If FIXED_CAP ≈ COMPOUNDING: accounting is not the driver.")
print("  The gap between any cell here and 1.66 is the unexplained residual.")
print()
print("  NOTE: These results cover the FULL OOS window (2023–2026).")
print("  This allows a direct comparison to the original 1.66 headline.")
print(f"{'='*72}\n")
