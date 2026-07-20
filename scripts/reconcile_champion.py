"""
CHAMPION RECONCILIATION RUN — Exp_R01
======================================
Single, non-iterated run of the champion configuration:
  XGBoost Alpha (Regr_Residual) + MVO (OAS covariance) + Sigmoidal Gold Hedge
  + REGIME_BANDS exposure (the live production default)

Run exactly once. Log git commit hash. Report validation and frozen-test windows
separately, never blended. Do not re-run if numbers look uncomfortable.

Architecture note (read before modifying):
  - ChimeraEngineML._allocate_gross_budget() overrides the parent's exposure logic
    with a p_bear-based heuristic (not REGIME_BANDS / CONSTANT_DELEVERAGE flags).
    To use pure REGIME_BANDS, we patch the override so it falls back to the parent
    method which respects self.exposure_mode.
  - USE_MVO=True routes weight-setting through _optimize_portfolio_weights(), which
    applies sector caps as hard SLSQP constraints on the joint pool (post-aggregation
    by construction). The pre-aggregation sleeve bug was only in the heuristic else
    branch. This means the champion was NOT exposed to that bug.

Windows:
  Validation : 2023-01-01 – 2025-09-30  (same as all prior reports)
  Frozen Test : 2025-10-01 – present    (true blind window)
"""
from __future__ import annotations

import os
import sys
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# ── 1. Record git commit ────────────────────────────────────────────────────
def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
GIT_HASH = get_git_hash()
print("=" * 70)
print("  CHIMERA CHAMPION RECONCILIATION RUN — Exp_R01")
print(f"  Timestamp : {RUN_TIMESTAMP}")
print(f"  Git hash  : {GIT_HASH}")
print("=" * 70)

# ── 2. Load engine with champion configuration ──────────────────────────────
from engine.ml_engine import RollingChimeraEngineML
import engine.signal as sig_module

CONFIG = sig_module.CONFIG
CONFIG["USE_MVO"] = True
CONFIG["COV_ESTIMATOR"] = "oas"
CONFIG["GOLD_MODEL"] = "sigmoid"   # sigmoidal gold hedge = champion spec
CONFIG["FRICTION_BPS"] = 15        # default (15 bps) — same as all prior runs

FEATURES = ["fip_z", "mom20_z", "mom60_z", "vol20_z", "beta",
            "rsi14", "structure_score", "rvol20", "vol_comp"]

class ChampionEngine(RollingChimeraEngineML):
    """
    Thin wrapper that forces exposure allocation back to the parent
    ChimeraEngineNormal._allocate_gross_budget() (REGIME_BANDS mode),
    bypassing ChimeraEngineML's p_bear heuristic override.
    """
    def __init__(self):
        super().__init__(model_prefix="Regr_Residual", features=FEATURES, is_ranker=False)
        self.exposure_mode = "REGIME_BANDS"

    def _allocate_gross_budget(self, regime: str, num_longs: int,
                                num_shorts: int, confidence: float):
        # Skip ML engine override — call grandparent (ChimeraEngineNormal)
        from engine.signal import ChimeraEngineNormal
        return ChimeraEngineNormal._allocate_gross_budget(
            self, regime, num_longs, num_shorts, confidence
        )

print("\n[CONFIG]")
print(f"  USE_MVO        = {CONFIG['USE_MVO']}")
print(f"  COV_ESTIMATOR  = {CONFIG['COV_ESTIMATOR']}")
print(f"  GOLD_MODEL     = {CONFIG['GOLD_MODEL']}")
print(f"  FRICTION_BPS   = {CONFIG['FRICTION_BPS']}")
print(f"  exposure_mode  = REGIME_BANDS (via ChampionEngine override)")
print(f"  Features       = {FEATURES}")

# ── 3. Run the simulation ───────────────────────────────────────────────────
engine = ChampionEngine()
engine.run_simulation()

# ── 4. Load trade log ──────────────────────────────────────────────────────
from config.paths import PRIMARY_TRADELOG
trades = pd.read_csv(PRIMARY_TRADELOG, parse_dates=["date"])
print(f"\n[TRADE LOG] {len(trades)} rows | "
      f"{trades['date'].min().date()} → {trades['date'].max().date()}")

# ── 5. Window definitions ──────────────────────────────────────────────────
VAL_START   = pd.Timestamp("2023-01-01")
VAL_END     = pd.Timestamp("2025-09-30")
FROZEN_START = pd.Timestamp("2025-10-01")
FROZEN_END  = trades["date"].max()

CAPITAL = CONFIG["CAPITAL"]

# ── 6. Metrics helper ──────────────────────────────────────────────────────
def compute_metrics(df_window: pd.DataFrame, label: str) -> dict:
    if df_window.empty:
        print(f"\n[{label}] No data in window.")
        return {}

    weekly = (
        df_window.groupby("date")
        .agg(net_pnl=("net_pnl", "sum"),
             gross_exp=("gross_weight", "sum"),
             turnover=("weight", lambda x: x.abs().sum()))
        .sort_index()
        .reset_index()
    )
    weekly["ret"] = weekly["net_pnl"] / CAPITAL

    rets = weekly["ret"].to_numpy(dtype=float)
    n = len(rets)
    if n < 2:
        print(f"\n[{label}] Insufficient weeks ({n}).")
        return {}

    ann_factor = 52.0
    cum_ret = float((1 + rets).prod() - 1)
    ann_vol  = float(np.std(rets, ddof=1) * np.sqrt(ann_factor))
    sharpe   = float((np.mean(rets) * ann_factor) / (ann_vol + 1e-12))

    years    = n / ann_factor
    cagr     = float((1 + cum_ret) ** (1 / years) - 1) if years > 0 else 0.0

    equity   = (1 + pd.Series(rets)).cumprod()
    peak     = equity.cummax()
    dd       = (equity - peak) / (peak + 1e-9)
    max_dd   = float(dd.min())

    # Calmar
    calmar   = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else np.nan

    # Turnover
    total_turnover     = float(weekly["turnover"].sum())
    weeks_count        = n
    avg_weekly_turnover = total_turnover / weeks_count if weeks_count else 0

    # Average gross exposure
    avg_gross = float(weekly["gross_exp"].mean())

    # Sector cap breaches (post-aggregation check)
    # Build sector_map if available
    import json
    sector_map = {}
    smap_path = REPO_ROOT / "chimera_data" / "sector_map.json"
    if smap_path.exists():
        with open(smap_path) as f:
            sector_map = json.load(f)

    breaches = 0
    for date, grp in df_window.groupby("date"):
        eq_grp = grp[~grp["ticker"].isin(["GOLDBEES", "CASH"])]
        sector_totals = {}
        for _, row in eq_grp.iterrows():
            sec = sector_map.get(row["ticker"], "Unknown")
            sector_totals[sec] = sector_totals.get(sec, 0.0) + abs(row.get("weight", 0.0))
        for sec, w in sector_totals.items():
            if w > 0.25 + 1e-4:
                breaches += 1

    return {
        "label": label,
        "weeks": n,
        "cum_ret": cum_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "avg_gross_exp": avg_gross,
        "avg_weekly_turnover": avg_weekly_turnover,
        "sector_cap_breaches": breaches,
    }

# ── 7. Compute metrics per window ──────────────────────────────────────────
val_df   = trades[(trades["date"] >= VAL_START)   & (trades["date"] <= VAL_END)]
frozen_df = trades[(trades["date"] >= FROZEN_START) & (trades["date"] <= FROZEN_END)]
full_df  = trades[trades["date"] >= VAL_START]

val_m    = compute_metrics(val_df,   f"VALIDATION ({VAL_START.date()} – {VAL_END.date()})")
frozen_m = compute_metrics(frozen_df, f"FROZEN TEST ({FROZEN_START.date()} – {FROZEN_END.date()})")
full_m   = compute_metrics(full_df,  f"FULL OOS ({VAL_START.date()} – {FROZEN_END.date()})")

# ── 8. Print report ────────────────────────────────────────────────────────
def print_window(m: dict):
    if not m:
        return
    print(f"\n{'='*70}")
    print(f"  {m['label']}")
    print(f"{'='*70}")
    print(f"  Weeks              : {m['weeks']}")
    print(f"  CAGR               : {m['cagr']*100:+.2f}%")
    print(f"  Sharpe Ratio       : {m['sharpe']:.4f}")
    print(f"  Max Drawdown       : {m['max_dd']*100:.2f}%")
    print(f"  Calmar Ratio       : {m['calmar']:.2f}" if not np.isnan(m['calmar']) else "  Calmar Ratio       : N/A")
    print(f"  Cum Return         : {m['cum_ret']*100:.2f}%")
    print(f"  Annualized Vol     : {m['ann_vol']*100:.2f}%")
    print(f"  Avg Gross Exposure : {m['avg_gross_exp']:.4f}")
    print(f"  Avg Weekly Turnover: {m['avg_weekly_turnover']:.4f}")
    print(f"  Sector Cap Breaches: {m['sector_cap_breaches']}")

print_window(val_m)
print_window(frozen_m)
print_window(full_m)

# ── 9. Gold allocation check ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("  GOLD SLEEVE AUDIT")
print(f"{'='*70}")
gold_trades = trades[(trades["ticker"] == "GOLDBEES") & (trades["date"] >= VAL_START)]
if not gold_trades.empty:
    print(f"  Gold weeks active  : {gold_trades['date'].nunique()}")
    print(f"  Gold weight — mean : {gold_trades['weight'].mean():.4f}")
    print(f"  Gold weight — max  : {gold_trades['weight'].max():.4f}")
    print(f"  Gold weight — min  : {gold_trades['weight'].min():.4f}")
    print(f"  Policy cap breach  : {(gold_trades['weight'] > 0.25 + 1e-4).sum()} weeks")
else:
    print("  No GOLDBEES trades found in OOS window.")

# ── 10. Sector concentration check ────────────────────────────────────────
print(f"\n{'='*70}")
print("  SECTOR CONCENTRATION AUDIT (Validation window)")
print(f"{'='*70}")
try:
    import json
    sector_map = {}
    smap_path = REPO_ROOT / "chimera_data" / "sector_map.json"
    if smap_path.exists():
        with open(smap_path) as f:
            sector_map = json.load(f)
    eq_trades = val_df[~val_df["ticker"].isin(["GOLDBEES", "CASH"])].copy()
    eq_trades["sector"] = eq_trades["ticker"].map(sector_map).fillna("Unknown")
    sec_avg = (
        eq_trades.groupby(["date", "sector"])["weight"]
        .apply(lambda x: x.abs().sum())
        .reset_index()
        .groupby("sector")["weight"]
        .mean()
        .sort_values(ascending=False)
    )
    print("  Average sector weight (validation window):")
    for sec, w in sec_avg.head(10).items():
        print(f"    {sec:<25} {w*100:.2f}%")
except Exception as e:
    print(f"  Sector audit skipped: {e}")

# ── 11. Summary banner ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  RECONCILIATION SUMMARY")
print(f"{'='*70}")
print(f"  Run timestamp  : {RUN_TIMESTAMP}")
print(f"  Git hash       : {GIT_HASH}")
print(f"  Config         : XGBoost Regr_Residual + MVO(OAS) + Sigmoid Gold + REGIME_BANDS")
print()
if val_m:
    print(f"  VALIDATION     Sharpe={val_m['sharpe']:.4f}  CAGR={val_m['cagr']*100:+.2f}%  MaxDD={val_m['max_dd']*100:.2f}%")
if frozen_m:
    print(f"  FROZEN TEST    Sharpe={frozen_m['sharpe']:.4f}  CAGR={frozen_m['cagr']*100:+.2f}%  MaxDD={frozen_m['max_dd']*100:.2f}%")
if full_m:
    print(f"  FULL OOS       Sharpe={full_m['sharpe']:.4f}  CAGR={full_m['cagr']*100:+.2f}%  MaxDD={full_m['max_dd']*100:.2f}%")
print(f"\n  Prior headline (all prior reports): Sharpe=1.66  CAGR=+19.76%")
if val_m:
    delta = val_m['sharpe'] - 1.66
    print(f"  Sharpe delta vs prior headline    : {delta:+.4f}")
print(f"{'='*70}\n")
