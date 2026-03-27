"""
CHIMERA MASTER BACKTEST RUNNER
==============================
Runs strategy variants and generates interactive HTML tearsheets.
All strategies run with 1.0x max leverage (cash only, no borrowing).

Usage:
    python run_all_backtests.py
"""

import sys
import os
import pandas as pd
import numpy as np
import polars as pl
import warnings

warnings.filterwarnings("ignore")

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chimera_analytics import PolarsTearSheet


# ============================================================
# UTILITY: Convert Trade Log CSV → Equity Curve (Polars)
# ============================================================
def trade_log_to_equity(csv_path, initial_capital=1_000_000):
    """
    Reads a trade log CSV and converts it into a daily equity curve.

    Logic:
    - Group trades by date (each date = one rebalance)
    - Portfolio return for that period = sum(weight_i * fwd_return_i)
    - Compound from initial_capital

    Returns: (equity_df: pl.DataFrame, trades_df: pl.DataFrame)
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])

    if df.empty:
        print(f"  WARNING: Empty trade log in {csv_path}")
        return None, None

    # Ensure numeric columns
    for col in ["weight", "fwd_return", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalize column names across strategies
    # V1 uses 'energy'/'structure', V2/V3 use 'kinetic_energy'/'structure_tag'
    if "kinetic_energy" not in df.columns and "energy" in df.columns:
        df["kinetic_energy"] = (
            pd.to_numeric(df["energy"], errors="coerce").fillna(0) / 100.0
        )
    if "structure_tag" not in df.columns and "structure" in df.columns:
        df["structure_tag"] = df["structure"].fillna("N/A").astype(str)

    # Compute net_pnl if missing
    if "net_pnl" not in df.columns:
        df["net_pnl"] = df["fwd_return"] * initial_capital * df["weight"]
    else:
        df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0)

    # Add missing columns expected by analytics (with defaults)
    defaults = {
        "kinetic_energy": 0.0,
        "structure_tag": "N/A",
        "market_state": "BULL",
        "leverage_mult": 1.0,
        "nifty_vol": 0.15,
        "efficiency": 0.0,
        "decision_reason": "N/A",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Ensure string columns don't have NaN
    for col in ["structure_tag", "market_state", "decision_reason", "ticker"]:
        if col in df.columns:
            df[col] = df[col].fillna("N/A").astype(str)

    # Ensure numeric columns don't have NaN
    for col in [
        "kinetic_energy",
        "leverage_mult",
        "nifty_vol",
        "efficiency",
        "net_pnl",
        "weight",
        "fwd_return",
        "close",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Weekly portfolio returns (sum of weighted forward returns per rebalance date)
    weekly_rets = (
        df.groupby("date")
        .apply(lambda g: (g["weight"] * g["fwd_return"]).sum(), include_groups=False)
        .reset_index()
    )
    weekly_rets.columns = ["date", "portfolio_return"]
    weekly_rets = weekly_rets.sort_values("date").reset_index(drop=True)

    # Build compounding equity curve (ensure float from the start)
    equity_values = [float(initial_capital)]
    for ret in weekly_rets["portfolio_return"]:
        equity_values.append(equity_values[-1] * (1 + ret))

    # Create equity DataFrame
    first_date = weekly_rets["date"].iloc[0]
    dates = [first_date - pd.Timedelta(days=7)] + list(weekly_rets["date"])

    equity_df = pl.DataFrame({"date": dates, "equity": equity_values}).with_columns(
        pl.col("date").cast(pl.Date)
    )

    # Select only the columns the tearsheet needs, with clean types
    trades_clean = df[
        [
            "date",
            "ticker",
            "close",
            "weight",
            "fwd_return",
            "net_pnl",
            "kinetic_energy",
            "structure_tag",
            "market_state",
            "leverage_mult",
            "nifty_vol",
            "efficiency",
        ]
    ].copy()
    trades_clean["date"] = pd.to_datetime(trades_clean["date"])

    trades_pl = pl.from_pandas(trades_clean)
    trades_pl = trades_pl.with_columns(pl.col("date").cast(pl.Date))

    return equity_df, trades_pl


# ============================================================
# STRATEGY RUNNERS
# ============================================================


def run_chimera_fip():
    """Chimera: Frog-in-the-Pan (Main Strategy)"""
    print("\n" + "=" * 60)
    print("  CHIMERA FROG-IN-THE-PAN (MAIN)")
    print("=" * 60)

    import chimera_engine_local

    engine = chimera_engine_local.ChimeraEngineFinal()
    engine.run_simulation()

    csv_path = os.path.join(
        os.path.dirname(__file__), "data", "tradelog_chimera_fip.csv"
    )
    print(f"  Trade log saved: {csv_path}")

    return csv_path


# ============================================================
# TEARSHEET GENERATOR
# ============================================================
def generate_tearsheet(csv_path, strategy_name, output_html):
    """Converts a trade log CSV into an interactive Plotly tearsheet."""
    print(f"\n--- Generating Tearsheet: {strategy_name} ---")

    equity_df, trades_df = trade_log_to_equity(csv_path)

    if equity_df is None:
        print(f"  SKIPPED: No data for {strategy_name}")
        return False

    print(f"  Equity curve: {equity_df.height} data points")
    print(f"  Trade log: {trades_df.height} entries")
    print(f"  Date range: {equity_df['date'][0]} -> {equity_df['date'][-1]}")

    initial = equity_df["equity"][0]
    final = equity_df["equity"][-1]
    total_ret = (final / initial - 1) * 100
    print(f"  Total Return: {total_ret:.2f}%")

    # Generate tearsheet
    ts = PolarsTearSheet(risk_free_rate=0.06, benchmark_ticker="^NSEI")
    output_path = os.path.join(os.path.dirname(__file__), output_html)
    ts.generate(equity_df, trades_df, open_browser=False, output_file=output_path)

    return True


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CHIMERA MASTER BACKTEST RUNNER")
    print("  All strategies: 1.0x max (Cash Only, No Leverage)")
    print("=" * 60)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data", exist_ok=True)

    results = {}

    # --- RUN ALL STRATEGIES ---
    strategies = [
        (
            "Chimera: Frog-in-the-Pan (Main)",
            run_chimera_fip,
            "tearsheet_chimera_fip.html",
        ),
    ]

    csv_files = {}
    for name, runner, html_file in strategies:
        try:
            csv_path = runner()
            csv_files[name] = (csv_path, html_file)
        except Exception as e:
            print(f"\n  ERROR running {name}: {e}")
            import traceback

            traceback.print_exc()

    # --- GENERATE TEARSHEETS ---
    print("\n" + "=" * 60)
    print("  GENERATING TEARSHEETS")
    print("=" * 60)

    for name, (csv_path, html_file) in csv_files.items():
        if csv_path and os.path.exists(csv_path):
            try:
                success = generate_tearsheet(csv_path, name, html_file)
                if success:
                    results[name] = html_file
            except Exception as e:
                print(f"  ERROR generating tearsheet for {name}: {e}")
                import traceback

                traceback.print_exc()

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE — TEARSHEET SUMMARY")
    print("=" * 60)

    for name, html_file in results.items():
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), html_file)
        size_kb = os.path.getsize(full_path) / 1024
        print(f"  OK {name}: {html_file} ({size_kb:.0f} KB)")

    if results:
        # Open the first tearsheet in browser
        import webbrowser

        first_file = list(results.values())[0]
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), first_file)
        print(f"\n  Opening {first_file} in browser...")
        webbrowser.open("file://" + os.path.realpath(full_path))
    else:
        print("  !!! No tearsheets were generated.")
