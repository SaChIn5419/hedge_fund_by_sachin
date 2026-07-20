"""
CHIMERA FORWARD TEST PIPELINE
==============================
Single-file executable that orchestrates the full forward test:

  Step 0: Backup old trade log
  Step 1: Update market data (stocks + indices + macro) via yfinance
  Step 2: Re-run the Chimera engine with updated data
  Step 3: Generate static backtest reports (PNG + TXT)
  Step 4: Forward test analysis — isolate out-of-sample period

Usage:
    python run_forward_test.py
    python run_forward_test.py --skip-data-update   # skip yfinance download
    python run_forward_test.py --workers 10          # parallel download threads
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Project paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config.paths import (
    DATA_DIR,
    OUTPUT_DATA_DIR,
    PRIMARY_TRADELOG,
    REGIME_TRACE_PATH,
    WEEKLY_TRACE_PATH,
)

STOCKS_DIR = DATA_DIR / 'stocks'
INDICES_DIR = DATA_DIR / 'indices'
MACRO_DIR = DATA_DIR / 'macro'
FORWARD_REPORT_DIR = OUTPUT_DATA_DIR / 'forward_test'

# Yahoo Finance ticker mapping for indices and macro
INDEX_YF_MAP = {
    'nifty50': '^NSEI',
    'banknifty': '^NSEBANK',
}
MACRO_YF_MAP = {
    'india_vix': '^INDIAVIX',
    'goldbees': 'GOLDBEES.NS',
    'usdinr': 'INR=X',
    'us10y': '^TNX',
    'crude_oil': 'CL=F',
    'silver': 'SI=F',
    'gpr': None,          # no yfinance source
    'bankbeES': 'BANKBEES.NS',
    'juniorbeES': 'JUNIORBEES.NS',
    'niftybeES': 'NIFTYBEES.NS',
}


def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: BACKUP
# ═══════════════════════════════════════════════════════════════════════════
def step0_backup():
    log('STEP 0: Backing up old trade log...')
    backup_dir = OUTPUT_DATA_DIR / 'forward_test' / 'backup'
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in [PRIMARY_TRADELOG, REGIME_TRACE_PATH, WEEKLY_TRACE_PATH]:
        if path.exists():
            dst = backup_dir / f'{path.stem}_pre_forward{path.suffix}'
            shutil.copy2(path, dst)
            log(f'  Backed up: {path.name} -> {dst.name}')
    log('  Backup complete.')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: UPDATE MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════
def _get_last_date(parquet_path: Path) -> pd.Timestamp:
    df = pd.read_parquet(parquet_path)
    date_col = 'Date' if 'Date' in df.columns else 'date'
    return pd.to_datetime(df[date_col]).max()


def _update_single_parquet(parquet_path: Path, yf_ticker: str, end_date: str) -> dict:
    """Download new data and append to existing parquet. Returns status dict."""
    import yfinance as yf

    try:
        last_date = _get_last_date(parquet_path)
        start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        if start >= end_date:
            return {'ticker': parquet_path.stem, 'status': 'up-to-date'}

        new_data = yf.download(
            yf_ticker, start=start, end=end_date,
            progress=False, auto_adjust=True, timeout=15,
        )
        if new_data is None or new_data.empty:
            return {'ticker': parquet_path.stem, 'status': 'no-new-data'}

        # Flatten MultiIndex columns if present
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)

        new_data = new_data.reset_index()
        date_col = 'Date' if 'Date' in new_data.columns else new_data.columns[0]
        new_data = new_data.rename(columns={date_col: 'Date'})
        new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)

        # Read existing and match schema
        existing = pd.read_parquet(parquet_path)
        ticker_name = existing['Ticker'].iloc[0] if 'Ticker' in existing.columns else parquet_path.stem
        new_data['Ticker'] = ticker_name
        if 'Adj Close' not in new_data.columns:
            new_data['Adj Close'] = new_data['Close']

        # Ensure column order matches
        cols = [c for c in existing.columns if c in new_data.columns]
        new_data = new_data[cols]

        # Cast types to match
        for col in cols:
            new_data[col] = new_data[col].astype(existing[col].dtype)

        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        combined.to_parquet(parquet_path, index=False)

        added = len(combined) - len(existing)
        return {'ticker': parquet_path.stem, 'status': 'updated', 'added': added}

    except Exception as e:
        return {'ticker': parquet_path.stem, 'status': 'error', 'error': str(e)[:80]}


def step1_update_market_data(workers: int = 5):
    log('STEP 1: Updating market data via yfinance...')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # 1a. Update stocks (batch by groups to avoid rate limits)
    stock_files = sorted(STOCKS_DIR.glob('*.parquet'))
    log(f'  Found {len(stock_files)} stock files to check.')

    results = {'updated': 0, 'up-to-date': 0, 'no-new-data': 0, 'error': 0}
    errors = []

    for i, path in enumerate(stock_files):
        yf_ticker = f'{path.stem}.NS'
        res = _update_single_parquet(path, yf_ticker, end_date)
        results[res['status']] = results.get(res['status'], 0) + 1
        if res['status'] == 'error':
            errors.append(res)
        if (i + 1) % 50 == 0:
            log(f'    Progress: {i+1}/{len(stock_files)} | updated={results["updated"]} errors={results["error"]}')
            time.sleep(1.0)  # Rate limit pause

    log(f'  Stocks: updated={results["updated"]} up-to-date={results["up-to-date"]} '
        f'no-new={results["no-new-data"]} errors={results["error"]}')

    # 1b. Update indices
    log('  Updating indices...')
    for path in INDICES_DIR.glob('*.parquet'):
        yf_ticker = INDEX_YF_MAP.get(path.stem)
        if yf_ticker is None:
            yf_ticker = f'{path.stem}.NS'
        res = _update_single_parquet(path, yf_ticker, end_date)
        log(f'    {path.stem}: {res["status"]}')

    # 1c. Update macro
    log('  Updating macro...')
    for path in MACRO_DIR.glob('*.parquet'):
        yf_ticker = MACRO_YF_MAP.get(path.stem)
        if yf_ticker is None:
            log(f'    {path.stem}: skipped (no yfinance mapping)')
            continue
        res = _update_single_parquet(path, yf_ticker, end_date)
        log(f'    {path.stem}: {res["status"]}')

    if errors:
        log(f'  ⚠ {len(errors)} tickers failed (this is normal for delisted stocks)')

    log('  Market data update complete.')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: RUN CHIMERA ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def step2_run_engine():
    log('STEP 2: Running Chimera engine simulation...')
    from engine.ml_engine import RollingChimeraEngineML
    from engine.signal import CONFIG
    
    # Enable champion MVO optimization
    CONFIG['USE_MVO'] = True
    CONFIG['COV_ESTIMATOR'] = 'oas'
    
    # Initialize rolling engine with champion feature set
    features = ['fip_z', 'mom20_z', 'mom60_z', 'vol20_z', 'beta', 'rsi14', 'structure_score', 'rvol20', 'vol_comp']
    engine = RollingChimeraEngineML(model_prefix='Regr_Residual', features=features, is_ranker=False)
    engine.run_simulation()
    log('  Engine simulation complete.')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: GENERATE BACKTEST REPORTS
# ═══════════════════════════════════════════════════════════════════════════
def step3_generate_reports():
    log('STEP 3: Generating static backtest reports...')
    from research.experiments.backtest_report import StaticBacktestReporter
    from research.experiments.regime_validation import RegimeValidationReporter

    csv_path = str(PRIMARY_TRADELOG)
    if not os.path.exists(csv_path):
        log('  ERROR: Trade log not found!')
        return

    trades = pd.read_csv(csv_path, parse_dates=['date'])
    log(f'  Trade log: {len(trades)} rows, {trades["date"].iloc[0].date()} -> {trades["date"].iloc[-1].date()}')

    reporter = StaticBacktestReporter(benchmark_ticker='^NSEI', capital=1_000_000)
    out = reporter.generate(csv_path, str(OUTPUT_DATA_DIR / 'report_chimera_fip'), open_browser=False)
    log(f'  PNG report: {out.png_path}')
    log(f'  TXT report: {out.txt_path}')

    try:
        regime_reporter = RegimeValidationReporter(capital=1_000_000)
        rout = regime_reporter.generate(csv_path, str(OUTPUT_DATA_DIR / 'regime_validation'))
        log(f'  Regime validation: {rout.csv_path}')
    except Exception as e:
        log(f'  Regime validation skipped: {e}')

    log('  Reports complete.')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: FORWARD TEST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def step4_forward_test_analysis(cutoff_date: str = '2026-03-27'):
    log('STEP 4: Forward test analysis...')
    FORWARD_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use manual cutoff date
    old_last_date = pd.Timestamp(cutoff_date)
    new_path = PRIMARY_TRADELOG

    log(f'  Using cutoff date  : {old_last_date.date()}')
    new_trades = pd.read_csv(new_path, parse_dates=['date'])
    new_last_date = new_trades['date'].max()
    log(f'  New backtest ends  : {new_last_date.date()}')

    # Isolate forward period trades
    fwd_trades = new_trades[new_trades['date'] > old_last_date].copy()
    if fwd_trades.empty:
        log('  ⚠ No new trades in forward period! Data may not have updated.')
        return

    log(f'  Forward period: {fwd_trades["date"].min().date()} -> {fwd_trades["date"].max().date()}')
    log(f'  Forward trades: {len(fwd_trades)} rows across {fwd_trades["date"].nunique()} weeks')

    # ── Forward period metrics ─────────────────────────────────────────
    capital = 1_000_000
    fwd_daily = fwd_trades.groupby('date', as_index=False).agg(
        net_pnl=('net_pnl', 'sum'),
        gross_exposure=('gross_weight', 'sum'),
        net_exposure=('weight', 'sum'),
        regime=('market_state', lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        trade_count=('ticker', 'count'),
    ).sort_values('date')

    fwd_daily['portfolio_return'] = fwd_daily['net_pnl'] / capital
    fwd_daily['equity'] = capital * (1 + fwd_daily['portfolio_return']).cumprod()
    fwd_daily['peak'] = fwd_daily['equity'].cummax()
    fwd_daily['drawdown'] = fwd_daily['equity'] / fwd_daily['peak'] - 1.0

    fwd_rets = fwd_daily['portfolio_return'].to_numpy()
    fwd_total = float((1 + fwd_rets).prod() - 1)
    fwd_vol = float(np.std(fwd_rets, ddof=1) * np.sqrt(52)) if len(fwd_rets) > 1 else 0
    fwd_sharpe = float((np.mean(fwd_rets) * 52) / (fwd_vol + 1e-12)) if fwd_vol > 0 else 0
    fwd_dd = float(fwd_daily['drawdown'].min())
    fwd_win_rate = float((fwd_rets > 0).mean())
    fwd_final = float(fwd_daily['equity'].iloc[-1])

    # Regime breakdown
    regime_counts = fwd_daily['regime'].value_counts().to_dict()
    active_fwd = fwd_trades[fwd_trades['weight'].abs() > 1e-12]
    regime_pnl = active_fwd.groupby('market_state')['net_pnl'].sum().to_dict()

    # Top winners/losers in forward period
    top_win = active_fwd.nlargest(10, 'net_pnl')[['date', 'ticker', 'net_pnl', 'weight', 'market_state']]
    top_lose = active_fwd.nsmallest(10, 'net_pnl')[['date', 'ticker', 'net_pnl', 'weight', 'market_state']]

    # Benchmark comparison
    try:
        import yfinance as yf
        bench = yf.download('^NSEI', start=fwd_daily['date'].min(), end=fwd_daily['date'].max() + timedelta(days=1),
                            progress=False, auto_adjust=True)
        if isinstance(bench.columns, pd.MultiIndex):
            bench = bench.xs('Close', axis=1, level=0)
        elif 'Close' in bench.columns:
            bench = bench['Close']
        bench = bench.dropna()
        bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1) if len(bench) > 1 else 0
    except Exception:
        bench_ret = float('nan')

    # ── Write text report ──────────────────────────────────────────────
    txt_path = FORWARD_REPORT_DIR / 'forward_test_report.txt'
    with open(txt_path, 'w') as f:
        f.write('CHIMERA FORWARD TEST REPORT\n')
        f.write('=' * 60 + '\n')
        f.write(f'Analysis date      : {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'Old backtest ended : {old_last_date.date()}\n')
        f.write(f'Forward period     : {fwd_daily["date"].min().date()} -> {fwd_daily["date"].max().date()}\n')
        f.write(f'Forward weeks      : {fwd_daily["date"].nunique()}\n')
        f.write(f'Forward trade rows : {len(fwd_trades)}\n\n')
        f.write('FORWARD PERIOD PERFORMANCE\n')
        f.write(f'  Starting capital : ₹{capital:,.0f}\n')
        f.write(f'  Ending equity    : ₹{fwd_final:,.0f}\n')
        f.write(f'  Forward return   : {fwd_total * 100:.2f}%\n')
        f.write(f'  Volatility (ann.): {fwd_vol * 100:.2f}%\n')
        f.write(f'  Sharpe ratio     : {fwd_sharpe:.2f}\n')
        f.write(f'  Max drawdown     : {fwd_dd * 100:.2f}%\n')
        f.write(f'  Win rate (weeks) : {fwd_win_rate * 100:.1f}%\n')
        f.write(f'  Nifty50 return   : {bench_ret * 100:.2f}%\n' if not np.isnan(bench_ret) else '  Nifty50 return   : N/A\n')
        if not np.isnan(bench_ret):
            f.write(f'  Excess return    : {(fwd_total - bench_ret) * 100:.2f}%\n')
        f.write('\nREGIME DISTRIBUTION\n')
        for regime, count in sorted(regime_counts.items()):
            pnl = regime_pnl.get(regime, 0)
            f.write(f'  {regime:<8} weeks={count} pnl=₹{pnl:,.0f}\n')
        f.write('\nWEEKLY PNL BREAKDOWN\n')
        for _, row in fwd_daily.iterrows():
            sign = '+' if row['net_pnl'] >= 0 else ''
            f.write(f'  {row["date"].date()}  {row["regime"]:<6} PnL=₹{sign}{row["net_pnl"]:,.0f}  '
                    f'Equity=₹{row["equity"]:,.0f}  DD={row["drawdown"]*100:.1f}%\n')
        f.write('\nTOP FORWARD WINNERS\n')
        for _, r in top_win.iterrows():
            f.write(f'  {r["date"].date()}  {r["ticker"]:<14} ₹{r["net_pnl"]:,.0f}  w={r["weight"]:.3f}  {r["market_state"]}\n')
        f.write('\nTOP FORWARD LOSERS\n')
        for _, r in top_lose.iterrows():
            f.write(f'  {r["date"].date()}  {r["ticker"]:<14} ₹{r["net_pnl"]:,.0f}  w={r["weight"]:.3f}  {r["market_state"]}\n')

    log(f'  Forward test report: {txt_path}')

    # ── Write forward equity CSV ───────────────────────────────────────
    csv_out = FORWARD_REPORT_DIR / 'forward_equity_curve.csv'
    fwd_daily.to_csv(csv_out, index=False)
    log(f'  Forward equity CSV:  {csv_out}')

    # ── Print summary to console ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('  CHIMERA FORWARD TEST SUMMARY')
    print('=' * 60)
    print(f'  Period        : {fwd_daily["date"].min().date()} -> {fwd_daily["date"].max().date()}')
    print(f'  Return        : {fwd_total * 100:+.2f}%')
    print(f'  Sharpe        : {fwd_sharpe:.2f}')
    print(f'  Max Drawdown  : {fwd_dd * 100:.2f}%')
    print(f'  Win Rate      : {fwd_win_rate * 100:.1f}%')
    if not np.isnan(bench_ret):
        print(f'  Nifty50       : {bench_ret * 100:+.2f}%')
        print(f'  Excess        : {(fwd_total - bench_ret) * 100:+.2f}%')
    print(f'  Regimes       : {regime_counts}')
    print('=' * 60)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Chimera Forward Test Pipeline (resumable — re-run safely after interruptions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_forward_test.py                  # Full pipeline (resumes data download automatically)
  python run_forward_test.py --data-only      # Only download remaining stale stocks, then exit
  python run_forward_test.py --skip-data-update  # Skip download, run engine + reports with current data
        """,
    )
    parser.add_argument('--skip-data-update', action='store_true', help='Skip yfinance data download entirely')
    parser.add_argument('--data-only', action='store_true',
                        help='Only run the data update step (resume downloads), then exit. '
                             'Already-updated stocks are skipped automatically.')
    parser.add_argument('--skip-reports', action='store_true', help='Skip static report generation')
    parser.add_argument('--workers', type=int, default=5, help='Parallel download workers (unused currently)')
    parser.add_argument('--cutoff-date', type=str, default='2026-03-27',
                        help='Manual cutoff date for forward period (YYYY-MM-DD)')
    args = parser.parse_args()

    t0 = time.time()
    print('=' * 60)
    print('  CHIMERA FORWARD TEST PIPELINE')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    os.chdir(REPO_ROOT)

    # Step 0: Backup
    step0_backup()

    # Step 1: Update market data
    if args.skip_data_update:
        log('STEP 1: SKIPPED (--skip-data-update)')
    else:
        step1_update_market_data(workers=args.workers)

    if args.data_only:
        elapsed = time.time() - t0
        print(f'\n[Data update completed in {elapsed / 60:.1f} minutes — run without --data-only for full pipeline]')
        return

    # Step 2: Run engine
    step2_run_engine()

    # Step 3: Generate reports
    if args.skip_reports:
        log('STEP 3: SKIPPED (--skip-reports)')
    else:
        step3_generate_reports()

    # Step 4: Forward test analysis
    step4_forward_test_analysis(cutoff_date=args.cutoff_date)

    elapsed = time.time() - t0
    print(f'\n[Pipeline completed in {elapsed / 60:.1f} minutes]')


if __name__ == '__main__':
    main()
