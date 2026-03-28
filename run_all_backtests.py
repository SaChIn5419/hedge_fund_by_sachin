"""
CHIMERA MASTER BACKTEST RUNNER
==============================
Runs the strategy and creates a static report:
- PNG with regime shading, benchmark comparison, rolling diagnostics
- TXT with deeper metrics and break analysis
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from chimera_engine import ChimeraEngineNormal
from chimera_backtest_report import StaticBacktestReporter


def run_chimera_fip() -> str:
    print('\n' + '=' * 60)
    print('  CHIMERA FROG-IN-THE-PAN (MAIN)')
    print('=' * 60)
    engine = ChimeraEngineNormal()
    engine.run_simulation()
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'tradelog_chimera_fip.csv')
    print(f'  Trade log saved: {csv_path}')
    return csv_path


def generate_static_report(csv_path: str, strategy_name: str, output_prefix: str) -> bool:
    print(f'\n--- Generating Static Report: {strategy_name} ---')
    if not os.path.exists(csv_path):
        print('  SKIPPED: CSV not found')
        return False
    trades = pd.read_csv(csv_path, parse_dates=['date'])
    if trades.empty:
        print('  SKIPPED: Empty trade log')
        return False
    print(f'  Trade log: {len(trades)} rows')
    print(f'  Date range: {trades["date"].iloc[0].date()} -> {trades["date"].iloc[-1].date()}')

    reporter = StaticBacktestReporter(benchmark_ticker='^NSEI', capital=1_000_000)
    out = reporter.generate(csv_path, output_prefix, open_browser=False)
    print(f'  PNG report: {out.png_path}')
    print(f'  TXT report: {out.txt_path}')
    return True


if __name__ == '__main__':
    print('=' * 60)
    print('  CHIMERA MASTER BACKTEST RUNNER')
    print('  Static report mode: PNG + TXT')
    print('=' * 60)

    os.chdir(PROJECT_ROOT)
    os.makedirs(PROJECT_ROOT / 'data', exist_ok=True)

    csv_path = run_chimera_fip()
    print('\n' + '=' * 60)
    print('  GENERATING STATIC REPORTS')
    print('=' * 60)
    generate_static_report(csv_path, 'Chimera: Frog-in-the-Pan (Main)', str(PROJECT_ROOT / 'data' / 'report_chimera_fip'))

    print('\n' + '=' * 60)
    print('  BACKTEST COMPLETE — STATIC REPORT SUMMARY')
    print('=' * 60)
