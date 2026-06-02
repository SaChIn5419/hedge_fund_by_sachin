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

from config.paths import OUTPUT_DATA_DIR, REPO_ROOT
from engine.ml_engine import ChimeraEngineML
from research.experiments.backtest_report import StaticBacktestReporter
from research.experiments.regime_validation import RegimeValidationReporter

warnings.filterwarnings('ignore')

PROJECT_ROOT = REPO_ROOT
sys.path.insert(0, str(PROJECT_ROOT))


def run_chimera_fip() -> str:
    print('\n' + '=' * 60)
    print('  CHIMERA FROG-IN-THE-PAN (MAIN)')
    print('=' * 60)
    engine = ChimeraEngineML()
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


def generate_regime_validation(csv_path: str, output_prefix: str) -> bool:
    print('\n--- Generating Regime Validation Report ---')
    if not os.path.exists(csv_path):
        print('  SKIPPED: CSV not found')
        return False
    reporter = RegimeValidationReporter(capital=1_000_000)
    out = reporter.generate(csv_path, output_prefix)
    print(f'  Validation CSV: {out.csv_path}')
    print(f'  Validation TXT: {out.txt_path}')
    return True


if __name__ == '__main__':
    print('=' * 60)
    print('  CHIMERA MASTER BACKTEST RUNNER')
    print('  Static report mode: PNG + TXT')
    print('=' * 60)

    os.chdir(PROJECT_ROOT)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    csv_path = run_chimera_fip()
    print('\n' + '=' * 60)
    print('  GENERATING STATIC REPORTS')
    print('=' * 60)
    generate_static_report(csv_path, 'Chimera: Frog-in-the-Pan (Main)', str(OUTPUT_DATA_DIR / 'report_chimera_fip'))
    generate_regime_validation(csv_path, str(OUTPUT_DATA_DIR / 'regime_validation'))

    print('\n' + '=' * 60)
    print('  BACKTEST COMPLETE — STATIC REPORT SUMMARY')
    print('=' * 60)
