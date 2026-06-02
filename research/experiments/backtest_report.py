from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from config.paths import DATA_DIR, REPO_ROOT

try:
    import yfinance as yf
except Exception:
    yf = None

PROJECT_ROOT = str(REPO_ROOT)

REGIME_COLORS = {
    'BULL': '#2ca02c',
    'BEAR': '#d62728',
    'CHOP': '#ff7f0e',
    'CASH': '#7f7f7f',
}


def _annual_factor_from_dates(dates: pd.Series) -> int:
    if len(dates) < 3:
        return 52
    gap = (pd.to_datetime(dates.iloc[-1]) - pd.to_datetime(dates.iloc[0])).days / max(len(dates), 1)
    return 252 if gap <= 4 else 52


def _safe_sharpe(rets: np.ndarray, annual_factor: int) -> float:
    if len(rets) < 2:
        return 0.0
    sd = np.std(rets, ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((np.mean(rets) * annual_factor) / (sd * np.sqrt(annual_factor)))


def _safe_sortino(rets: np.ndarray, annual_factor: int) -> float:
    if len(rets) < 2:
        return 0.0
    downside = rets[rets < 0]
    if len(downside) == 0:
        return 0.0
    sd = np.std(downside, ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((np.mean(rets) * annual_factor) / (sd * np.sqrt(annual_factor)))


def _rolling_max_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _download_benchmark(start_date: pd.Timestamp, end_date: pd.Timestamp, ticker: str = '^NSEI') -> Optional[pd.Series]:
    if yf is None:
        return None
    try:
        bench = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False, auto_adjust=True)
        if bench is None or bench.empty:
            return None
        if isinstance(bench.columns, pd.MultiIndex):
            if 'Close' in bench.columns.get_level_values(0):
                bench = bench.xs('Close', axis=1, level=0)
                if isinstance(bench, pd.DataFrame):
                    bench = bench.iloc[:, 0]
        elif 'Close' in bench.columns:
            bench = bench['Close']
        bench = pd.Series(bench, name='benchmark_close')
        bench.index = pd.to_datetime(bench.index).tz_localize(None)
        return bench.sort_index()
    except Exception:
        return None


def _build_equity_curve(trades: pd.DataFrame, capital: float = 1_000_000) -> pd.DataFrame:
    trades = trades.copy()
    trades['date'] = pd.to_datetime(trades['date'])
    trades['net_pnl'] = pd.to_numeric(trades.get('net_pnl', 0), errors='coerce').fillna(0.0)
    daily = trades.groupby('date', as_index=False)['net_pnl'].sum().sort_values('date')
    daily['portfolio_return'] = daily['net_pnl'] / float(capital)
    daily['equity'] = float(capital) * (1.0 + daily['portfolio_return']).cumprod()
    daily['peak'] = daily['equity'].cummax()
    daily['drawdown'] = daily['equity'] / daily['peak'] - 1.0
    return daily


def _detect_multiple_breaks(strategy_ret: pd.Series, bench_ret: pd.Series, dates: pd.Series, max_breaks: int = 3, min_size: int = 30) -> List[Tuple[pd.Timestamp, float]]:
    """
    Recursively detects structural breaks using the CUSUM of Alpha.
    Returns a list of (date, score) sorted by prominence.
    """
    all_breaks = []

    def find_best_break(s_ret, b_ret, d_series):
        alpha = (s_ret - b_ret).fillna(0.0)
        if len(alpha) < min_size:
            return None, 0.0
        demeaned = alpha - alpha.mean()
        cusum = demeaned.cumsum()
        idx = int(np.argmax(np.abs(cusum.to_numpy())))
        return pd.to_datetime(d_series.iloc[idx]), float(np.abs(cusum.iloc[idx]))

    segments = [(strategy_ret, bench_ret, dates)]
    
    while segments and len(all_breaks) < max_breaks:
        s_ret, b_ret, d_series = segments.pop(0)
        date, score = find_best_break(s_ret, b_ret, d_series)
        
        if date is None or score <= 0:
            continue
            
        all_breaks.append((date, score))
        split_idx = int(np.where(d_series == date)[0][0])
        segments.append((s_ret.iloc[:split_idx], b_ret.iloc[:split_idx], d_series.iloc[:split_idx]))
        segments.append((s_ret.iloc[split_idx+1:], b_ret.iloc[split_idx+1:], d_series.iloc[split_idx+1:]))
    
    return sorted(all_breaks, key=lambda x: x[1], reverse=True)[:max_breaks]


def _detect_break_date(strategy_ret: pd.Series, bench_ret: pd.Series, dates: pd.Series) -> Tuple[Optional[pd.Timestamp], float]:
    """Simple wrapper to find the single most prominent structural break."""
    breaks = _detect_multiple_breaks(strategy_ret, bench_ret, dates, max_breaks=1)
    if not breaks:
        return None, 0.0
    return breaks[0]












def _make_transition_table(state_series: pd.Series) -> pd.DataFrame:
    states = state_series.dropna().astype(str).tolist()
    if len(states) < 2:
        return pd.DataFrame()
    pairs = list(zip(states[:-1], states[1:]))
    df = pd.DataFrame(pairs, columns=['from', 'to'])
    table = pd.crosstab(df['from'], df['to'])
    return table


def _split_pre_post(daily: pd.DataFrame, split_date: Optional[pd.Timestamp]) -> Dict[str, Dict[str, float]]:
    if split_date is None or daily.empty:
        return {}
    pre = daily[daily['date'] < split_date]
    post = daily[daily['date'] >= split_date]
    out: Dict[str, Dict[str, float]] = {}
    for name, frame in [('pre', pre), ('post', post)]:
        if frame.empty:
            continue
        rets = frame['portfolio_return'].to_numpy(dtype=float)
        out[name] = {
            'total_return': float((1 + rets).prod() - 1),
            'sharpe': _safe_sharpe(rets, 52),
            'sortino': _safe_sortino(rets, 52),
            'win_rate': float((rets > 0).mean()),
            'avg_return': float(np.mean(rets)),
            'volatility': float(np.std(rets, ddof=1) * np.sqrt(52)) if len(rets) > 1 else 0.0,
            'max_dd': float(_rolling_max_drawdown((1 + frame['portfolio_return']).cumprod()).min()),
        }
    return out


@dataclass
class ReportOutputs:
    png_path: str
    txt_path: str
    equity_path: str
    rolling_path: str
    drawdown_path: str
    regime_path: str


class StaticBacktestReporter:
    def __init__(self, benchmark_ticker: str = '^NSEI', capital: float = 1_000_000):
        self.benchmark_ticker = benchmark_ticker
        self.capital = capital

    def _load_trade_log(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError('empty trade log')
        df['date'] = pd.to_datetime(df['date'])
        if 'signal_date' in df.columns:
            df['signal_date'] = pd.to_datetime(df['signal_date'])
        if 'exit_date' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_date'])
        for col in ['weight', 'gross_weight', 'fwd_return', 'net_pnl', 'leverage_mult', 'regime_confidence', 'regime_score', 'breadth', 'macro_score', 'news_bias', 'suppression_score', 'p_bull', 'p_chop', 'p_bear', 'transition_risk', 'mom_z', 'break_score', 'score', 'long_score', 'short_score', 'nifty_vol', 'efficiency']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _daily_summary(self, trades: pd.DataFrame) -> pd.DataFrame:
        trades = trades.copy()
        trades['date'] = pd.to_datetime(trades['date'])
        if 'market_state' not in trades.columns:
            trades['market_state'] = 'UNKNOWN'
        if 'side' not in trades.columns:
            trades['side'] = np.where(trades['weight'] >= 0, 'LONG', 'SHORT')
        grouped = trades.groupby('date')
        daily = grouped.agg(
            net_pnl=('net_pnl', 'sum'),
            gross_exposure=('gross_weight', 'sum'),
            net_exposure=('weight', 'sum'),
            regime_confidence=('regime_confidence', 'mean'),
            regime_score=('regime_score', 'mean'),
            p_bull=('p_bull', 'mean'),
            p_chop=('p_chop', 'mean'),
            p_bear=('p_bear', 'mean'),
            transition_risk=('transition_risk', 'mean'),
            breadth=('breadth', 'mean'),
            macro_score=('macro_score', 'mean'),
            news_bias=('news_bias', 'mean'),
            suppression_score=('suppression_score', 'mean'),
            avg_mom_z=('mom_z', 'mean'),
            avg_efficiency=('efficiency', 'mean'),
        ).reset_index()
        # Compute long_pnl separately to avoid broken lambda closure in groupby
        long_pnl = trades[trades['side'] == 'LONG'].groupby('date')['net_pnl'].sum().rename('long_pnl')
        daily = daily.merge(long_pnl, on='date', how='left')
        daily['long_pnl'] = daily['long_pnl'].fillna(0.0)
        daily['portfolio_return'] = daily['net_pnl'] / float(self.capital)
        daily['equity'] = float(self.capital) * (1.0 + daily['portfolio_return']).cumprod()
        daily['peak'] = daily['equity'].cummax()
        daily['drawdown'] = daily['equity'] / daily['peak'] - 1.0
        daily['market_state'] = grouped['market_state'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]).values
        return daily


    def _benchmark_series(self, daily_dates: pd.Series) -> Optional[pd.Series]:
        start, end = pd.to_datetime(daily_dates.min()), pd.to_datetime(daily_dates.max())
        return _download_benchmark(start, end, self.benchmark_ticker)

    def _make_plot(self, daily: pd.DataFrame, bench: Optional[pd.Series], regime_trace: Optional[pd.DataFrame], break_date: Optional[pd.Timestamp], out_png: str):
        fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True, constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes

        x = pd.to_datetime(daily['date'])
        bench_eq = None
        bench_ret = None
        if bench is not None and not bench.empty:
            bench = bench.reindex(x).ffill().dropna()
            if not bench.empty:
                bench_eq = bench * (self.capital / float(bench.iloc[0]))
                bench_ret = bench.pct_change().fillna(0.0)

        # Regime shading
        reg = regime_trace.copy() if regime_trace is not None and not regime_trace.empty else None
        if reg is not None:
            reg['date'] = pd.to_datetime(reg['date'])
            reg = reg.sort_values('date').dropna(subset=['regime'])
            starts = reg['date'].tolist()
            states = reg['regime'].astype(str).tolist()
            for i, start in enumerate(starts):
                end = starts[i + 1] if i + 1 < len(starts) else x.iloc[-1]
                color = REGIME_COLORS.get(states[i], '#999999')
                for ax in axes:
                    ax.axvspan(start, end, color=color, alpha=0.06)

        ax1.plot(x, daily['equity'], label='Strategy', linewidth=2.2)
        if bench_eq is not None:
            ax1.plot(bench_eq.index, bench_eq.values, label='Nifty50', linestyle='--', linewidth=1.8)
        ax1.set_yscale('log')
        ax1.set_title('Chimera Equity vs Benchmark with Regime Shading')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.2)

        if bench_ret is not None and len(bench_ret) > 0:
            # Strategy returns must share the same DatetimeIndex as benchmark
            strat_ret_series = daily.set_index(pd.to_datetime(daily['date']))['portfolio_return']
            strat_ret = strat_ret_series.reindex(bench_ret.index).fillna(0.0)
            rolling = 26 if len(daily) > 60 else max(5, len(daily) // 4)
            cov_sb = strat_ret.rolling(rolling, min_periods=max(6, rolling // 2)).cov(bench_ret)
            var_b = bench_ret.rolling(rolling, min_periods=max(6, rolling // 2)).var()
            beta = (cov_sb / var_b.replace(0, np.nan)).dropna()
            roll_sharpe = (
                strat_ret.rolling(rolling, min_periods=max(6, rolling // 2)).mean()
                / strat_ret.rolling(rolling, min_periods=max(6, rolling // 2)).std().replace(0, np.nan)
                * np.sqrt(52)
            ).dropna()
            ax2.plot(beta.index, beta.values, label='Rolling Beta', linewidth=1.5)
            ax2.plot(roll_sharpe.index, roll_sharpe.values, label='Rolling Sharpe', linewidth=1.5, color='orange')
            ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax2.axhline(1, color='grey', linewidth=0.5, alpha=0.3, linestyle='--')
            ax2.legend(loc='upper left')
            ax2.grid(alpha=0.2)
            ax2.set_title('Rolling Beta and Rolling Sharpe')
        else:
            ax2.plot(x, daily['regime_confidence'], label='Regime confidence')
            ax2.legend(loc='upper left')
            ax2.grid(alpha=0.2)

        ax3.plot(x, daily['drawdown'], label='Strategy Drawdown')
        if bench_eq is not None:
            bench_dd = bench_eq / bench_eq.cummax() - 1.0
            ax3.plot(bench_dd.index, bench_dd.values, linestyle='--', label='Nifty50 Drawdown')
        ax3.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax3.legend(loc='lower left')
        ax3.grid(alpha=0.2)
        ax3.set_title('Drawdown')

        ax4.plot(x, daily['regime_confidence'], label='Regime Confidence')
        if 'p_bull' in daily.columns:
            ax4.plot(x, daily['p_bull'], label='P(Bull)')
        if 'p_bear' in daily.columns:
            ax4.plot(x, daily['p_bear'], label='P(Bear)')
        if 'transition_risk' in daily.columns:
            ax4.plot(x, daily['transition_risk'], label='Transition Risk')
        if 'breadth' in daily.columns:
            ax4.plot(x, daily['breadth'], label='Breadth')
        if 'macro_score' in daily.columns:
            ax4.plot(x, daily['macro_score'], label='Macro score')
        ax4.legend(loc='upper left')
        ax4.grid(alpha=0.2)
        ax4.set_title('Regime Inputs')

        if break_date is not None:
            for ax in axes:
                ax.axvline(break_date, color='purple', linestyle=':', linewidth=2.2)
            ax1.text(break_date, ax1.get_ylim()[1], ' break', color='purple', fontsize=10, va='top')

        regime_handles = [Patch(facecolor=c, alpha=0.18, label=k) for k, c in REGIME_COLORS.items()]
        ax1.legend(handles=ax1.get_legend_handles_labels()[0] + regime_handles, labels=ax1.get_legend_handles_labels()[1] + [p.get_label() for p in regime_handles], loc='upper left', fontsize=9)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0)
        fig.savefig(out_png, dpi=180, bbox_inches='tight')
        plt.close(fig)

    def _write_txt(self, daily: pd.DataFrame, trades: pd.DataFrame, bench: Optional[pd.Series], regime_trace: Optional[pd.DataFrame], out_txt: str, break_date: Optional[pd.Timestamp], break_score: float):
        rets = daily['portfolio_return'].to_numpy(dtype=float)
        annual_factor = _annual_factor_from_dates(daily['date'])
        initial = float(self.capital)
        final = float(daily['equity'].iloc[-1])
        cum_ret = final / initial - 1.0
        years = max((pd.to_datetime(daily['date'].iloc[-1]) - pd.to_datetime(daily['date'].iloc[0])).days / 365.25, 1e-9)
        cagr = (final / initial) ** (1 / years) - 1.0
        vol_ann = float(np.std(rets, ddof=1) * np.sqrt(annual_factor)) if len(rets) > 1 else 0.0
        sharpe = _safe_sharpe(rets, annual_factor)
        sortino = _safe_sortino(rets, annual_factor)
        max_dd = float(daily['drawdown'].min())
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
        ulcer = float(np.sqrt(np.mean(np.square(daily['drawdown'].to_numpy(dtype=float))))) if len(daily) else 0.0
        win_rate = float((rets > 0).mean()) if len(rets) else 0.0
        avg_win = float(np.mean(rets[rets > 0])) if np.any(rets > 0) else 0.0
        avg_loss = float(np.mean(rets[rets < 0])) if np.any(rets < 0) else 0.0
        win_loss = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-12 else 0.0
        gain_pain = abs(rets[rets > 0].sum()) / abs(rets[rets < 0].sum()) if abs(rets[rets < 0].sum()) > 1e-12 else 0.0
        var95 = float(np.percentile(rets, 5)) if len(rets) else 0.0
        cvar95 = float(rets[rets <= var95].mean()) if np.any(rets <= var95) else 0.0
        skew = float(stats.skew(rets)) if len(rets) > 2 else 0.0
        kurt = float(stats.kurtosis(rets)) if len(rets) > 3 else 0.0

        total_trades = int((trades['weight'].abs() > 1e-12).sum())
        active_rows = total_trades
        active_dates = int(trades['date'].nunique())
        long_rows = int((trades['weight'] > 0).sum())
        short_rows = int((trades['weight'] < 0).sum())
        avg_gross = float(trades.groupby('date')['gross_weight'].sum().mean()) if 'gross_weight' in trades.columns else 0.0
        avg_net = float(trades.groupby('date')['weight'].sum().mean()) if 'weight' in trades.columns else 0.0

        bench_info = {}
        if bench is not None and not bench.empty:
            bench = bench.reindex(pd.to_datetime(daily['date'])).ffill().dropna()
            if not bench.empty:
                bench_eq = bench * (initial / float(bench.iloc[0]))
                bench_ret = bench.pct_change().fillna(0.0).to_numpy(dtype=float)
                bench_total = float(bench_eq.iloc[-1] / bench_eq.iloc[0] - 1.0)
                bench_years = max((bench_eq.index[-1] - bench_eq.index[0]).days / 365.25, 1e-9)
                bench_cagr = (float(bench_eq.iloc[-1]) / float(bench_eq.iloc[0])) ** (1 / bench_years) - 1.0
                bench_dd = float((bench_eq / bench_eq.cummax() - 1.0).min())
                bench_info = {'bench_total': bench_total, 'bench_cagr': bench_cagr, 'bench_dd': bench_dd}
            else:
                bench_info = {'bench_total': np.nan, 'bench_cagr': np.nan, 'bench_dd': np.nan}
        else:
            bench_info = {'bench_total': np.nan, 'bench_cagr': np.nan, 'bench_dd': np.nan}

        state_perf = trades[trades['weight'].abs() > 1e-12].copy()
        if not state_perf.empty:
            state_perf['state_ret'] = state_perf['net_pnl'] / float(self.capital)
            regime_table = state_perf.groupby('market_state').agg(
                trades=('net_pnl', 'count'),
                pnl=('net_pnl', 'sum'),
                gross=('gross_weight', 'mean'),
                net=('weight', 'mean'),
                avg_conf=('regime_confidence', 'mean'),
                avg_momz=('mom_z', 'mean'),
            ).sort_values('pnl', ascending=False)
        else:
            regime_table = pd.DataFrame()

        side_table = trades[trades['weight'].abs() > 1e-12].groupby('side').agg(
            trades=('net_pnl', 'count'),
            pnl=('net_pnl', 'sum'),
            avg_w=('weight', 'mean'),
        ) if not trades.empty else pd.DataFrame()

        transition_table = _make_transition_table(trades.groupby('date')['market_state'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]))
        split = _split_pre_post(daily, break_date)

        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write('CHIMERA STATIC BACKTEST REPORT\n')
            f.write('=' * 56 + '\n')
            f.write(f'Date range         : {daily["date"].iloc[0].date()} -> {daily["date"].iloc[-1].date()}\n')
            f.write(f'Benchmark          : {self.benchmark_ticker}\n')
            f.write('\nPERFORMANCE\n')
            f.write(f'  Initial capital  : ₹{initial:,.2f}\n')
            f.write(f'  Final equity     : ₹{final:,.2f}\n')
            f.write(f'  Total return     : {cum_ret * 100:.2f}%\n')
            f.write(f'  CAGR             : {cagr * 100:.2f}%\n')
            f.write('\nRISK\n')
            f.write(f'  Volatility (ann.): {vol_ann * 100:.2f}%\n')
            f.write(f'  Sharpe ratio     : {sharpe:.2f}\n')
            f.write(f'  Sortino ratio    : {sortino:.2f}\n')
            f.write(f'  Max drawdown     : {max_dd * 100:.2f}%\n')
            f.write(f'  Calmar ratio     : {calmar:.2f}\n')
            f.write(f'  Ulcer index      : {ulcer:.4f}\n')
            f.write(f'  VaR 95%          : {var95 * 100:.2f}%\n')
            f.write(f'  CVaR 95%         : {cvar95 * 100:.2f}%\n')
            f.write(f'  Skew / Kurtosis  : {skew:.2f} / {kurt:.2f}\n')
            f.write('\nACTIVITY\n')
            f.write(f'  Trade rows       : {total_trades}\n')
            f.write(f'  Active rows      : {active_rows}\n')
            f.write(f'  Active dates     : {active_dates}\n')
            f.write(f'  Long rows        : {long_rows}\n')
            f.write(f'  Short rows       : {short_rows}\n')
            f.write(f'  Avg gross expo   : {avg_gross:.3f}\n')
            f.write(f'  Avg net expo     : {avg_net:.3f}\n')
            f.write(f'  Sum net PnL      : ₹{trades["net_pnl"].sum():,.0f}\n')
            f.write('\nBENCHMARK COMPARISON\n')
            f.write(f'  Benchmark return : {bench_info["bench_total"] * 100:.2f}%\n' if pd.notna(bench_info['bench_total']) else '  Benchmark return : N/A\n')
            f.write(f'  Benchmark CAGR   : {bench_info["bench_cagr"] * 100:.2f}%\n' if pd.notna(bench_info['bench_cagr']) else '  Benchmark CAGR   : N/A\n')
            f.write(f'  Benchmark max DD : {bench_info["bench_dd"] * 100:.2f}%\n' if pd.notna(bench_info['bench_dd']) else '  Benchmark max DD : N/A\n')
            if pd.notna(bench_info['bench_total']):
                f.write(f'  Excess return    : {(cum_ret - bench_info["bench_total"]) * 100:.2f}%\n')
            f.write('\nREGIME BREAKDOWN\n')
            if not regime_table.empty:
                for state, row in regime_table.iterrows():
                    f.write(f'  {state:<8} trades={int(row.trades):<4} pnl=₹{row.pnl:,.0f} avg_gross={row.gross:.3f} avg_conf={row.avg_conf:.2f}\n')
            else:
                f.write('  No regime data\n')
            f.write('\nSIDE BREAKDOWN\n')
            if not side_table.empty:
                for side, row in side_table.iterrows():
                    f.write(f'  {side:<5} trades={int(row.trades):<4} pnl=₹{row.pnl:,.0f} avg_w={row.avg_w:.3f}\n')
            f.write('\nTRANSITION MATRIX\n')
            if not transition_table.empty:
                f.write(transition_table.to_string() + '\n')
            else:
                f.write('  N/A\n')
            f.write('\nBREAK DIAGNOSTICS\n')
            if break_date is not None:
                f.write(f'  Candidate break date : {break_date.date()}\n')
                f.write(f'  Break score          : {break_score:.4f}\n')
            else:
                f.write('  Candidate break date : N/A\n')
            if split:
                f.write('\nPRE / POST BREAK SPLIT\n')
                for name, vals in split.items():
                    f.write(f"  {name.upper():<4} ret={vals['total_return'] * 100:.2f}% sharpe={vals['sharpe']:.2f} sortino={vals['sortino']:.2f} win={vals['win_rate'] * 100:.1f}% vol={vals['volatility'] * 100:.2f}% dd={vals['max_dd'] * 100:.2f}%\n")
            f.write('\nTOP ACTIVE WINNERS\n')
            active = trades[trades['weight'].abs() > 1e-12].copy().sort_values('net_pnl', ascending=False).head(10)
            for _, r in active.iterrows():
                f.write(f"  {pd.to_datetime(r['date']).date()}  {r['ticker']:<12} ₹{r['net_pnl']:,.0f}  w={r['weight']:.3f}  state={r.get('market_state', 'N/A')}\n")
            f.write('\nTOP ACTIVE LOSERS\n')
            active = trades[trades['weight'].abs() > 1e-12].copy().sort_values('net_pnl', ascending=True).head(10)
            for _, r in active.iterrows():
                f.write(f"  {pd.to_datetime(r['date']).date()}  {r['ticker']:<12} ₹{r['net_pnl']:,.0f}  w={r['weight']:.3f}  state={r.get('market_state', 'N/A')}\n")

    def _make_equity_plot(self, daily: pd.DataFrame, bench: Optional[pd.Series], regime_trace: Optional[pd.DataFrame], break_date: Optional[pd.Timestamp], filename: str):
        fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        x = pd.to_datetime(daily['date'])

        # Shading
        if regime_trace is not None and not regime_trace.empty:
            reg = regime_trace.copy()
            reg['date'] = pd.to_datetime(reg['date'])
            reg = reg.sort_values('date').dropna(subset=['regime'])
            starts = reg['date'].tolist()
            states = reg['regime'].astype(str).tolist()
            for i, start in enumerate(starts):
                end = starts[i + 1] if i + 1 < len(starts) else x.iloc[-1]
                color = REGIME_COLORS.get(states[i], '#999999')
                ax.axvspan(start, end, color=color, alpha=0.08)

        ax.plot(x, daily['equity'], label='Chimera Strategy', linewidth=2.2, color='#1f77b4')
        if bench is not None and not bench.empty:
            bench = bench.reindex(x).ffill().dropna()
            if not bench.empty:
                bench_eq = bench * (self.capital / float(bench.iloc[0]))
                ax.plot(bench_eq.index, bench_eq.values, label=f'Benchmark ({self.benchmark_ticker})', linestyle='--', linewidth=1.8, color='#d62728')
        ax.set_yscale('log')
        ax.set_title('Chimera Equity Curve & Benchmark (Log Scale with Regime Shading)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (₹)')
        ax.grid(alpha=0.25)

        if break_date is not None:
            ax.axvline(break_date, color='purple', linestyle=':', linewidth=2.2)
            ax.text(break_date, ax.get_ylim()[1] * 0.9, ' Structural Break', color='purple', fontsize=10, va='top')

        regime_handles = [Patch(facecolor=c, alpha=0.25, label=f'{k} Regime') for k, c in REGIME_COLORS.items()]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + regime_handles, labels=ax.get_legend_handles_labels()[1] + [p.get_label() for p in regime_handles], loc='upper left', fontsize=9.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.savefig(filename, dpi=180, bbox_inches='tight')
        plt.close(fig)

    def _make_rolling_plot(self, daily: pd.DataFrame, bench: Optional[pd.Series], filename: str):
        fig, ax = plt.subplots(figsize=(12, 6.0), constrained_layout=True)
        x = pd.to_datetime(daily['date'])

        # Calculate Rolling Sharpe & Beta
        if bench is not None and not bench.empty:
            bench = bench.reindex(x).ffill().dropna()
            if not bench.empty:
                bench_ret = bench.pct_change().fillna(0.0)
                strat_ret = daily.set_index(pd.to_datetime(daily['date']))['portfolio_return'].reindex(bench_ret.index).fillna(0.0)
                rolling = 26 if len(daily) > 60 else max(5, len(daily) // 4)
                cov_sb = strat_ret.rolling(rolling, min_periods=max(6, rolling // 2)).cov(bench_ret)
                var_b = bench_ret.rolling(rolling, min_periods=max(6, rolling // 2)).var()
                beta = (cov_sb / var_b.replace(0, np.nan)).dropna()
                roll_sharpe = (
                    strat_ret.rolling(rolling, min_periods=max(6, rolling // 2)).mean()
                    / strat_ret.rolling(rolling, min_periods=max(6, rolling // 2)).std().replace(0, np.nan)
                    * np.sqrt(52)
                ).dropna()

                ax.plot(beta.index, beta.values, label='Rolling Beta (NSE/Benchmark Sensitivity)', linewidth=1.8, color='#1f77b4')
                ax.plot(roll_sharpe.index, roll_sharpe.values, label='Rolling Sharpe Ratio (Annualised)', linewidth=1.8, color='#ff7f0e')
                ax.axhline(0, color='black', linewidth=1.0, alpha=0.6)
                ax.axhline(1, color='grey', linewidth=0.8, alpha=0.4, linestyle='--')
                ax.set_title('Rolling Portfolio Metrics (Sensitivity & Efficiency)', fontsize=13, fontweight='bold')
                ax.set_xlabel('Date')
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(alpha=0.25)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.savefig(filename, dpi=180, bbox_inches='tight')
        plt.close(fig)

    def _make_drawdown_plot(self, daily: pd.DataFrame, bench: Optional[pd.Series], filename: str):
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        x = pd.to_datetime(daily['date'])

        ax.fill_between(x, daily['drawdown'] * 100, 0, color='#d62728', alpha=0.3, label='Strategy Drawdown')
        ax.plot(x, daily['drawdown'] * 100, color='#d62728', linewidth=1.5)

        if bench is not None and not bench.empty:
            bench = bench.reindex(x).ffill().dropna()
            if not bench.empty:
                bench_eq = bench * (self.capital / float(bench.iloc[0]))
                bench_dd = (bench_eq / bench_eq.cummax() - 1.0) * 100
                ax.plot(bench_dd.index, bench_dd.values, linestyle='--', color='#7f7f7f', label=f'Benchmark ({self.benchmark_ticker}) Drawdown', linewidth=1.5)

        ax.axhline(0, color='black', linewidth=1.0, alpha=0.6)
        ax.set_title('Strategy Peak-to-Trough Drawdown Analysis', fontsize=13, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.savefig(filename, dpi=180, bbox_inches='tight')
        plt.close(fig)

    def _make_regime_plot(self, daily: pd.DataFrame, filename: str):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
        x = pd.to_datetime(daily['date'])

        # Panel 1: Probabilities
        if 'p_bull' in daily.columns:
            ax1.plot(x, daily['p_bull'], label='P(Bull) - Up Trend Strength', color='#2ca02c', linewidth=2.0)
        if 'p_bear' in daily.columns:
            ax1.plot(x, daily['p_bear'], label='P(Bear) - Down Trend Strength', color='#d62728', linewidth=2.0)
        if 'p_chop' in daily.columns:
            ax1.plot(x, daily['p_chop'], label='P(Chop) - Whipsaw/Range', color='#ff7f0e', linewidth=1.5, linestyle=':')
        ax1.set_title('Panel A: Underlying Regime Probability States', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Probability')
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper left', fontsize=9.5)
        ax1.grid(alpha=0.25)

        # Panel 2: Confidence and Transition Risk
        ax2.plot(x, daily['regime_confidence'], label='Regime Decision Confidence', color='#1f77b4', linewidth=2.0)
        if 'transition_risk' in daily.columns:
            ax2.plot(x, daily['transition_risk'], label='Transition Risk (Vulnerability)', color='#9467bd', linewidth=1.8, linestyle='--')
        ax2.set_title('Panel B: State Stability & Transition Vulnerability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper left', fontsize=9.5)
        ax2.grid(alpha=0.25)

        # Panel 3: Technical / Sentiment Drivers
        if 'breadth' in daily.columns:
            ax3.plot(x, daily['breadth'], label='Market Breadth (Volume Rank)', color='#000080', linewidth=1.5)
        if 'macro_score' in daily.columns:
            ax3.plot(x, daily['macro_score'], label='Macro Driver Score', color='#e377c2', linewidth=1.5)
        if 'news_bias' in daily.columns:
            ax3.plot(x, daily['news_bias'], label='Lightweight News Sentiment Bias', color='#17becf', linewidth=1.5)
        ax3.set_title('Panel C: Technical, Macro, & News Sentiment Drivers', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Normalized Score')
        ax3.legend(loc='upper left', fontsize=9.5)
        ax3.grid(alpha=0.25)

        # Format dates on last axis
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.suptitle('Chimera Regime Analysis Dashboard', fontsize=15, fontweight='bold', y=1.02)
        fig.savefig(filename, dpi=180, bbox_inches='tight')
        plt.close(fig)

    def generate(self, csv_path: str, output_prefix: str, open_browser: bool = False) -> ReportOutputs:
        trades = self._load_trade_log(csv_path)
        daily = self._daily_summary(trades)
        regime_path = os.path.join(os.path.dirname(csv_path), 'regime_trace_chimera_fip.csv')
        regime_trace = pd.read_csv(regime_path) if os.path.exists(regime_path) else None
        bench = self._benchmark_series(daily['date'])

        # break date based on strategy minus benchmark returns.
        if bench is not None and not bench.empty:
            bench = bench.reindex(pd.to_datetime(daily['date'])).ffill()
            strat_ret = daily['portfolio_return'].reindex(bench.index).fillna(0.0)
            break_date, break_score = _detect_break_date(strat_ret, bench.pct_change().fillna(0.0), pd.Series(bench.index))
        else:
            break_date, break_score = None, 0.0

        out_png = f'{output_prefix}.png'
        out_txt = f'{output_prefix}.txt'
        out_eq = f'{output_prefix}_equity.png'
        out_roll = f'{output_prefix}_rolling_metrics.png'
        out_dd = f'{output_prefix}_drawdown.png'
        out_reg = f'{output_prefix}_regime_analysis.png'

        self._make_plot(daily, bench, regime_trace, break_date, out_png)
        self._write_txt(daily, trades, bench, regime_trace, out_txt, break_date, break_score)

        # Generate separate clean reports
        self._make_equity_plot(daily, bench, regime_trace, break_date, out_eq)
        self._make_rolling_plot(daily, bench, out_roll)
        self._make_drawdown_plot(daily, bench, out_dd)
        self._make_regime_plot(daily, out_reg)

        return ReportOutputs(
            png_path=out_png,
            txt_path=out_txt,
            equity_path=out_eq,
            rolling_path=out_roll,
            drawdown_path=out_dd,
            regime_path=out_reg
        )
