import os
import json
import pandas as pd
import numpy as np

# Resolve base project directory dynamically to avoid CWD issues
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_trade_log(filepath="data/tradelog_chimera_fip.csv"):
    filepath = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        if 'signal_date' in df.columns:
            df['signal_date'] = pd.to_datetime(df['signal_date'])
        if 'exit_date' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_date'])
        for col in ['weight', 'gross_weight', 'fwd_return', 'net_pnl', 'leverage_mult', 'regime_confidence', 'regime_score', 'breadth', 'macro_score', 'news_bias', 'suppression_score', 'p_bull', 'p_chop', 'p_bear', 'transition_risk', 'mom_z', 'break_score', 'score', 'long_score', 'short_score', 'nifty_vol', 'efficiency']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_weekly_returns(filepath="data/weekly_returns_chimera_fip.csv"):
    filepath = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def load_regime_trace(filepath="data/regime_trace_chimera_fip.csv"):
    filepath = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def load_geometry_log(filepath="data/geometry_log.csv"):
    filepath = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def load_evidence_registry(filepath="data/evidence_registry.json"):
    filepath = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_daily_summary(trades, capital=1_000_000):
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
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
        breadth=('breadth', 'mean'),
        macro_score=('macro_score', 'mean'),
    ).reset_index()

    weekly_path = os.path.join(BASE_DIR, "data/weekly_returns_chimera_fip.csv")
    if os.path.exists(weekly_path):
        weekly = pd.read_csv(weekly_path)
        weekly['date'] = pd.to_datetime(weekly['date'])
        
        weekly['pnl_rs'] = weekly['portfolio_return'] * float(capital)
        weekly['equity_val'] = float(capital) + weekly['pnl_rs'].cumsum()
        weekly['true_return'] = weekly['pnl_rs'] / weekly['equity_val'].shift(1).fillna(float(capital))
        
        daily = daily.merge(weekly[['date', 'true_return', 'equity_val']], on='date', how='left')
        daily['portfolio_return'] = daily['true_return']
        daily['equity'] = daily['equity_val']
        daily.drop(columns=['true_return', 'equity_val'], errors='ignore', inplace=True)
    else:
        daily['equity'] = float(capital) + daily['net_pnl'].cumsum()
        daily['portfolio_return'] = daily['net_pnl'] / daily['equity'].shift(1).fillna(float(capital))
    daily['peak'] = daily['equity'].cummax()
    daily['drawdown'] = daily['equity'] / daily['peak'] - 1.0
    daily['market_state'] = grouped['market_state'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]).values

    return daily

def get_kpis(daily, capital=1_000_000):
    if daily.empty:
        return {}
    rets = daily['portfolio_return'].to_numpy(dtype=float)
    initial = float(capital)
    final = float(daily['equity'].iloc[-1])
    cum_ret = final / initial - 1.0

    years = max((pd.to_datetime(daily['date'].iloc[-1]) - pd.to_datetime(daily['date'].iloc[0])).days / 365.25, 1e-9)
    cagr = (final / initial) ** (1 / years) - 1.0

    annual_factor = 52
    sd = np.std(rets, ddof=1)
    sharpe = float((np.mean(rets) * annual_factor) / (sd * np.sqrt(annual_factor))) if sd > 1e-12 else 0.0

    downside = rets[rets < 0]
    sd_down = np.std(downside, ddof=1) if len(downside) > 0 else 0.0
    sortino = float((np.mean(rets) * annual_factor) / (sd_down * np.sqrt(annual_factor))) if sd_down > 1e-12 else 0.0

    max_dd = float(daily['drawdown'].min())

    return {
        'Total Return': f"{cum_ret * 100:.2f}%",
        'CAGR': f"{cagr * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Max Drawdown': f"{max_dd * 100:.2f}%"
    }

def get_current_holdings(trades, current_date=None):
    if trades.empty:
        return pd.DataFrame()
    if current_date is None:
        current_date = trades['date'].max()

    current_trades = trades[(trades['date'] == current_date) & (trades['weight'].abs() > 1e-12)].copy()
    current_trades['Entry Date'] = current_trades['signal_date'].dt.strftime('%Y-%m-%d')
    return current_trades[['ticker', 'side', 'weight', 'Entry Date', 'close', 'fwd_return', 'net_pnl', 'market_state', 'score']]
