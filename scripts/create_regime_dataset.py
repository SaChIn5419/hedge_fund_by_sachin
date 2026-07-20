"""
Create Regime Training Dataset
Generates ground-truth regime labels and rich features for training an ML-based regime classifier.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from engine.signal import ChimeraEngineNormal, CONFIG, _resolve_key

def build_regime_dataset():
    engine = ChimeraEngineNormal()
    
    print("Loading data...")
    data_map = engine.load_all_stocks()
    data_map = engine.load_indices(data_map)
    
    broad_key = _resolve_key(data_map, CONFIG['BROAD_TICKER_CANDIDATES'])
    risk_key = _resolve_key(data_map, CONFIG['RISK_TICKER_CANDIDATES'])
    fear_key = _resolve_key(data_map, CONFIG['FEAR_TICKER_CANDIDATES'])
    usd_key = _resolve_key(data_map, CONFIG['USDINR_CANDIDATES'])
    us10y_key = _resolve_key(data_map, CONFIG['US10Y_CANDIDATES'])
    gold_key = _resolve_key(data_map, CONFIG['GOLD_CANDIDATES'])
    
    calendar = data_map[broad_key].copy().sort_index()
    dates = calendar.index
    
    broad = calendar.copy()
    broad['Close'] = broad['Close'].astype(float)
    broad['ret'] = broad['Close'].pct_change()
    
    # ==========================================
    # FEATURE ENGINEERING (all backward-looking)
    # ==========================================
    features = pd.DataFrame(index=dates)
    
    # 1. Price vs Moving Averages (continuous, not binary)
    sma50 = broad['Close'].rolling(50, min_periods=25).mean()
    sma100 = broad['Close'].rolling(100, min_periods=50).mean()
    sma200 = broad['Close'].rolling(200, min_periods=100).mean()
    
    features['dist_sma50'] = (broad['Close'] - sma50) / sma50  # % distance from SMA50
    features['dist_sma100'] = (broad['Close'] - sma100) / sma100
    features['dist_sma200'] = (broad['Close'] - sma200) / sma200
    
    # 2. Trend Quality
    features['eff_ratio_20'] = _efficiency_ratio(broad['Close'], 20)
    features['eff_ratio_50'] = _efficiency_ratio(broad['Close'], 50)
    features['eff_ratio_100'] = _efficiency_ratio(broad['Close'], 100)
    
    # 3. Momentum at multiple timeframes
    features['mom5'] = broad['Close'].pct_change(5)
    features['mom10'] = broad['Close'].pct_change(10)
    features['mom20'] = broad['Close'].pct_change(20)
    features['mom60'] = broad['Close'].pct_change(60)
    
    # 4. Volatility regime
    vol20 = broad['ret'].rolling(20, min_periods=10).std() * np.sqrt(252)
    vol60 = broad['ret'].rolling(60, min_periods=30).std() * np.sqrt(252)
    features['vol20'] = vol20
    features['vol60'] = vol60
    features['vol_ratio'] = vol20 / vol60.replace(0, np.nan)  # Vol regime shift
    
    # 5. Rolling Sharpe (risk-adjusted momentum)
    features['sharpe_20'] = (broad['ret'].rolling(20, min_periods=10).mean() * 252) / (vol20.replace(0, np.nan))
    features['sharpe_60'] = (broad['ret'].rolling(60, min_periods=30).mean() * 252) / (vol60.replace(0, np.nan))
    
    # 6. Breadth (% of stocks above SMA200)
    asset_cache = engine._build_asset_cache(data_map, dates)
    breadth_series = pd.Series(index=dates, dtype=float)
    for idx in range(len(dates)):
        vals = [a.above200[idx] for a in asset_cache if idx < len(a.above200) and np.isfinite(a.above200[idx])]
        breadth_series.iloc[idx] = np.nanmean(vals) if vals else np.nan
    
    features['breadth'] = breadth_series
    features['breadth_mom20'] = breadth_series.diff(20)  # Breadth momentum
    
    # 7. VIX features
    if fear_key:
        vix = data_map[fear_key]['Close'].reindex(dates).astype(float)
        features['vix'] = vix
        features['vix_sma20'] = vix.rolling(20, min_periods=10).mean()
        features['vix_z'] = (vix - vix.rolling(60, min_periods=30).mean()) / vix.rolling(60, min_periods=30).std().replace(0, np.nan)
    
    # 8. Risk asset (BankNifty vs Nifty relative strength)
    if risk_key:
        risk_close = data_map[risk_key]['Close'].reindex(dates).astype(float)
        features['risk_rel_strength'] = (risk_close.pct_change(20)) - features['mom20']
    
    # 9. Macro features
    if usd_key:
        usd_close = data_map[usd_key]['Close'].reindex(dates).astype(float)
        features['usd_mom20'] = usd_close.pct_change(20)
    if us10y_key:
        us10y_close = data_map[us10y_key]['Close'].reindex(dates).astype(float)
        features['us10y_mom20'] = us10y_close.pct_change(20)
    if gold_key:
        gold_close = data_map[gold_key]['Close'].reindex(dates).astype(float)
        features['gold_mom20'] = gold_close.pct_change(20)
    
    # 10. Cross-sectional dispersion (how spread out are stock returns?)
    dispersion_series = pd.Series(index=dates, dtype=float)
    for idx in range(len(dates)):
        rets = [a.mom20[idx] for a in asset_cache if idx < len(a.mom20) and np.isfinite(a.mom20[idx])]
        dispersion_series.iloc[idx] = np.std(rets) if len(rets) > 10 else np.nan
    features['cross_dispersion'] = dispersion_series
    
    # 11. Drawdown from peak
    rolling_max = broad['Close'].cummax()
    features['drawdown'] = (broad['Close'] / rolling_max) - 1.0
    
    # 12. Mean reversion signal
    features['rsi14_market'] = _rsi_series(broad['Close'], 14)
    
    # ==========================================
    # GROUND TRUTH REGIME LABELS
    # ==========================================
    # Use FORWARD 5-day market return to define regime
    # This is the TARGET - it uses future data (which is correct for supervised training)
    fwd_ret_5 = broad['Close'].pct_change(5).shift(-5)  # Forward 5-day return
    fwd_vol_5 = broad['ret'].rolling(5).std().shift(-5) * np.sqrt(252)
    
    # Regime labels based on forward return + volatility context
    # BULL: market goes up meaningfully (> +0.75%)
    # BEAR: market goes down meaningfully (< -0.75%)
    # CHOP: market goes sideways or is uncertain
    regime_label = pd.Series('CHOP', index=dates)
    regime_label[fwd_ret_5 > 0.0075] = 'BULL'
    regime_label[fwd_ret_5 < -0.0075] = 'BEAR'
    
    features['regime_label'] = regime_label
    features['fwd_ret_5'] = fwd_ret_5
    
    # Only keep weekly (Friday) samples to match rebalance schedule
    rebalance_dates = dates[dates.weekday == 4]
    weekly_features = features.loc[features.index.isin(rebalance_dates)].copy()
    
    # Drop rows with too many NaNs
    weekly_features = weekly_features.dropna(subset=['dist_sma200', 'vol20', 'breadth', 'regime_label'])
    weekly_features = weekly_features[weekly_features['regime_label'] != 'CHOP']  # Keep for now, will re-add
    # Actually keep all labels
    weekly_features = features.loc[features.index.isin(rebalance_dates)].copy()
    weekly_features = weekly_features.dropna(subset=['dist_sma200', 'vol20', 'breadth', 'regime_label', 'fwd_ret_5'])
    
    out_path = 'data/regime_dataset.parquet'
    weekly_features.to_parquet(out_path)
    
    print(f"\nSaved regime dataset: {len(weekly_features)} samples to {out_path}")
    print(f"Label distribution:")
    print(weekly_features['regime_label'].value_counts())
    print(f"\nFeature columns ({len(weekly_features.columns)}):")
    print(weekly_features.columns.tolist())


def _efficiency_ratio(prices, lookback):
    """Kaufman Efficiency Ratio: net price change / sum of absolute changes"""
    net_change = (prices - prices.shift(lookback)).abs()
    sum_changes = prices.diff().abs().rolling(lookback, min_periods=lookback//2).sum()
    return net_change / sum_changes.replace(0, np.nan)


def _rsi_series(prices, period=14):
    """RSI for a price series"""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


if __name__ == '__main__':
    build_regime_dataset()
