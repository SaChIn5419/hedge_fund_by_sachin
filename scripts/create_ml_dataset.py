import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from engine.signal import ChimeraEngineNormal

def build_dataset():
    engine = ChimeraEngineNormal()
    
    print("Loading data...")
    data_map = engine.load_all_stocks()
    data_map = engine.load_indices(data_map)
    
    broad_key = None
    for k in ['nifty50', '^NSEI', 'nifty_50']:
        if k in data_map:
            broad_key = k
            break
    if broad_key is None:
        raise ValueError("Missing Nifty reference series in data_map")
        
    calendar = data_map[broad_key].copy().sort_index()
    dates = calendar.index
    rebalance_dates = dates[dates.weekday == 4]
    
    nifty_series_open = data_map[broad_key]['Open'].astype(float)
    nifty_series_close = data_map[broad_key]['Close'].astype(float)
    
    # Load sector map
    import json
    sector_map = {}
    if os.path.exists('chimera_data/sector_map.json'):
        with open('chimera_data/sector_map.json', 'r') as f:
            sector_map = json.load(f)
            
    # Precalculate daily returns matrices for rolling factor regressions
    daily_returns_df = pd.DataFrame({
        tk: data_map[tk]['Close'].pct_change().reindex(dates)
        for tk in data_map if 'Close' in data_map[tk].columns
    })
    nifty_daily_ret = nifty_series_close.pct_change().reindex(dates)
    
    daily_sector_returns = {}
    sectors = set(sector_map.values())
    for s in sectors:
        if s == 'Unknown':
            continue
        tickers_in_sector = [tk for tk in sector_map if sector_map[tk] == s]
        if tickers_in_sector:
            daily_sector_returns[s] = daily_returns_df[tickers_in_sector].mean(axis=1)
    daily_sector_returns_df = pd.DataFrame(daily_sector_returns)
    daily_sector_returns_df['Unknown'] = daily_returns_df.mean(axis=1)
    
    def fit_rolling_betas(y_series, mkt_series, sec_series):
        mask = np.isfinite(y_series) & np.isfinite(mkt_series) & np.isfinite(sec_series)
        if mask.sum() < 100:
            return 1.0, 1.0
        y_clean = y_series[mask]
        X_clean = np.column_stack([np.ones(mask.sum()), mkt_series[mask], sec_series[mask]])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
            return float(beta[1]), float(beta[2])
        except:
            return 1.0, 1.0

    asset_cache = engine._build_asset_cache(data_map, dates)
    
    all_features = []
    
    print("Extracting features and targets...")
    for i in tqdm(range(len(rebalance_dates) - 1)):
        d = rebalance_dates[i]
        next_d = rebalance_dates[i+1]
        
        signal_idx = np.searchsorted(dates, d) - 1
        entry_idx = np.searchsorted(dates, d)
        exit_idx = np.searchsorted(dates, next_d)
        
        df_features = engine._score_universe(asset_cache, signal_idx)
        if df_features.empty:
            continue
            
        df_features['date'] = d
        
        # Get nifty return
        nifty_open = nifty_series_open.iloc[entry_idx] if entry_idx < len(nifty_series_open) else np.nan
        nifty_close = nifty_series_close.iloc[exit_idx - 1] if (exit_idx - 1) < len(nifty_series_close) else np.nan
        nifty_ret = (nifty_close / nifty_open) - 1.0 if np.isfinite(nifty_open) and nifty_open > 0 else np.nan
        
        # Calculate forward return for each ticker
        fwd_returns = []
        for _, row in df_features.iterrows():
            ticker = row['ticker']
            asset = next((a for a in asset_cache if a.ticker == ticker), None)
            if asset:
                entry_open = asset.open[entry_idx]
                exit_close = asset.close[exit_idx - 1] # Close before next open
                if np.isfinite(entry_open) and np.isfinite(exit_close) and entry_open > 0:
                    ret = (exit_close / entry_open) - 1.0
                    fwd_returns.append(ret)
                else:
                    fwd_returns.append(np.nan)
            else:
                fwd_returns.append(np.nan)
                
        df_features['target_fwd_ret'] = fwd_returns
        df_features = df_features.dropna(subset=['target_fwd_ret'])
        if df_features.empty:
            continue
            
        # Target A: Market Excess Return
        df_features['target_excess_nifty'] = df_features['target_fwd_ret'] - (nifty_ret if np.isfinite(nifty_ret) else 0.0)
        
        # Target B: Sector Excess Return
        df_features['sector'] = df_features['ticker'].map(lambda tk: sector_map.get(tk, 'Unknown'))
        sector_returns = df_features.groupby('sector')['target_fwd_ret'].mean().to_dict()
        df_features['target_excess_sector'] = df_features['target_fwd_ret'] - df_features['sector'].map(sector_returns)
        
        # Target C: Cross-Sectional Percentile Rank
        df_features['target_cs_rank'] = df_features['target_fwd_ret'].rank(pct=True)
        
        # Target D: Residual Idiosyncratic Return
        fwd_sector_ret_dict = sector_returns
        slice_idx = slice(max(0, entry_idx - 252), entry_idx)
        mkt_daily = nifty_daily_ret.iloc[slice_idx].values
        
        residual_returns = []
        for _, row in df_features.iterrows():
            tk = row['ticker']
            s = row['sector']
            fwd_ret = row['target_fwd_ret']
            
            y_daily = daily_returns_df[tk].iloc[slice_idx].values if tk in daily_returns_df.columns else np.zeros(len(mkt_daily))
            sec_daily = daily_sector_returns_df[s].iloc[slice_idx].values if s in daily_sector_returns_df.columns else mkt_daily
            
            beta_mkt, beta_sector = fit_rolling_betas(y_daily, mkt_daily, sec_daily)
            
            f_sec_ret = fwd_sector_ret_dict.get(s, 0.0)
            res_ret = fwd_ret - beta_mkt * (nifty_ret if np.isfinite(nifty_ret) else 0.0) - beta_sector * f_sec_ret
            residual_returns.append(res_ret)
            
        df_features['target_residual'] = residual_returns
        
        all_features.append(df_features)
        
    final_df = pd.concat(all_features, ignore_index=True)
    
    out_path = os.path.join('data', 'ml_dataset.parquet')
    final_df.to_parquet(out_path, index=False)
    print(f"Saved dataset with {len(final_df)} samples to {out_path}")

if __name__ == "__main__":
    build_dataset()
