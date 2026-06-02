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
    broad_key = engine.broad_ticker if hasattr(engine, 'broad_ticker') else 'nifty50'
    calendar = data_map[broad_key].copy().sort_index()
    dates = calendar.index
    rebalance_dates = dates[dates.weekday == 4]
    
    asset_cache = engine._build_asset_cache(data_map, dates)
    
    all_features = []
    
    print("Extracting features and targets...")
    for i in tqdm(range(len(rebalance_dates) - 1)):
        d = rebalance_dates[i]
        next_d = rebalance_dates[i+1]
        
        signal_idx = np.searchsorted(dates, d) - 1
        entry_idx = np.searchsorted(dates, d)
        exit_idx = np.searchsorted(dates, next_d)
        
        # We need the macro context just to pass it along (even though we don't use it directly here)
        # Actually, let's just grab the dataframe directly using _score_universe
        df_features = engine._score_universe(asset_cache, signal_idx)
        if df_features.empty:
            continue
            
        df_features['date'] = d
        
        # Calculate forward return for each ticker
        fwd_returns = []
        for _, row in df_features.iterrows():
            ticker = row['ticker']
            # Find the asset in cache
            asset = next((a for a in asset_cache if a.ticker == ticker), None)
            if asset:
                entry_open = asset.open[entry_idx]
                exit_close = asset.close[exit_idx - 1] # Close before the next open
                if np.isfinite(entry_open) and np.isfinite(exit_close) and entry_open > 0:
                    ret = (exit_close / entry_open) - 1.0
                    fwd_returns.append(ret)
                else:
                    fwd_returns.append(np.nan)
            else:
                fwd_returns.append(np.nan)
                
        df_features['target_fwd_ret'] = fwd_returns
        
        # Drop rows with NaN returns
        df_features = df_features.dropna(subset=['target_fwd_ret'])
        
        all_features.append(df_features)
        
    final_df = pd.concat(all_features, ignore_index=True)
    
    out_path = os.path.join('data', 'ml_dataset.parquet')
    final_df.to_parquet(out_path, index=False)
    print(f"Saved dataset with {len(final_df)} samples to {out_path}")

if __name__ == "__main__":
    build_dataset()
