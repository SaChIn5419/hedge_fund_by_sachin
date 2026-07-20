import pandas as pd
import numpy as np
import xgboost as xgb
from stable_baselines3 import PPO
from engine.signal import ChimeraEngineNormal, AssetSnapshot
from typing import List

class ChimeraEngineML(ChimeraEngineNormal):
    def __init__(self, model_path='engine/ml/models/xgb_baseline.json', rl_path='engine/ml/models/ppo_agent.zip', features=None):
        super().__init__()
        # 1. Load XGBoost Stock Selection Model
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        # If no features provided, default to the production 9 features
        if features is None:
            self.features = ['fip_z', 'mom20_z', 'mom60_z', 'vol20_z', 'beta', 'rsi14', 'structure_score', 'rvol20', 'vol_comp']
        else:
            self.features = features
            
        self.rl_features = [
            'regime_confidence', 'p_bull', 'p_chop', 'p_bear', 
            'transition_risk', 'breadth', 'vix', 'macro_score',
            'news_bias', 'suppression_score'
        ]
            
        print(f"ML Engine initialized. Loaded XGB from {model_path} with {len(self.features)} features.")

    def _score_universe(self, asset_cache: List[AssetSnapshot], signal_idx: int) -> pd.DataFrame:
        # 1. Base Feature Extraction (Call parent method to get the raw dataframe)
        df = super()._score_universe(asset_cache, signal_idx)
        
        if df.empty:
            return df
            
        # 2. Predict using XGBoost Model
        # We need to fill NaNs since we dropped them in training, or XGBoost can handle them natively
        # But to match training exactly, let's keep it safe. XGBoost handles NaNs natively if left as np.nan
        X = df[self.features]
        
        # We use the prediction as our primary long_score
        preds = self.model.predict(X)
        df['ml_pred_return'] = preds
        
        # 3. Hybrid Scoring
        # We still want to apply common sense constraints (RSI penalty, moving average)
        # We'll use the ML prediction as the primary ranker
        
        df['long_score'] = df['ml_pred_return'].rank(pct=True) + df['rsi_penalty_long'] + df['mom5_penalty']
        
        # Shorts are disabled anyway based on CONFIG, but we can set short_score to inverted prediction
        df['short_score'] = (1.0 - df['ml_pred_return'].rank(pct=True)) + df['rsi_penalty_short']
        
        return df.sort_values('long_score', ascending=False).reset_index(drop=True)

    def _allocate_gross_budget(self, regime: str, num_longs: int, num_shorts: int, confidence: float) -> tuple[float, float]:
        # Fetch the macro state we saved right before this call
        state_dict = self.current_macro_state
        
        # Format the observation for the RL Agent
        obs = np.array([state_dict[f] for f in self.rl_features], dtype=np.float32)
        
        # Apply the exact same normalization used during training
        obs[6] = obs[6] / 100.0  # VIX normalization
        
        # Fill any NaNs to prevent agent crashing
        obs = np.nan_to_num(obs, nan=0.0)
        
        # Heuristic sizing replacing PPO agent
        p_bear = state_dict.get('p_bear', 0.0)
        long_gross = float(np.clip(1.0 - (p_bear * 0.5), 0.3, 1.0))
        
        short_gross = 0.0 # Shorts are permanently disabled
        
        return long_gross, short_gross


class RollingChimeraEngineML(ChimeraEngineML):
    def __init__(self, model_prefix, features, is_ranker=False):
        # Call grandparent constructor directly to bypass parent's default load
        super(ChimeraEngineML, self).__init__()
        
        import os
        fallback_path = f"engine/ml/models/xgb_{model_prefix}_fallback.json"
        if is_ranker:
            self.model = xgb.XGBRanker()
        else:
            self.model = xgb.XGBRegressor()
        self.model.load_model(fallback_path)
        
        self.features = features
        self.rl_features = [
            'regime_confidence', 'p_bull', 'p_chop', 'p_bear', 
            'transition_risk', 'breadth', 'vix', 'macro_score',
            'news_bias', 'suppression_score'
        ]
        self.model_prefix = model_prefix
        self.is_ranker = is_ranker
        self.current_ym = None
        
        print(f"Rolling ML Engine initialized for {model_prefix}. Loaded {fallback_path} with {len(self.features)} features.")
        
    def _score_universe(self, asset_cache, signal_idx):
        if not hasattr(self, 'dates') or self.dates is None:
            return super()._score_universe(asset_cache, signal_idx)
            
        current_date = self.dates[signal_idx]
        ym = f"{current_date.year}-{current_date.month:02d}"
        
        # Load the causal rolling model trained prior to this month
        if ym != self.current_ym:
            self.current_ym = ym
            model_path = f"engine/ml/models/xgb_{self.model_prefix}_{ym}.json"
            import os
            if os.path.exists(model_path):
                if self.is_ranker:
                    self.model = xgb.XGBRanker()
                else:
                    self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
                
        return super()._score_universe(asset_cache, signal_idx)
