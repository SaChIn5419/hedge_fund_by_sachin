import pandas as pd
import numpy as np
import xgboost as xgb
from stable_baselines3 import PPO
from engine.signal import ChimeraEngineNormal, AssetSnapshot
from typing import List

class ChimeraEngineML(ChimeraEngineNormal):
    def __init__(self, model_path='engine/ml/models/xgb_baseline.json', rl_path='engine/ml/models/ppo_agent.zip'):
        super().__init__()
        # 1. Load XGBoost Stock Selection Model
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.features = ['fip_z', 'mom20_z', 'mom60_z', 'vol20_z', 'beta', 'rsi14', 'structure_score']
        
        # 2. Load DRL Portfolio Sizing Agent
        self.rl_agent = PPO.load(rl_path)
        self.rl_features = [
            'regime_confidence', 'p_bull', 'p_chop', 'p_bear', 
            'transition_risk', 'breadth', 'vix', 'macro_score',
            'news_bias', 'suppression_score'
        ]
        
        print(f"ML Engine initialized. Loaded XGB from {model_path} and PPO from {rl_path}")

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
        
        # Ask the RL Agent for the long_gross exposure
        action, _states = self.rl_agent.predict(obs, deterministic=True)
        long_gross = float(np.clip(action[0], 0.0, 1.0))
        
        short_gross = 0.0 # Shorts are permanently disabled
        
        return long_gross, short_gross
