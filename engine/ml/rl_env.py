import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ChimeraPortfolioEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The agent learns to size the portfolio based on macro state.
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, data_path='data/weekly_returns_chimera_fip_zero_short_base.csv'):
        super().__init__()
        # Load the precomputed weekly returns and macro states
        self.df = pd.read_csv(data_path)
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df = self.df.ffill().fillna(0.0) # Fix NaNs in VIX or other columns
        
        # State features
        self.features = [
            'regime_confidence', 'p_bull', 'p_chop', 'p_bear', 
            'transition_risk', 'breadth', 'vix', 'macro_score',
            'news_bias', 'suppression_score'
        ]
        
        # Action space: Continuous value between 0.0 (Cash) and 1.0 (Full Long)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(self.df) - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = row[self.features].values.astype(np.float32)
        # Simple normalization for VIX
        obs[6] = obs[6] / 100.0
        return obs

    def step(self, action):
        # Action is the chosen gross exposure for this week
        exposure = np.clip(action[0], 0.0, 1.0)
        
        # Calculate what the return WOULD have been with this exposure
        row = self.df.iloc[self.current_step]
        actual_portfolio_return = row['portfolio_return']
        actual_gross = row['gross_exposure']
        
        # Base return of the underlying selected stocks (unlevered)
        base_return = actual_portfolio_return / max(0.01, actual_gross)
        
        # Agent's return
        agent_return = base_return * exposure
        
        # Reward: We want to maximize return but heavily penalize drawdown/volatility
        # A simple Sharpe-like reward per step:
        risk_free = 0.05 / 52
        reward = agent_return - risk_free
        
        # Optional: Add a small penalty for high exposure during high VIX to encourage hedging
        # reward -= (exposure * (row['vix'] / 100.0) * 0.01)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'agent_return': agent_return,
            'exposure': exposure,
            'base_return': base_return
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass
