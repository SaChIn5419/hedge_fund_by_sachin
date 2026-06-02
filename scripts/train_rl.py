import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from engine.ml.rl_env import ChimeraPortfolioEnv

def train_agent():
    # 1. Instantiate the env
    env = ChimeraPortfolioEnv(data_path='data/weekly_returns_chimera_fip_zero_short_base.csv')
    
    # Check that the env follows the Gym API
    check_env(env, warn=True)
    
    print("Environment verified. Starting training...")
    
    # 2. Instantiate the agent
    # We use a small policy network (MlpPolicy) since the state is very low dimensional (10 features)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
    
    # 3. Train the agent
    # We'll train for a relatively small number of timesteps since the dataset is only ~315 rows.
    # To prevent overfitting on small data, we keep timesteps moderate.
    model.learn(total_timesteps=50000, progress_bar=True)
    
    # 4. Save the agent
    os.makedirs('engine/ml/models', exist_ok=True)
    model.save("engine/ml/models/ppo_agent")
    print("Model saved to engine/ml/models/ppo_agent.zip")
    
    # 5. Quick Backtest Verification
    obs, info = env.reset()
    agent_returns = []
    base_returns = []
    exposures = []
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_returns.append(info['agent_return'])
        base_returns.append(info['base_return'])
        exposures.append(info['exposure'])
        done = terminated or truncated
        
    cagr_agent = _calc_cagr(agent_returns)
    cagr_base = _calc_cagr(base_returns)
    print(f"\n--- In-Sample Training Results ---")
    print(f"Base Underlying CAGR (100% exposed): {cagr_base*100:.2f}%")
    print(f"RL Agent Managed CAGR: {cagr_agent*100:.2f}%")
    print(f"Average Exposure: {np.mean(exposures)*100:.1f}%")

def _calc_cagr(returns):
    years = len(returns) / 52.0
    total = np.prod([1 + r for r in returns])
    return (total ** (1 / max(1e-5, years))) - 1

if __name__ == "__main__":
    train_agent()
