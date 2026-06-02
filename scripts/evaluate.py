import pandas as pd
import numpy as np

def evaluate(returns_file):
    df = pd.read_csv(returns_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    returns = df['portfolio_return'].values
    
    # CAGR
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    total_return = np.prod(1 + returns)
    cagr = (total_return ** (1 / years)) - 1
    
    # Sharpe (Weekly annualized)
    vol = np.std(returns) * np.sqrt(52)
    sharpe = (cagr - 0.05) / vol if vol > 0 else 0
    
    # Drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns / running_max) - 1
    max_dd = np.min(drawdowns)
    
    print(f"--- ML Engine Results ---")
    print(f"CAGR: {cagr*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    evaluate('data/weekly_returns_chimera_fip.csv')
