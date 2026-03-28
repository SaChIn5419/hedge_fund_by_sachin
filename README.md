# 🌌 Chimera Dispersion Engine v2

**Autonomous Institutional-Grade Trading System for Indian Equities**

Chimera is a market-neutral dispersion engine designed to exploit cross-sectional momentum and fractal integration while strictly controlling for market regime shifts. This v2 overhaul transforms the system into a high-performance quant engine with significant risk mitigation and alpha generation.

## 📊 Performance Statistics (2019-2026)

| Metric | Performance |
|--------|-------------|
| **Initial Capital** | ₹1,000,000 |
| **Final Equity** | **₹4,600,165** |
| **Total Return** | **360.02%** |
| **CAGR** | **27.48%** |
| **Sharpe Ratio** | **1.24** |
| **Sortino Ratio** | **1.98** |
| **Max Drawdown** | -32.35% |
| **Excess Return vs Nifty** | 266.13% |

---

## 🛠️ Key Components

### 🧠 Signal Layer: Alpha Engine
- **FIP (Fractal Integration Physics)**: Continuity-weighted momentum with volatility-adjusted normalization.
- **RSI Overbought Filter**: Prevents momentum-crash entries in overextended stocks (RSI > 75).
- **Mom5 Crash Filter**: Immediate exclusion of stocks experiencing high-velocity short-term crashes (>10%/week).
- **CS-Z Scores**: Cross-sectional Z-score normalization for better dispersion capture.

### 🛡️ Risk & Regime Layer
- **Regime Classifier**: Multi-stage classification (BULL, CHOP, BEAR) using Nifty-200SMA and broad-market breadth.
- **Zero-Correlation Setup**: Engineered for near-zero correlation with the benchmark index.
- **Diversification**: 20 long-position names with 10% hard gross weight caps.

---

## 📈 Visual Report

![Backtest Report](data/report_chimera_fip.png)

## 📁 Repository Structure

- `chimera_engine.py`: The core simulation and alpha engine.
- `chimera_backtest_report.py`: Advanced diagnostics and visualization suite.
- `run_all_backtests.py`: Master execution script.
- `data/`: Reports and trade logs (raw data excluded).

---

> **Note**: This repository contains code and generated reports only. Market data in `chimera_data/` is excluded from the version control for size and license considerations.
