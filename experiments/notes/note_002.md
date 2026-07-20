# Research Note #2: Temporal Context (Stacked MLP vs LSTM)

**Date:** 2026-07-14
**Workstream:** 1 (Information - Contextual) & 2 (Decision Making)
**Experiment IDs:** EXP_002A through EXP_002E

## 1. Question
Does explicit feature stacking (history) eliminate the need for Recurrent architectures (LSTM) when predicting a slowly evolving latent state?

## 2. Hypothesis
Explicitly stacking historical features captures the majority of the temporal signal (History Gain), rendering the marginal benefit of full LSTM recurrence (Memory Gain) statistically insignificant relative to its computational cost.

## 3. Experimental Design
- **Data:** A synthetic benchmark (Benchmark A) featuring a slowly varying hidden sine wave. `returns[t]` is the future derivative of the wave. `features[t]` is a noisy single-point observation.
- **Controlled Variables:** Learning rate, batch size, PPO hyperparameters, training steps (50k).
- **Independent Variable:** State representation (Current Week, 4-Week Stack, 12-Week Stack) and Policy Architecture (MLP vs LSTM).
- **Dependent Variable:** Out-of-sample Sharpe Ratio on a 20k holdout set.

## 4. Expected Theoretical Answer
Stacking should drastically outperform a single observation. The performance should scale as the observation window spans a larger segment of the dominant cycle.

## 5. Results
| Experiment | Configuration | OOS Sharpe | Training Time |
| :--- | :--- | :--- | :--- |
| **002A** | Current Week / MLP | 0.0000 | ~2.0 min |
| **002B** | 4-Week Stack / MLP | 0.2532 | ~2.0 min |
| **002C** | 12-Week Stack / MLP | 2.2186 | ~2.0 min |
| **002D** | Current Week / LSTM | 1.8919 | ~7.5 min |
| **002E** | 12-Week Stack / LSTM | 3.0911 | ~7.5 min |

## 6. Statistics
- **History Gain (002C - 002A):** +2.22 Sharpe
- **Memory Gain (002E - 002C):** +0.87 Sharpe
- **95% Confidence Interval (Block Bootstrap, p=1/252, n=1000):**
  - **002C:** [1.61, 2.91]
  - **002E:** [2.04, 4.21]
- **Computational Efficiency (Sharpe / Training Hour):**
  - **002C:** 66.6
  - **002E:** 24.7

## 7. Interpretation
For slowly evolving latent-state problems represented by noisy observations, explicit temporal stacking captures most of the information required by a recurrent policy. While the LSTM (002E) achieved the highest absolute point estimate, its 95% Confidence Interval overlaps substantially with the 12-Week Stacked MLP (002C). Therefore, the Memory Gain (+0.87) is not robustly statistically significant, yet the computational cost is roughly 3.7x higher.

## 8. Limitations
This experiment was conducted on **Benchmark A** (Hidden sine wave), which is highly favorable for recurrence due to stationarity and smoothness. Real markets contain structural breaks, changing volatility, and non-stationarity, which may further confound the LSTM's learning dynamics.

## 9. Decision
**Status: Confirmed (Synthetic Benchmark A)**
For rapid iteration, stacked observations provide an excellent cost-performance tradeoff (yielding vastly superior Sharpe/Hour). Whether recurrent policies justify their additional complexity on real financial data remains an open research question.

## 10. Future Work
- Execute the **Lookback Curve (Experiment 002F)** to find the exact plateau instead of guessing 12 weeks.
- Perform **Information Analysis** (SNR, PCA variance) to explain why the specific window wins.
- Elevate to **Benchmark B and C** to test against multi-frequency noise and regime switching.
