# Research Note #8B: Observation SNR and Lookback Adaptability (Benchmark C2.5)

**Date:** 2026-07-14
**Workstream:** 1 (Information) & 7 (Optimization Science)
**Experiment ID:** EXP_006B
**Reproducibility:**
- Kernel: Python 3.10 / SB3
- Dataset: hmm_system_v1 (P=0.95, VolScale=1.0)
- Git Commit: Phase 6.6D

## 1. Question
Does Convergence Reliability decay smoothly as the Signal-to-Noise Ratio (SNR) decays, and does the optimal lookback window size interact with observation noise?

## 2. Hypothesis (Prediction P1B)
Convergence Reliability will decay smoothly as SNR decays (i.e. as observation noise increases). Additionally, the optimal lookback $L^*$ must shift:
- At **high SNR (low noise)**, shorter lookbacks should dominate because the current state is immediately observable (minimizing Staleness Cost).
- At **low SNR (high noise)**, longer lookbacks must be used to filter out observation noise (maximizing Signal Gain), despite the Staleness Cost.

## 3. Experimental Design
- **Environment:** Benchmark C2.5 (HMM with P=0.95, VolScale=1.0 fixed).
- **Independent Variables:**
  - `Observation Noise (feature_noise_var)`: `[0.125, 0.5, 2.0, 8.0]` (representing High, Medium, Low, Very Low SNR).
  - `Lags (L)`: `[4, 12, 36]`.
- **Sample Size:** 10 seeds per cell (120 runs total).
- **Metrics:** Median Sharpe, Convergence Reliability (Prob > 0), Failure Severity (Expected Downside).

---

## 4. Results (The SNR-Lags Response Surface)

### Median Sharpe / Convergence Reliability (Axis 1) / Failure Severity (Axis 2)
| Feature Noise Var (SNR) | Lags = 4 | Lags = 12 | Lags = 36 | Best Lookback |
| :--- | :--- | :--- | :--- | :--- |
| **0.125 (High SNR)** | **-1.30 [30%] / -5.78** | -4.11 [10%] / -6.96 | -7.83 [0%] / -7.62 | **Lags=4** |
| **0.500 (Med SNR)** | **-7.70 [0%] / -9.31** | -11.03 [10%] / -16.13 | -15.37 [0%] / -17.60 | **Lags=4 / 12** |
| **2.000 (Low SNR)** | **-19.62 [0%] / -21.53** | -23.89 [0%] / -24.06 | -31.77 [0%] / -30.83 | **Lags=4** |
| **8.000 (Very Low SNR)**| **-26.25 [0%] / -26.05** | -27.68 [0%] / -27.54 | -34.19 [0%] / -32.56 | **Lags=4** |

---

## 5. Statistics & Interpretation
The hypothesis that "Lowering SNR forces $L^*$ to expand" is **Falsified**. 

1. **State Contamination vs Denoising:** Simply adding more history (increasing lags to 36) was *not* an effective denoising mechanism. Across all noise scales (from 0.125 to 8.0), **Lags = 36 remained the worst-performing lookback configuration**. 
2. **Obsolete Information Contamination:** In a regime-switching environment, historical observations span multiple distinct regimes. Averaging these observations does not improve the current state's SNR; instead, it mixes conflicting trends, causing **State Contamination** (representation bias) alongside observation noise.
3. **The Nonstationarity Barrier:** Lowering SNR substantially degrades convergence reliability. Within the tested architectures (flat stacked MLPs) and environments, increasing the lookback window did not compensate for this degradation.

---

## 6. Competing Hypotheses & Explanations (Resolved)
- **Signal Denoising via Context Stacking:** Falsified. Flat stack averaging is unable to separate observation noise from transition-induced state contamination.
- **Selective Memory Bottleneck:** Primary hypothesis. An MLP is unable to perform adaptive filtering on stale history. A recurrent network (LSTM) with active input/forget gates should be able to dynamically contract its memory state when crossing transitions, resolving the tradeoff.

## 7. Working Theory Revision
We update **[Working Theory #1](file:///home/sachindb/.gemini/antigravity/brain/3f02f66e-cf8c-45df-bfd3-15e0e4bbb7cd/working_theory_001_temporal_representation.md)**:
* **Update:** We define $L^* = f(\tau, \text{Observation Noise}, \text{State Contamination})$. State contamination is distinct from observation noise. If the learning architecture lacks a mechanism to dynamically filter out obsolete history (selective memory), increasing $L$ to combat noise is always dominated by representation contamination.

## 8. Next Roadmap Task
Proceed to **Phase 6.6E: LSTM Selective Memory Sweep (Experiment 007)**. We will evaluate a recurrent policy (LSTM) across varying observation noise and lookback bounds to test if selective memory gating allows the agent to prevent state contamination and recover performance under noise.
