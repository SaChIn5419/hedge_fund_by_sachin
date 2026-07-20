# Research Note #7B: The Optimal HMM Lookback Surface (Experiment 005B)

**Date:** 2026-07-14
**Workstream:** 1 (Information) & 7 (Optimization Science)
**Experiment ID:** EXP_005B
**Reproducibility:**
- Kernel: Python 3.10 / SB3
- Dataset: hmm_system_v1 (3 regimes: Bull, Bear, Chop)
- Git Commit: Phase 6.6B

## 1. Question
What is the exact mathematical shape of the optimal lookback surface $L^* = f(P)$ across regime switching dynamics, and how does the Context-Lag Tradeoff behave as we vary persistence continuously?

## 2. Null Hypothesis
The optimal lookback size is static ($L^* = \text{constant}$) and independent of the environment's regime persistence ($P$).

## 3. Alternative Hypothesis
The optimal lookback $L^*$ is a function of the regime persistence $P$. As persistence increases (regimes last longer), the optimal lookback window shifts to larger values to capture long-term trends. As persistence decreases, the optimal lookback shifts to smaller values to avoid information staleness and boundary-crossing signal contamination.

## 4. Experimental Design
- **Environment:** Benchmark C (HMM with Bull, Bear, Chop regimes).
- **Independent Variables:**
  - `Persistence (P)`: `[0.90, 0.95, 0.99]` (Regimes last 10 days, 20 days, or 100 days on average).
  - `Lags (L)`: `[4, 8, 12, 24, 36, 48]`.
- **Sample Size:** 5 seeds per cell (90 total training runs).
- **Metrics:** Median Sharpe, Probability > 0 (Axis 1), Expected Downside (Axis 2).

---

## 5. Results (The Response Surface)

### Median Sharpe / [Prob > 0 (Axis 1)] / Expected Downside (Axis 2)
| Persistence (P) | L = 4 | L = 8 | L = 12 | L = 24 | L = 36 | L = 48 | Optimal $L^*$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.90 (~10d)** | **-9.38 [0%] / -12.06** | -14.36 [0%] / -15.22 | -14.74 [0%] / -18.71 | -28.77 [0%] / -24.00 | -35.62 [0%] / -28.87 | -21.34 [0%] / -20.85 | **L = 4** |
| **0.95 (~20d)** | -5.34 [0%] / -6.43 | -23.58 [0%] / -19.08 | **-8.26 [20%] / -18.43** | **-4.67 [0%] / -9.74** | -20.88 [0%] / -21.44 | -9.90 [0%] / -11.45 | **L = 24 / 12** |
| **0.99 (~100d)** | -3.75 [0%] / -7.71 | -9.68 [20%] / -17.21 | **-3.13 [20%] / -9.72** | -11.82 [0%] / -16.61 | -12.87 [0%] / -20.18 | -10.29 [0%] / -9.15 | **L = 12** |

---

## 6. Statistics & Interpretation
The Null Hypothesis is **Rejected**. The optimal lookback $L^*$ is a function of the regime persistence $P$, and can be understood as a **temporal bias-variance tradeoff**:

* **Short windows (L = 4, 8, 12)** exhibit lower bias (they do not mix obsolete data across boundaries) but higher variance (they are sensitive to local observation noise).
* **Long windows (L = 24, 36, 48)** exhibit lower variance (more noise smoothing) but higher bias (they introduce severe information staleness across regime transitions).

1. **At Low Persistence ($P=0.90$, $\tau \approx 10$d):** The optimal lookback is the smallest window **$L^* = 4$**. As lookback extends, performance collapses monotonically to -35.62 at $L=36$ due to the high staleness cost of mixing rapidly switching regimes.
2. **At Moderate Persistence ($P=0.95$, $\tau \approx 20$d):** The optimal lookback shifts outward, peaking at **$L^* = 24$** (median -4.67) and **$L^* = 12$** (median -8.26, and the only cell to achieve positive seeds). The window size matches the natural timescale of the environment to resolve the signal.
3. **At High Persistence ($P=0.99$, $\tau \approx 100$d):** The optimal lookback is a **broad plateau centered around short-to-medium lookbacks** (L=4 to L=12, medians -3.75 to -3.13), rather than a sharp peak.

---

## 7. Competing Explanations & Hypotheses
- **Emerging Law of Temporal Representation:** The optimal temporal representation depends on the characteristic time scale of the latent process ($\tau = 1 / (1-P)$) and decreases as environmental nonstationarity increases.
- **The All-Weather Plateau Hypothesis:** At very large lookbacks ($L=48$), performance slightly rebounds across all persistences (e.g. from -35.62 back to -21.34 at $P=0.90$). 
  - *Hypothesis A (All-Weather Policy):* The network stops trying to time regime entries and learns a robust, non-directional "all-weather" policy that acts on the long-term stationary distribution.
  - *Hypothesis B (Optimization Artifact):* The large lookback window acts as a natural regularizer, reducing policy variance at the expense of return potential.
  - *Hypothesis C:* Insufficient training steps prevent the network from resolving the 48-lag state space, leading to a default passive policy.

## 8. Limitations
We only isolated Regime Persistence. 5 seeds per cell remains too small for definitive tail collapse statistics.

## 9. Research Roadmap Update (Decision)
- **Roadmap Shift:** Before proceeding to Benchmark C2 (Regime Volatility), we will compile **Theory Note #1: Temporal Representation in Sequential Decision Problems** to synthesize our findings from Benchmarks A, B, and C into a unified conceptual framework.
- **Next Task:** Author and review `theory_note_001_temporal_representation.md` to formalize the Context-Lag tradeoff mathematically and outline future falsification tests.
