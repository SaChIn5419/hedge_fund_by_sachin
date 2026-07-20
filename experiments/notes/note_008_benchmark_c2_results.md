# Research Note #8: Regime Volatility and Scale Invariance (Benchmark C2)

**Date:** 2026-07-14
**Workstream:** 1 (Information) & 7 (Optimization Science)
**Experiment ID:** EXP_006
**Reproducibility:**
- Kernel: Python 3.10 / SB3
- Dataset: hmm_system_v1 (P=0.95, Lags=12)
- Git Commit: Phase 6.6C

## 1. Question
Do regime volatility shifts worsen optimization Failure Severity, and does expanding the rollout horizon ($n\_steps$) cushion this downside (Prediction P1)?

## 2. Hypothesis (Prediction P1)
Volatility scale shifts affect Failure Severity (Expected Downside) more than Convergence Reliability. Higher volatility increases gradient variance during training, causing deeper failure basins, which can be mitigated by scaling the rollout horizon ($n\_steps = 8192$).

## 3. Experimental Design
- **Environment:** Benchmark C2 (HMM with P=0.95, Lags=12 fixed).
- **Independent Variables:**
  - `Volatility Scale (vol_scale)`: `[0.5, 1.0, 2.0]`.
  - `Rollout Horizon (n_steps)`: `[2048, 8192]`.
- **Sample Size:** 10 seeds per cell (60 runs total).
- **Metrics:** Convergence Reliability ($P(\text{Sharpe} > 0)$), Failure Severity (Expected Downside on Sharpe $\le 0$).

---

## 4. Results

### Median Sharpe / Convergence Reliability (Axis 1) / Failure Severity (Axis 2)
| Volatility Scale | $n\_steps = 2048$ | $n\_steps = 8192$ | Impact of $n\_steps$ |
| :--- | :--- | :--- | :--- |
| **0.5 (Low Vol)** | -11.15 / 0.0% / -14.91 | -12.97 / 0.0% / -13.27 | Insignificant |
| **1.0 (Base Vol)** | -11.03 / 10.0% / -16.13 | -13.17 / 0.0% / -13.29 | Insignificant |
| **2.0 (High Vol)** | -10.82 / 10.0% / -15.98 | -13.56 / 0.0% / -13.16 | Insignificant |

---

## 5. Statistics & Interpretation
Prediction P1 is **Falsified** under the specific conditions of this experiment. Volatility scale shifts had no statistically significant impact on either Convergence Reliability or Failure Severity:

1. **Absolute Scale Invariance:** At $n\_steps = 2048$, as volatility scaled from 0.5 to 2.0 (a 4× increase), the median Sharpe remained flat around -11.0. This indicates that **uniform scaling of both signal and noise (absolute scale) leaves PPO performance approximately invariant under advantage normalization.**
2. **Constant Information Geometry (SNR):** The HMM generator scales the regime drift returns and return volatility by the same factor, leaving the latent state transition SNR unchanged. The oracle still sees the identical problem geometric representation, and PPO's advantage normalization:
   $$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A}$$
   cancels out the scale factor, preventing gradient variance shifts.
3. **Contextual Scope:** Within the environments studied so far, persistence and transition dynamics appear to have a larger effect than uniform scale shifts.

---

## 6. Competing Hypotheses & Explanations (Resolved)
- **Volatility Gradient Explosion:** Falsified for scale-equivalent systems. Standardized advantages prevent scale shifts from inflating gradient variance.
- **Constant Latent SNR:** Confirmed. Scaling volatility scales both the drift and the return variance, leaving the underlying state transition SNR constant.

## 7. Working Theory Revision
Based on this falsification, we update **Working Theory #1**:
* **Update:** We distinguish between **absolute scale** and **information geometry**. Uniform scale scaling is invariant for normalized policy optimization. However, the constituent variables of volatility—specifically the Signal-to-Noise Ratio (SNR) and the information rates—remain active parameters of the learning system.

## 8. Next Roadmap Task
Proceed to **Benchmark C2.5 (SNR Sweep)**. We will fix the volatility scale at $1.0$ and vary the observation noise variance to directly test whether performance decays smoothly as SNR decays, isolating the impact of information geometry on Convergence Reliability.
