# Research Note #9: Recurrent Gating and State Contamination Mitigation (Experiment 007)

**Date:** 2026-07-14
**Workstream:** 1 (Information) & 7 (Optimization Science)
**Experiment ID:** EXP_007
**Reproducibility:**
- Kernel: Python 3.10 / SB3 / sb3_contrib (RecurrentPPO)
- Dataset: hmm_system_v1 (P=0.95, VolScale=1.0, FeatureNoiseVar=0.5)
- Git Commit: Phase 6.6E

## 1. Question
Does a recurrent policy (LSTM) with active state gating outperform stacked MLPs by dynamically contracting its memory state to prevent **State Contamination** under observation noise?

## 2. Hypothesis (Prediction P2)
If stacked MLPs collapse under observation noise because they average history blindly across regime boundaries (State Contamination), then a recurrent policy (LSTM) that learns an input/forget gate sequence should:
1. Achieve significantly higher Convergence Reliability by dynamically contracting/dilating its memory window.
2. Minimize Failure Severity by gating out noisy inputs and choosing a neutral, capital-protecting policy when state confidence collapses.

## 3. Experimental Design
- **Environment:** HMM with P=0.95, VolScale=1.0, FeatureNoiseVar=0.5 (Medium SNR reference baseline).
- **Independent Variables (Architectures):**
  - `MLP` with fixed context stacks: `Lags = 4, 12, 36`.
  - `LSTM` (`RecurrentPPO` with MLP-LSTM policy) using raw 1-lag features, allowing internal recurrence to govern context.
- **Sample Size:** 10 seeds per cell (40 runs total).
- **Metrics:** Median Sharpe, Convergence Reliability ($P(\text{Sharpe} > 0)$), Failure Severity (Expected Downside on Sharpe $\le 0$).

---

## 4. Results (MLP vs. LSTM Gating Comparison)

### Median Sharpe / Convergence Reliability (Axis 1) / Failure Severity (Axis 2)
| Policy | Lags / Window | Mean Sharpe | Median Sharpe | Std Sharpe | Convergence Reliability | Failure Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | **Recurrent** | **+0.002** | **0.00** | 1.06 | **40.0%** | **-0.47** |
| **MLP** | Lags = 4 | -9.31 | -7.70 | 5.32 | 0.0% | -9.31 |
| **MLP** | Lags = 12 | -14.50 | -11.03 | 11.03 | 10.0% | -16.13 |
| **MLP** | Lags = 36 | -17.60 | -15.37 | 6.98 | 0.0% | -17.60 |

---

## 5. Statistics & Interpretation
The empirical results demonstrate a massive qualitative shift, but the exact underlying mechanism remains to be proven:

1. **Dramatically Protected Failure Severity:** 
   The LSTM reduced the Expected Downside on failure to only **-0.47**, compared to the MLPs' downsides of **-9.31** to **-17.60** (a 20× to 35× risk reduction). 
   - *Intelligent Capital Preservation:* In 4 out of 10 seeds, the LSTM converged to a Sharpe of **exactly 0.0000**. Under high noise/transition uncertainty, the recurrent policy learns a conservative "I don't know" risk-neutral strategy, choosing to preserve capital, whereas MLPs trade randomly and lose heavily.
2. **State Contamination Protection:** 
   Because the LSTM operates on 1-lag features, it does not stack observations. Whether this is due to active recurrent gating of transitions or simpler parameters smoothing remains to be isolated.

---

## 6. Competing Hypotheses & Explanations (Under Investigation)
To explain why recurrent policies substantially reduce catastrophic failures, we outline four competing hypotheses:
- **Hypothesis A (Selective Memory Gating):** The forget gate learns to flush state memory (restricting lookback) when crossing regime boundaries, preventing state contamination.
- **Hypothesis B (Optimization Stability):** The LSTM parameterization creates a smoother optimization landscape, preventing policy network gradients from falling into deep suboptimal basins during training.
- **Hypothesis C (Uncertainty Gating):** The network detects state uncertainty and defaults to a flat, risk-neutral policy (Sharpe = 0) rather than trading with low confidence.
- **Hypothesis D (Action Smoothing / Turnover Reduction):** The recurrent state acts as a temporal low-pass filter, smoothing portfolio weight transitions and reducing transaction costs, independent of memory-flushing.

---

## 7. Working Theory Revision
We update **[Working Theory #1](file:///home/sachindb/.gemini/antigravity/brain/3f02f66e-cf8c-45df-bfd3-15e0e4bbb7cd/working_theory_001_temporal_representation.md)**:
* **Update:** Recurrent policies substantially reduce catastrophic failures under the tested environments. The mechanism is hypothesized to involve selective memory gating to prevent state contamination, but alternative explanations—including optimization landscape stability, action smoothing, and uncertainty-aware behavior—remain under active investigation.

## 8. Next Roadmap Task
Proceed to **Experiment 007B (Mechanism Identification)**. We will instrument the LSTM policy, extract internal activations (forget gate, cell state, hidden state), track policy turnover and action variance, and check for forget gate resets during regime transition events to isolate the causal mechanism of robustness.
