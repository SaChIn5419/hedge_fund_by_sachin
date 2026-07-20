# Research Note #5: Mapping the Optimization Landscape (Benchmark A+)

**Date:** 2026-07-14
**Workstream:** 7 (Optimization Science)
**Experiment ID:** EXP_003C
**Reproducibility:**
- Kernel: Python 3.10 / SB3
- Environment: Benchmark A (36-Week Lags)
- Git Commit: Phase 6.4

## 1. Question
Can standard PPO hyperparameters (Entropy, Learning Rate, Batch Size) eliminate the probability of falling into catastrophic degenerate basins when facing high-dimensional inputs with low SNR?

## 2. Null Hypothesis
Hyperparameter adjustments have no measurable impact on the probability of convergence to the "Good Basin".

## 3. Alternative Hypothesis
Higher entropy coefficients or larger batch sizes will smooth the optimization landscape and prevent early collapse into degenerate local minima, significantly increasing the probability of a positive Sharpe.

## 4. Experimental Design
- **Environment:** Benchmark A (Noisy Sine Wave, 36-Week Lags).
- **Configurations (5):** Baseline, High Entropy, Slow LR, Large Batch, Combined Robust.
- **Sample Size:** 10 independent seeds per configuration (50 runs total).
- **Metrics:** Mean, Median, Probability > 0, Probability < -10.

## 5. Expected Theoretical Answer
Large batches reduce gradient variance. High entropy prevents premature convergence to a deterministic policy trap. Both should independently increase the probability of successful convergence.

## 6. Results
| Configuration | Mean Sharpe | Median | Prob > 0 | Prob < -10 | Worst Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | -1.58 | +1.82 | 70% | 20% | -15.97 |
| **High Entropy** | -1.26 | +2.10 | **80%** | 20% | -15.00 |
| **Slow LR** | **-0.20** | **+5.23** | 70% | **10%** | -33.40 |
| **Large Batch** | -9.32 | +2.66 | 60% | 30% | -47.81 |
| **Combined** | -5.48 | +4.71 | 60% | 20% | -43.29 |

## 7. Statistics
High Entropy yielded the highest convergence reliability (80%). Slow Learning Rate yielded the highest median performance (+5.23) and lowest collapse rate (10%). However, standard deviation across all configurations remained massive (>6.0), driven by the devastating magnitude of the catastrophic collapses (up to -47.81).

## 8. Optimization Profile & Interpretation
Hyperparameter tuning changes the *probability* of success but does not fundamentally alter the optimization landscape. Slow LR halves the collapse frequency (10% vs 20%) and dramatically improves median performance. High Entropy modestly lifts the convergence rate (80%). However, no configuration eliminates the degenerate basins. Increasing batch size or combining hyperparameters created *deeper* catastrophic basins (−47.81, −43.29) than the baseline. Once PPO falls into these traps, it reinforces the catastrophic behavior confidently.

## 9. Competing Explanations
- **Insufficient Tuning:** A highly specific grid search might find a completely safe region. (Low Likelihood: the bimodal trap appears structural to high-dim, low-SNR state space).
- **Algorithm Fragility:** PPO's clipping mechanism may be fundamentally insufficient for this geometry. (Moderate Likelihood: warrants testing alternative algorithms).

## 10. Limitations
10 seeds per configuration is relatively small for accurately estimating tail collapse probabilities (10% = 1 seed). Confidence on tail estimates is low.

## 11. Decision
**Status: Null Hypothesis Not Rejected.**
Hyperparameter tuning changes the probability of successful convergence but does not fundamentally alter the optimization landscape. Catastrophic degenerate basins remain structurally present across all tested configurations. Mitigation rather than elimination is the achievable standard.

## 12. Future Work & Roadmap Update
- Adopt **Slow LR (5e-5)** or **High Entropy (0.05)** as the new default for future experiments, as they offer the best Risk/Reward optimization profile.
- Proceed to **Phase 6.5: Benchmark B (Parameterized Latent State)**, running at least 10 seeds on all future architectural tests.
