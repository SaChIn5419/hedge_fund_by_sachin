# Research Note #11: The Quantitative Law of Adaptive Memory (EXP_009)

**Date:** 2026-07-15  
**Workstream:** 1 (Information) & 7 (Optimization Science)  
**Experiment ID:** EXP_009  

---

## 1. Executive Summary
We swept the temporal structure of false positive resets (Uniform vs. Bursty vs. Correlated) under a constant noise rate (300 total resets) and evaluated the response surface $S(\lambda)$ across Sharpe and Maximum Drawdown.

We discovered that:
1. **Temporal Clustering Spikes Risk (Drawdown):** Bursty false alarms (consecutive blocks of 5 resets) spike Maximum Drawdown by **2.8× (4.61% vs. 1.64%)** compared to Uniform false alarms, even though the Sharpe ratio remains identical (~3.68). Wiping memory consecutively blinds the policy dynamically, causing clustered trading errors.
2. **The Recall Dependency Rule:** Wiping memory on false alarms ($\lambda = 0.0$) is highly tolerated *if and only if* Recall is high (transition resets are caught). When Recall is low (as in the Surprise detector, 20.8%), false resets trigger severe state fragmentation drag, collapsing performance.

---

## 2. Experimental Results

We evaluated 18 configurations under Uniform, Bursty, and Correlated false resets:

| Noise Structure | $\lambda = 0.0$ | $\lambda = 0.2$ | $\lambda = 0.4$ | $\lambda = 0.6$ | $\lambda = 0.8$ | $\lambda = 1.0$ (Control) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform (Sharpe)** | **3.66** | 3.44 | 2.99 | 2.47 | 1.58 | 0.19 |
| **Uniform (Drawdown)**| **1.64%** | 3.22% | 6.21% | 12.63%| 20.74%| 49.14% |
| **Bursty (Sharpe)** | **3.69** | 3.42 | 2.93 | 2.40 | 1.50 | 0.19 |
| **Bursty (Drawdown)** | **4.61%** | 4.73% | 6.25% | 12.86%| 20.93%| 49.14% |
| **Correlated (Sharpe)**| **3.73** | 3.48 | 2.98 | 2.45 | 1.51 | 0.19 |
| **Correlated (Drawdown)**| **1.90%** | 3.09% | 6.09% | 12.31%| 21.07%| 49.14% |

---

## 3. Discussion & Analysis

### A. The Temporal Clustering Drawdown Spike
- For a single isolated false alarm (Uniform mode), the model's hidden state is wiped to 0, but it reconstructs its belief state within 1–2 steps due to stable observation cues.
- For bursty false alarms (Bursty mode), the model is forced into consecutive resets. The state is wiped repeatedly before it can accumulate any historical context. This drives the policy into **prolonged blindness**, which manifests as a cluster of losing trades and spikes the Maximum Drawdown to **4.61%**.

### B. The Recall Dependency Rule
Comparing this with Experiment 008B (where Surprise detector resets collapsed Sharpe to -0.34):
- Wiping state on false alarms is harmless if the model also resets at true transitions (high Recall).
- If the detector misses true transitions (low Recall), representation inertia remains active. Wiping the state on false alarms then fragments the recovery process, destroying the model's slow adaptation.

---

## 4. Next Steps
We will update our working theory and proceed to **Phase 6.6L: Benchmark C3 (Transition Asymmetry)**.
