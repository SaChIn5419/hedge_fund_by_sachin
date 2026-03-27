# Particle Learning Filter — Full Technical Design

## Overview: Two New Files

```
strategies/mystic_pulse/
├── particle_filter.py          ← [NEW] Core algorithm (standalone, no dependencies)
├── run_mystic_bt_plf.py        ← [NEW] Backtest that uses the filter
├── mystic_engine.py            ← [UNCHANGED] Numba trend score calculator
├── run_mystic_bt.py            ← [UNCHANGED] Original backtest (kept for comparison)
└── test_bench.py               ← [UNCHANGED]
```

---

## File 1: `particle_filter.py` — The Core Algorithm

### Constants (Fixed, Never Tuned)
```python
K = 3           # Number of regimes: discovered as TREND / CHOP / CRASH
N_PARTICLES = 200  # Particle count (paper default)
NU = 5.0        # Inverse-Gamma degrees of freedom (fat tails)
K_FACTORS = 1   # Number of latent factors (1 = market factor)
ALPHA_0 = 1.0   # Dirichlet prior (uniform start)
PHI_SQ = 0.98   # Forgetting factor for sufficient statistics (slow adaptation)
```

---

### Class: `ParticleLearningFilter`

```python
class ParticleLearningFilter:
    def __init__(self, K=3, N=200, nu=5.0, k_factors=1):
```

**What it stores (per particle i = 1..N):**

| Attribute | Shape | Purpose |
|-----------|-------|---------|
| `self.regimes[i]` | int | Current regime label r_t for particle i |
| `self.x[i]` | (k,) | Latent factor state x_t |
| `self.m_x[i]` | (k,) | Kalman mean of x_t |
| `self.C_x[i]` | (k,k) | Kalman covariance of x_t |
| `self.B[i]` | (K, s, k) | Regime-dependent factor loadings |
| `self.A[i]` | (k, k) | State transition matrix |
| `self.R_s[i]` | (s, s) | Observation noise covariance |
| `self.alpha[i]` | (K, K) | Dirichlet counts for transition matrix |
| `self.v[i]` | float | Sufficient stat for R_s estimation |
| `self.Phi[i]` | (k, k) | Sufficient stat (precision) for B |
| `self.Psi[i]` | (k, k) | Sufficient stat (precision) for A |

Where `k=1` (latent factors) and `s=1` (observing 1D portfolio return). So all matrices collapse to **scalars** in practice. This makes it extremely fast.

---

### Function 1: [__init__(self, K, N, nu, k_factors)](file:///d:/finance/hedgefund_chimera/engine/chimera_engine.py#40-45)

**What it does:** Initializes all N particles with diffuse (uninformative) priors.

```python
def __init__(self, K=3, N=200, nu=5.0, k_factors=1):
    self.K = K
    self.N = N
    self.nu = nu
    self.k = k_factors
    
    # Per-particle state
    self.regimes = np.zeros(N, dtype=int)           # Start in regime 0
    self.x = np.zeros((N, k_factors))               # Latent factor = 0
    self.m_x = np.zeros((N, k_factors))             # Kalman mean
    self.C_x = np.ones((N, k_factors, k_factors))   # Kalman cov = identity (diffuse)
    
    # Per-particle sufficient statistics (all start diffuse)
    self.B = np.zeros((N, K, 1, k_factors))         # Factor loadings per regime
    self.A = np.eye(k_factors)[None, :, :].repeat(N, axis=0)  # A = identity
    self.R_s = np.ones((N, 1, 1)) * 0.01            # Small initial obs noise
    self.v = np.ones(N) * 10.0                      # IG sufficient stat
    self.Phi = np.eye(k_factors)[None,:,:].repeat(N, axis=0) * 0.01  # Diffuse B prior
    self.Psi = np.eye(k_factors)[None,:,:].repeat(N, axis=0) * 0.01  # Diffuse A prior
    self.alpha = np.ones((N, K, K)) * ALPHA_0       # Uniform Dirichlet
```

**Why these values:** All priors are diffuse/uninformative → the filter "knows nothing" at t=0 and learns purely from data. No look-ahead.

---

### Function 2: `step(self, y_t)` — The Core 6-Step Algorithm

**Input:** [y_t](file:///d:/finance/hedgefund_chimera/chimera_analytics.py#43-51) — a single scalar (today's realized portfolio return)  
**Output:** Updates all internal particle states. Call `get_regime_probs()` after to read the result.

```python
def step(self, y_t):
    # STEP 1: Data Augmentation — draw heavy-tail variance multiplier
    lambda_t = self._sample_inv_gamma()          # shape: (N,)
    
    # STEP 2: Lookahead Resampling — weight particles by predictive likelihood
    weights = self._compute_weights(y_t, lambda_t)  # shape: (N,)
    indices = self._resample(weights)               # shape: (N,), int
    self._apply_resampling(indices)
    
    # STEP 3: Regime Propagation — sample new regime from posterior
    self._propagate_regimes(y_t, lambda_t)
    
    # STEP 4: Factor State Propagation — sample latent factor x_t
    self._propagate_factors(y_t, lambda_t)
    
    # STEP 5: Sufficient Statistics Update — deterministic RLS
    self._update_sufficient_stats(y_t, lambda_t)
    
    # STEP 6: Kalman Update — update state tracking
    self._kalman_update(y_t, lambda_t)
```

Now let me break down each sub-function:

---

### Function 2a: `_sample_inv_gamma(self)` — Step 1

**Purpose:** Draw fat-tail innovation multiplier for each particle.

```python
def _sample_inv_gamma(self):
    """
    λ_t ~ InverseGamma(ν/2, ν/2)
    When ν=5, this produces occasional λ >> 1 (fat tails)
    """
    # scipy: invgamma.rvs(a=ν/2, scale=ν/2, size=N)
    # Equivalent: 1 / gamma(ν/2, 2/ν)
    return 1.0 / np.random.gamma(self.nu / 2, 2.0 / self.nu, size=self.N)
```

**Why:** Financial returns have fat tails. λ_t > 1 means "today's return variance is larger than normal" — the filter adapts to volatility spikes without overfitting.

---

### Function 2b: `_compute_weights(self, y_t, lambda_t)` — Step 2 (Lookahead)

**Purpose:** Each particle predicts what y_t should look like given its current state. Particles that predicted well get higher weight.

```python
def _compute_weights(self, y_t, lambda_t):
    """
    W_i ∝ Σ_k N(y_t; m_y(k), C_y(k)) * p(r_t=k | r_{t-1}^i)
    
    For our 1D case (s=1, k=1):
      m_y(k) = B[k] * A * m_x     (predicted observation mean)
      C_y(k) = B[k]^2 * A^2 * C_x + B[k]^2 + λ*R_s  (predicted variance)
    """
    log_weights = np.zeros(self.N)
    
    for i in range(self.N):
        total_lik = 0.0
        for k in range(self.K):
            # Predictive mean and variance for regime k
            m_y = self.B[i, k] * self.A[i] * self.m_x[i]
            C_y = (self.B[i,k]**2 * self.A[i]**2 * self.C_x[i] 
                   + self.B[i,k]**2 + lambda_t[i] * self.R_s[i])
            
            # Gaussian likelihood of y_t under this prediction
            lik = normal_pdf(y_t, m_y, C_y)
            
            # Transition probability from current regime to k
            trans_prob = self.alpha[i, self.regimes[i], k] 
            trans_prob /= self.alpha[i, self.regimes[i], :].sum()
            
            total_lik += lik * trans_prob
        
        log_weights[i] = np.log(max(total_lik, 1e-300))
    
    # Normalize
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()
```

**Why lookahead:** This is the "auxiliary" part. Standard particle filters just propagate blindly. The lookahead step checks "does this new data point agree with my prediction?" and kills off particles with bad predictions BEFORE propagating. This dramatically reduces particle degeneracy.

---

### Function 2c: `_resample(self, weights)` — Step 2 (continued)

```python
def _resample(self, weights):
    """Multinomial resampling: draw N indices proportional to weights."""
    return np.random.choice(self.N, size=self.N, replace=True, p=weights)
```

### Function 2d: `_apply_resampling(self, indices)`

```python
def _apply_resampling(self, indices):
    """Copy all particle states from the resampled indices."""
    self.regimes = self.regimes[indices]
    self.x = self.x[indices].copy()
    self.m_x = self.m_x[indices].copy()
    self.C_x = self.C_x[indices].copy()
    self.B = self.B[indices].copy()
    self.A = self.A[indices].copy()
    self.R_s = self.R_s[indices].copy()
    self.alpha = self.alpha[indices].copy()
    self.v = self.v[indices].copy()
    self.Phi = self.Phi[indices].copy()
    self.Psi = self.Psi[indices].copy()
```

---

### Function 2e: `_propagate_regimes(self, y_t, lambda_t)` — Step 3

**Purpose:** For each particle, sample the new regime from the posterior (given the new observation).

```python
def _propagate_regimes(self, y_t, lambda_t):
    """
    p(r_t = k | ...) ∝ N(y_t; m_y(k), C_y(k)) * transition_prob(r_{t-1} → k)
    Draw from this multinomial for each particle.
    """
    for i in range(self.N):
        probs = np.zeros(self.K)
        for k in range(self.K):
            m_y = self.B[i, k] * self.A[i] * self.m_x[i]
            C_y = (self.B[i,k]**2 * self.A[i]**2 * self.C_x[i]
                   + self.B[i,k]**2 + lambda_t[i] * self.R_s[i])
            
            lik = normal_pdf(y_t, m_y, C_y)
            trans = self.alpha[i, self.regimes[i], k] / self.alpha[i, self.regimes[i], :].sum()
            probs[k] = lik * trans
        
        probs /= probs.sum() + 1e-300
        self.regimes[i] = np.random.choice(self.K, p=probs)
```

**Why this matters:** This is where the filter decides "which regime am I in NOW?" based on the return it just observed + its prior belief about transitions.

---

### Function 2f: `_propagate_factors(self, y_t, lambda_t)` — Step 4

**Purpose:** Sample the hidden factor value x_t given the new regime and observation.

```python
def _propagate_factors(self, y_t, lambda_t):
    """
    x_t ~ N(m_x_new, C_x_new)
    C_x_new^{-1} = B(r_t)^T * (λ*R_s)^{-1} * B(r_t) + I
    m_x_new = C_x_new * (B(r_t)^T * (λ*R_s)^{-1} * y_t + A * x_{t-1})
    
    For 1D: these are all scalar operations.
    """
    for i in range(self.N):
        r = self.regimes[i]
        b = self.B[i, r].item()        # scalar
        a = self.A[i].item()            # scalar
        lam_rs = lambda_t[i] * self.R_s[i].item()  # scalar
        
        C_x_inv = b**2 / lam_rs + 1.0
        C_x_new = 1.0 / C_x_inv
        m_x_new = C_x_new * (b / lam_rs * y_t + a * self.x[i].item())
        
        self.x[i] = np.random.normal(m_x_new, np.sqrt(max(C_x_new, 1e-10)))
        self.m_x[i] = m_x_new
        self.C_x[i] = C_x_new
```

---

### Function 2g: `_update_sufficient_stats(self, y_t, lambda_t)` — Step 5

**Purpose:** Deterministically update the running parameter estimates using recursive least squares. This is where the "learning" happens — no sampling, pure math.

```python
def _update_sufficient_stats(self, y_t, lambda_t):
    """
    Updates B (factor loadings per regime), A (transition), 
    R_s (noise), and alpha (Dirichlet counts).
    All formulas from paper's Step 5.
    """
    phi_sq = 0.98  # Forgetting factor
    
    for i in range(self.N):
        r = self.regimes[i]
        x_curr = self.x[i].item()
        
        # --- Observation layer (B, R_s) ---
        self.v[i] = phi_sq * self.v[i] + 1.0
        g = self.Phi[i].item() * x_curr
        zeta_sq = lambda_t[i] * phi_sq + x_curr * g
        
        e_hat = y_t - self.B[i, r].item() * x_curr  # prediction error
        
        # Update B for the active regime only
        self.B[i, r] += (g / zeta_sq) * e_hat
        
        # Update Phi (precision of B)
        self.Phi[i] = (1.0 / phi_sq) * (self.Phi[i] - (g * g / zeta_sq))
        self.Phi[i] = max(self.Phi[i].item(), 1e-6)  # Clamp for stability
        
        # Update R_s (observation noise)
        self.R_s[i] = phi_sq * (self.R_s[i] + e_hat**2 / zeta_sq)
        
        # --- State transition layer (A) ---
        # (same RLS structure for state equation)
        x_prev = self.x[i].item()  # from before propagation
        g_x = self.Psi[i].item() * x_prev
        sigma_sq = phi_sq + x_prev * g_x
        
        e_hat_x = x_curr - self.A[i].item() * x_prev
        self.A[i] += (g_x / sigma_sq) * e_hat_x
        self.Psi[i] = (1.0 / phi_sq) * (self.Psi[i] - g_x**2 / sigma_sq)
        self.Psi[i] = max(self.Psi[i].item(), 1e-6)
        
        # --- Transition matrix (Dirichlet counts) ---
        r_prev = self.regimes[i]  # regime from this step (after propagation)
        self.alpha[i, r_prev, r] += 1.0
```

**Why "sufficient statistics":** Instead of storing the entire history, we compress it into these running estimates. This is what makes Particle Learning memory-efficient and avoids overfitting to ancient data (the forgetting factor φ²=0.98 means data from ~50 steps ago has half-weight).

---

### Function 3: `get_regime_probs(self)` — Read the Output

```python
def get_regime_probs(self):
    """
    Returns the posterior probability of each regime by counting
    how many particles voted for each regime.
    """
    counts = np.bincount(self.regimes, minlength=self.K)
    return counts / self.N

def get_dominant_regime(self):
    """Returns the regime with highest posterior probability."""
    probs = self.get_regime_probs()
    return np.argmax(probs), probs
```

---

### Function 4: `classify_regimes(self, return_history)` — Post-hoc Labeling

**Purpose:** After running the filter over all data, label regime 0/1/2 as TREND/CHOP/CRASH based on their observed characteristics. **This does NOT affect the filter's decisions** — it's purely for human readability.

```python
def classify_regimes(self, daily_returns, regime_labels):
    """
    Compute mean return per regime and assign names:
      - Highest mean return → TREND
      - Lowest mean return  → CRASH  
      - Middle              → CHOP
    """
    regime_means = {}
    for k in range(self.K):
        mask = regime_labels == k
        if mask.sum() > 0:
            regime_means[k] = daily_returns[mask].mean()
        else:
            regime_means[k] = 0.0
    
    sorted_regimes = sorted(regime_means, key=regime_means.get, reverse=True)
    labels = {}
    labels[sorted_regimes[0]] = "TREND"
    labels[sorted_regimes[1]] = "CHOP"
    labels[sorted_regimes[2]] = "CRASH"
    return labels
```

---

### Helper: `normal_pdf(x, mean, var)`

```python
def normal_pdf(x, mean, var):
    """Univariate Gaussian PDF (no scipy dependency)."""
    return np.exp(-0.5 * (x - mean)**2 / var) / np.sqrt(2 * np.pi * var)
```

---

## File 2: `run_mystic_bt_plf.py` — Backtest Integration

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Load data & compute Mystic signals (SAME AS V0)  │
│  DuckDB → Polars → Numba trend_score → is_invested         │
│  NO volume filter. Just trend_score > 0.                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Build daily UNFILTERED portfolio return stream    │
│  For each day: take Top 10 by trend_score, equal weight     │
│  Compute: daily_ret = Σ(weight_i × fwd_return_i)           │
│  This is the V0 baseline return stream.                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Run Particle Learning Filter SEQUENTIALLY         │
│                                                             │
│  for day in sorted_dates:                                   │
│      regime = plf.get_dominant_regime()   ← uses t-1 state  │
│      regime_map[day] = regime             ← store decision  │
│      plf.step(daily_ret[day])             ← feed REALIZED   │
│                                              return to      │
│                                              update beliefs │
│                                                             │
│  KEY: Decision at time t uses regime from BEFORE step(y_t)  │
│       This guarantees zero lookahead bias.                  │ 
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Filter trades by regime                           │
│                                                             │
│  For each day:                                              │
│    if regime_map[day] == TREND:                              │ 
│        keep all Top 10 trades (full allocation)             │
│    elif regime_map[day] == CHOP:                             │
│        reduce to Top 5, half weight (defensive)             │
│    elif regime_map[day] == CRASH:                             │
│        go to cash (no trades)                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Generate trade log → Deep Diagnostic + Tearsheet  │
└─────────────────────────────────────────────────────────────┘
```

### Key Functions in `run_mystic_bt_plf.py`

```python
def run_backtest_plf(df: pl.DataFrame):
    """
    Full backtest with Particle Learning Filter regime gating.
    
    Steps:
    1. Compute trend_score for all stocks (Numba, same as V0)
    2. Build UNFILTERED daily Top-10 portfolio returns
    3. Run PLF sequentially over daily returns
    4. Gate trades: TREND → full, CHOP → defensive, CRASH → cash
    5. Output trade log CSV for diagnostic
    """
```

#### Sub-function: `compute_unfiltered_returns(df)`
```python
def compute_unfiltered_returns(df):
    """
    Input: Panel data with trend_score computed
    Output: Dict[date → float] of daily aggregate portfolio returns
    
    Logic:
    - Filter trend_score > 0
    - Rank top 10 per day
    - Equal weight 10%
    - Return = Σ(weight × fwd_return)
    
    This is IDENTICAL to V0 baseline logic.
    No volume filter, no regime filter yet.
    """
```

#### Sub-function: `apply_regime_gate(trades_df, regime_map, regime_labels)`
```python
def apply_regime_gate(trades_df, regime_map, regime_labels):
    """
    Input: 
      - trades_df: All V0 baseline trades (Polars DataFrame)
      - regime_map: Dict[date → int] regime per day
      - regime_labels: Dict[int → str] mapping regime_id to TREND/CHOP/CRASH
    
    Output: Filtered trades DataFrame
    
    Logic:
      TREND  → keep all trades, weight = 10% (aggressive)
      CHOP   → keep top 5 only, weight = 10% (defensive)
      CRASH  → drop all trades (100% cash)
    """
```

---

## Overfitting & Lookahead Bias Safeguards

### Causal Guarantee (Zero Lookahead)

The critical line in the backtest loop:
```python
for day in sorted_dates:
    regime = plf.get_dominant_regime()   # ← reads YESTERDAY's belief
    regime_map[day] = regime             # ← decision for TODAY
    plf.step(daily_ret[day])             # ← NOW update with today's data
```

The regime decision at time t is made **BEFORE** the filter sees y_t. This is mathematically equivalent to trading on yesterday's close with today's regime belief.

### No Hyperparameter Tuning
All constants (K=3, N=200, ν=5, φ²=0.98, α₀=1) are fixed from paper defaults and financial theory. None will be changed based on backtest results.

### Split-Sample Validation
```
IS:  2018–2021  →  Record Sharpe, CAGR, MaxDD
OOS: 2022–2026  →  Same filter continues (no reset)
Pass condition: OOS_Sharpe > 0.5 × IS_Sharpe
```
