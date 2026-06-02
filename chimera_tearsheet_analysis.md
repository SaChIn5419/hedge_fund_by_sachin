# Chimera v2 Tearsheet — Deep Analysis

**Period**: 2019-12-06 → 2026-05-22 · 315 weeks (6.5 years)  
**Starting Capital**: ₹10,00,000 → **Final Equity: ₹46,86,251**

---

## Headline Performance

| Metric | Value | Verdict |
|--------|-------|---------|
| **Total Return** | **368.63%** | 4.7× money in 6.5 years |
| **CAGR** | **27.02%** | Exceptional — beats most fund managers |
| **Sharpe Ratio** | **1.26** | Strong risk-adjusted returns |
| **Sortino Ratio** | **1.94** | Downside-controlled — losses are smaller than wins |
| **Max Drawdown** | **-28.71%** | Painful but within equity strategy norms |
| **Calmar Ratio** | **0.94** | Decent, slightly below 1.0 ideal |
| **Win Rate** | **58.1%** | Solid edge — wins more often than loses |
| **Win/Loss Ratio** | **1.18×** | Wins are 18% larger than losses on average |
| **Gain/Pain Ratio** | **1.64×** | For every ₹1 lost, ₹1.64 gained |

> [!TIP]
> The Sortino of 1.94 being ~50% higher than the Sharpe of 1.26 reveals an important asymmetry: the strategy's volatility is **skewed to the upside**. The positive skew (0.22) and moderate kurtosis (1.52) confirm this — fat right tail, thin left tail. This is exactly the profile you want.

---

## Regime Breakdown — Where the Alpha Lives

| Regime | Time Spent | Cum Return | Weekly Avg | Win Rate | Sharpe |
|--------|-----------|------------|------------|----------|--------|
| 🟢 **BULL** | 57% (178w) | **+300.8%** | +0.86% | 58.4% | **1.61** |
| 🟡 **CHOP** | 30% (94w) | +12.3% | +0.13% | 59.6% | 0.73 |
| 🔴 **BEAR** | 14% (43w) | +4.1% | +0.11% | 53.5% | 0.43 |

> [!IMPORTANT]
> **97% of all profits come from BULL regime.** CHOP and BEAR are essentially capital preservation modes — they don't lose much, but they don't make much either. The strategy's entire thesis hinges on riding BULL correctly and surviving everything else.

### Key Insight: CHOP is Actually Well-Behaved
- CHOP has the **highest win rate** (59.6%) but the **lowest volatility** (9.47% annualized)
- This means Chimera is correctly de-risking during sideways markets — taking small, high-probability bets
- The 0.73 Sharpe in CHOP is respectable for what's essentially a "do no harm" mode

### Key Insight: BEAR is Net Positive (Barely)
- +4.1% cumulative over 43 BEAR weeks is remarkable — most momentum strategies hemorrhage in bear markets
- The 53.5% win rate means the short book is just barely profitable
- Volatility spikes to 13.3% in BEAR despite the reduced exposure — this is the inherent nature of bear regimes

---

## Yearly Returns — The Full Picture

| Year | Return | Volatility | Sharpe | Max DD | Verdict |
|------|--------|-----------|--------|--------|---------|
| 2019 | +0.95% | 2.6% | 4.83 | 0.0% | 4 weeks only, startup |
| **2020** | **+41.3%** | 21.2% | 1.92 | -14.4% | COVID crash navigated brilliantly |
| **2021** | **+130.2%** | 30.8% | **3.11** | -9.4% | 🏆 **Best year — monster run** |
| 2022 | **-23.5%** | 20.2% | -1.28 | -23.8% | ⚠️ Bear market pain |
| **2023** | **+70.6%** | 18.4% | **3.13** | -6.8% | 🏆 **Cleanest year — low DD, high return** |
| 2024 | +35.0% | 26.2% | 1.38 | -13.1% | Solid but volatile |
| 2025 | **-15.9%** | 10.6% | -1.69 | -13.8% | ⚠️ Current slump |
| 2026 | -3.5% | 8.7% | -1.09 | -7.5% | 19 weeks, still in drawdown |

> [!WARNING]
> **2025 and 2026 are both negative.** The strategy has been underwater for ~80 weeks (since Oct 2024). This is the second-deepest drawdown at -26.17% and it has **NOT recovered yet**. This needs attention.

---

## Drawdown Episodes — The Scars

| Rank | Depth | Start | Trough | Recovery | Duration |
|------|-------|-------|--------|----------|----------|
| **#1** | **-28.71%** | 2022-01-07 | 2023-03-17 | 2023-08-04 | 78 weeks |
| **#2** | **-26.17%** | 2024-10-04 | 2026-03-20 | **⚠️ ONGOING** | **80+ weeks** |
| #3 | -14.37% | 2020-02-07 | 2020-03-27 | 2020-08-21 | 25 weeks |
| #4 | -13.11% | 2024-02-02 | 2024-03-01 | 2024-04-26 | 10 weeks |
| #5 | -9.41% | 2021-03-12 | 2021-03-12 | 2021-04-23 | 5 weeks |

> [!CAUTION]
> **The current drawdown (#2) is already longer than the worst drawdown (#1) in duration — 80 vs 78 weeks — and still hasn't recovered.** While the depth is slightly shallower (-26.2% vs -28.7%), the extended duration is a concern. The trough was hit on 2026-03-20, and the strategy has recovered slightly since, but still has ~26% to claw back to the Oct 2024 peak.

---

## Streak Analysis

| Metric | Value |
|--------|-------|
| Longest winning streak | **11 weeks** |
| Longest losing streak | 7 weeks |
| Avg winning streak | 2.4 weeks |
| Avg losing streak | 1.8 weeks |

The asymmetry here is healthy — winning streaks are both longer (max and average) than losing streaks. This confirms the strategy has **trend-following DNA** that lets winners compound.

---

## Recent 13-Week Momentum

| Metric | Value |
|--------|-------|
| 13-week return | **-0.51%** |
| Annualized volatility | 9.07% |
| Win rate | 46.2% |
| Sharpe | -0.18 |
| Regime mix | CHOP (8w), BEAR (5w) |

> [!NOTE]
> The recent quarter has been essentially flat — no BULL regime at all. The strategy is correctly reducing exposure (low vol of 9%) but isn't finding opportunities. This is expected behavior when the market is stuck in CHOP/BEAR — the system is designed to preserve capital here, not generate alpha.

---

## Regime Transition Matrix

```
to      BEAR  BULL  CHOP
from
BEAR      34     0     9
BULL       1   163    14
CHOP       8    15    70
```

### Key Observations:
- **BULL is extremely sticky**: 163 self-transitions vs only 15 exits (92% persistence). Once Chimera identifies a bull, it stays in it.
- **BEAR never transitions directly to BULL**: All 9 BEAR exits go through CHOP first. This is realistic market behavior — markets don't V-reverse from bear to bull without a bottoming process.
- **CHOP is the gateway regime**: It's the only state that transitions to both BULL and BEAR, making it the critical decision point.
- **BULL→BEAR is extremely rare** (only 1 occurrence): Chimera almost always routes through CHOP before calling a bear market.

---

## Actionable Insights

### 1. The Ongoing Drawdown Is the #1 Priority
The strategy has been in its second-worst drawdown for 80+ weeks. While the system is behaving correctly (low exposure in CHOP/BEAR), the duration raises questions:
- Is the regime classifier being too cautious — calling CHOP when it should be calling BULL?
- Are there missed opportunities in the recent market rally that Chimera isn't capturing?

### 2. 2023 Was the Gold Standard Year
- 70.6% return with only -6.8% max drawdown is a Sharpe of 3.13
- This is what Chimera looks like when the regime classifier is nailing it — high confidence BULL calls with clean trend following

### 3. Short Book Needs Work
- Shorts contribute only -₹27,954 total PnL (net negative)
- BEAR shorts are barely break-even despite correct regime identification
- Consider whether the short book is worth the complexity, or if it should be purely defensive (reduce long exposure) rather than active (take short positions)

### 4. The Win/Loss Ratio (1.18×) Could Be Higher
- A momentum strategy should ideally have 1.5×+ win/loss ratio
- The current 1.18× suggests position sizing or exit timing could be tightened
- The Gain/Pain ratio of 1.64× is better, which means the issue is more about per-trade sizing than overall exposure management
