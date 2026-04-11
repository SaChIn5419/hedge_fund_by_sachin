from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.paths import OUTPUT_DATA_DIR


def _safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").dropna().mean())


def _classify_baseline(row: pd.Series) -> str:
    regime_score = float(row.get("regime_score", 0.0))
    breadth = float(row.get("breadth", np.nan))
    if regime_score > 0.20 and np.isfinite(breadth) and breadth >= 0.40:
        return "BULL"
    if regime_score < -0.20 and (not np.isfinite(breadth) or breadth <= 0.25):
        return "BEAR"
    return "CHOP"


def _classify_probabilistic(row: pd.Series) -> str:
    probs = {
        "BULL": float(row.get("p_bull", 1 / 3)),
        "CHOP": float(row.get("p_chop", 1 / 3)),
        "BEAR": float(row.get("p_bear", 1 / 3)),
    }
    return max(probs, key=probs.get)


def _classify_probabilistic_no_news(row: pd.Series) -> str:
    p_bull = float(row.get("p_bull", 1 / 3))
    p_bear = float(row.get("p_bear", 1 / 3))
    p_chop = float(row.get("p_chop", 1 / 3))
    news_bias = float(row.get("news_bias", 0.0))
    suppression = float(row.get("suppression_score", 0.0))
    adj_bull = p_bull - 0.10 * news_bias + 0.08 * suppression
    adj_bear = p_bear + 0.06 * news_bias - 0.10 * suppression
    adj_chop = p_chop + 0.04 * suppression
    probs = {"BULL": adj_bull, "CHOP": adj_chop, "BEAR": adj_bear}
    return max(probs, key=probs.get)


@dataclass
class ValidationOutputs:
    txt_path: str
    csv_path: str


class RegimeValidationReporter:
    def __init__(self, capital: float = 1_000_000):
        self.capital = capital

    def _load_inputs(self, trade_csv: str, regime_csv: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        trades = pd.read_csv(trade_csv, parse_dates=["date", "signal_date", "exit_date"])
        if trades.empty:
            raise ValueError("Empty trade log")

        if regime_csv is None:
            regime_csv = str(Path(trade_csv).with_name("regime_trace_chimera_fip.csv"))
        regime_trace = pd.read_csv(regime_csv, parse_dates=["date", "signal_date"]) if Path(regime_csv).exists() else pd.DataFrame()
        return trades, regime_trace

    def _build_daily(self, trades: pd.DataFrame, regime_trace: pd.DataFrame) -> pd.DataFrame:
        active = trades.copy()
        active["date"] = pd.to_datetime(active["date"])
        grouped = active.groupby("date", as_index=False).agg(
            net_pnl=("net_pnl", "sum"),
            regime_confidence=("regime_confidence", "mean"),
            regime_score=("regime_score", "mean"),
            p_bull=("p_bull", "mean"),
            p_chop=("p_chop", "mean"),
            p_bear=("p_bear", "mean"),
            transition_risk=("transition_risk", "mean"),
            breadth=("breadth", "mean"),
            news_bias=("news_bias", "mean"),
            suppression_score=("suppression_score", "mean"),
        )
        grouped["portfolio_return"] = grouped["net_pnl"] / float(self.capital)
        grouped["baseline_state"] = grouped.apply(_classify_baseline, axis=1)
        grouped["prob_state"] = grouped.apply(_classify_probabilistic, axis=1)
        grouped["prob_no_news_state"] = grouped.apply(_classify_probabilistic_no_news, axis=1)

        if not regime_trace.empty:
            trace = regime_trace.copy()
            trace["date"] = pd.to_datetime(trace["date"])
            grouped = grouped.merge(
                trace[["date", "regime", "regime_confidence", "p_bull", "p_chop", "p_bear", "transition_risk"]],
                on="date",
                how="left",
                suffixes=("", "_trace"),
            )
        return grouped.sort_values("date").reset_index(drop=True)

    def _regime_metrics(self, daily: pd.DataFrame, label_col: str) -> dict[str, float]:
        labels = daily[label_col].astype(str)
        returns = daily["portfolio_return"].astype(float)
        state_change = (labels != labels.shift(1)).fillna(False)
        transition_days = int(state_change.sum())
        unstable_days = int((daily["transition_risk"].fillna(0.0) >= 0.60).sum()) if "transition_risk" in daily.columns else 0
        bad_trade_days = int((returns < 0).sum())
        bad_trade_loss = float(returns[returns < 0].sum()) if bad_trade_days else 0.0
        suppressed_bad_days = int(((returns < 0) & (daily["suppression_score"].fillna(0.0) >= 0.60)).sum()) if "suppression_score" in daily.columns else 0

        return {
            "avg_confidence": _safe_mean(daily["regime_confidence"]),
            "avg_transition_risk": _safe_mean(daily["transition_risk"]) if "transition_risk" in daily.columns else 0.0,
            "state_changes": transition_days,
            "state_change_rate": transition_days / max(len(daily), 1),
            "unstable_days": unstable_days,
            "bad_days": bad_trade_days,
            "bad_day_loss": bad_trade_loss,
            "suppressed_bad_days": suppressed_bad_days,
            "bull_share": float((labels == "BULL").mean()),
            "bear_share": float((labels == "BEAR").mean()),
            "chop_share": float((labels == "CHOP").mean()),
        }

    def _write_txt(self, daily: pd.DataFrame, out_txt: str) -> None:
        metrics = {
            "baseline": self._regime_metrics(daily, "baseline_state"),
            "probabilistic_no_news": self._regime_metrics(daily, "prob_no_news_state"),
            "probabilistic_with_news": self._regime_metrics(daily, "prob_state"),
        }
        confusion = pd.crosstab(daily["baseline_state"], daily["prob_state"], rownames=["baseline"], colnames=["prob_with_news"])
        no_news_confusion = pd.crosstab(daily["baseline_state"], daily["prob_no_news_state"], rownames=["baseline"], colnames=["prob_no_news"])

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("CHIMERA REGIME VALIDATION REPORT\n")
            f.write("=" * 64 + "\n")
            f.write(f"Date range         : {daily['date'].iloc[0].date()} -> {daily['date'].iloc[-1].date()}\n")
            f.write(f"Observations       : {len(daily)} daily rows\n")
            f.write("\nMODEL COMPARISON\n")
            for name, vals in metrics.items():
                f.write(
                    f"  {name:<24} conf={vals['avg_confidence']:.3f} "
                    f"trans_risk={vals['avg_transition_risk']:.3f} "
                    f"changes={vals['state_changes']:<4} "
                    f"bad_days={vals['bad_days']:<4} "
                    f"bad_loss={vals['bad_day_loss'] * 100:.2f}% "
                    f"suppressed_bad={vals['suppressed_bad_days']:<4}\n"
                )
            f.write("\nSTATE MIX\n")
            for name, vals in metrics.items():
                f.write(
                    f"  {name:<24} bull={vals['bull_share'] * 100:5.1f}% "
                    f"chop={vals['chop_share'] * 100:5.1f}% "
                    f"bear={vals['bear_share'] * 100:5.1f}%\n"
                )
            f.write("\nBASELINE VS PROBABILISTIC+NEWS\n")
            f.write(confusion.to_string() + "\n")
            f.write("\nBASELINE VS PROBABILISTIC WITHOUT NEWS\n")
            f.write(no_news_confusion.to_string() + "\n")
            if "suppression_score" in daily.columns:
                suppressed = daily[daily["suppression_score"].fillna(0.0) >= 0.60]
                f.write("\nSUPPRESSION WINDOW SUMMARY\n")
                f.write(f"  Suppression days  : {len(suppressed)}\n")
                if not suppressed.empty:
                    f.write(f"  Avg return        : {suppressed['portfolio_return'].mean() * 100:.3f}%\n")
                    f.write(f"  Avg transition risk: {suppressed['transition_risk'].mean():.3f}\n")
                    f.write(f"  Avg news bias     : {suppressed['news_bias'].mean():.3f}\n")

    def generate(self, trade_csv: str, output_prefix: str) -> ValidationOutputs:
        trades, regime_trace = self._load_inputs(trade_csv)
        daily = self._build_daily(trades, regime_trace)
        csv_path = f"{output_prefix}.csv"
        txt_path = f"{output_prefix}.txt"
        daily.to_csv(csv_path, index=False)
        self._write_txt(daily, txt_path)
        return ValidationOutputs(txt_path=txt_path, csv_path=csv_path)


def generate_default_validation(trade_csv: str) -> ValidationOutputs:
    reporter = RegimeValidationReporter()
    return reporter.generate(trade_csv, str(OUTPUT_DATA_DIR / "regime_validation"))
