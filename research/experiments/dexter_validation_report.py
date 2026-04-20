from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.dexter.normalize import prepare_for_calendar
from agents.dexter.validation import VALIDATION_CONTRACT
from config.paths import (
    DEXTER_RESEARCH_FEATURES_PATH,
    NEWS_DAILY_FEATURES_PATH,
    OUTPUT_DATA_DIR,
    REGIME_TRACE_PATH,
    WEEKLY_TRACE_PATH,
)


DEXTER_Z_PREFIX = "dexter_"
MIN_OOS_OBSERVATIONS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Dexter features against FinBERT/news and regime targets.")
    parser.add_argument("--dexter-path", default=str(DEXTER_RESEARCH_FEATURES_PATH))
    parser.add_argument("--news-path", default=str(NEWS_DAILY_FEATURES_PATH))
    parser.add_argument("--weekly-path", default=str(WEEKLY_TRACE_PATH))
    parser.add_argument("--regime-path", default=str(REGIME_TRACE_PATH))
    parser.add_argument("--output-prefix", default=str(OUTPUT_DATA_DIR / "dexter_validation"))
    return parser.parse_args()


def _read_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _date_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("frame requires a date column")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values("date").drop_duplicates("date", keep="last")


def _numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 5 or tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(tmp.iloc[:, 0].rank().corr(tmp.iloc[:, 1].rank()))


def _rolling_spearman(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    values = []
    idx = []
    for end in range(window, len(a) + 1):
        sub_a = a.iloc[end - window:end]
        sub_b = b.iloc[end - window:end]
        values.append(_spearman(sub_a, sub_b))
        idx.append(a.index[end - 1])
    return pd.Series(values, index=idx, dtype=float)


def _vif_table(features: pd.DataFrame) -> pd.DataFrame:
    x = features.dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["feature", "vif", "status"])
    usable_cols = [c for c in x.columns if x[c].nunique() > 1]
    x = x[usable_cols]
    if len(usable_cols) < 2 or len(x) <= len(usable_cols) + 2:
        return pd.DataFrame({"feature": usable_cols, "vif": np.nan, "status": "insufficient_observations"})

    rows = []
    for col in usable_cols:
        y = x[col].to_numpy(dtype=float)
        others = x[[c for c in usable_cols if c != col]].to_numpy(dtype=float)
        others = np.column_stack([np.ones(len(others)), others])
        try:
            beta = np.linalg.lstsq(others, y, rcond=None)[0]
            pred = others @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            vif = 1.0 / max(1e-9, 1.0 - r2) if np.isfinite(r2) else np.nan
        except np.linalg.LinAlgError:
            vif = np.nan
        if np.isfinite(vif) and vif > VALIDATION_CONTRACT.redundancy.hard_max_vif:
            status = "hard_fail"
        elif np.isfinite(vif) and vif > VALIDATION_CONTRACT.redundancy.max_vif:
            status = "fail"
        elif np.isfinite(vif):
            status = "pass"
        else:
            status = "not_computed"
        rows.append({"feature": col, "vif": vif, "status": status})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last")


def _ridge_fit_predict(
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    *,
    train_fraction: float = 0.70,
    alpha: float = 1.0,
) -> Tuple[pd.Series, float]:
    data = frame[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < MIN_OOS_OBSERVATIONS or len(feature_cols) == 0:
        return pd.Series(dtype=float), np.nan
    split = int(len(data) * train_fraction)
    if split < 12 or len(data) - split < 8:
        return pd.Series(dtype=float), np.nan

    train = data.iloc[:split]
    test = data.iloc[split:]
    mu = train[feature_cols].mean()
    sd = train[feature_cols].std().replace(0, np.nan).fillna(1.0)
    x_train = ((train[feature_cols] - mu) / sd).to_numpy(dtype=float)
    x_test = ((test[feature_cols] - mu) / sd).to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    pred = pd.Series(x_test @ beta, index=test.index, name="prediction")

    baseline = float(np.mean(y_train))
    ss_model = float(np.sum((y_test - pred.to_numpy()) ** 2))
    ss_base = float(np.sum((y_test - baseline) ** 2))
    oos_r2 = 1.0 - ss_model / ss_base if ss_base > 0 else np.nan
    return pred, float(oos_r2)


def _auc_score(y_true: pd.Series, score: pd.Series) -> float:
    tmp = pd.concat([y_true, score], axis=1).dropna()
    if tmp.empty:
        return np.nan
    y = tmp.iloc[:, 0].astype(int)
    s = tmp.iloc[:, 1].astype(float)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _balanced_accuracy(y_true: pd.Series, score: pd.Series) -> float:
    tmp = pd.concat([y_true, score], axis=1).dropna()
    if tmp.empty:
        return np.nan
    y = tmp.iloc[:, 0].astype(int)
    s = tmp.iloc[:, 1].astype(float)
    threshold = float(s.median())
    pred = (s >= threshold).astype(int)
    positives = y == 1
    negatives = y == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return np.nan
    tpr = float((pred[positives] == 1).mean())
    tnr = float((pred[negatives] == 0).mean())
    return 0.5 * (tpr + tnr)


def _build_model_frame(dexter: pd.DataFrame, news: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = _date_col(weekly)
    calendar = pd.DatetimeIndex(weekly["date"])
    dexter_prepared = prepare_for_calendar(dexter, calendar) if not dexter.empty else pd.DataFrame({"date": calendar})
    dexter_prepared = _date_col(dexter_prepared).add_prefix(DEXTER_Z_PREFIX).rename(columns={f"{DEXTER_Z_PREFIX}date": "date"})

    if not news.empty:
        news = _date_col(news).set_index("date").reindex(calendar).ffill().reset_index().rename(columns={"index": "date"})
    else:
        news = pd.DataFrame({"date": calendar})

    frame = weekly.merge(news, on="date", how="left").merge(dexter_prepared, on="date", how="left")
    frame["target_next_week_return"] = pd.to_numeric(frame["portfolio_return"], errors="coerce").shift(-1)
    frame["target_next_week_downside"] = (frame["target_next_week_return"] < 0.0).astype(float)
    frame["target_next_week_stress"] = (
        frame["target_next_week_return"]
        <= frame["target_next_week_return"].rolling(52, min_periods=20).quantile(0.25)
    ).astype(float)
    return frame


def _feature_sets(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    news_cols = [c for c in frame.columns if c.endswith("_z") and not c.startswith(DEXTER_Z_PREFIX)]
    dexter_cols = [c for c in frame.columns if c.startswith(DEXTER_Z_PREFIX) and (c.endswith("_z60") or c.endswith("_effective"))]
    dexter_cols = [c for c in dexter_cols if frame[c].nunique(dropna=True) > 1]
    news_cols = [c for c in news_cols if frame[c].nunique(dropna=True) > 1]
    return news_cols, dexter_cols


def _correlation_report(frame: pd.DataFrame, news_cols: List[str], dexter_cols: List[str]) -> pd.DataFrame:
    rows = []
    for dcol in dexter_cols:
        for ncol in news_cols:
            rho = _spearman(frame[dcol], frame[ncol])
            roll = _rolling_spearman(frame[dcol], frame[ncol], 60)
            rows.append(
                {
                    "dexter_feature": dcol,
                    "news_feature": ncol,
                    "spearman": rho,
                    "rolling_median_abs_spearman": float(roll.abs().median()) if not roll.empty else np.nan,
                    "rolling_p75_abs_spearman": float(roll.abs().quantile(0.75)) if not roll.empty else np.nan,
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame(columns=["dexter_feature", "news_feature", "spearman", "rolling_median_abs_spearman", "rolling_p75_abs_spearman", "status"])
    gate = VALIDATION_CONTRACT.redundancy
    report["status"] = np.where(
        (report["spearman"].abs() > gate.max_abs_spearman)
        | (report["rolling_median_abs_spearman"] > gate.max_rolling_median_abs_spearman)
        | (report["rolling_p75_abs_spearman"] > gate.max_rolling_p75_abs_spearman),
        "fail",
        "pass",
    )
    return report.sort_values(["status", "spearman"], ascending=[True, False])


def _incremental_report(frame: pd.DataFrame, news_cols: List[str], dexter_cols: List[str]) -> pd.DataFrame:
    rows = []
    insufficient_dexter = len(dexter_cols) == 0
    for target in ["target_next_week_return"]:
        base_pred, base_r2 = _ridge_fit_predict(frame, news_cols, target)
        full_pred, full_r2 = _ridge_fit_predict(frame, news_cols + dexter_cols, target)
        gain = full_r2 - base_r2 if np.isfinite(base_r2) and np.isfinite(full_r2) else np.nan
        status = "insufficient_dexter_history" if insufficient_dexter else (
            "pass" if np.isfinite(gain) and gain >= VALIDATION_CONTRACT.incremental_value.min_weekly_return_oos_r2_gain else "fail"
        )
        rows.append(
            {
                "target": target,
                "metric": "oos_r2",
                "baseline": base_r2,
                "with_dexter": full_r2,
                "incremental_gain": gain,
                "threshold": VALIDATION_CONTRACT.incremental_value.min_weekly_return_oos_r2_gain,
                "status": status,
            }
        )

    for target in ["target_next_week_downside", "target_next_week_stress"]:
        base_pred, _ = _ridge_fit_predict(frame, news_cols, target)
        full_pred, _ = _ridge_fit_predict(frame, news_cols + dexter_cols, target)
        y = frame[target]
        base_auc = _auc_score(y, base_pred)
        full_auc = _auc_score(y, full_pred)
        auc_gain = full_auc - base_auc if np.isfinite(base_auc) and np.isfinite(full_auc) else np.nan
        base_ba = _balanced_accuracy(y, base_pred)
        full_ba = _balanced_accuracy(y, full_pred)
        ba_gain = full_ba - base_ba if np.isfinite(base_ba) and np.isfinite(full_ba) else np.nan
        rows.append(
            {
                "target": target,
                "metric": "auc",
                "baseline": base_auc,
                "with_dexter": full_auc,
                "incremental_gain": auc_gain,
                "threshold": VALIDATION_CONTRACT.incremental_value.min_classification_auc_gain,
                "status": "insufficient_dexter_history" if insufficient_dexter else (
                    "pass" if np.isfinite(auc_gain) and auc_gain >= VALIDATION_CONTRACT.incremental_value.min_classification_auc_gain else "fail"
                ),
            }
        )
        rows.append(
            {
                "target": target,
                "metric": "balanced_accuracy",
                "baseline": base_ba,
                "with_dexter": full_ba,
                "incremental_gain": ba_gain,
                "threshold": VALIDATION_CONTRACT.incremental_value.min_balanced_accuracy_gain,
                "status": "insufficient_dexter_history" if insufficient_dexter else (
                    "pass" if np.isfinite(ba_gain) and ba_gain >= VALIDATION_CONTRACT.incremental_value.min_balanced_accuracy_gain else "fail"
                ),
            }
        )
    return pd.DataFrame(rows)


def _spillover_report(frame: pd.DataFrame) -> pd.DataFrame:
    amplifier_cols = [c for c in frame.columns if c.endswith("spillover_amplifier_z60") or c.endswith("spillover_amplifier_effective")]
    rows = []
    for col in amplifier_cols:
        usable = frame[[col, "target_next_week_return"]].dropna()
        if usable[col].nunique() < 2 or len(usable) < 30:
            rows.append(
                {
                    "feature": col,
                    "test": "dynamic_corr_proxy",
                    "value": np.nan,
                    "threshold": VALIDATION_CONTRACT.spillover.min_dynamic_corr_beta,
                    "status": "insufficient_observations",
                    "note": "DCC-GARCH validation requires a global benchmark return series and more Dexter history.",
                }
            )
            continue
        # Fallback proxy until a true DCC-GARCH input is supplied: beta of next-week
        # absolute return on the spillover feature. This is not a production pass.
        x = np.column_stack([np.ones(len(usable)), usable[col].to_numpy(dtype=float)])
        y = usable["target_next_week_return"].abs().to_numpy(dtype=float)
        beta = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
        rows.append(
            {
                "feature": col,
                "test": "absolute_return_beta_proxy",
                "value": beta,
                "threshold": VALIDATION_CONTRACT.spillover.min_dynamic_corr_beta,
                "status": "diagnostic_only",
                "note": "Replace with DCC-GARCH India/global dynamic correlation once global benchmark returns are available.",
            }
        )
    return pd.DataFrame(rows)


def _regime_utility_report(frame: pd.DataFrame) -> pd.DataFrame:
    if "market_state" not in frame.columns:
        return pd.DataFrame()
    states = frame["market_state"].astype(str)
    false_defensive = (
        states.isin(["CHOP", "BEAR"])
        & (states.shift(1) == "BULL")
        & (states.shift(-1) == "BULL")
    )
    defensive_count = int((states.isin(["CHOP", "BEAR"]) & (states.shift(1) == "BULL")).sum())
    false_count = int(false_defensive.sum())
    false_rate = false_count / max(defensive_count, 1)
    return pd.DataFrame(
        [
            {
                "metric": "false_defensive_rate",
                "value": false_rate,
                "threshold": VALIDATION_CONTRACT.regime_utility.max_false_defensive_rate_increase,
                "status": "diagnostic_baseline",
                "note": "Requires base-vs-Dexter regime traces and friction assumptions for final pass/fail.",
            },
            {
                "metric": "saved_capital_to_friction_multiple",
                "value": np.nan,
                "threshold": VALIDATION_CONTRACT.regime_utility.min_saved_capital_to_friction_multiple,
                "status": "pending",
                "note": "Compute after paired base/Dexter trace run with turnover and friction estimates.",
            },
        ]
    )


def _write_text(
    output_path: str,
    frame: pd.DataFrame,
    corr: pd.DataFrame,
    vif: pd.DataFrame,
    incremental: pd.DataFrame,
    spillover: pd.DataFrame,
    regime: pd.DataFrame,
    news_cols: List[str],
    dexter_cols: List[str],
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("DEXTER VALIDATION REPORT\n")
        f.write("=" * 72 + "\n")
        f.write(f"Rows                 : {len(frame)}\n")
        f.write(f"Date range           : {frame['date'].min().date()} -> {frame['date'].max().date()}\n")
        f.write(f"News features        : {len(news_cols)}\n")
        f.write(f"Dexter features      : {len(dexter_cols)}\n")
        f.write(f"Validation contract  : {asdict(VALIDATION_CONTRACT)}\n")
        if len(dexter_cols) == 0:
            f.write("\nWARNING: No non-constant stationarized Dexter features. Forward history is insufficient.\n")
        f.write("\nREDUNDANCY SUMMARY\n")
        if corr.empty:
            f.write("  No correlation pairs computed.\n")
        else:
            f.write(corr.head(20).to_string(index=False) + "\n")
        f.write("\nVIF SUMMARY\n")
        f.write(vif.head(20).to_string(index=False) + "\n" if not vif.empty else "  No VIF computed.\n")
        f.write("\nINCREMENTAL VALUE\n")
        f.write(incremental.to_string(index=False) + "\n")
        f.write("\nSPILLOVER\n")
        f.write(spillover.to_string(index=False) + "\n" if not spillover.empty else "  No spillover tests computed.\n")
        f.write("\nREGIME UTILITY\n")
        f.write(regime.to_string(index=False) + "\n" if not regime.empty else "  No regime utility tests computed.\n")


def generate_report(
    *,
    dexter_path: str,
    news_path: str,
    weekly_path: str,
    regime_path: str,
    output_prefix: str,
) -> Dict[str, str]:
    dexter = _read_frame(dexter_path)
    news = _read_frame(news_path)
    weekly = _read_frame(weekly_path)
    regime = _read_frame(regime_path)
    if weekly.empty:
        raise SystemExit(f"No weekly trace found at {weekly_path}")

    if not regime.empty and "date" in regime.columns:
        regime = _date_col(regime)
        weekly = _date_col(weekly).merge(regime[["date", "regime", "news_reason"]], on="date", how="left")

    frame = _build_model_frame(dexter, news, weekly)
    news_cols, dexter_cols = _feature_sets(frame)

    corr = _correlation_report(frame, news_cols, dexter_cols)
    vif = _vif_table(_numeric(frame, news_cols + dexter_cols))
    incremental = _incremental_report(frame, news_cols, dexter_cols)
    spillover = _spillover_report(frame)
    regime_report = _regime_utility_report(frame)

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "joined_csv": str(out_prefix.with_suffix(".joined.csv")),
        "correlation_csv": str(out_prefix.with_suffix(".correlation.csv")),
        "vif_csv": str(out_prefix.with_suffix(".vif.csv")),
        "incremental_csv": str(out_prefix.with_suffix(".incremental.csv")),
        "spillover_csv": str(out_prefix.with_suffix(".spillover.csv")),
        "regime_csv": str(out_prefix.with_suffix(".regime.csv")),
        "txt": str(out_prefix.with_suffix(".txt")),
    }
    frame.to_csv(outputs["joined_csv"], index=False)
    corr.to_csv(outputs["correlation_csv"], index=False)
    vif.to_csv(outputs["vif_csv"], index=False)
    incremental.to_csv(outputs["incremental_csv"], index=False)
    spillover.to_csv(outputs["spillover_csv"], index=False)
    regime_report.to_csv(outputs["regime_csv"], index=False)
    _write_text(outputs["txt"], frame, corr, vif, incremental, spillover, regime_report, news_cols, dexter_cols)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_report(
        dexter_path=args.dexter_path,
        news_path=args.news_path,
        weekly_path=args.weekly_path,
        regime_path=args.regime_path,
        output_prefix=args.output_prefix,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
