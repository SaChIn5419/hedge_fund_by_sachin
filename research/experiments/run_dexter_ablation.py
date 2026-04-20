from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config.paths import OUTPUT_DATA_DIR, PRIMARY_TRADELOG, REGIME_TRACE_PATH, WEEKLY_TRACE_PATH
from engine.signal import ChimeraEngineNormal


OUTPUT_FILES = {
    "trades": PRIMARY_TRADELOG,
    "regime": REGIME_TRACE_PATH,
    "weekly": WEEKLY_TRACE_PATH,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired base-vs-Dexter Chimera ablation.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DATA_DIR / "dexter_ablation"))
    parser.add_argument("--friction-bps", type=float, default=30.0, help="Round-trip rebalance friction estimate in bps.")
    parser.add_argument("--skip-runs", action="store_true", help="Only compare existing ablation outputs.")
    return parser.parse_args()


def _run_engine(label: str, output_dir: Path, *, disable_dexter: bool) -> Dict[str, Path]:
    old_flag = os.environ.get("CHIMERA_DISABLE_DEXTER")
    if disable_dexter:
        os.environ["CHIMERA_DISABLE_DEXTER"] = "1"
    else:
        os.environ.pop("CHIMERA_DISABLE_DEXTER", None)
    try:
        engine = ChimeraEngineNormal()
        engine.run_simulation()
    finally:
        if old_flag is None:
            os.environ.pop("CHIMERA_DISABLE_DEXTER", None)
        else:
            os.environ["CHIMERA_DISABLE_DEXTER"] = old_flag

    output_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, Path] = {}
    for key, src in OUTPUT_FILES.items():
        dst = output_dir / f"{label}_{Path(src).name}"
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


def _backup_primary_outputs(tmp_dir: Path) -> Dict[Path, Path]:
    backups: Dict[Path, Path] = {}
    for src in OUTPUT_FILES.values():
        if Path(src).exists():
            dst = tmp_dir / Path(src).name
            shutil.copy2(src, dst)
            backups[Path(src)] = dst
    return backups


def _restore_primary_outputs(backups: Dict[Path, Path]) -> None:
    for src, backup in backups.items():
        if backup.exists():
            shutil.copy2(backup, src)


def _turnover(trades: pd.DataFrame) -> pd.DataFrame:
    work = trades.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work[work["ticker"] != "CASH"]
    pivot = work.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum").fillna(0.0)
    turnover = pivot.diff().abs().sum(axis=1).fillna(pivot.abs().sum(axis=1))
    return turnover.rename("turnover").reset_index()


def _compare(output_dir: Path, friction_bps: float) -> tuple[pd.DataFrame, str]:
    required = [
        output_dir / f"base_{WEEKLY_TRACE_PATH.name}",
        output_dir / f"dexter_{WEEKLY_TRACE_PATH.name}",
        output_dir / f"base_{PRIMARY_TRADELOG.name}",
        output_dir / f"dexter_{PRIMARY_TRADELOG.name}",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing ablation outputs. Run without --skip-runs first.\n"
            + "\n".join(f"  missing: {path}" for path in missing)
        )
    base_weekly = pd.read_csv(output_dir / f"base_{WEEKLY_TRACE_PATH.name}", parse_dates=["date"])
    dexter_weekly = pd.read_csv(output_dir / f"dexter_{WEEKLY_TRACE_PATH.name}", parse_dates=["date"])
    base_trades = pd.read_csv(output_dir / f"base_{PRIMARY_TRADELOG.name}", parse_dates=["date"])
    dexter_trades = pd.read_csv(output_dir / f"dexter_{PRIMARY_TRADELOG.name}", parse_dates=["date"])

    base_weekly["date"] = base_weekly["date"].dt.normalize()
    dexter_weekly["date"] = dexter_weekly["date"].dt.normalize()
    joined = base_weekly.merge(dexter_weekly, on="date", suffixes=("_base", "_dexter"))
    joined = joined.merge(_turnover(base_trades), on="date", how="left").rename(columns={"turnover": "turnover_base"})
    joined = joined.merge(_turnover(dexter_trades), on="date", how="left").rename(columns={"turnover": "turnover_dexter"})
    joined[["turnover_base", "turnover_dexter"]] = joined[["turnover_base", "turnover_dexter"]].fillna(0.0)

    friction = float(friction_bps) / 10_000.0
    joined["extra_turnover"] = (joined["turnover_dexter"] - joined["turnover_base"]).clip(lower=0.0)
    joined["extra_friction_cost"] = joined["extra_turnover"] * friction
    joined["return_delta"] = joined["portfolio_return_dexter"] - joined["portfolio_return_base"]
    joined["defensive_shift"] = (
        (joined["market_state_base"] == "BULL")
        & (joined["market_state_dexter"].isin(["CHOP", "BEAR"]))
    )
    joined["false_defensive_roundtrip"] = (
        joined["defensive_shift"]
        & (joined["market_state_base"].shift(-1) == "BULL")
        & (joined["market_state_dexter"].shift(-1) == "BULL")
    )
    joined["clears_friction_gate"] = joined["return_delta"] > (
        2.0 * joined["extra_friction_cost"]
    )

    defensive = joined[joined["defensive_shift"]]
    false_rate = float(joined["false_defensive_roundtrip"].sum() / max(joined["defensive_shift"].sum(), 1))
    avg_return_delta = float(defensive["return_delta"].mean()) if not defensive.empty else np.nan
    clear_rate = float(defensive["clears_friction_gate"].mean()) if not defensive.empty else np.nan

    text = "\n".join(
        [
            "DEXTER ABLATION SUMMARY",
            "=" * 72,
            f"Rows                         : {len(joined)}",
            f"Defensive shifts             : {int(joined['defensive_shift'].sum())}",
            f"False defensive roundtrips   : {int(joined['false_defensive_roundtrip'].sum())}",
            f"False defensive rate         : {false_rate:.3f}",
            f"Avg return delta on shifts   : {avg_return_delta:.6f}",
            f"Friction gate clear rate     : {clear_rate:.3f}",
            f"Friction estimate            : {friction_bps:.1f} bps",
            "",
            "A defensive shift only passes production utility if preserved capital exceeds 2x extra rebalance friction.",
        ]
    )
    return joined, text


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.skip_runs:
        with tempfile.TemporaryDirectory(prefix="chimera_dexter_ablation_") as tmp:
            backups = _backup_primary_outputs(Path(tmp))
            try:
                _run_engine("base", output_dir, disable_dexter=True)
                _run_engine("dexter", output_dir, disable_dexter=False)
            finally:
                _restore_primary_outputs(backups)

    comparison, text = _compare(output_dir, args.friction_bps)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    summary_path = output_dir / "summary.txt"
    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote summary: {summary_path}")
    print(text)


if __name__ == "__main__":
    main()
