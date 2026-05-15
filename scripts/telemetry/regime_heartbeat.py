import os
import pandas as pd
from datetime import datetime
from config.paths import REGIME_TRACE_PATH
from engine.signal import ChimeraEngineNormal


def generate_regime_heartbeat():
    """
    Extracts the last recorded regime state from the engine's trace
    to track alpha-divergence between Base and Dexter.
    """
    if not os.path.exists(REGIME_TRACE_PATH):
        print("Regime trace not found. Skipping heartbeat.")
        return

    df = pd.read_csv(REGIME_TRACE_PATH)
    if df.empty:
        print("Regime trace is empty.")
        return

    # Get the most recent observation
    last_row = df.iloc[-1]

    # In the current engine, 'regime' is the final decided state (incorporating Dexter if enabled)
    # We can identify if Dexter pushed it by looking at the 'regime_reason'
    regime = last_row["regime"]
    reason = last_row["regime_reason"]
    confidence = last_row["regime_confidence"]
    date = last_row["date"]

    # Simple logic to detect if Dexter influenced the decision
    # (Dexter's influence usually shows up as 'Sourced from Dexter' or related keywords in the reason)
    is_dexter_shift = "Dexter" in reason if pd.notna(reason) else False

    heartbeat = {
        "date": date,
        "regime": regime,
        "dexter_influence": 1 if is_dexter_shift else 0,
        "confidence": confidence,
        "reason": reason,
    }

    heartbeat_df = pd.DataFrame([heartbeat])
    path = "data/telemetry/regime_heartbeat.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Append to CSV
    file_exists = os.path.exists(path)
    heartbeat_df.to_csv(path, mode="a", index=False, header=not file_exists)
    print(
        f"Heartbeat updated: {date} | Regime: {regime} | Dexter Shift: {is_dexter_shift}"
    )


if __name__ == "__main__":
    generate_regime_heartbeat()
