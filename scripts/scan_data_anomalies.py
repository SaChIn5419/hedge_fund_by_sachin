import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

DATA_DIR = "/home/sachindb/Documents/hedgefund_chimera/chimera_data"

def scan_file_anomalies(path: str) -> List[Dict]:
    ticker = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return [{"ticker": ticker, "date": "N/A", "type": "LOAD_ERROR", "msg": f"Failed to load: {e}"}]

    # Ensure required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            return [{"ticker": ticker, "date": "N/A", "type": "MISSING_COLUMNS", "msg": f"Missing column {col}"}]

    df = df.sort_index()
    if df.empty or len(df) < 10:
        return []

    anomalies = []
    
    # 1. Check for flatlined closes (prices remain identical for >= 5 consecutive active days)
    # Exclude volume = 0 days since it might just be non-trading or halted, but consecutive identical prices with trading is suspect.
    close_vals = df["Close"].values
    vol_vals = df["Volume"].values
    dates = df.index
    
    consecutive_flat = 0
    for i in range(1, len(df)):
        if close_vals[i] == close_vals[i-1] and vol_vals[i] > 0:
            consecutive_flat += 1
            if consecutive_flat >= 10:  # 10 days of flat prices with active volume is highly suspicious
                anomalies.append({
                    "ticker": ticker,
                    "date": str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i]),
                    "type": "FLATLINED_CLOSE",
                    "msg": f"Price flatlined at {close_vals[i]} for {consecutive_flat} consecutive trading days."
                })
        else:
            consecutive_flat = 0
            
    # 2. Check for daily return outliers and price discontinuities
    # Single-day returns
    df["Returns"] = df["Close"].pct_change()
    ret_vals = df["Returns"].values
    
    # Trailing 60-day realized volatility of returns
    df["Vol60"] = df["Returns"].rolling(60).std()
    vol60_vals = df["Vol60"].values
    
    for i in range(60, len(df)):
        ret = ret_vals[i]
        vol = vol60_vals[i-1] # previous day vol
        date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])
        
        if np.isnan(ret) or np.isnan(vol) or vol == 0:
            continue
            
        z_score = np.abs(ret) / vol
        
        # Flag returns beyond 8 sigma of trailing volatility
        if z_score > 8.0 and np.abs(ret) > 0.05:
            anomalies.append({
                "ticker": ticker,
                "date": date_str,
                "type": "RETURN_VOL_OUTLIER",
                "msg": f"Daily return {ret:.2%} is {z_score:.1f} sigma of trailing vol ({vol:.2%})."
            })
            
        # Flag single-day absolute returns exceeding 40% (potential splits/unadjusted data)
        if np.abs(ret) > 0.40:
            anomalies.append({
                "ticker": ticker,
                "date": date_str,
                "type": "PRICE_DISCONTINUITY",
                "msg": f"Extreme daily return of {ret:.2%} (close jumped from {close_vals[i-1]:.2f} to {close_vals[i]:.2f})."
            })
            
        # Flag invalid price checks (High < Low, Close < 0, etc.)
        if df["High"].values[i] < df["Low"].values[i]:
            anomalies.append({
                "ticker": ticker,
                "date": date_str,
                "type": "INVALID_HIGH_LOW",
                "msg": f"High {df['High'].values[i]} is lower than Low {df['Low'].values[i]}."
            })
            
        if close_vals[i] <= 0:
            anomalies.append({
                "ticker": ticker,
                "date": date_str,
                "type": "NON_POSITIVE_PRICE",
                "msg": f"Negative or zero Close price: {close_vals[i]}."
            })

    return anomalies

def run_anomaly_scan():
    print("================================================================================")
    print("CHIMERA DATA QUALITY & ANOMALY SCANNER")
    print("================================================================================")
    
    all_anomalies = []
    
    # Locate all parquet files
    for subdir in ["stocks", "indices", "macro"]:
        folder = os.path.join(DATA_DIR, subdir)
        if not os.path.isdir(folder):
            continue
        print(f"Scanning subdirectory: {subdir}...")
        files = glob.glob(os.path.join(folder, "*.parquet"))
        for path in files:
            file_anomalies = scan_file_anomalies(path)
            all_anomalies.extend(file_anomalies)
            
    print(f"Scan complete. Total anomalies detected: {len(all_anomalies)}")
    
    if all_anomalies:
        print("\n--- DETECTED ANOMALIES ---")
        df_anom = pd.DataFrame(all_anomalies)
        # Format display
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)
        print(df_anom.to_string(index=False))
        
        # Save report
        report_path = "data/detected_data_anomalies.csv"
        df_anom.to_csv(report_path, index=False)
        print(f"\nSaved detailed anomaly report to {report_path}")
    else:
        print("No anomalies detected across all stock, index, and macro series! Data quality check PASSED.")

if __name__ == "__main__":
    run_anomaly_scan()
