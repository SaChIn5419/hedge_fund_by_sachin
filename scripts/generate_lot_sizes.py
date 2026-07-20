"""Script to generate static lot_sizes.json from Angel One Scrip Master."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Resolve project paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.pairs.data_pipeline import download_scrip_master


def main():
    print("Downloading and parsing Scrip Master...")
    cache_dir = REPO_ROOT / "data"
    df = download_scrip_master(str(cache_dir))
    
    if df is None or df.empty:
        print("Error: Could not load Scrip Master.")
        return
        
    print("Filtering for NSE Stock Futures (exch_seg: NFO, instrumenttype: FUTSTK)...")
    fo_df = df[(df["exch_seg"] == "NFO") & (df["instrumenttype"] == "FUTSTK")]
    
    if fo_df.empty:
        print("Error: No stock futures found in Scrip Master.")
        return
        
    lot_map = {}
    for _, row in fo_df.iterrows():
        name = row["name"].upper().strip()
        try:
            lot_size = int(row["lotsize"])
            # Keep the first/current lot size or maximum lot size
            if name not in lot_map or lot_size > lot_map[name]:
                lot_map[name] = lot_size
        except (ValueError, TypeError):
            continue
            
    output_dir = REPO_ROOT / "chimera_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lot_sizes.json"
    
    print(f"Extracted {len(lot_map)} F&O tickers and their lot sizes.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lot_map, f, indent=4, sort_keys=True)
        
    print(f"Saved lot sizes mapping to {output_path}")


if __name__ == "__main__":
    main()
