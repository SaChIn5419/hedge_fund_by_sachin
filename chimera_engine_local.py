import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
from datetime import datetime, timedelta
import os
import glob

DATA_DIR = "/home/sachindb/Documents/hedgefund_chimera/chimera_data"

CONFIG = {
    "CAPITAL": 1000000,
    "TARGET_VOL": 0.20,
    "MAX_LEVERAGE": 1.5,
    "REBALANCE_FREQ": "W-FRI",
    "LOOKBACK_GAUSSIAN": 200,
    "LOOKBACK_VP": 60,
    "LOOKBACK_ENERGY": 20,
    "FIP_LAMBDA": 0.97,
    "FIP_RISK_ADJUST": "DOWNSIDE_VOL",
    "FIP_VOLUME_WEIGHTING": False,
    "FIP_BLEND_RAW": False,
    "FIP_REGIME_ADAPTIVE": False,
    "INDICES": {"BROAD": "nifty50", "RISK": "banknifty", "FEAR": "india_vix"},
    "SAFE_HAVEN": "goldbees",
}


class ChimeraEngineFinal:
    def __init__(self):
        print("--- CHIMERA PROTOCOL: INITIALIZING ---")
        self.trade_log = []
        self.weekly_returns = []
        self.last_allocations = []

    def load_all_stocks(self):
        """Load all stocks from parquet files"""
        stocks_dir = os.path.join(DATA_DIR, "stocks")
        stock_files = glob.glob(os.path.join(stocks_dir, "*.parquet"))

        data_map = {}
        for f in stock_files:
            ticker = os.path.basename(f).replace(".parquet", "")
            try:
                df = pd.read_parquet(f)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                data_map[ticker] = df
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        print(f"Loaded {len(data_map)} stocks from local data")
        return data_map

    def load_indices(self, data_map):
        """Load indices from local parquet or download if missing"""
        indices_dir = os.path.join(DATA_DIR, "indices")

        for name, ticker in CONFIG["INDICES"].items():
            local_path = os.path.join(indices_dir, f"{ticker}.parquet")
            if os.path.exists(local_path):
                df = pd.read_parquet(local_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                data_map[ticker] = df
                print(f"Loaded index {name} ({ticker}) from local data")
            else:
                print(f"Index {name} ({ticker}) not found locally")

        safe_haven_path = os.path.join(DATA_DIR, "macro", "goldbees.parquet")
        if os.path.exists(safe_haven_path):
            df = pd.read_parquet(safe_haven_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            data_map["GOLDBEES"] = df
            print("Loaded GOLDBEES from local data")
        else:
            print("GOLDBEES not found, downloading...")
            try:
                df = yf.download("GOLDBEES.NS", start="2019-01-01", progress=False)
                if not df.empty:
                    data_map["GOLDBEES"] = df
            except:
                pass

        missing_indices = []
        for name, ticker in CONFIG["INDICES"].items():
            if ticker not in data_map:
                yf_ticker = {
                    "BROAD": "^NSEI",
                    "RISK": "^CNXSC",
                    "FEAR": "^INDIAVIX",
                }.get(name)
                if yf_ticker:
                    missing_indices.append(yf_ticker)

        if missing_indices:
            print(f"Downloading missing indices: {missing_indices}")
            try:
                idx_data = yf.download(
                    missing_indices,
                    start="2019-01-01",
                    progress=False,
                    group_by="ticker",
                )
                for t in missing_indices:
                    if t in idx_data.columns.levels[0]:
                        data_map[t] = idx_data[t].dropna(how="all")
            except Exception as e:
                print(f"Error downloading indices: {e}")

        return data_map

    def get_frog_in_pan(self, stock_slice, current_vix=15.0):
        """FIP Momentum with shorting support"""
        if len(stock_slice) < 61:
            return 0.0

        prices = stock_slice["Close"].iloc[-61:]
        volumes = stock_slice["Volume"].iloc[-61:]
        returns = prices.pct_change().dropna()

        if CONFIG.get("FIP_REGIME_ADAPTIVE", False):
            lambda_ = 0.90 if current_vix > 20.0 else 0.98
        else:
            lambda_ = CONFIG.get("FIP_LAMBDA", 0.95)

        n = len(returns)
        weights = np.array([lambda_ ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        pos_mask = (returns > 0).astype(float).values
        neg_mask = (returns < 0).astype(float).values

        frac_positive = np.sum(pos_mask * weights)
        frac_negative = np.sum(neg_mask * weights)

        total_mom = prices.iloc[-1] / prices.iloc[0] - 1.0

        raw_fip = (frac_positive - frac_negative) * np.sign(total_mom) * abs(total_mom)
        fip = raw_fip

        adj_type = CONFIG.get("FIP_RISK_ADJUST", "TOTAL_VOL")

        if adj_type == "TOTAL_VOL" or adj_type is True:
            daily_vol = returns.std()
            if daily_vol == 0 or np.isnan(daily_vol):
                daily_vol = 1e-6
            fip = fip / daily_vol
        elif adj_type == "DOWNSIDE_VOL":
            down_returns = returns[returns < 0]
            down_vol = down_returns.std() if len(down_returns) > 1 else 1e-6
            if down_vol == 0 or np.isnan(down_vol):
                down_vol = 1e-6
            fip = fip / down_vol

        if CONFIG.get("FIP_BLEND_RAW", False):
            fip = (fip + (raw_fip * 100)) / 2.0

        return fip * 100.0 if not pd.isna(fip) else 0.0

    def check_structure(self, df):
        lookback = CONFIG["LOOKBACK_VP"]
        if len(df) < lookback:
            return "N/A"

        hist = df.iloc[-(lookback + 1) : -1]
        current = df["Close"].iloc[-1]

        mean = hist["Close"].mean()
        std = hist["Close"].std()
        if std == 0:
            return "TRAPPED"
        z = (current - mean) / std

        if abs(z) < 1.0:
            return "TRAPPED"
        return "VACUUM"

    def get_gaussian_signal(self, df):
        period = CONFIG["LOOKBACK_GAUSSIAN"]
        if len(df) < period:
            return "NEUTRAL"
        mean = df["Close"].rolling(period).mean().iloc[-1]
        std = df["Close"].rolling(period).std().iloc[-1]
        price = df["Close"].iloc[-1]

        if price > mean + 2 * std:
            return "BREAKOUT"
        elif price < mean - 2 * std:
            return "BREAKDOWN"
        return "NEUTRAL"

    def get_composite_regime(self, data_map, current_date):
        broad = data_map.get(CONFIG["INDICES"]["BROAD"])
        broad_status = "BULL"
        vix = 15.0

        if broad is not None:
            slice_b = broad[broad.index <= current_date]
            if not slice_b.empty:
                sma200 = slice_b["Close"].rolling(200).mean().iloc[-1]
                sma20 = slice_b["Close"].rolling(20).mean().iloc[-1]
                curr_price = slice_b["Close"].iloc[-1]

                if curr_price < sma200 and curr_price < sma20:
                    broad_status = "BEAR"

        risk = data_map.get(CONFIG["INDICES"]["RISK"])
        risk_status = "BULL"
        if risk is not None:
            slice_r = risk[risk.index <= current_date]
            if not slice_r.empty:
                sma50 = slice_r["Close"].rolling(50).mean().iloc[-1]
                if slice_r["Close"].iloc[-1] < sma50:
                    risk_status = "BEAR"

        fear = data_map.get(CONFIG["INDICES"]["FEAR"])
        if fear is not None:
            slice_f = fear[fear.index <= current_date]
            if not slice_f.empty:
                vix = slice_f["Close"].iloc[-1]

        if risk_status == "BEAR" or vix > 24.0:
            return "BEAR", vix, "Canary Died / VIX Spike"
        if broad_status == "BEAR":
            return "BEAR", vix, "Broad Market Trend"

        return "BULL", vix, "All Systems Go"

    def run_simulation(self):
        print("Loading data...")
        data_map = self.load_all_stocks()
        data_map = self.load_indices(data_map)

        calendar = data_map.get(CONFIG["INDICES"]["BROAD"])
        if calendar is None:
            print("CRITICAL: No calendar data available.")
            return

        print(f"Calendar loaded with {len(calendar)} rows")
        dates = calendar.index
        print(f"--- SIMULATING {len(dates)} DAYS ---")

        for i in range(250, len(dates)):
            current_date = dates[i]

            if current_date.weekday() != 4:
                continue

            regime, vix, reason = self.get_composite_regime(data_map, current_date)

            weekly_pnl = 0.0
            if self.last_allocations:
                for t, w, entry_p in self.last_allocations:
                    try:
                        curr_p = (
                            data_map[t]
                            .loc[data_map[t].index <= current_date]
                            .iloc[-1]["Close"]
                        )
                        ret = ((curr_p - entry_p) / entry_p) * w
                        weekly_pnl += ret
                    except:
                        pass
                self.weekly_returns.append(weekly_pnl)

            perf_penalty = 1.0
            rolling_sharpe = 0.0
            if len(self.weekly_returns) >= 20:
                rh = np.array(self.weekly_returns[-20:])
                if np.std(rh) > 0:
                    rolling_sharpe = (np.mean(rh) / np.std(rh)) * np.sqrt(52)

            if len(self.weekly_returns) > 20 and rolling_sharpe < -0.5:
                perf_penalty = 0.5
                reason += f" [KILL-SWITCH: Sharpe {rolling_sharpe:.2f}]"
            elif len(self.weekly_returns) > 20 and rolling_sharpe < 0.0:
                perf_penalty = 0.75
                reason += f" [CAUTION: Sharpe {rolling_sharpe:.2f}]"

            broad_slice = calendar.iloc[: i + 1]
            returns = broad_slice["Close"].pct_change()
            current_vol = returns.tail(20).std() * np.sqrt(252)

            if current_vol == 0 or np.isnan(current_vol):
                current_vol = 0.15

            base_leverage = (
                min(CONFIG["TARGET_VOL"] / current_vol, CONFIG["MAX_LEVERAGE"])
                * perf_penalty
            )

            candidates = []
            available_tickers = [
                t
                for t in data_map.keys()
                if t not in CONFIG["INDICES"].values() and t != "GOLDBEES"
            ]

            for ticker in available_tickers[:100]:
                if ticker not in data_map:
                    continue
                stock_df = data_map[ticker]
                stock_slice = stock_df[stock_df.index <= current_date]

                if len(stock_slice) < 200:
                    continue

                sig = self.get_gaussian_signal(stock_slice)
                fip = self.get_frog_in_pan(stock_slice, current_vix=vix)
                candidates.append((ticker, fip, stock_slice, sig))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_picks = candidates[:10]
            bottom_picks = [c for c in candidates[-5:] if c[1] < 0]

            if not top_picks and not bottom_picks:
                self.trade_log.append(
                    {
                        "date": current_date,
                        "ticker": "CASH",
                        "close": 1.0,
                        "weight": 1.0,
                        "kinetic_energy": 0.0,
                        "structure_tag": "NONE",
                        "leverage_mult": 0.0,
                        "market_state": "BULL_IDLE",
                        "decision_reason": "No Candidates",
                        "nifty_vol": vix / 100,
                        "fwd_return": 0.0,
                        "net_pnl": 0.0,
                        "efficiency": 0.0,
                    }
                )
                continue

            final_weights = {}

            for ticker, energy, stock_slice, sig in top_picks:
                if energy <= 0:
                    continue
                structure = self.check_structure(stock_slice)

                lev_mult = 1.0
                trade_note = "Standard Long"

                if vix > 24:
                    lev_mult = 0.5
                    trade_note = "VIX Penalty"
                elif energy > 0.25 and structure == "VACUUM":
                    lev_mult = 1.5
                    trade_note = "Turbo (High Energy)"
                elif structure == "TRAPPED":
                    lev_mult = 0.0
                    trade_note = "Volume Profile (Chop)"

                if lev_mult > 0:
                    final_weights[ticker] = {
                        "weight": (base_leverage / len(top_picks)) * lev_mult,
                        "energy": energy,
                        "stock_slice": stock_slice,
                        "trade_note": trade_note,
                        "side": "LONG",
                    }

            for ticker, energy, stock_slice, sig in bottom_picks:
                if energy >= 0:
                    continue
                structure = self.check_structure(stock_slice)

                lev_mult = 1.0
                trade_note = "Standard Short"

                if vix > 24:
                    lev_mult = 0.5
                    trade_note = "VIX Penalty"
                elif energy < -0.25 and structure == "VACUUM":
                    lev_mult = 1.5
                    trade_note = "Turbo Short"

                final_weights[ticker] = {
                    "weight": (base_leverage / len(bottom_picks)) * lev_mult * -1,
                    "energy": energy,
                    "stock_slice": stock_slice,
                    "trade_note": trade_note,
                    "side": "SHORT",
                }

            total_net_exposure = sum(
                abs(item["weight"]) for item in final_weights.values()
            )
            max_allowed = CONFIG["MAX_LEVERAGE"] * perf_penalty

            if total_net_exposure > max_allowed:
                scale_factor = max_allowed / total_net_exposure
                for t in final_weights:
                    final_weights[t]["weight"] *= scale_factor
                    final_weights[t]["trade_note"] += " (Norm)"

            self.last_allocations = []

            for ticker, item in final_weights.items():
                final_weight = item["weight"]
                energy = item["energy"]
                stock_slice = item["stock_slice"]
                trade_note = item["trade_note"]
                side = item["side"]

                curr_price = stock_slice["Close"].iloc[-1]
                fwd_ret = 0.0
                try:
                    future_slice = data_map[ticker][
                        data_map[ticker].index > current_date
                    ]
                    if not future_slice.empty:
                        if len(future_slice) >= 5:
                            next_price = future_slice["Close"].iloc[4]
                        else:
                            next_price = future_slice["Close"].iloc[-1]
                        fwd_ret = (next_price - curr_price) / curr_price
                        if side == "SHORT":
                            fwd_ret = -fwd_ret
                except:
                    fwd_ret = 0.0

                self.trade_log.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "close": curr_price,
                        "weight": final_weight,
                        "kinetic_energy": energy / 100,
                        "efficiency": 1.0
                        if self.check_structure(stock_slice) == "VACUUM"
                        else 0.0,
                        "leverage_mult": abs(final_weight / (base_leverage / 10))
                        if final_weight != 0
                        else 0,
                        "market_state": f"{regime}_{side}",
                        "decision_reason": trade_note,
                        "nifty_vol": vix / 100,
                        "fwd_return": fwd_ret,
                        "net_pnl": fwd_ret * 100000 * final_weight,
                        "structure_tag": self.check_structure(stock_slice),
                    }
                )

                self.last_allocations.append((ticker, final_weight, curr_price))

        log_df = pd.DataFrame(self.trade_log)
        filename = "data/tradelog_chimera_fip.csv"
        os.makedirs("data", exist_ok=True)
        log_df.to_csv(filename, index=False)
        print(f"--- PROTOCOL COMPLETE. SAVED: {filename} ---")


if __name__ == "__main__":
    eng = ChimeraEngineFinal()
    eng.run_simulation()
