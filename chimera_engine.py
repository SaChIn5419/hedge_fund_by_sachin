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


class ChimeraEngineNormal:
    def __init__(self):
        print("--- CHIMERA NORMAL: PURE FIP MOMENTUM ---")
        self.trade_log = []
        self.weekly_returns = []
        self.last_allocations = []

    def load_all_stocks(self):
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
        indices_dir = os.path.join(DATA_DIR, "indices")

        for name, ticker in CONFIG["INDICES"].items():
            local_path = os.path.join(indices_dir, f"{ticker}.parquet")
            if os.path.exists(local_path):
                df = pd.read_parquet(local_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                data_map[ticker] = df
                print(f"Loaded index {name} ({ticker})")

        safe_haven_path = os.path.join(DATA_DIR, "macro", "goldbees.parquet")
        if os.path.exists(safe_haven_path):
            df = pd.read_parquet(safe_haven_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            data_map["GOLDBEES"] = df
            print("Loaded GOLDBEES")

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
            print(f"Downloading missing: {missing_indices}")
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
                print(f"Error: {e}")

        return data_map

    def get_frog_in_pan(self, stock_slice, current_vix=15.0):
        if len(stock_slice) < 61:
            return 0.0

        prices = stock_slice["Close"].iloc[-61:]
        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return 0.0

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

        down_returns = returns[returns < 0]
        down_vol = down_returns.std() if len(down_returns) > 1 else 0.001
        down_vol = max(down_vol, 0.001)
        fip = fip / down_vol * 100

        # Only cap extreme negative values, keep positive unbounded
        fip = max(-100, fip)
        return fip

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
            return "BEAR", vix, "Canary/VIX"
        if broad_status == "BEAR":
            return "BEAR", vix, "Broad"

        return "BULL", vix, "OK"

    def run_simulation(self):
        print("Loading data...")
        data_map = self.load_all_stocks()
        data_map = self.load_indices(data_map)

        calendar = data_map.get(CONFIG["INDICES"]["BROAD"])
        if calendar is None:
            print("No calendar data!")
            return

        print(f"Calendar: {len(calendar)} rows")
        dates = calendar.index

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
            if len(self.weekly_returns) >= 20:
                rh = np.array(self.weekly_returns[-20:])
                if np.std(rh) > 0:
                    rolling_sharpe = (np.mean(rh) / np.std(rh)) * np.sqrt(52)
                    if rolling_sharpe < -0.5:
                        perf_penalty = 0.5
                    elif rolling_sharpe < 0.0:
                        perf_penalty = 0.75

            broad_slice = calendar.iloc[: i + 1]
            returns = broad_slice["Close"].pct_change()
            current_vol = returns.tail(20).std() * np.sqrt(252)
            current_vol = max(current_vol, 0.15)

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

            for ticker in available_tickers[:150]:
                if ticker not in data_map:
                    continue
                stock_df = data_map[ticker]
                stock_slice = stock_df[stock_df.index <= current_date]

                if len(stock_slice) < 200:
                    continue

                fip = self.get_frog_in_pan(stock_slice, current_vix=vix)
                candidates.append((ticker, fip, stock_slice))

            candidates.sort(key=lambda x: x[1], reverse=True)

            # Regime-based: more shorts in bear
            if regime == "BEAR":
                top_picks = [c for c in candidates[:10] if c[1] > 0]
                bottom_picks = [c for c in candidates[-10:] if c[1] < 0]
            else:
                top_picks = [c for c in candidates[:10] if c[1] > 0]
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
                        "market_state": "IDLE",
                        "decision_reason": "No Signal",
                        "nifty_vol": vix / 100,
                        "fwd_return": 0.0,
                        "net_pnl": 0.0,
                        "efficiency": 0.0,
                    }
                )
                continue

            final_weights = {}

            for ticker, energy, stock_slice in top_picks:
                structure = self.check_structure(stock_slice)
                lev_mult = 1.0
                trade_note = "Long"

                if vix > 24:
                    lev_mult = 0.5
                elif energy > 25 and structure == "VACUUM":
                    lev_mult = 1.5

                if lev_mult > 0:
                    final_weights[ticker] = {
                        "weight": (base_leverage / len(top_picks)) * lev_mult,
                        "energy": energy,
                        "stock_slice": stock_slice,
                        "trade_note": trade_note,
                        "side": "LONG",
                    }

            for ticker, energy, stock_slice in bottom_picks:
                structure = self.check_structure(stock_slice)
                lev_mult = 1.0
                trade_note = "Short"

                if vix > 24:
                    lev_mult = 0.5
                elif energy < -25 and structure == "VACUUM":
                    lev_mult = 1.5

                final_weights[ticker] = {
                    "weight": (base_leverage / len(bottom_picks)) * lev_mult * -1,
                    "energy": energy,
                    "stock_slice": stock_slice,
                    "trade_note": trade_note,
                    "side": "SHORT",
                }

            total_exposure = sum(abs(item["weight"]) for item in final_weights.values())
            max_allowed = CONFIG["MAX_LEVERAGE"] * perf_penalty

            if total_exposure > max_allowed:
                scale = max_allowed / total_exposure
                for t in final_weights:
                    final_weights[t]["weight"] *= scale

            self.last_allocations = []

            for ticker, item in final_weights.items():
                stock_slice = item["stock_slice"]
                curr_price = stock_slice["Close"].iloc[-1]
                fwd_ret = 0.0
                try:
                    future = data_map[ticker][data_map[ticker].index > current_date]
                    if not future.empty:
                        next_price = future["Close"].iloc[min(4, len(future) - 1)]
                        fwd_ret = (next_price - curr_price) / curr_price
                        if item["side"] == "SHORT":
                            fwd_ret = -fwd_ret
                except:
                    pass

                self.trade_log.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "close": curr_price,
                        "weight": item["weight"],
                        "kinetic_energy": item["energy"],
                        "efficiency": 1.0
                        if self.check_structure(stock_slice) == "VACUUM"
                        else 0.0,
                        "leverage_mult": abs(item["weight"] / (base_leverage / 8)),
                        "market_state": f"{regime}_{item['side']}",
                        "decision_reason": item["trade_note"],
                        "nifty_vol": vix / 100,
                        "fwd_return": fwd_ret,
                        "net_pnl": fwd_ret * 100000 * item["weight"],
                        "structure_tag": self.check_structure(stock_slice),
                    }
                )

                self.last_allocations.append((ticker, item["weight"], curr_price))

        log_df = pd.DataFrame(self.trade_log)
        filename = "data/tradelog_normal.csv"
        os.makedirs("data", exist_ok=True)
        log_df.to_csv(filename, index=False)
        print(f"--- SAVED: {filename} ---")


if __name__ == "__main__":
    eng = ChimeraEngineNormal()
    eng.run_simulation()
