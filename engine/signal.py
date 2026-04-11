from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config.paths import DATA_DIR, NEWS_DAILY_FEATURES_PATH, PRIMARY_TRADELOG, REGIME_TRACE_PATH, REPO_ROOT, WEEKLY_TRACE_PATH
from models.regime.context import score_news_regime_context
from models.regime.probabilistic import RegimeFeatures, RegimeProbabilities, infer_regime_probabilities

PROJECT_ROOT = str(REPO_ROOT)
LEGACY_TRADELOGS = [
    os.path.join(PROJECT_ROOT, 'data', 'tradelog_normal.csv'),
    os.path.join(PROJECT_ROOT, 'data', 'chimera_blackbox_final.csv'),
]

CONFIG = {
    'CAPITAL': 1_000_000,
    'MAX_LEVERAGE': 1.00,
    'LOOKBACK_FIP': 60,
    'LOOKBACK_STRUCTURE': 60,
    'LOOKBACK_MOM_FAST': 20,
    'LOOKBACK_MOM_SLOW': 60,
    'LOOKBACK_VOL': 20,
    'LOOKBACK_BETA': 60,
    'LOOKBACK_BREADTH': 200,
    'LOOKBACK_TREND': 100,
    'LOOKBACK_CONF': 60,
    'MIN_HISTORY': 220,
    'MIN_PRICE': 50.0,
    'MIN_ADV20': 200_000.0,
    'LONG_NAMES': 20,
    'SHORT_NAMES': 5,
    'LONG_GROSS_BULL': 1.00,
    'SHORT_GROSS_BULL': 0.00,
    'LONG_GROSS_CHOP': 0.70,
    'SHORT_GROSS_CHOP': 0.10,
    'LONG_GROSS_BEAR': 0.40,
    'SHORT_GROSS_BEAR': 0.50,
    # Breadth calibrated for 2200+ stock universe (avg ~0.43, healthy bull ~0.55+)
    'BREADTH_BULL': 0.40,
    'BREADTH_BEAR': 0.25,
    'VIX_LOW': 18.0,
    'VIX_HIGH': 28.0,
    'BROAD_TICKER_CANDIDATES': ['nifty50', '^NSEI', 'nifty_50'],
    'RISK_TICKER_CANDIDATES': ['banknifty', '^NSEBANK', '^CNXBANK'],
    'FEAR_TICKER_CANDIDATES': ['india_vix', 'vix', '^INDIAVIX', '^NSEI_VIX'],
    'USDINR_CANDIDATES': ['usd_inr', 'usdinr', 'inr=x', 'inrusd'],
    'US10Y_CANDIDATES': ['us10y', '10y_usdbond_rates', '^tnx', 'tnx'],
    'GOLD_CANDIDATES': ['goldbees', 'gold', 'gold_bees', 'gc=f'],
}


@dataclass
class AssetSnapshot:
    ticker: str
    close: np.ndarray
    open: np.ndarray
    volume: np.ndarray
    ret: np.ndarray
    mom20: np.ndarray
    mom60: np.ndarray
    mom20_z: np.ndarray
    mom60_z: np.ndarray
    vol20: np.ndarray
    vol20_z: np.ndarray
    adv20: np.ndarray
    beta: np.ndarray
    above200: np.ndarray
    structure: np.ndarray
    fip: np.ndarray
    fip_z: np.ndarray
    rsi14: np.ndarray
    mom5: np.ndarray


def _safe_name(name: str) -> str:
    return str(name).replace(' ', '_').replace('-', '_').lower()


def _rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(10, w // 3)).mean()


def _rolling_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(10, w // 3)).std()


def _zscore(s: pd.Series, w: int) -> pd.Series:
    mu = _rolling_mean(s, w)
    sd = _rolling_std(s, w).replace(0, np.nan)
    return (s - mu) / sd


def _read_price_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if 'Date' not in df.columns and 'date' not in df.columns:
        raise ValueError('missing Date column')
    if 'Close' not in df.columns and 'close' not in df.columns:
        raise ValueError('missing Close column')
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    ren = {}
    for src, dst in [('close', 'Close'), ('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('volume', 'Volume')]:
        if src in df.columns and dst not in df.columns:
            ren[src] = dst
    if ren:
        df = df.rename(columns=ren)
    return df


def _read_price_frame_from_download(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if 'Close' not in frame.columns and 'Adj Close' in frame.columns:
        frame['Close'] = frame['Adj Close']
    if 'Date' not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: 'Date'})
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame = frame.set_index('Date').sort_index()
    frame = frame[~frame.index.duplicated(keep='last')]
    return frame


def _resolve_key(data_map: Dict[str, pd.DataFrame], candidates: List[str]) -> Optional[str]:
    lookup = {_safe_name(k): k for k in data_map}
    for cand in candidates:
        key = lookup.get(_safe_name(cand))
        if key is not None:
            return key
    return None


def _trend_slope(series: pd.Series, lookback: int = 100) -> pd.Series:
    logp = np.log(series.astype(float).replace(0, np.nan))
    return (logp - logp.shift(lookback)) / float(lookback)


def _efficiency_ratio(close: pd.Series, lookback: int = 100) -> pd.Series:
    change = (close - close.shift(lookback)).abs()
    noise = close.diff().abs().rolling(lookback, min_periods=max(20, lookback // 3)).sum()
    return (change / (noise + 1e-9)).clip(0, 1)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI with EWM smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _ewm_fip(close: pd.Series, lambda_: float = 0.97, lookback: int = 60) -> pd.Series:
    """FIP signal: measures continuity-weighted momentum with vol normalization.
    Returns unbounded z-score-like values (NOT clipped to [-100,100]).
    """
    ret = close.pct_change()
    mp = max(20, lookback // 3)
    up = ret.clip(lower=0).ewm(alpha=1 - lambda_, adjust=False, min_periods=mp).mean()
    down = (-ret.clip(upper=0)).ewm(alpha=1 - lambda_, adjust=False, min_periods=mp).mean()
    total = up + down + 1e-9
    # Continuity ratio: fraction of move that is up vs down (-1 to +1)
    continuity = (up - down) / total
    # Trend magnitude (log return over lookback)
    trend = close.pct_change(lookback).fillna(0.0)
    # Vol-normalize by rolling std (NOT downside-only, avoids explosion)
    vol = ret.rolling(lookback, min_periods=mp).std().fillna(ret.std()) + 1e-6
    # Combine: continuity × trend / vol → spread
    raw = continuity * trend / vol
    return raw.fillna(0.0)


def _rolling_structure(close: pd.Series, lookback: int = 60) -> pd.Series:
    hist_mean = close.shift(1).rolling(lookback, min_periods=max(20, lookback // 3)).mean()
    hist_std = close.shift(1).rolling(lookback, min_periods=max(20, lookback // 3)).std().replace(0, np.nan)
    z = (close - hist_mean) / hist_std
    out = pd.Series(np.where(z.abs() >= 1.0, 'VACUUM', 'TRAPPED'), index=close.index)
    out[z.isna()] = 'UNKNOWN'
    return out


class ChimeraEngineNormal:
    def __init__(self):
        print('--- CHIMERA NORMAL: CAUSAL LONG/SHORT REBALANCER ---')
        self.trade_log: List[dict] = []
        self.weekly_returns: List[dict] = []
        self.regime_trace: List[dict] = []
        self.news_context: Optional[pd.DataFrame] = None
        self.prev_regime_probs: Optional[RegimeProbabilities] = None

    def load_news_context(self, calendar_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        if not os.path.exists(NEWS_DAILY_FEATURES_PATH):
            return None
        try:
            if str(NEWS_DAILY_FEATURES_PATH).endswith('.csv'):
                context = pd.read_csv(NEWS_DAILY_FEATURES_PATH)
            else:
                context = pd.read_parquet(NEWS_DAILY_FEATURES_PATH)
        except Exception as exc:
            print(f'Error loading news context: {exc}')
            return None

        if context.empty or 'date' not in context.columns:
            return None

        context = context.copy()
        context['date'] = pd.to_datetime(context['date']).dt.normalize()
        context = context.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        context = context.set_index('date').reindex(pd.to_datetime(calendar_index).normalize()).ffill()
        context.index.name = 'date'
        return context

    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        data_map: Dict[str, pd.DataFrame] = {}
        for subdir in ['stocks', 'indices', 'macro']:
            folder = os.path.join(DATA_DIR, subdir)
            if not os.path.isdir(folder):
                continue
            for path in sorted(glob.glob(os.path.join(folder, '*.parquet'))):
                ticker = os.path.splitext(os.path.basename(path))[0]
                try:
                    data_map[ticker] = _read_price_frame(path)
                except Exception as exc:
                    print(f'Error loading {ticker}: {exc}')
        print(f'Loaded {len(data_map)} instruments from local data')
        return data_map

    def load_indices(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        missing = {
            'BROAD': _resolve_key(data_map, CONFIG['BROAD_TICKER_CANDIDATES']),
            'RISK': _resolve_key(data_map, CONFIG['RISK_TICKER_CANDIDATES']),
            'FEAR': _resolve_key(data_map, CONFIG['FEAR_TICKER_CANDIDATES']),
        }
        if all(v is not None for v in missing.values()):
            return data_map
        fallback_map = {'BROAD': '^NSEI', 'RISK': '^CNXBANK', 'FEAR': '^INDIAVIX'}
        yf_tickers = [fallback_map[k] for k, v in missing.items() if v is None]
        if not yf_tickers:
            return data_map
        try:
            idx_data = yf.download(yf_tickers, start='2019-01-01', progress=False, group_by='ticker', auto_adjust=True)
            for role, key in missing.items():
                if key is not None:
                    continue
                yf_name = fallback_map[role]
                if isinstance(idx_data.columns, pd.MultiIndex) and yf_name in idx_data.columns.get_level_values(0):
                    frame = idx_data[yf_name].dropna(how='all')
                else:
                    frame = idx_data.copy()
                if frame.empty:
                    continue
                frame = frame.reset_index()
                if 'Date' not in frame.columns:
                    frame = frame.rename(columns={frame.columns[0]: 'Date'})
                data_map[yf_name] = _read_price_frame_from_download(frame)
        except Exception as exc:
            print(f'Error downloading fallback indices: {exc}')
        return data_map

    def _build_asset_cache(self, data_map: Dict[str, pd.DataFrame], calendar_index: pd.DatetimeIndex) -> List[AssetSnapshot]:
        broad_key = _resolve_key(data_map, CONFIG['BROAD_TICKER_CANDIDATES'])
        if broad_key is None:
            raise ValueError('Missing broad market reference series')
        broad_close = data_map[broad_key]['Close'].reindex(calendar_index).astype(float)
        broad_ret = broad_close.pct_change()

        excluded = {
            _safe_name(k)
            for k in CONFIG['BROAD_TICKER_CANDIDATES'] + CONFIG['RISK_TICKER_CANDIDATES'] + CONFIG['FEAR_TICKER_CANDIDATES'] + CONFIG['USDINR_CANDIDATES'] + CONFIG['US10Y_CANDIDATES'] + CONFIG['GOLD_CANDIDATES']
        }

        cache: List[AssetSnapshot] = []
        for ticker, df in data_map.items():
            if _safe_name(ticker) in excluded or 'Close' not in df.columns:
                continue
            aligned = df.reindex(calendar_index)
            close = aligned['Close'].astype(float)
            open_price = aligned['Open'].astype(float) if 'Open' in aligned.columns else close.copy()
            volume = aligned['Volume'].astype(float) if 'Volume' in aligned.columns else pd.Series(index=calendar_index, dtype=float)
            ret = close.pct_change()
            mom20 = close.pct_change(CONFIG['LOOKBACK_MOM_FAST'])
            mom60 = close.pct_change(CONFIG['LOOKBACK_MOM_SLOW'])
            vol20 = ret.rolling(CONFIG['LOOKBACK_VOL'], min_periods=max(10, CONFIG['LOOKBACK_VOL'] // 2)).std() * np.sqrt(252)
            adv20 = volume.rolling(20, min_periods=10).mean()
            above200 = (close > close.rolling(CONFIG['LOOKBACK_BREADTH'], min_periods=100).mean()).astype(float)
            structure = _rolling_structure(close, CONFIG['LOOKBACK_STRUCTURE'])
            fip = _ewm_fip(close, lambda_=0.97, lookback=CONFIG['LOOKBACK_FIP'])
            beta = ret.rolling(CONFIG['LOOKBACK_BETA'], min_periods=max(20, CONFIG['LOOKBACK_BETA'] // 2)).cov(broad_ret) / broad_ret.rolling(CONFIG['LOOKBACK_BETA'], min_periods=max(20, CONFIG['LOOKBACK_BETA'] // 2)).var()
            rsi14 = _rsi(close, 14)
            mom5 = close.pct_change(5)

            cache.append(AssetSnapshot(
                ticker=ticker,
                close=close.to_numpy(dtype=float),
                open=open_price.to_numpy(dtype=float),
                volume=volume.to_numpy(dtype=float),
                ret=ret.to_numpy(dtype=float),
                mom20=mom20.to_numpy(dtype=float),
                mom60=mom60.to_numpy(dtype=float),
                mom20_z=_z(mom20, CONFIG['LOOKBACK_CONF']).to_numpy(dtype=float),
                mom60_z=_z(mom60, CONFIG['LOOKBACK_CONF']).to_numpy(dtype=float),
                vol20=vol20.to_numpy(dtype=float),
                vol20_z=_z(vol20, CONFIG['LOOKBACK_CONF']).to_numpy(dtype=float),
                adv20=adv20.to_numpy(dtype=float),
                beta=beta.to_numpy(dtype=float),
                above200=above200.to_numpy(dtype=float),
                structure=structure.to_numpy(dtype=object),
                fip=fip.to_numpy(dtype=float),
                fip_z=_z(fip, CONFIG['LOOKBACK_CONF']).to_numpy(dtype=float),
                rsi14=rsi14.to_numpy(dtype=float),
                mom5=mom5.to_numpy(dtype=float),
            ))
        return cache

    def _build_market_frame(self, data_map: Dict[str, pd.DataFrame], calendar_index: pd.DatetimeIndex):
        def prep(cands: List[str]) -> Optional[pd.DataFrame]:
            key = _resolve_key(data_map, cands)
            if key is None:
                return None
            frame = data_map[key].reindex(calendar_index).copy()
            if 'Close' not in frame.columns:
                return None
            frame['Close'] = frame['Close'].astype(float)
            frame['RET1'] = frame['Close'].pct_change()
            frame['SMA50'] = frame['Close'].rolling(50, min_periods=25).mean()
            frame['SMA200'] = frame['Close'].rolling(200, min_periods=100).mean()
            frame['VOL20'] = frame['RET1'].rolling(20, min_periods=10).std() * np.sqrt(252)
            return frame

        broad = prep(CONFIG['BROAD_TICKER_CANDIDATES'])
        risk = prep(CONFIG['RISK_TICKER_CANDIDATES'])
        fear = prep(CONFIG['FEAR_TICKER_CANDIDATES'])
        usd = prep(CONFIG['USDINR_CANDIDATES'])
        us10y = prep(CONFIG['US10Y_CANDIDATES'])
        gold = prep(CONFIG['GOLD_CANDIDATES'])
        if broad is None:
            raise ValueError('Missing broad market reference series')
        broad['EFF100'] = _efficiency_ratio(broad['Close'], CONFIG['LOOKBACK_TREND'])
        broad['SLOPE100'] = _trend_slope(broad['Close'], CONFIG['LOOKBACK_TREND'])
        broad['MOM20_Z'] = _z(broad['RET1'].rolling(20, min_periods=10).mean(), CONFIG['LOOKBACK_CONF'])
        broad['VOL_Z'] = _z(broad['VOL20'], CONFIG['LOOKBACK_CONF'])
        return broad, risk, fear, usd, us10y, gold

    def _classify_regime(self, idx: int, broad, risk, fear, usd, us10y, gold, breadth: float) -> Tuple[str, float, float, float, str, float, float, float, str, float, float, float, float]:
        price = float(broad['Close'].iloc[idx])
        sma200 = float(broad['SMA200'].iloc[idx]) if pd.notna(broad['SMA200'].iloc[idx]) else np.nan
        eff = float(broad['EFF100'].iloc[idx]) if pd.notna(broad['EFF100'].iloc[idx]) else np.nan
        slope = float(broad['SLOPE100'].iloc[idx]) if pd.notna(broad['SLOPE100'].iloc[idx]) else np.nan
        momz = float(broad['MOM20_Z'].iloc[idx]) if pd.notna(broad['MOM20_Z'].iloc[idx]) else np.nan
        vix = float(fear['Close'].iloc[idx]) if fear is not None and pd.notna(fear['Close'].iloc[idx]) else np.nan
        news_row = None if self.news_context is None or idx >= len(self.news_context) else self.news_context.iloc[idx]
        news_context = score_news_regime_context(news_row)

        risk_ok = True
        if risk is not None and pd.notna(risk['SMA50'].iloc[idx]) and pd.notna(risk['Close'].iloc[idx]):
            risk_ok = bool(float(risk['Close'].iloc[idx]) >= float(risk['SMA50'].iloc[idx]))

        macro_score, macro_count = 0.0, 0
        for frame, sign in [(usd, -1.0), (us10y, -1.0), (gold, 1.0)]:
            if frame is None:
                continue
            v = frame['RET1'].rolling(20, min_periods=10).mean().iloc[idx]
            if pd.notna(v):
                macro_score += sign * float(v)
                macro_count += 1
        if fear is not None and pd.notna(vix):
            macro_score += -0.02 * (vix - 20.0)
            macro_count += 1
        macro_score = macro_score / macro_count if macro_count else 0.0

        macro_score += 0.15 * news_context.regime_bias

        if any(pd.isna(x) for x in [sma200, eff, slope, momz]):
            probs = infer_regime_probabilities(
                RegimeFeatures(
                    above_sma=0.0,
                    trend_score=0.0,
                    breadth_score=0.0,
                    risk_score=0.0,
                    macro_score=macro_score,
                    news_bias=news_context.regime_bias,
                    suppression_score=news_context.suppression_score,
                ),
                previous=self.prev_regime_probs,
            )
            self.prev_regime_probs = probs
            return 'CHOP', breadth, vix, macro_score, 'Insufficient history', probs.confidence, news_context.regime_bias, news_context.suppression_score, news_context.reason, probs.bull, probs.chop, probs.bear, probs.transition_risk

        # PRIMARY: Nifty above/below SMA200 is the dominant signal
        above_sma = 1.0 if price >= sma200 else -1.0
        trend_score = 1.50 * above_sma
        trend_score += 0.40 * np.tanh(float(slope) * 20.0)
        trend_score += 0.30 * np.tanh(float(eff) - 0.15)
        trend_score += 0.30 * np.tanh(float(momz) / 2.0)

        # Breadth centered on universe median (~0.43) not 0.50
        breadth_center = 0.43
        breadth_score = 0.0 if not np.isfinite(breadth) else (breadth - breadth_center) * 3.0

        risk_score = 0.0
        if np.isfinite(vix):
            risk_score += np.interp(vix, [10.0, 16.0, 22.0, 30.0, 45.0], [0.6, 0.3, 0.0, -0.5, -1.0])
        if not risk_ok:
            risk_score -= 0.3

        probs = infer_regime_probabilities(
            RegimeFeatures(
                above_sma=above_sma,
                trend_score=trend_score,
                breadth_score=breadth_score,
                risk_score=risk_score,
                macro_score=macro_score,
                news_bias=news_context.regime_bias,
                suppression_score=news_context.suppression_score,
            ),
            previous=self.prev_regime_probs,
        )
        self.prev_regime_probs = probs
        confidence = float(np.clip(probs.confidence + news_context.confidence_delta, 0.0, 1.0))

        if probs.selected_regime == 'BULL' and breadth >= CONFIG['BREADTH_BULL']:
            reason = f'Probabilistic bull regime | {news_context.reason}'
            return 'BULL', breadth, vix, macro_score, reason, confidence, news_context.regime_bias, news_context.suppression_score, news_context.reason, probs.bull, probs.chop, probs.bear, probs.transition_risk
        if probs.selected_regime == 'BEAR' and (breadth < CONFIG['BREADTH_BEAR'] or (np.isfinite(vix) and vix >= CONFIG['VIX_HIGH'])):
            reason = f'Probabilistic bear regime | {news_context.reason}'
            return 'BEAR', breadth, vix, macro_score, reason, confidence, news_context.regime_bias, news_context.suppression_score, news_context.reason, probs.bull, probs.chop, probs.bear, probs.transition_risk
        reason = f'Probabilistic mixed/chop regime | {news_context.reason}'
        return 'CHOP', breadth, vix, macro_score, reason, confidence, news_context.regime_bias, news_context.suppression_score, news_context.reason, probs.bull, probs.chop, probs.bear, probs.transition_risk

    def _score_universe(self, asset_cache: List[AssetSnapshot], signal_idx: int) -> pd.DataFrame:
        rows = []
        for a in asset_cache:
            if signal_idx >= len(a.close):
                continue
            close = a.close[signal_idx]
            openp = a.open[signal_idx]
            adv20 = a.adv20[signal_idx]
            if not np.isfinite(close) or not np.isfinite(openp):
                continue
            if not np.isfinite(adv20) or adv20 < CONFIG['MIN_ADV20'] or close < CONFIG['MIN_PRICE']:
                continue
            row = {
                'ticker': a.ticker,
                'idx': signal_idx,
                'close': float(close),
                'open_price': float(openp),
                'fip': float(a.fip[signal_idx]) if np.isfinite(a.fip[signal_idx]) else np.nan,
                'fip_z': float(a.fip_z[signal_idx]) if np.isfinite(a.fip_z[signal_idx]) else np.nan,
                'mom20': float(a.mom20[signal_idx]) if np.isfinite(a.mom20[signal_idx]) else np.nan,
                'mom60': float(a.mom60[signal_idx]) if np.isfinite(a.mom60[signal_idx]) else np.nan,
                'mom20_z': float(a.mom20_z[signal_idx]) if np.isfinite(a.mom20_z[signal_idx]) else np.nan,
                'mom60_z': float(a.mom60_z[signal_idx]) if np.isfinite(a.mom60_z[signal_idx]) else np.nan,
                'vol20': float(a.vol20[signal_idx]) if np.isfinite(a.vol20[signal_idx]) else np.nan,
                'vol20_z': float(a.vol20_z[signal_idx]) if np.isfinite(a.vol20_z[signal_idx]) else np.nan,
                'adv20': float(adv20),
                'beta': float(a.beta[signal_idx]) if np.isfinite(a.beta[signal_idx]) else np.nan,
                'above200': float(a.above200[signal_idx]) if np.isfinite(a.above200[signal_idx]) else np.nan,
                'structure': str(a.structure[signal_idx]),
                'rsi14': float(a.rsi14[signal_idx]) if np.isfinite(a.rsi14[signal_idx]) else np.nan,
                'mom5': float(a.mom5[signal_idx]) if np.isfinite(a.mom5[signal_idx]) else np.nan,
            }
            if not np.isfinite(row['fip']) or not np.isfinite(row['mom60']) or not np.isfinite(row['vol20']):
                continue
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['structure_score'] = np.where(
            (df['structure'] == 'VACUUM') & (df['mom20'] > 0) & (df['above200'] >= 0.5), 1.0,
            np.where(df['structure'] == 'VACUUM', 0.7, 0.35)
        )
        # Cross-sectional z-score of momentum (better than raw rank for dispersion)
        for col in ['mom20', 'mom60']:
            mu, sd = df[col].mean(), df[col].std()
            df[f'cs_z_{col}'] = (df[col] - mu) / (sd + 1e-9)
        for col in ['fip', 'fip_z', 'mom20', 'mom60', 'cs_z_mom20', 'cs_z_mom60', 'mom20_z', 'mom60_z', 'vol20', 'vol20_z', 'beta', 'adv20', 'structure_score', 'rsi14']:
            df[f'rank_{col}'] = df[col].rank(pct=True, method='average')

        # RSI-based overbought/oversold penalty (key for momentum crash protection)
        # Penalize longs with RSI > 75 (overbought → likely to mean-revert)
        # Penalize shorts with RSI < 25 (oversold → likely to bounce)
        df['rsi_penalty_long'] = np.where(df['rsi14'] > 75, -0.15, np.where(df['rsi14'] > 70, -0.05, 0.0))
        df['rsi_penalty_short'] = np.where(df['rsi14'] < 25, -0.15, np.where(df['rsi14'] < 30, -0.05, 0.0))

        # Short-term momentum confirmation: penalize falling knives
        df['mom5_penalty'] = np.where(df['mom5'] < -0.08, -0.10, np.where(df['mom5'] < -0.04, -0.05, 0.0))

        # LONG: favor momentum + low vol + above SMA200 + RSI filter
        df['long_score'] = (
            0.28 * df['rank_cs_z_mom60'].fillna(df['rank_mom60']) +
            0.18 * df['rank_cs_z_mom20'].fillna(df['rank_mom20']) +
            0.10 * df['rank_fip_z'].fillna(df['rank_fip']) +
            0.10 * df['rank_structure_score'] +
            0.12 * (1.0 - df['rank_vol20_z'].fillna(df['rank_vol20'])) +
            0.06 * (1.0 - df['rank_beta'].fillna(0.5)) +
            0.06 * df['rank_adv20'] +
            0.10 * (1.0 - df['rank_rsi14'].fillna(0.5)) +  # Prefer stocks NOT overbought
            df['rsi_penalty_long'] +
            df['mom5_penalty']
        )
        # SHORT: worst momentum + below SMA200 + high vol + high beta
        df['short_score'] = (
            0.28 * (1.0 - df['rank_cs_z_mom60'].fillna(df['rank_mom60'])) +
            0.18 * (1.0 - df['rank_cs_z_mom20'].fillna(df['rank_mom20'])) +
            0.10 * (1.0 - df['rank_fip_z'].fillna(df['rank_fip'])) +
            0.10 * (1.0 - df['rank_structure_score']) +
            0.14 * df['rank_vol20_z'].fillna(df['rank_vol20']) +
            0.10 * df['rank_beta'].fillna(0.5) +
            0.05 * (1.0 - df['rank_adv20']) +
            0.05 * df['rank_rsi14'].fillna(0.5) +  # Prefer to short overbought stocks
            df['rsi_penalty_short']
        )
        return df.sort_values('long_score', ascending=False).reset_index(drop=True)

    def _allocate_gross_budget(self, regime: str, long_count: int, short_count: int, confidence: float) -> Tuple[float, float]:
        if regime == 'BULL':
            long_gross = CONFIG['LONG_GROSS_BULL']
            short_gross = CONFIG['SHORT_GROSS_BULL'] if short_count > 0 else 0.0
        elif regime == 'BEAR':
            long_gross = CONFIG['LONG_GROSS_BEAR'] if long_count > 0 else 0.0
            short_gross = CONFIG['SHORT_GROSS_BEAR'] if short_count > 0 else 0.0
        else:
            long_gross = CONFIG['LONG_GROSS_CHOP'] if long_count > 0 else 0.0
            short_gross = CONFIG['SHORT_GROSS_CHOP'] if short_count > 0 else 0.0
        scale = 0.75 + 0.50 * float(np.clip(confidence, 0.0, 1.0))
        return long_gross * scale, short_gross * scale

    def _write_trade_logs(self, log_df: pd.DataFrame):
        os.makedirs(os.path.dirname(PRIMARY_TRADELOG), exist_ok=True)
        for path in [PRIMARY_TRADELOG] + LEGACY_TRADELOGS:
            log_df.to_csv(path, index=False)
        print(f'--- SAVED: {PRIMARY_TRADELOG} ---')

    def run_simulation(self):
        print('Loading data...')
        self.trade_log = []
        self.weekly_returns = []
        self.regime_trace = []
        self.prev_regime_probs = None

        data_map = self.load_all_stocks()
        data_map = self.load_indices(data_map)
        broad_key = _resolve_key(data_map, CONFIG['BROAD_TICKER_CANDIDATES'])
        if broad_key is None:
            print('No calendar data!')
            return

        calendar = data_map[broad_key].copy().sort_index()
        dates = calendar.index
        rebalance_dates = dates[dates.weekday == 4]
        if len(rebalance_dates) <= 250:
            print('Not enough rebalance dates to run simulation.')
            return

        start_idx = max(CONFIG['MIN_HISTORY'], CONFIG['LOOKBACK_TREND'] + 5)
        start_rebalance_idx = int(np.searchsorted(rebalance_dates, dates[start_idx], side='left'))
        if start_rebalance_idx >= len(rebalance_dates) - 1:
            print('Not enough post-warmup rebalance dates.')
            return

        asset_cache = self._build_asset_cache(data_map, dates)
        broad, risk, fear, usd, us10y, gold = self._build_market_frame(data_map, dates)
        self.news_context = self.load_news_context(dates)
        print(f'Calendar: {len(calendar)} rows')
        print(f'Universe: {len(asset_cache)} stocks')
        print(f'News context: {"loaded" if self.news_context is not None else "not available"}')

        asset_lookup = {a.ticker: a for a in asset_cache}

        for rebalance_idx in range(start_rebalance_idx, len(rebalance_dates) - 1):
            current_date = rebalance_dates[rebalance_idx]
            next_date = rebalance_dates[rebalance_idx + 1]
            current_idx = int(dates.get_loc(current_date))
            next_idx = int(dates.get_loc(next_date))
            signal_idx = current_idx - 1
            if signal_idx < 0:
                continue

            breadth_vals = [a.above200[signal_idx] for a in asset_cache if np.isfinite(a.above200[signal_idx])]
            breadth = float(np.nanmean(breadth_vals)) if breadth_vals else np.nan
            regime, breadth, vix, macro_score, regime_reason, confidence, news_bias, suppression_score, news_reason, p_bull, p_chop, p_bear, transition_risk = self._classify_regime(signal_idx, broad, risk, fear, usd, us10y, gold, breadth)
            self.regime_trace.append({
                'date': current_date,
                'signal_date': dates[signal_idx],
                'regime': regime,
                'regime_reason': regime_reason,
                'regime_confidence': confidence,
                'p_bull': p_bull,
                'p_chop': p_chop,
                'p_bear': p_bear,
                'transition_risk': transition_risk,
                'breadth': breadth,
                'vix': vix,
                'macro_score': macro_score,
                'news_bias': news_bias,
                'suppression_score': suppression_score,
                'news_reason': news_reason,
            })

            scored = self._score_universe(asset_cache, signal_idx)
            if scored.empty:
                self.trade_log.append({
                    'date': current_date, 'signal_date': dates[signal_idx], 'exit_date': next_date, 'ticker': 'CASH',
                    'close': 1.0, 'entry_price': 1.0, 'weight': 0.0, 'gross_weight': 0.0,
                    'kinetic_energy': 0.0, 'efficiency': 0.0, 'leverage_mult': 0.0,
                    'market_state': regime, 'regime_confidence': confidence, 'regime_score': p_bull - p_bear,
                    'decision_reason': f'No qualified universe ({regime_reason})',
                    'nifty_vol': vix / 100 if np.isfinite(vix) else np.nan,
                    'universe_size': len(asset_cache), 'fwd_return': 0.0, 'net_pnl': 0.0,
                    'structure_tag': 'NONE', 'side': 'CASH', 'score': 0.0,
                    'long_score': 0.0, 'short_score': 0.0, 'breadth': breadth, 'macro_score': macro_score,
                    'mom_z': np.nan, 'break_score': np.nan, 'news_bias': news_bias, 'suppression_score': suppression_score,
                    'p_bull': p_bull, 'p_chop': p_chop, 'p_bear': p_bear, 'transition_risk': transition_risk,
                })
                continue

            # LONG POOL: positive 60-day momentum + above median score + not crashing short-term
            long_pool = scored[
                (scored['mom60'].notna()) &
                (scored['mom60'] > 0) &
                (scored['long_score'] >= scored['long_score'].median()) &
                (scored['mom5'] > -0.10)  # Skip stocks that crashed >10% in last week
            ].copy()
            # SHORT POOL: much stricter filters — only short confirmed losers
            short_pool = scored[
                (scored['mom60'] < 0) &  # Must have negative 60-day momentum
                (scored['above200'] < 0.5) &  # Must be BELOW its SMA200
                (scored['short_score'] >= scored['short_score'].quantile(0.80))
            ].copy()
            # In BULL regime, no shorts at all
            if regime == 'BULL':
                short_pool = pd.DataFrame()
            elif regime == 'CHOP':
                short_pool = short_pool[short_pool['short_score'] >= short_pool['short_score'].quantile(0.90)] if not short_pool.empty else short_pool
            if suppression_score >= 0.60 or transition_risk >= 0.75:
                short_pool = pd.DataFrame()
                long_pool = long_pool.head(max(5, CONFIG['LONG_NAMES'] // 2)).copy()

            top_longs = long_pool.head(CONFIG['LONG_NAMES']).copy()
            if not top_longs.empty and not short_pool.empty:
                short_pool = short_pool[~short_pool['ticker'].isin(top_longs['ticker'])]
            top_shorts = short_pool.head(CONFIG['SHORT_NAMES']).copy()

            if top_longs.empty and top_shorts.empty:
                self.trade_log.append({
                    'date': current_date, 'signal_date': dates[signal_idx], 'exit_date': next_date, 'ticker': 'CASH',
                    'close': 1.0, 'entry_price': 1.0, 'weight': 0.0, 'gross_weight': 0.0,
                    'kinetic_energy': 0.0, 'efficiency': 0.0, 'leverage_mult': 0.0,
                    'market_state': regime, 'regime_confidence': confidence, 'regime_score': p_bull - p_bear,
                    'decision_reason': f'No Signal ({regime_reason})',
                    'nifty_vol': vix / 100 if np.isfinite(vix) else np.nan,
                    'universe_size': len(asset_cache), 'fwd_return': 0.0, 'net_pnl': 0.0,
                    'structure_tag': 'NONE', 'side': 'CASH', 'score': 0.0,
                    'long_score': 0.0, 'short_score': 0.0, 'breadth': breadth, 'macro_score': macro_score,
                    'mom_z': np.nan, 'break_score': np.nan, 'news_bias': news_bias, 'suppression_score': suppression_score,
                    'p_bull': p_bull, 'p_chop': p_chop, 'p_bear': p_bear, 'transition_risk': transition_risk,
                })
                continue

            long_budget, short_budget = self._allocate_gross_budget(regime, len(top_longs), len(top_shorts), confidence)
            if transition_risk >= 0.60:
                long_budget *= max(0.40, 1.0 - 0.60 * transition_risk)
                short_budget *= max(0.25, 1.0 - 0.80 * transition_risk)
            # Cap short budget — never more than 30% of long budget
            if short_budget > long_budget * 0.30:
                short_budget = long_budget * 0.30

            if not top_longs.empty:
                long_raw = (top_longs['long_score'] - top_longs['long_score'].min() + 1e-6).clip(lower=1e-6)
                long_raw = long_raw / long_raw.sum()
                inv_vol = 1.0 / top_longs['vol20'].replace(0, np.nan).fillna(top_longs['vol20'].median())
                inv_vol = inv_vol / inv_vol.sum()
                long_raw = 0.60 * long_raw + 0.40 * inv_vol
                long_raw = long_raw / long_raw.sum()
            else:
                long_raw = pd.Series(dtype=float)

            if not top_shorts.empty:
                short_raw = (top_shorts['short_score'] - top_shorts['short_score'].min() + 1e-6).clip(lower=1e-6)
                short_raw = short_raw / short_raw.sum()
                inv_vol = 1.0 / top_shorts['vol20'].replace(0, np.nan).fillna(top_shorts['vol20'].median())
                inv_vol = inv_vol / inv_vol.sum()
                short_raw = 0.55 * short_raw + 0.45 * inv_vol
                short_raw = short_raw / short_raw.sum()
            else:
                short_raw = pd.Series(dtype=float)

            final_weights = {}
            MAX_SINGLE_WEIGHT = 0.10  # Hard cap: no single stock > 10% of capital
            for i, (_, row) in enumerate(top_longs.iterrows()):
                base_w = float(long_budget * long_raw.iloc[i]) if len(long_raw) else 0.0
                w = base_w * (0.75 if np.isfinite(vix) and vix > 24 else 1.0)
                if row['structure'] == 'VACUUM' and row['fip'] > 0:
                    w *= 1.10
                w = min(w, MAX_SINGLE_WEIGHT)
                final_weights[row['ticker']] = {'side': 'LONG', 'weight': w, 'base_weight': base_w, 'score': float(row['long_score']), 'row': row}
            for i, (_, row) in enumerate(top_shorts.iterrows()):
                base_w = float(short_budget * short_raw.iloc[i]) if len(short_raw) else 0.0
                w = base_w * (0.85 if np.isfinite(vix) and vix > 24 else 1.0)
                if row['structure'] == 'TRAPPED' and np.isfinite(row['mom60']) and row['mom60'] < 0:
                    w *= 1.10
                w = min(w, MAX_SINGLE_WEIGHT)
                final_weights[row['ticker']] = {'side': 'SHORT', 'weight': -w, 'base_weight': base_w, 'score': float(row['short_score']), 'row': row}

            total_gross = float(sum(abs(v['weight']) for v in final_weights.values()))
            if total_gross > CONFIG['MAX_LEVERAGE'] and total_gross > 0:
                scale = CONFIG['MAX_LEVERAGE'] / total_gross
                for item in final_weights.values():
                    item['weight'] *= scale

            date_pnl = 0.0
            for ticker, item in final_weights.items():
                asset = asset_lookup[ticker]
                entry_open = asset.open[current_idx] if current_idx < len(asset.open) else np.nan
                exit_open = asset.open[next_idx] if next_idx < len(asset.open) else np.nan
                if not np.isfinite(entry_open) or not np.isfinite(exit_open) or entry_open <= 0:
                    raw_ret = 0.0
                    entry_open = float(item['row']['open_price'])
                else:
                    raw_ret = (exit_open / entry_open) - 1.0
                gross_weight = abs(item['weight'])
                net_pnl = CONFIG['CAPITAL'] * item['weight'] * raw_ret
                date_pnl += net_pnl
                row = item['row']
                self.trade_log.append({
                    'date': current_date,
                    'signal_date': dates[signal_idx],
                    'exit_date': next_date,
                    'ticker': ticker,
                    'close': float(item['row']['close']),
                    'entry_price': float(entry_open),
                    'weight': float(item['weight']),
                    'gross_weight': gross_weight,
                    'kinetic_energy': float(abs(row['fip'])),
                    'efficiency': float(broad['EFF100'].iloc[signal_idx]) if pd.notna(broad['EFF100'].iloc[signal_idx]) else np.nan,
                    'leverage_mult': gross_weight / max(1e-9, item['base_weight']),
                    'market_state': regime,
                    'regime_confidence': confidence,
                    'regime_score': p_bull - p_bear,
                    'decision_reason': regime_reason,
                    'nifty_vol': float(fear['Close'].iloc[signal_idx] / 100.0) if fear is not None and pd.notna(fear['Close'].iloc[signal_idx]) else np.nan,
                    'universe_size': len(asset_cache),
                    'fwd_return': float(raw_ret),
                    'net_pnl': float(net_pnl),
                    'structure_tag': item['row']['structure'],
                    'side': item['side'],
                    'score': float(item['score']),
                    'long_score': float(item['score']) if item['side'] == 'LONG' else np.nan,
                    'short_score': float(item['score']) if item['side'] == 'SHORT' else np.nan,
                    'breadth': breadth,
                    'macro_score': macro_score,
                    'news_bias': news_bias,
                    'suppression_score': suppression_score,
                    'p_bull': p_bull,
                    'p_chop': p_chop,
                    'p_bear': p_bear,
                    'transition_risk': transition_risk,
                    'mom_z': float(row['mom60_z']) if pd.notna(row['mom60_z']) else np.nan,
                    'break_score': np.nan,
                })

            self.weekly_returns.append({
                'date': current_date,
                'market_state': regime,
                'regime_confidence': confidence,
                'p_bull': p_bull,
                'p_chop': p_chop,
                'p_bear': p_bear,
                'transition_risk': transition_risk,
                'breadth': breadth,
                'vix': vix,
                'macro_score': macro_score,
                'news_bias': news_bias,
                'suppression_score': suppression_score,
                'portfolio_return': date_pnl / CONFIG['CAPITAL'],
                'gross_exposure': float(sum(abs(v['weight']) for v in final_weights.values())),
                'net_exposure': float(sum(v['weight'] for v in final_weights.values())),
            })

        if not self.trade_log:
            print('No trades generated.')
            return

        trades_df = pd.DataFrame(self.trade_log)
        weekly_df = pd.DataFrame(self.weekly_returns)
        regime_df = pd.DataFrame(self.regime_trace)
        os.makedirs(os.path.dirname(PRIMARY_TRADELOG), exist_ok=True)
        self._write_trade_logs(trades_df)
        regime_df.to_csv(REGIME_TRACE_PATH, index=False)
        weekly_df.to_csv(WEEKLY_TRACE_PATH, index=False)
        print(f'Active trades: {(trades_df["weight"].abs() > 1e-12).sum()} across {trades_df["date"].nunique()} rebalance dates')
        print(f'Gross exposure avg: {trades_df.groupby("date")["gross_weight"].sum().mean():.3f}')
        print(f'Net PnL sum: {trades_df["net_pnl"].sum():,.2f}')
        print(f'--- REGIME TRACE: {REGIME_TRACE_PATH} ---')
        print(f'--- WEEKLY RETURN TRACE: {WEEKLY_TRACE_PATH} ---')


def _z(series: pd.Series, window: int) -> pd.Series:
    return _zscore(series, window)
