"""
Macro feature fetcher.
Sources:
  1. FRED API — Fed funds rate, CPI, unemployment, yield curve spread
  2. Yahoo Finance — VIX, DXY, Gold, Oil, SPY, sector ETFs
All series are aligned to business-day frequency and forward-filled.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# FRED series IDs we care about
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "yield_spread_10y2y": "T10Y2Y",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
}

# Yahoo Finance macro proxies
YF_MACRO = {
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
    "oil": "CL=F",
    "spy": "SPY",
}

# Sector ETFs
SECTOR_ETFS = {
    "xlk": "XLK",   # Tech
    "xlf": "XLF",   # Financials
    "xle": "XLE",   # Energy
    "xlv": "XLV",   # Healthcare
    "xli": "XLI",   # Industrials
}


class MacroFetcher:
    def __init__(self, fred_api_key: str = ""):
        self.fred_api_key = fred_api_key
        self._cache: dict = {}
        self.cache_ttl = 3600  # 1 hour

    def _fred_available(self) -> bool:
        return bool(self.fred_api_key)

    def _get_fred_data(self, start: str, end: str) -> pd.DataFrame:
        if not self._fred_available():
            return pd.DataFrame()
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_api_key)
            frames = {}
            for name, series_id in FRED_SERIES.items():
                try:
                    s = fred.get_series(series_id, observation_start=start, observation_end=end)
                    frames[name] = s
                except Exception:
                    pass
            if not frames:
                return pd.DataFrame()
            df = pd.DataFrame(frames)
            df = df.resample("B").last().ffill()
            return df
        except Exception:
            return pd.DataFrame()

    def _get_yf_macro(self, start: str, end: str) -> pd.DataFrame:
        all_symbols = {**YF_MACRO, **SECTOR_ETFS}
        frames = []
        for name, symbol in all_symbols.items():
            try:
                hist = yf.Ticker(symbol).history(start=start, end=end)
                if not hist.empty:
                    s = hist["Close"].rename(name)
                    s.index = s.index.tz_localize(None)
                    frames.append(s)
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1).ffill()
        # Compute 1-day returns alongside levels
        ret_frames = []
        for col in df.columns:
            ret_frames.append(df[col].pct_change().rename(f"{col}_ret"))
        df = pd.concat([df] + ret_frames, axis=1)
        return df

    def get_macro_features(self, start: str, end: str) -> pd.DataFrame:
        """
        Returns merged FRED + YF macro feature DataFrame aligned to the given date range.
        Index is business-day DatetimeIndex.
        """
        fred_df = self._get_fred_data(start, end)
        yf_df = self._get_yf_macro(start, end)

        if fred_df.empty and yf_df.empty:
            return pd.DataFrame()
        if fred_df.empty:
            return yf_df
        if yf_df.empty:
            return fred_df

        merged = fred_df.join(yf_df, how="outer").ffill().bfill()
        return merged

    def get_latest_snapshot(self) -> dict:
        """Latest macro values as a flat dict for the API response / UI."""
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        df = self.get_macro_features(start, end)
        if df.empty:
            return {}
        latest = df.iloc[-1].dropna().to_dict()
        return {k: round(float(v), 4) for k, v in latest.items()}
