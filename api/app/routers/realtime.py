"""
Real-time WebSocket endpoint.
Connects at: ws://host/v1/realtime/ws/{ticker}?interval=5m

Every 30 seconds sends:
{
  "type": "update",
  "ticker": "AAPL",
  "interval": "5m",
  "candles": [...],       // real OHLCV candles (last 100)
  "predicted": [...],     // predicted next 10 candles (different color on frontend)
  "sentiment": {...},
  "macro_snapshot": {...},
  "timestamp": "..."
}
"""
import asyncio
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

log = logging.getLogger(__name__)
router = APIRouter()

POLL_INTERVAL = 30   # seconds between updates
PRED_CANDLES = 10    # how many future candles to predict
HISTORY_CANDLES = 100  # how many historical candles to send


def _fetch_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame:
    """Fetch intraday OHLCV from yfinance. Free, no key needed."""
    period_map = {"1m": "1d", "5m": "5d", "15m": "60d", "1d": "2y"}
    period = period_map.get(interval, "5d")
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.tz_localize(None)
        return df.tail(HISTORY_CANDLES)
    except Exception as e:
        log.warning(f"yfinance intraday fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _df_to_candles(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "time": int(idx.timestamp()),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
            "volume": int(r["volume"]),
        }
        for idx, r in df.iterrows()
    ]


def _predict_next_candles(df: pd.DataFrame, n: int = PRED_CANDLES, interval: str = "5m") -> list[dict]:
    """
    Lightweight next-candle prediction using:
      - EMA trend extrapolation for close price direction
      - Historical volatility for realistic OHLC spread
      - Momentum (last 5 returns) weighted average
    No heavy model load — this runs every 30 seconds for the live chart.
    The offline-trained model is used for the longer-horizon dashboard prediction.
    """
    if len(df) < 20:
        return []

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # Estimate next close via exponentially weighted mean of recent returns
    returns = np.diff(closes[-20:]) / closes[-20:-1]
    weights = np.exp(np.linspace(0, 1, len(returns)))
    weights /= weights.sum()
    avg_return = float(np.dot(weights, returns))

    # Clamp to ±3% per candle to avoid runaway predictions
    avg_return = max(-0.03, min(0.03, avg_return))

    # Historical spread stats
    avg_high_spread = float(np.mean((highs[-20:] - closes[-20:]) / closes[-20:]))
    avg_low_spread = float(np.mean((closes[-20:] - lows[-20:]) / closes[-20:]))
    avg_high_spread = max(0.0005, avg_high_spread)
    avg_low_spread = max(0.0005, avg_low_spread)

    # Infer interval in seconds for timestamp stepping
    interval_seconds = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "1d": 86400,
    }.get(interval, 300)

    last_ts = int(df.index[-1].timestamp())
    last_close = float(closes[-1])

    predicted = []
    current_close = last_close
    for i in range(1, n + 1):
        # Add slight mean-reversion dampening for longer horizons
        dampening = 0.95 ** i
        step_return = avg_return * dampening
        next_close = round(current_close * (1 + step_return), 2)
        next_open = round(current_close, 2)  # gap-less open
        next_high = round(next_close * (1 + avg_high_spread), 2)
        next_low = round(next_close * (1 - avg_low_spread), 2)
        # Ensure OHLC consistency
        next_high = max(next_high, next_open, next_close)
        next_low = min(next_low, next_open, next_close)

        predicted.append({
            "time": last_ts + i * interval_seconds,
            "open": next_open,
            "high": next_high,
            "low": next_low,
            "close": next_close,
            "predicted": True,
        })
        current_close = next_close

    return predicted


async def _get_sentiment(ticker: str) -> dict:
    try:
        from app.config import get_settings
        from app.services.sentiment_service import SentimentService
        settings = get_settings()
        svc = SentimentService(news_api_key=settings.news_api_key)
        return await svc.get_sentiment(ticker)
    except Exception:
        return {}


def _get_macro_snapshot() -> dict:
    try:
        from app.config import get_settings
        from app.services.macro_fetcher import MacroFetcher
        settings = get_settings()
        return MacroFetcher(fred_api_key=settings.fred_api_key).get_latest_snapshot()
    except Exception:
        return {}


@router.websocket("/ws/{ticker}")
async def realtime_ws(
    websocket: WebSocket,
    ticker: str,
    interval: str = Query("5m", description="Candle interval: 1m, 5m, 15m, 1d"),
):
    """
    Stream live OHLCV candles + AI-predicted next candles for a given ticker.
    Reconnect-safe: client should retry on disconnect.
    """
    await websocket.accept()
    ticker = ticker.upper()
    log.info(f"WebSocket connected: {ticker} @ {interval}")

    # Fetch sentiment + macro once per connection (cached, so fast on repeat)
    sentiment_task = asyncio.create_task(_get_sentiment(ticker))
    macro_snapshot = _get_macro_snapshot()

    try:
        while True:
            df = _fetch_intraday(ticker, interval)

            if df.empty:
                await websocket.send_json({
                    "type": "error",
                    "message": f"No data available for {ticker}",
                })
                await asyncio.sleep(POLL_INTERVAL)
                continue

            candles = _df_to_candles(df)
            predicted = _predict_next_candles(df, n=PRED_CANDLES, interval=interval)

            # Resolve sentiment if ready
            sentiment = {}
            if sentiment_task.done():
                try:
                    sentiment = sentiment_task.result()
                except Exception:
                    pass

            payload = {
                "type": "update",
                "ticker": ticker,
                "interval": interval,
                "candles": candles,
                "predicted": predicted,
                "sentiment": sentiment.get("aggregate", {}),
                "macro_snapshot": macro_snapshot,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await websocket.send_json(payload)
            await asyncio.sleep(POLL_INTERVAL)

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: {ticker}")
    except Exception as e:
        log.error(f"WebSocket error for {ticker}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
