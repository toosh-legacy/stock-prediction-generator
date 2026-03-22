import yfinance as yf
import pandas as pd
import json
from app.config import get_settings

settings = get_settings()


def _get_redis_client():
    """Returns Upstash Redis client in production, standard redis in local dev."""
    if settings.using_upstash:
        from upstash_redis import Redis
        return Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token
        )
    else:
        import redis
        return redis.from_url(settings.redis_url, decode_responses=True)


class DataFetcher:
    def __init__(self):
        self.redis = _get_redis_client()
        self.cache_ttl = 3600  # 1 hour

    def _cache_get(self, key: str):
        try:
            return self.redis.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: str, ttl: int):
        try:
            self.redis.setex(key, ttl, value)
        except Exception:
            pass  # Cache miss is acceptable — degrade gracefully

    async def get_ohlcv(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        cache_key = f"ohlcv:{ticker}:{period}:{start}:{end}"
        cached = self._cache_get(cache_key)
        if cached:
            return pd.read_json(cached)

        t = yf.Ticker(ticker)
        if start and end:
            df = t.history(start=start, end=end)
        else:
            df = t.history(period=period)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        df.index = df.index.tz_localize(None)
        df.columns = [c.lower() for c in df.columns]

        self._cache_set(cache_key, df.to_json(), self.cache_ttl)
        return df

    async def get_info(self, ticker: str) -> dict:
        cache_key = f"info:{ticker}"
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)

        t = yf.Ticker(ticker)
        info = t.info or {}
        self._cache_set(cache_key, json.dumps(info), 3600 * 24)
        return info
