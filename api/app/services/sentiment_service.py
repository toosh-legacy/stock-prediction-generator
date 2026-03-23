"""
Sentiment analysis service.
Sources (all free):
  1. NewsAPI — structured news articles for the ticker
  2. Yahoo Finance RSS — live headline feed
  3. Finviz scraper — news table from finviz.com
All sources run VADER sentiment. Results cached 15 minutes.
Reddit support can be added when PRAW credentials are available.
"""
import asyncio
from datetime import datetime, timedelta

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class SentimentService:
    def __init__(self, news_api_key: str = ""):
        self.news_api_key = news_api_key
        self.vader = SentimentIntensityAnalyzer()
        self._cache: dict = {}
        self.cache_ttl = 900  # 15 minutes

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts, _ = self._cache[key]
        return (datetime.utcnow() - ts).total_seconds() < self.cache_ttl

    def _cache_set(self, key: str, value) -> None:
        self._cache[key] = (datetime.utcnow(), value)

    def _cache_get(self, key: str):
        return self._cache[key][1]

    # ------------------------------------------------------------------
    # VADER scoring
    # ------------------------------------------------------------------

    def _score(self, headlines: list[str]) -> dict:
        if not headlines:
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0, "count": 0}
        scores = [self.vader.polarity_scores(h) for h in headlines]
        n = len(scores)
        return {
            "compound": round(sum(s["compound"] for s in scores) / n, 4),
            "positive": round(sum(s["pos"] for s in scores) / n, 4),
            "negative": round(sum(s["neg"] for s in scores) / n, 4),
            "neutral": round(sum(s["neu"] for s in scores) / n, 4),
            "count": n,
        }

    # ------------------------------------------------------------------
    # Fetchers
    # ------------------------------------------------------------------

    async def _fetch_newsapi(self, ticker: str, company_name: str = "") -> list[str]:
        if not self.news_api_key:
            return []
        query = f'"{ticker}" stock' if not company_name else f'"{ticker}" OR "{company_name}" stock'
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 30,
            "from": (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "apiKey": self.news_api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://newsapi.org/v2/everything", params=params)
                articles = r.json().get("articles", [])
                return [
                    f"{a.get('title', '')} {a.get('description') or ''}".strip()
                    for a in articles
                    if a.get("title")
                ]
        except Exception:
            return []

    async def _fetch_yahoo_rss(self, ticker: str) -> list[str]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 StockSage/1.0"})
            if not BS4_AVAILABLE:
                return []
            soup = BeautifulSoup(r.text, "lxml-xml")
            return [
                item.find("title").text
                for item in soup.find_all("item")[:20]
                if item.find("title")
            ]
        except Exception:
            return []

    async def _fetch_finviz(self, ticker: str) -> list[str]:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 StockSage/1.0"})
            if not BS4_AVAILABLE:
                return []
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find("table", {"class": "fullview-news-outer"})
            if not table:
                return []
            return [a.text.strip() for a in table.find_all("a") if a.text.strip()][:20]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_sentiment(self, ticker: str, company_name: str = "") -> dict:
        """
        Fetch live sentiment for a ticker from all sources.
        Returns aggregated VADER scores + per-source breakdown.
        Cached 15 minutes.
        """
        cache_key = ticker.upper()
        if self._cache_valid(cache_key):
            return self._cache_get(cache_key)

        newsapi, yahoo, finviz = await asyncio.gather(
            self._fetch_newsapi(ticker, company_name),
            self._fetch_yahoo_rss(ticker),
            self._fetch_finviz(ticker),
        )

        all_headlines = newsapi + yahoo + finviz

        result = {
            "ticker": ticker.upper(),
            "sources": {
                "newsapi": self._score(newsapi),
                "yahoo_rss": self._score(yahoo),
                "finviz": self._score(finviz),
            },
            "aggregate": self._score(all_headlines),
            "total_articles": len(all_headlines),
            "sample_headlines": all_headlines[:5],
            "fetched_at": datetime.utcnow().isoformat(),
        }

        self._cache_set(cache_key, result)
        return result

    def sentiment_to_features(self, sentiment: dict) -> dict:
        """Convert a sentiment result dict into a flat feature dict for model input."""
        agg = sentiment.get("aggregate", {})
        return {
            "sentiment_compound": agg.get("compound", 0.0),
            "sentiment_positive": agg.get("positive", 0.0),
            "sentiment_negative": agg.get("negative", 0.0),
            "sentiment_article_count": float(agg.get("count", 0)),
        }
