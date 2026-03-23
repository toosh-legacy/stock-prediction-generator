"""REST endpoint to fetch sentiment for a ticker (also available via WebSocket)."""
from fastapi import APIRouter, Depends, HTTPException
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.config import get_settings
from app.services.sentiment_service import SentimentService
from app.services.macro_fetcher import MacroFetcher

router = APIRouter()
settings = get_settings()
_sentiment_svc = SentimentService(news_api_key=settings.news_api_key)
_macro_fetcher = MacroFetcher(fred_api_key=settings.fred_api_key)


@router.get("/{ticker}", summary="Get live sentiment for a ticker")
async def get_sentiment(ticker: str, api_key: APIKey = Depends(verify_api_key)):
    try:
        result = await _sentiment_svc.get_sentiment(ticker.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/macro/snapshot", summary="Get latest macro indicators")
async def get_macro(api_key: APIKey = Depends(verify_api_key)):
    try:
        return _macro_fetcher.get_latest_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
