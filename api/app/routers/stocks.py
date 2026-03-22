from fastapi import APIRouter, Depends, HTTPException
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
from app.services.feature_engineer import FeatureEngineer

router = APIRouter()
fetcher = DataFetcher()


@router.get("/{ticker}", summary="Get stock info and recent OHLCV")
async def get_stock(
    ticker: str,
    period: str = "1mo",
    api_key: APIKey = Depends(verify_api_key),
):
    try:
        df = await fetcher.get_ohlcv(ticker.upper(), period=period)
        info = await fetcher.get_info(ticker.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "ohlcv": [
            {
                "date": str(idx.date()),
                "open": round(float(r["open"]), 2),
                "high": round(float(r["high"]), 2),
                "low": round(float(r["low"]), 2),
                "close": round(float(r["close"]), 2),
                "volume": int(r["volume"]),
            }
            for idx, r in df.iterrows()
        ],
    }


@router.get("/{ticker}/indicators", summary="Get technical indicators for a ticker")
async def get_indicators(
    ticker: str,
    api_key: APIKey = Depends(verify_api_key),
):
    try:
        df = await fetcher.get_ohlcv(ticker.upper(), period="6mo")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    fe = FeatureEngineer()
    df_feat = fe.add_technical_indicators(df)
    latest = df_feat.iloc[-1].dropna().to_dict()

    return {
        "ticker": ticker.upper(),
        "date": str(df_feat.index[-1].date()),
        "indicators": {k: round(float(v), 4) for k, v in latest.items()},
    }
