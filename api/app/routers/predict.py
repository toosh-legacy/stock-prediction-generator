from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
from app.services.feature_engineer import FeatureEngineer
from app.services.macro_fetcher import MacroFetcher
from app.services.sentiment_service import SentimentService
from app.services.model_registry import load_or_train_model
from app.config import get_settings

router = APIRouter()
fetcher = DataFetcher()
fe = FeatureEngineer()
settings = get_settings()


class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock ticker symbol")
    model: Literal["lstm", "xgboost", "ensemble"] = Field(
        "ensemble", description="Model to use"
    )
    horizon: int = Field(5, ge=1, le=30, description="Days to predict ahead (1–30)")


class PredictionResponse(BaseModel):
    ticker: str
    model: str
    horizon: int
    predictions: list[dict]
    confidence: float
    disclaimer: str = "Predictions are for informational purposes only. Not financial advice."


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Predict stock prices",
)
async def predict_stock(
    body: PredictionRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    ticker = body.ticker.upper()

    try:
        df = await fetcher.get_ohlcv(ticker, period="2y")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # Build full feature set with live macro + sentiment (same as training)
        start = df.index[0].strftime("%Y-%m-%d")
        end = df.index[-1].strftime("%Y-%m-%d")

        macro_df = None
        try:
            macro_fetcher = MacroFetcher(fred_api_key=settings.fred_api_key)
            macro_df = macro_fetcher.get_macro_features(start, end)
        except Exception:
            pass

        sentiment_features = {}
        try:
            svc = SentimentService(news_api_key=settings.news_api_key)
            sentiment_result = await svc.get_sentiment(ticker)
            from app.services.sentiment_service import sentiment_to_features
            sentiment_features = sentiment_to_features(sentiment_result)
        except Exception:
            pass

        df_feat = fe.build_features(df, macro_df=macro_df, sentiment_features=sentiment_features)

        m = load_or_train_model(ticker, body.model, body.horizon, df_feat)
        result = m.predict(df_feat, horizon=body.horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    return PredictionResponse(
        ticker=ticker,
        model=body.model,
        horizon=body.horizon,
        predictions=[
            {"date": d, "predicted": p, "lower_ci": l, "upper_ci": u}
            for d, p, l, u in zip(
                result["dates"],
                result["predicted"],
                result["lower_ci"],
                result["upper_ci"],
            )
        ],
        confidence=result["confidence"],
    )
