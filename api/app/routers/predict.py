from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
from app.services.model_registry import load_or_train_model

router = APIRouter()
fetcher = DataFetcher()


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
    description="""
Predict future closing prices for a given ticker.

Returns a list of `{date, predicted, lower_ci, upper_ci}` objects for the requested horizon.

**Note:** Models are pre-trained on 2 years of daily data. The first request for a
ticker may train a model on-the-fly (~30s). Subsequent requests use cached models.
    """,
)
async def predict_stock(
    body: PredictionRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    try:
        df = await fetcher.get_ohlcv(body.ticker.upper(), period="2y")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        m = load_or_train_model(body.ticker.upper(), body.model, body.horizon, df)
        result = m.predict(df, horizon=body.horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    return PredictionResponse(
        ticker=body.ticker.upper(),
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
