from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel
from app.ml.ensemble_model import EnsembleModel
from app.services.backtester import Backtester
import pandas as pd

router = APIRouter()
fetcher = DataFetcher()


class BacktestRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    model: Literal["lstm", "xgboost", "ensemble"] = "ensemble"
    start_date: str = Field("2022-01-01", example="2022-01-01")
    end_date: str = Field("2023-12-31", example="2023-12-31")
    initial_capital: float = Field(10000.0, ge=1000)
    transaction_cost_bps: float = Field(
        10, ge=0, le=100, description="Transaction cost in basis points"
    )


@router.post("/", summary="Backtest a model on historical data")
async def backtest(
    body: BacktestRequest,
    api_key: APIKey = Depends(verify_api_key),
):
    try:
        df = await fetcher.get_ohlcv(
            body.ticker.upper(), start=body.start_date, end=body.end_date
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail="Not enough data in the specified date range. Need at least 30 trading days.",
        )

    model_map = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}
    m = model_map[body.model]()

    train_df = df.iloc[: int(len(df) * 0.7)]
    test_df = df.iloc[int(len(df) * 0.7):]

    try:
        m.train(train_df, horizon=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

    # Walk-forward prediction on test set
    preds = []
    for i in range(len(test_df)):
        context = pd.concat([train_df, test_df.iloc[:i]])
        try:
            result = m.predict(context, horizon=1)
            preds.append(result["predicted"][0])
        except Exception:
            # Fallback to last known price if prediction fails
            preds.append(float(context["close"].iloc[-1]))

    pred_series = pd.Series(preds, index=test_df.index)
    bt = Backtester(
        initial_capital=body.initial_capital,
        transaction_cost=body.transaction_cost_bps / 10000,
    )
    result = bt.run(test_df, pred_series, ticker=body.ticker.upper(), model=body.model)

    return {
        **result.__dict__,
        "disclaimer": "Backtest results are hypothetical. Past performance does not guarantee future results.",
    }
