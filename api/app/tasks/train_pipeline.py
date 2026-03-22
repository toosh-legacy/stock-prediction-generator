"""
Standalone training pipeline that can be triggered via API or CLI.
"""
import asyncio
import os
from app.config import get_settings

settings = get_settings()


async def train_for_ticker(ticker: str, model_name: str, horizon: int) -> dict:
    """
    Train a model for a ticker and save to artifacts dir.
    Used by the model_registry when no cached model exists.
    """
    import yfinance as yf
    from app.ml.lstm_model import LSTMModel
    from app.ml.xgboost_model import XGBoostModel
    from app.ml.ensemble_model import EnsembleModel

    model_map = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}

    df = yf.Ticker(ticker).history(period="2y")
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)

    m = model_map[model_name]()
    metrics = m.train(df, horizon=horizon)

    os.makedirs(settings.model_artifacts_dir, exist_ok=True)
    path = os.path.join(
        settings.model_artifacts_dir, f"{ticker}_{model_name}_h{horizon}"
    )
    m.save(path)
    return {"ticker": ticker, "model": model_name, "horizon": horizon, "metrics": metrics}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    result = asyncio.run(train_for_ticker(args.ticker, args.model, args.horizon))
    print(result)
