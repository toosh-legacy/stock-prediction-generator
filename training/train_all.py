"""
Train all models for a list of tickers and save artifacts to MODEL_ARTIFACTS_DIR.
Metrics are logged to MLflow.

Usage:
    python training/train_all.py --tickers AAPL MSFT GOOGL TSLA NVDA --horizon 5
    python training/train_all.py --tickers AAPL --models xgboost --horizon 1
"""
import argparse
import os
import sys

# Add api/ to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import mlflow
import yfinance as yf

from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel
from app.ml.ensemble_model import EnsembleModel

MODELS = {
    "lstm": LSTMModel,
    "xgboost": XGBoostModel,
    "ensemble": EnsembleModel,
}

DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]


def fetch_data(ticker: str):
    print(f"  Fetching 2y OHLCV for {ticker}...")
    df = yf.Ticker(ticker).history(period="2y")
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    return df


def train_ticker(ticker: str, horizon: int, model_names: list[str], artifacts_dir: str):
    df = fetch_data(ticker)
    os.makedirs(artifacts_dir, exist_ok=True)

    for model_name in model_names:
        print(f"  Training {model_name} for {ticker} (horizon={horizon})...")
        with mlflow.start_run(run_name=f"{ticker}_{model_name}_h{horizon}"):
            mlflow.set_tags({"ticker": ticker, "model": model_name, "horizon": horizon})
            m = MODELS[model_name]()
            try:
                metrics = m.train(df, horizon=horizon)
                mlflow.log_params({"horizon": horizon, "ticker": ticker, "model": model_name})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

                local_path = os.path.join(artifacts_dir, f"{ticker}_{model_name}_h{horizon}")
                m.save(local_path)
                print(f"    Saved to {local_path}. Metrics: {metrics}")
            except Exception as e:
                print(f"    ERROR training {model_name} for {ticker}: {e}")
                mlflow.set_tag("status", f"FAILED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StockSage models for a set of tickers.")
    parser.add_argument("--tickers", nargs="+", default=DEMO_TICKERS, help="Ticker symbols")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days")
    parser.add_argument("--models", nargs="+", default=["xgboost", "lstm", "ensemble"], help="Models to train")
    parser.add_argument("--artifacts-dir", default=os.getenv("MODEL_ARTIFACTS_DIR", "./models"), help="Output dir")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"), help="MLflow URI")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("stocksage-training")

    print(f"Training {len(args.tickers)} tickers × {len(args.models)} models at horizon={args.horizon}")
    print(f"Artifacts → {args.artifacts_dir}")
    print(f"MLflow → {args.mlflow_uri}\n")

    for ticker in args.tickers:
        print(f"\n{'='*50}\nTicker: {ticker}\n{'='*50}")
        train_ticker(ticker, args.horizon, args.models, args.artifacts_dir)

    print("\nDone! Run `mlflow ui` to view experiment results.")
