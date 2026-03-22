"""
Evaluate trained models on a held-out test set and report key metrics.

Usage:
    python training/evaluate.py --ticker AAPL --model xgboost --horizon 5
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import yfinance as yf
from app.ml.xgboost_model import XGBoostModel
from app.ml.lstm_model import LSTMModel
from app.ml.ensemble_model import EnsembleModel
from app.services.backtester import Backtester

MODELS = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}


def evaluate(ticker: str, model_name: str, horizon: int, artifacts_dir: str):
    print(f"Fetching {ticker}...")
    df = yf.Ticker(ticker).history(period="3y")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)

    # Use last 20% as test set
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    path = os.path.join(artifacts_dir, f"{ticker}_{model_name}_h{horizon}")
    m = MODELS[model_name]()

    try:
        m.load(path)
        print(f"Loaded model from {path}")
    except Exception:
        print(f"No saved model found at {path}. Training fresh...")
        m.train(train_df, horizon=horizon)

    # Walk-forward predictions
    print("Running walk-forward evaluation...")
    preds, actuals = [], []
    for i in range(len(test_df)):
        ctx = pd.concat([train_df, test_df.iloc[:i]])
        try:
            res = m.predict(ctx, horizon=1)
            preds.append(res["predicted"][0])
        except Exception:
            preds.append(float(ctx["close"].iloc[-1]))
        actuals.append(float(test_df["close"].iloc[i]))

    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)

    rmse = float(np.sqrt(np.mean((preds_arr - actuals_arr) ** 2)))
    mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
    mape = float(np.mean(np.abs((preds_arr - actuals_arr) / actuals_arr)) * 100)

    pred_series = pd.Series(preds, index=test_df.index)
    bt = Backtester(initial_capital=10_000.0, transaction_cost=0.001)
    bt_result = bt.run(test_df, pred_series, ticker=ticker, model=model_name)

    print(f"\n{'='*50}")
    print(f"Evaluation: {ticker} | {model_name} | horizon={horizon}")
    print(f"{'='*50}")
    print(f"Test set: {len(test_df)} days")
    print(f"\nPrice Prediction Accuracy:")
    print(f"  RMSE:  ${rmse:.2f}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"\nBacktest Performance:")
    print(f"  Total Return:     {bt_result.total_return*100:.2f}%")
    print(f"  Benchmark Return: {bt_result.benchmark_return*100:.2f}%")
    print(f"  Sharpe Ratio:     {bt_result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:     {bt_result.max_drawdown*100:.2f}%")
    print(f"  Win Rate:         {bt_result.win_rate*100:.1f}%")

    return {
        "ticker": ticker, "model": model_name, "horizon": horizon,
        "rmse": rmse, "mae": mae, "mape": mape,
        **bt_result.__dict__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model", choices=list(MODELS), default="xgboost")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--artifacts-dir", default=os.getenv("MODEL_ARTIFACTS_DIR", "./models"))
    args = parser.parse_args()

    evaluate(args.ticker, args.model, args.horizon, args.artifacts_dir)
