"""
Train a single model for a single ticker. Useful for quick experiments.

Usage:
    python training/train_single.py --ticker AAPL --model xgboost --horizon 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import yfinance as yf
from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel
from app.ml.ensemble_model import EnsembleModel

MODELS = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", choices=list(MODELS), default="xgboost")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--artifacts-dir", default=os.getenv("MODEL_ARTIFACTS_DIR", "./models"))
    args = parser.parse_args()

    print(f"Fetching {args.ticker}...")
    df = yf.Ticker(args.ticker).history(period="2y")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)

    print(f"Training {args.model}...")
    m = MODELS[args.model]()
    metrics = m.train(df, horizon=args.horizon)
    print(f"Metrics: {metrics}")

    os.makedirs(args.artifacts_dir, exist_ok=True)
    path = os.path.join(args.artifacts_dir, f"{args.ticker}_{args.model}_h{args.horizon}")
    m.save(path)
    print(f"Saved to {path}")

    # Quick prediction test
    result = m.predict(df, horizon=args.horizon)
    print(f"\nPredictions for next {args.horizon} days:")
    for d, p, lo, hi in zip(result["dates"], result["predicted"], result["lower_ci"], result["upper_ci"]):
        print(f"  {d}: ${p:.2f} [{lo:.2f}, {hi:.2f}]")
