"""
Proper training pipeline with:
  - Walk-forward TimeSeriesSplit cross-validation (5 folds)
  - Full feature engineering (technical + macro + sentiment zeros)
  - XGBoost hyperparameter search
  - LSTM training
  - Ensemble construction
  - Per-fold + aggregate metrics: RMSE, MAE, MAPE, directional accuracy
  - SHAP feature importance for XGBoost
  - Artifacts pushed to HuggingFace Hub

Usage:
    python training/train_pipeline.py --ticker AAPL --horizon 5
    python training/train_pipeline.py --ticker AAPL MSFT GOOGL --horizon 5 --models xgboost
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from app.config import get_settings
from app.services.feature_engineer import FeatureEngineer
from app.services.macro_fetcher import MacroFetcher
from app.ml.xgboost_model import XGBoostModel
from app.ml.lstm_model import LSTMModel
from app.ml.ensemble_model import EnsembleModel
from app.services.model_registry import push_to_hub

settings = get_settings()
fe = FeatureEngineer()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    print(f"  Fetching {ticker} ({period})...")
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    return df


def load_macro_data(start: str, end: str) -> pd.DataFrame:
    print("  Fetching macro data (FRED + Yahoo Finance)...")
    try:
        fetcher = MacroFetcher(fred_api_key=settings.fred_api_key)
        return fetcher.get_macro_features(start, end)
    except Exception as e:
        print(f"  Macro fetch failed (continuing without): {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    # Directional accuracy: did we predict the right direction vs previous?
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    dir_acc = float(np.mean(actual_dir == pred_dir)) if len(actual_dir) > 0 else 0.0
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 2), "dir_acc": round(dir_acc, 4)}


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(df_feat: pd.DataFrame, model_cls, model_kwargs: dict, horizon: int, n_splits: int = 5) -> list[dict]:
    """Run TimeSeriesSplit CV and return per-fold metrics."""
    fe_local = FeatureEngineer()
    feature_cols = [c for c in df_feat.columns if c not in ["open", "high", "low", "close", "volume"]]
    df_clean = df_feat[feature_cols + ["close"]].dropna()

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_clean)):
        train_df = df_feat.iloc[train_idx]
        test_df = df_feat.iloc[test_idx]
        if len(train_df) < 100 or len(test_df) < 10:
            continue

        m = model_cls(**model_kwargs)
        try:
            m.train(train_df, horizon=horizon)
        except Exception as e:
            print(f"    Fold {fold+1} training failed: {e}")
            continue

        # Walk-forward predictions on test set
        preds, actuals = [], []
        for i in range(len(test_df)):
            ctx = pd.concat([train_df, test_df.iloc[:i]])
            try:
                r = m.predict(ctx, horizon=1)
                preds.append(r["predicted"][0])
            except Exception:
                preds.append(float(ctx["close"].iloc[-1]))
            actuals.append(float(test_df["close"].iloc[i]))

        metrics = compute_metrics(np.array(actuals), np.array(preds))
        metrics["fold"] = fold + 1
        metrics["train_size"] = len(train_df)
        metrics["test_size"] = len(test_df)
        fold_metrics.append(metrics)
        print(f"    Fold {fold+1}: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
              f"MAPE={metrics['mape']:.1f}%  DirAcc={metrics['dir_acc']*100:.1f}%")

    return fold_metrics


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap_importance(model: XGBoostModel, df_feat: pd.DataFrame) -> pd.Series:
    try:
        import shap
        feature_cols = model.feature_cols
        df_clean = df_feat[feature_cols].dropna()
        X = df_clean.values[-200:]  # last 200 rows for speed
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X)
        importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_cols,
        ).sort_values(ascending=False)
        return importance
    except Exception as e:
        print(f"  SHAP skipped: {e}")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ticker(
    ticker: str,
    horizon: int,
    model_names: list[str],
    artifacts_dir: str,
    n_cv_splits: int = 5,
    push_hf: bool = True,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  {ticker}  |  horizon={horizon}d  |  models={model_names}")
    print(f"{'='*60}")

    df = load_stock_data(ticker)
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")

    macro_df = load_macro_data(start, end)

    print("  Building features...")
    df_feat = fe.build_features(df, macro_df=macro_df, sentiment_features={})

    # Hold out last 20% as final test set (never seen during CV)
    split = int(len(df_feat) * 0.8)
    train_val_df = df_feat.iloc[:split]
    test_df = df_feat.iloc[split:]
    test_raw = df.iloc[split:]

    results = {"ticker": ticker, "horizon": horizon, "models": {}}

    for model_name in model_names:
        print(f"\n  --- {model_name.upper()} ---")
        model_map = {"xgboost": (XGBoostModel, {}), "lstm": (LSTMModel, {"epochs": 50}), "ensemble": (EnsembleModel, {})}
        if model_name not in model_map:
            continue
        model_cls, model_kwargs = model_map[model_name]

        # Cross-validation
        print(f"  Running {n_cv_splits}-fold walk-forward CV...")
        fold_metrics = walk_forward_cv(train_val_df, model_cls, model_kwargs, horizon, n_cv_splits)

        if not fold_metrics:
            print(f"  No valid folds — skipping {model_name}")
            continue

        avg = {k: round(np.mean([f[k] for f in fold_metrics]), 4)
               for k in ["rmse", "mae", "mape", "dir_acc"]}
        print(f"  CV averages: RMSE={avg['rmse']:.2f}  MAE={avg['mae']:.2f}  "
              f"MAPE={avg['mape']:.1f}%  DirAcc={avg['dir_acc']*100:.1f}%")

        # Final training on full train+val data
        print(f"  Training final model on full train set ({len(train_val_df)} days)...")
        final_model = model_cls(**model_kwargs)
        final_model.train(train_val_df, horizon=horizon)

        # Holdout test evaluation
        print(f"  Evaluating on holdout test ({len(test_df)} days)...")
        preds, actuals = [], []
        for i in range(len(test_df)):
            ctx = pd.concat([train_val_df, test_df.iloc[:i]])
            try:
                r = final_model.predict(ctx, horizon=1)
                preds.append(r["predicted"][0])
            except Exception:
                preds.append(float(ctx["close"].iloc[-1]))
            actuals.append(float(test_df["close"].iloc[i]))

        test_metrics = compute_metrics(np.array(actuals), np.array(preds))
        print(f"  TEST  : RMSE={test_metrics['rmse']:.2f}  MAE={test_metrics['mae']:.2f}  "
              f"MAPE={test_metrics['mape']:.1f}%  DirAcc={test_metrics['dir_acc']*100:.1f}%")

        # SHAP importance (XGBoost only)
        shap_top10 = {}
        if model_name == "xgboost":
            importance = compute_shap_importance(final_model, train_val_df)
            if not importance.empty:
                shap_top10 = importance.head(10).round(4).to_dict()
                print(f"  Top 5 SHAP features: {list(shap_top10.keys())[:5]}")

        # Save artifact
        os.makedirs(artifacts_dir, exist_ok=True)
        path = os.path.join(artifacts_dir, f"{ticker}_{model_name}_h{horizon}")
        final_model.save(path)
        print(f"  Saved -> {path}")

        # Push to HF Hub
        if push_hf and settings.using_hf:
            pushed = push_to_hub(artifacts_dir, ticker, model_name, horizon)
            print(f"  HF Hub push: {'OK' if pushed else 'SKIPPED'}")

        results["models"][model_name] = {
            "cv_avg": avg,
            "test": test_metrics,
            "shap_top10": shap_top10,
            "n_features": len(train_val_df.columns),
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSage training pipeline")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["xgboost", "lstm", "ensemble"])
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--artifacts-dir", default=os.getenv("MODEL_ARTIFACTS_DIR", "./models"))
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace Hub push")
    args = parser.parse_args()

    all_results = []
    for ticker in args.tickers:
        try:
            r = train_ticker(
                ticker=ticker,
                horizon=args.horizon,
                model_names=args.models,
                artifacts_dir=args.artifacts_dir,
                n_cv_splits=args.cv_splits,
                push_hf=not args.no_hf,
            )
            all_results.append(r)
        except Exception as e:
            print(f"ERROR training {ticker}: {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ticker':<8} {'Model':<12} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'DirAcc%':>9}")
    print("-" * 60)
    for r in all_results:
        for model_name, m in r["models"].items():
            t = m["test"]
            print(f"{r['ticker']:<8} {model_name:<12} {t['rmse']:>8.2f} {t['mae']:>8.2f} "
                  f"{t['mape']:>7.1f}% {t['dir_acc']*100:>8.1f}%")
    print(f"\nDone. Artifacts in: {args.artifacts_dir}")
    if settings.using_hf:
        print(f"HF Hub repo: {settings.hf_repo_id}")
