"""
APScheduler job — re-trains demo tickers daily to keep artifacts fresh.
Register this in main.py startup if you want automated retraining.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.config import get_settings

settings = get_settings()

DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]
DEMO_HORIZONS = [1, 5]

scheduler = AsyncIOScheduler()


async def retrain_demo_models():
    """Re-train XGBoost models for demo tickers. Runs nightly."""
    import yfinance as yf
    import os
    from app.ml.xgboost_model import XGBoostModel

    for ticker in DEMO_TICKERS:
        for horizon in DEMO_HORIZONS:
            try:
                df = yf.Ticker(ticker).history(period="2y")
                df.columns = [c.lower() for c in df.columns]
                df.index = df.index.tz_localize(None)

                m = XGBoostModel()
                m.train(df, horizon=horizon)

                os.makedirs(settings.model_artifacts_dir, exist_ok=True)
                path = os.path.join(
                    settings.model_artifacts_dir, f"{ticker}_xgboost_h{horizon}"
                )
                m.save(path)

                # Update in-memory cache
                from app.services.model_registry import set_cached_model
                set_cached_model(ticker, "xgboost", horizon, m)

            except Exception as e:
                print(f"[scheduler] Failed to retrain {ticker} h={horizon}: {e}")


def start_scheduler():
    scheduler.add_job(retrain_demo_models, "cron", hour=2, minute=0)
    scheduler.start()
