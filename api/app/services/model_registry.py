"""
Model registry: caches trained models in memory to avoid re-training on every request.
On cold start, models are loaded from MODEL_ARTIFACTS_DIR if artifacts exist,
otherwise trained fresh and saved.
"""
import os
import threading
from typing import Optional
from app.config import get_settings

settings = get_settings()

# In-memory cache: {"{ticker}_{model}_{horizon}": model_instance}
_model_cache: dict = {}
_cache_lock = threading.Lock()


def get_model_path(ticker: str, model_name: str, horizon: int) -> str:
    return os.path.join(settings.model_artifacts_dir, f"{ticker}_{model_name}_h{horizon}")


def get_cached_model(ticker: str, model_name: str, horizon: int):
    key = f"{ticker}_{model_name}_{horizon}"
    with _cache_lock:
        return _model_cache.get(key)


def set_cached_model(ticker: str, model_name: str, horizon: int, model) -> None:
    key = f"{ticker}_{model_name}_{horizon}"
    with _cache_lock:
        _model_cache[key] = model


def load_or_train_model(ticker: str, model_name: str, horizon: int, df):
    """
    1. Check in-memory cache
    2. Try loading from disk artifacts
    3. Train fresh, save to disk, cache in memory
    """
    from app.ml.lstm_model import LSTMModel
    from app.ml.xgboost_model import XGBoostModel
    from app.ml.ensemble_model import EnsembleModel

    model_map = {
        "lstm": LSTMModel,
        "xgboost": XGBoostModel,
        "ensemble": EnsembleModel,
    }

    # 1. Memory cache
    cached = get_cached_model(ticker, model_name, horizon)
    if cached is not None:
        return cached

    model_cls = model_map[model_name]
    m = model_cls()
    path = get_model_path(ticker, model_name, horizon)

    # 2. Disk artifacts
    try:
        m.load(path)
        set_cached_model(ticker, model_name, horizon, m)
        return m
    except Exception:
        pass

    # 3. Train fresh
    m.train(df, horizon=horizon)
    os.makedirs(settings.model_artifacts_dir, exist_ok=True)
    try:
        m.save(path)
    except Exception:
        pass  # Saving is best-effort; serve the prediction regardless

    set_cached_model(ticker, model_name, horizon, m)
    return m
