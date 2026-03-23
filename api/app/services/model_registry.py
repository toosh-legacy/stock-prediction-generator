"""
Model registry with three-tier loading:
  1. In-memory cache (fastest — survives request lifetime)
  2. Local disk artifacts at MODEL_ARTIFACTS_DIR
  3. HuggingFace Hub (persistent across redeploys)
  4. Train fresh as last resort
"""
import os
import threading
import glob
import logging

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_model_cache: dict = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(ticker: str, model_name: str, horizon: int) -> str:
    return f"{ticker}_{model_name}_{horizon}"


def get_cached_model(ticker: str, model_name: str, horizon: int):
    with _cache_lock:
        return _model_cache.get(_cache_key(ticker, model_name, horizon))


def set_cached_model(ticker: str, model_name: str, horizon: int, model) -> None:
    with _cache_lock:
        _model_cache[_cache_key(ticker, model_name, horizon)] = model


# ---------------------------------------------------------------------------
# HuggingFace Hub helpers
# ---------------------------------------------------------------------------

def _hf_artifact_prefix(ticker: str, model_name: str, horizon: int) -> str:
    return f"{ticker}_{model_name}_h{horizon}"


def push_to_hub(local_dir: str, ticker: str, model_name: str, horizon: int) -> bool:
    """Upload all artifact files for this model to HF Hub. Returns True on success."""
    if not settings.using_hf:
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=settings.hf_token)
        prefix = _hf_artifact_prefix(ticker, model_name, horizon)
        pattern = os.path.join(local_dir, f"{prefix}*")
        files = glob.glob(pattern)
        for fpath in files:
            fname = os.path.basename(fpath)
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"models/{fname}",
                repo_id=settings.hf_repo_id,
                repo_type="dataset",
            )
        log.info(f"Pushed {len(files)} artifact(s) for {ticker}/{model_name}/h{horizon} to HF Hub")
        return True
    except Exception as e:
        log.warning(f"HF Hub push failed: {e}")
        return False


def pull_from_hub(local_dir: str, ticker: str, model_name: str, horizon: int) -> bool:
    """Download artifact files from HF Hub to local_dir. Returns True if any files found."""
    if not settings.using_hf:
        return False
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=settings.hf_token)
        prefix = _hf_artifact_prefix(ticker, model_name, horizon)
        # List all files in the repo matching this prefix
        files = [
            f.rfilename
            for f in api.list_repo_files(repo_id=settings.hf_repo_id, repo_type="dataset")
            if f.rfilename.startswith(f"models/{prefix}")
        ]
        if not files:
            return False
        os.makedirs(local_dir, exist_ok=True)
        for repo_path in files:
            hf_hub_download(
                repo_id=settings.hf_repo_id,
                filename=repo_path,
                repo_type="dataset",
                local_dir=local_dir,
                token=settings.hf_token,
            )
        log.info(f"Pulled {len(files)} artifact(s) for {ticker}/{model_name}/h{horizon} from HF Hub")
        return True
    except Exception as e:
        log.warning(f"HF Hub pull failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def load_or_train_model(ticker: str, model_name: str, horizon: int, df):
    """
    Load or train a model using the four-tier strategy:
    memory → disk → HF Hub → train fresh.
    """
    from app.ml.lstm_model import LSTMModel
    from app.ml.xgboost_model import XGBoostModel
    from app.ml.ensemble_model import EnsembleModel

    model_map = {
        "lstm": LSTMModel,
        "xgboost": XGBoostModel,
        "ensemble": EnsembleModel,
    }

    # 1. Memory
    cached = get_cached_model(ticker, model_name, horizon)
    if cached is not None:
        return cached

    m = model_map[model_name]()
    local_dir = settings.model_artifacts_dir
    path = os.path.join(local_dir, f"{ticker}_{model_name}_h{horizon}")

    # 2. Local disk
    try:
        m.load(path)
        log.info(f"Loaded {ticker}/{model_name}/h{horizon} from disk")
        set_cached_model(ticker, model_name, horizon, m)
        return m
    except Exception:
        pass

    # 3. HuggingFace Hub
    if pull_from_hub(local_dir, ticker, model_name, horizon):
        try:
            m.load(path)
            log.info(f"Loaded {ticker}/{model_name}/h{horizon} from HF Hub")
            set_cached_model(ticker, model_name, horizon, m)
            return m
        except Exception:
            pass

    # 4. Train fresh
    log.info(f"Training {ticker}/{model_name}/h{horizon} from scratch...")
    m.train(df, horizon=horizon)
    os.makedirs(local_dir, exist_ok=True)
    try:
        m.save(path)
        push_to_hub(local_dir, ticker, model_name, horizon)
    except Exception as e:
        log.warning(f"Could not save model: {e}")

    set_cached_model(ticker, model_name, horizon, m)
    return m
