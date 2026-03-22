import pytest
import pandas as pd
import numpy as np
from app.services.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )
    return df


def test_add_technical_indicators(sample_df):
    fe = FeatureEngineer()
    result = fe.add_technical_indicators(sample_df)
    assert "sma_20" in result.columns
    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "bb_upper" in result.columns
    assert "atr" in result.columns
    assert "obv" in result.columns
    assert "return_1d" in result.columns


def test_add_time_features(sample_df):
    fe = FeatureEngineer()
    result = fe.add_time_features(sample_df)
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert result["day_of_week"].between(0, 4).all()  # Mon–Fri only (business days)


def test_prepare_sequences(sample_df):
    fe = FeatureEngineer()
    df_feat = fe.add_technical_indicators(sample_df)
    df_feat = fe.add_time_features(df_feat)
    X, y, scaler, cols = fe.prepare_sequences(df_feat, seq_len=20, horizon=1)
    assert X.ndim == 3
    assert X.shape[1] == 20
    assert len(y) == len(X)
    assert scaler is not None


def test_prepare_tabular(sample_df):
    fe = FeatureEngineer()
    df_feat = fe.add_technical_indicators(sample_df)
    df_feat = fe.add_time_features(df_feat)
    X, y, cols = fe.prepare_tabular(df_feat, horizon=1)
    assert X.ndim == 2
    assert len(X) == len(y)
    assert len(cols) > 0
