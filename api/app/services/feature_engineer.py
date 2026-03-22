import pandas as pd
import numpy as np
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from app.config import get_settings

settings = get_settings()


class FeatureEngineer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Trend
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        # Momentum
        df['rsi'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_sig'] = macd['MACDs_12_26_9']
        # Volatility
        bb = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # Volume
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        # Lag returns
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'return_{lag}d'] = df['close'].pct_change(lag)
        df['volatility_20d'] = df[f'return_1d'].rolling(20).std()
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        return df

    def prepare_sequences(
        self, df: pd.DataFrame, seq_len: int = 60, horizon: int = 1
    ) -> tuple:
        """Returns (X, y, scaler, feature_cols) numpy arrays for sequence models."""
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        df_clean = df[feature_cols + ['close']].dropna()
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df_clean)
        X, y = [], []
        for i in range(seq_len, len(scaled) - horizon):
            X.append(scaled[i - seq_len:i])
            y.append(scaled[i + horizon - 1, -1])  # close price index
        return np.array(X), np.array(y), scaler, df_clean.columns.tolist()

    def prepare_tabular(
        self, df: pd.DataFrame, horizon: int = 1
    ) -> tuple:
        """Returns (X, y, feature_cols) for tree models."""
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        df_clean = df[feature_cols + ['close']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['close'].shift(-horizon).dropna().values
        X = X[:len(y)]
        return X, y, feature_cols
