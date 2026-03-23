"""
Extended feature engineering pipeline.
Produces 60+ features per trading day:
  - Technical indicators (trend, momentum, volatility, volume)
  - Lag returns & rolling statistics
  - Calendar / time features
  - Macro features (injected from MacroFetcher)
  - Sentiment features (injected at inference time; 0 during training)
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler


class FeatureEngineer:
    """
    Stateless feature builder.  All methods take a DataFrame with lowercase
    columns [open, high, low, close, volume] and return an augmented DataFrame.
    """

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Trend ---
        df["sma_10"] = ta.sma(df["close"], length=10)
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_12"] = ta.ema(df["close"], length=12)
        df["ema_26"] = ta.ema(df["close"], length=26)
        df["ema_50"] = ta.ema(df["close"], length=50)
        # Price relative to moving averages
        df["price_to_sma20"] = df["close"] / df["sma_20"].replace(0, np.nan) - 1
        df["price_to_sma50"] = df["close"] / df["sma_50"].replace(0, np.nan) - 1
        df["sma20_to_sma50"] = df["sma_20"] / df["sma_50"].replace(0, np.nan) - 1

        # --- Momentum ---
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)
        df["rsi_21"] = ta.rsi(df["close"], length=21)
        macd = ta.macd(df["close"])
        if macd is not None:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 1]
            df["macd_hist"] = macd.iloc[:, 2]
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and len(stoch.columns) >= 2:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        df["roc_10"] = ta.roc(df["close"], length=10)
        df["roc_20"] = ta.roc(df["close"], length=20)
        df["mom_10"] = ta.mom(df["close"], length=10)

        # --- Volatility ---
        bb = ta.bbands(df["close"], length=20)
        if bb is not None:
            df["bb_upper"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_lower"] = bb.iloc[:, 2]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_7"] = ta.atr(df["high"], df["low"], df["close"], length=7)
        df["natr"] = ta.natr(df["high"], df["low"], df["close"], length=14)
        # Daily range & gap
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # --- Trend strength ---
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            df["adx"] = adx.iloc[:, 0]
        aroon = ta.aroon(df["high"], df["low"], length=25)
        if aroon is not None and len(aroon.columns) >= 2:
            df["aroon_up"] = aroon.iloc[:, 0]
            df["aroon_down"] = aroon.iloc[:, 1]

        # --- Volume ---
        df["obv"] = ta.obv(df["close"], df["volume"])
        try:
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        except Exception:
            df["vwap"] = df["close"]
        df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["volume_sma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma20"].replace(0, np.nan)

        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lags = [1, 2, 3, 5, 10, 20, 60]
        for lag in lags:
            df[f"return_{lag}d"] = df["close"].pct_change(lag)
        # Rolling statistics
        for window in [5, 10, 20, 60]:
            df[f"roll_mean_{window}"] = df["close"].rolling(window).mean()
            df[f"roll_std_{window}"] = df["close"].rolling(window).std()
            df[f"roll_max_{window}"] = df["close"].rolling(window).max()
            df[f"roll_min_{window}"] = df["close"].rolling(window).min()
        df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
        df["volatility_60d"] = df["return_1d"].rolling(60).std() * np.sqrt(252)
        # Candle body / shadow features
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"]
        df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"]
        return df

    # ------------------------------------------------------------------
    # Calendar / time features
    # ------------------------------------------------------------------

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(int)
        return df

    # ------------------------------------------------------------------
    # Macro feature injection
    # ------------------------------------------------------------------

    def add_macro_features(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Join macro features onto the stock DataFrame by date index."""
        if macro_df is None or macro_df.empty:
            return df
        df = df.copy()
        # Align on business day index
        macro_aligned = macro_df.reindex(df.index, method="ffill")
        for col in macro_aligned.columns:
            df[f"macro_{col}"] = macro_aligned[col].values
        return df

    # ------------------------------------------------------------------
    # Sentiment injection (inference-time only)
    # ------------------------------------------------------------------

    def inject_sentiment(self, df: pd.DataFrame, sentiment_features: dict) -> pd.DataFrame:
        """
        Append sentiment features as constant columns to the last row(s).
        During training, sentiment_features should be {} (zeros will be used).
        At inference, pass the live sentiment dict.
        """
        df = df.copy()
        defaults = {
            "sentiment_compound": 0.0,
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_article_count": 0.0,
        }
        for k, default in defaults.items():
            df[k] = sentiment_features.get(k, default)
        return df

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_features(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
        sentiment_features: dict | None = None,
    ) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        df = self.add_technical_indicators(df)
        df = self.add_lag_features(df)
        df = self.add_time_features(df)
        if macro_df is not None and not macro_df.empty:
            df = self.add_macro_features(df, macro_df)
        if sentiment_features:
            df = self.inject_sentiment(df, sentiment_features)
        else:
            df = self.inject_sentiment(df, {})  # zero sentiment during training
        return df

    # ------------------------------------------------------------------
    # Sequence preparation (for LSTM)
    # ------------------------------------------------------------------

    def prepare_sequences(
        self, df: pd.DataFrame, seq_len: int = 60, horizon: int = 1
    ) -> tuple:
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        df_clean = df[feature_cols + ["close"]].dropna()
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df_clean)
        X, y = [], []
        for i in range(seq_len, len(scaled) - horizon):
            X.append(scaled[i - seq_len:i])
            y.append(scaled[i + horizon - 1, -1])
        return np.array(X), np.array(y), scaler, df_clean.columns.tolist()

    # ------------------------------------------------------------------
    # Tabular preparation (for XGBoost / tree models)
    # ------------------------------------------------------------------

    def prepare_tabular(self, df: pd.DataFrame, horizon: int = 1) -> tuple:
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        df_clean = df[feature_cols + ["close"]].dropna()
        X = df_clean[feature_cols].values
        y = df_clean["close"].shift(-horizon).dropna().values
        X = X[: len(y)]
        return X, y, feature_cols
