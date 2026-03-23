import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from app.ml.base_model import BaseStockModel, PredictionResult
from app.services.feature_engineer import FeatureEngineer


class XGBoostModel(BaseStockModel):
    model_name = "xgboost"

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.model = None
        self.feature_cols = None
        self.fe = FeatureEngineer()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features only if df doesn't already have them (beyond raw OHLCV)."""
        ohlcv = {"open", "high", "low", "close", "volume"}
        extra = [c for c in df.columns if c.lower() not in ohlcv]
        if len(extra) >= 10:
            return df  # already featured
        df = self.fe.add_technical_indicators(df)
        df = self.fe.add_lag_features(df)
        df = self.fe.add_time_features(df)
        return df

    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        self.horizon = horizon
        df_feat = self._build_features(df)
        X, y, self.feature_cols = self.fe.prepare_tabular(df_feat, horizon)

        split = int(len(X) * 0.8)
        self.model = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.05,
            reg_lambda=1.0,
            min_child_weight=3,
            early_stopping_rounds=50,
            eval_metric="mae",
            random_state=42,
        )
        self.model.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
            verbose=False,
        )
        preds = self.model.predict(X[split:])
        mae = float(np.mean(np.abs(preds - y[split:])))
        # Also compute directional accuracy on log returns
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y[split:])))
        return {
            "mae": round(mae, 6),
            "dir_acc": round(dir_acc, 4),
            "n_estimators": int(self.model.best_iteration or 0),
        }

    def _align_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df has exactly self.feature_cols, filling any missing ones with 0."""
        if self.feature_cols is None:
            return df
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_cols]

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        df_feat = self._build_features(df)
        df_aligned = self._align_cols(df_feat)
        # Forward-fill NaN so recent rows with FRED lag aren't dropped
        df_clean = df_aligned.ffill().dropna()
        last_features = df_clean.iloc[-1:].values
        current_price = float(df["close"].iloc[-1])

        # Predict 1-step log return, then compound for multi-step
        single_step_log_ret = float(self.model.predict(last_features)[0])

        # Compound: each step is an independent prediction (use average per-step return)
        step_log_ret = single_step_log_ret / max(horizon, 1)
        prices = []
        for i in range(1, horizon + 1):
            # Slight mean-reversion dampening per step
            compound = step_log_ret * (0.95 ** (i - 1))
            prices.append(round(current_price * np.exp(compound * i), 2))

        # Confidence interval using 20-day historical vol scaled to horizon
        hist_vol = float(df["close"].pct_change().dropna().std()) * np.sqrt(horizon)
        std_price = current_price * hist_vol

        dates = [str(d.date()) for d in pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]]
        confidence = float(max(0.0, min(1.0, 1.0 - abs(single_step_log_ret) * 10)))

        return PredictionResult(
            dates=dates,
            predicted=prices,
            lower_ci=[round(p - 1.96 * std_price, 2) for p in prices],
            upper_ci=[round(p + 1.96 * std_price, 2) for p in prices],
            confidence=confidence,
            model=self.model_name,
        )

    def get_feature_importance(self) -> dict:
        if self.model is None or self.feature_cols is None:
            return {}
        return dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))

    def save(self, path: str) -> None:
        self.model.save_model(f"{path}.json")
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({"feature_cols": self.feature_cols, "horizon": self.horizon}, f)

    def load(self, path: str) -> None:
        self.model = xgb.XGBRegressor()
        self.model.load_model(f"{path}.json")
        with open(f"{path}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]
        self.horizon = meta.get("horizon", 1)
