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
        self.scaler = None
        self.feature_cols = None
        self.fe = FeatureEngineer()

    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        self.horizon = horizon
        df_feat = self.fe.add_technical_indicators(df)
        df_feat = self.fe.add_time_features(df_feat)
        X, y, self.feature_cols = self.fe.prepare_tabular(df_feat, horizon)
        split = int(len(X) * 0.8)
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
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
        return {"mae": mae, "n_estimators": self.model.best_iteration}

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        df_feat = self.fe.add_technical_indicators(df)
        df_feat = self.fe.add_time_features(df_feat)
        df_clean = df_feat[self.feature_cols].dropna()
        last_features = df_clean.iloc[-1:].values

        base_pred = float(self.model.predict(last_features)[0])
        # For multi-step, apply a simple drift multiplier per step
        preds = [round(base_pred * (1 + 0.0001 * i), 2) for i in range(horizon)]

        dates = [str(d.date()) for d in pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]]
        std = float(np.std(df['close'].pct_change().dropna()) * base_pred)

        return PredictionResult(
            dates=dates,
            predicted=preds,
            lower_ci=[round(p - 1.96 * std, 2) for p in preds],
            upper_ci=[round(p + 1.96 * std, 2) for p in preds],
            confidence=0.7,
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
