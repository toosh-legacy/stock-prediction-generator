import numpy as np
import pandas as pd
import pickle
from app.ml.base_model import BaseStockModel, PredictionResult
from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel


class EnsembleModel(BaseStockModel):
    model_name = "ensemble"

    def __init__(self):
        self.lstm = LSTMModel(epochs=30)
        self.xgb = XGBoostModel()
        # Blending weights (equal by default)
        self.lstm_weight = 0.5
        self.xgb_weight = 0.5

    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        lstm_metrics = self.lstm.train(df, horizon)
        xgb_metrics = self.xgb.train(df, horizon)
        return {"lstm": lstm_metrics, "xgb": xgb_metrics, "status": "trained"}

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        lstm_result = self.lstm.predict(df, horizon)
        xgb_result = self.xgb.predict(df, horizon)

        blended = [
            round(self.lstm_weight * l + self.xgb_weight * x, 2)
            for l, x in zip(lstm_result['predicted'], xgb_result['predicted'])
        ]
        lower = [min(l, x) for l, x in zip(lstm_result['lower_ci'], xgb_result['lower_ci'])]
        upper = [max(l, x) for l, x in zip(lstm_result['upper_ci'], xgb_result['upper_ci'])]

        return PredictionResult(
            dates=lstm_result['dates'],
            predicted=blended,
            lower_ci=lower,
            upper_ci=upper,
            confidence=float(np.mean([lstm_result['confidence'], xgb_result['confidence']])),
            model=self.model_name,
        )

    def save(self, path: str) -> None:
        self.lstm.save(f"{path}_lstm")
        self.xgb.save(f"{path}_xgb")
        with open(f"{path}_ensemble_meta.pkl", "wb") as f:
            pickle.dump({"lstm_weight": self.lstm_weight, "xgb_weight": self.xgb_weight}, f)

    def load(self, path: str) -> None:
        self.lstm.load(f"{path}_lstm")
        self.xgb.load(f"{path}_xgb")
        try:
            with open(f"{path}_ensemble_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            self.lstm_weight = meta.get("lstm_weight", 0.5)
            self.xgb_weight = meta.get("xgb_weight", 0.5)
        except FileNotFoundError:
            pass
