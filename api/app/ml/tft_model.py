"""
Temporal Fusion Transformer — placeholder that wraps the ensemble model.
Full TFT implementation via pytorch-forecasting requires significant data
preprocessing. This module provides a compatible interface for future use.
"""
from app.ml.ensemble_model import EnsembleModel
from app.ml.base_model import PredictionResult
import pandas as pd


class TFTModel(EnsembleModel):
    """
    TFT placeholder: currently delegates to the ensemble model.
    A full pytorch-forecasting TFT implementation can be dropped in here
    without changing any router or service code.
    """
    model_name = "tft"

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        result = super().predict(df, horizon)
        result["model"] = self.model_name
        return result
