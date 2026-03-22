from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import TypedDict


class PredictionResult(TypedDict):
    dates: list[str]
    predicted: list[float]
    lower_ci: list[float]
    upper_ci: list[float]
    confidence: float
    model: str


class BaseStockModel(ABC):
    model_name: str = "base"

    @abstractmethod
    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        """Train on df. Returns metrics dict."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        """Predict next `horizon` trading days."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
