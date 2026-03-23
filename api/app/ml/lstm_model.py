import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from app.ml.base_model import BaseStockModel, PredictionResult
from app.services.feature_engineer import FeatureEngineer


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = (attn_weights * out).sum(dim=1)
        return self.fc(context).squeeze(-1)


class LSTMModel(BaseStockModel):
    model_name = "lstm"

    def __init__(self, seq_len: int = 60, hidden_size: int = 128, num_layers: int = 2,
                 epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.fe = FeatureEngineer()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ohlcv = {"open", "high", "low", "close", "volume"}
        extra = [c for c in df.columns if c.lower() not in ohlcv]
        if len(extra) >= 10:
            return df
        df = self.fe.add_technical_indicators(df)
        df = self.fe.add_lag_features(df)
        df = self.fe.add_time_features(df)
        return df

    def _align_to_training_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure predict-time df has exactly the same columns as training (fill missing with 0).

        Note: self.feature_cols already includes 'close' (stored from prepare_sequences).
        """
        if self.feature_cols is None:
            return df
        df = df.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_cols]

    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        df_feat = self._build_features(df)
        X, y, self.scaler, self.feature_cols = self.fe.prepare_sequences(df_feat, self.seq_len, horizon)

        if len(X) == 0:
            raise ValueError("Not enough data to train LSTM. Need at least seq_len + horizon rows.")

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        input_size = X_train.shape[2]
        self.model = LSTMNet(input_size, self.hidden_size, self.num_layers).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        best_val_loss = float('inf')
        best_state = None
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(X_train_t))
            for i in range(0, len(X_train_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                pred = self.model(X_train_t[idx])
                loss = loss_fn(pred, y_train_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)
        return {"val_loss": round(best_val_loss, 6), "epochs": self.epochs}

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        df_feat = self._build_features(df)
        df_aligned = self._align_to_training_cols(df_feat)
        # Forward-fill NaN (FRED lag causes recent rows to have NaN macro values).
        # This is much better than dropping rows or filling with 0.
        df_clean = df_aligned.ffill().dropna()
        scaled = self.scaler.transform(df_clean)

        seq = torch.FloatTensor(scaled[-self.seq_len:]).unsqueeze(0).to(self.device)

        self.model.eval()
        predictions_scaled = []
        with torch.no_grad():
            current_seq = seq.clone()
            for _ in range(horizon):
                pred_scaled = self.model(current_seq).item()
                predictions_scaled.append(pred_scaled)
                new_row = current_seq[0, -1, :].clone()
                new_row[-1] = pred_scaled
                current_seq = torch.cat(
                    [current_seq[:, 1:, :], new_row.unsqueeze(0).unsqueeze(0)], dim=1
                )

        # The model outputs log returns (log(close[t+h] / close[t])).
        # Convert each prediction back to an absolute price.
        current_price = float(df["close"].iloc[-1])
        # Each autoregressive step predicts the log return from the current time.
        # We use a slight mean-reversion dampening per step.
        base_log_ret = predictions_scaled[0]  # 1-step log return
        step_log_ret = base_log_ret / max(horizon, 1)
        inverted = np.array([
            current_price * np.exp(step_log_ret * (i + 1) * (0.95 ** i))
            for i in range(horizon)
        ])

        last_date = df.index[-1]
        dates = [str(d.date()) for d in pd.bdate_range(last_date, periods=horizon + 1)[1:]]

        std_dev = float(np.std(inverted)) if len(inverted) > 1 else float(np.abs(inverted[0])) * 0.02
        mean_pred = float(np.mean(np.abs(inverted)))
        confidence = float(max(0.0, min(1.0, 1.0 - (std_dev / mean_pred)))) if mean_pred > 0 else 0.5

        return PredictionResult(
            dates=dates,
            predicted=[round(float(v), 2) for v in inverted],
            lower_ci=[round(float(v - 1.96 * std_dev), 2) for v in inverted],
            upper_ci=[round(float(v + 1.96 * std_dev), 2) for v in inverted],
            confidence=confidence,
            model=self.model_name,
        )

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), f"{path}_weights.pt")
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "input_size": self.model.lstm.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
            }, f)

    def load(self, path: str) -> None:
        with open(f"{path}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.scaler = meta["scaler"]
        self.feature_cols = meta["feature_cols"]
        self.seq_len = meta.get("seq_len", self.seq_len)
        self.hidden_size = meta["hidden_size"]
        self.num_layers = meta["num_layers"]
        self.model = LSTMNet(meta["input_size"], meta["hidden_size"], meta["num_layers"]).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{path}_weights.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()
