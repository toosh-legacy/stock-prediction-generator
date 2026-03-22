# StockSage — Full-Stack Stock Prediction Platform
## Master Spec for Claude Code

> Build this entire project end-to-end. Follow every section in order.
> Do not skip any phase. Ask no clarifying questions — all decisions are made below.

---

## 1. Project Overview

**StockSage** is a full-stack AI stock prediction platform with:
- A **public REST API** (FastAPI, hosted on **Koyeb** — free forever, no credit card) that any developer can call
- A **Next.js web app** (hosted on **Vercel** — free forever, no credit card) that consumes that same public API
- A **model training pipeline** (runs locally) that trains LSTM, XGBoost, and an ensemble, tracks with MLflow, and caches artifacts to the **local filesystem** (persisted in Koyeb's ephemeral storage, re-trained on cold start if missing)
- A **backtesting engine** that evaluates every model on held-out data and reports Sharpe ratio, max drawdown, win rate, and cumulative return vs buy-and-hold

**Deployment stack is 100% free with no credit card required:**
| Service | Provider | Cost |
|---|---|---|
| Frontend | Vercel | Free forever |
| API + Postgres | Koyeb (Starter plan) | Free forever |
| Redis cache | Upstash | Free (10k req/day) |
| CI/CD | GitHub Actions | Free |
| Stock data | yfinance | Free |

The API is first-class: it has OpenAPI docs, API key auth, rate limiting, and is intended to be published publicly so other developers can build on top of it.

---

## 2. Monorepo Structure

```
stocksage/
├── api/                        # FastAPI backend (deployed to Koyeb — free, no CC)
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models/             # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── stock.py
│   │   │   ├── prediction.py
│   │   │   └── api_key.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py
│   │   │   ├── backtest.py
│   │   │   ├── stocks.py
│   │   │   └── auth.py
│   │   ├── services/
│   │   │   ├── data_fetcher.py
│   │   │   ├── feature_engineer.py
│   │   │   ├── model_registry.py
│   │   │   └── backtester.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   ├── lstm_model.py
│   │   │   ├── tft_model.py
│   │   │   ├── xgboost_model.py
│   │   │   └── ensemble_model.py
│   │   ├── tasks/
│   │   │   ├── train_pipeline.py
│   │   │   └── scheduler.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── rate_limit.py
│   ├── tests/
│   │   ├── test_predict.py
│   │   ├── test_backtest.py
│   │   └── test_features.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── koyeb.yaml
│
├── web/                        # Next.js 14 frontend (deployed to Vercel)
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx            # Landing / hero
│   │   ├── dashboard/
│   │   │   └── page.tsx        # Main app — ticker search + charts
│   │   ├── backtest/
│   │   │   └── page.tsx        # Backtest results page
│   │   ├── docs/
│   │   │   └── page.tsx        # API documentation page
│   │   └── api/
│   │       └── proxy/
│   │           └── route.ts    # Next.js route handler (proxies to FastAPI)
│   ├── components/
│   │   ├── ui/                 # shadcn/ui components
│   │   ├── charts/
│   │   │   ├── CandlestickChart.tsx
│   │   │   ├── PredictionOverlay.tsx
│   │   │   ├── EquityCurve.tsx
│   │   │   └── FeatureImportance.tsx
│   │   ├── TickerSearch.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── PredictionCard.tsx
│   │   ├── BacktestTable.tsx
│   │   └── ApiKeyManager.tsx
│   ├── lib/
│   │   ├── api.ts              # Typed API client
│   │   └── utils.ts
│   ├── types/
│   │   └── index.ts
│   ├── public/
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── package.json
│
├── training/                   # Standalone training scripts (run locally or cron)
│   ├── train_all.py
│   ├── train_single.py
│   ├── evaluate.py
│   └── upload_artifacts.py
│
├── docker-compose.yml          # Local dev: API + Postgres + Redis + MLflow
├── .env.example
└── README.md
```

---

## 3. Environment Variables

### `.env.example` (copy to `.env`, never commit `.env`)
```
# Database (local dev — Koyeb injects DATABASE_URL automatically in production)
DATABASE_URL=postgresql://postgres:password@localhost:5432/stocksage

# Redis — use Upstash in production (free, no CC)
# Local dev: redis://localhost:6379
# Production: get UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN from upstash.com
REDIS_URL=redis://localhost:6379
UPSTASH_REDIS_REST_URL=https://your-upstash-endpoint.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_upstash_token

# External APIs (all optional — yfinance works without any key)
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Model artifact storage — local filesystem (no S3 needed)
# Models are saved to ./models/ directory and cached in memory between requests
MODEL_ARTIFACTS_DIR=./models

# MLflow (local only — run `mlflow ui` to view experiments)
MLFLOW_TRACKING_URI=http://localhost:5000

# App
SECRET_KEY=generate_a_random_64_char_hex_string_here
API_KEY_SALT=another_random_string
ENVIRONMENT=development

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=3600

# CORS (comma-separated — add your Vercel URL here)
ALLOWED_ORIGINS=http://localhost:3000,https://your-app.vercel.app
```

---

## 4. Backend — FastAPI (`api/`)

### 4.1 `api/requirements.txt`
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy==2.0.30
alembic==1.13.1
asyncpg==0.29.0
psycopg2-binary==2.9.9
upstash-redis==1.1.0
pydantic==2.7.1
pydantic-settings==2.2.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
httpx==0.27.0
slowapi==0.1.9

# Data
yfinance==0.2.40
pandas==2.2.2
numpy==1.26.4
pandas-ta==0.3.14b
scikit-learn==1.4.2
scipy==1.13.0

# NLP / Sentiment
transformers==4.41.1
torch==2.3.0
vaderSentiment==3.3.2

# ML Models
xgboost==2.0.3
lightgbm==4.3.0
pytorch-forecasting==1.0.0
mlflow==2.13.0

# Backtesting
vectorbt==0.26.2

# Scheduling
apscheduler==3.10.4

# Testing
pytest==8.2.0
pytest-asyncio==0.23.7
httpx==0.27.0
```

### 4.2 `api/app/config.py`
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    # Upstash Redis (used in production on Koyeb — free, no CC)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    alpha_vantage_api_key: str = ""
    news_api_key: str = ""
    # Local filesystem for model artifacts — no S3 needed
    model_artifacts_dir: str = "./models"
    mlflow_tracking_uri: str = "http://localhost:5000"
    secret_key: str
    api_key_salt: str
    environment: str = "development"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 3600
    allowed_origins: list[str] = ["http://localhost:3000"]

    @property
    def using_upstash(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 4.3 `api/app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.database import engine, Base
from app.routers import predict, backtest, stocks, auth

settings = get_settings()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="StockSage API",
    description="""
## StockSage — AI Stock Prediction API

Predict stock prices using an ensemble of LSTM, Temporal Fusion Transformer, and XGBoost models.

### Authentication
All endpoints except `/health` require an API key passed as a header:
```
X-API-Key: your_api_key_here
```

Get a free API key at [stocksage.app/keys](https://stocksage.app/keys).

### Rate Limits
- Free tier: 100 requests/hour
- Pro tier: 10,000 requests/hour

### Models
| Model | Description |
|---|---|
| `lstm` | Long Short-Term Memory — sequential baseline |
| `tft` | Temporal Fusion Transformer — state-of-the-art multivariate |
| `xgboost` | Gradient boosted trees — strong tabular baseline |
| `ensemble` | Meta-learner blending all three (recommended) |
    """,
    version="1.0.0",
    contact={"name": "StockSage", "url": "https://stocksage.app"},
    license_info={"name": "MIT"},
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,     prefix="/v1/auth",     tags=["Authentication"])
app.include_router(stocks.router,   prefix="/v1/stocks",   tags=["Stocks"])
app.include_router(predict.router,  prefix="/v1/predict",  tags=["Predictions"])
app.include_router(backtest.router, prefix="/v1/backtest", tags=["Backtesting"])

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}
```

### 4.4 `api/app/database.py`
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.environment == "development",
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 4.5 `api/app/models/api_key.py`
```python
from sqlalchemy import Column, String, DateTime, Boolean, Integer
from sqlalchemy.sql import func
from app.database import Base
import uuid

class APIKey(Base):
    __tablename__ = "api_keys"

    id       = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash = Column(String, unique=True, nullable=False)  # bcrypt hash
    name     = Column(String, nullable=False)               # e.g. "My App"
    tier     = Column(String, default="free")               # free | pro
    is_active= Column(Boolean, default=True)
    requests_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used  = Column(DateTime(timezone=True), nullable=True)
```

### 4.6 `api/app/models/prediction.py`
```python
from sqlalchemy import Column, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base
import uuid

class Prediction(Base):
    __tablename__ = "predictions"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker      = Column(String, nullable=False, index=True)
    model_name  = Column(String, nullable=False)
    horizon_days= Column(String, nullable=False)  # "1d" | "5d" | "30d"
    predictions = Column(JSON, nullable=False)    # [{date, price, lower_ci, upper_ci}]
    confidence  = Column(Float)
    features_used = Column(JSON)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
```

### 4.7 `api/app/middleware/auth.py`
```python
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from app.models.api_key import APIKey
from app.database import get_db
from fastapi import Depends

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def verify_api_key(
    api_key: str = Security(API_KEY_HEADER),
    db: AsyncSession = Depends(get_db)
) -> APIKey:
    # Hash the incoming key and look it up
    result = await db.execute(select(APIKey).where(APIKey.is_active == True))
    keys = result.scalars().all()
    for key_record in keys:
        if pwd_context.verify(api_key, key_record.key_hash):
            return key_record
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key. Get one at https://stocksage.app/keys",
        headers={"WWW-Authenticate": "ApiKey"},
    )
```

### 4.8 `api/app/services/data_fetcher.py`
```python
import yfinance as yf
import pandas as pd
import json
from app.config import get_settings

settings = get_settings()

def _get_redis_client():
    """Returns Upstash Redis client in production, standard redis in local dev."""
    if settings.using_upstash:
        from upstash_redis import Redis
        return Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token
        )
    else:
        import redis
        return redis.from_url(settings.redis_url, decode_responses=True)

class DataFetcher:
    def __init__(self):
        self.redis = _get_redis_client()
        self.cache_ttl = 3600  # 1 hour

    def _cache_get(self, key: str):
        try:
            return self.redis.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: str, ttl: int):
        try:
            self.redis.setex(key, ttl, value)
        except Exception:
            pass  # Cache miss is acceptable — degrade gracefully

    async def get_ohlcv(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        cache_key = f"ohlcv:{ticker}:{period}:{start}:{end}"
        cached = self._cache_get(cache_key)
        if cached:
            return pd.read_json(cached)

        t = yf.Ticker(ticker)
        if start and end:
            df = t.history(start=start, end=end)
        else:
            df = t.history(period=period)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        df.index = df.index.tz_localize(None)
        df.columns = [c.lower() for c in df.columns]

        self._cache_set(cache_key, df.to_json(), self.cache_ttl)
        return df

    async def get_info(self, ticker: str) -> dict:
        cache_key = f"info:{ticker}"
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)

        t = yf.Ticker(ticker)
        info = t.info or {}
        self._cache_set(cache_key, json.dumps(info), 3600 * 24)
        return info
```

### 4.9 `api/app/services/feature_engineer.py`
```python
import pandas as pd
import numpy as np
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import httpx
from app.config import get_settings

settings = get_settings()

class FeatureEngineer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Trend
        df['sma_20']  = ta.sma(df['close'], length=20)
        df['sma_50']  = ta.sma(df['close'], length=50)
        df['ema_12']  = ta.ema(df['close'], length=12)
        df['ema_26']  = ta.ema(df['close'], length=26)
        # Momentum
        df['rsi']     = ta.rsi(df['close'], length=14)
        macd          = ta.macd(df['close'])
        df['macd']    = macd['MACD_12_26_9']
        df['macd_sig']= macd['MACDs_12_26_9']
        # Volatility
        bb            = ta.bbands(df['close'], length=20)
        df['bb_upper']= bb['BBU_20_2.0']
        df['bb_lower']= bb['BBL_20_2.0']
        df['bb_width']= (df['bb_upper'] - df['bb_lower']) / df['close']
        df['atr']     = ta.atr(df['high'], df['low'], df['close'], length=14)
        # Volume
        df['obv']     = ta.obv(df['close'], df['volume'])
        df['vwap']    = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        # Lag returns
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'return_{lag}d'] = df['close'].pct_change(lag)
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['month']       = df.index.month
        df['quarter']     = df.index.quarter
        df['is_month_end']= df.index.is_month_end.astype(int)
        return df

    def prepare_sequences(
        self, df: pd.DataFrame, seq_len: int = 60, horizon: int = 1
    ) -> tuple:
        """Returns (X, y) numpy arrays for sequence models."""
        feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume']]
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
        """Returns (X, y) for tree models."""
        feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume']]
        df_clean = df[feature_cols + ['close']].dropna()
        X = df_clean[feature_cols].values
        y = df_clean['close'].shift(-horizon).dropna().values
        X = X[:len(y)]
        return X, y, feature_cols
```

### 4.10 `api/app/ml/base_model.py`
```python
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
```

### 4.11 `api/app/ml/lstm_model.py`
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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
        # Attention over sequence
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

    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        df_feat = self.fe.add_technical_indicators(df)
        df_feat = self.fe.add_time_features(df_feat)
        X, y, self.scaler, self.feature_cols = self.fe.prepare_sequences(df_feat, self.seq_len, horizon)

        # 80/20 train/val split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t   = torch.FloatTensor(X_val).to(self.device)
        y_val_t   = torch.FloatTensor(y_val).to(self.device)

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
                idx = perm[i:i+self.batch_size]
                pred = self.model(X_train_t[idx])
                loss = loss_fn(pred, y_train_t[idx])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        return {"val_loss": best_val_loss, "epochs": self.epochs}

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        df_feat = self.fe.add_technical_indicators(df)
        df_feat = self.fe.add_time_features(df_feat)
        feature_cols_only = [c for c in df_feat.columns if c not in ['open','high','low','close','volume']]
        df_clean = df_feat[feature_cols_only + ['close']].dropna()
        scaled = self.scaler.transform(df_clean)

        # Use the last seq_len rows
        seq = torch.FloatTensor(scaled[-self.seq_len:]).unsqueeze(0).to(self.device)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            # Multi-step: autoregressively predict horizon steps
            current_seq = seq.clone()
            for _ in range(horizon):
                pred_scaled = self.model(current_seq).item()
                predictions.append(pred_scaled)
                # Shift sequence
                new_row = current_seq[0, -1, :].clone()
                new_row[-1] = pred_scaled
                current_seq = torch.cat([current_seq[:, 1:, :], new_row.unsqueeze(0).unsqueeze(0)], dim=1)

        # Inverse transform predictions
        close_idx = -1
        dummy = np.zeros((horizon, df_clean.shape[1]))
        dummy[:, close_idx] = predictions
        inverted = self.scaler.inverse_transform(dummy)[:, close_idx]

        import pandas as pd
        from datetime import timedelta
        last_date = df.index[-1]
        business_days = pd.bdate_range(last_date, periods=horizon+1)[1:]
        dates = [str(d.date()) for d in business_days]

        std_dev = float(np.std(inverted))
        return PredictionResult(
            dates=dates,
            predicted=[round(float(v), 2) for v in inverted],
            lower_ci=[round(float(v - 1.96 * std_dev), 2) for v in inverted],
            upper_ci=[round(float(v + 1.96 * std_dev), 2) for v in inverted],
            confidence=float(max(0.0, 1.0 - (std_dev / float(np.mean(inverted))))),
            model=self.model_name
        )

    def save(self, path: str) -> None:
        import pickle
        torch.save(self.model.state_dict(), f"{path}_weights.pt")
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_cols": self.feature_cols,
                         "input_size": self.model.lstm.input_size,
                         "hidden_size": self.hidden_size, "num_layers": self.num_layers}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(f"{path}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.scaler = meta["scaler"]
        self.feature_cols = meta["feature_cols"]
        self.model = LSTMNet(meta["input_size"], meta["hidden_size"], meta["num_layers"]).to(self.device)
        self.model.load_state_dict(torch.load(f"{path}_weights.pt", map_location=self.device))
```

### 4.12 `api/app/ml/xgboost_model.py`
```python
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
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=50, eval_metric="mae", random_state=42
        )
        self.model.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
            verbose=False
        )
        preds = self.model.predict(X[split:])
        mae = float(np.mean(np.abs(preds - y[split:])))
        return {"mae": mae, "n_estimators": self.model.best_iteration}

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        df_feat = self.fe.add_technical_indicators(df)
        df_feat = self.fe.add_time_features(df_feat)
        df_clean = df_feat[self.feature_cols].dropna()
        last_features = df_clean.iloc[-1:].values

        preds = [float(self.model.predict(last_features)[0])]
        importances = dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))

        dates = [str(d.date()) for d in pd.bdate_range(df.index[-1], periods=horizon+1)[1:]]
        std = float(np.std(df['close'].pct_change().dropna()) * preds[0])

        return PredictionResult(
            dates=dates, predicted=[round(preds[0], 2)] * horizon,
            lower_ci=[round(preds[0] - 1.96*std, 2)] * horizon,
            upper_ci=[round(preds[0] + 1.96*std, 2)] * horizon,
            confidence=0.7, model=self.model_name
        )

    def get_feature_importance(self) -> dict:
        return dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))

    def save(self, path: str) -> None:
        self.model.save_model(f"{path}.json")
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({"feature_cols": self.feature_cols}, f)

    def load(self, path: str) -> None:
        self.model = xgb.XGBRegressor()
        self.model.load_model(f"{path}.json")
        with open(f"{path}_meta.pkl", "rb") as f:
            self.feature_cols = pickle.load(f)["feature_cols"]
```

### 4.13 `api/app/ml/ensemble_model.py`
```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import pickle
from app.ml.base_model import BaseStockModel, PredictionResult
from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel

class EnsembleModel(BaseStockModel):
    model_name = "ensemble"

    def __init__(self):
        self.lstm = LSTMModel(epochs=30)
        self.xgb  = XGBoostModel()
        self.meta_learner = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)

    def train(self, df: pd.DataFrame, horizon: int = 1) -> dict:
        lstm_metrics = self.lstm.train(df, horizon)
        xgb_metrics  = self.xgb.train(df, horizon)

        # Build meta-features from OOF predictions (last 20% of training data)
        split = int(len(df) * 0.6)
        train_df, oof_df = df.iloc[:split], df.iloc[split:]

        self.lstm.train(train_df, horizon)
        self.xgb.train(train_df, horizon)

        lstm_pred = self.lstm.predict(oof_df, horizon)['predicted'][0]
        xgb_pred  = self.xgb.predict(oof_df, horizon)['predicted'][0]
        actual    = float(oof_df['close'].iloc[-1])

        # Re-train base models on full data
        self.lstm.train(df, horizon)
        self.xgb.train(df, horizon)

        return {"lstm": lstm_metrics, "xgb": xgb_metrics, "status": "trained"}

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> PredictionResult:
        lstm_result = self.lstm.predict(df, horizon)
        xgb_result  = self.xgb.predict(df, horizon)

        blended = [
            round(0.5 * l + 0.5 * x, 2)
            for l, x in zip(lstm_result['predicted'], xgb_result['predicted'])
        ]
        lower = [min(l, x) for l, x in zip(lstm_result['lower_ci'], xgb_result['lower_ci'])]
        upper = [max(l, x) for l, x in zip(lstm_result['upper_ci'], xgb_result['upper_ci'])]

        return PredictionResult(
            dates=lstm_result['dates'], predicted=blended,
            lower_ci=lower, upper_ci=upper,
            confidence=float(np.mean([lstm_result['confidence'], xgb_result['confidence']])),
            model=self.model_name
        )

    def save(self, path: str) -> None:
        self.lstm.save(f"{path}_lstm")
        self.xgb.save(f"{path}_xgb")

    def load(self, path: str) -> None:
        self.lstm.load(f"{path}_lstm")
        self.xgb.load(f"{path}_xgb")
```

### 4.14 `api/app/services/backtester.py`
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    ticker: str
    model: str
    start_date: str
    end_date: str
    total_return: float
    benchmark_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: list[dict]  # [{date, portfolio_value, benchmark_value}]

class Backtester:
    def __init__(self, initial_capital: float = 10_000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run(
        self,
        df: pd.DataFrame,
        predictions: pd.Series,
        ticker: str = "UNKNOWN",
        model: str = "unknown"
    ) -> BacktestResult:
        """
        Simple long/short strategy: buy if predicted > current, sell otherwise.
        df must have 'close' column. predictions indexed same as df.
        """
        df = df.copy()
        df['signal'] = np.where(predictions.values > df['close'].values, 1, -1)
        df['daily_return'] = df['close'].pct_change()
        df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
        df['strategy_return'] -= abs(df['signal'].diff()) * self.transaction_cost / 2

        df.dropna(inplace=True)

        portfolio = (1 + df['strategy_return']).cumprod() * self.initial_capital
        benchmark = (1 + df['daily_return']).cumprod() * self.initial_capital

        total_return    = float((portfolio.iloc[-1] / self.initial_capital) - 1)
        bench_return    = float((benchmark.iloc[-1] / self.initial_capital) - 1)
        excess_returns  = df['strategy_return'] - 0.02 / 252  # risk-free rate daily
        sharpe          = float(np.sqrt(252) * excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0
        downside        = excess_returns[excess_returns < 0].std()
        sortino         = float(np.sqrt(252) * excess_returns.mean() / downside) if downside != 0 else 0
        rolling_max     = portfolio.cummax()
        drawdown        = (portfolio - rolling_max) / rolling_max
        max_drawdown    = float(drawdown.min())
        wins            = df['strategy_return'][df['signal'].shift(1) != 0] > 0
        win_rate        = float(wins.mean()) if len(wins) > 0 else 0
        total_trades    = int(abs(df['signal'].diff()).sum() / 2)

        equity_curve = [
            {
                "date": str(idx.date()),
                "portfolio": round(float(p), 2),
                "benchmark": round(float(b), 2),
            }
            for idx, p, b in zip(df.index, portfolio, benchmark)
        ]

        return BacktestResult(
            ticker=ticker, model=model,
            start_date=str(df.index[0].date()), end_date=str(df.index[-1].date()),
            total_return=round(total_return, 4), benchmark_return=round(bench_return, 4),
            sharpe_ratio=round(sharpe, 3), sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_drawdown, 4), win_rate=round(win_rate, 4),
            total_trades=total_trades, equity_curve=equity_curve
        )
```

### 4.15 `api/app/routers/predict.py`
```python
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
from app.services.feature_engineer import FeatureEngineer
from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel
from app.ml.ensemble_model import EnsembleModel
from typing import Literal

router = APIRouter()
fetcher = DataFetcher()

class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock ticker symbol")
    model: Literal["lstm", "xgboost", "ensemble"] = Field("ensemble", description="Model to use")
    horizon: int = Field(5, ge=1, le=30, description="Days to predict ahead (1–30)")

class PredictionResponse(BaseModel):
    ticker: str
    model: str
    horizon: int
    predictions: list[dict]
    confidence: float
    disclaimer: str = "Predictions are for informational purposes only. Not financial advice."

@router.post("/", response_model=PredictionResponse, summary="Predict stock prices")
async def predict_stock(
    body: PredictionRequest,
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Predict future closing prices for a given ticker.

    Returns a list of `{date, predicted, lower_ci, upper_ci}` objects for the requested horizon.

    **Note:** Models are pre-trained on 2 years of daily data. The first request for a
    ticker may train a model on-the-fly (~30s). Subsequent requests use cached models.
    """
    try:
        df = await fetcher.get_ohlcv(body.ticker.upper(), period="2y")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_map = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}
    model_cls = model_map[body.model]
    m = model_cls()
    m.train(df, horizon=body.horizon)
    result = m.predict(df, horizon=body.horizon)

    return PredictionResponse(
        ticker=body.ticker.upper(),
        model=body.model,
        horizon=body.horizon,
        predictions=[
            {"date": d, "predicted": p, "lower_ci": l, "upper_ci": u}
            for d, p, l, u in zip(
                result["dates"], result["predicted"],
                result["lower_ci"], result["upper_ci"]
            )
        ],
        confidence=result["confidence"]
    )
```

### 4.16 `api/app/routers/backtest.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel
from app.ml.ensemble_model import EnsembleModel
from app.services.backtester import Backtester
import pandas as pd
import numpy as np

router = APIRouter()
fetcher = DataFetcher()

class BacktestRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    model: Literal["lstm", "xgboost", "ensemble"] = "ensemble"
    start_date: str = Field("2022-01-01", example="2022-01-01")
    end_date: str = Field("2023-12-31", example="2023-12-31")
    initial_capital: float = Field(10000.0, ge=1000)
    transaction_cost_bps: float = Field(10, ge=0, le=100, description="Transaction cost in basis points")

@router.post("/", summary="Backtest a model on historical data")
async def backtest(
    body: BacktestRequest,
    api_key: APIKey = Depends(verify_api_key)
):
    try:
        df = await fetcher.get_ohlcv(body.ticker.upper(), start=body.start_date, end=body.end_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_map = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}
    m = model_map[body.model]()

    train_df = df.iloc[:int(len(df)*0.7)]
    test_df  = df.iloc[int(len(df)*0.7):]

    m.train(train_df, horizon=1)

    preds = []
    for i in range(len(test_df)):
        context = pd.concat([train_df, test_df.iloc[:i]])
        result  = m.predict(context, horizon=1)
        preds.append(result["predicted"][0])

    pred_series = pd.Series(preds, index=test_df.index)
    bt = Backtester(initial_capital=body.initial_capital, transaction_cost=body.transaction_cost_bps/10000)
    result = bt.run(test_df, pred_series, ticker=body.ticker.upper(), model=body.model)

    return result.__dict__
```

### 4.17 `api/app/routers/stocks.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from app.middleware.auth import verify_api_key
from app.models.api_key import APIKey
from app.services.data_fetcher import DataFetcher
import yfinance as yf

router = APIRouter()
fetcher = DataFetcher()

@router.get("/{ticker}", summary="Get stock info and recent OHLCV")
async def get_stock(ticker: str, period: str = "1mo", api_key: APIKey = Depends(verify_api_key)):
    try:
        df = await fetcher.get_ohlcv(ticker.upper(), period=period)
        info = await fetcher.get_info(ticker.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "ohlcv": [
            {"date": str(idx.date()), "open": round(float(r["open"]),2),
             "high": round(float(r["high"]),2), "low": round(float(r["low"]),2),
             "close": round(float(r["close"]),2), "volume": int(r["volume"])}
            for idx, r in df.iterrows()
        ]
    }

@router.get("/{ticker}/indicators", summary="Get technical indicators")
async def get_indicators(ticker: str, api_key: APIKey = Depends(verify_api_key)):
    from app.services.feature_engineer import FeatureEngineer
    df = await fetcher.get_ohlcv(ticker.upper(), period="6mo")
    fe = FeatureEngineer()
    df_feat = fe.add_technical_indicators(df)
    latest = df_feat.iloc[-1].dropna().to_dict()
    return {"ticker": ticker.upper(), "date": str(df_feat.index[-1].date()), "indicators": {k: round(float(v),4) for k, v in latest.items()}}
```

### 4.18 `api/app/routers/auth.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from passlib.context import CryptContext
from app.database import get_db
from app.models.api_key import APIKey
import secrets
import uuid

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class CreateKeyRequest(BaseModel):
    name: str

class CreateKeyResponse(BaseModel):
    api_key: str
    key_id: str
    message: str = "Store this key securely — it will not be shown again."

@router.post("/keys", response_model=CreateKeyResponse, summary="Generate a new API key")
async def create_api_key(body: CreateKeyRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate a new API key. The raw key is returned once and never stored in plaintext.
    """
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    key_hash = pwd_context.hash(raw_key)
    record = APIKey(id=str(uuid.uuid4()), key_hash=key_hash, name=body.name, tier="free")
    db.add(record)
    await db.commit()
    return CreateKeyResponse(api_key=raw_key, key_id=record.id)

@router.delete("/keys/{key_id}", summary="Revoke an API key")
async def revoke_api_key(key_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    key.is_active = False
    await db.commit()
    return {"message": "Key revoked"}
```

### 4.19 `api/Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.20 `api/koyeb.yaml`
```yaml
# Koyeb deployment config — free Starter plan, no credit card required
# Deploy via: koyeb app init stocksage-api --git github.com/YOUR_USERNAME/stocksage --git-branch main --git-build-command "pip install -r requirements.txt" --git-run-command "uvicorn app.main:app --host 0.0.0.0 --port 8000" --ports 8000:http --routes /:8000
# Or connect GitHub repo directly from the Koyeb dashboard (easier)

name: stocksage-api
services:
  - name: api
    type: web
    instance_type: free          # 512MB RAM, 0.1 vCPU — free forever, no CC
    ports:
      - port: 8000
        protocol: http
    routes:
      - path: /
        port: 8000
    build:
      type: dockerfile
      dockerfile: Dockerfile
    env:
      # DATABASE_URL is auto-injected by Koyeb when you attach the free Postgres addon
      - key: ENVIRONMENT
        value: production
      - key: SECRET_KEY
        secret: stocksage-secret-key        # set in Koyeb dashboard → Secrets
      - key: API_KEY_SALT
        secret: stocksage-api-key-salt
      - key: UPSTASH_REDIS_REST_URL
        secret: upstash-redis-url           # from upstash.com free tier
      - key: UPSTASH_REDIS_REST_TOKEN
        secret: upstash-redis-token
      - key: ALLOWED_ORIGINS
        value: "https://your-app.vercel.app,http://localhost:3000"
    health_checks:
      - path: /health
        port: 8000
        initial_delay_seconds: 30
        period_seconds: 30
```

**To deploy on Koyeb (step by step):**
1. Sign up at koyeb.com — no credit card needed, just GitHub login
2. Create a new App → connect your GitHub repo → select the `api/` subfolder as root
3. Set build command: `pip install -r requirements.txt`
4. Set run command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Select **Free** instance type (512MB)
6. Add a **Postgres** database addon from the Koyeb dashboard (free, injected as `DATABASE_URL`)
7. Add secrets in Koyeb dashboard → Settings → Secrets: `SECRET_KEY`, `API_KEY_SALT`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`
8. Deploy — your API is live at `https://stocksage-api-YOUR_USERNAME.koyeb.app`

---

## 5. Frontend — Next.js 14 (`web/`)

### 5.1 `web/package.json`
```json
{
  "name": "stocksage-web",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.3",
    "react": "^18",
    "react-dom": "^18",
    "@tanstack/react-query": "^5.37.1",
    "axios": "^1.7.2",
    "recharts": "^2.12.7",
    "lightweight-charts": "^4.1.6",
    "lucide-react": "^0.383.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.3.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-toast": "^1.1.5",
    "framer-motion": "^11.2.10",
    "zustand": "^4.5.2",
    "date-fns": "^3.6.0"
  },
  "devDependencies": {
    "typescript": "^5",
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "tailwindcss": "^3.4.4",
    "postcss": "^8",
    "autoprefixer": "^10",
    "eslint": "^8",
    "eslint-config-next": "14.2.3"
  }
}
```

### 5.2 `web/next.config.ts`
```typescript
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL}/v1/:path*`,
      },
    ]
  },
}

export default nextConfig
```

### 5.3 `web/lib/api.ts`
```typescript
import axios from 'axios'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

apiClient.interceptors.request.use((config) => {
  const key = localStorage.getItem('stocksage_api_key')
  if (key) config.headers['X-API-Key'] = key
  return config
})

export interface OHLCVPoint {
  date: string; open: number; high: number; low: number; close: number; volume: number
}

export interface PredictionPoint {
  date: string; predicted: number; lower_ci: number; upper_ci: number
}

export interface PredictionResponse {
  ticker: string; model: string; horizon: number
  predictions: PredictionPoint[]; confidence: number; disclaimer: string
}

export interface BacktestResponse {
  ticker: string; model: string; start_date: string; end_date: string
  total_return: number; benchmark_return: number; sharpe_ratio: number
  sortino_ratio: number; max_drawdown: number; win_rate: number
  total_trades: number; equity_curve: Array<{date: string; portfolio: number; benchmark: number}>
}

export interface StockResponse {
  ticker: string; name: string; sector: string
  market_cap: number | null; pe_ratio: number | null; ohlcv: OHLCVPoint[]
}

export const stocksageApi = {
  getStock: (ticker: string, period = '3mo') =>
    apiClient.get<StockResponse>(`/v1/stocks/${ticker}`, { params: { period } }).then(r => r.data),

  predict: (ticker: string, model = 'ensemble', horizon = 5) =>
    apiClient.post<PredictionResponse>('/v1/predict/', { ticker, model, horizon }).then(r => r.data),

  backtest: (params: {
    ticker: string; model: string; start_date: string; end_date: string
    initial_capital: number; transaction_cost_bps: number
  }) => apiClient.post<BacktestResponse>('/v1/backtest/', params).then(r => r.data),

  getIndicators: (ticker: string) =>
    apiClient.get(`/v1/stocks/${ticker}/indicators`).then(r => r.data),

  createApiKey: (name: string) =>
    apiClient.post('/v1/auth/keys', { name }).then(r => r.data),
}
```

### 5.4 `web/app/layout.tsx`
```tsx
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { QueryProvider } from '@/components/providers/QueryProvider'
import { Toaster } from '@/components/ui/toaster'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'StockSage — AI Stock Prediction',
  description: 'Predict stock prices using LSTM, Temporal Fusion Transformer, and XGBoost ensemble models.',
  openGraph: {
    title: 'StockSage',
    description: 'AI-powered stock prediction platform',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <QueryProvider>
          {children}
          <Toaster />
        </QueryProvider>
      </body>
    </html>
  )
}
```

### 5.5 `web/app/page.tsx` — Landing Page
Build a beautiful dark-themed landing page with:
- Hero: "Predict the market with AI" tagline, animated gradient text, CTA buttons ("Try the App" → `/dashboard`, "View API Docs" → `/docs`)
- Feature cards: Data pipeline, Model zoo, Backtesting, Public API
- Live demo ticker widget showing a real prediction for AAPL
- "How it works" section with 4 steps
- API code snippet showing a real `curl` request
- Footer with GitHub link

### 5.6 `web/app/dashboard/page.tsx` — Main App
Build the full dashboard with these sections:

**Top bar:**
- Logo + nav (Dashboard | Backtest | API Docs)
- API key indicator (green dot if set, "Set API Key" button if not)

**Main content — left panel (60%):**
- `TickerSearch` component: text input, debounced, validates against API
- `CandlestickChart` component (use `lightweight-charts`): OHLCV candlestick chart with volume bars below
- `PredictionOverlay`: after prediction runs, overlay the prediction line + CI band on the chart in a different color

**Right panel (40%):**
- `ModelSelector`: radio cards for lstm / xgboost / ensemble with short descriptions
- Horizon slider: 1 to 30 days
- "Run Prediction" button (loading state while fetching)
- `PredictionCard`: shows predicted price, % change, confidence score, CI range for each predicted day
- `FeatureImportance`: horizontal bar chart of top 10 features (from XGBoost importance or SHAP approximation)
- Stock info card: company name, sector, market cap, P/E

### 5.7 `web/app/backtest/page.tsx` — Backtesting Page
Build a form + results page:

**Form:**
- Ticker input
- Model select (lstm / xgboost / ensemble)
- Date range pickers (start / end)
- Initial capital input
- Transaction cost (basis points) input
- "Run Backtest" button

**Results (shown after fetch):**
- 4 metric cards: Total Return vs Benchmark, Sharpe Ratio, Max Drawdown, Win Rate — color coded (green/red)
- `EquityCurve` component: line chart with two lines (portfolio vs benchmark) from `recharts`
- Stats table: start/end date, total trades, Sortino ratio
- Disclaimer banner

### 5.8 `web/app/docs/page.tsx` — API Docs Page
Build a clean API documentation page:
- Introduction section with base URL and authentication instructions
- Code snippets for getting an API key (form on the page — calls `POST /v1/auth/keys`)
- Endpoint reference table for all 6 endpoints
- Interactive "Try it" section: input ticker + model + horizon, run request, show raw JSON response
- Rate limits table (free vs pro)
- SDK snippets in Python, JavaScript, and curl for each endpoint

### 5.9 `web/components/charts/CandlestickChart.tsx`
```tsx
'use client'
import { useEffect, useRef } from 'react'
import { createChart, ColorType, CandlestickSeries, HistogramSeries } from 'lightweight-charts'
import type { OHLCVPoint, PredictionPoint } from '@/lib/api'

interface Props {
  data: OHLCVPoint[]
  predictions?: PredictionPoint[]
  ticker: string
}

export function CandlestickChart({ data, predictions, ticker }: Props) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data.length) return

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 400,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#9ca3af' },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      crosshair: { mode: 1 },
      timeScale: { borderColor: '#374151' },
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    })

    candleSeries.setData(data.map(d => ({
      time: d.date, open: d.open, high: d.high, low: d.low, close: d.close
    })))

    if (predictions?.length) {
      const predSeries = chart.addSeries(CandlestickSeries, { // use line series
        color: '#818cf8', lineWidth: 2,
      })
      // Add prediction line + CI area
    }

    chart.timeScale().fitContent()
    return () => chart.remove()
  }, [data, predictions])

  return <div ref={chartRef} className="w-full rounded-lg overflow-hidden" />
}
```

### 5.10 `web/components/ApiKeyManager.tsx`
```tsx
'use client'
import { useState, useEffect } from 'react'
import { stocksageApi } from '@/lib/api'

export function ApiKeyManager() {
  const [key, setKey] = useState('')
  const [saved, setSaved] = useState(false)
  const [generating, setGenerating] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('stocksage_api_key')
    if (stored) { setKey(stored); setSaved(true) }
  }, [])

  const saveKey = () => {
    localStorage.setItem('stocksage_api_key', key)
    setSaved(true)
  }

  const generateKey = async () => {
    setGenerating(true)
    try {
      const result = await stocksageApi.createApiKey('My StockSage Key')
      setKey(result.api_key)
      localStorage.setItem('stocksage_api_key', result.api_key)
      setSaved(true)
    } finally { setGenerating(false) }
  }

  return (
    <div className="flex items-center gap-3">
      <div className={`w-2 h-2 rounded-full ${saved ? 'bg-green-400' : 'bg-gray-500'}`} />
      {!saved ? (
        <button onClick={generateKey} disabled={generating}
          className="text-sm text-indigo-400 hover:text-indigo-300">
          {generating ? 'Generating...' : 'Get free API key'}
        </button>
      ) : (
        <span className="text-sm text-gray-400">API key active</span>
      )}
    </div>
  )
}
```

---

## 6. Docker Compose (local dev only)

> This is **local development only**. In production, Koyeb provides Postgres and Upstash provides Redis — no Docker needed in the cloud.

### `docker-compose.yml`
```yaml
version: '3.9'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: stocksage
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports: ["5432:5432"]
    volumes: [postgres_data:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlartifacts
    volumes: [mlflow_data:/mlartifacts]

  api:
    build: ./api
    ports: ["8000:8000"]
    env_file: .env
    depends_on:
      postgres: { condition: service_healthy }
      redis:    { condition: service_healthy }
    volumes:
      - ./api:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  web:
    image: node:20-alpine
    working_dir: /app
    ports: ["3000:3000"]
    volumes: [./web:/app]
    command: sh -c "npm install && npm run dev"
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000

volumes:
  postgres_data:
  mlflow_data:
```

---

## 7. Database Migrations

Use **Alembic** for migrations.

### Initial setup commands (run these after `docker compose up`):
```bash
cd api
alembic init alembic
# Edit alembic/env.py to use DATABASE_URL from settings
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

### `api/alembic/env.py` — add this to connect to async engine:
```python
from app.config import get_settings
from app.database import Base
from app.models import api_key, prediction, stock  # import all models

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata
```

---

## 8. Training Pipeline

### `training/train_all.py`
```python
"""
Run this script to train all models for a list of tickers.
Artifacts are saved to the local ./models/ directory and metrics logged to MLflow.

Usage:
    python training/train_all.py --tickers AAPL MSFT GOOGL TSLA NVDA --horizon 5
"""
import argparse, mlflow, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from app.ml.lstm_model import LSTMModel
from app.ml.xgboost_model import XGBoostModel
from app.ml.ensemble_model import EnsembleModel
import yfinance as yf

MODELS = {"lstm": LSTMModel, "xgboost": XGBoostModel, "ensemble": EnsembleModel}

def train_ticker(ticker: str, horizon: int, models: list[str]):
    df = yf.Ticker(ticker).history(period="2y")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)

    for model_name in models:
        print(f"  Training {model_name} for {ticker}...")
        with mlflow.start_run(run_name=f"{ticker}_{model_name}_h{horizon}"):
            mlflow.set_tags({"ticker": ticker, "model": model_name, "horizon": horizon})
            m = MODELS[model_name]()
            metrics = m.train(df, horizon=horizon)
            mlflow.log_params({"horizon": horizon, "ticker": ticker})
            mlflow.log_metrics(metrics)

            # Save artifacts locally to MODEL_ARTIFACTS_DIR
            artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR", "./models")
            os.makedirs(artifacts_dir, exist_ok=True)
            local_path = os.path.join(artifacts_dir, f"{ticker}_{model_name}_h{horizon}")
            m.save(local_path)

            print(f"    Done. Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["lstm", "xgboost", "ensemble"])
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("stocksage-training")

    for ticker in args.tickers:
        print(f"\nTraining {ticker}...")
        train_ticker(ticker, args.horizon, args.models)
```

---

## 9. Tests

### `api/tests/test_predict.py`
```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_predict_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/v1/predict/", json={"ticker": "AAPL", "model": "xgboost", "horizon": 1})
    assert r.status_code == 403

@pytest.mark.asyncio
async def test_predict_invalid_ticker(valid_api_key):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"X-API-Key": valid_api_key}) as ac:
        r = await ac.post("/v1/predict/", json={"ticker": "INVALIDTICKER123", "model": "xgboost", "horizon": 1})
    assert r.status_code == 404
```

---

## 10. Deployment

### 10.1 Koyeb (API) — Free, no credit card
1. Sign up at **koyeb.com** with GitHub — no CC required
2. New App → Connect GitHub → select your repo → set **Root directory** to `api/`
3. Build command: `pip install -r requirements.txt`
4. Run command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Instance type: **Free** (512MB RAM)
6. Click **Add Database** → PostgreSQL → Free → Koyeb auto-injects `DATABASE_URL`
7. Go to **Settings → Secrets** and add:
   - `SECRET_KEY` (run `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `API_KEY_SALT` (another random string)
   - `UPSTASH_REDIS_REST_URL` (from upstash.com free tier)
   - `UPSTASH_REDIS_REST_TOKEN` (from upstash.com free tier)
   - `ALLOWED_ORIGINS` = `https://your-app.vercel.app,http://localhost:3000`
8. Deploy → your API is live at `https://stocksage-api-USERNAME.koyeb.app`
9. Run Alembic migrations via Koyeb's one-time job or the Koyeb shell

### 10.2 Upstash Redis — Free, no credit card
1. Sign up at **upstash.com** with GitHub — no CC required
2. Create Database → Redis → Free tier → Select closest region
3. Copy `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` from the dashboard
4. Add both as secrets in Koyeb (step 7 above) and as env vars in Vercel

### 10.3 Vercel (Frontend) — Free, no credit card
1. `cd web && npx vercel` — or connect GitHub repo from vercel.com dashboard
2. Set environment variable in Vercel dashboard:
   - `NEXT_PUBLIC_API_URL` = `https://stocksage-api-USERNAME.koyeb.app`
3. Every push to `main` auto-deploys

### 10.4 GitHub Actions CI/CD
Create `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test-api:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: api
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  test-web:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: npm run build
```

---

## 11. README.md (top level)

Write a professional README with:
- Project badges (build status, license, Python version, Next.js version)
- Screenshot/gif of the dashboard
- "What is this?" summary
- Architecture diagram (text)
- Quick start (5 commands to run locally with Docker Compose)
- API documentation link
- Deployment guide (Koyeb free tier setup + Vercel + Upstash — all no credit card)
- Model performance table (RMSE, MAE, Sharpe for AAPL/MSFT/GOOGL on 2023 holdout)
- Tech stack section
- "Using the Public API" section with curl examples
- Contributing guide
- License (MIT)

---

## 12. Build Order for Claude Code

Execute in this exact order:

1. **Scaffold monorepo**: Create all directories and empty files from the structure in section 2
2. **`.env.example`** and **`docker-compose.yml`**: Create first so dev env can start
3. **API — models & database**: `database.py`, `models/`, `alembic/`
4. **API — services**: `data_fetcher.py`, `feature_engineer.py`
5. **API — ML models**: `base_model.py`, `lstm_model.py`, `xgboost_model.py`, `ensemble_model.py`
6. **API — backtester**: `backtester.py`
7. **API — routers + middleware**: `auth.py` middleware, then all 4 routers
8. **API — `main.py`**: Wire everything together
9. **API — Dockerfile + koyeb.yaml**: Containerize
10. **API — tests**: Write and verify all tests pass
11. **Frontend — `package.json` + config files**: Scaffold Next.js project
12. **Frontend — types + api client**: `types/index.ts`, `lib/api.ts`
13. **Frontend — components**: Charts, cards, forms (bottom-up)
14. **Frontend — pages**: Layout → Landing → Dashboard → Backtest → Docs
15. **Training scripts**: `training/train_all.py`
16. **CI/CD**: `.github/workflows/ci.yml`
17. **README.md**: Final, comprehensive

---

## 13. Important Notes

- **Never commit `.env`** — only `.env.example`
- **The API is the product** — the Next.js app is a consumer of the API, just like any external developer. Never hardcode business logic in the frontend.
- **Model training is slow** — on first prediction for a new ticker, train on the fly and cache the model to `MODEL_ARTIFACTS_DIR`. Show a loading state in the UI. For demo tickers (AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META), pre-train and cache artifacts before deploying.
- **No S3 needed** — models are saved to the local filesystem (`./models/`). On Koyeb's free tier the filesystem is ephemeral (resets on redeploy), so the API re-trains on first request after a cold deploy. This is acceptable for a portfolio project — mention it honestly in your README.
- **Redis graceful degradation** — the `DataFetcher` wraps all cache calls in try/except. If Upstash is unavailable, the API degrades gracefully and fetches fresh data from yfinance. This means rate limits won't be enforced in degraded mode — acceptable for a personal project.
- **yfinance is free and sufficient** — do not require users to set up an Alpha Vantage key to run the app.
- **Confidence scores** — calculate as `1 - (std_dev / mean_prediction)` clamped to [0, 1]. Be honest: these are statistical confidence intervals, not certainty.
- **Disclaimer** — always show "Not financial advice" on predictions and backtests.
- **CORS** — the Koyeb API must allow the Vercel domain. Set `ALLOWED_ORIGINS` in Koyeb secrets.
- **Koyeb abuse prevention** — if your account gets flagged (rare for legit projects), email their support explaining your use case. It gets resolved quickly.
- **Zero dollar, zero credit card** — Vercel (frontend) + Koyeb (API + Postgres) + Upstash (Redis) + GitHub (CI/CD) = complete production stack at $0.00/month with no payment method required.
