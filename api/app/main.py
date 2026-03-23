from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.routers import predict, backtest, stocks, auth, realtime, sentiment

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="StockSage API",
    description="""
## StockSage — AI Stock Prediction API

Predict stock prices using an ensemble of LSTM + XGBoost models trained on
technical indicators, macro features (FRED + Yahoo Finance), and live sentiment
(NewsAPI + Yahoo RSS + Finviz).

### Authentication
All endpoints except `/health` and `POST /v1/auth/keys` require:
```
X-API-Key: sk-your_key_here
```

### Real-time WebSocket
Connect to `ws://host/v1/realtime/ws/{ticker}?interval=5m` for live candles
and predicted next-candle overlays. No auth required on the WebSocket.

### Models
| Model | Description |
|---|---|
| `lstm` | Attention LSTM on 60-day sequences |
| `xgboost` | Gradient-boosted trees on 60+ engineered features |
| `ensemble` | LSTM + XGBoost blend (recommended) |
    """,
    version="2.0.0",
    contact={"name": "StockSage"},
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

app.include_router(auth.router,      prefix="/v1/auth",      tags=["Authentication"])
app.include_router(stocks.router,    prefix="/v1/stocks",    tags=["Stocks"])
app.include_router(predict.router,   prefix="/v1/predict",   tags=["Predictions"])
app.include_router(backtest.router,  prefix="/v1/backtest",  tags=["Backtesting"])
app.include_router(sentiment.router, prefix="/v1/sentiment", tags=["Sentiment & Macro"])
app.include_router(realtime.router,  prefix="/v1/realtime",  tags=["Real-time"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.on_event("startup")
async def startup_event():
    from app.database import engine, Base
    import app.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
