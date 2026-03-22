from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.routers import predict, backtest, stocks, auth

settings = get_settings()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="StockSage API",
    description="""
## StockSage — AI Stock Prediction API

Predict stock prices using an ensemble of LSTM and XGBoost models.

### Authentication
All endpoints except `/health` and `POST /v1/auth/keys` require an API key passed as a header:
```
X-API-Key: your_api_key_here
```

Get a free API key by calling `POST /v1/auth/keys` with your chosen name.

### Rate Limits
- Free tier: 100 requests/hour
- Pro tier: 10,000 requests/hour

### Models
| Model | Description |
|---|---|
| `lstm` | Long Short-Term Memory — sequential baseline with attention |
| `xgboost` | Gradient boosted trees — strong tabular baseline |
| `ensemble` | Meta-learner blending LSTM + XGBoost (recommended) |
    """,
    version="1.0.0",
    contact={"name": "StockSage", "url": "https://github.com/your-username/stocksage"},
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


@app.on_event("startup")
async def startup_event():
    """Create DB tables on startup if they don't exist."""
    from app.database import engine, Base
    # Import models so they are registered with Base.metadata
    import app.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
