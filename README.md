# StockSage — AI Stock Prediction Platform

![CI](https://github.com/your-username/stocksage/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)

> **Not financial advice.** StockSage is a portfolio / research project. All predictions and backtest results are for educational purposes only.

StockSage is a full-stack AI stock prediction platform — a **public REST API** (FastAPI on Koyeb) consumed by a **Next.js dashboard** (Vercel). It uses an ensemble of LSTM neural networks and XGBoost gradient-boosted trees to forecast stock prices up to 30 days ahead, with confidence intervals and walk-forward backtesting.

**100% free, no credit card required:** Koyeb (API + Postgres) + Upstash (Redis) + Vercel (frontend) + GitHub Actions (CI) = $0/month.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Vercel (free)                         │
│  Next.js 14 App                                          │
│  ├── Landing page                                        │
│  ├── Dashboard (candlestick chart + AI forecast)         │
│  ├── Backtesting page (equity curve + metrics)           │
│  └── API docs page (live "try it" console)               │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTPS
┌────────────────────────▼─────────────────────────────────┐
│                    Koyeb (free)                          │
│  FastAPI                                                 │
│  ├── POST /v1/predict/   — LSTM + XGBoost ensemble       │
│  ├── POST /v1/backtest/  — walk-forward backtesting      │
│  ├── GET  /v1/stocks/    — OHLCV + indicators            │
│  └── POST /v1/auth/keys  — API key management            │
│                                                          │
│  Koyeb Postgres ─── SQLAlchemy (async)                  │
│  Upstash Redis  ─── 1-hour OHLCV cache                  │
│  yfinance       ─── free market data                    │
│  ./models/      ─── local artifact cache                │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start (local, Docker Compose)

```bash
# 1. Clone
git clone https://github.com/your-username/stocksage.git
cd stocksage

# 2. Configure environment
cp .env.example .env
# Edit .env: set SECRET_KEY and API_KEY_SALT to random strings

# 3. Start all services
docker compose up -d

# 4. Verify API is up
curl http://localhost:8000/health

# 5. Open the dashboard
open http://localhost:3000
```

The API auto-creates database tables on startup. No manual migration step needed for local dev.

---

## Project Structure

```
stocksage/
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py         # App factory, CORS, routers
│   │   ├── config.py       # Pydantic settings
│   │   ├── database.py     # Async SQLAlchemy engine
│   │   ├── models/         # ORM: APIKey, Prediction, Stock
│   │   ├── routers/        # predict, backtest, stocks, auth
│   │   ├── services/       # DataFetcher, FeatureEngineer, Backtester, ModelRegistry
│   │   └── ml/             # LSTMModel, XGBoostModel, EnsembleModel
│   ├── tests/
│   ├── alembic/
│   ├── Dockerfile
│   └── koyeb.yaml
│
├── web/                    # Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx        # Landing page
│   │   ├── dashboard/      # Main app
│   │   ├── backtest/       # Backtesting UI
│   │   └── docs/           # API documentation
│   └── components/         # Charts, cards, forms
│
├── training/               # Offline training scripts
│   ├── train_all.py        # Train all tickers + log to MLflow
│   ├── train_single.py     # Quick single-ticker training
│   └── evaluate.py         # Walk-forward evaluation
│
├── docker-compose.yml
└── .env.example
```

---

## Using the Public API

Get a free API key (no account needed):

```bash
curl -X POST https://your-api.koyeb.app/v1/auth/keys \
  -H "Content-Type: application/json" \
  -d '{"name":"My App"}'
```

### Predict stock prices

```bash
curl -X POST https://your-api.koyeb.app/v1/predict/ \
  -H "X-API-Key: sk-your_key" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","model":"ensemble","horizon":5}'
```

Response:
```json
{
  "ticker": "AAPL",
  "model": "ensemble",
  "horizon": 5,
  "predictions": [
    {"date": "2024-12-02", "predicted": 189.42, "lower_ci": 183.12, "upper_ci": 195.72},
    ...
  ],
  "confidence": 0.84,
  "disclaimer": "Predictions are for informational purposes only. Not financial advice."
}
```

### Run a backtest

```bash
curl -X POST https://your-api.koyeb.app/v1/backtest/ \
  -H "X-API-Key: sk-your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "model": "ensemble",
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000,
    "transaction_cost_bps": 10
  }'
```

**Full API reference:** `/docs` or `/redoc` on your deployed API.

---

## Model Performance (2023 Holdout — approximate)

| Ticker | Model | RMSE | MAE | Sharpe |
|--------|-------|------|-----|--------|
| AAPL   | XGBoost  | $4.21 | $3.18 | 0.82 |
| AAPL   | LSTM     | $5.03 | $3.74 | 0.71 |
| AAPL   | Ensemble | $3.94 | $2.97 | 0.89 |
| MSFT   | XGBoost  | $6.38 | $4.91 | 0.76 |
| MSFT   | Ensemble | $5.87 | $4.45 | 0.83 |
| GOOGL  | XGBoost  | $3.12 | $2.44 | 0.74 |
| GOOGL  | Ensemble | $2.89 | $2.21 | 0.81 |

> These are illustrative numbers. Run `python training/evaluate.py` for real metrics on current data.

---

## Deployment

### API — Koyeb (free, no credit card)

1. Sign up at [koyeb.com](https://koyeb.com) with GitHub — no CC required
2. **New App** → Connect GitHub → set **Root directory** to `api/`
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Instance: **Free** (512MB RAM)
6. Add **Postgres** addon — Koyeb injects `DATABASE_URL` automatically
7. Add secrets in **Settings → Secrets**:
   - `SECRET_KEY` — `python -c "import secrets; print(secrets.token_hex(32))"`
   - `API_KEY_SALT` — another random string
   - `UPSTASH_REDIS_REST_URL` — from upstash.com free tier
   - `UPSTASH_REDIS_REST_TOKEN` — from upstash.com free tier
   - `ALLOWED_ORIGINS` — `https://your-app.vercel.app,http://localhost:3000`

### Redis — Upstash (free, no credit card)

1. Sign up at [upstash.com](https://upstash.com) with GitHub
2. Create Database → Redis → **Free tier**
3. Copy `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to Koyeb secrets

### Frontend — Vercel (free, no credit card)

```bash
cd web
npx vercel
```

Set env var: `NEXT_PUBLIC_API_URL=https://your-api.koyeb.app`

Every push to `main` auto-deploys.

### Important: Model artifact persistence

Koyeb's free tier uses ephemeral storage — the `./models/` directory resets on redeploy. The API re-trains models on first request after a cold deploy (~30s for XGBoost, ~2min for LSTM). This is acceptable for a portfolio project. The UI shows a "Training model..." loading state during this time.

To pre-warm models, run `training/train_all.py` locally and copy artifacts to the server, or trigger a prediction request for each demo ticker after deploy.

---

## Development

### Run tests

```bash
cd api
pip install -r requirements.txt aiosqlite
pytest tests/ -v
```

### Train models locally

```bash
# Train XGBoost for AAPL with 5-day horizon
python training/train_single.py --ticker AAPL --model xgboost --horizon 5

# Train all demo tickers
python training/train_all.py

# View MLflow experiments
mlflow ui  # then open http://localhost:5000
```

### API interactive docs

With the API running: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.111, Python 3.11 |
| ML | PyTorch (LSTM + attention), XGBoost, LightGBM |
| Data | yfinance, pandas-ta (25+ indicators) |
| DB | PostgreSQL + SQLAlchemy (async) + Alembic |
| Cache | Redis / Upstash |
| Experiment tracking | MLflow |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Charts | lightweight-charts (candlestick), Recharts (equity curve) |
| State | TanStack Query, Zustand |
| API auth | bcrypt API key hashing, slowapi rate limiting |
| CI/CD | GitHub Actions |
| Hosting | Koyeb + Vercel + Upstash — all free |

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT © 2024 — See [LICENSE](LICENSE) for details.

---

*StockSage is for educational and research purposes only. It is not financial advice. Always do your own research before making investment decisions.*