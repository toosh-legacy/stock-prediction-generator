import axios from 'axios'
import type {
  StockResponse,
  PredictionResponse,
  BacktestResponse,
  IndicatorsResponse,
  ApiKeyResponse,
  ModelType,
} from '@/types'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 120_000, // 2 min — model training can be slow on first run
})

apiClient.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const key = localStorage.getItem('stocksage_api_key')
    if (key) config.headers['X-API-Key'] = key
  }
  return config
})

export const stocksageApi = {
  getStock: (ticker: string, period = '3mo') =>
    apiClient
      .get<StockResponse>(`/v1/stocks/${ticker}`, { params: { period } })
      .then((r) => r.data),

  predict: (ticker: string, model: ModelType = 'ensemble', horizon = 5) =>
    apiClient
      .post<PredictionResponse>('/v1/predict/', { ticker, model, horizon })
      .then((r) => r.data),

  backtest: (params: {
    ticker: string
    model: ModelType
    start_date: string
    end_date: string
    initial_capital: number
    transaction_cost_bps: number
  }) =>
    apiClient.post<BacktestResponse>('/v1/backtest/', params).then((r) => r.data),

  getIndicators: (ticker: string) =>
    apiClient
      .get<IndicatorsResponse>(`/v1/stocks/${ticker}/indicators`)
      .then((r) => r.data),

  createApiKey: (name: string) =>
    apiClient.post<ApiKeyResponse>('/v1/auth/keys', { name }).then((r) => r.data),
}
