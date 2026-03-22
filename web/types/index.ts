export interface OHLCVPoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface PredictionPoint {
  date: string
  predicted: number
  lower_ci: number
  upper_ci: number
}

export interface PredictionResponse {
  ticker: string
  model: string
  horizon: number
  predictions: PredictionPoint[]
  confidence: number
  disclaimer: string
}

export interface BacktestResponse {
  ticker: string
  model: string
  start_date: string
  end_date: string
  total_return: number
  benchmark_return: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  equity_curve: EquityPoint[]
  disclaimer: string
}

export interface EquityPoint {
  date: string
  portfolio: number
  benchmark: number
}

export interface StockResponse {
  ticker: string
  name: string
  sector: string
  market_cap: number | null
  pe_ratio: number | null
  ohlcv: OHLCVPoint[]
}

export interface IndicatorsResponse {
  ticker: string
  date: string
  indicators: Record<string, number>
}

export interface ApiKeyResponse {
  api_key: string
  key_id: string
  message: string
}

export type ModelType = 'lstm' | 'xgboost' | 'ensemble'
