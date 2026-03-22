'use client'
import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import { BarChart3, LayoutDashboard, FlaskConical, BookOpen, Play, Loader2 } from 'lucide-react'
import { stocksageApi } from '@/lib/api'
import { CandlestickChart } from '@/components/charts/CandlestickChart'
import { FeatureImportance } from '@/components/charts/FeatureImportance'
import { TickerSearch } from '@/components/TickerSearch'
import { ModelSelector } from '@/components/ModelSelector'
import { PredictionCard } from '@/components/PredictionCard'
import { ApiKeyManager } from '@/components/ApiKeyManager'
import { formatCurrency, formatLargeNumber } from '@/lib/utils'
import type { ModelType, PredictionResponse } from '@/types'

export default function DashboardPage() {
  const [ticker, setTicker] = useState('AAPL')
  const [model, setModel] = useState<ModelType>('ensemble')
  const [horizon, setHorizon] = useState(5)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [indicators, setIndicators] = useState<Record<string, number>>({})

  const { data: stock, isLoading: stockLoading, error: stockError } = useQuery({
    queryKey: ['stock', ticker],
    queryFn: () => stocksageApi.getStock(ticker, '3mo'),
    enabled: !!ticker,
    retry: 1,
  })

  const predictMutation = useMutation({
    mutationFn: () => stocksageApi.predict(ticker, model, horizon),
    onSuccess: (data) => {
      setPrediction(data)
      // Fetch indicators after prediction
      stocksageApi.getIndicators(ticker).then((ind) => setIndicators(ind.indicators)).catch(() => {})
    },
  })

  const currentPrice = stock?.ohlcv[stock.ohlcv.length - 1]?.close

  return (
    <div className="min-h-screen bg-surface-900 text-gray-100">
      {/* Top bar */}
      <header className="sticky top-0 z-40 border-b border-gray-800 bg-surface-900/90 backdrop-blur-sm">
        <div className="max-w-[1400px] mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2 font-bold text-white">
              <BarChart3 size={18} className="text-indigo-400" />
              StockSage
            </Link>
            <nav className="hidden md:flex items-center gap-4 text-sm">
              <Link href="/dashboard" className="flex items-center gap-1.5 text-white font-medium">
                <LayoutDashboard size={14} /> Dashboard
              </Link>
              <Link href="/backtest" className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors">
                <FlaskConical size={14} /> Backtest
              </Link>
              <Link href="/docs" className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors">
                <BookOpen size={14} /> API Docs
              </Link>
            </nav>
          </div>
          <ApiKeyManager />
        </div>
      </header>

      <div className="max-w-[1400px] mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
        {/* Left: chart panel */}
        <div className="space-y-4">
          {/* Ticker search */}
          <div className="card">
            <TickerSearch
              value={ticker}
              onChange={(t) => { setTicker(t); setPrediction(null) }}
              loading={stockLoading}
            />
          </div>

          {/* Stock info bar */}
          {stock && (
            <div className="card flex flex-wrap gap-4 items-center text-sm py-3">
              <div>
                <span className="text-gray-500 text-xs">Company</span>
                <p className="font-medium text-white">{stock.name}</p>
              </div>
              <div>
                <span className="text-gray-500 text-xs">Sector</span>
                <p className="text-gray-300">{stock.sector}</p>
              </div>
              <div>
                <span className="text-gray-500 text-xs">Market Cap</span>
                <p className="text-gray-300 font-mono">{formatLargeNumber(stock.market_cap)}</p>
              </div>
              <div>
                <span className="text-gray-500 text-xs">P/E Ratio</span>
                <p className="text-gray-300 font-mono">
                  {stock.pe_ratio ? stock.pe_ratio.toFixed(2) : 'N/A'}
                </p>
              </div>
              <div>
                <span className="text-gray-500 text-xs">Last Close</span>
                <p className="text-white font-mono font-bold">{formatCurrency(currentPrice)}</p>
              </div>
            </div>
          )}

          {stockError && (
            <div className="card border-red-900 bg-red-950/30 text-red-400 text-sm">
              Failed to load {ticker}. Check the ticker symbol and your API key.
            </div>
          )}

          {/* Candlestick chart */}
          <div className="card min-h-[420px]">
            {stockLoading ? (
              <div className="h-96 flex items-center justify-center">
                <Loader2 className="animate-spin text-indigo-400" size={28} />
              </div>
            ) : stock ? (
              <CandlestickChart
                data={stock.ohlcv}
                predictions={prediction?.predictions}
                ticker={ticker}
              />
            ) : null}
          </div>

          {/* Feature importance */}
          {Object.keys(indicators).length > 0 && (
            <div className="card">
              <FeatureImportance importances={indicators} />
            </div>
          )}
        </div>

        {/* Right: controls + prediction */}
        <div className="space-y-4">
          <div className="card space-y-5">
            <ModelSelector value={model} onChange={setModel} />

            {/* Horizon slider */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Forecast Horizon
                </label>
                <span className="text-sm font-mono text-indigo-300">{horizon} days</span>
              </div>
              <input
                type="range"
                min={1}
                max={30}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>1d</span><span>15d</span><span>30d</span>
              </div>
            </div>

            {/* Run button */}
            <button
              onClick={() => predictMutation.mutate()}
              disabled={predictMutation.isPending || !stock}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-all disabled:opacity-60 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-indigo-900/50"
            >
              {predictMutation.isPending ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Training model...
                </>
              ) : (
                <>
                  <Play size={16} />
                  Run Prediction
                </>
              )}
            </button>

            {predictMutation.isPending && (
              <p className="text-xs text-gray-500 text-center">
                First request trains the model (~30s). Subsequent requests use cache.
              </p>
            )}

            {predictMutation.error && (
              <p className="text-xs text-red-400 text-center">
                Prediction failed. Ensure your API key is set.
              </p>
            )}
          </div>

          {/* Prediction results */}
          {prediction && (
            <div className="card">
              <PredictionCard prediction={prediction} currentPrice={currentPrice} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
