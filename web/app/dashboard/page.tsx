'use client'
import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import {
  BarChart3, LayoutDashboard, FlaskConical, BookOpen,
  Play, Loader2, Radio,
} from 'lucide-react'
import { stocksageApi } from '@/lib/api'
import { LiveCandlestickChart } from '@/components/charts/LiveCandlestickChart'
import { CandlestickChart } from '@/components/charts/CandlestickChart'
import { FeatureImportance } from '@/components/charts/FeatureImportance'
import { TickerSearch } from '@/components/TickerSearch'
import { ModelSelector } from '@/components/ModelSelector'
import { PredictionCard } from '@/components/PredictionCard'
import { ApiKeyManager } from '@/components/ApiKeyManager'
import { formatCurrency, formatLargeNumber } from '@/lib/utils'
import { cn } from '@/lib/utils'
import type { ModelType, PredictionResponse } from '@/types'

type ViewMode = 'live' | 'analysis'

const INTERVAL_OPTIONS = [
  { value: '1m', label: '1m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '1d', label: '1D' },
]

export default function DashboardPage() {
  const [ticker, setTicker] = useState('AAPL')
  const [model, setModel] = useState<ModelType>('ensemble')
  const [horizon, setHorizon] = useState(5)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [indicators, setIndicators] = useState<Record<string, number>>({})
  const [viewMode, setViewMode] = useState<ViewMode>('live')
  const [interval, setInterval] = useState<'1m' | '5m' | '15m' | '1d'>('5m')

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
        {/* Left panel */}
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
                <span className="text-gray-500 text-xs">P/E</span>
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
              Failed to load {ticker}. Check the ticker and your API key.
            </div>
          )}

          {/* View mode toggle + interval selector */}
          <div className="flex items-center justify-between">
            <div className="flex gap-1 bg-surface-700 rounded-lg p-1 border border-gray-700">
              <button
                onClick={() => setViewMode('live')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors',
                  viewMode === 'live'
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-400 hover:text-white'
                )}
              >
                <Radio size={12} />
                Live Chart
              </button>
              <button
                onClick={() => setViewMode('analysis')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors',
                  viewMode === 'analysis'
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-400 hover:text-white'
                )}
              >
                <BarChart3 size={12} />
                Analysis
              </button>
            </div>
            {viewMode === 'live' && (
              <div className="flex gap-1">
                {INTERVAL_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setInterval(opt.value as any)}
                    className={cn(
                      'px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors',
                      interval === opt.value
                        ? 'bg-surface-600 text-white border border-gray-600'
                        : 'text-gray-500 hover:text-gray-300'
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Chart */}
          <div className="card min-h-[440px]">
            {viewMode === 'live' ? (
              <LiveCandlestickChart ticker={ticker} interval={interval} height={420} />
            ) : stockLoading ? (
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

        {/* Right panel */}
        <div className="space-y-4">
          <div className="card space-y-5">
            <ModelSelector value={model} onChange={setModel} />

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Forecast Horizon
                </label>
                <span className="text-sm font-mono text-indigo-300">{horizon} days</span>
              </div>
              <input
                type="range" min={1} max={30} value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>1d</span><span>15d</span><span>30d</span>
              </div>
            </div>

            <button
              onClick={() => predictMutation.mutate()}
              disabled={predictMutation.isPending || !stock}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-all disabled:opacity-60 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-indigo-900/50"
            >
              {predictMutation.isPending ? (
                <><Loader2 size={16} className="animate-spin" /> Training model...</>
              ) : (
                <><Play size={16} /> Run Prediction</>
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
