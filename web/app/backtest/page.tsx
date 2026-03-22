'use client'
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import {
  BarChart3,
  LayoutDashboard,
  FlaskConical,
  BookOpen,
  Play,
  Loader2,
} from 'lucide-react'
import { stocksageApi } from '@/lib/api'
import { EquityCurve } from '@/components/charts/EquityCurve'
import { BacktestTable } from '@/components/BacktestTable'
import { ApiKeyManager } from '@/components/ApiKeyManager'
import type { ModelType, BacktestResponse } from '@/types'
import { cn } from '@/lib/utils'

export default function BacktestPage() {
  const [form, setForm] = useState({
    ticker: 'AAPL',
    model: 'ensemble' as ModelType,
    start_date: '2022-01-01',
    end_date: '2023-12-31',
    initial_capital: 10000,
    transaction_cost_bps: 10,
  })
  const [result, setResult] = useState<BacktestResponse | null>(null)

  const mutation = useMutation({
    mutationFn: () => stocksageApi.backtest(form),
    onSuccess: setResult,
  })

  const set = (k: string, v: any) => setForm((f) => ({ ...f, [k]: v }))

  return (
    <div className="min-h-screen bg-surface-900 text-gray-100">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-gray-800 bg-surface-900/90 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2 font-bold text-white">
              <BarChart3 size={18} className="text-indigo-400" />
              StockSage
            </Link>
            <nav className="hidden md:flex items-center gap-4 text-sm">
              <Link href="/dashboard" className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors">
                <LayoutDashboard size={14} /> Dashboard
              </Link>
              <Link href="/backtest" className="flex items-center gap-1.5 text-white font-medium">
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

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Backtesting Engine</h1>
          <p className="text-gray-400 text-sm mt-1">
            Simulate a long/short strategy on historical data and evaluate risk-adjusted returns.
          </p>
        </div>

        {/* Form */}
        <div className="card">
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-xs text-gray-400 font-medium block mb-1">Ticker</label>
              <input
                value={form.ticker}
                onChange={(e) => set('ticker', e.target.value.toUpperCase())}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="AAPL"
                maxLength={10}
              />
            </div>

            <div>
              <label className="text-xs text-gray-400 font-medium block mb-1">Model</label>
              <select
                value={form.model}
                onChange={(e) => set('model', e.target.value as ModelType)}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="ensemble">Ensemble (recommended)</option>
                <option value="lstm">LSTM</option>
                <option value="xgboost">XGBoost</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-400 font-medium block mb-1">Start Date</label>
              <input
                type="date"
                value={form.start_date}
                onChange={(e) => set('start_date', e.target.value)}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs text-gray-400 font-medium block mb-1">End Date</label>
              <input
                type="date"
                value={form.end_date}
                onChange={(e) => set('end_date', e.target.value)}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs text-gray-400 font-medium block mb-1">
                Initial Capital (USD)
              </label>
              <input
                type="number"
                value={form.initial_capital}
                onChange={(e) => set('initial_capital', Number(e.target.value))}
                min={1000}
                step={1000}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs text-gray-400 font-medium block mb-1">
                Transaction Cost (bps)
              </label>
              <input
                type="number"
                value={form.transaction_cost_bps}
                onChange={(e) => set('transaction_cost_bps', Number(e.target.value))}
                min={0}
                max={100}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>

          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="mt-4 flex items-center gap-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-all disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {mutation.isPending ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Running backtest...
              </>
            ) : (
              <>
                <Play size={16} /> Run Backtest
              </>
            )}
          </button>

          {mutation.isPending && (
            <p className="text-xs text-gray-500 mt-2">
              Walk-forward backtesting can take 1–2 minutes. Please wait.
            </p>
          )}
          {mutation.error && (
            <p className="text-xs text-red-400 mt-2">
              Backtest failed. Ensure your API key is set and the date range has sufficient data.
            </p>
          )}
        </div>

        {/* Results */}
        {result && (
          <div className="grid lg:grid-cols-[1fr_340px] gap-6 animate-fade-in">
            <div className="space-y-4">
              <div className="card">
                <h2 className="text-sm font-medium text-gray-400 mb-4">Equity Curve</h2>
                <EquityCurve data={result.equity_curve} />
              </div>
            </div>
            <div>
              <BacktestTable result={result} />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
