'use client'
import type { PredictionResponse } from '@/types'
import { formatCurrency, formatPercent, colorForReturn } from '@/lib/utils'
import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Props {
  prediction: PredictionResponse
  currentPrice?: number
}

export function PredictionCard({ prediction, currentPrice }: Props) {
  const first = prediction.predictions[0]
  const last = prediction.predictions[prediction.predictions.length - 1]
  const change = currentPrice ? (last.predicted - currentPrice) / currentPrice : 0
  const isPositive = change >= 0

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className={cn('rounded-xl p-4 border', isPositive ? 'bg-green-950/40 border-green-800' : 'bg-red-950/40 border-red-800')}>
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-gray-400 mb-1">
              {prediction.horizon}-day target ({prediction.model})
            </p>
            <p className="text-2xl font-bold text-white font-mono">
              {formatCurrency(last.predicted)}
            </p>
            {currentPrice && (
              <p className={cn('text-sm font-medium mt-0.5', colorForReturn(change))}>
                {formatPercent(change)} from current
              </p>
            )}
          </div>
          <div className={cn('p-2 rounded-full', isPositive ? 'bg-green-900/50' : 'bg-red-900/50')}>
            {isPositive ? (
              <TrendingUp size={20} className="text-green-400" />
            ) : (
              <TrendingDown size={20} className="text-red-400" />
            )}
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-gray-500">Confidence</span>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 rounded-full"
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
              <span className="text-gray-300 font-mono">
                {Math.round(prediction.confidence * 100)}%
              </span>
            </div>
          </div>
          <div>
            <span className="text-gray-500">CI Range (last)</span>
            <p className="text-gray-300 font-mono mt-1">
              {formatCurrency(last.lower_ci)} – {formatCurrency(last.upper_ci)}
            </p>
          </div>
        </div>
      </div>

      {/* Day-by-day table */}
      <div>
        <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
          Daily Forecast
        </h4>
        <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
          {prediction.predictions.map((p, i) => {
            const dayChange = currentPrice ? (p.predicted - currentPrice) / currentPrice : 0
            return (
              <div
                key={p.date}
                className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-surface-700 text-sm"
              >
                <span className="text-gray-500 w-20">
                  Day {i + 1} <span className="text-[10px]">({p.date})</span>
                </span>
                <span className="font-mono text-gray-200">{formatCurrency(p.predicted)}</span>
                <span className={cn('text-xs font-medium', colorForReturn(dayChange))}>
                  {formatPercent(dayChange)}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex gap-2 p-3 rounded-lg bg-yellow-950/30 border border-yellow-800/40 text-xs text-yellow-700">
        <AlertTriangle size={12} className="flex-shrink-0 mt-0.5" />
        <span>{prediction.disclaimer}</span>
      </div>
    </div>
  )
}
