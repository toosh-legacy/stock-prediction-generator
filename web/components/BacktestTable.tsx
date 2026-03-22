'use client'
import type { BacktestResponse } from '@/types'
import { formatPercent, colorForReturn, formatCurrency } from '@/lib/utils'
import { cn } from '@/lib/utils'
import { AlertTriangle } from 'lucide-react'

interface Props {
  result: BacktestResponse
}

function MetricCard({
  label,
  value,
  sub,
  positive,
}: {
  label: string
  value: string
  sub?: string
  positive?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded-xl border p-4',
        positive === true
          ? 'bg-green-950/40 border-green-800'
          : positive === false
          ? 'bg-red-950/40 border-red-800'
          : 'bg-surface-700 border-gray-700'
      )}
    >
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-xl font-bold font-mono mt-1 text-white">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  )
}

export function BacktestTable({ result }: Props) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Total Return"
          value={formatPercent(result.total_return)}
          sub={`vs buy-and-hold ${formatPercent(result.benchmark_return)}`}
          positive={result.total_return > result.benchmark_return}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={result.sharpe_ratio.toFixed(3)}
          sub="Risk-adjusted return"
          positive={result.sharpe_ratio > 1}
        />
        <MetricCard
          label="Max Drawdown"
          value={formatPercent(result.max_drawdown)}
          sub="Peak-to-trough loss"
          positive={result.max_drawdown > -0.15}
        />
        <MetricCard
          label="Win Rate"
          value={`${(result.win_rate * 100).toFixed(1)}%`}
          sub={`${result.total_trades} trades`}
          positive={result.win_rate > 0.5}
        />
      </div>

      <div className="card">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Statistics</h3>
        <table className="w-full text-sm">
          <tbody className="divide-y divide-gray-800">
            {[
              ['Period', `${result.start_date} → ${result.end_date}`],
              ['Model', result.model],
              ['Ticker', result.ticker],
              ['Sortino Ratio', result.sortino_ratio.toFixed(3)],
              ['Total Trades', result.total_trades],
              ['Benchmark Return', formatPercent(result.benchmark_return)],
            ].map(([k, v]) => (
              <tr key={k}>
                <td className="py-2 text-gray-500">{k}</td>
                <td className="py-2 text-gray-200 font-mono text-right">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-2 p-3 rounded-lg bg-yellow-950/30 border border-yellow-800/40 text-xs text-yellow-700">
        <AlertTriangle size={12} className="flex-shrink-0 mt-0.5" />
        <span>{result.disclaimer}</span>
      </div>
    </div>
  )
}
