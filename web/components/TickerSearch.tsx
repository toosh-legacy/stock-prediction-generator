'use client'
import { useState, useCallback } from 'react'
import { Search, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

const POPULAR_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY']

interface Props {
  value: string
  onChange: (ticker: string) => void
  loading?: boolean
}

export function TickerSearch({ value, onChange, loading }: Props) {
  const [input, setInput] = useState(value)

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      const ticker = input.trim().toUpperCase()
      if (ticker) onChange(ticker)
    },
    [input, onChange]
  )

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="relative">
        <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
          {loading ? (
            <Loader2 size={16} className="text-indigo-400 animate-spin" />
          ) : (
            <Search size={16} className="text-gray-500" />
          )}
        </div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          placeholder="AAPL, MSFT, GOOGL..."
          className={cn(
            'w-full bg-surface-700 border border-gray-700 rounded-lg',
            'pl-9 pr-20 py-3 text-sm text-gray-100 placeholder-gray-500',
            'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent',
            'transition-all'
          )}
          maxLength={10}
          spellCheck={false}
          autoComplete="off"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className={cn(
            'absolute inset-y-0 right-0 px-4 rounded-r-lg text-sm font-medium',
            'bg-indigo-600 hover:bg-indigo-500 text-white transition-colors',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
        >
          Search
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        {POPULAR_TICKERS.map((t) => (
          <button
            key={t}
            onClick={() => {
              setInput(t)
              onChange(t)
            }}
            className={cn(
              'px-3 py-1 rounded-full text-xs font-mono font-medium transition-colors',
              value === t
                ? 'bg-indigo-600 text-white'
                : 'bg-surface-700 text-gray-400 hover:bg-surface-600 hover:text-gray-200 border border-gray-700'
            )}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  )
}
