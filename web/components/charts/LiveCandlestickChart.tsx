'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import {
  createChart,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
  IChartApi,
  ISeriesApi,
} from 'lightweight-charts'
import { Wifi, WifiOff, Loader2 } from 'lucide-react'

interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
  predicted?: boolean
}

interface SentimentData {
  compound?: number
  positive?: number
  negative?: number
  count?: number
}

interface MacroSnapshot {
  vix?: number
  fed_funds_rate?: number
  yield_spread_10y2y?: number
  [key: string]: number | undefined
}

interface WsMessage {
  type: string
  ticker: string
  interval: string
  candles: Candle[]
  predicted: Candle[]
  sentiment: SentimentData
  macro_snapshot: MacroSnapshot
  timestamp: string
}

interface Props {
  ticker: string
  interval?: '1m' | '5m' | '15m' | '1d'
  apiBaseUrl?: string
  height?: number
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

const WS_BASE = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
  .replace(/^https/, 'wss')
  .replace(/^http/, 'ws')

export function LiveCandlestickChart({ ticker, interval = '5m', height = 420 }: Props) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const realSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const predSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectDelay = useRef(1000)

  const [status, setStatus] = useState<ConnectionStatus>('connecting')
  const [lastUpdate, setLastUpdate] = useState<string | null>(null)
  const [sentiment, setSentiment] = useState<SentimentData>({})
  const [macro, setMacro] = useState<MacroSnapshot>({})
  const [predCount, setPredCount] = useState(0)

  // ------------------------------------------------------------------
  // Chart initialisation
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f293760' },
        horzLines: { color: '#1f293760' },
      },
      crosshair: { mode: 1 },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: interval === '1m',
      },
      rightPriceScale: { borderColor: '#374151' },
    })

    chartRef.current = chart

    // Real candles series
    const realSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })
    realSeriesRef.current = realSeries

    // Predicted candles — indigo, hollow look
    const predSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#818cf840',
      downColor: '#a78bfa40',
      borderUpColor: '#818cf8',
      borderDownColor: '#a78bfa',
      wickUpColor: '#818cf8',
      wickDownColor: '#a78bfa',
    })
    predSeriesRef.current = predSeries

    // Volume
    const volSeries = chart.addSeries(HistogramSeries, {
      color: '#6366f140',
      priceFormat: { type: 'volume' },
      priceScaleId: 'vol',
    })
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } })
    volSeriesRef.current = volSeries

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartRef.current = null
    }
  }, [height, interval])

  // ------------------------------------------------------------------
  // WebSocket connection
  // ------------------------------------------------------------------
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const url = `${WS_BASE}/v1/realtime/ws/${ticker}?interval=${interval}`
    setStatus('connecting')

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('connected')
      reconnectDelay.current = 1000
    }

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)
        if (msg.type !== 'update') return

        // Update real candle series
        if (realSeriesRef.current && msg.candles?.length) {
          const data = msg.candles.map((c) => ({
            time: c.time as any,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          }))
          realSeriesRef.current.setData(data)
        }

        // Update predicted candle series
        if (predSeriesRef.current && msg.predicted?.length) {
          const data = msg.predicted.map((c) => ({
            time: c.time as any,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          }))
          predSeriesRef.current.setData(data)
          setPredCount(data.length)
        }

        // Volume
        if (volSeriesRef.current && msg.candles?.length) {
          const data = msg.candles
            .filter((c) => c.volume != null)
            .map((c) => ({
              time: c.time as any,
              value: c.volume!,
              color: c.close >= c.open ? '#22c55e30' : '#ef444430',
            }))
          if (data.length) volSeriesRef.current.setData(data)
        }

        chartRef.current?.timeScale().fitContent()

        setSentiment(msg.sentiment || {})
        setMacro(msg.macro_snapshot || {})
        setLastUpdate(new Date(msg.timestamp).toLocaleTimeString())
      } catch {
        // malformed message, ignore
      }
    }

    ws.onerror = () => setStatus('error')

    ws.onclose = () => {
      setStatus('disconnected')
      // Exponential backoff reconnect
      const delay = Math.min(reconnectDelay.current, 30000)
      reconnectDelay.current = delay * 2
      reconnectTimer.current = setTimeout(connect, delay)
    }
  }, [ticker, interval])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  // ------------------------------------------------------------------
  // Sentiment helpers
  // ------------------------------------------------------------------
  const sentimentLabel = () => {
    const c = sentiment.compound ?? 0
    if (c > 0.15) return { text: 'Bullish', color: 'text-green-400' }
    if (c < -0.15) return { text: 'Bearish', color: 'text-red-400' }
    return { text: 'Neutral', color: 'text-gray-400' }
  }
  const sent = sentimentLabel()

  return (
    <div className="space-y-2">
      {/* Status bar */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <span className="font-mono font-bold text-white">{ticker}</span>
          <span className="text-gray-500">·</span>
          <span className="text-gray-500">{interval} candles</span>
          {predCount > 0 && (
            <>
              <span className="text-gray-500">·</span>
              <span className="flex items-center gap-1 text-indigo-400">
                <span className="inline-block w-3 border-t border-dashed border-indigo-400" />
                {predCount} predicted candles
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          {Object.keys(macro).length > 0 && (
            <span className="text-gray-500">
              VIX {macro.vix?.toFixed(1) ?? '—'}
            </span>
          )}
          {sentiment.compound !== undefined && (
            <span className={sent.color}>{sent.text}</span>
          )}
          <div className="flex items-center gap-1">
            {status === 'connected' ? (
              <Wifi size={12} className="text-green-400" />
            ) : status === 'connecting' ? (
              <Loader2 size={12} className="text-yellow-400 animate-spin" />
            ) : (
              <WifiOff size={12} className="text-red-400" />
            )}
            <span
              className={
                status === 'connected'
                  ? 'text-green-400'
                  : status === 'connecting'
                  ? 'text-yellow-400'
                  : 'text-red-400'
              }
            >
              {status === 'connected' && lastUpdate ? `Live · ${lastUpdate}` : status}
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div ref={chartContainerRef} className="w-full rounded-lg overflow-hidden" />

      {/* Legend */}
      <div className="flex gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-green-400 inline-block" /> Real candles
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-indigo-400 inline-block border-dashed" /> AI prediction
        </span>
        {sentiment.count != null && sentiment.count > 0 && (
          <span className="ml-auto">
            Based on {sentiment.count} articles · compound {(sentiment.compound ?? 0).toFixed(3)}
          </span>
        )}
      </div>
    </div>
  )
}
