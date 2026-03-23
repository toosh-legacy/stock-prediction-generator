'use client'
import { useEffect, useRef } from 'react'
import { createChart, ColorType, LineStyle } from 'lightweight-charts'
import type { OHLCVPoint, PredictionPoint } from '@/types'

interface Props {
  data: OHLCVPoint[]
  predictions?: PredictionPoint[]
  ticker: string
}

export function CandlestickChart({ data, predictions, ticker }: Props) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data.length) return
    let disposed = false

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 380,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      crosshair: { mode: 1 },
      timeScale: { borderColor: '#374151', timeVisible: true },
      rightPriceScale: { borderColor: '#374151' },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    candleSeries.setData(
      data.map((d) => ({
        time: d.date as any,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }))
    )

    // Volume histogram
    const volumeSeries = chart.addHistogramSeries({
      color: '#6366f140',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    })
    volumeSeries.setData(
      data.map((d) => ({
        time: d.date as any,
        value: d.volume,
        color: d.close >= d.open ? '#22c55e30' : '#ef444430',
      }))
    )

    // Prediction overlay
    if (predictions?.length) {
      const predLine = chart.addLineSeries({
        color: '#818cf8',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        title: 'Forecast',
      })
      const upperLine = chart.addLineSeries({
        color: '#818cf820',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Upper CI',
      })
      const lowerLine = chart.addLineSeries({
        color: '#818cf820',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Lower CI',
      })

      predLine.setData(predictions.map((p) => ({ time: p.date as any, value: p.predicted })))
      upperLine.setData(predictions.map((p) => ({ time: p.date as any, value: p.upper_ci })))
      lowerLine.setData(predictions.map((p) => ({ time: p.date as any, value: p.lower_ci })))
    }

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (!disposed && chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)

    return () => {
      disposed = true
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data, predictions])

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-400">{ticker} — Price Chart</span>
        {predictions?.length ? (
          <span className="text-xs text-indigo-400 flex items-center gap-1">
            <span className="inline-block w-4 border-t-2 border-dashed border-indigo-400" />
            AI Forecast
          </span>
        ) : null}
      </div>
      <div ref={chartRef} className="w-full rounded-lg overflow-hidden" />
    </div>
  )
}
