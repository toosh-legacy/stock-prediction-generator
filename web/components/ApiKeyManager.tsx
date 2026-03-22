'use client'
import { useState, useEffect } from 'react'
import { stocksageApi } from '@/lib/api'
import { KeyRound, CheckCircle } from 'lucide-react'

export function ApiKeyManager() {
  const [key, setKey] = useState('')
  const [saved, setSaved] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const stored = localStorage.getItem('stocksage_api_key')
    if (stored) {
      setKey(stored)
      setSaved(true)
    }
  }, [])

  const saveKey = () => {
    if (!key.trim()) return
    localStorage.setItem('stocksage_api_key', key.trim())
    setSaved(true)
    setError('')
  }

  const clearKey = () => {
    localStorage.removeItem('stocksage_api_key')
    setKey('')
    setSaved(false)
  }

  const generateKey = async () => {
    setGenerating(true)
    setError('')
    try {
      const result = await stocksageApi.createApiKey('StockSage Web')
      setKey(result.api_key)
      localStorage.setItem('stocksage_api_key', result.api_key)
      setSaved(true)
    } catch {
      setError('Failed to generate key. Is the API running?')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="flex items-center gap-3">
      {saved ? (
        <button
          onClick={clearKey}
          className="flex items-center gap-2 text-sm text-green-400 hover:text-green-300 transition-colors"
        >
          <CheckCircle size={14} />
          <span>API key active</span>
        </button>
      ) : (
        <button
          onClick={generateKey}
          disabled={generating}
          className="flex items-center gap-2 text-sm text-indigo-400 hover:text-indigo-300 transition-colors disabled:opacity-60"
        >
          <KeyRound size={14} />
          <span>{generating ? 'Generating...' : 'Get free API key'}</span>
        </button>
      )}
      {error && <span className="text-xs text-red-400">{error}</span>}
    </div>
  )
}
