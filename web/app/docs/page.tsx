'use client'
import { useState } from 'react'
import Link from 'next/link'
import { useMutation } from '@tanstack/react-query'
import {
  BarChart3,
  LayoutDashboard,
  FlaskConical,
  BookOpen,
  Copy,
  Check,
  Play,
} from 'lucide-react'
import { stocksageApi } from '@/lib/api'
import { ApiKeyManager } from '@/components/ApiKeyManager'
import { cn } from '@/lib/utils'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const ENDPOINTS = [
  { method: 'POST', path: '/v1/auth/keys', auth: false, desc: 'Generate a free API key' },
  { method: 'DELETE', path: '/v1/auth/keys/{key_id}', auth: false, desc: 'Revoke an API key' },
  { method: 'GET', path: '/v1/stocks/{ticker}', auth: true, desc: 'OHLCV + company info' },
  { method: 'GET', path: '/v1/stocks/{ticker}/indicators', auth: true, desc: 'Technical indicators' },
  { method: 'POST', path: '/v1/predict/', auth: true, desc: 'AI price forecast' },
  { method: 'POST', path: '/v1/backtest/', auth: true, desc: 'Walk-forward backtest' },
]

const CODE_SAMPLES: Record<string, Record<string, string>> = {
  curl: {
    'Get API Key': `curl -X POST ${BASE_URL}/v1/auth/keys \\
  -H "Content-Type: application/json" \\
  -d '{"name":"My App"}'`,
    'Predict': `curl -X POST ${BASE_URL}/v1/predict/ \\
  -H "X-API-Key: sk-your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"ticker":"AAPL","model":"ensemble","horizon":5}'`,
    'Backtest': `curl -X POST ${BASE_URL}/v1/backtest/ \\
  -H "X-API-Key: sk-your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"ticker":"AAPL","model":"ensemble","start_date":"2022-01-01","end_date":"2023-12-31","initial_capital":10000,"transaction_cost_bps":10}'`,
  },
  python: {
    'Get API Key': `import requests

r = requests.post("${BASE_URL}/v1/auth/keys", json={"name": "My App"})
api_key = r.json()["api_key"]
print(f"Your API key: {api_key}")`,
    'Predict': `import requests

headers = {"X-API-Key": "sk-your_key"}
r = requests.post(
    "${BASE_URL}/v1/predict/",
    json={"ticker": "AAPL", "model": "ensemble", "horizon": 5},
    headers=headers
)
result = r.json()
for p in result["predictions"]:
    print(f"{p['date']}: ${p['predicted']:.2f} [{p['lower_ci']:.2f}, {p['upper_ci']:.2f}]")`,
    'Backtest': `import requests

headers = {"X-API-Key": "sk-your_key"}
r = requests.post(
    "${BASE_URL}/v1/backtest/",
    json={
        "ticker": "AAPL", "model": "ensemble",
        "start_date": "2022-01-01", "end_date": "2023-12-31",
        "initial_capital": 10000, "transaction_cost_bps": 10
    },
    headers=headers
)
bt = r.json()
print(f"Total return: {bt['total_return']*100:.2f}%")
print(f"Sharpe ratio: {bt['sharpe_ratio']:.3f}")`,
  },
  javascript: {
    'Get API Key': `const res = await fetch("${BASE_URL}/v1/auth/keys", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "My App" })
});
const { api_key } = await res.json();
console.log("Your API key:", api_key);`,
    'Predict': `const res = await fetch("${BASE_URL}/v1/predict/", {
  method: "POST",
  headers: {
    "X-API-Key": "sk-your_key",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ ticker: "AAPL", model: "ensemble", horizon: 5 })
});
const data = await res.json();
console.log(data.predictions);`,
    'Backtest': `const res = await fetch("${BASE_URL}/v1/backtest/", {
  method: "POST",
  headers: {
    "X-API-Key": "sk-your_key",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    ticker: "AAPL", model: "ensemble",
    start_date: "2022-01-01", end_date: "2023-12-31",
    initial_capital: 10000, transaction_cost_bps: 10
  })
});
const bt = await res.json();
console.log(\`Sharpe: \${bt.sharpe_ratio}\`);`,
  },
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="relative bg-surface-700 rounded-lg border border-gray-700 overflow-hidden">
      <button
        onClick={copy}
        className="absolute top-2 right-2 p-1.5 rounded hover:bg-surface-600 transition-colors text-gray-500 hover:text-gray-300"
      >
        {copied ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
      </button>
      <pre className="p-4 text-xs text-gray-300 font-mono overflow-x-auto leading-relaxed whitespace-pre">
        {code}
      </pre>
    </div>
  )
}

export default function DocsPage() {
  const [lang, setLang] = useState<'curl' | 'python' | 'javascript'>('curl')
  const [keyName, setKeyName] = useState('My App')
  const [tryTicker, setTryTicker] = useState('AAPL')
  const [tryModel, setTryModel] = useState('ensemble')
  const [tryHorizon, setTryHorizon] = useState(5)
  const [tryResult, setTryResult] = useState<string | null>(null)

  const tryMutation = useMutation({
    mutationFn: () => stocksageApi.predict(tryTicker, tryModel as any, tryHorizon),
    onSuccess: (data) => setTryResult(JSON.stringify(data, null, 2)),
    onError: () => setTryResult('{"error": "Request failed. Ensure your API key is set above."}'),
  })

  const keyMutation = useMutation({
    mutationFn: () => stocksageApi.createApiKey(keyName),
    onSuccess: (data) => {
      if (typeof window !== 'undefined') {
        localStorage.setItem('stocksage_api_key', data.api_key)
      }
      setTryResult(JSON.stringify(data, null, 2))
    },
  })

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
              <Link href="/backtest" className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors">
                <FlaskConical size={14} /> Backtest
              </Link>
              <Link href="/docs" className="flex items-center gap-1.5 text-white font-medium">
                <BookOpen size={14} /> API Docs
              </Link>
            </nav>
          </div>
          <ApiKeyManager />
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-10">
        <div>
          <h1 className="text-3xl font-bold text-white">API Reference</h1>
          <p className="text-gray-400 mt-2">
            Base URL:{' '}
            <code className="text-indigo-300 bg-surface-700 px-2 py-0.5 rounded text-sm">
              {BASE_URL}
            </code>
          </p>
        </div>

        {/* Authentication */}
        <section className="card space-y-4">
          <h2 className="text-lg font-semibold text-white">Authentication</h2>
          <p className="text-sm text-gray-400">
            All endpoints except <code className="text-indigo-300">POST /v1/auth/keys</code>{' '}
            require an API key in the request header:
          </p>
          <CodeBlock code={`X-API-Key: sk-your_api_key_here`} />

          <div className="pt-2 space-y-3">
            <h3 className="text-sm font-medium text-gray-300">Generate your free API key</h3>
            <div className="flex gap-2">
              <input
                value={keyName}
                onChange={(e) => setKeyName(e.target.value)}
                placeholder="App name"
                className="flex-1 bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <button
                onClick={() => keyMutation.mutate()}
                disabled={keyMutation.isPending}
                className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors disabled:opacity-60"
              >
                {keyMutation.isPending ? 'Generating...' : 'Generate Key'}
              </button>
            </div>
            {keyMutation.data && (
              <div className="p-3 bg-green-950/30 border border-green-800/50 rounded-lg text-xs text-green-400 font-mono break-all">
                {keyMutation.data.api_key}
                <p className="text-green-600 mt-1">Key saved to browser. Not shown again.</p>
              </div>
            )}
          </div>
        </section>

        {/* Endpoints table */}
        <section className="card">
          <h2 className="text-lg font-semibold text-white mb-4">Endpoints</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-2 pr-4 text-gray-500 font-medium">Method</th>
                  <th className="text-left py-2 pr-4 text-gray-500 font-medium">Path</th>
                  <th className="text-left py-2 pr-4 text-gray-500 font-medium">Auth</th>
                  <th className="text-left py-2 text-gray-500 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {ENDPOINTS.map((ep) => (
                  <tr key={ep.path + ep.method}>
                    <td className="py-2.5 pr-4">
                      <span className={cn(
                        'px-2 py-0.5 rounded text-xs font-mono font-bold',
                        ep.method === 'GET' ? 'bg-blue-900/50 text-blue-300' :
                        ep.method === 'POST' ? 'bg-green-900/50 text-green-300' :
                        'bg-red-900/50 text-red-300'
                      )}>
                        {ep.method}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-indigo-300 text-xs">{ep.path}</td>
                    <td className="py-2.5 pr-4 text-xs">
                      {ep.auth ? (
                        <span className="text-yellow-500">Required</span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>
                    <td className="py-2.5 text-gray-400">{ep.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Code samples */}
        <section className="card space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">SDK Examples</h2>
            <div className="flex gap-1">
              {(['curl', 'python', 'javascript'] as const).map((l) => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={cn(
                    'px-3 py-1 rounded text-xs font-medium transition-colors',
                    lang === l
                      ? 'bg-indigo-600 text-white'
                      : 'text-gray-400 hover:text-white'
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
          {Object.entries(CODE_SAMPLES[lang]).map(([title, code]) => (
            <div key={title}>
              <h3 className="text-sm font-medium text-gray-400 mb-2">{title}</h3>
              <CodeBlock code={code} />
            </div>
          ))}
        </section>

        {/* Rate limits */}
        <section className="card">
          <h2 className="text-lg font-semibold text-white mb-4">Rate Limits</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-2 pr-4 text-gray-500 font-medium">Tier</th>
                <th className="text-left py-2 pr-4 text-gray-500 font-medium">Requests / hour</th>
                <th className="text-left py-2 text-gray-500 font-medium">Cost</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              <tr>
                <td className="py-2.5 pr-4 text-gray-300">Free</td>
                <td className="py-2.5 pr-4 text-gray-300 font-mono">100</td>
                <td className="py-2.5 text-green-400">$0 / month</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 text-gray-300">Pro</td>
                <td className="py-2.5 pr-4 text-gray-300 font-mono">10,000</td>
                <td className="py-2.5 text-gray-400">Contact us</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* Try it */}
        <section className="card space-y-4">
          <h2 className="text-lg font-semibold text-white">Try it live</h2>
          <div className="grid sm:grid-cols-3 gap-3">
            <div>
              <label className="text-xs text-gray-500 block mb-1">Ticker</label>
              <input
                value={tryTicker}
                onChange={(e) => setTryTicker(e.target.value.toUpperCase())}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">Model</label>
              <select
                value={tryModel}
                onChange={(e) => setTryModel(e.target.value)}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="ensemble">ensemble</option>
                <option value="lstm">lstm</option>
                <option value="xgboost">xgboost</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">Horizon (days)</label>
              <input
                type="number"
                value={tryHorizon}
                onChange={(e) => setTryHorizon(Number(e.target.value))}
                min={1}
                max={30}
                className="w-full bg-surface-700 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>
          <button
            onClick={() => tryMutation.mutate()}
            disabled={tryMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors disabled:opacity-60"
          >
            {tryMutation.isPending ? (
              <>Loading...</>
            ) : (
              <>
                <Play size={14} /> Send Request
              </>
            )}
          </button>
          {tryResult && (
            <div className="bg-surface-700 rounded-lg border border-gray-700 overflow-hidden">
              <pre className="p-4 text-xs text-gray-300 font-mono overflow-x-auto max-h-80 leading-relaxed">
                {tryResult}
              </pre>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
