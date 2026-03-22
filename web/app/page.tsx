import Link from 'next/link'
import {
  BarChart3,
  Zap,
  Shield,
  Globe,
  ArrowRight,
  ChevronRight,
  GitFork,
} from 'lucide-react'

const FEATURE_CARDS = [
  {
    icon: Zap,
    title: 'Real-time Data Pipeline',
    desc: 'Live OHLCV + 25+ technical indicators computed fresh for every request via yfinance.',
  },
  {
    icon: BarChart3,
    title: 'Ensemble Model Zoo',
    desc: 'LSTM with attention, XGBoost on engineered features, blended for best-of-both accuracy.',
  },
  {
    icon: Shield,
    title: 'Rigorous Backtesting',
    desc: 'Walk-forward evaluation with Sharpe ratio, max drawdown, win rate, vs buy-and-hold.',
  },
  {
    icon: Globe,
    title: 'Public REST API',
    desc: 'OpenAPI docs, API key auth, rate limiting. Build your own apps on top of StockSage.',
  },
]

const STEPS = [
  { n: '01', title: 'Get an API key', desc: 'Free, no credit card. Generated instantly.' },
  { n: '02', title: 'Choose a ticker', desc: 'Any stock on US exchanges supported via yfinance.' },
  { n: '03', title: 'Select a model', desc: 'LSTM, XGBoost, or the recommended ensemble.' },
  { n: '04', title: 'Get predictions', desc: 'Up to 30-day forecast with confidence intervals.' },
]

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-surface-900 text-gray-100">
      {/* Nav */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-gray-800/50 bg-surface-900/80 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 size={20} className="text-indigo-400" />
            <span className="font-bold text-white">StockSage</span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <Link href="/dashboard" className="text-gray-400 hover:text-white transition-colors">
              App
            </Link>
            <Link href="/backtest" className="text-gray-400 hover:text-white transition-colors">
              Backtest
            </Link>
            <Link href="/docs" className="text-gray-400 hover:text-white transition-colors">
              API Docs
            </Link>
            <Link
              href="/dashboard"
              className="px-4 py-1.5 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition-colors"
            >
              Try it free
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-4 text-center max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-indigo-800 bg-indigo-950/50 text-indigo-300 text-xs font-medium mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse-slow" />
          Free forever — no credit card required
        </div>

        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight mb-6 leading-tight">
          Predict the market
          <br />
          <span className="gradient-text animate-gradient-x bg-size-200">
            with AI confidence
          </span>
        </h1>

        <p className="text-lg text-gray-400 max-w-2xl mx-auto mb-10">
          StockSage combines LSTM neural networks and XGBoost to forecast stock prices up to 30
          days ahead — with confidence intervals, backtesting, and a free public REST API any
          developer can build on.
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-all hover:shadow-lg hover:shadow-indigo-900/50"
          >
            Try the App <ArrowRight size={16} />
          </Link>
          <Link
            href="/docs"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl border border-gray-700 hover:border-gray-600 text-gray-300 hover:text-white font-semibold transition-all"
          >
            View API Docs <ChevronRight size={16} />
          </Link>
        </div>
      </section>

      {/* Feature cards */}
      <section className="py-16 px-4">
        <div className="max-w-5xl mx-auto grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURE_CARDS.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="card hover:border-gray-700 transition-colors group">
              <div className="p-2 rounded-lg bg-indigo-950/50 border border-indigo-900/50 w-fit mb-4 group-hover:border-indigo-700 transition-colors">
                <Icon size={18} className="text-indigo-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">{title}</h3>
              <p className="text-sm text-gray-500">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="py-16 px-4 border-t border-gray-800">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-center text-white mb-12">How it works</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {STEPS.map(({ n, title, desc }) => (
              <div key={n} className="text-center">
                <div className="text-3xl font-black text-indigo-900 mb-3">{n}</div>
                <h3 className="font-semibold text-white mb-2">{title}</h3>
                <p className="text-sm text-gray-500">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Code snippet */}
      <section className="py-16 px-4 border-t border-gray-800">
        <div className="max-w-3xl mx-auto text-center mb-8">
          <h2 className="text-2xl font-bold text-white mb-3">Integrate in 60 seconds</h2>
          <p className="text-gray-400">One API call. Structured JSON. No complex setup.</p>
        </div>
        <div className="max-w-2xl mx-auto bg-surface-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-surface-700 border-b border-gray-700">
            <span className="text-xs text-gray-500 font-mono">curl</span>
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-gray-600" />
              <div className="w-2.5 h-2.5 rounded-full bg-gray-600" />
              <div className="w-2.5 h-2.5 rounded-full bg-gray-600" />
            </div>
          </div>
          <pre className="p-4 text-xs text-gray-300 font-mono overflow-x-auto leading-relaxed">
{`curl -X POST https://your-api.koyeb.app/v1/predict/ \\
  -H "X-API-Key: sk-your_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{"ticker":"AAPL","model":"ensemble","horizon":5}'

# Response:
{
  "ticker": "AAPL",
  "model": "ensemble",
  "horizon": 5,
  "predictions": [
    {"date":"2024-12-02","predicted":189.42,"lower_ci":183.12,"upper_ci":195.72},
    {"date":"2024-12-03","predicted":191.15,"lower_ci":184.21,"upper_ci":198.09},
    ...
  ],
  "confidence": 0.84
}`}
          </pre>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-8 px-4 text-center text-sm text-gray-600">
        <div className="flex items-center justify-center gap-6 mb-4">
          <a
            href="https://github.com/your-username/stocksage"
            className="flex items-center gap-2 hover:text-gray-400 transition-colors"
          >
            <GitFork size={14} /> GitHub
          </a>
          <Link href="/docs" className="hover:text-gray-400 transition-colors">
            API Docs
          </Link>
          <Link href="/backtest" className="hover:text-gray-400 transition-colors">
            Backtesting
          </Link>
        </div>
        <p>StockSage — for educational and research purposes only. Not financial advice.</p>
        <p className="mt-1">MIT License · Built with FastAPI + Next.js · Hosted on Koyeb + Vercel</p>
      </footer>
    </main>
  )
}
