'use client'
import { cn } from '@/lib/utils'
import type { ModelType } from '@/types'

const MODELS: { id: ModelType; name: string; desc: string; badge?: string }[] = [
  {
    id: 'ensemble',
    name: 'Ensemble',
    desc: 'LSTM + XGBoost blend. Best accuracy.',
    badge: 'Recommended',
  },
  {
    id: 'lstm',
    name: 'LSTM',
    desc: 'Attention-based recurrent network for sequences.',
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    desc: 'Gradient-boosted trees on technical indicators.',
  },
]

interface Props {
  value: ModelType
  onChange: (model: ModelType) => void
}

export function ModelSelector({ value, onChange }: Props) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
        Model
      </label>
      <div className="space-y-2">
        {MODELS.map((m) => (
          <button
            key={m.id}
            onClick={() => onChange(m.id)}
            className={cn(
              'w-full text-left rounded-lg border p-3 transition-all',
              value === m.id
                ? 'border-indigo-500 bg-indigo-950/50'
                : 'border-gray-700 bg-surface-700 hover:border-gray-600'
            )}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    'w-3.5 h-3.5 rounded-full border-2 flex-shrink-0',
                    value === m.id ? 'border-indigo-400 bg-indigo-400' : 'border-gray-600'
                  )}
                />
                <span className="text-sm font-medium text-gray-200">{m.name}</span>
              </div>
              {m.badge && (
                <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-indigo-900 text-indigo-300 border border-indigo-700">
                  {m.badge}
                </span>
              )}
            </div>
            <p className="mt-1 text-xs text-gray-500 pl-5">{m.desc}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
