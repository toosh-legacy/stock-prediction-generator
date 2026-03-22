'use client'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface Props {
  importances: Record<string, number>
}

export function FeatureImportance({ importances }: Props) {
  const sorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, value]) => ({ name: name.replace(/_/g, ' '), value: Math.round(value * 1000) / 10 }))

  if (!sorted.length) return null

  return (
    <div>
      <h3 className="text-sm font-medium text-gray-400 mb-3">Top Feature Importances</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={sorted} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
          <XAxis
            type="number"
            tick={{ fill: '#6b7280', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={75}
          />
          <Tooltip
            formatter={(v: number) => [`${v}%`, 'Importance']}
            contentStyle={{
              background: '#1a1a24',
              border: '1px solid #374151',
              borderRadius: 8,
              color: '#e5e7eb',
              fontSize: 12,
            }}
          />
          <Bar dataKey="value" fill="#6366f1" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
