import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { QueryProvider } from '@/components/providers/QueryProvider'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'StockSage — AI Stock Prediction',
  description:
    'Predict stock prices using LSTM and XGBoost ensemble models. Free public API for developers.',
  openGraph: {
    title: 'StockSage — AI Stock Prediction',
    description: 'AI-powered stock prediction platform with a free public API.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  )
}
