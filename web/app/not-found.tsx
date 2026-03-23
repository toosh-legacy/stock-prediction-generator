import Link from 'next/link'

export default function NotFound() {
  return (
    <div className="min-h-screen bg-surface-900 flex flex-col items-center justify-center text-center px-4">
      <p className="text-7xl font-black text-gray-800 mb-4">404</p>
      <h1 className="text-2xl font-bold text-white mb-2">Page not found</h1>
      <p className="text-gray-500 mb-8">The page you&apos;re looking for doesn&apos;t exist.</p>
      <Link
        href="/"
        className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-colors"
      >
        Back to home
      </Link>
    </div>
  )
}
