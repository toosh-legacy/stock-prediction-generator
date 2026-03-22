/**
 * Next.js route handler — proxies all requests to the FastAPI backend.
 * This allows the frontend to call /api/v1/... and have it forwarded
 * to the backend without CORS issues in production.
 *
 * The `rewrites` in next.config.ts handle this automatically for most cases.
 * This file serves as a fallback for edge cases.
 */
import { NextRequest, NextResponse } from 'next/server'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET(req: NextRequest) {
  return proxy(req)
}

export async function POST(req: NextRequest) {
  return proxy(req)
}

export async function DELETE(req: NextRequest) {
  return proxy(req)
}

async function proxy(req: NextRequest): Promise<NextResponse> {
  const url = new URL(req.url)
  // Strip /api prefix and forward to backend
  const backendPath = url.pathname.replace(/^\/api/, '')
  const backendUrl = `${API_URL}${backendPath}${url.search}`

  const headers = new Headers(req.headers)
  headers.delete('host')

  try {
    const response = await fetch(backendUrl, {
      method: req.method,
      headers,
      body: req.method !== 'GET' && req.method !== 'HEAD' ? req.body : undefined,
      // @ts-ignore — Next.js specific
      duplex: 'half',
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to reach backend API' },
      { status: 502 }
    )
  }
}
