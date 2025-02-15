import { DiscoveredPage, CrawlResult, DiscoverOptions, DiscoverResponse } from './types'

const BACKEND_URL = 'http://localhost:24125'

export async function discoverSubdomains({ url, depth = 3 }: DiscoverOptions): Promise<DiscoverResponse> {
  try {
    console.log('Making request to backend:', `${BACKEND_URL}/api/discover`)
    console.log('Request payload:', { url, depth })

    const response = await fetch(`${BACKEND_URL}/api/discover`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url, depth }),
    })

    console.log('Response status:', response.status)
    const data = await response.json()
    console.log('Response data:', data)

    if (!response.ok) {
      console.error('Error response:', data)
      throw new Error(data.detail || 'Failed to discover and crawl pages')
    }

    return data
  } catch (error) {
    console.error('Error discovering and crawling pages:', error)
    throw error
  }
}

export function validateUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 KB'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}