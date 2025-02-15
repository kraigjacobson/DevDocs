export interface InternalLink {
  href: string
  text: string
  status?: 'pending' | 'crawled' | 'error'
}

export interface DiscoveredPage {
  url: string
  title?: string
  status: 'pending' | 'crawled' | 'error'
  internalLinks?: InternalLink[]  // Add internal links to each discovered page
}

export interface CrawlStats {
  subdomainsParsed: number
  pagesCrawled: number
  dataExtracted: string
  errorsEncountered: number
}

export interface CrawlResult {
  markdown: string
  stats: CrawlStats
}

export interface DiscoverResponse {
  message: string
  success: boolean
  pages: DiscoveredPage[]
  crawl_result: CrawlResult | null
}

export interface DiscoverOptions {
  url: string
  depth?: number
}