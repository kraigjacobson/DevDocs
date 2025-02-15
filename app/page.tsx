'use client'

import { useState, useEffect } from 'react'
import UrlInput from '@/components/UrlInput'
import ProcessingBlock from '@/components/ProcessingBlock'
import MarkdownOutput from '@/components/MarkdownOutput'
import StoredFiles from '@/components/StoredFiles'
import MCPServerStatus from '@/components/MCPServerStatus'
import { discoverSubdomains, validateUrl, formatBytes } from '@/lib/crawl-service'
import { saveMarkdown, loadMarkdown } from '@/lib/storage'
import { useToast } from "@/components/ui/use-toast"
import { DiscoveredPage } from '@/lib/types'

export default function Home() {
  const [url, setUrl] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [markdown, setMarkdown] = useState('')
  const [stats, setStats] = useState({
    subdomainsParsed: 0,
    pagesCrawled: 0,
    dataExtracted: '0 KB',
    errorsEncountered: 0
  })
  const { toast } = useToast()

  const handleSubmit = async (submittedUrl: string, depth: number) => {
    if (!validateUrl(submittedUrl)) {
      toast({
        title: "Invalid URL",
        description: "Please enter a valid URL including the protocol (http:// or https://)",
        variant: "destructive"
      })
      return
    }

    setUrl(submittedUrl)
    setIsProcessing(true)
    setMarkdown('')
    
    try {
      console.log('Starting discovery and crawl for:', submittedUrl, 'with depth:', depth)
      const result = await discoverSubdomains({ url: submittedUrl, depth })
      console.log('Discovery and crawl result:', result)
      
      if (result.crawl_result) {
        setMarkdown(result.crawl_result.markdown)
        setStats({
          subdomainsParsed: result.pages.length,
          pagesCrawled: result.pages.length,
          dataExtracted: formatBytes(result.crawl_result.markdown.length),
          errorsEncountered: 0
        })
      }
      
      toast({
        title: "Processing Complete",
        description: `Found and processed ${result.pages?.length || 0} pages at depth ${depth}`
      })
    } catch (error) {
      console.error('Error processing URL:', error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process URL",
        variant: "destructive"
      })
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900">
      <header className="w-full py-12 bg-gradient-to-r from-gray-900/50 to-gray-800/50 backdrop-blur-sm border-b border-gray-700">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500 mb-4">
            DevDocs
          </h1>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Discover and extract documentation for quicker development
          </p>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 space-y-6">
        <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
          <h2 className="text-2xl font-semibold mb-4 text-purple-400">Processing Status</h2>
          <div className="flex gap-4">
            <ProcessingBlock
              isProcessing={isProcessing}
              stats={stats}
            />
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
          <h2 className="text-2xl font-semibold mb-4 text-blue-400">Start Exploration</h2>
          <UrlInput onSubmit={handleSubmit} />
        </div>

        <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
          <h2 className="text-2xl font-semibold mb-4 text-yellow-400">Extracted Content</h2>
          <MarkdownOutput
            markdown={markdown}
            isVisible={markdown !== ''}
          />
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
            <h2 className="text-2xl font-semibold mb-4 text-white">MCP Server</h2>
            <MCPServerStatus />
          </div>

          <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
            <h2 className="text-2xl font-semibold mb-4 text-blue-400">Stored Files</h2>
            <StoredFiles />
          </div>
        </div>
      </div>

      <footer className="py-8 text-center text-gray-400">
        <p className="flex items-center justify-center gap-2">
          Made with <span className="text-red-500">❤️</span> by{' '}
          <a 
            href="https://www.cyberagi.ai/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 transition-colors"
          >
            CyberAGI Inc
          </a>{' '}
          in <span>🇺🇸</span>
        </p>
      </footer>
    </main>
  )
}
