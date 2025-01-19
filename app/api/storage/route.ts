import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

const STORAGE_DIR = path.join(process.cwd(), 'storage/markdown')

export async function POST(request: Request) {
  try {
    const { url, content } = await request.json()
    
    // Create storage directory if it doesn't exist
    await fs.mkdir(STORAGE_DIR, { recursive: true })
    
    // Generate filename from URL
    const filename = url
      .replace(/^https?:\/\//, '')
      .replace(/[^a-z0-9]/gi, '_')
      .toLowerCase() + '.md'
    
    const filePath = path.join(STORAGE_DIR, filename)
    await fs.writeFile(filePath, content, 'utf-8')
    
    return NextResponse.json({ success: true })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to save markdown' },
      { status: 500 }
    )
  }
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const url = searchParams.get('url')
    
    // Handle list request
    if (!url) {
      // Only get .md files
      const files = await fs.readdir(STORAGE_DIR)
      const mdFiles = files.filter(f => f.endsWith('.md'))
      
      const fileDetails = await Promise.all(
        mdFiles.map(async (filename) => {
          const mdPath = path.join(STORAGE_DIR, filename)
          const jsonPath = path.join(STORAGE_DIR, filename.replace('.md', '.json'))
          const stats = await fs.stat(mdPath)
          const content = await fs.readFile(mdPath, 'utf-8')
          
          // Create JSON file if it doesn't exist
          if (!files.includes(filename.replace('.md', '.json'))) {
            const jsonContent = JSON.stringify({
              content,
              metadata: {
                wordCount: content.split(/\s+/).length,
                charCount: content.length,
                timestamp: stats.mtime
              }
            }, null, 2)
            await fs.writeFile(jsonPath, jsonContent, 'utf-8')
          }
          
          return {
            name: filename.replace('.md', ''),
            jsonPath,
            markdownPath: mdPath,
            timestamp: stats.mtime,
            size: stats.size,
            wordCount: content.split(/\s+/).length,
            charCount: content.length
          }
        })
      )
      
      return NextResponse.json({
        success: true,
        files: fileDetails
      })
    }
    
    // Handle single file request
    const filename = url
      .replace(/^https?:\/\//, '')
      .replace(/[^a-z0-9]/gi, '_')
      .toLowerCase() + '.md'
    
    const filePath = path.join(STORAGE_DIR, filename)
    const content = await fs.readFile(filePath, 'utf-8')
    
    return NextResponse.json({ success: true, content })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to load markdown' },
      { status: 500 }
    )
  }
}