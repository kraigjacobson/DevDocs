import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

const STORAGE_DIR = path.join(process.cwd(), 'storage/markdown')

export async function POST(request: Request) {
  try {
    const { url, content, metadata } = await request.json()
    
    // Create storage directory if it doesn't exist
    await fs.mkdir(STORAGE_DIR, { recursive: true })
    
    // Generate base filename from URL
    const baseFilename = url
      .replace(/^https?:\/\//, '')
      .replace(/[^a-z0-9]/gi, '_')
      .toLowerCase()
    
    const mdPath = path.join(STORAGE_DIR, `${baseFilename}.md`)
    const jsonPath = path.join(STORAGE_DIR, `${baseFilename}.json`)
    
    // Update or create markdown file
    await fs.writeFile(mdPath, content, 'utf-8')
    
    // Update or create JSON file
    let jsonContent = {
      url,
      content,
      metadata: {
        ...metadata,
        lastUpdated: new Date().toISOString(),
        wordCount: content.split(/\s+/).length,
        charCount: content.length
      }
    }
    
    // If JSON file exists, preserve existing metadata
    try {
      const existingJson = await fs.readFile(jsonPath, 'utf-8')
      const existingData = JSON.parse(existingJson)
      jsonContent = {
        ...existingData,
        ...jsonContent,
        metadata: {
          ...existingData.metadata,
          ...jsonContent.metadata
        }
      }
    } catch (e) {
      // No existing JSON file, use new content
    }
    
    await fs.writeFile(jsonPath, JSON.stringify(jsonContent, null, 2), 'utf-8')
    
    return NextResponse.json({
      success: true,
      files: {
        markdown: mdPath,
        json: jsonPath
      }
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to save content' },
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