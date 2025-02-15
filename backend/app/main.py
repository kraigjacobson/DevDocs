from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List
import uvicorn
import logging
import psutil
import os
from pathlib import Path
from .crawler import discover_pages, crawl_pages, DiscoveredPage, CrawlResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crawl4AI Backend")

# Configure CORS to allow requests from our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiscoverRequest(BaseModel):
    url: str
    depth: int = Field(default=3, ge=1, le=5)  # Enforce depth between 1 and 5

    @validator('depth')
    def validate_depth(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Depth must be between 1 and 5')
        return v

class CrawlRequest(BaseModel):
    pages: List[DiscoveredPage]

class MCPStatusResponse(BaseModel):
    status: str
    pid: int | None = None
    details: str | None = None

class MCPLogsResponse(BaseModel):
    logs: List[str]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/mcp/config")
async def get_mcp_config():
    """Get MCP server configuration"""
    try:
        # Get the project root directory
        root_dir = str(Path(__file__).parents[2].resolve())
        logger.info(f"Project root directory: {root_dir}")

        config = {
            "mcpServers": {
                "fast-markdown": {
                    "command": str(Path(root_dir) / "fast-markdown-mcp/venv/bin/python"),
                    "args": [
                        "-m", "fast_markdown_mcp.server",
                        str(Path(root_dir) / "storage/markdown")
                    ],
                    "env": {
                        "PYTHONPATH": str(Path(root_dir) / "fast-markdown-mcp/src")
                    },
                    "disabled": False
                }
            }
        }
        return config
    except Exception as e:
        logger.error(f"Error generating config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mcp/status", response_model=MCPStatusResponse)
async def get_mcp_status():
    """Check if the MCP server is running with multiple verification steps"""
    try:
        logger.info("Checking MCP server status")
        mcp_pid = None
        process_found = False
        socket_active = False
        details = []

        # Step 1: Check for the process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'python' in cmdline.lower() and 'fast_markdown_mcp.server' in cmdline:
                        process_found = True
                        mcp_pid = proc.info['pid']
                        details.append(f"Process found (PID: {mcp_pid})")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Step 2: Verify the process is responding
        if process_found:
            try:
                process = psutil.Process(mcp_pid)
                if process.status() == psutil.STATUS_ZOMBIE:
                    details.append("Process is zombie")
                    process_found = False
                else:
                    details.append("Process is active")
            except psutil.NoSuchProcess:
                details.append("Process disappeared")
                process_found = False

        # Determine final status
        if process_found:
            logger.info("MCP server is operational")
            return {
                "status": "running",
                "pid": mcp_pid,
                "details": "; ".join(details)
            }
        else:
            logger.info("MCP server is not running")
            return {
                "status": "stopped",
                "details": "; ".join(details)
            }

    except Exception as e:
        logger.error(f"Error checking MCP status: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "details": f"Error checking status: {str(e)}"
        }

@app.get("/api/mcp/logs", response_model=MCPLogsResponse)
async def get_mcp_logs():
    """Get MCP server logs"""
    try:
        logger.info("Fetching MCP server logs")
        log_path = Path("logs/mcp.log")
        if not log_path.exists():
            logger.info("No log file found")
            return {"logs": []}
        
        with open(log_path, 'r') as f:
            # Get last 50 lines but skip empty lines
            logs = [line.strip() for line in f.readlines() if line.strip()][-50:]
        
        logger.info(f"Retrieved {len(logs)} log lines")
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}", exc_info=True)
        return {"logs": [f"Error reading logs: {str(e)}"]}

@app.post("/api/discover")
async def discover_endpoint(request: DiscoverRequest):
    """Discover and crawl pages related to the provided URL"""
    try:
        logger.info(f"Received discover request for URL: {request.url} with depth: {request.depth}")
        pages = await discover_pages(request.url, max_depth=request.depth)
        
        # Log the discovery results
        if pages:
            logger.info(f"Successfully discovered {len(pages)} pages")
            for page in pages:
                logger.debug(f"Discovered page: {page.url} ({page.status})")
        else:
            logger.warning("No pages discovered")
            return {
                "message": "No pages discovered",
                "success": True,
                "pages": [],
                "crawl_result": None
            }
            
        # Automatically crawl the discovered pages
        logger.info(f"Starting crawl for {len(pages)} discovered pages")
        crawl_result = await crawl_pages(pages)
        logger.info(f"Crawl completed with stats: {crawl_result.stats}")
        
        return {
            "message": f"Found and crawled {len(pages)} pages",
            "success": True,
            "pages": pages,
            "crawl_result": crawl_result
        }
    except Exception as e:
        logger.error(f"Error discovering pages: {str(e)}", exc_info=True)
        # Return a structured error response
        return {
            "pages": [],
            "message": f"Error discovering pages: {str(e)}",
            "success": False,
            "error": str(e)
        }

@app.post("/api/crawl")
async def crawl_endpoint(request: CrawlRequest):
    """Crawl the provided pages and save content incrementally"""
    try:
        logger.info(f"Received crawl request for {len(request.pages)} pages")
        result = await crawl_pages(request.pages)
        
        # Log the results
        logger.info(f"Successfully crawled pages. Stats: {result.stats}")
        
        # Return stats and list of processed files
        storage_dir = Path("storage/markdown")
        processed_files = []
        
        for page in request.pages:
            base_name = page.url.replace('https://', '').replace('http://', '').replace('/', '_').lower()
            md_path = storage_dir / f"{base_name}.md"
            json_path = storage_dir / f"{base_name}.json"
            
            if md_path.exists() and json_path.exists():
                try:
                    json_data = json.loads(json_path.read_text(encoding='utf-8'))
                    processed_files.append({
                        "url": page.url,
                        "status": page.status,
                        "files": {
                            "markdown": str(md_path),
                            "json": str(json_path)
                        },
                        "metadata": json_data.get("metadata", {})
                    })
                except Exception as e:
                    logger.error(f"Error reading processed file data: {e}")
        
        return {
            "stats": result.stats.dict(),
            "processed_files": processed_files,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error crawling pages: {str(e)}", exc_info=True)
        return {
            "stats": {
                "subdomains_parsed": 0,
                "pages_crawled": 0,
                "data_extracted": "0 KB",
                "errors_encountered": 1
            },
            "processed_files": [],
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=24125,
        reload=True,
        log_level="info"
    )