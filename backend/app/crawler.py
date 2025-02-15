from typing import List, Optional, Set
import logging
import sys
from datetime import datetime
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from urllib.parse import urljoin, urlparse

# Configure logging
logger = logging.getLogger(__name__)

# Increase recursion limit for complex pages
sys.setrecursionlimit(10000)

class InternalLink(BaseModel):
    href: str
    text: str
    status: str = 'pending'  # Default status for internal links

class DiscoveredPage(BaseModel):
    url: str
    title: Optional[str] = None
    status: str = "pending"  # Default status for parent pages
    internalLinks: Optional[List[InternalLink]] = None

class CrawlStats(BaseModel):
    subdomains_parsed: int = 0
    pages_crawled: int = 0
    data_extracted: str = "0 KB"
    errors_encountered: int = 0

class CrawlResult(BaseModel):
    markdown: str
    stats: CrawlStats

def get_browser_config() -> BrowserConfig:
    """Get browser configuration that launches a local instance"""
    return BrowserConfig(
        browser_type="chromium",
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        verbose=True,
        text_mode=True,
        light_mode=True,
        extra_args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu"
        ]
    )

def get_crawler_config(session_id: str = None) -> CrawlerRunConfig:
    """Get crawler configuration optimized for developer documentation"""
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.4,  # Increased threshold to focus on main content
            threshold_type="dynamic",
            min_word_threshold=3  # Reduced to catch short code examples
        ),
        options={
            "body_width": 120,  # Wider for code blocks
            "ignore_images": True,
            "escape_html": False,
            "code_block_style": "fenced",
            "preserve_newlines": True,
            "emphasis_marks": ['_'],  # Simplified emphasis
            "strip_comments": False,  # Keep code comments
            "headers_as_sections": True,  # Better structure
            "strip_multiple_newlines": True  # Clean output
        }
    )
    
    return CrawlerRunConfig(
        word_count_threshold=5,
        markdown_generator=markdown_generator,
        cache_mode=CacheMode.ENABLED,
        verbose=True,
        wait_until='domcontentloaded',
        wait_for_images=True,
        scan_full_page=True,
        scroll_delay=0.5,
        page_timeout=120000,
        screenshot=False,
        pdf=False,
        magic=True
    )

def normalize_url(url: str) -> str:
    """Normalize URL by removing trailing slashes and fragments"""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    if not path:
        path = '/'
    return f"{parsed.scheme}://{parsed.netloc}{path}"

async def discover_pages(
    url: str,
    max_depth: int = 3,
    current_depth: int = 1,
    seen_urls: Set[str] = None,
    parent_urls: Set[str] = None,
    all_internal_links: Set[str] = None
) -> List[DiscoveredPage]:
    if seen_urls is None:
        seen_urls = set()
    if parent_urls is None:
        parent_urls = set()
    if all_internal_links is None:
        all_internal_links = set()
    
    url = normalize_url(url)
    discovered_pages = []
    logger.info(f"Starting discovery for URL: {url} at depth {current_depth}/{max_depth}")
    
    if url in seen_urls or current_depth > max_depth:
        logger.info(f"Skipping URL: {url} (seen: {url in seen_urls}, depth: {current_depth})")
        return discovered_pages
        
    seen_urls.add(url)
    parent_urls.add(url)
    
    try:
        browser_config = get_browser_config()
        crawler_config = get_crawler_config()
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                result = await crawler.arun(url=url, config=crawler_config)
                
                title = "Untitled Page"
                if result.markdown_v2 and result.markdown_v2.fit_markdown:
                    content_lines = result.markdown_v2.fit_markdown.split('\n')
                    if content_lines:
                        potential_title = content_lines[0].strip('# ').strip()
                        if potential_title:
                            title = potential_title

                internal_links = []
                if hasattr(result, 'links') and isinstance(result.links, dict):
                    seen_internal_links = set()
                    
                    for link in result.links.get("internal", []):
                        href = link.get("href", "")
                        if not href:
                            continue
                            
                        if not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        href = normalize_url(href)
                            
                        if (href in parent_urls or 
                            href in all_internal_links or 
                            href in seen_internal_links):
                            continue
                            
                        if any(excluded in href.lower() for excluded in [
                            "login", "signup", "register", "logout",
                            "account", "profile", "admin"
                        ]):
                            continue
                            
                        base_domain = urlparse(url).netloc
                        link_domain = urlparse(href).netloc
                        if base_domain != link_domain:
                            continue
                            
                        seen_internal_links.add(href)
                        all_internal_links.add(href)
                        
                        internal_links.append(InternalLink(
                            href=href,
                            text=link.get("text", "").strip()
                        ))
                    
                    logger.info(f"Found {len(internal_links)} unique internal links at depth {current_depth}")

                primary_page = DiscoveredPage(
                    url=url,
                    title=title,
                    internalLinks=internal_links
                )
                discovered_pages.append(primary_page)

                if current_depth < max_depth:
                    for link in internal_links:
                        sub_pages = await discover_pages(
                            url=link.href,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            seen_urls=seen_urls,
                            parent_urls=parent_urls,
                            all_internal_links=all_internal_links
                        )
                        discovered_pages.extend(sub_pages)

            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
                discovered_pages.append(
                    DiscoveredPage(
                        url=url,
                        title="Error Page",
                        status="error"
                    )
                )

            return discovered_pages

    except Exception as e:
        logger.error(f"Error discovering pages: {str(e)}")
        return [DiscoveredPage(url=url, title="Main Page", status="error")]

import json
import os
import asyncio
from datetime import datetime
from pathlib import Path

# Use absolute path that matches Docker volume mount
STORAGE_DIR = Path("/app/storage/markdown")

class PageContent(BaseModel):
    """Structured content for AI consumption"""
    url: str
    title: str
    content: str
    metadata: dict
    last_crawled: datetime
    internal_links: List[str]
    sections: List[dict]  # Hierarchical content structure

async def process_content(content: str) -> dict:
    """Process markdown content into structured sections with code blocks"""
    sections = []
    current_section = {"title": None, "content": [], "subsections": [], "code_blocks": []}
    in_code_block = False
    current_code_block = {"language": "", "content": []}
    
    for line in content.split('\n'):
        if line.startswith('```'):
            if in_code_block:
                # End code block
                if current_code_block["content"]:
                    current_section["code_blocks"].append(current_code_block)
                current_code_block = {"language": "", "content": []}
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
                current_code_block["language"] = line[3:].strip()
        elif in_code_block:
            current_code_block["content"].append(line)
        elif line.startswith('# '):
            if current_section["content"] or current_section["code_blocks"]:
                sections.append(current_section)
            current_section = {"title": line[2:], "content": [], "subsections": [], "code_blocks": []}
        elif line.startswith('## '):
            if current_section["content"]:
                current_section["subsections"].append({
                    "title": line[3:],
                    "content": [],
                    "code_blocks": []
                })
        else:
            if current_section["subsections"]:
                current_section["subsections"][-1]["content"].append(line)
            else:
                current_section["content"].append(line)
    
    if current_section["content"] or current_section["code_blocks"]:
        sections.append(current_section)
    
    return sections

async def save_page_content(page_content: PageContent):
    """Save page content to both .md and .json files"""
    try:
        logger.info(f"Creating storage directory at: {STORAGE_DIR}")
        STORAGE_DIR.mkdir(exist_ok=True)
        logger.info(f"Storage directory exists: {STORAGE_DIR.exists()}, Is writable: {os.access(STORAGE_DIR, os.W_OK)}")
        
        # Generate filenames
        base_name = page_content.url.replace('https://', '').replace('http://', '').replace('/', '_').lower()
        md_path = STORAGE_DIR / f"{base_name}.md"
        json_path = STORAGE_DIR / f"{base_name}.json"
        logger.info(f"Generated paths - MD: {md_path}, JSON: {json_path}")
        
        # Save markdown with minimal headers and preserved code blocks
        markdown_content = f"""# {page_content.title}
{page_content.url}

{page_content.content.strip()}
"""
        logger.info(f"Writing markdown file ({len(markdown_content)} bytes) to: {md_path}")
        await asyncio.to_thread(lambda: md_path.write_text(markdown_content, encoding='utf-8'))
        logger.info(f"Markdown file written successfully: {md_path.exists()}, Size: {md_path.stat().st_size} bytes")
        
        # Save JSON with AI-optimized structure
        # Create minified but AI-friendly JSON structure
        json_content = {
            "u": page_content.url,  # url
            "t": page_content.title,  # title
            "s": [  # sections
                {
                    "h": s.get("title"),  # heading
                    "c": "\n".join(s.get("content", [])),  # content
                    "code": [  # code blocks
                        {
                            "l": b.get("language", ""),  # language
                            "c": "\n".join(b.get("content", []))  # code
                        }
                        for b in s.get("code_blocks", [])
                        if b.get("content")
                    ]
                }
                for s in page_content.sections
                if s.get("title") or s.get("content") or s.get("code_blocks")
            ],
            "l": [  # links
                link for link in page_content.internal_links
                if link and not link.startswith("#")  # Skip anchor links
            ]
        }
        json_str = json.dumps(json_content, separators=(',', ':'))  # Minified
        logger.info(f"Writing JSON file ({len(json_str)} bytes) to: {json_path}")
        await asyncio.to_thread(lambda: json_path.write_text(json_str, encoding='utf-8'))
        logger.info(f"JSON file written successfully: {json_path.exists()}, Size: {json_path.stat().st_size} bytes")
        
        # List directory contents after saving
        logger.info("Storage directory contents:")
        for file in STORAGE_DIR.iterdir():
            logger.info(f"- {file.name}: {file.stat().st_size} bytes")
            
    except Exception as e:
        logger.error(f"Error saving page content: {str(e)}", exc_info=True)
        raise

async def crawl_pages(pages: List[DiscoveredPage]) -> CrawlResult:
    """
    Crawl pages and save content incrementally, skipping already crawled pages
    """
    total_size = 0
    errors = 0
    stats = CrawlStats()
    
    try:
        logger.info(f"Starting crawl for {len(pages)} pages")
        logger.info(f"Storage directory: {STORAGE_DIR}")
        logger.info(f"Storage directory exists: {STORAGE_DIR.exists()}")
        if STORAGE_DIR.exists():
            logger.info(f"Current storage contents: {[f.name for f in STORAGE_DIR.iterdir()]}")
        
        browser_config = get_browser_config()
        crawler_config = get_crawler_config()
        logger.info("Initializing crawler with browser config: %s", browser_config)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for page in pages:
                # Check if recently crawled
                base_name = page.url.replace('https://', '').replace('http://', '').replace('/', '_').lower()
                json_path = STORAGE_DIR / f"{base_name}.json"
                logger.info(f"Checking existing file for {page.url} at {json_path}")
                logger.info(f"File exists: {json_path.exists()}")
                
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding='utf-8'))
                        last_crawled = datetime.fromisoformat(data['last_crawled'])
                        time_since_crawl = datetime.now() - last_crawled
                        logger.info(f"Last crawled: {last_crawled}, Time since crawl: {time_since_crawl}")
                        
                        # Skip if crawled in last 24 hours
                        if time_since_crawl.days < 1:
                            logger.info(f"Skipping recently crawled page: {page.url} (crawled {time_since_crawl.total_seconds()/3600:.1f} hours ago)")
                            continue
                        else:
                            logger.info(f"Recrawling page: {page.url} (last crawled {time_since_crawl.days} days ago)")
                    except Exception as e:
                        logger.warning(f"Error reading existing JSON for {page.url}, will recrawl: {e}", exc_info=True)
                
                try:
                    logger.info(f"Crawling page: {page.url}")
                    result = await crawler.arun(url=page.url, config=crawler_config)
                    
                    if result and hasattr(result, 'markdown_v2') and result.markdown_v2:
                        content = None
                        if hasattr(result.markdown_v2, 'fit_markdown') and result.markdown_v2.fit_markdown:
                            content = result.markdown_v2.fit_markdown
                        elif hasattr(result.markdown_v2, 'raw_markdown') and result.markdown_v2.raw_markdown:
                            content = result.markdown_v2.raw_markdown
                        
                        if content:
                            # Process content
                            sections = await process_content(content)
                            
                            # Create structured content
                            page_content = PageContent(
                                url=page.url,
                                title=page.title or "Untitled Page",
                                content=content,
                                metadata={
                                    "crawl_timestamp": datetime.now().isoformat(),
                                    "content_type": "documentation",
                                },
                                last_crawled=datetime.now(),
                                internal_links=[link.href for link in (page.internalLinks or [])],
                                sections=sections
                            )
                            
                            # Save content
                            await save_page_content(page_content)
                            
                            total_size += len(content.encode('utf-8'))
                            stats.pages_crawled += 1
                            page.status = "crawled"
                            
                            logger.info(f"Successfully processed and saved content from {page.url}")
                        else:
                            logger.warning(f"No content extracted from {page.url}")
                            errors += 1
                            page.status = "error"
                    
                except Exception as e:
                    logger.error(f"Error crawling page {page.url}: {str(e)}")
                    errors += 1
                    page.status = "error"
            
            size_str = f"{total_size} B"
            if total_size > 1024:
                size_str = f"{total_size/1024:.2f} KB"
            if total_size > 1024*1024:
                size_str = f"{total_size/(1024*1024):.2f} MB"
            
            stats.subdomains_parsed = len(pages)
            stats.data_extracted = size_str
            stats.errors_encountered = errors
            
            logger.info(f"Completed crawling with stats: {stats}")
            
            # Return empty markdown since we're saving files directly
            return CrawlResult(
                markdown="",
                stats=stats
            )
            
    except Exception as e:
        logger.error(f"Error in crawl_pages: {str(e)}")
        return CrawlResult(
            markdown="",
            stats=CrawlStats(
                subdomains_parsed=len(pages),
                pages_crawled=0,
                data_extracted="0 KB",
                errors_encountered=1
            )
        )