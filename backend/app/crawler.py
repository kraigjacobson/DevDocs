from typing import List, Optional, Set
import logging
import sys
from datetime import datetime
from pydantic import BaseModel
from bs4 import BeautifulSoup
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
        markdown_generator=markdown_generator,
        cache_mode=CacheMode.ENABLED,
        verbose=True,
        wait_until='networkidle',  # Wait for network to be idle
        wait_for_timeout=5000,  # Wait 5 seconds after page load
        page_timeout=60000,  # 1 minute timeout for page load
        screenshot=False,
        pdf=False
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
                if result.page_content:
                    # Try to find the first heading for title
                    soup = BeautifulSoup(result.page_content, 'html5lib')
                    first_heading = soup.find(['h1', 'h2', 'h3'])
                    if first_heading:
                        title = first_heading.get_text().strip()

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
    """Process HTML content into structured sections optimized for AI consumption"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, 'html5lib')  # Using html5lib for consistent and lenient HTML parsing
    
    def extract_code(element):
        """Extract code block with language"""
        pre = element.find('pre')
        if not pre:
            return None
        code = pre.find('code')
        if not code:
            code = pre  # Some pre tags don't have nested code tags
        lang = code.get('class', [''])[0].replace('language-', '') if code.get('class') else ''
        return {
            "lang": lang,
            "code": code.get_text().strip()
        }
    
    def extract_table(element):
        """Extract table data with headers"""
        headers = []
        rows = []
        
        # Extract headers
        thead = element.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
        
        # Extract rows
        tbody = element.find('tbody') or element
        for tr in tbody.find_all('tr'):
            row = [td.get_text().strip() for td in tr.find_all('td')]
            if row:  # Skip empty rows
                rows.append(row)
        
        return {
            "type": "table",
            "headers": headers,
            "rows": rows
        }
    
    def extract_list(element):
        """Extract list items maintaining nested structure"""
        items = []
        for li in element.find_all('li', recursive=False):
            item = {
                "text": li.get_text().strip(),
                "sub_items": []
            }
            # Handle nested lists
            nested_list = li.find(['ul', 'ol'])
            if nested_list:
                item["sub_items"] = extract_list(nested_list)["items"]
            items.append(item)
        
        return {
            "type": "list",
            "list_type": "ordered" if element.name == "ol" else "unordered",
            "items": items
        }
    
    def process_element(element, level=1):
        """Recursively process elements maintaining hierarchy"""
        if not element or not hasattr(element, 'name'):
            return None
            
        # Handle headings
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return {
                "t": element.get_text().strip(),  # title
                "l": int(element.name[1]),  # level
                "c": []  # children
            }
        
        # Handle notes, warnings, tips
        if element.name == 'div' and any(c in element.get('class', []) for c in ['note', 'warning', 'tip', 'info']):
            note_type = next((c for c in element.get('class', []) if c in ['note', 'warning', 'tip', 'info']), 'note')
            return {
                "type": note_type,
                "text": element.get_text().strip()
            }
        
        # Handle code blocks
        if element.name == 'pre':
            code_info = extract_code(element)
            if code_info:
                return {
                    "type": "code",
                    "lang": code_info["lang"],
                    "code": code_info["code"]
                }
        
        # Handle tables
        if element.name == 'table':
            return extract_table(element)
        
        # Handle lists
        if element.name in ['ul', 'ol']:
            return extract_list(element)
        
        # Handle paragraphs and other text content
        if element.name == 'p':
            # Check for inline code
            inline_code = element.find('code')
            if inline_code:
                return {
                    "type": "inline_code",
                    "text": element.get_text().strip(),
                    "code": inline_code.get_text().strip()
                }
            return {
                "type": "text",
                "text": element.get_text().strip()
            }
        
        # Handle links
        if element.name == 'a' and element.get('href'):
            return {
                "type": "link",
                "text": element.get_text().strip(),
                "url": element['href']
            }
        
        return None
    
    sections = []
    current_section = None
    
    # Process all relevant elements
    for element in soup.find_all([
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'p', 'div', 'pre', 'table', 'ul', 'ol',
        'blockquote', 'a'
    ]):
        result = process_element(element)
        if result:
            if result.get('l'):  # It's a heading
                if current_section and current_section.get('c'):
                    sections.append(current_section)
                current_section = result
            elif current_section:
                current_section['c'].append(result)
            else:
                # Handle content before first heading
                current_section = {
                    "t": "Introduction",
                    "l": 1,
                    "c": [result]
                }
    
    if current_section and current_section.get('c'):
        sections.append(current_section)
    
    # If no sections were created but we have content, create a default section
    if not sections and soup.find(['p', 'div', 'pre', 'table', 'ul', 'ol']):
        sections.append({
            "t": "Content",
            "l": 1,
            "c": [
                result for element in soup.find_all(['p', 'div', 'pre', 'table', 'ul', 'ol'])
                if (result := process_element(element))
            ]
        })
    
    return sections

async def save_page_content(page_content: PageContent):
    """Save page content to both .md and .json files with enhanced structure"""
    try:
        logger.info(f"Creating storage directory at: {STORAGE_DIR}")
        STORAGE_DIR.mkdir(exist_ok=True)
        logger.info(f"Storage directory exists: {STORAGE_DIR.exists()}, Is writable: {os.access(STORAGE_DIR, os.W_OK)}")
        
        # Generate filenames
        base_name = page_content.url.replace('https://', '').replace('http://', '').replace('/', '_').lower()
        md_path = STORAGE_DIR / f"{base_name}.md"
        json_path = STORAGE_DIR / f"{base_name}.json"
        logger.info(f"Generated paths - MD: {md_path}, JSON: {json_path}")
        
        # Process the content through BeautifulSoup
        sections = await process_content(page_content.content)
        
        def format_list_items_md(items, list_type="unordered", indent=""):
            """Format list items for markdown with proper indentation"""
            result = []
            for i, item in enumerate(items, 1):
                bullet = "- " if list_type == "unordered" else f"{i}. "
                result.append(f"{indent}{bullet}{item['text']}")
                if item.get('sub_items'):
                    sub_indent = indent + "  "
                    result.extend(format_list_items_md(item['sub_items'], list_type, sub_indent))
            return result
        
        def format_table_md(table):
            """Format table for markdown"""
            if not table.get('rows'):
                return ""
            
            headers = table.get('headers', [])
            if not headers and table['rows']:
                # Create numeric headers if none exist
                headers = [f"Column {i+1}" for i in range(len(table['rows'][0]))]
            
            # Create header row
            result = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"
            ]
            
            # Add data rows
            for row in table['rows']:
                result.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            return "\n".join(result)
        
        def format_content_md(content_items):
            """Format content items for markdown with enhanced styling"""
            md_lines = []
            for item in content_items:
                if not item:
                    continue
                    
                item_type = item.get('type', 'text')
                
                if item_type == 'text':
                    md_lines.append(f"\n{item['text']}\n")
                
                elif item_type in ['note', 'warning', 'tip', 'info']:
                    symbol = {
                        'note': 'ℹ️',
                        'warning': '⚠️',
                        'tip': '💡',
                        'info': 'ℹ️'
                    }.get(item_type, 'ℹ️')
                    md_lines.append(f"\n> {symbol} **{item_type.upper()}**: {item['text']}\n")
                
                elif item_type == 'code':
                    lang = item.get('lang', '')
                    md_lines.append(f"\n```{lang}\n{item['code']}\n```\n")
                
                elif item_type == 'inline_code':
                    md_lines.append(f"`{item['code']}`")
                
                elif item_type == 'list':
                    md_lines.extend(format_list_items_md(item['items'], item.get('list_type', 'unordered')))
                    md_lines.append("")  # Add spacing after list
                
                elif item_type == 'table':
                    md_lines.append("\n" + format_table_md(item) + "\n")
                
                elif item_type == 'link':
                    md_lines.append(f"[{item['text']}]({item['url']})")
            
            return "\n".join(md_lines)
        
        # Generate markdown content with enhanced formatting
        markdown_lines = [
            f"# {page_content.title}",
            f"\nSource: {page_content.url}",
            f"\nLast Updated: {page_content.last_crawled.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Table of Contents\n"
        ]
        
        # Generate table of contents
        for section in sections:
            if section.get('t'):
                level = section.get('l', 1)
                indent = "  " * (level - 1)
                # Create GitHub-compatible anchor links
                anchor = section['t'].lower().replace(' ', '-').replace('/', '').replace('(', '').replace(')', '')
                markdown_lines.append(f"{indent}- [{section['t']}](#{anchor})")
        
        markdown_lines.append("\n---\n")  # Add separator after TOC
        
        # Add section content
        for section in sections:
            if section.get('t'):
                level = section.get('l', 1)
                markdown_lines.append(f"\n{'#' * level} {section['t']}\n")
                if section.get('c'):
                    markdown_lines.append(format_content_md(section['c']))
        
        markdown_content = "\n".join(markdown_lines)
        logger.info(f"Writing markdown file ({len(markdown_content)} bytes) to: {md_path}")
        await asyncio.to_thread(lambda: md_path.write_text(markdown_content, encoding='utf-8'))
        logger.info(f"Markdown file written successfully: {md_path.exists()}, Size: {md_path.stat().st_size} bytes")
        
        # Create enhanced AI-friendly JSON structure
        json_content = {
            "d": {  # document
                "t": page_content.title,  # title
                "u": page_content.url,  # url
                "m": page_content.metadata,  # metadata
                "ts": page_content.last_crawled.isoformat(),  # timestamp
                "s": [  # sections
                    {
                        "t": s.get("t"),  # title
                        "l": s.get("l", 1),  # level
                        "c": [  # content
                            {
                                "k": item.get("type", "text"),  # kind
                                **({"t": item["text"]} if "text" in item else {}),  # text
                                **({"l": item["lang"], "c": item["code"]} if item.get("type") == "code" else {}),  # code
                                **({"i": item["items"]} if item.get("type") == "list" else {}),  # list items
                                **({"h": item["headers"], "r": item["rows"]} if item.get("type") == "table" else {}),  # table
                                **({"u": item["url"]} if item.get("type") == "link" else {})  # link
                            }
                            for item in s.get("c", [])
                            if item
                        ]
                    }
                    for s in sections
                    if s and s.get("t")
                ]
            },
            "r": [  # references
                {
                    "u": link,  # url
                    "t": datetime.now().isoformat()  # timestamp when reference was found
                }
                for link in page_content.internal_links
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
                    
                    if result and hasattr(result, 'page_content') and result.page_content:
                        content = result.page_content
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