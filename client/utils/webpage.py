from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter

from serpapi import GoogleSearch
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

async def google_search(query: str, topk: int = 10) -> list[dict]:
    """
    Run a Google search via SerpAPI and return the organic results.
    
    Parameters
    ----------
    query : str
        The search query string (e.g., "Coffee")
    topk : int
        The number of the search result 
    Returns
    -------
    list[dict]
        The list of organic search results from Google.
    """
    import os
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError("Please set SERPAPI_KEY environment variable")
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": topk
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

async def crawl_page(url: str) -> str:
    """Deep crawl and extract key content from a web page (Markdown format).

    This tool is designed to perform *deep analysis* on a specific link
    retrieved from an earlier search step (e.g., via a web search tool).
    Given a fully qualified HTTP(S) URL, it fetches the web page,
    removes boilerplate content (menus, ads, nav bars, etc.), and
    extracts the core readable content, returning it as a clean,
    structured Markdown string.

    This Markdown output is well-suited for downstream processing by
    large language models (LLMs) for tasks such as:
    - Answering user questions from a specific page
    - Summarizing long articles or reports
    - Extracting facts, definitions, lists, or instructions
    - Contextual search over high‑signal content

    This is often used as a **follow-up** step after a general-purpose
    search tool (e.g., via SearxNG), when the agent needs to "click through"
    to an individual link and analyze its full content in a readable form.

    Args:
        url (str): A valid, fully-qualified URL (http:// or https://) that
            points to a real and accessible web page (e.g. news article,
            blog post, research page).

    Returns:
        str: Markdown-formatted main content of the page. If the crawl fails
            (due to network errors, access restrictions, or page layout
            issues), a plain-text error message is returned instead.
    """
    browser_config = BrowserConfig(
        headless=True,  
        verbose=True,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=0)
        ),
        # markdown_generator=DefaultMarkdownGenerator(
        #     content_filter=BM25ContentFilter(user_query="WHEN_WE_FOCUS_BASED_ON_A_USER_QUERY", bm25_threshold=1.0)
        # ),
    )
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            return result.markdown
    except Exception as exc:
        return f"⚠️ Crawl error: {exc!s}"