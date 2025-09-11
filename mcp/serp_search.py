from serpapi import GoogleSearch
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("serpapi")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
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

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
