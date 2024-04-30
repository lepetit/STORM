from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from duckduckgo_search import DDGS


'''
# Tavily is typically a better search engine, but your free queries are limited
tavily_search = TavilySearchResults(max_results=4)

@tool
def search_engine(query: str):
    """Search engine to the internet."""

    results = tavily_search.invoke(query)
    return [{"content": r["content"], "url": r["url"]} for r in results]
'''

# Duck Duck Go search
ddg_engine = DDGS()

"""DuckDuckGo text search generator. Query params: https://duckduckgo.com/params.

Args:
    keywords: keywords for query.
    region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
    safesearch: on, moderate, off. Defaults to "moderate".
    timelimit: d, w, m, y. Defaults to None.
    backend: api, html, lite. Defaults to api.
        api - collect data from https://duckduckgo.com,
        html - collect data from https://html.duckduckgo.com,
        lite - collect data from https://lite.duckduckgo.com.
    max_results: max number of results. If None, returns results only from the first response. Defaults to None.

Returns:
    List of dictionaries with search results.
"""
@tool
async def search_engine(query: str, region= "wt-wt"):
    """Search engine to the internet."""
    results = ddg_engine.text(keywords = query,
                              region= region,
                              safesearch='moderate',
                              timelimit=None,
                              max_results=50)
    
    print("Search: ", query, " found:", results.length)
    return [{"content": r["body"], "url": r["href"]} for r in results]

@tool
async def news_search_engine(query: str, region= "wt-wt"):
    """Search engine to the internet."""
    results = ddg_engine.news(keywords = query,
                              region= region,
                              safesearch='moderate',
                              timelimit='5g',
                              max_results=50)
    
    print("Search: ", query, " found:", results.length)
    return [{"content": r["body"], "url": r["href"]} for r in results]
