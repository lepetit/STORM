from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool


# Tavily is typically a better search engine, but your free queries are limited
tavily_search = TavilySearchResults(max_results=4)

@tool
def search_engine(query: str):
    """Search engine to the internet."""

    results = tavily_search.invoke(query)
    return [{"content": r["content"], "url": r["url"]} for r in results]

'''
# DDG
search_engine = DuckDuckGoSearchAPIWrapper()

@tool
async def search_engine(query: str):
    """Search engine to the internet."""
    results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
    return [{"content": r["body"], "url": r["href"]} for r in results]
'''