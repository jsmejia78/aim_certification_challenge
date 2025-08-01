from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# ----------------------------------------
# Tavily Tool Decorator
# ----------------------------------------

tavily_tool = TavilySearchResults(max_results=5)

@tool
def run_tavily_search(query: str) -> str:
    """Perform a Tavily web search and return a summarized result block"""
    results = tavily_tool.invoke({"query": query})
    if not results or not results.get("results"):
        return "No relevant results found."
    return "\n\n".join([f"- {r['content']}" for r in results["results"]])
