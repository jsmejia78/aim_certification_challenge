import os
from langchain_core.tools import Tool
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------------------------
# Tavily Tool Decorator
# ----------------------------------------

#tavily_tool = TavilySearchResults(max_results=5)
tavily_client = TavilyClient(api_key = os.environ["TAVILY_API_KEY"])

def tavily_search(query: str, max_results: int = 5) -> str:
    results = tavily_client.search(query=query, max_results=max_results)

    if not results["results"]:
        return "No results found."

    summaries = []
    for i, r in enumerate(results["results"]):
        title = r.get("title", "Untitled")
        url = r.get("url", "No URL")
        snippet = r.get("content", r.get("snippet", "No content available."))

        summaries.append(f"{i+1}. {title}\n{snippet}\nSource: {url}")

    return "\n\n".join(summaries)

tavily_tool = Tool.from_function(
    func=tavily_search,
    name="tavily_search",
    description="Search the web for the latest information related to query"
)

