import os
from langchain_core.tools import Tool, tool
from tavily import TavilyClient
from dotenv import load_dotenv
from pydantic import BaseModel
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
    description="Search the web for the latest information to build context related to the query"
)

class RAGInput(BaseModel):
    query: str
    retriever: object  # or use your specific retriever type if available

@tool
def custom_rag_tool(input: RAGInput) -> str:
    """
    A custom RAG tool that uses the provided retriever to fetch relevant context for the input query.
    """
    docs = input.retriever(input.query)

    # Extract content for display and logging
    if isinstance(docs, str):
        return docs
    elif isinstance(docs, dict):
        return docs.get("answer", str(docs))  # Prefer 'answer' field if it exists
    else:
        return str(docs)