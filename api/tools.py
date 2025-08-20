import os
from langchain_core.tools import Tool, tool
from typing import Dict, Any, List, Optional
from tavily import TavilyClient
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.chains.base import Chain
# Load environment variables
load_dotenv()


def create_retrival_tool(retrieval_chain):
    """Create a retrieval tool."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information related to the query"""
        try:
            result = retrieval_chain.invoke({"question": query})
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_tools(rag_chain : Optional[Chain] = None) -> List:
    """Get default tools for the agent."""

    # set up tools
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=3))
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_retrival_tool(rag_chain))
    
    return tools
