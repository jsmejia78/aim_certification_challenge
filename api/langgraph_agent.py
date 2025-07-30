import os
from typing import Dict
from fastapi import HTTPException
import asyncio
#from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
import yfinance as yf
import json
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from uuid import uuid4
import wikipedia

# Optional: set language
wikipedia.set_lang("en")

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

class LangGraphAgent:
    def __init__(self):
        
        self.agent_graph = None
        self.model = None
        self.tool_belt = None
        self.tool_node = None

    def _call_model(self, state):
        messages = state["messages"]
        if self.model is not None:
            response = self.model.invoke(messages)
            return {"messages" : [response]}
        else:
            raise HTTPException(status_code=500, detail="Model not initialized")

    def _should_continue(self,state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "action"
        return END

    async def _initialization(self, api_keys: Dict[str, str], model_name: str, temperature: float):
        """Initialize everything needed for the agent"""
        try:
            # Set the API key as environment variable for the models\Agent
            os.environ["OPENAI_API_KEY"] = api_keys['OPENAI_API_KEY']
            os.environ["TAVILY_API_KEY"] = api_keys['TAVILY_API_KEY']
            os.environ["LANGCHAIN_API_KEY"] = api_keys['LANGCHAIN_API_KEY']
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = f"AIE7 - LangGraph - {uuid4().hex[0:8]}"

            # Yahoo Finance tool
            def get_stock_info(symbol: str) -> str:
                """Get comprehensive stock information including price, news, and financials"""
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get recent news
                    news = ticker.news[:3] if ticker.news else []
                    
                    # Format the response
                    result = {
                        "symbol": symbol,
                        "company_name": info.get("longName", "N/A"),
                        "current_price": info.get("currentPrice", "N/A"),
                        "market_cap": info.get("marketCap", "N/A"),
                        "pe_ratio": info.get("trailingPE", "N/A"),
                        "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                        "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                        "recent_news": [{"title": n.get("title", ""), "link": n.get("link", "")} for n in news]
                    }
                    return json.dumps(result, indent=2)
                except Exception as e:
                    return f"Error getting data for {symbol}: {str(e)}"
            
            stock_info_tool = Tool.from_function(
                func=get_stock_info,
                name="yahoo_finance_stock_info",
                description="Get detailed stock information including current price, market cap, PE ratio, 52-week range, and recent news for any stock symbol"
            )

            #tavily_tool = TavilySearchResults(max_results=5)
            tavily_client = TavilyClient(api_key = os.environ["TAVILY_API_KEY"])

            def tavily_search(query: str, max_results: int = 3) -> str:
                results = tavily_client.search(query=query, max_results=max_results)
                summaries = "\n".join(f"{i+1}. {r['title']} - {r['url']}" for i, r in enumerate(results["results"]))
                return summaries or "No results found."

            tavily_tool = Tool.from_function(
                func=tavily_search,
                name="tavily_search",
                description="Search the web for the latest news and information"
            )

            # Wikipedia tool
            def wikipedia_lookup(query: str) -> str:
                """
                Searches Wikipedia for the query and returns a summary.
                """
                try:
                    summary = wikipedia.summary(query, sentences=5)
                    return f"Wikipedia summary for '{query}':\n{summary}"
                except wikipedia.DisambiguationError as e:
                    options = ", ".join(e.options[:5])
                    return f"The term '{query}' is ambiguous. Did you mean: {options}?"
                except wikipedia.PageError:
                    return f"No Wikipedia page found for '{query}'."
                except Exception as e:
                    return f"Error searching Wikipedia: {str(e)}"

            wikipedia_tool = Tool.from_function(
                func=wikipedia_lookup,
                name="wikipedia_lookup",
                description="Search Wikipedia for the queried information."
            )

            # Tool belt
            self.tool_belt = [
                tavily_tool,
                stock_info_tool,
                wikipedia_tool
            ]

            self.model = ChatOpenAI(model=model_name, temperature=temperature)
            self.model = self.model.bind_tools(self.tool_belt)

            self.tool_node = ToolNode(self.tool_belt)

            uncompiled_graph = StateGraph(AgentState)

            uncompiled_graph.add_node("agent", self._call_model)
            uncompiled_graph.add_node("action", self.tool_node)
            uncompiled_graph.set_entry_point("agent")

            uncompiled_graph.add_conditional_edges(
                "agent",
                self._should_continue
            )

            uncompiled_graph.add_edge("action", "agent")
            self.agent_graph = uncompiled_graph.compile()

            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")
    
    
    async def chat(self, user_message: str, api_keys: Dict[str, str], system_message: str = "", model_name: str = "gpt-4o-mini", temperature: float = 0):
        """ chat with the agent """
        try:
            if not self.agent_graph:
                await self._initialization(api_keys, model_name, temperature)
            
            # Prepare input messages, including system message if provided
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=user_message))
            
            inputs : AgentState = {"messages": messages}

            # Store messages and tool calls simply
            all_messages = []
            tool_calls = []
            final_response = ""
            
            if self.agent_graph is not None:
                async for chunk in self.agent_graph.astream(inputs, stream_mode="updates"):
                    for node, values in chunk.items():

                        # Store messages from nodes
                        if "messages" in values:
                            for message in values["messages"]:
                                all_messages.append(message)
                                
                                # Store tool calls if present
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    tool_calls.extend(message.tool_calls)
                                
                                # Update final response if this is from the agent node
                                if node == "agent" and hasattr(message, 'content') and message.content:
                                    final_response = message.content

            # Handle empty response case
            if not final_response:
                final_response = "I apologize, but I couldn't generate a response. Please try again."
            
            # Return structured response
            return {
                "response": final_response,
                "messages": all_messages,
                "tool_calls": tool_calls,
                "metadata": {
                    "model": "gpt-4o-mini",
                    "total_messages": len(all_messages),
                    "total_tool_calls": len(tool_calls),
                    "system_message_used": bool(system_message)
                },
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error in chat method: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

