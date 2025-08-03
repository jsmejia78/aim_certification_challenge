import os
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException
from uuid import uuid4

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

from prompts import SYSTEM_PROMPT
from tools import run_tavily_search

# ----------------------------------------
# Agent State Definition
# ----------------------------------------

class AgentState(TypedDict):
    query: str
    current_messages: Annotated[List[BaseMessage], add_messages]
    memory: List[BaseMessage]
    context: Dict[str, Any]
    response: str

# ----------------------------------------
# Main Agent Class
# ----------------------------------------

class LangGraphAgent():
    def __init__(self, api_keys: Dict[str, str]):

        self.agent_graph = None
        self.model = None
        self.tool_belt = None
        self.memory = []

        os.environ["OPENAI_API_KEY"] = api_keys['OPENAI_API_KEY']
        os.environ["TAVILY_API_KEY"] = api_keys['TAVILY_API_KEY']
        os.environ["LANGCHAIN_API_KEY"] = api_keys['LANGCHAIN_API_KEY']
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = f"AIM-CERT-{uuid4().hex[0:8]}"

        self._initialization(api_keys)


    def _call_model(self, state: AgentState):
        """Generate reasoning output using context + prior messages"""

        # Flatten the context from tools into a single string
        flat_context = "\n\n".join(
            f"{k.capitalize()} Result:\n{v}" for k, v in state["context"].items() if v
        ).strip()

        # Format the prompt with the context and the question
        prompt_text = SYSTEM_PROMPT.format(
            context=flat_context or "No context available.",
            question=state["query"]
        )

        sys_msg = SystemMessage(content=prompt_text)

        # Combine the current messages with the memory: could use messages = [sys_msg] + state["memory"][-3:] + state["current_messages"] later
        messages = messages = [sys_msg] + state["current_messages"] + state["memory"][-3:]

        if self.model:
            response = self.model.invoke(messages)
            return {
                "current_messages": [response],
                "response": response.content
            }
        else:
            raise HTTPException(status_code=500, detail="Model not initialized")


    def _should_continue(self, state: AgentState):
        last_message = state["current_messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "search"
        return END


    def _initialization(self, api_keys: Dict[str, str]):
        """Initialize model and graph"""
        try:
            self.tool_belt = [run_tavily_search]
            self.model = ChatOpenAI(model="gpt-4.1-mini", temperature=0).bind_tools(self.tool_belt)

            graph = StateGraph(AgentState, name="companion-agent-graph")
            graph.add_node("agent", self._call_model)
            graph.add_node("search", self._search_node)

            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", self._should_continue)
            graph.add_edge("search", "agent")

            self.agent_graph = graph.compile()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")


    def _search_node(self, state: AgentState):
        """Explicit node to run Tavily and store result"""
        query = state["query"]
        search_result = run_tavily_search.invoke(query)

        state["context"]["search"] = search_result

        tool_message = ToolMessage(
            tool_call_id="tavily_search",
            content=search_result
        )

        return {
            "current_messages": [tool_message],
            "context": state["context"]
        }


    async def chat(self, user_message: str):
        """Chat loop entrypoint"""
        try:
            inputs: AgentState = {
                "query": user_message,
                "current_messages": [HumanMessage(content=user_message)],
                "memory": self.memory,
                "context": {},
                "response": ""
            }

            final_response = ""
            tool_calls = []
            final_current_messages = []

            if self.agent_graph:
                async for chunk in self.agent_graph.astream(inputs, stream_mode="updates"):
                    for node, values in chunk.items():
                        if "current_messages" in values:
                            final_current_messages.extend(values["current_messages"])  # Keep for this loop only
                        if "response" in values:
                            final_response = values["response"]
                        if "tool_calls" in values:
                            tool_calls.extend(values["tool_calls"])

            # Append this ReAct interaction to long-term memory
            self.memory.extend(final_current_messages)

            return {
                "response": final_response or "I apologize, but I couldn't generate a response.",
                "messages": final_current_messages,
                "tool_calls": tool_calls,
                "metadata": {
                    "model": "gpt-4.1-mini",
                    "total_messages": len(final_current_messages),
                    "total_tool_calls": len(tool_calls),
                    "system_message_used": True
                },
                "status": "success"
            }

        except Exception as e:
            print(f"Error in chat method: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
