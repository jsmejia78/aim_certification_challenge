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
from dotenv import load_dotenv


from prompts import LLM_PROMPT
from tools import tavily_tool

# ----------------------------------------
# Agent State Definition
# ----------------------------------------

class AgentState(TypedDict):
    query: str
    current_messages: Annotated[List[BaseMessage], add_messages]
    agent_memory: List[BaseMessage]
    context: Dict[str, Any]
    response: str

# ----------------------------------------
# Main Agent Class
# ----------------------------------------

class LangGraphAgent():
    def __init__(self):

        self.agent_graph = None
        self.model = None
        self.tool_belt = None
        self.agent_memory = []
        self.count = 0
        # Automatically loads variables from .env file into os.environ
        load_dotenv()

        assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
        assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
        assert os.getenv("LANGCHAIN_API_KEY"), "Missing LANGCHAIN_API_KEY"

        #os.environ["LANGCHAIN_TRACING_V2"] = "true"
        #os.environ["LANGCHAIN_PROJECT"] = f"AIM-CERT-{uuid4().hex[0:8]}"

        self._initialization()


    def _call_model(self, state: AgentState):
        """Generate reasoning output using context + prior messages"""

        # SAFELY get context, agent_memory, query, current_messages
        context = state.get("context", {})
        agent_memory = state.get("agent_memory", [])
        query = state.get("query", "")
        current = state.get("current_messages", [])

        # Flatten the context from tools into a single string
        flat_context = "\n\n".join(
            f"{k.capitalize()} Result:\n{v}" for k, v in context.items() if v
        ).strip()

        # Format the prompt with the context and the question
        prompt_text = LLM_PROMPT.format(
            context=flat_context or "No context available.",
            question=query
        )

        sys_msg = SystemMessage(content=prompt_text)

        # Combine messages: last 3 from agent_memory + current messages
        if len(agent_memory) >= 3:
            messages = [sys_msg] + current + agent_memory[-3:]
        elif len(agent_memory) >= 2:
            messages = [sys_msg] + current + agent_memory[-2:]
        elif len(agent_memory) >= 1:
            messages = [sys_msg] + current + agent_memory[-1:]
        else:
            messages = [sys_msg] + current + agent_memory

        if self.model:
            response = self.model.invoke(messages)
            return {
                **state,  # propagate all values, including agent_memory
                "current_messages": [response],
                "response": response.content,
                "agent_memory": state.get("agent_memory", []) 
            }
        else:
            raise HTTPException(status_code=500, detail="Model not initialized")



    def _should_continue(self, state: AgentState):
        current = state.get("current_messages", [])
        last_message = current[-1] if current else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "search"
        return END


    def _initialization(self):
        """Initialize model and graph"""
        try:
            self.tool_belt = [tavily_tool]
            self.model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7).bind_tools(self.tool_belt)

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
        query = state.get("query", "")
        search_result = tavily_tool.invoke(query)

        updated_context = state.get("context", {}).copy()
        updated_context["search"] = search_result

        last_message = state.get("current_messages", [])[-1]

        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            raise HTTPException(status_code=500, detail="Tool was expected to be called, but no tool_calls found.")

        # Create a ToolMessage for every tool_call ID (you could have >1 in a future case)
        tool_messages = [
            ToolMessage(
                tool_call_id=call["id"],  # ID from the assistant's tool_call
                content=search_result     # You can refine this per tool call if needed
            )
            for call in tool_calls
        ]

        return {
            **state,
            "current_messages": tool_messages,
            "context": updated_context,
            "memory": state.get("memory", [])
        }

    from langchain_core.messages import HumanMessage, BaseMessage

    async def chat(self, user_message: str):
        """Chat loop entrypoint"""
        try:
            inputs: AgentState = {
                "query": user_message,
                "current_messages": [HumanMessage(content=user_message)],
                "agent_memory": self.agent_memory,
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
                            for msg in values["current_messages"]:
                                final_current_messages.append(msg)
                                # Extract tool calls if they exist in AssistantMessage
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    tool_calls.extend(msg.tool_calls)
                        if "response" in values:
                            final_response = values["response"]

            # Append ReAct interaction to long-term agent memory
            self.agent_memory.extend(final_current_messages)

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
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


        except Exception as e:
            print(f"Error in chat method: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
