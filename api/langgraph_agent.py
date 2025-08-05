import os
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException
from uuid import uuid4

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from enum import Enum

from prompts import SYSTEM_PROMPT
from tools import tavily_tool, custom_rag_tool
from retrievers import get_retrieval_chains_and_wrappers
from vector_stores import VectorStoresManager
from data_loader import DataLoader

class RetrievalEnums(Enum):
    NAIVE = "base_retrieval_chain"
    BM25 = "bm25_retrieval_chain"
    CONTEXTUAL_COMPRESSION = "contextual_compression_retrieval_chain"
    MULTI_QUERY = "multi_query_retrieval_chain"
    PARENT_DOCUMENT = "parent_document_retrieval_chain"
    ENSEMBLE = "ensemble_retrieval_chain"

# ----------------------------------------
# Agent State Definition
# ----------------------------------------

class AgentState(TypedDict):
    query: str
    current_messages: Annotated[List[BaseMessage], add_messages]
    agent_memory: List[BaseMessage]
    context: Dict[str, List[Any]]
    response: str

# ----------------------------------------
# Main Agent Class
# ----------------------------------------

class LangGraphAgent():
    def __init__(self, retriever_mode: RetrievalEnums, MODE: str, langchain_project_name: str):

        self.agent_graph = None
        self.react_model = None
        self.tool_belt = None
        self.agent_memory = []
        self.count = 0

        self.retrievers_config = None
        self.retriever_mode = retriever_mode
        self.retrieval_llm = None
        self.rag_prompt = None
        self.rag_model = None
        self.retrival_chains = None
        self.retrival_wrappers = None
        self.MODE = MODE
        self.loaded_rag_data = None
        self.dbs_manager = None

        # Automatically loads variables from .env file into os.environ
        load_dotenv()

        assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
        assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
        assert os.getenv("LANGCHAIN_API_KEY"), "Missing LANGCHAIN_API_KEY"
        assert os.getenv("COHERE_API_KEY"), "Missing COHERE_API_KEY"

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = langchain_project_name #f"AIM-CERT-{uuid4().hex[0:8]}"

        self._initialization()


    def _call_model(self, state: AgentState):
        """Generate reasoning output using context + prior messages"""

        agent_memory = state.get("agent_memory", [])
        current = state.get("current_messages", [])

        # Combine messages: last 3 from agent_memory + current messages
        if len(agent_memory) >= 3:
            messages =  + current + agent_memory[-3:]
        else:
            messages =  current + agent_memory

        if self.react_model:
            response = self.react_model.invoke(messages)
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

        if last_message and hasattr(last_message, "tool_calls"):
            for call in last_message.tool_calls:
                if call["name"] == "tavily_search":
                    return "search"
                elif call["name"] == "custom_rag_tool":
                    return "rag"
        return END

    def _initialization(self):
        """Initialize models, graph, and dependencies"""
        try:

            # set up tool belt
            self.tool_belt = [tavily_tool, custom_rag_tool]

            # set up model
            self.react_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7).bind_tools(self.tool_belt)
            self.rag_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

            graph = StateGraph(AgentState, name="companion-agent-graph")

            graph.add_node("agent", self._call_model)
            graph.add_node("search", self._search_node)
            graph.add_node("rag", self._rag_node)

            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", self._should_continue)
            graph.add_edge("search", "agent")
            graph.add_edge("rag", "agent")

            self.agent_graph = graph.compile()

            # set up retrievers
            data_loader = DataLoader("pd_blogs_filtered")
            self.loaded_rag_data = data_loader.load_data()

            TEST = "First Improved"

            if TEST == "First Naive":
                self.dbs_manager = VectorStoresManager(    
                    MODE="baseline",
                    loaded_data=self.loaded_rag_data,
                    chunk_config={"enabled": True, "params": {"chunk_size": 750, "chunk_overlap": 0}},
                    embeddings_model_name="text-embedding-3-small",
                    chat_model="gpt-4.1-mini",
                    collection_name="Rag Loaded Data Naive"
                 )
            else:
                self.dbs_manager = VectorStoresManager(    
                    MODE="baseline",
                    loaded_data=self.loaded_rag_data,
                    chunk_config={"enabled": True, "params": {"chunk_size": 1000, "chunk_overlap": 200}},
                    embeddings_model_name="text-embedding-3-small",
                    chat_model="gpt-4.1-mini",
                    collection_name="Rag Loaded Data Improved"
                )

            self.retrievers_config = {
                "base": {
                    "vectorstore": self.dbs_manager.get_base_vectorstore()
                },
                "parent_document": {
                    "vectorstore": self.dbs_manager.get_parent_document_vectorstore(),
                    "in_memory_store": self.dbs_manager.get_in_memory_store(),
                    "child_splitter": self.dbs_manager.get_child_splitter()
                }
            }

            # set up retrievers
            self.retrival_chains, self.retrival_wrappers = get_retrieval_chains_and_wrappers(
                self.retrievers_config, 
                self.loaded_rag_data,
                self.rag_model,
                self.MODE)

        except Exception as e:
            print(f"Error in initialization: {str(e)}") 
            raise HTTPException(status_code=500, detail=f"Failed to initialize Agent and dependencies: {str(e)}")

    def _search_node(self, state: AgentState):
        """Search the web for the latest information related to query"""

        query = state.get("query", "")
        search_result = tavily_tool.invoke(query)

        updated_context = state.get("context", {}).copy()
        updated_context.setdefault("search", []).append(search_result)

        last_message = state.get("current_messages", [])[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            raise HTTPException(status_code=500, detail="Tool was expected to be called, but no tool_calls found.")

        tool_messages = [
            ToolMessage(
                tool_call_id=call["id"],
                content=search_result
            )
            for call in tool_calls
        ]

        return {
            **state,
            "current_messages": tool_messages,
            "context": updated_context
        }

    def _rag_node(self, state: AgentState):
        """Custom RAG-based search for relevant info."""

        query = state.get("query", "")

        rag_result = custom_rag_tool.invoke(
            {"input" : {
                "query": query,
                "retriever": self.retrival_wrappers[self.retriever_mode.value]
            }}
        )

        updated_context = state.get("context", {}).copy()
        updated_context.setdefault("rag", []).append(rag_result)

        last_message = state.get("current_messages", [])[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            raise HTTPException(status_code=500, detail="Expected tool_calls not found.")

        tool_messages = [
            ToolMessage(
                tool_call_id=call["id"],
                content=rag_result
            )
            for call in tool_calls
        ]

        return {
            **state,
            "current_messages": tool_messages,
            "context": updated_context
        }
    
    async def chat(self, user_message: str):
        """Chat loop entrypoint"""
        try:

            sys_msg = SystemMessage(content=SYSTEM_PROMPT)
            user_msg =  HumanMessage(content=user_message)

            inputs: AgentState = {
                "query": user_message,
                "current_messages": [sys_msg, user_msg],
                "agent_memory": self.agent_memory,
                "context": {},
                "response": ""
            }

            final_response = ""
            tool_calls = []
            final_current_messages = []
            final_context = {}

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
                        if "context" in values:
                            final_context = values["context"]

            # Append ReAct interaction to long-term agent memory
            self.agent_memory.extend(final_current_messages)

            return {
                "response": final_response or "I apologize, but I couldn't generate a response.",
                "messages": final_current_messages,
                "tool_calls": tool_calls,
                "context": final_context,
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


    def reset_longer_term_memory(self):
        self.agent_memory = []

#Agent = LangGraphAgent(retriever_mode=RetrievalEnums.NAIVE, MODE="CERT", langchain_project_name="AIM-CERT-LANGGRAPH-NAIVE")