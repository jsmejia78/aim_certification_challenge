import os
from typing import List, Dict, Any
from fastapi import HTTPException
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from enum import Enum

from langgraph.prebuilt import ToolNode

from prompts import SYSTEM_PROMPT
from tools import get_tools, create_retrieval_tool
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
    messages: Annotated[List[BaseMessage], add_messages]
    response: str

# ----------------------------------------
# Main Agent Class
# ----------------------------------------

class LangGraphAgent():
    def __init__(self, retriever_mode: RetrievalEnums, MODE: str, langchain_project_name: str):

        self.agent_graph = None
        self.react_model = None
        self.tool_belt = None
        self.retrievers_config = None
        self.retriever_mode = retriever_mode
        self.retrieval_llm = None
        self.rag_prompt = None
        self.retriver_model = None
        self.retrival_chains = None
        self.retrival_wrappers = None
        self.MODE = MODE
        self.loaded_rag_data = None
        self.dbs_manager = None
        self.memory = None
        self.retrival_chain = None
        self.retriever_wrapper = None

        # Automatically loads variables from .env file into os.environ
        load_dotenv()

        assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
        assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
        assert os.getenv("LANGCHAIN_API_KEY"), "Missing LANGCHAIN_API_KEY"
        assert os.getenv("COHERE_API_KEY"), "Missing COHERE_API_KEY"

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = langchain_project_name #f"AIM-CERT-{uuid4().hex[0:8]}"

        self._initialization()

    def _prefetch_node(self, state: AgentState) -> AgentState:
        """Prefetch data from the retrieval chain"""
        result = self.retrival_chain.invoke({"question": state["query"]})
    
        return {
        "messages": [AIMessage(content=result)]
        }

    def _call_model(self, state: AgentState):
        """Generate reasoning output using context + prior messages"""

        if self.react_model:
            response = self.react_model.invoke(state["messages"])
            return {
                "messages": [response],
                "response": response.content
            }
        else:
            raise HTTPException(status_code=500, detail="Model not initialized")


    def _should_continue(self, state: AgentState):
        """Route to tools if the last message has tool calls."""
        try:    
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "action"
            return END
        except Exception as e:
            print(f"Error in _should_continue: {str(e)}")
            return END

    def _initialization(self):
        """Initialize models, graph, and dependencies"""
        try:

            # data loader
            data_loader = DataLoader("pd_blogs_filtered")
            self.loaded_rag_data = data_loader.load_data()

            # set up retriever model (in case it is needed)
            self.retriver_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

            # set up vector stores
            self.dbs_manager = VectorStoresManager(    
                MODE="baseline",
                loaded_data=self.loaded_rag_data,
                chunk_config={"enabled": True, "params": {"chunk_size": 1000, "chunk_overlap": 200}},
                embeddings_model_name="text-embedding-3-small",
                chat_model="gpt-4.1-mini",
                collection_name="Rag Loaded Data Improved"
            )

            # set up retrievers config
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
                self.retriver_model,
                self.MODE
            )

            # set up retrieval chain
            self.retrival_chain = self.retrival_chains[self.retriever_mode.value]
            self.retriever_wrapper = create_retrieval_tool(self.rag_chain)

            # set up tools belt
            self.tool_belt = get_tools()

            # set up memory
            self.memory = MemorySaver()

            # set up model
            self.react_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7).bind_tools(self.tool_belt)

            graph = StateGraph(AgentState, name="companion-agent-graph")

            # set up tool node
            self.tool_node = ToolNode(self.tool_belt)

            # set up nodes and edges
            graph.add_node("prefetch", self._prefetch_node)
            graph.add_node("agent", self._call_model)
            graph.add_node("action", self.tool_node)
            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", self._should_continue,  {"action": "action", END: END})
            graph.add_edge("action", "agent")
            graph.add_edge("prefetch", "agent")

            self.agent_graph = graph.compile(checkpointer=self.memory)

        except Exception as e:
            print(f"Error in initialization: {str(e)}") 
            raise HTTPException(status_code=500, detail=f"Failed to initialize Agent and dependencies: {str(e)}")


    async def chat(self, user_message: str, config_thread: dict):
        """Chat loop entrypoint with streaming support"""
        try:
            force_message = "Use your RAG tool or web search tool to get context to answer my question"
            sys_msg = SystemMessage(content=SYSTEM_PROMPT)
            user_msg = HumanMessage(content=user_message + " " + force_message)

            inputs: AgentState = {
                "query": user_message,
                "messages": [sys_msg, user_msg],
                "response": ""
            }

            # Return a generator for streaming
            async def stream_response():
                final_response = ""
                tool_calls = []
                final_messages = []
                
                if self.agent_graph:
                    async for chunk in self.agent_graph.astream(inputs, stream_mode="updates", config=config_thread):
                        for node, values in chunk.items():
                            # Stream each update as it happens
                            if "messages" in values:
                                for msg in values["messages"]:
                                    final_messages.append(msg)
                                    # Extract tool calls if they exist in AssistantMessage
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        tool_calls.extend(msg.tool_calls)
                                    
                                    # Stream the message content
                                    if hasattr(msg, 'content') and msg.content:
                                        yield {
                                            "type": "message",
                                            "content": msg.content,
                                            "role": msg.__class__.__name__.lower().replace('message', ''),
                                            "timestamp": str(datetime.now())
                                        }
                            
                            if "response" in values:
                                final_response = values["response"]
                                # Stream the response
                                if final_response:
                                    yield {
                                        "type": "response",
                                        "content": final_response,
                                        "timestamp": str(datetime.now())
                                    }
                            
                            # Stream tool call information
                            if "action" in node and tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "content": {
                                        "tool_calls": tool_calls,
                                        "node": node
                                    },
                                    "timestamp": str(datetime.now())
                                }
                
                # Final summary
                yield {
                    "type": "final",
                    "content": {
                        "response": final_response or "I apologize, but I couldn't generate a response.",
                        "messages": len(final_messages),
                        "tool_calls": len(tool_calls),
                        "metadata": {
                            "model": "gpt-4.1-mini",
                            "total_messages": len(final_messages),
                            "total_tool_calls": len(tool_calls),
                            "system_message_used": True
                        }
                    },
                    "timestamp": str(datetime.now())
                }

            return stream_response()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

    def reset_longer_term_memory(self):
        """Reset the agent's memory"""
        if self.memory:
            self.memory.clear()
