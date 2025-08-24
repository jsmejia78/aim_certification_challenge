import os
import json
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

from prompts import SYSTEM_PROMPT, FAMILY_SYSTEM_PROMPT_TEMPLATE, router_prompt_template
from tools import get_tools, create_retrieval_tool
from retrievers import get_retrieval_chains_and_wrappers
from vector_stores import VectorStoresManager
from data_loader import DataLoader

from guardrails.hub import (
    RestrictToTopic,
    DetectJailbreak, 
    ProfanityFree,
    GuardrailsPII
)
from guardrails import Guard

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
    last_router_response: str

# ----------------------------------------
# Main Agent Class
# ----------------------------------------

class LangGraphAgent:
    def __init__(self, retriever_mode: RetrievalEnums, MODE: str, langchain_project_name: str):
        # Validate required parameters
        if not retriever_mode or not MODE or not langchain_project_name:
            raise ValueError("All parameters are required")
        
        # Set required attributes
        self.retriever_mode = retriever_mode
        self.MODE = MODE
        self.langchain_project_name = langchain_project_name
        
        # Initialize mood variable
        self.mood = None
        
        # Initialize feedback variable
        self.feedback = None
        
        # Setup environment and initialize immediately
        self._setup_environment()
        self._initialize_components()

    def set_mood(self, mood: str):
        """Set the current mood of the user"""
        self.mood = mood
        
    def get_mood(self) -> str:
        """Get the current mood of the user"""
        return self.mood

    def set_feedback(self, feedback: str):
        """Set the feedback from the user about the session"""
        self.feedback = feedback
        print(f"Agent feedback set to: {feedback}")
        
    def get_feedback(self) -> str:
        """Get the feedback from the user about the session"""
        return self.feedback

    def generate_family_system_prompt(self, family_data_path: str = "family_data.json") -> str:
        """
        Generate a personalized system prompt based on family data and parent mood.
        
        Args:
            family_data_path (str): Path to the family_data.json file
            
        Returns:
            str: Personalized system prompt string
        """
        try:
            # Load family data
            with open(family_data_path, 'r') as f:
                family_data = json.load(f)
            
            # Extract family information
            mother_name = family_data.get("mother_name", "the mother")
            father_name = family_data.get("father_name", "the father")
            number_of_kids = family_data.get("number_of_kids", 0)
            children = family_data.get("children", [])
            
            # Build children information string
            children_info = ""
            if children:
                children_details = []
                for child in children:
                    name = child.get("name", "unnamed child")
                    age = child.get("age", 0)
                    strengths = child.get("strengths", [])
                    growth_areas = child.get("growth_areas", [])
                    
                    child_detail = f"{name} (age {age})"
                    if strengths:
                        child_detail += f" with strengths in {', '.join(strengths)}"
                    if growth_areas:
                        child_detail += f" and areas for growth in {', '.join(growth_areas)}"
                    
                    children_details.append(child_detail)
                
                children_info = f"Children: {', '.join(children_details)}"
            else:
                children_info = f"Number of children: {number_of_kids}"
            
            # Get current mood
            current_mood = self.get_mood() or "neutral"
            
            # Generate personalized system prompt using template
            system_prompt = FAMILY_SYSTEM_PROMPT_TEMPLATE.format(
                mother_name=mother_name,
                father_name=father_name,
                children_info=children_info,
                current_mood=current_mood
            )
            
            return system_prompt
            
        except FileNotFoundError:
            # Fallback to default system prompt if family data file not found
            return SYSTEM_PROMPT
        except Exception as e:
            print(f"Error generating family system prompt: {str(e)}")
            # Return default system prompt on any error
            return SYSTEM_PROMPT

    def _setup_environment(self):
        """Setup environment variables and validate configuration"""
        load_dotenv()
        
        # Validate environment variables
        required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "COHERE_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {missing_keys}")
        
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project_name

    def _initialize_components(self):
        """Initialize all components in dependency order"""
        try:
            self._load_data()
            self._setup_vector_stores()
            self._setup_retrievers()
            self._setup_tools()
            self._setup_guards()
            self._setup_memory()
            self._setup_model()
            self._setup_graph()
        except Exception as e:
            print(f"Error in initialization: {str(e)}") 
            raise HTTPException(status_code=500, detail=f"Failed to initialize Agent and dependencies: {str(e)}")

    def _load_data(self):
        """Load RAG data and setup retriever model"""
        data_loader = DataLoader("pd_blogs_filtered")
        self.loaded_rag_data = data_loader.load_data()
        self.retriver_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    def _setup_vector_stores(self):
        """Setup vector stores manager"""
        self.dbs_manager = VectorStoresManager(    
            MODE="baseline",
            loaded_data=self.loaded_rag_data,
            chunk_config={"enabled": True, "params": {"chunk_size": 1000, "chunk_overlap": 200}},
            embeddings_model_name="text-embedding-3-small",
            chat_model="gpt-4.1-mini",
            collection_name="Rag Loaded Data Improved"
        )

    def _setup_retrievers(self):
        """Setup retrieval chains and wrappers"""
        # Set up retrievers config
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

        # Set up retrievers
        self.retrival_chains, self.retrival_wrappers = get_retrieval_chains_and_wrappers(
            self.retrievers_config, 
            self.loaded_rag_data,
            self.retriver_model,
            self.MODE
        )

        # Set up retrieval chain
        self.retrival_chain = self.retrival_chains[self.retriever_mode.value]
        self.retriever_wrapper = create_retrieval_tool(self.retrival_chain)

    def _setup_tools(self):
        """Setup tool belt"""
        self.tool_belt = get_tools(self.retrival_chain)

    def _setup_memory(self):
        """Setup memory manager"""
        self.memory = MemorySaver()

    def _setup_model(self):
        """Setup the language model with tools"""
        self.react_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7).bind_tools(self.tool_belt)
        self.router_and_bridge_model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

    def _setup_guards(self):
        """Setup guards"""

        # Topic Restriction Guard - Keep conversations focused on family
        self.topic_guard = Guard().use(
            RestrictToTopic(
                valid_topics=[
                    "parenting", "family", "children", "education", "life", "positive discipline",
                    "greeting", "hello", "hi", "how are you", "conversation", "chat", "help",
                    "support", "advice", "guidance", "behavior", "emotions", "feelings",
                    "communication", "relationships", "development", "growth", "learning", "plans", "follow ups", "goals", "goals and plans"
                ],
                invalid_topics=["investment advice", "crypto", "gambling", "politics", "self-harm", "suicide", "medical advice", "mental health advice"],
                disable_classifier=True,
                disable_llm=False,
                on_fail="filter"
            )
        )

        self.jailbreak_guard = Guard().use(DetectJailbreak())
        self.profanity_guard = Guard().use(
            ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="filter")
        )
        self.pii_guard = Guard().use(
        GuardrailsPII(
            entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"], 
            on_fail="fix"
            )
        )

        self.guard_list = [self.topic_guard, self.jailbreak_guard]

    # ----------------------------------------
    # Node Definitions
    # ----------------------------------------

    def _prefetch_node(self, state: AgentState) -> AgentState:
        """Prefetch data from the retrieval chain"""
        if not self.retrival_chain:
            raise HTTPException(status_code=500, detail="Retrieval chain not initialized")
            
        result = self.retrival_chain.invoke({"question": state["query"]})
        
        # Convert result to string format for AIMessage
        if hasattr(result, 'content'):
            content = result.content
        elif isinstance(result, list):
            # If it's a list of documents, extract their content
            content = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in result])
        else:
            content = str(result)
    
        return {
            "messages": [AIMessage(content=content)]
        }

    def _call_model(self, state: AgentState):
        """Generate reasoning output using context + prior messages"""
        if not self.react_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
            
        if "messages" not in state or not state["messages"]:
            raise HTTPException(status_code=500, detail="No messages in state")
            
        response = self.react_model.invoke(state["messages"])
        return {
            "messages": [response],
            "response": response.content
        }

    def _should_continue(self, state: AgentState):
        """Route to tools if the last message has tool calls."""
        try:    
            if "messages" not in state or not state["messages"]:
                return END
                
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "action"
            return "bridge_chat"
        except Exception as e:
            print(f"Error in _should_continue: {str(e)}")
            return END

    def _router_node(self, state: AgentState):
        """Router node to determine the next node to execute"""
        if state.get("query") is None:
            raise HTTPException(status_code=400, detail="Query not found in state")
        
        formatted_prompt = router_prompt_template.format(query=state["query"])
        sys_msg = SystemMessage(content=formatted_prompt)
        
        output = self.router_and_bridge_model.invoke([sys_msg])
        
        return {
            "last_router_response": output.content
        }

    def _bridge_chat_node(self, state: AgentState):
        """Bridge chat node to bridge the chat"""
        if "CLARIFY" in state["last_router_response"]:
            # Extract the clarifying question and clean it up
            clarifying_question = state["last_router_response"].split("::")[1]
            
            # Remove angle brackets if they exist
            if clarifying_question.startswith("<") and clarifying_question.endswith(">"):
                clarifying_question = clarifying_question[1:-1]
            
            # Capitalize first letter
            clarifying_question = clarifying_question.strip().capitalize()
            
            return {
                "messages": [AIMessage(content=clarifying_question)],
                "response": clarifying_question
            }
        
        # For non-clarifying cases, return minimal state update
        return {
            "messages": []  # Empty messages list to maintain state consistency
        }


    def _router_next(self, state: AgentState):
        """Route to tools if the last message has tool calls."""
        try:    
            if "CONTEXT" in state["last_router_response"]:
                return "prefetch"
            elif "CONTINUE" in state["last_router_response"]:
                return "agent"
            elif "CLARIFY" in state["last_router_response"]:
                return "bridge_chat"
            else:
                return END
        except Exception as e:
            print(f"Error in _should_continue: {str(e)}")
            return END

    # ----------------------------------------
    # Graph Setup
    # ----------------------------------------

    def _setup_graph(self):
        """Setup and compile the LangGraph workflow"""
        graph = StateGraph(AgentState, name="companion-agent-graph")
        
        # Set up tool node
        self.tool_node = ToolNode(self.tool_belt)
        
        # Set up nodes and edges
        graph.add_node("router", self._router_node)
        graph.add_node("prefetch", self._prefetch_node)
        graph.add_node("agent", self._call_model)
        graph.add_node("action", self.tool_node)
        graph.add_node("bridge_chat", self._bridge_chat_node)
        
        # Fix: Start with router, not prefetch
        graph.set_entry_point("router")
        graph.add_conditional_edges("router", self._router_next, {
            "prefetch": "prefetch", 
            "agent": "agent", 
            "bridge_chat": "bridge_chat"
        })
        graph.add_edge("prefetch", "agent")
        graph.add_conditional_edges("agent", self._should_continue, {
            "action": "action", 
            "bridge_chat": "bridge_chat",
            END:END
        })
        graph.add_edge("action", "agent")
        graph.add_edge("bridge_chat", END)
        

        self.agent_graph = graph.compile(checkpointer=self.memory)

    # ----------------------------------------
    # Chat Loop
    # ----------------------------------------
    async def chat(self, user_message: str, config_thread: dict):
        """Chat loop entrypoint with streaming support"""

        try:
            # ----------------------------------------
            # Input validation guards
            # ----------------------------------------
            
            # Check if user_message is provided and valid
            if not user_message or not isinstance(user_message, str):
                return {
                    "type": "error",
                    "content": "Please provide a valid message to continue our conversation.",
                    "timestamp": str(datetime.now())
                }
            
            # Check if user_message is not empty or just whitespace
            if not user_message.strip():
                return {
                    "type": "error",
                    "content": "Your message cannot be empty. Please type something to continue.",
                    "timestamp": str(datetime.now())
                }
            
            # Check if user_message is not too long (reasonable limit)
            if len(user_message) > 5000:
                return {
                    "type": "error",
                    "content": "Your message is too long. Please keep it under 5000 characters.",
                    "timestamp": str(datetime.now())
                }
            
            # Check if config_thread is provided and valid
            if not config_thread or not isinstance(config_thread, dict):
                return {
                    "type": "error",
                    "content": "Configuration error. Please try refreshing the page.",
                    "timestamp": str(datetime.now())
                }

            # ----------------------------------------
            # Check content guards
            # ----------------------------------------

            # Track guard validation results
            guard_errors = []
            
            for guard in self.guard_list:
                try:
                    guard_response = guard.validate(user_message)
                    
                    # Debug: Print what the guard actually returned
                    print(f"Guard {guard.__class__.__name__} response: {guard_response}")
                    print(f"Response type: {type(guard_response)}")
                    
                    # Handle the actual guard response structure
                    # Guards return ValidationOutcome objects with validation_passed attribute
                    if hasattr(guard_response, 'validation_passed'):
                        # This is a ValidationOutcome object
                        passed = guard_response.validation_passed
                        print(f"  ValidationOutcome - validation_passed: {passed}")
                        
                        # Get detailed error information from validation_summaries
                        if not passed and hasattr(guard_response, 'validation_summaries'):
                            for summary in guard_response.validation_summaries:
                                if hasattr(summary, 'failure_reason'):
                                    error_msg = summary.failure_reason
                                    print(f"  Failure reason: {error_msg}")
                                    guard_errors.append(f"Guard validation failed: {error_msg}")
                                    break
                            else:
                                # No specific failure reason found
                                guard_errors.append("Guard validation failed: Please try rephrasing your question.")
                        elif passed:
                            print(f"  ✓ Guard {guard.__class__.__name__} passed validation")
                        else:
                            guard_errors.append("Guard validation failed: Please try rephrasing your question.")
                            
                    elif hasattr(guard_response, 'get'):
                        # Dictionary-like access (fallback)
                        success = guard_response.get('success', False)
                        status = guard_response.get('status', 'unknown')
                        action = guard_response.get('action', 'unknown')
                        print(f"  Dict access - success: {success}, status: {status}, action: {action}")
                        
                        passed = (
                            (success and status == 'success') or 
                            (action == 'passed') or
                            (status == 'passed')
                        )
                        
                        if not passed:
                            error_msg = guard_response.get('error') or guard_response.get('reasons', [None])[0]
                            if error_msg:
                                guard_errors.append(f"Guard validation failed: {error_msg}")
                            else:
                                guard_errors.append("Guard validation failed: Please try rephrasing your question.")
                        else:
                            print(f"  ✓ Guard {guard.__class__.__name__} passed validation")
                    else:
                        # Fallback to attribute access
                        success = getattr(guard_response, 'success', False)
                        status = getattr(guard_response, 'status', 'unknown')
                        action = getattr(guard_response, 'action', 'unknown')
                        print(f"  Attr access - success: {success}, status: {status}, action: {action}")
                        
                        passed = (
                            (success and status == 'success') or 
                            (action == 'passed') or
                            (status == 'passed')
                        )
                        
                        if not passed:
                            error_msg = getattr(guard_response, 'error', None) or getattr(guard_response, 'reasons', [None])[0]
                            if error_msg:
                                guard_errors.append(f"Guard validation failed: {error_msg}")
                            else:
                                guard_errors.append("Guard validation failed: Please try rephrasing your question.")
                        else:
                            print(f"  ✓ Guard {guard.__class__.__name__} passed validation")
                    
                    print(f"  Final guard passed: {passed}")
                        
                except Exception as guard_error:
                    print(f"Guard validation error: {str(guard_error)}")
                    guard_errors.append("Guard validation error: Please try again.")
            
            # If any guards failed, we'll handle it in the stream_response function
            if guard_errors:
                print(f"Guard validation errors: {guard_errors}")
                
                # Create a copy of guard errors to ensure they're preserved
                final_guard_errors = guard_errors.copy()
                print(f"Final guard errors (copy): {final_guard_errors}")
            else:
                # Initialize empty list if no guard errors
                final_guard_errors = []
                print(f"No guard errors, final_guard_errors initialized as empty: {final_guard_errors}")

            # ----------------------------------------
            # Chat actions and stream response
            # ----------------------------------------

            # Track component initialization errors
            component_errors = []
            
            # Guard: Check if required components are initialized
            if not hasattr(self, 'react_model') or not self.react_model:
                component_errors.append("Language model is not available. Please try again later.")
            
            if not hasattr(self, 'retrival_chain') or not self.retrival_chain:
                component_errors.append("Knowledge base is not available. Please try again later.")
            
            # Guard: Check if memory system is initialized
            if not hasattr(self, 'memory') or not self.memory:
                component_errors.append("Memory system is not available. Please try again later.")

            # Create user message
            try:
                sys_msg = SystemMessage(content=self.generate_family_system_prompt())
                user_msg = HumanMessage(content=user_message)
            except Exception as e:
                print(f"Error creating system message: {str(e)}")
                component_errors.append("Failed to initialize conversation. Please try again.")

            # Always include the user message in inputs
            # LangGraph will handle conversation state through checkpointing
            inputs: AgentState = {
                "query": user_message,
                "messages": [sys_msg, user_msg],
                "response": "",
                "last_router_response": None
            }

            # Return a generator for streaming
            async def stream_response():
                final_response = ""
                tool_calls = []
                final_messages = []
                last_router_response = ""

                try:
                    # Debug: Check what guard_errors contains
                    print(f"🔍 stream_response: guard_errors = {guard_errors}")
                    print(f"🔍 stream_response: len(guard_errors) = {len(guard_errors)}")
                    print(f"🔍 stream_response: guard_errors type = {type(guard_errors)}")
                    
                    # Check if guards failed validation
                    if final_guard_errors:
                        error_response = {
                            "type": "error",
                            "content": f"I'm sorry, but I can't help with that request. {'; '.join(final_guard_errors)}",
                            "timestamp": str(datetime.now())
                        }
                        print(f"🚨 Yielding error response: {error_response}")
                        yield error_response
                        return
                    
                    # Check if components failed initialization
                    if component_errors:
                        yield {
                            "type": "error",
                            "content": f"System error: {'; '.join(component_errors)}",
                            "timestamp": str(datetime.now())
                        }
                        return
                    
                    print(f"Starting stream_response with inputs: {inputs}")
                    print(f"Config thread: {config_thread}")
                    
                    # Guard: Check if agent_graph exists
                    if not self.agent_graph:
                        yield {
                            "type": "error",
                            "content": "Agent system is not properly initialized. Please try again later.",
                            "timestamp": str(datetime.now())
                        }
                        return
                    
                    print("Agent graph exists, starting astream...")
                    async for chunk in self.agent_graph.astream(inputs, stream_mode="updates", config=config_thread):
                            for node, values in chunk.items():
                                # Only stream content from the "agent" node (LLM responses)
                                # Don't stream prefetch content or tool results
                                if node == "agent" and "messages" in values:
                                    for msg in values["messages"]:
                                        final_messages.append(msg)
                                        # Extract tool calls if they exist in AssistantMessage
                                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                                            tool_calls.extend(msg.tool_calls)
                                        
                                        # Stream the LLM message content only
                                        if hasattr(msg, 'content') and msg.content and msg.__class__.__name__ == "AIMessage":
                                            yield {
                                                "type": "message",
                                                "content": msg.content,
                                                "role": "assistant",
                                                "timestamp": str(datetime.now()),
                                                "metadata": {
                                                    "router_response": last_router_response
                                                }
                                            }
                                
                                # Stream the final response from agent node or bridge_chat node
                                if (node == "agent" or node == "bridge_chat") and "response" in values:
                                    final_response = values["response"]
                                    # Stream the response
                                    if final_response:
                                        yield {
                                            "type": "response",
                                            "content": final_response,
                                            "timestamp": str(datetime.now()),
                                            "metadata": {
                                                "router_response": last_router_response
                                            }
                                        }
                                
                                # Collect all messages from all nodes for final summary (but don't stream them)
                                if "messages" in values:
                                    for msg in values["messages"]:
                                        if msg not in final_messages:
                                            final_messages.append(msg)
                                        # Extract tool calls if they exist
                                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                                            tool_calls.extend(msg.tool_calls)
                                
                                if node == "router" and "last_router_response" in values:
                                    last_router_response = values["last_router_response"]
                                    yield {
                                        "type": "router_response",
                                        "content": last_router_response,
                                        "timestamp": str(datetime.now())
                                    }
                                    
                                # Stream tool call information only when tools are executed
                                if node == "action" and tool_calls:
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
                except Exception as e:
                    print(f"Error in stream_response: {str(e)}")
                    # Return error response with more user-friendly message
                    error_content = "I encountered an issue while processing your request. "
                    
                    # Provide specific error messages for common issues
                    if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                        error_content += "The service is currently busy. Please try again in a moment."
                    elif "timeout" in str(e).lower():
                        error_content += "The request took too long to process. Please try again."
                    elif "network" in str(e).lower() or "connection" in str(e).lower():
                        error_content += "There was a network issue. Please check your connection and try again."
                    else:
                        error_content += "Please try again or rephrase your question."
                    
                    yield {
                        "type": "error",
                        "content": error_content,
                        "timestamp": str(datetime.now())
                    }

            return stream_response()

        except Exception as e:
            print(f"Critical error in chat function: {str(e)}")
            # Return error response instead of raising HTTPException for better user experience
            return {
                "type": "error",
                "content": "I'm experiencing technical difficulties. Please try again later or contact support if the issue persists.",
                "timestamp": str(datetime.now())
            }

    def reset_longer_term_memory(self):
        """Reset the agent's memory"""
        print(f"Resetting agent memory. Previous feedback: {self.feedback}")
        self.mood = None
        self.feedback = None
        if self.memory:
            # Clear the memory storage directly - keep using same instance
            self.memory.storage.clear()
