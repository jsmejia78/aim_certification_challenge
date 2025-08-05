# Import required FastAPI components for building the API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Import Pydantic for data validation and settings management
from pydantic import BaseModel
# Import OpenAI client for interacting with OpenAI's API
from typing import Optional, Dict
import os
import sys

# Add the current directory to Python path for Vercel compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import langgraph_agent with error handling for deployment
try:
    from langgraph_agent import LangGraphAgent, RetrievalEnums
except ImportError:
    # Fallback for different import paths
    try:
        import langgraph_agent as langgraph_agent
        LangGraphAgent = langgraph_agent.LangGraphAgent
        RetrievalEnums = langgraph_agent.RetrievalEnums
    except ImportError:
        print("Warning: LangGraphAgent could not be imported")


# Initialize FastAPI application with a title
app = FastAPI(title="ParentALLm Agent")

# Initialize LangGraphAgent
Agent = LangGraphAgent(retriever_mode=RetrievalEnums.PARENT_DOCUMENT, 
                       MODE="CHALLENGE", 
                       langchain_project_name= "AIM-CERT-LANGGRAPH-PARENT")

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows the API to be accessed from different domains/origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers in requests
)

# Define the data model for chat requests using Pydantic
# This ensures incoming request data is properly validated
class ChatRequest(BaseModel):
    user_message: str      # Message from the user

# Define the main chat endpoint that handles POST requests
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        
        reply = await Agent.chat(request.user_message)
        return {
            "response": reply["response"],
            "context": reply.get("context", {})
        }
    
    except Exception as e:
        # Handle any errors that occur during processing
        raise HTTPException(status_code=500, detail=str(e))

# Define a health check endpoint to verify API status
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Define an endpoint to clear agent memory
@app.post("/api/clear-memory")
async def clear_memory():
    try:
        Agent.reset_longer_term_memory()
        return {"status": "memory_cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    # Start the server on all network interfaces (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
