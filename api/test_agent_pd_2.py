from langgraph_agent import LangGraphAgent
import asyncio
from dotenv import load_dotenv
import os

# Entry point
async def main():

    load_dotenv()

    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
    assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
    assert os.getenv("LANGCHAIN_API_KEY"), "Missing LANGCHAIN_API_KEY"
    assert os.getenv("COHERE_API_KEY"), "Missing COHERE_API_KEY"

    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    langsmith_project_name = "AIM-CERT-LANGGRAPH-BASE-PIPE"
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project_name
    print(langsmith_project_name)

    MODE = "PIPETEST"

    from langgraph_agent import LangGraphAgent, RetrievalEnums

    Agent = LangGraphAgent(retriever_mode=RetrievalEnums.NAIVE, 
                        MODE=MODE, 
                        langchain_project_name= langsmith_project_name)
                        
    user_message = "My 7-year-old insists he’s working with an invisible time-traveling robot named Professor Blip, who gives him nightly missions. He refuses to do homework because it ‘interferes with the timeline’ and wants us to register Blip as his legal guardian. It’s affecting bedtime and school. How do we respect his imagination without letting it disrupt daily life?"

    reply = await Agent.chat(user_message)

    print(f"\n\nQuery: {user_message}")
    print(f"\n\nResponce: {reply['response']}")
    print(f"\n\nContext: {reply['context']}")
    print(f"\n\nNum tool calls: {reply['metadata']['total_tool_calls']}\n\n")

if __name__ == "__main__":
    asyncio.run(main())