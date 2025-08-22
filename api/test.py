from langgraph_agent import LangGraphAgent, RetrievalEnums
from pydantic import BaseModel

Agent = LangGraphAgent(retriever_mode=RetrievalEnums.PARENT_DOCUMENT, 
                       MODE="DEMO_DAY", 
                       langchain_project_name= "AIM-DEMO-DAY")


config_thread = {"configurable": {"thread_id": 1}}

request = "My kid has contants meltdows at the store, and those can last up to 20 minutes crying non-stop (and usually involves peeing himself from how upset he is), how can i help the situation?"

async def main(request):
    stream_generator = await Agent.chat(request, config_thread)
    async for chunk in stream_generator:
        print(chunk, end='', flush=True)

# Run the async function
import asyncio
asyncio.run(main(request))