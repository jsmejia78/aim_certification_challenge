from langgraph_agent import LangGraphAgent
import asyncio


# Entry point
async def main():
    agent = LangGraphAgent()
    user_message = "what was the weather in Tokyo on 7/30/2025?"

    reply = await agent.chat(user_message)

    print(f"\n\n\nResponce: {reply['response']}\n\n\n")
    print(reply["metadata"]["total_tool_calls"])
if __name__ == "__main__":
    asyncio.run(main())