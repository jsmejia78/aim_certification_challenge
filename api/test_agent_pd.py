from langgraph_agent import LangGraphAgent
import asyncio


# Entry point
async def main():
    agent = LangGraphAgent()
    user_message = "what are good kid tamtrum managing techniques in 2025?"

    reply = await agent.chat(user_message)

    print(f"\n\nQuery: {user_message}")
    print(f"\n\nResponce: {reply['response']}")
    print(f"\n\nNum tool calls: {reply['metadata']['total_tool_calls']}\n\n")

    user_message = "any difference with techniques from the previous year"

    reply = await agent.chat(user_message)

    print(f"\n\nQuery: {user_message}")
    print(f"\n\nResponce: {reply['response']}")
    print(f"\n\nNum tool calls: {reply['metadata']['total_tool_calls']}\n\n")

if __name__ == "__main__":
    asyncio.run(main())