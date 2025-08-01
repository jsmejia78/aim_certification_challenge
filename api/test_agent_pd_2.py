from langgraph_agent import LangGraphAgent
import asyncio


# Entry point
async def main():
    agent = LangGraphAgent()
    user_message = "My 7-year-old insists he’s working with an invisible time-traveling robot named Professor Blip, who gives him nightly missions. He refuses to do homework because it ‘interferes with the timeline’ and wants us to register Blip as his legal guardian. It’s affecting bedtime and school. How do we respect his imagination without letting it disrupt daily life?"

    reply = await agent.chat(user_message)

    print(f"\n\nQuery: {user_message}")
    print(f"\n\nResponce: {reply['response']}")
    print(f"\n\nNum tool calls: {reply['metadata']['total_tool_calls']}\n\n")

    user_message = "Should I take him to a psycologist, this year (2025)?"

    reply = await agent.chat(user_message)

    print(f"\n\nQuery: {user_message}")
    print(f"\n\nResponce: {reply['response']}")
    print(f"\n\nNum tool calls: {reply['metadata']['total_tool_calls']}\n\n")

if __name__ == "__main__":
    asyncio.run(main())