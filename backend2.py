from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent  # ← Use this instead
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import aiosqlite
import asyncio
import os

load_dotenv()

# -------------------
# 1. LLM
# ------------------- 
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# -------------------
# 2. MCP Client & Tools
# ------------------- 
CLICKUP_API_KEY = os.getenv("CLICKUP_API_KEY")
CLICKUP_TEAM_ID = os.getenv("CLICKUP_TEAM_ID")

client = MultiServerMCPClient({
    "clickup": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "mcp-remote", "https://mcp.clickup.com/mcp"]
    }
})

async def load_mcp_tools() -> list[BaseTool]:
    try:
        print("🔄 Attempting to load MCP tools...")
        tools = await client.get_tools()
        print(f"✅ Successfully loaded {len(tools)} MCP tools")
        for tool in tools:
            print(f"  - {tool.name}")
        return tools
    except Exception as e:
        print(f"❌ Failed to load MCP tools: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

# -------------------
# 3. Checkpointer
# -------------------
async def init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

# -------------------
# 4. Agent (using create_react_agent)
# -------------------
async def create_chatbot():
    mcp_tools = await load_mcp_tools()
    checkpointer = await init_checkpointer()
    
    # Use create_react_agent which properly handles async tools
    chatbot = create_react_agent(
        llm, 
        mcp_tools, 
        checkpointer=checkpointer
    )
    
    return chatbot

# -------------------
# 5. Chat function
# -------------------
async def chat(user_message: str, thread_id: str = "default"):
    chatbot = await create_chatbot()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n🧑 User: {user_message}\n")
    
    async for event in chatbot.astream(
        {"messages": [("user", user_message)]},
        config=config,
        stream_mode="values"
    ):
        # Get the last message
        last_message = event["messages"][-1]
        
        # Print tool calls if any
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                print(f"🔧 Calling tool: {tool_call['name']}")
                print(f"   Args: {tool_call['args']}\n")
        
        # Print assistant response
        if last_message.type == "ai" and last_message.content:
            print(f"🤖 Assistant: {last_message.content}\n")

# -------------------
# 6. Helper functions
# -------------------
async def retrieve_all_threads():
    checkpointer = await init_checkpointer()
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

# -------------------
# 7. Main execution
# -------------------
async def main():
    # Test the chatbot
    await chat("can u check my task? what is the update?")
    
    # You can continue the conversation
    # await chat("what about my other tasks?", thread_id="default")

if __name__ == "__main__":
    asyncio.run(main())