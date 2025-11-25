from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition 
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langsmith import traceable  # ← ADDED
import aiosqlite
import requests
import asyncio
import threading
import os

load_dotenv()
 

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# ------------------- 
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")  


# -------------------
# 2. Tools
# ------------------- 
@tool
def get_siam_fav_number()->int:
    """A simple tool that returns Siam's favourite number.""" 
    return 73
 
CLICKUP_API_KEY = os.getenv("CLICKUP_API_KEY")
CLICKUP_TEAM_ID = os.getenv("CLICKUP_TEAM_ID")

client = MultiServerMCPClient(
    { 
    "clickup": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.clickup.com/mcp"]
    }
  }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        print("🔄 Attempting to load MCP tools...")
        tools = run_async(client.get_tools())
        print(f"✅ Successfully loaded {len(tools)} MCP tools")
        for tool in tools:
            print(f"  - {tool.name}")
        return tools
    except Exception as e:
        print(f"❌ Failed to load MCP tools: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []


mcp_tools = load_mcp_tools()
 
# Bind tools to LLM
if mcp_tools:
    print(f"🔧 Binding {len(mcp_tools)} tools to LLM")
    llm_with_tools = llm.bind_tools(mcp_tools)
else:
    print("⚠️  No tools to bind - LLM will run without tools")
    llm_with_tools = llm 
    
# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
@traceable(name="chat_node", run_type="llm")  # ← ADDED: Trace LLM calls
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(mcp_tools) if mcp_tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())