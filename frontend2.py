import streamlit as st
import asyncio
from datetime import datetime
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent  # Updated import
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import aiosqlite
import os
import atexit

load_dotenv()

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="ClickUp AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Custom CSS
# -------------------
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .tool-call {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        font-family: monospace;
    }
    .thinking {
        color: #666;
        font-style: italic;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------
# Global MCP Client Manager
# -------------------
class MCPClientManager:
    _instance = None
    _client = None
    _tools = None
    _lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def get_client(self):
        async with self._lock:
            if self._client is None:
                self._client = MultiServerMCPClient({
                    "clickup": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "mcp-remote", "https://mcp.clickup.com/mcp"]
                    }
                })
            return self._client
    
    async def get_tools(self):
        async with self._lock:
            if self._tools is None:
                client = await self.get_client()
                try:
                    self._tools = await client.get_tools()
                except Exception as e:
                    st.error(f"❌ Failed to load MCP tools: {e}")
                    self._tools = []
            return self._tools
    
    async def cleanup(self):
        async with self._lock:
            if self._client is not None:
                try:
                    await self._client.close()
                except:
                    pass
                self._client = None
                self._tools = None

# -------------------
# LLM Setup
# -------------------
@st.cache_resource
def get_llm():
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# -------------------
# Checkpointer with caching
# -------------------
_checkpointer_cache = {}

async def get_checkpointer():
    """Initialize SQLite checkpointer for conversation history"""
    if "checkpointer" not in _checkpointer_cache:
        conn = await aiosqlite.connect(database="chatbot.db")
        _checkpointer_cache["checkpointer"] = AsyncSqliteSaver(conn)
    return _checkpointer_cache["checkpointer"]

# -------------------
# Agent Creation with proper async handling
# -------------------
async def create_chatbot():
    """Create the agent with MCP tools"""
    llm = get_llm()
    
    # Get tools from the singleton manager
    manager = MCPClientManager.get_instance()
    mcp_tools = await manager.get_tools()
    
    checkpointer = await get_checkpointer()
    
    # Create agent with tools
    from langgraph.prebuilt import create_react_agent
    chatbot = create_react_agent(
        llm, 
        mcp_tools, 
        checkpointer=checkpointer
    )
    
    return chatbot, len(mcp_tools)

# -------------------
# Chat Logic
# -------------------
async def process_message(user_message: str, thread_id: str):
    """Process a user message and stream the response"""
    chatbot, num_tools = await create_chatbot()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    full_response = ""
    tool_calls_made = []
    
    # Create placeholders for streaming
    response_placeholder = st.empty()
    tool_placeholder = st.empty()
    
    try:
        async for event in chatbot.astream(
            {"messages": [("user", user_message)]},
            config=config,
            stream_mode="values"
        ):
            last_message = event["messages"][-1]
            
            # Track tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_info = {
                        "name": tool_call.get('name', 'unknown'),
                        "args": tool_call.get('args', {})
                    }
                    if tool_info not in tool_calls_made:
                        tool_calls_made.append(tool_info)
            
            # Update tool calls display
            if tool_calls_made:
                with tool_placeholder:
                    with st.expander("🔧 Tool Calls", expanded=True):
                        for tc in tool_calls_made:
                            st.markdown(f"**{tc['name']}**")
                            if tc['args']:
                                st.json(tc['args'])
            
            # Stream assistant response
            if last_message.type == "ai" and last_message.content:
                full_response = last_message.content
                response_placeholder.markdown(full_response)
        
        return full_response, tool_calls_made
        
    except Exception as e:
        st.error(f"❌ Error processing message: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, []

# -------------------
# Retrieve Threads
# -------------------
async def get_all_threads():
    """Get all conversation threads from the database"""
    try:
        checkpointer = await get_checkpointer()
        all_threads = set()
        async for checkpoint in checkpointer.alist(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
        return sorted(list(all_threads))
    except Exception as e:
        st.error(f"Error retrieving threads: {e}")
        return []

# -------------------
# Initialize tools on startup
# -------------------
async def init_tools():
    """Initialize MCP tools"""
    manager = MCPClientManager.get_instance()
    tools = await manager.get_tools()
    return len(tools)

# -------------------
# Session State Init
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if "tools_loaded" not in st.session_state:
    st.session_state.tools_loaded = False
    st.session_state.num_tools = 0

# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.title("🤖 ClickUp Assistant")
    st.markdown("---")
    
    # Thread management
    st.subheader("💬 Conversations")
    
    # New conversation button
    if st.button("➕ New Conversation"):
        st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.messages = []
        st.rerun()
    
    # Display current thread ID
    st.info(f"**Current Thread:**\n`{st.session_state.thread_id[-16:]}`")
    
    # Load existing threads button
    if st.button("🔄 Load Threads"):
        with st.spinner("Loading threads..."):
            threads = asyncio.run(get_all_threads())
            st.session_state.available_threads = threads
    
    # Thread selector
    if "available_threads" in st.session_state and st.session_state.available_threads:
        selected_thread = st.selectbox(
            "Previous Threads",
            options=st.session_state.available_threads,
            index=st.session_state.available_threads.index(st.session_state.thread_id) 
                if st.session_state.thread_id in st.session_state.available_threads else 0
        )
        
        if selected_thread != st.session_state.thread_id:
            if st.button("📂 Load Selected"):
                st.session_state.thread_id = selected_thread
                st.session_state.messages = []
                st.rerun()
    
    st.markdown("---")
    
    # Tools status
    st.subheader("🔧 Tools Status")
    if not st.session_state.tools_loaded:
        with st.spinner("Loading MCP tools..."):
            try:
                num_tools = asyncio.run(init_tools())
                st.session_state.num_tools = num_tools
                st.session_state.tools_loaded = True
                st.success(f"✅ {num_tools} tools loaded")
            except Exception as e:
                st.error(f"❌ Failed to load tools")
                st.exception(e)
    else:
        st.success(f"✅ {st.session_state.num_tools} tools ready")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Current Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with LangGraph + MCP")

# -------------------
# Main Chat Interface
# -------------------
st.title("💬 ClickUp AI Assistant")
st.markdown("Ask me anything about your ClickUp tasks, projects, and workspace!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show tool calls if present
        if "tool_calls" in message and message["tool_calls"]:
            with st.expander("🔧 Tool Calls Used", expanded=False):
                for tc in message["tool_calls"]:
                    st.markdown(f"**{tc['name']}**")
                    if tc.get('args'):
                        st.json(tc['args'])

# Chat input
if prompt := st.chat_input("Ask about your ClickUp tasks..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response, tool_calls = asyncio.run(
                process_message(prompt, st.session_state.thread_id)
            )
        
        if response:
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "tool_calls": tool_calls
            })
            st.rerun()

# -------------------
# Welcome Message
# -------------------
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! Ask me anything about your ClickUp workspace. Try:\n\n"
            "- 'What tasks do I have?'\n"
            "- 'Show me my urgent tasks'\n"
            "- 'What's the status of my latest project?'\n"
            "- 'Can you check my task? What is the update?'")
 