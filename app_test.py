import os
import uuid
import chainlit as cl
import chromadb
import ollama
import pypdf
import docx
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configuration
MODEL_NAME = "llama3.1:8b"
DB_PATH = r"C:\Users\hari7\Documents\Anokha Hackthon\ChromaDB"

# Initialize Client
print(f"Initializing Memory at: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)

chat_collection = client.get_or_create_collection(name="user_chat_facts")
doc_collection = client.get_or_create_collection(name="user_documents")

# ==========================================
# 1. TOOLS
# ==========================================

search_engine = DuckDuckGoSearchRun()

@tool
def web_search_tool(query: str):
    """
    Use this tool for finding jobs, internships, or researching companies/technologies.
    """
    try:
        return search_engine.invoke(query)
    except Exception as e:
        return f"Search Error: {e}"

tools = [web_search_tool]
tool_node = ToolNode(tools)

# ==========================================
# 2. HELPERS (Memory & Parsing)
# ==========================================

def parse_file(file_path):
    text = ""
    try:
        if file_path.endswith('.pdf'):
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages: text += page.extract_text() or ""
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text += "\n".join([p.text for p in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
        return text[:15000]
    except: return ""

def get_combined_memory(text):
    context = ""
    try:
        # Get Chat History
        res = chat_collection.query(query_texts=[text], n_results=3)
        if res['documents'][0]:
            context += "FROM CHAT HISTORY:\n" + "\n".join([f"- {d}" for d in res['documents'][0]]) + "\n\n"
        
        # Get Doc History
        res = doc_collection.query(query_texts=[text], n_results=2)
        if res['documents'][0]:
            context += "FROM DOCUMENTS:\n" + "\n".join([f"- {d}" for d in res['documents'][0]]) + "\n"
    except: pass
    return context

def auto_save_chat_fact(user_text):
    prompt = f"""Extract personal career facts from: "{user_text}". 
    Return ONLY the fact (Name, Skill, Goal). If none, return "NO_FACT"."""
    try:
        res = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        fact = res['message']['content'].strip().replace('"', '')
        if "NO_FACT" not in fact and len(fact) > 5:
            chat_collection.add(documents=[fact], ids=[str(uuid.uuid4())])
            return fact
    except: pass
    return None

def process_document(file_text, name):
    prompt = f"Summarize this document (Resume/Paper) for a Career DB. Content: {file_text}"
    try:
        res = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        summary = res['message']['content'].strip()
        doc_collection.add(documents=[f"Source: {name}\nSummary: {summary}"], ids=[str(uuid.uuid4())])
        return summary
    except: return None

# ==========================================
# 3. ROUTER & AGENT LOGIC
# ==========================================

class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: Literal["general_chat", "web_search", "resume_parser"]

# Bind tools for the agent usage
llm = ChatOllama(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- A. ROUTER NODE ---
def router_node(state: AgentState):
    """
    Decides if we need external tools or just simple chat.
    """
    messages = state['messages']
    last_msg = messages[-1].content.lower()
    
    # Simple Keyword Routing Logic (Fast & Cheap)
    # You can also use an LLM call here for smarter routing if preferred.
    
    if any(k in last_msg for k in ["find", "search", "job", "internship", "opening", "salary", "news"]):
        print("--- ROUTER: Decision -> Web Search ---")
        return {"next_step": "web_search"}
    
    # (Resume parsing is handled via UI events, but logic could go here too)
    
    print("--- ROUTER: Decision -> General Chat ---")
    return {"next_step": "general_chat"}

# --- B. GENERAL CHAT NODE ---
def general_chat_node(state: AgentState):
    messages = state['messages']
    last_msg = messages[-1]
    
    # 1. Background Save (The "Listener")
    fact = auto_save_chat_fact(last_msg.content)
    
    # 2. Retrieve Context
    memory = get_combined_memory(last_msg.content)
    
    prompt = f"""You are a Career Assistant.
    
    USER MEMORY:
    {memory}
    
    INSTRUCTIONS:
    - Answer conversational queries naturally.
    - If you found a new fact (like "{fact}"), acknowledge it briefly.
    - Do NOT invent fake jobs.
    """
    
    final_msgs = [SystemMessage(content=prompt)] + messages
    response = llm.invoke(final_msgs)
    
    return {"messages": [response]}

# --- C. SEARCH AGENT NODE ---
def search_agent_node(state: AgentState):
    messages = state['messages']
    last_msg = messages[-1]
    
    # Retrieve Context to refine search (e.g., user skills)
    memory = get_combined_memory(last_msg.content)
    
    prompt = f"""You are a Job Search Assistant.
    
    USER PROFILE:
    {memory}
    
    INSTRUCTIONS:
    - The user wants to find jobs/info.
    - Call the `web_search_tool` with a specific query based on their profile.
    - Example: If they know Python, search "Python Internships".
    """
    
    final_msgs = [SystemMessage(content=prompt)] + messages
    response = llm_with_tools.invoke(final_msgs)
    
    return {"messages": [response]}

# --- D. CONDITIONAL EDGES ---
def route_decision(state: AgentState):
    return state["next_step"]

def continue_tool(state: AgentState):
    if state['messages'][-1].tool_calls:
        return "tools"
    return END

# --- GRAPH BUILD ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("web_search", search_agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("router")

# Router Logic
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "general_chat": "general_chat",
        "web_search": "web_search"
    }
)

# Chat ends after general response
workflow.add_edge("general_chat", END)

# Search loop (Agent -> Tool -> Agent -> End)
workflow.add_conditional_edges("web_search", continue_tool)
workflow.add_edge("tools", "web_search")

app_graph = workflow.compile()

# ==========================================
# 4. CHAINLIT UI
# ==========================================

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    await cl.Message(content="**Smart Career Agent Ready.**\nI can Chat (Memory) or Search (Web).").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("messages")
    
    # 1. FILE UPLOAD HANDLING (Bypasses Router, goes straight to Memory)
    if message.elements:
        for element in message.elements:
            if any(x in element.mime for x in ["pdf", "word", "text"]):
                await cl.Message(content=f"Analyzing {element.name}...").send()
                text = parse_file(element.path)
                summary = process_document(text, element.name)
                if summary:
                    await cl.Message(content="**File Analyzed & Memorized.**").send()
                    history.append(HumanMessage(content=f"User uploaded {element.name}. Summary: {summary}"))
    
    # 2. TEXT HANDLING (Goes to Router)
    else:
        history.append(HumanMessage(content=message.content))

    ui_msg = cl.Message(content="")
    await ui_msg.send()
    
    inputs = {"messages": history}
    
    for event in app_graph.stream(inputs):
        for key, value in event.items():
            
            # Show "Thinking" for different nodes
            if key == "web_search":
                # Only show if it's actually calling a tool
                msg = value["messages"][-1]
                if msg.tool_calls:
                    await ui_msg.stream_token("\n*🔍 Activated Web Search...*\n")
                    await ui_msg.update()
                else:
                    ui_msg.content = msg.content
                    await ui_msg.update()
            
            elif key == "general_chat":
                msg = value["messages"][-1]
                ui_msg.content = msg.content
                await ui_msg.update()

    if ui_msg.content:
        history.append(AIMessage(content=ui_msg.content))
        cl.user_session.set("messages", history)   