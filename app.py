import os
import uuid
import chainlit as cl
import chromadb
import ollama
import pypdf
import docx
import json
import datetime
from typing import TypedDict, Literal, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Configuration
MODEL_NAME = "llama3.1:8b"
DB_PATH = r"C:\Users\hari7\Documents\Anokha Hackthon\ChromaDB"

# Initialize Client
client = chromadb.PersistentClient(path=DB_PATH)
chat_collection = client.get_or_create_collection(name="user_chat_facts")
doc_collection = client.get_or_create_collection(name="user_documents")

# ==========================================
# 1. TOOLS DEFINITION
# ==========================================

search_engine = DuckDuckGoSearchRun()

@tool
def web_search_tool(query: str):
    """Finds jobs, internships, or researches companies."""
    try:
        return search_engine.invoke(query)
    except Exception as e:
        return f"Search Error: {e}"

# Separate LLM for strict JSON generation
tool_llm = ChatOllama(model=MODEL_NAME, temperature=0.1)

@tool
def generate_roadmap_tool(skill: str):
    """
    Generates a structured learning path for a skill.
    Returns a JSON string.
    """
    print(f"Tool: Architecting roadmap for {skill}...")
    prompt = f"""
    Create a step-by-step learning roadmap for: {skill}.
    Strictly output ONLY a JSON array. Format:
    [{{"topic": "Name", "hours_needed": integer, "description": "Summary"}}]
    """
    try:
        response = tool_llm.invoke(prompt)
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return clean_json
    except Exception as e:
        return f"Roadmap Error: {e}"

@tool
def create_schedule_tool(roadmap_json: str, hours_per_day: int = 2):
    """
    Maps a roadmap JSON to a calendar.
    """
    print(f"Tool: Scheduling ({hours_per_day} hrs/day)...")
    try:
        data = json.loads(roadmap_json)
        schedule = []
        current_date = datetime.date.today()
        
        for module in data:
            needed = module.get('hours_needed', 2)
            days = (needed // hours_per_day) + (1 if needed % hours_per_day > 0 else 0)
            end_date = current_date + datetime.timedelta(days=max(1, days - 1))
            
            schedule.append({
                "topic": module['topic'],
                "dates": f"{current_date} to {end_date}",
                "focus": module.get('description', '')
            })
            current_date = end_date + datetime.timedelta(days=1)
            
        return json.dumps(schedule, indent=2)
    except Exception as e:
        return f"Scheduling Error: {e}"

# Unified Tool Set
all_tools = [web_search_tool, generate_roadmap_tool, create_schedule_tool]
tool_node = ToolNode(all_tools)

# ==========================================
# 2. HELPERS
# ==========================================

def get_combined_memory(text):
    context = ""
    try:
        res = chat_collection.query(query_texts=[text], n_results=3)
        if res['documents'][0]:
            context += "FROM CHAT HISTORY:\n" + "\n".join([f"- {d}" for d in res['documents'][0]]) + "\n"
    except: pass
    return context

def auto_save_chat_fact(user_text):
    if "my name" in user_text.lower() or "i know" in user_text.lower():
        chat_collection.add(documents=[user_text], ids=[str(uuid.uuid4())])
        return "Memory Updated"
    return None

# ==========================================
# 3. AGENT LOGIC (THE FIX)
# ==========================================

class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: Literal["action_agent", "general_chat"]

# Main LLM with ALL tools bound
llm = ChatOllama(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# --- A. ROUTER NODE ---
def router_node(state: AgentState):
    messages = state['messages']
    last_msg = messages[-1].content.lower()
    
    # If users asks for plans, search, or roadmap -> ACTION AGENT
    keywords = ["plan", "roadmap", "schedule", "find", "search", "job", "internship"]
    if any(k in last_msg for k in keywords):
        return {"next_step": "action_agent"}
    
    return {"next_step": "general_chat"}

# --- B. ACTION AGENT NODE (The Loop Fix) ---
def action_agent_node(state: AgentState):
    """
    This node handles ALL tool interactions.
    It loops: Agent -> Tool -> Agent -> Final Answer.
    """
    messages = state['messages']
    
    # We don't add system prompt every loop, just ensuring context exists
    # If the last message was a ToolMessage, the Agent will naturally read it and respond.
    
    if not isinstance(messages[-1], ToolMessage):
        # Only add prompt if starting a new action, not returning from tool
        memory = get_combined_memory(messages[-1].content)
        system_prompt = f"""You are a Career Architect.
        USER CONTEXT: {memory}
        
        TOOLS AVAILABLE:
        - web_search_tool: For jobs/internships.
        - generate_roadmap_tool: First step for learning a skill.
        - create_schedule_tool: Second step (requires roadmap output).
        """
        # Note: In LangGraph, it's safer to keep messages clean. 
        # We prepend system prompt only for the invocation.
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    else:
        # Returning from a tool, just continue conversation
        response = llm_with_tools.invoke(messages)
        
    return {"messages": [response]}

# --- C. GENERAL CHAT NODE ---
def general_chat_node(state: AgentState):
    messages = state['messages']
    auto_save_chat_fact(messages[-1].content)
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- D. CONDITIONAL EDGES ---
def route_decision(state: AgentState):
    return state["next_step"]

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # IF the agent called a tool -> Go to 'tools' node
    if last_message.tool_calls:
        return "tools"
    # ELSE (Agent is done) -> End
    return END

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("action_agent", action_agent_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("router")

# 1. Router Split
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "action_agent": "action_agent",
        "general_chat": "general_chat"
    }
)

# 2. Action Loop (The Fix: Tools go back to Agent)
workflow.add_conditional_edges("action_agent", should_continue)
workflow.add_edge("tools", "action_agent")  # <--- CRITICAL FIX

# 3. Chat End
workflow.add_edge("general_chat", END)

app_graph = workflow.compile()

# ==========================================
# 4. UI
# ==========================================

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    await cl.Message(content="**Career Agent Ready.**\nTry: 'Create a study plan for Docker'").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("messages")
    history.append(HumanMessage(content=message.content))
    
    inputs = {"messages": history}
    ui_msg = cl.Message(content="")
    await ui_msg.send()

    for event in app_graph.stream(inputs):
        for key, value in event.items():
            
            # If Agent is "Thinking" (Generating tool calls)
            if key == "action_agent":
                msg = value["messages"][-1]
                if msg.tool_calls:
                    tool_names = ", ".join([t['name'] for t in msg.tool_calls])
                    await ui_msg.stream_token(f"\n*⚙️ Calling Tool: {tool_names}...*\n")
                    await ui_msg.update()
                elif msg.content:
                    # Final Answer from Agent
                    ui_msg.content = msg.content
                    await ui_msg.update()

            # If Tool has finished
            elif key == "tools":
                 await ui_msg.stream_token("\n*✅ Data Processed. Analyzing results...*\n")
                 await ui_msg.update()
    
    if ui_msg.content:
        history.append(AIMessage(content=ui_msg.content))
        cl.user_session.set("messages", history)