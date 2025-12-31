import os
import uuid
import chainlit as cl
import chromadb
import json
import datetime
from typing import TypedDict, Literal, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
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
# 3. AGENT LOGIC
# ==========================================

class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: Literal["action_agent", "general_chat"]

llm = ChatOllama(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# --- A. ROUTER NODE ---
def router_node(state: AgentState):
    messages = state['messages']
    last_msg = messages[-1].content.lower()
    
    keywords = ["plan", "roadmap", "schedule", "find", "search", "job", "internship", "learn"]
    if any(k in last_msg for k in keywords):
        return {"next_step": "action_agent"}
    return {"next_step": "general_chat"}

# --- B. ACTION AGENT NODE ---
def action_agent_node(state: AgentState):
    messages = state['messages']
    
    # 1. Fetch context (use the first user message for context stability)
    user_msg_content = messages[0].content if messages else ""
    memory = get_combined_memory(user_msg_content)
    
    # 2. Define System Prompt
    system_prompt = f"""You are a Career Architect.
    USER CONTEXT: {memory}
    
    TOOLS:
    - web_search_tool: For jobs/internships.
    - generate_roadmap_tool: First step for learning a skill.
    - create_schedule_tool: Second step (requires roadmap output).
    
    INSTRUCTIONS:
    If the user asks for a plan:
    1. Call generate_roadmap_tool.
    2. Call create_schedule_tool with the result.
    3. Present the schedule clearly in the final response.
    """
    
    # 3. Invoke LLM with System Prompt + History
    # We reconstruct the list to ensure the System Prompt is ALWAYS present
    # Filter out previous system messages to avoid duplication
    chat_history = [m for m in messages if not isinstance(m, SystemMessage)]
    final_input = [SystemMessage(content=system_prompt)] + chat_history
    
    response = llm_with_tools.invoke(final_input)
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
    if last_message.tool_calls:
        return "tools"
    return END

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("action_agent", action_agent_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", route_decision, {"action_agent": "action_agent", "general_chat": "general_chat"})
workflow.add_conditional_edges("action_agent", should_continue)
workflow.add_edge("tools", "action_agent")
workflow.add_edge("general_chat", END)

app_graph = workflow.compile()

# ==========================================
# 4. UI (ROBUST GUARDRAILS)
# ==========================================

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    await cl.Message(content="Career Agent Ready.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("messages")
    history.append(HumanMessage(content=message.content))
    
    inputs = {"messages": history}
    
    # 1. Status Message (For "Thinking..." updates)
    status_msg = cl.Message(content="")
    await status_msg.send()
    
    final_response_content = ""

    for event in app_graph.stream(inputs):
        for key, value in event.items():
            
            # AGENT NODE
            if key == "action_agent":
                msg = value["messages"][-1]
                
                # Case A: Tool Call (Thinking)
                if msg.tool_calls:
                    tool_names = ", ".join([t['name'] for t in msg.tool_calls])
                    await status_msg.stream_token(f"Architecting Plan ({tool_names})... \n")
                    await status_msg.update()
                
                # Case B: Final Response (Text)
                elif msg.content:
                    final_response_content = msg.content

            # TOOL NODE
            elif key == "tools":
                 await status_msg.stream_token("Data retrieved. Finalizing schedule... \n")
                 await status_msg.update()
            
            # CHAT NODE
            elif key == "general_chat":
                msg = value["messages"][-1]
                final_response_content = msg.content

    # 2. Send Final Response as a FRESH message (Ensures visibility)
    if final_response_content:
        # Clear the "Thinking" status to keep chat clean (Optional)
        # await status_msg.remove() 
        
        await cl.Message(content=final_response_content).send()
        history.append(AIMessage(content=final_response_content))
        cl.user_session.set("messages", history)