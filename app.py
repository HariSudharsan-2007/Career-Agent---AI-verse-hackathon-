import os
import json
import chainlit as cl
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configuration
MODEL_NAME = "llama3.1:8b" 
MEMORY_FILE = "user_profile.json"

# ==========================================
# 1. SMART MEMORY TOOLS
# ==========================================

def _save_to_json(key, value):
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            try:
                data = json.load(f)
            except:
                data = {}
    else:
        data = {}
    
    # Check if data is already there to avoid unnecessary writes
    if data.get(key) == value:
        return f"SKIP_UPDATE: Already knew {key} is {value}"
    
    data[key] = value
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    return f"SUCCESS: Updated {key} to {value}"

@tool
def update_user_details(name: str = None, skills: str = None, goal: str = None):
    """
    Call this tool ONLY when the user explicitly states a fact about themselves.
    """
    status = []
    if name: status.append(_save_to_json('name', name))
    if skills: status.append(_save_to_json('skills', skills))
    if goal: status.append(_save_to_json('goal', goal))
    
    # This return message is for the LLM, not the user.
    return "\n".join(status) if status else "No changes made."

# ==========================================
# 2. AGENT ORCHESTRATION
# ==========================================

class AgentState(TypedDict):
    messages: list[BaseMessage]

llm = ChatOllama(model=MODEL_NAME, temperature=0.3)
tools = [update_user_details]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    current_messages = state['messages']
    
    # Load Memory silently
    profile_text = "{}"
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            try:
                profile_text = json.dumps(json.load(f))
            except:
                pass

    # --- INTELLIGENT SYSTEM PROMPT ---
    system_prompt = f"""You are a helpful Career Assistant with a persistent memory.
    
    CURRENT SAVED PROFILE: {profile_text}
    
    **YOUR GOAL:** Chat naturally with the user while building their profile in the background.
    
    **RULES FOR UPDATING MEMORY:**
    1. **Strict Extraction:** Only call `update_user_details` if the user *explicitly* introduces a fact.
       - "I am Hari" -> UPDATE Name.
       - "My skills are Python and Java" -> UPDATE Skills.
       - "Super super", "Cool", "Okay" -> **DO NOT** update anything. Just chat.
    
    2. **Handling Updates:**
       - If you call the tool, do NOT mention "I updated the JSON" or show technical logs.
       - Just say something natural like: "Nice to meet you, Hari!" or "Great, Python is a useful skill."
    
    3. **Answering Questions:**
       - If the user asks "What is my name?", read the CURRENT SAVED PROFILE and answer directly. DO NOT call the tool.
    """
    
    messages_with_context = [SystemMessage(content=system_prompt)] + current_messages
    
    response = llm_with_tools.invoke(messages_with_context)
    return {"messages": [response]}

# Graph Setup
tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()

# ==========================================
# 3. CLEAN UI (CHAINLIT)
# ==========================================

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    await cl.Message(content="**Assistant Ready.** I'm listening!").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("messages")
    history.append(HumanMessage(content=message.content))

    ui_msg = cl.Message(content="")
    await ui_msg.send()
    
    inputs = {"messages": history}
    
    for event in app_graph.stream(inputs):
        for key, value in event.items():
            if key == "agent":
                final_response = value["messages"][-1]
                
                # If it's a tool call, show a subtle "Thinking" message, not the raw log
                if final_response.tool_calls:
                    await ui_msg.stream_token("") # Do nothing, keep it silent or show a small icon
                
            elif key == "tools":
                # The tool has finished. We keep this silent too.
                # The LLM will speak next and confirm the action naturally.
                pass

    if final_response:
        # We only display the FINAL text response from the Agent to the User
        if final_response.content:
            history.append(final_response)
            cl.user_session.set("messages", history)
            ui_msg.content = final_response.content
            await ui_msg.update()