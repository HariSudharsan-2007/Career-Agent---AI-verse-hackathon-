import os
import json
import chainlit as cl
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. SETUP
MODEL_NAME = "llama3.1:8b" 
MEMORY_FILE = "user_profile.json"

# 2. TOOL: SAVE PROFILE
def _save_to_profile(key, value):
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = value
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    return f"Saved {key}: {value}"

@tool
def update_user_details(name: str = None, skills: str = None, experience: str = None):
    """Call this tool to save user details like name, skills, or experience."""
    status = []
    if name: status.append(_save_to_profile('name', name))
    if skills: status.append(_save_to_profile('skills', skills))
    if experience: status.append(_save_to_profile('experience', experience))
    return "\n".join(status) if status else "No details updated."

# 3. AGENT LOGIC
class AgentState(TypedDict):
    messages: list[BaseMessage]

llm = ChatOllama(model=MODEL_NAME, temperature=0)
tools = [update_user_details] # ONLY ONE TOOL FOR NOW
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    messages = state['messages']
    # Load memory to show the LLM what it already knows
    profile_text = "{}"
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            profile_text = json.dumps(json.load(f))
            
    system_prompt = f"""You are a helpful assistant.
    CURRENT MEMORY: {profile_text}
    INSTRUCTION: If the user tells you their name or skills, call `update_user_details` immediately."""
    
    # Add system prompt to the start
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 4. GRAPH BUILD
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", lambda x: "tools" if x['messages'][-1].tool_calls else END)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()

# 5. UI (CHAINLIT)
@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    await cl.Message(content="**Step 1 Agent Ready.** Tell me your name!").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("messages")
    history.append(HumanMessage(content=message.content))
    
    msg = cl.Message(content="")
    inputs = {"messages": history}
    
    for event in app_graph.stream(inputs):
        for key, value in event.items():
            if key == "agent":
                response = value["messages"][-1]
                if response.content:
                    await msg.stream_token(response.content)
            elif key == "tools":
                await msg.stream_token("\n*Memory Updated...*\n")
    
    history.append(response)
    cl.user_session.set("messages", history)
    await msg.update()