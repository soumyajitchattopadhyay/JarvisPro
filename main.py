import os
import sys
from typing import Annotated, TypedDict, List, Literal, Union, Dict, cast
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# FastAPI imports for deployment
from fastapi import FastAPI

import actions 

# --- SAFE HARDWARE IMPORTS ---
# This prevents the "Status 1" crash on Render/Railway
try:
    from speak import speak 
    from listen import listen_for_command
except Exception:
    speak = None
    listen_for_command = None

load_dotenv()

# --- 1. WEB SERVER INIT (The engine's heart for Render) ---
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "online", "system": "TESrACT", "mode": "cloud"}

# --- 2. BRAIN PROTOCOLS ---
PROTOCOLS = {
    "CORE": "You are TESrACT. Execute first, talk second. Use the 'execute_python_code' tool for ALL math, logic, and file ops.",
    "CODE_STYLE": "All files created must include Type Hints (PEP 484) and Google-style docstrings.",
    "ERROR_RECOVERY": "If a tool fails, read the traceback, search for the fix, and retry WITHOUT asking the user.",
    "COMMUNICATION": "Keep verbal updates under 15 words.",
    "PERMISSION": "If CURRENT_CONTROL is RESTRICTED, you are FORBIDDEN from using system tools. You must first ask the user: 'I need to perform a system action. Allow Control?'",
    "ACTIVATION": "Only once the user says 'Allow Control' will CURRENT_CONTROL become ACTIVE."
}

class JProState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: NotRequired[str]
    control_allowed: bool 

# --- 3. NODES & LOGIC ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(actions.tools)

def call_brain(state: JProState):
    is_allowed = state.get("control_allowed", False)
    status = "ACTIVE" if is_allowed else "RESTRICTED"
    sys_msg = SystemMessage(content=f"SYSTEM PROTOCOLS:\n{PROTOCOLS}\nCURRENT_CONTROL: {status}")
    response = llm_with_tools.invoke([sys_msg] + state["messages"][-10:])
    
    if response.tool_calls:
        for t in response.tool_calls:
            t['args']['control_allowed'] = is_allowed
    return {"messages": [response]}

def reviewer_node(state: JProState):
    last_msg = state["messages"][-1]
    content_str = str(last_msg.content).lower() if last_msg.content else ""
    if "allow control" in content_str: return {"control_allowed": True}
    if "stop control" in content_str: return {"control_allowed": False}
    return state

# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(JProState)
workflow.add_node("agent", call_brain)
workflow.add_node("tools", ToolNode(actions.tools))
workflow.add_node("reviewer", reviewer_node)
workflow.add_edge(START, "agent")

def route_after_agent(state: JProState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "reviewer")
workflow.add_edge("reviewer", "agent")

memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)

# --- 5. WEB API ENDPOINT (Bridged to HUD) ---
@app.get("/", response_class=HTMLResponse)
async def serve_hud():
    """Serves the Stark HUD interface at the root URL."""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>TESrACT HUD Error: index.html not found in root.</h1>"

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Processes JSON input from the HUD and returns the brain's response."""
    try:
        # 1. Parse JSON body from your HUD
        data = await request.json()
        user_text = data.get("text", "")
        thread_id = data.get("thread_id", "web_user_001")

        # 2. Extract control state for the HUD
        # We check the memory to see if control is active
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = compiled_graph.get_state(config)
        current_allowed = state.values.get("control_allowed", False)

        # 3. If user is explicitly allowing control in this message:
        if "allow control" in user_text.lower():
            current_allowed = True

        inputs = cast(JProState, {
            "messages": [HumanMessage(content=user_text)], 
            "control_allowed": current_allowed
        })
        
        # 4. Execute Graph (Using invoke for stable web response)
        result = compiled_graph.invoke(inputs, config=config)
        final_reply = result["messages"][-1].content
        
        # 5. Return JSON in the exact format your HTML script expects
        return {
            "reply": final_reply, 
            "control_status": current_allowed
        }
    except Exception as e:
        return {"reply": f"LATTICE_ERROR: {str(e)}", "control_status": False}
    
# --- 6. LOCAL VOICE LOOP (Untouched) ---
def main_loop():
    if speak: speak("TESrACT system online.", wait_for_speech=True)
    config: RunnableConfig = {"configurable": {"thread_id": "jarvis_session_001"}}
    
    while True:
        command = listen_for_command() if listen_for_command else input("User: ")
        if not command: continue
        if any(word in command.lower() for word in ["shutdown", "exit"]):
            if speak: speak("Powering down.")
            break

        state_vals = compiled_graph.get_state(config).values
        is_allowed = state_vals.get("control_allowed", False)

        inputs = cast(JProState, {"messages": [HumanMessage(content=command)], "control_allowed": is_allowed})
        
        try:
            for event in compiled_graph.stream(inputs, config=config, stream_mode="values"):
                final_msg = event["messages"][-1]
            
            if isinstance(final_msg, AIMessage) and not final_msg.tool_calls:
                output_text = str(final_msg.content).replace("`", "")
                print(f"TESrACT: {output_text}")
                if speak: speak(output_text, wait_for_speech=True)
        except Exception as e:
            print(f"Graph Error: {e}")

if __name__ == "__main__":
    # RENDER START LOGIC
    # Render will provide a $PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # If we are in a cloud environment, we run the API
    if os.environ.get("RENDER") or len(sys.argv) > 1:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        main_loop()
