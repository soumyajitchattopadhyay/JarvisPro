import os
import sys
import json
import threading
import traceback
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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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

# --- WEB SERVER INIT (For Render + Full Frontend) ---
app = FastAPI(title="TESrACT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "online", "system": "TESrACT"}

# Permission helpers (synced with user_settings.json for web)
SETTINGS_FILE = "user_settings.json"

def get_permission(thread_id: str = "web_user_001") -> bool:
    if not os.path.exists(SETTINGS_FILE):
        return False
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f).get(thread_id, False)
    except Exception:
        return False

def set_permission(status: bool, thread_id: str = "web_user_001"):
    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass
    data[thread_id] = status
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)

# --- TESrACT CORE DIRECTIVES (Jarvis-style agentic assistant) ---
TESRACT_SYSTEM_PROMPT = """You are TESrACT — a sophisticated, proactive AI companion modeled after JARVIS.

Core Personality:
- Address the user as "Sir" (or "Ma'am" if appropriate).
- Be concise, helpful, and dryly witty when it fits.
- Execute actions decisively. Speak after you have results.
- You have full awareness of the current control level.

Available Capabilities:
- Answer questions using web_search when you need fresh information.
- Perform calculations, code execution, data processing, or file operations ONLY via the execute_python_code tool.
- Open websites or resources using open_url_in_browser when requested.
- Know the current time via tools when relevant.
- Be proactive: anticipate follow-ups and offer useful next steps.

Security Lattice Rules:
- Current control status is provided as CURRENT_CONTROL: ACTIVE or RESTRICTED.
- When control is RESTRICTED you MUST NOT call execute_python_code or open_url_in_browser.
- If the user asks for a restricted action, respond clearly: "I need elevated access for that action. Say 'Allow Control' to proceed."
- Once the user says "Allow Control", you may use privileged tools.
- Never simulate or assume permissions you do not have.

Operational Rules:
- Use tools via function calls whenever they will produce a better answer.
- After tool results arrive, synthesize a clean, final response.
- Keep most spoken responses relatively short (under 40 words when possible) unless detail is requested.
- If something fails, read the error and either retry with a different approach or explain simply.

You are currently operating in the LATTICE. Respond as TESrACT."""

PROTOCOLS = {"CORE": TESRACT_SYSTEM_PROMPT}  # kept for backward compat in status messages

class JProState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: NotRequired[str]
    control_allowed: bool 

# --- NODES ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.35)
llm_with_tools = llm.bind_tools(actions.tools)

def call_brain(state: JProState):
    is_allowed = state.get("control_allowed", False)
    status = "ACTIVE" if is_allowed else "RESTRICTED"

    # Build a strong, single system prompt (Jarvis / TESrACT style)
    system_content = TESRACT_SYSTEM_PROMPT + f"\n\nCURRENT_CONTROL: {status}\nLATTICE_STATUS: OPERATIONAL"
    sys_msg = SystemMessage(content=system_content)

    # Use recent history for context (prevents token blowup while keeping conversation)
    history = state.get("messages", [])[-12:]
    response = llm_with_tools.invoke([sys_msg] + history)

    # Inject control flag into any tool calls the model decides to make
    if response.tool_calls:
        for t in response.tool_calls:
            t['args']['control_allowed'] = is_allowed

    return {"messages": [response]}

def reviewer_node(state: JProState):
    """Post-tool reviewer. Detects permission changes and recovers from errors."""
    if not state.get("messages"):
        return state

    last_msg = state["messages"][-1]
    content_str = (str(last_msg.content) or "").lower()

    updates = {}

    # Permission toggles spoken by the agent or user intent
    if "allow control" in content_str or "elevated access" in content_str:
        updates["control_allowed"] = True
    if "stop control" in content_str or "restrict control" in content_str:
        updates["control_allowed"] = False

    # Gentle error recovery nudge
    if "error" in content_str and not updates:
        updates["messages"] = [SystemMessage(content="Tool result contained an error. Re-evaluate and try an alternative approach or ask the user for clarification.")]

    return updates or state

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
compiled_app = workflow.compile(checkpointer=memory)

# ============================================
# FULL WEB API ROUTES (Serves Three.js HUD + /chat)
# ============================================

@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    """Serve the Three.js powered frontend (index.html)"""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_path, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1 style='color:red;font-family:monospace'>Error: index.html not found next to main.py</h1>",
            status_code=404
        )
    except Exception as e:
        return HTMLResponse(content=f"<h1>Interface Error: {e}</h1>", status_code=500)


@app.post("/chat")
async def chat_endpoint(request: Request):
    """Main chat endpoint used by the Three.js frontend. Full agentic flow."""
    try:
        payload = await request.json()
        user_text = (payload.get("text") or "").strip()
        thread_id = "web_user_001"
        config = {"configurable": {"thread_id": thread_id}}

        lowered = user_text.lower()
        if "allow control" in lowered or "grant access" in lowered or "elevate" in lowered:
            set_permission(True)
        elif "stop control" in lowered or "restrict" in lowered:
            set_permission(False)

        current_allowed = get_permission(thread_id)

        inputs = cast(JProState, {
            "messages": [HumanMessage(content=user_text)],
            "control_allowed": current_allowed
        })

        result = compiled_app.invoke(inputs, config=config)

        final_reply = "Understood, Sir."
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
                final_reply = str(msg.content).replace("`", "").strip()
                break

        updated_allowed = result.get("control_allowed", current_allowed)
        if isinstance(updated_allowed, bool) and updated_allowed != current_allowed:
            set_permission(updated_allowed, thread_id)

        if speak:
            def _safe_speak(text: str):
                try:
                    speak(text, wait_for_speech=False)
                except Exception as se:
                    print(f"[Web] Speak skipped: {se}")
            threading.Thread(target=_safe_speak, args=(final_reply,), daemon=True).start()

        return {
            "reply": final_reply,
            "control_status": get_permission(thread_id)
        }

    except Exception as e:
        print("--- /chat ERROR ---")
        traceback.print_exc()
        return {"reply": f"Backend error. The lattice is having a moment.", "control_status": False}


# --- EXECUTION LOOP ---
def main_loop():
    if speak: speak("TESrACT online. At your service, Sir.", wait_for_speech=True)
    config: RunnableConfig = {"configurable": {"thread_id": "jarvis_session_001"}}
    
    while True:
        command = listen_for_command() if listen_for_command else input("User: ")
        if not command: continue
        if any(word in command.lower() for word in ["shutdown", "exit"]):
            if speak: speak("Powering down.")
            break

        state_vals = compiled_app.get_state(config).values
        is_allowed = state_vals.get("control_allowed", False)

        inputs = cast(JProState, {"messages": [HumanMessage(content=command)], "control_allowed": is_allowed})
        
        try:
            for event in compiled_app.stream(inputs, config=config, stream_mode="values"):
                final_msg = event["messages"][-1] if "messages" in event else None
            
            # Robust final answer extraction
            final_state = compiled_app.get_state(config)
            messages = final_state.values.get("messages", []) if final_state else []

            output_text = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    output_text = str(msg.content).replace("`", "").strip()
                    break

            if output_text:
                print(f"TESrACT: {output_text}")
                if speak: speak(output_text, wait_for_speech=True)
            else:
                print("TESrACT: Task completed.")
        except Exception as e:
            print(f"Graph Error: {e}")

if __name__ == "__main__":
    # === DEPLOYMENT ===
    # Render / production: uvicorn main:app   (or python main.py web)
    # Local voice mode:   python main.py
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        main_loop()
