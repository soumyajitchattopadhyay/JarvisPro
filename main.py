"""
TESrACT — JARVIS-style agent with smart LLM routing.

Routing (llm_router.py) — local-first on Apple Silicon unified RAM:
  - Ollama  → light models (Llama 3.2 3B) for Q&A/tools; heavy (Gemma2 9B) for analysis
  - MLX     → optional Apple Silicon text path when OLLAMA is unavailable
  - Groq    → cloud fallback + reliable tool calling
  - Colab   → deepest inference when local RAM is insufficient
  - Web search results stored in Chroma in-RAM for recall across turns

Configure via .env: OLLAMA_* , GROQ_API_KEY, COLAB_LLM_URL, LLM_ROUTING_MODE.
"""
from dotenv import load_dotenv

load_dotenv()  # Must run before any other import — llm_router reads env at import time.

import os
import sys
import json
import threading
import traceback
import datetime
import webbrowser
import re
from typing import Annotated, TypedDict, List, Callable, cast
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from groq import RateLimitError
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import actions
import llm_router
from llm_router import (
    route_and_invoke,
    colab_configured,
    colab_is_healthy,
    build_groq_llm,
    local_status,
)
from local_search_memory import memory_stats

# --- SAFE HARDWARE IMPORTS ---
# This prevents the "Status 1" crash on Render/Railway
SpeakFn = Callable[..., None]
ListenFn = Callable[[], str | None]

try:
    from speak import speak as _speak_impl
    from listen import listen_for_command as _listen_impl
    speak: SpeakFn | None = _speak_impl
    listen_for_command: ListenFn | None = _listen_impl
except Exception:
    speak = None
    listen_for_command = None


def _run_speak(text: str, *, wait_for_speech: bool = False) -> None:
    """Speak in a background thread when available; safe if speak is None."""
    speak_fn = speak
    if speak_fn is None:
        return

    def _do_speak() -> None:
        try:
            speak_fn(text, wait_for_speech=wait_for_speech)
        except Exception as se:
            print(f"[Web] Speak skipped: {se}")

    if wait_for_speech:
        _do_speak()
    else:
        threading.Thread(target=_do_speak, daemon=True).start()

# --- ROUTING STARTUP STATUS ---
_routing_mode = os.getenv("LLM_ROUTING_MODE", "auto")
print(f"[TESrACT] LLM routing mode: {_routing_mode} (local-first when auto)")
_local = local_status()
print(
    f"[TESrACT] Ollama (local RAM): "
    f"{'online' if _local['ollama_healthy'] else 'offline'} "
    f"light={_local['light_model']} heavy={_local['heavy_model']}"
)
if _local.get("mlx_enabled"):
    print(f"[TESrACT] MLX fallback: enabled ({_local.get('mlx_model')})")
if colab_configured():
    _colab_ok = colab_is_healthy()
    _colab_host = (os.getenv("COLAB_LLM_URL") or "")[:48]
    print(f"[TESrACT] Colab uplink (cloud): {'online' if _colab_ok else 'offline'} ({_colab_host})")
else:
    print("[TESrACT] Colab uplink: not configured — local + Groq fallback")

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
    # [ROUTING] Surface local + cloud backends for deploy monitoring.
    return {
        "status": "online",
        "system": "TESrACT",
        "local": local_status(),
        "search_memory": memory_stats(),
        "colab_configured": colab_configured(),
        "colab_healthy": colab_is_healthy() if colab_configured() else False,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "llm_routing_mode": os.getenv("LLM_ROUTING_MODE", "auto"),
        "last_route": llm_router.last_route_used,
        "last_local_model": llm_router.last_local_model or None,
    }

_icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
if os.path.isdir(_icons_dir):
    app.mount("/icons", StaticFiles(directory=_icons_dir), name="icons")

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

def try_simple_command(text: str, control_allowed: bool) -> str | None:
    """Handle very simple commands directly (bypass LLM to save tokens and work under rate limits)."""
    t = text.lower().strip()

    # Time queries
    if any(kw in t for kw in ["what time", "current time", "tell time", "what's the time"]):
        now = datetime.datetime.now()
        return now.strftime("It is %A, %B %d %Y, %I:%M %p, Sir.")

    # Open sites / URLs (basic support)
    open_match = re.search(r"open\s+([a-zA-Z0-9\.\-]+)", t)
    if open_match:
        site = open_match.group(1).strip()
        if "google" in site:
            url = "https://google.com"
        elif "youtube" in site:
            url = "https://youtube.com"
        elif "github" in site:
            url = "https://github.com"
        else:
            if "." not in site:
                url = f"https://{site}.com"
            else:
                url = site if site.startswith(("http://", "https://")) else f"https://{site}"
        if not control_allowed:
            return "I need elevated access to open pages, Sir. Please say 'Allow Control'."
        try:
            webbrowser.open(url)
            return f"Opened {url}, Sir."
        except Exception:
            return f"Attempted to open {url}, Sir."

    return None

# --- TESrACT CORE DIRECTIVES (minimal Jarvis-style) ---
TESRACT_SYSTEM_PROMPT = """You are TESrACT, like JARVIS. Call user "Sir". Be calm, concise, formal.

Rules: Use tools for actions/info. Never claim results without tool confirmation.
Tools: web_search, execute_python_code (if allowed), open_url_in_browser (if allowed), get_current_time.
If RESTRICTED, ask for "Allow Control".

Reply briefly, stay in character."""

PROTOCOLS = {"CORE": TESRACT_SYSTEM_PROMPT}  # kept for backward compat in status messages

class JProState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: NotRequired[str]
    control_allowed: bool 

# --- NODES ---
# [ROUTING] Groq LLM for cloud fallback; local Ollama is built inside llm_router.
llm = build_groq_llm()
llm_with_tools = llm.bind_tools(actions.tools)

def call_brain(state: JProState):
    is_allowed = state.get("control_allowed", False)
    status = "ACTIVE" if is_allowed else "RESTRICTED"

    system_content = TESRACT_SYSTEM_PROMPT + f"\n\nCURRENT_CONTROL: {status}"
    sys_msg = SystemMessage(content=system_content)

    raw_history = state.get("messages", [])
    history = [m for m in raw_history if "rate-limited" not in str(getattr(m, "content", "") or "").lower()][-4:]
    user_text = ""
    for msg in reversed(raw_history):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content or "")
            break

    # Delegate to llm_router — local Ollama first, then Groq/Colab fallback.
    try:
        response, route = route_and_invoke(
            llm_with_tools=llm_with_tools,
            sys_msg=sys_msg,
            history=history,
            control_allowed=is_allowed,
            user_text=user_text,
            history_len=len(history),
            tools=actions.tools,
        )
        reason_suffix = f" — {llm_router.last_route_reason}" if llm_router.last_route_reason else ""
        model_suffix = ""
        if route == "local" and llm_router.last_local_model:
            model_suffix = f" [{llm_router.last_local_tier or 'light'}: {llm_router.last_local_model}]"
        print(f"[TESrACT] Agent turn served by: {route.upper()}{model_suffix}{reason_suffix}")
        return {"messages": [response]}
    except RateLimitError as e:
        # Both Groq (rate-limited) and Colab (if configured) failed — agent stays usable via simple commands.
        rate_msg = "TESrACT is rate-limited right now, Sir. Basic commands like time or open may still work."
        if colab_configured() and not colab_is_healthy(log_failure=False):
            rate_msg += " Colab GPU uplink is also offline."
        print(f"[TESrACT] All LLM backends exhausted (local + Groq rate limit): {e}")
        return {"messages": [AIMessage(content=rate_msg)]}
    except Exception as e:
        # Last-resort graceful reply — graph continues; simple commands still work.
        err_msg = "I encountered a temporary issue. Please try again shortly, Sir."
        _colab_up = colab_configured() and colab_is_healthy(log_failure=False)
        print(f"[TESrACT] All LLM backends failed (local/Ollama, Groq, Colab {'up' if _colab_up else 'down'}): {e}")
        return {"messages": [AIMessage(content=err_msg)]}

def reviewer_node(state: JProState) -> dict[str, object]:
    """Post-tool reviewer. Detects permission changes and recovers from errors.
    Looks across recent messages to catch control keywords reliably."""
    if not state.get("messages"):
        return {}

    # Check last few messages for control phrases (more reliable than just last)
    recent_content = " ".join(
        str(m.content or "").lower() for m in state["messages"][-3:]
    )

    updates = {}

    if "allow control" in recent_content or "elevated access" in recent_content:
        updates["control_allowed"] = True
    if "stop control" in recent_content or "restrict control" in recent_content:
        updates["control_allowed"] = False

    # In-character error recovery
    if "error" in recent_content and not updates:
        updates["messages"] = [SystemMessage(
            content="The previous action encountered an issue. I will reassess and attempt another approach if possible, Sir."
        )]

    return updates

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
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        lowered = user_text.lower()
        if "allow control" in lowered or "grant access" in lowered or "elevate" in lowered:
            set_permission(True)
        elif "stop control" in lowered or "restrict" in lowered:
            set_permission(False)

        current_allowed = get_permission(thread_id)

        # Handle simple commands directly (bypass LLM entirely to save tokens and remain usable under rate limits)
        simple_reply = try_simple_command(user_text, current_allowed)
        if simple_reply:
            _run_speak(simple_reply)
            return {"reply": simple_reply, "control_status": current_allowed}

        # Standard way: pass only the new user message.
        # The checkpointer + add_messages reducer will append it to the thread history automatically.
        inputs = cast(JProState, {
            "messages": [HumanMessage(content=user_text)],
            "control_allowed": current_allowed
        })

        try:
            compiled_app.invoke(inputs, config=config)
        except RateLimitError as e:
            # On rate limit, still return friendly msg but note that basic commands work
            print(f"[TESrACT] RateLimitError caught in chat_endpoint: {e}")
            return {"reply": "TESrACT is rate-limited right now, Sir. Basic commands like time or open may still work.", "control_status": current_allowed}
        except Exception as e:
            print(f"[TESrACT] Graph invoke error: {e}")
            return {"reply": "I encountered a temporary issue. Please try again shortly, Sir.", "control_status": current_allowed}

        # Always fetch the latest state after the full graph run (agent/tools/reviewer loop)
        final_state = compiled_app.get_state(config)
        state_values = final_state.values if final_state else None
        messages = (state_values or {}).get("messages", [])

        final_reply = "At your service, Sir."
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
                final_reply = str(msg.content).replace("`", "").strip()
                break

        updated_allowed = (state_values or {}).get("control_allowed", current_allowed)
        if isinstance(updated_allowed, bool) and updated_allowed != current_allowed:
            set_permission(updated_allowed, thread_id)

        _run_speak(final_reply)

        return {
            "reply": final_reply,
            "control_status": get_permission(thread_id)
        }

    except Exception as e:
        print("--- /chat ERROR ---")
        traceback.print_exc()
        return {"reply": "TESrACT is having a temporary issue. Please try again shortly, Sir.", "control_status": False}


# --- EXECUTION LOOP ---
def main_loop():
    _run_speak("TESrACT online. At your service, Sir.", wait_for_speech=True)
    config: RunnableConfig = {"configurable": {"thread_id": "jarvis_session_001"}}
    
    while True:
        command = listen_for_command() if listen_for_command else input("User: ")
        if not command: continue
        if any(word in command.lower() for word in ["shutdown", "exit"]):
            _run_speak("Powering down.", wait_for_speech=True)
            break

        state = compiled_app.get_state(config)
        is_allowed = state.values.get("control_allowed", False) if state and state.values else False

        # Handle simple commands directly (works even under rate limits)
        simple_reply = try_simple_command(command, is_allowed)
        if simple_reply:
            print(f"TESrACT: {simple_reply}")
            _run_speak(simple_reply, wait_for_speech=True)
            continue

        inputs = cast(JProState, {"messages": [HumanMessage(content=command)], "control_allowed": is_allowed})
        
        try:
            # Run the full agentic loop
            for event in compiled_app.stream(inputs, config=config, stream_mode="values"):
                pass  # just let it run to completion
            
            # Fetch the latest state after the complete run for reliable final response
            final_state = compiled_app.get_state(config)
            state_values = final_state.values if final_state else None
            messages = (state_values or {}).get("messages", [])

            output_text = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    output_text = str(msg.content).replace("`", "").strip()
                    break

            if output_text:
                print(f"TESrACT: {output_text}")
                _run_speak(output_text, wait_for_speech=True)
            else:
                print("TESrACT: Very well, Sir.")
        except RateLimitError as e:
            print(f"[TESrACT] Rate limit in voice loop: {e}")
            _run_speak(
                "TESrACT is rate-limited right now, Sir. Basic commands like time or open may still work.",
                wait_for_speech=True,
            )
        except Exception as e:
            print(f"Graph Error: {e}")

def run_routing_tests() -> int:
    """Run LLM routing integration tests (Groq vs Colab + tool calling)."""
    from test_routing import run_all_tests
    return run_all_tests()


if __name__ == "__main__":
    # === DEPLOYMENT ===
    # Render / production: uvicorn main:app   (or python main.py web)
    # Local voice mode:   python main.py
    # Routing tests:      python main.py test  (or python test_routing.py)
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        raise SystemExit(run_routing_tests())
    else:
        main_loop()
