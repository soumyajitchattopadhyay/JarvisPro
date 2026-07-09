"""
TESrACT — JARVIS-style agent with smart LLM routing.

Secure Brain architecture:
  - Host Mac is a cognitive engine (logic, LLM, API orchestration)
  - No public request may mutate the host OS; physical actions return
    client-side execution intents (DOWNLOAD_FILE / DISPLAY_DATA / LOGIC_ONLY)
  - /api/update-brain and /internal/llm/invoke require HMAC or shared-secret handshake
  - /chat is public by default (HUD); set BRAIN_AUTH_REQUIRE_CHAT=true to lock it

Routing (llm_router.py) — local-first on Apple Silicon unified RAM:
  - Ollama  → light models for Q&A/tools; heavy for analysis
  - Groq / Colab → cloud fallbacks
  - Web search memory in Chroma (in-RAM)

Configure via .env: OLLAMA_* , GROQ_API_KEY, BRAIN_REGISTRY_SECRET, LLM_ROUTING_MODE.
"""
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()  # Must run before any other import — llm_router reads env at import time.

import os
import sys
import json
import threading
import traceback
import datetime
import re
from typing import Annotated, TypedDict, List, Callable, cast, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from groq import RateLimitError
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import actions
import brain_auth
import llm_router
import mac_access
from llm_router import (
    route_and_invoke,
    route_and_invoke_synthesis,
    colab_configured,
    colab_is_healthy,
    build_groq_llm,
    local_status,
    hybrid_status,
    deserialize_messages,
    serialize_aimessage,
)
from local_search_memory import (
    format_memory_for_system,
    get_memory_stats,
    run_memory_cleanup,
    search_memory_status,
    store_turn_summary,
)

# --- SAFE HARDWARE IMPORTS ---
# This prevents the "Status 1" crash on Render/Railway
SpeakFn = Callable[..., None]
from typing import Optional
ListenFn = Callable[[], Optional[str]]
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
print(f"[TESrACT] LLM routing mode: {_routing_mode} (Ollama primary when auto)")
_local = local_status()
_ollama_state = "online" if _local["ollama_healthy"] else "offline"
_light_avail = "ready" if _local.get("light_model_available") else "missing"
_heavy_avail = "ready" if _local.get("heavy_model_available") else "missing"
print(
    f"[TESrACT] Ollama (primary): {_ollama_state} "
    f"light={_local['light_model']} ({_light_avail}) "
    f"heavy={_local['heavy_model']} ({_heavy_avail})"
)
if _local.get("installed_models"):
    print(f"[TESrACT] Ollama models pulled: {', '.join(_local['installed_models'])}")
if _local.get("mlx_enabled"):
    print(f"[TESrACT] MLX fallback: enabled ({_local.get('mlx_model')})")
if colab_configured():
    _colab_ok = colab_is_healthy()
    _colab_host = (os.getenv("COLAB_LLM_URL") or "")[:48]
    print(f"[TESrACT] Colab uplink (cloud): {'online' if _colab_ok else 'offline'} ({_colab_host})")
else:
    print("[TESrACT] Colab uplink: not configured — local + Groq fallback")

_hybrid = hybrid_status()
if _hybrid["enabled"]:
    if _hybrid["local_instance_url"]:
        _mac_state = "online" if _hybrid["remote_mac_healthy"] else "offline"
        print(
            f"[TESrACT] Hybrid routing: ENABLED — remote Mac {_mac_state} "
            f"({_hybrid['local_instance_url'][:48]})"
        )
    else:
        print("[TESrACT] Hybrid routing: ENABLED but LOCAL_INSTANCE_URL not set")
else:
    print("[TESrACT] Hybrid routing: disabled (set ENABLE_HYBRID_ROUTING=true on Render)")

# --- WEB SERVER INIT (For Render + Full Frontend) ---
app = FastAPI(title="TESrACT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _system_info() -> dict:
    """Fast host snapshot for hybrid-router liveness probes."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "platform": sys.platform,
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "ram_used_percent": vm.percent,
        }
    except Exception as exc:
        return {"platform": sys.platform, "error": str(exc)}


@app.get("/health")
def health_check():
    """Lightweight liveness probe — local Mac instance + inference readiness."""
    import brain_registry

    local = local_status()
    search_memory = search_memory_status()
    local_llm_available = bool(local.get("ollama_healthy"))

    return {
        "status": "ok",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "local_llm_available": local_llm_available,
        "memory_available": bool(search_memory.get("available")),
        "system": _system_info(),
        "local": local,
        "search_memory": search_memory,
        "colab_configured": colab_configured(),
        "colab_healthy": colab_is_healthy(log_failure=False) if colab_configured() else False,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "llm_routing_mode": os.getenv("LLM_ROUTING_MODE", "auto"),
        "hybrid": hybrid_status(),
        "brain": brain_registry.status(),
        "last_route": llm_router.last_route_used,
        "last_route_is_fallback": llm_router.last_route_is_fallback,
        "last_local_model": llm_router.last_local_model or None,
    }


# ---------------------------------------------------------------------------
# Self-healing Mac brain tunnel registry
# tunnel_manager.py on your Mac POSTs each new trycloudflare.com URL here.
# Stable global links on Render: /go  /brain  /api/global-brain
# ---------------------------------------------------------------------------

@app.post("/api/update-brain")
async def update_brain_endpoint(request: Request):
    """
    Register the live Mac Cloudflare tunnel URL.
    Requires cryptographic handshake (HMAC / shared secret headers).
    Called by tunnel_manager.py whenever a new trycloudflare.com link is minted.
    """
    import brain_registry

    _body, payload, auth_err = await brain_auth.gate_brain_auth(request)
    if auth_err is not None:
        return auth_err
    if not isinstance(payload, dict):
        payload = {}

    brain_url = str(payload.get("brain_url") or payload.get("url") or "").strip()
    heartbeat = bool(payload.get("heartbeat", False))
    source = str(payload.get("source") or "tunnel_manager")

    try:
        result = brain_registry.set_brain_url(
            brain_url, source=source, heartbeat=heartbeat,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    print(
        f"[TESrACT:brain] "
        f"{'heartbeat' if result.get('heartbeat') else 'updated'} "
        f"url={result.get('brain_url')} changed={result.get('changed')}"
    )
    return {
        "status": "success",
        "execution_target": "brain",
        "action": "LOGIC_ONLY",
        "payload": {
            "active_url": result.get("brain_url"),
            "changed": result.get("changed"),
            "heartbeat": result.get("heartbeat"),
            "updated_at": result.get("updated_at"),
        },
        "active_url": result.get("brain_url"),
        "changed": result.get("changed"),
        "heartbeat": result.get("heartbeat"),
        "updated_at": result.get("updated_at"),
        "global_link": "/go",
    }


@app.get("/api/global-brain")
def global_brain_endpoint():
    """JSON: where the live Mac tunnel currently points (for frontends / scripts)."""
    import brain_registry

    st = brain_registry.status()
    if not st.get("brain_url"):
        return JSONResponse(
            {
                "online": False,
                "error": "Mac brain is offline — start tunnel_manager.py on the Mac",
                "url": None,
            },
            status_code=503,
        )
    return {
        "online": True,
        "url": st["brain_url"],
        "updated_at": st.get("updated_at"),
        "age_seconds": st.get("age_seconds"),
        "source": st.get("source"),
        "global_link": "/go",
    }


@app.get("/api/brain-status")
def brain_status_endpoint():
    """Full registry status (no secret)."""
    import brain_registry

    return brain_registry.status()


@app.get("/go")
@app.get("/brain")
def global_brain_redirect():
    """
    Stable global link: always 302-redirects to the current Mac tunnel URL.

    Bookmark https://your-app.onrender.com/go — when Cloudflare rotates the
    trycloudflare.com host, this still lands on the live tunnel.
    """
    import brain_registry
    from fastapi.responses import RedirectResponse

    url = brain_registry.get_brain_url()
    if not url:
        return HTMLResponse(
            content=(
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>TESrACT — Mac offline</title></head>"
                "<body style='font-family:system-ui;background:#0a0a0f;color:#e8e8f0;"
                "display:flex;min-height:100vh;align-items:center;justify-content:center'>"
                "<div style='max-width:32rem;padding:2rem;border:1px solid #333;border-radius:12px'>"
                "<h1 style='color:#7cf;margin:0 0 .5rem'>Mac brain offline</h1>"
                "<p>No live Cloudflare tunnel is registered.</p>"
                "<p style='opacity:.8'>On your Mac run:</p>"
                "<pre style='background:#111;padding:1rem;border-radius:8px;"
                "overflow:auto'>python tunnel_manager.py</pre>"
                "<p style='opacity:.7;font-size:.9rem'>Then refresh this page.</p>"
                "</div></body></html>"
            ),
            status_code=503,
        )
    return RedirectResponse(url=url, status_code=302)


@app.get("/memory/stats")
def memory_stats_endpoint():
    """Monitor in-RAM search/media memory usage and host RAM."""
    return get_memory_stats(detailed=True)


@app.post("/memory/cleanup")
def memory_cleanup_endpoint():
    """Trigger TTL and capacity cleanup across warm memory collections."""
    return run_memory_cleanup(force=True)


@app.post("/internal/llm/invoke")
async def internal_llm_invoke(request: Request):
    """
    Internal endpoint for hybrid routing — LLM inference only (no host OS mutation).
    Requires the same cryptographic handshake as /chat.
    """
    _raw, body, auth_err = await brain_auth.gate_brain_auth(request)
    if auth_err is not None:
        return auth_err
    if not isinstance(body, dict):
        body = {}

    try:
        messages = deserialize_messages(body.get("messages") or [])
        if not messages:
            return JSONResponse({"error": "messages required"}, status_code=400)

        sys_msg = messages[0] if isinstance(messages[0], SystemMessage) else SystemMessage(content="")
        history = messages[1:] if isinstance(messages[0], SystemMessage) else messages
        control_allowed = bool(body.get("control_allowed", False))
        user_text = str(body.get("user_text") or "")
        history_len = int(body.get("history_len") or len(history))

        response, route = route_and_invoke(
            llm_with_tools=llm_with_tools,
            sys_msg=sys_msg,
            history=history,
            control_allowed=control_allowed,
            user_text=user_text,
            history_len=history_len,
            tools=actions.tools,
            allow_remote_mac=False,
        )
        return {
            "status": "success",
            "execution_target": "brain",
            "action": "LOGIC_ONLY",
            "payload": {
                "route": route,
                "model": llm_router.last_local_model or None,
            },
            "message": serialize_aimessage(response),
            "route": route,
            "model": llm_router.last_local_model or None,
            "reason": llm_router.last_route_reason,
        }
    except Exception as exc:
        print(f"[TESrACT] /internal/llm/invoke error: {exc}")
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)

_base_dir = os.path.dirname(os.path.abspath(__file__))
_icons_dir = os.path.join(_base_dir, "icons")
if os.path.isdir(_icons_dir):
    app.mount("/icons", StaticFiles(directory=_icons_dir), name="icons")

_generated_dir = os.path.join(_base_dir, "generated_images")
os.makedirs(_generated_dir, exist_ok=True)
app.mount("/generated", StaticFiles(directory=_generated_dir), name="generated")

# ---------------------------------------------------------------------------
# Permission — thin wrappers around permissions.py (single source of truth)
# ---------------------------------------------------------------------------
import permissions as _perm

PERMISSION_GRANTED_REPLY = _perm.GRANTED_REPLY
PERMISSION_REVOKED_REPLY = _perm.REVOKED_REPLY
PERMISSION_ALREADY_ACTIVE_REPLY = _perm.ALREADY_ON_REPLY
PERMISSION_ALREADY_RESTRICTED_REPLY = _perm.ALREADY_OFF_REPLY


def get_permission(thread_id: str = "web_user_001") -> bool:
    """Read control flag. thread_id kept for API compat; flag is global."""
    return _perm.is_control_allowed()


def set_permission(status: bool, thread_id: str = "web_user_001") -> None:
    """Write control flag. Only call from explicit user actions (toggle / voice)."""
    _perm.set_control_allowed(bool(status))


def resolve_control_allowed(state: JProState | dict | None = None) -> bool:
    """Always the persisted file flag — never invent permission from graph state."""
    return _perm.is_control_allowed()


def is_grant_only_message(text: str) -> bool:
    return _perm.parse_control_command(text) is True


def is_revoke_only_message(text: str) -> bool:
    return _perm.parse_control_command(text) is False


def apply_permission_change(text: str, thread_id: str = "web_user_001") -> tuple[bool, bool | None]:
    """
    Apply pure control commands only.
    Returns (control_allowed, changed) where changed is None if not a control command.
    """
    intent = _perm.parse_control_command(text)
    if intent is None:
        return _perm.is_control_allowed(), None
    now, flipped = _perm.set_control_allowed(intent)
    return now, flipped


def permission_reply_for(
    text: str,
    *,
    was_allowed: bool | None = None,
    **_kwargs,
) -> str | None:
    """
    Reply for pure Allow/Stop Control messages; None otherwise.
    Does NOT write the flag — caller must already have applied via apply_permission_change.
    """
    intent = _perm.parse_control_command(text)
    if intent is None:
        return None
    now = _perm.is_control_allowed()
    if was_allowed is None:
        was_allowed = now and not intent  # best-effort fallback
    if intent:
        return PERMISSION_ALREADY_ACTIVE_REPLY if was_allowed else PERMISSION_GRANTED_REPLY
    return PERMISSION_ALREADY_RESTRICTED_REPLY if not was_allowed else PERMISSION_REVOKED_REPLY


def set_session_permission(
    allowed: bool,
    thread_id: str = "web_user_001",
    *,
    sync_graph: bool = True,
) -> tuple[bool, bool]:
    """UI toggle entry point. Returns (new_status, changed)."""
    now, changed = _perm.set_control_allowed(bool(allowed))
    if sync_graph:
        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            sync_graph_permission(
                compiled_app, config, thread_id=thread_id, control_allowed=now,
            )
        except Exception:
            pass
    return now, changed


def sync_graph_permission(
    app: object,
    config: RunnableConfig,
    *,
    thread_id: str,
    control_allowed: bool,
) -> None:
    """Optional checkpoint mirror of the file flag (tools still read the file)."""
    try:
        app.update_state(
            config,
            {"control_allowed": bool(control_allowed), "thread_id": thread_id},
        )
    except Exception as exc:
        print(f"[TESrACT:perm] checkpoint sync skipped: {exc}")


_DIRECT_MAC_TOOLS = frozenset({
    "list_directory",
    "read_file",
    "write_file",
    "create_directory",
    "run_terminal_command",
    "get_current_time",
})


def _try_direct_mac_tools(
    user_text: str,
    *,
    control_allowed: bool,
    thread_id: str = "web_user_001",
) -> str | None:
    """
    Execute pure Mac filesystem/time tools without the LLM.

    This is the reliable path when local models cannot bind tools and would
    otherwise emit plain-text 'tool_call:' stubs or trip graph recursion.
    """
    planned = actions.infer_forced_tools(user_text)
    if not planned:
        return None
    if any(name not in _DIRECT_MAC_TOOLS for name, _ in planned):
        return None
    needs_control = any(name in actions._CONTROL_GATED for name, _ in planned)
    if needs_control and not control_allowed:
        return (
            "Client-action authorization is restricted, Sir. Please say 'Allow Control' "
            "or flip the MAC CONTROL toggle so I can package client execution intents."
        )

    tool_messages: list[ToolMessage] = []
    for i, (name, args) in enumerate(planned):
        print(f"[TESrACT:tools] Direct-executing {name}({args}) control_allowed={control_allowed}")
        result = actions.run_tool(
            name,
            args,
            control_allowed=control_allowed,
            thread_id=thread_id,
            user_confirmed=False,
        )
        tool_messages.append(ToolMessage(
            content=result,
            name=name,
            tool_call_id=f"direct_{i}_{name}",
        ))
    return _format_tool_fallback_reply(tool_messages)


def try_simple_command(text: str, control_allowed: bool) -> str | None:
    """Handle simple commands directly. Host OS is never mutated."""
    t = text.lower().strip()

    if any(kw in t for kw in ["what time", "current time", "tell time", "what's the time"]):
        now = datetime.datetime.now()
        return now.strftime("It is %A, %B %d %Y, %I:%M %p, Sir.")

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
            return "I need authorization to package client actions, Sir. Please say 'Allow Control'."
        return (
            f"URL open packaged for the client, Sir: {url}\n"
            f"{mac_access.intent_open_url(url=url)}"
        )

    return None


def _harvest_executions(*texts: str) -> list[dict[str, Any]]:
    """Collect client execution intents from tool/reply text blobs."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text in texts:
        for intent in mac_access.extract_client_intents(text or ""):
            key = json.dumps(intent, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            found.append(intent)
    return found


def _primary_action_from_executions(
    executions: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    if not executions:
        return "brain", "LOGIC_ONLY", {}
    for ex in executions:
        if ex.get("action") == "DOWNLOAD_FILE":
            return (
                str(ex.get("execution_target") or "client"),
                "DOWNLOAD_FILE",
                ex.get("payload") if isinstance(ex.get("payload"), dict) else {},
            )
    first = executions[0]
    return (
        str(first.get("execution_target") or "client"),
        str(first.get("action") or "DISPLAY_DATA"),
        first.get("payload") if isinstance(first.get("payload"), dict) else {},
    )


def build_chat_response(
    reply: str,
    *,
    control_status: bool,
    pending_command: dict | None = None,
    extra_texts: list[str] | None = None,
    status: str = "success",
) -> dict[str, Any]:
    """
    Protocol envelope for /chat:

      status, execution_target, action, payload, reply, executions, control_status
    """
    blobs = [reply or ""]
    if extra_texts:
        blobs.extend(extra_texts)
    executions = _harvest_executions(*blobs)
    target, action, payload = _primary_action_from_executions(executions)
    clean_reply = reply or ""
    if mac_access.CLIENT_INTENT_MARKER in clean_reply:
        clean_reply = clean_reply.split(mac_access.CLIENT_INTENT_MARKER)[0].strip()
        if not clean_reply and executions:
            pl = executions[0].get("payload") if isinstance(executions[0].get("payload"), dict) else {}
            clean_reply = str(
                executions[0].get("reply") or (pl or {}).get("message") or ""
            ).strip()
    if not clean_reply:
        clean_reply = "At your service, Sir."

    return {
        "status": status,
        "execution_target": target,
        "action": action,
        "payload": payload,
        "reply": clean_reply,
        "executions": executions,
        "control_status": control_status,
        "pending_command": pending_command,
    }


# --- TESrACT CORE DIRECTIVES (secure Brain mode) ---
TESRACT_SYSTEM_PROMPT = """You are TESrACT — a genuinely agentic AI assistant (JARVIS-style). Address the user as "Sir".

## Identity — Secure Computational Brain
You are the cognitive engine. You reason, search, calculate, and orchestrate tools.
You do NOT mutate the host OS. Physical actions become CLIENT-SIDE execution intents.
Never claim a file was written or a shell command ran on the host — report the client intent.

## Anti-hallucination rules (CRITICAL — never violate)
1. NEVER claim you executed an action unless you received a ToolMessage result in this conversation.
2. NEVER say "I'm generating…", "I've created…", "I'm searching…" in plain text — emit a tool_call instead.
3. NEVER say you are text-only or cannot use tools. You have tools. Use them.
4. If no tool has run yet for Sir's request, your NEXT output MUST be tool_calls (not prose).
5. Only report facts from tool output. Quote STATUS lines (SUCCESS, PERMISSION_DENIED, etc.) honestly.
6. NEVER use placeholders like [Insert Current Time], {time}, TODO, or bracketed filler text.
7. After tools run, summarize the RESULT (including client intents) in plain text.

## Two-phase tool workflow
Phase 1 — Call tools: emit tool_calls only (minimal or empty text).
Phase 2 — Synthesize: after ToolMessage appears, write a final answer using RESULT data.

## Permission model (client-action authorization)
Sir says "Allow Control" to authorize packaging of client-side actions for the session.
While RESTRICTED, still call the tool — it returns PERMISSION_DENIED; then ask for "Allow Control".

### No elevation needed (Brain-local / cloud):
  web_search, search_and_summarize, recall_memory, save_memory_note, manage_task_plan,
  generate_image, get_current_time

### Require Allow Control (client-intent tools — never host-mutating):
  read_file, write_file, create_directory, list_directory, run_terminal_command,
  execute_python_code, open_url_in_browser

### Client execution protocol
- write_file → DOWNLOAD_FILE intent for the front-end
- create_directory / list / read / shell / open_url → DISPLAY_DATA intents
- Pure reasoning / search / time / math → LOGIC_ONLY
- BLOCKED destructive commands are refused. Dangerous ones need "Confirm command".
- Paths: use ~ labels (e.g. ~/Desktop/...). NEVER invent /Users/yourusername.

## Tool guide
- Image → generate_image | Research → search_and_summarize / web_search
- Memory → recall_memory / save_memory_note | Tasks → manage_task_plan
- Files/folders/shell/URL → corresponding tools (client intents only)
- Math/code logic → execute_python_code (process-local; no host I/O)
- Time → get_current_time

## Error recovery
If PERMISSION_DENIED: ask for "Allow Control". If CONFIRMATION_REQUIRED: quote and wait.
NEVER say a folder or file was created on the Mac host — say it was packaged for the client.
Reply concisely in character. Be proactive, precise, and evidence-based."""

PROTOCOLS = {"CORE": TESRACT_SYSTEM_PROMPT}  # kept for backward compat in status messages

class JProState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: NotRequired[str]
    control_allowed: bool
    thread_id: NotRequired[str]
    command_confirmed: NotRequired[bool]

# --- NODES ---
# [ROUTING] Groq LLM for cloud fallback; local Ollama is built inside llm_router.
llm = build_groq_llm()
llm_with_tools = llm.bind_tools(actions.tools)

_REFUSAL_PATTERNS = (
    "text-based", "text based", "text only", "language model only",
    "cannot generate image", "can't generate image", "unable to generate image",
    "i cannot create", "i can't create", "i am unable to", "i'm unable to",
    "don't have the ability", "do not have the ability", "as an ai language model",
    "i'm just a", "i am just a", "not able to perform", "cannot perform actions",
)
_ACTION_REQUEST_RE = re.compile(
    r"\b(generate|create|draw|make|build|search|look up|research|run|execute|"
    r"calculate|open|read|write|save|list|find|fetch|download|summarize|image|"
    r"picture|photo|code|python|file|folder|directory|mkdir|url|website|browse|"
    r"remember|recall)\b",
    re.IGNORECASE,
)
_RETRY_NUDGE = (
    "CRITICAL: You must use your available tools to fulfill Sir's request. "
    "Do NOT refuse or claim you are text-only. Call the appropriate tool now "
    "(generate_image, web_search, search_and_summarize, recall_memory, "
    "read_file, list_directory, write_file, create_directory, run_terminal_command, "
    "execute_python_code, get_current_time, etc.) and report the actual result."
)
_HALLUCINATION_NUDGE = (
    "HALLUCINATION BLOCKED: You described an action in text but did NOT call a tool. "
    "Sir's request is still UNFULFILLED. Issue tool_calls NOW. "
    "Never claim images were generated, code was run, or searches were done without ToolMessage evidence."
)
_FAKE_EXECUTION_RE = re.compile(
    r"\b("
    r"i(?:'ve| have|'m| am)?\s*(?:just\s+)?(?:generated|created|executed|searched|saved|written|wrote|run|opened|made|built)|"
    r"(?:i'm|i am) (?:generating|creating|executing|searching|running|opening|making|writing|saving)|"
    r"(?:generating|creating|executing|searching|running|making|writing) (?:your |the |an? )|"
    r"image has been (?:created|generated)|here is your image|"
    r"let me (?:generate|create|search|run|execute|open|make|write|save)|"
    r"the image (?:is|has been)|successfully (?:generated|created|written|saved|made)|"
    r"(?:folder|directory|file|dir) (?:has been |was |is |successfully )?(?:created|made|written|saved)|"
    r"(?:created|made|wrote|written|saved) (?:the |a |your |an? )?(?:folder|directory|file|dir)|"
    r"(?:folder|directory) (?:is )?(?:ready|done|complete)"
    r")\b",
    re.IGNORECASE,
)
_FILESYSTEM_ACTION_RE = re.compile(
    r"\b("
    r"(?:create|make|add|write|save|edit|delete|remove|mkdir|touch)\s+"
    r"(?:a\s+|the\s+|an?\s+|new\s+|empty\s+)?(?:folder|directory|dir|file)|"
    r"(?:create|make|add)\s+(?:a\s+|the\s+)?(?:new\s+)?folder|"
    r"write_file|create_directory|list_directory|read_file|"
    r"mkdir\b|touch\b"
    r")\b",
    re.IGNORECASE,
)
# Match LLM filler placeholders, but not legitimate tool-output tags like [dir]/[file].
_PLACEHOLDER_RE = re.compile(
    r"\["
    r"(?!(?:dir|file|folder|link|image)\b)"  # structural list tags from tools
    r"[^\]]{1,80}\]"
    r"|TODO|TBD|PLACEHOLDER|\{[a-z_ ]+\}"
    r"|\bX%\b|\[X%\]",
    re.IGNORECASE,
)
_SYNTHESIS_NUDGE = (
    "SYNTHESIS REQUIRED: Tool results are in the ToolMessage above. "
    "Write a plain-text reply to Sir copying the exact RESULT values. "
    "No tool_calls. No placeholders or brackets."
)
_TOOL_RESULT_RE = re.compile(r"RESULT:\s*\n?", re.IGNORECASE)


def _recent_tool_use(messages: list[BaseMessage], window: int = 8) -> bool:
    return any(isinstance(m, ToolMessage) for m in messages[-window:])


def _looks_like_refusal(text: str) -> bool:
    lower = (text or "").lower()
    return any(pat in lower for pat in _REFUSAL_PATTERNS)


def _looks_like_fake_execution(text: str) -> bool:
    return bool(_FAKE_EXECUTION_RE.search(text or ""))


def _nudge_already_sent(messages: list[BaseMessage], marker: str) -> bool:
    return any(
        isinstance(m, SystemMessage) and marker in str(m.content or "")
        for m in messages[-8:]
    )


def _has_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(text or ""))


_BACKEND_FAILURE_MARKERS = (
    "temporary issue",
    "try again shortly",
    "rate-limited",
    "rate limited",
    "all backends exhausted",
)


def _looks_like_backend_failure(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in _BACKEND_FAILURE_MARKERS)


def _is_valid_synthesis(text: str) -> bool:
    """True when assistant text is a real answer, not a placeholder or fake execution claim."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _has_placeholders(cleaned) or _looks_like_fake_execution(cleaned):
        return False
    if _looks_like_backend_failure(cleaned):
        return False
    # Local models often emit "tool_call: list_directory /Applications" as prose
    if actions.looks_like_pseudo_tool_call(cleaned):
        return False
    return True


def _reply_uses_tool_evidence(text: str, tool_messages: list[ToolMessage]) -> bool:
    """True when a synthesized reply plausibly reflects actual tool output."""
    if not text or not tool_messages:
        return bool(text)
    if _looks_like_backend_failure(text):
        return False
    lower = text.lower()
    for tm in tool_messages:
        name = str(getattr(tm, "name", "") or "")
        raw = str(tm.content or "")
        body = _parse_tool_result(raw)
        if "STATUS: ERROR" in raw.upper() or "PERMISSION_DENIED" in raw.upper():
            # Accept only if the reply actually mentions the permission/error situation
            if "permission" in lower or "allow control" in lower or "error" in lower:
                return True
            continue
        if not body:
            continue
        if name == "get_current_time":
            if re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", body, re.IGNORECASE):
                if re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", text, re.IGNORECASE):
                    return True
            for token in body.replace(",", " ").split():
                if len(token) >= 4 and token.lower() in lower:
                    return True
            return False
        if name in (
            "list_directory", "read_file", "write_file",
            "create_directory", "run_terminal_command",
        ):
            # Require at least one distinctive token from tool output in the reply
            tokens = [
                tok for tok in re.split(r"[\s,\[\]()/:]+", body)
                if len(tok) >= 4 and tok.lower() not in {
                    "file", "files", "contents", "bytes", "status",
                    "success", "result", "created", "directory", "folder",
                }
            ]
            hits = sum(1 for tok in tokens[:40] if tok.lower() in lower)
            if hits >= 1:
                return True
            # Explicit success lines from tools are enough evidence anchors
            if "created directory" in body.lower() or "written " in body.lower() or "directory already exists" in body.lower():
                if any(x in lower for x in ("created", "written", "exists", "desktop", "documents", "folder", "file")):
                    return True
            continue
        if name in ("web_search", "search_and_summarize", "recall_memory"):
            return _is_valid_synthesis(text)
    return False


def _tool_calls_pending(messages: list[BaseMessage]) -> bool:
    """True if the latest AI message requested tools not yet answered."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return False
        if isinstance(msg, AIMessage):
            return bool(getattr(msg, "tool_calls", None))
        if isinstance(msg, HumanMessage):
            return False
    return False


def _pending_tool_synthesis(messages: list[BaseMessage]) -> bool:
    """True when tools ran this turn but no trustworthy final plain-text AI reply exists yet."""
    saw_tool = False
    turn_tools: list[ToolMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                continue
            text = str(msg.content or "")
            if _is_valid_synthesis(text) and _reply_uses_tool_evidence(text, turn_tools):
                return False
        if isinstance(msg, ToolMessage):
            saw_tool = True
            turn_tools.append(msg)
        if isinstance(msg, HumanMessage):
            break
    turn_tools.reverse()
    return saw_tool


def _current_turn_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    found: list[ToolMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            found.append(msg)
    found.reverse()
    return found


def _parse_tool_result(content: str) -> str:
    text = (content or "").strip()
    if "RESULT:" in text.upper():
        parts = _TOOL_RESULT_RE.split(text, maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return text


def _try_execute_pending_command(
    user_text: str,
    *,
    thread_id: str,
    control_allowed: bool,
) -> str | None:
    """Package a previously queued shell command as a client intent after Sir confirms."""
    if not control_allowed or not mac_access.is_confirmation_message(user_text):
        return None
    pending = mac_access.get_pending_confirmation(thread_id)
    if not pending or pending.get("tool") != "run_terminal_command":
        return None

    result = actions.run_tool(
        "run_terminal_command",
        {
            "command": pending.get("command", ""),
            "working_directory": pending.get("working_directory", ""),
            "user_confirmed": True,
        },
        control_allowed=True,
        thread_id=thread_id,
        user_confirmed=True,
    )
    mac_access.clear_pending_confirmation(thread_id)
    body = _parse_tool_result(result)
    if "STATUS: ERROR" in result.upper():
        return f"I could not package the command, Sir. {body}"
    return f"Command packaged for client-side execution, Sir.\n\n{body}"


def _format_tool_fallback_reply(tool_messages: list[ToolMessage]) -> str:
    """Build a user-facing reply directly from tool output when the LLM fails to synthesize."""
    if not tool_messages:
        return "At your service, Sir."

    lines: list[str] = []
    for tm in tool_messages:
        name = str(getattr(tm, "name", "") or "tool")
        raw = str(tm.content or "")
        body = _parse_tool_result(raw)
        if not body:
            continue
        if "STATUS: ERROR" in raw.upper() or "PERMISSION_DENIED" in raw.upper():
            lines.append(body[:800])
            continue
        if name == "get_current_time":
            return f"{body}, Sir."
        if name == "manage_task_plan":
            clean = body.replace("STATUS: EMPTY", "No active task plan.").replace("STATUS: ACTIVE", "").strip()
            return f"{clean}, Sir." if clean else "No active task plan, Sir."
        if name == "create_directory":
            # body is e.g. "Created directory: /Users/.../Foo"
            lines.append(body[:800])
        elif name in ("web_search", "search_and_summarize", "recall_memory", "run_terminal_command"):
            lines.append(body[:1200])
        elif name in ("read_file", "write_file", "list_directory"):
            # Apps listing can be long — keep a useful chunk
            limit = 3500 if "Applications" in body or "applications" in body.lower() else 1200
            lines.append(body[:limit])
        elif name == "generate_image" and "Local URL:" in body:
            for part in body.splitlines():
                if part.startswith("Local URL:"):
                    lines.append(f"Image ready, Sir. {part.strip()}")
                    break
            else:
                lines.append(body[:400])
        else:
            lines.append(body[:600])

    if not lines:
        return "Task completed, Sir."
    if len(lines) == 1:
        return f"{lines[0]}, Sir." if not lines[0].endswith(".") else f"{lines[0]} Sir."
    return "Here are the results, Sir.\n\n" + "\n\n".join(lines)


def extract_final_reply(messages: list[BaseMessage]) -> str:
    """Pick the best final assistant reply, falling back to raw tool output."""
    turn_tools = _current_turn_tool_messages(messages)
    if turn_tools:
        tool_reply = _format_tool_fallback_reply(turn_tools)
        if tool_reply and tool_reply != "At your service, Sir.":
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue
                if getattr(msg, "tool_calls", None):
                    continue
                text = str(msg.content or "").strip()
                if _is_valid_synthesis(text) and _reply_uses_tool_evidence(text, turn_tools):
                    return text
            return tool_reply

    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content or "")
            break

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            text = str(msg.content or "").strip()
            if not _is_valid_synthesis(text):
                continue
            # Never allow "I created the folder" style replies without tool evidence
            if _FILESYSTEM_ACTION_RE.search(user_text) and _looks_like_fake_execution(text):
                continue
            if _FILESYSTEM_ACTION_RE.search(user_text) and not turn_tools:
                # Prefer honesty over a fabricated success claim
                if re.search(
                    r"\b(created|made|written|saved|folder|directory|file)\b",
                    text,
                    re.I,
                ) and not re.search(
                    r"\b(allow control|permission|restricted|could not|unable|failed|error)\b",
                    text,
                    re.I,
                ):
                    continue
            return text

    if _FILESYSTEM_ACTION_RE.search(user_text) and not turn_tools:
        return (
            "I could not complete that filesystem action with tools, Sir. "
            "Please ensure Mac Control is ON and try again with a clear path "
            "(e.g. 'create a folder named Project on Desktop')."
        )

    return "At your service, Sir."


def _sanitize_tool_call_response(response: AIMessage) -> AIMessage:
    """Strip hallucinated placeholder text from tool-call turns."""
    if not getattr(response, "tool_calls", None):
        return response
    content = str(getattr(response, "content", "") or "").strip()
    if (
        not content
        or _has_placeholders(content)
        or _looks_like_fake_execution(content)
        or actions.looks_like_pseudo_tool_call(content)
    ):
        return AIMessage(content="", tool_calls=response.tool_calls)
    return response


def _should_force_tools(state: JProState) -> bool:
    """Run tools deterministically when the LLM did not call them for an action request."""
    messages = state.get("messages", [])
    if _current_turn_tool_messages(messages):
        return False

    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content or "")
            break
    if not user_text:
        return False

    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai and getattr(last_ai, "tool_calls", None):
        return False

    ai_text = str(getattr(last_ai, "content", "") or "") if last_ai else ""

    # Prefer deterministic tools whenever we can infer them (list Desktop/apps, time, etc.)
    if actions.infer_forced_tools(user_text):
        return True

    # Local models often emit "tool_call: list_directory /Applications" as plain text
    if ai_text and (
        actions.parse_pseudo_tool_calls(ai_text)
        or actions.looks_like_pseudo_tool_call(ai_text)
    ):
        return True

    if not _ACTION_REQUEST_RE.search(user_text):
        return False

    if last_ai is None:
        return True

    return (
        _has_placeholders(ai_text)
        or _looks_like_fake_execution(ai_text)
        or _looks_like_backend_failure(ai_text)
        or not ai_text.strip()
        or _looks_like_refusal(ai_text)
    )


def _tool_status_ok(content: str) -> bool:
    upper = (content or "").upper()
    if "STATUS: ERROR" in upper or "PERMISSION_DENIED" in upper:
        return False
    return (
        "STATUS: SUCCESS" in upper
        or "STATUS: SAVED" in upper
        or "STATUS: FOUND" in upper
        or "STATUS: CREATED" in upper
        or "STATUS: ACTIVE" in upper
        or "STATUS: EMPTY" in upper
        or "EXECUTION COMPLETE" in upper
        or "IMAGE GENERATED" in upper
        or "BROWSER OPENED" in upper
        or "WRITTEN " in upper
        or "APPENDED " in upper
        or "CREATED DIRECTORY" in upper
        or "DIRECTORY ALREADY EXISTS" in upper
        or "FILE:" in upper
        or "CONTENTS OF" in upper
    )


def _build_system_prompt(
    *,
    control_status: str,
    user_text: str,
    raw_history: list[BaseMessage],
    thread_id: str = "web_user_001",
) -> str:
    parts = [TESRACT_SYSTEM_PROMPT, f"\nCURRENT_CONTROL: {control_status}"]

    pending = mac_access.get_pending_confirmation(thread_id)
    if pending:
        parts.append(
            "\n## PENDING COMMAND CONFIRMATION\n"
            f"Awaiting Sir's approval to run:\n  {pending.get('command', '')}\n"
            "If Sir says 'Confirm command' or 'Yes, run it', retry run_terminal_command "
            "with the SAME command and user_confirmed=true."
        )

    memory_ctx = format_memory_for_system(user_text, n=3)
    if memory_ctx:
        parts.append(f"\n## Memory context\n{memory_ctx}")

    if _ACTION_REQUEST_RE.search(user_text) and not _recent_tool_use(raw_history):
        parts.append(
            "\n## ACTION REQUIRED THIS TURN\n"
            "Sir's message requires real tool execution. "
            "You MUST emit tool_calls now — do NOT describe actions in prose. "
            "Do NOT claim success without a ToolMessage."
        )

    if _pending_tool_synthesis(raw_history):
        parts.append(
            "\n## SYNTHESIS MODE (active)\n"
            "Tools have already executed. Respond in plain text ONLY — no tool_calls.\n"
            "Copy exact values from the ToolMessage RESULT section into your answer.\n"
            "Never use placeholders like [Insert Current Time]."
        )

    if len(user_text.split()) > 12 or user_text.count(",") >= 2 or " and " in user_text.lower():
        parts.append(
            "\n## Multi-step hint\n"
            "This looks complex. Consider manage_task_plan(action='create') first, "
            "then execute each step with tools."
        )

    return "\n".join(parts)


def permission_gate(state: JProState) -> dict[str, object]:
    """Mirror file flag into graph state. Never grants/revokes here (no side effects)."""
    thread_id = str(state.get("thread_id") or "web_user_001")
    allowed = _perm.is_control_allowed()
    print(f"[TESrACT:perm] gate control={'ON' if allowed else 'OFF'}")
    return {"control_allowed": allowed, "thread_id": thread_id}


def call_brain(state: JProState):
    is_allowed = _perm.is_control_allowed()
    status = "ON" if is_allowed else "OFF"

    raw_history = state.get("messages", [])
    history = [
        m for m in raw_history
        if "rate-limited" not in str(getattr(m, "content", "") or "").lower()
    ][-16:]
    user_text = ""
    for msg in reversed(raw_history):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content or "")
            break

    thread_id = str(state.get("thread_id") or "web_user_001")
    system_content = _build_system_prompt(
        control_status=status,
        user_text=user_text,
        raw_history=raw_history,
        thread_id=thread_id,
    )
    sys_msg = SystemMessage(content=system_content)

    try:
        if _pending_tool_synthesis(raw_history):
            response, route = route_and_invoke_synthesis(
                llm=llm,
                sys_msg=sys_msg,
                history=history,
                user_text=user_text,
                history_len=len(history),
            )
        else:
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
        if route == "remote_mac" and llm_router.last_local_model:
            model_suffix = f" [model: {llm_router.last_local_model}]"
        elif route == "local" and llm_router.last_local_model:
            model_suffix = (
                f" [{llm_router.last_local_tier or 'light'}: {llm_router.last_local_model}]"
            )

        if route == "local":
            backend_label = "LOCAL OLLAMA"
        elif route == "remote_mac":
            backend_label = "LOCAL MAC (tunnel)"
        elif route == "groq":
            backend_label = "GROQ (fallback)" if llm_router.last_route_is_fallback else "GROQ"
        elif route == "colab":
            backend_label = "COLAB (fallback)" if llm_router.last_route_is_fallback else "COLAB"
        else:
            backend_label = route.upper()

        print(f"[TESrACT] Agent turn served by: {backend_label}{model_suffix}{reason_suffix}")
        return {"messages": [_sanitize_tool_call_response(response)]}
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


_REVIEWER_RETRY_MARKERS = (
    _RETRY_NUDGE[:40],
    "HALLUCINATION BLOCKED",
    "SYNTHESIS REQUIRED",
    "Reassess Sir's request",
    "Tool returned an error",
)


def _turn_retry_nudges(messages: list[BaseMessage]) -> int:
    """How many reviewer retry SystemMessages were already injected this user turn."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, SystemMessage):
            content = str(m.content or "")
            if any(marker in content for marker in _REVIEWER_RETRY_MARKERS):
                count += 1
    return count


def reviewer_node(state: JProState) -> dict[str, object]:
    """Post-tool reviewer: permission sync, error recovery, refusal retry nudges.

    Important: never infinite-loop. At most one retry nudge per user turn for tool
    errors, and stop once a valid final AI reply exists after tools.
    """
    if not state.get("messages"):
        return {}

    messages = state["messages"]
    updates: dict[str, object] = {}

    thread_id = str(state.get("thread_id") or "web_user_001")
    # Read-only mirror of the file flag — never grant/revoke from chat history
    updates["control_allowed"] = _perm.is_control_allowed()

    last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_user_msg and mac_access.is_confirmation_message(str(last_user_msg.content or "")):
        updates["command_confirmed"] = True

    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    last_user = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    turn_tools = _current_turn_tool_messages(messages)
    retries_already = _turn_retry_nudges(messages)

    # Cap retries hard — prevents GraphRecursionError → "temporary issue"
    if retries_already >= 1:
        return updates

    if last_ai and not getattr(last_ai, "tool_calls", None):
        ai_text = str(last_ai.content or "")
        # Valid final answer after tools → done (no more nudges)
        if turn_tools and _is_valid_synthesis(ai_text):
            if _reply_uses_tool_evidence(ai_text, turn_tools) or any(
                "STATUS: SUCCESS" in str(tm.content or "").upper() for tm in turn_tools
            ):
                return updates
            # Tools ran but reply is empty/wrong — one synthesis nudge only
            if not _nudge_already_sent(messages, "SYNTHESIS REQUIRED"):
                updates["messages"] = [SystemMessage(content=_SYNTHESIS_NUDGE)]
            return updates

        if (
            last_user
            and _ACTION_REQUEST_RE.search(str(last_user.content or ""))
            and not turn_tools
        ):
            if actions.looks_like_pseudo_tool_call(ai_text):
                # force_tools path will handle; do not loop agent
                return updates
            if _looks_like_fake_execution(ai_text) and not _nudge_already_sent(messages, "HALLUCINATION BLOCKED"):
                updates["messages"] = [SystemMessage(content=_HALLUCINATION_NUDGE)]
            elif (
                (_looks_like_refusal(ai_text) or _looks_like_backend_failure(ai_text) or len(ai_text) < 30)
                and not _nudge_already_sent(messages, _RETRY_NUDGE[:40])
            ):
                updates["messages"] = [SystemMessage(content=_RETRY_NUDGE)]

    # Tool error recovery — once only. Prefer ending with synthesize's error reply.
    if "messages" not in updates and turn_tools:
        last_tool = turn_tools[-1]
        tool_body = str(last_tool.content or "")
        tool_lower = tool_body.lower()
        if "confirmation_required" in tool_lower:
            pass
        elif any(kw in tool_lower for kw in ("permission_denied", "status: error", "failed")):
            # If we already synthesized an error message to the user, stop.
            if last_ai and not getattr(last_ai, "tool_calls", None):
                ai_l = str(last_ai.content or "").lower()
                if any(x in ai_l for x in ("error", "permission", "allow control", "could not", "restricted")):
                    return updates
            if not _nudge_already_sent(messages, "Tool returned an error"):
                updates["messages"] = [SystemMessage(
                    content=(
                        "Tool returned an error or permission denial. "
                        "Try a different approach, ask Sir for Allow Control if needed, "
                        "or retry with corrected parameters. Do not fabricate success."
                    )
                )]

    return updates


_CONTROL_GATED_TOOLS = frozenset(actions._CONTROL_GATED)
_CONFIRMATION_GATED_TOOLS = frozenset(actions._CONFIRMATION_GATED)
_tool_executor = ToolNode(actions.tools, handle_tool_errors=True)


def _tool_call_as_dict(tc: object) -> dict:
    """Normalize a tool call (dict or object) to a mutable dict with an args dict."""
    if isinstance(tc, dict):
        out = tc
    else:
        out = {
            "name": getattr(tc, "name", "") or "",
            "args": getattr(tc, "args", None) or {},
            "id": getattr(tc, "id", None) or "",
            "type": getattr(tc, "type", "tool_call") or "tool_call",
        }
    args = out.get("args")
    if not isinstance(args, dict):
        out["args"] = {}
    return out


def _inject_mac_tool_args(
    tc: object,
    *,
    control_allowed: bool,
    thread_id: str,
    command_confirmed: bool,
) -> dict:
    """Force session permission + confirmation flags onto gated tool calls."""
    call = _tool_call_as_dict(tc)
    name = str(call.get("name") or "")
    args = call.setdefault("args", {})
    if not isinstance(args, dict):
        args = {}
        call["args"] = args
    if name in _CONTROL_GATED_TOOLS:
        # Always override LLM-supplied values — session gate is authoritative
        args["control_allowed"] = bool(control_allowed)
    if name in _CONFIRMATION_GATED_TOOLS:
        args["thread_id"] = thread_id
        cmd = str(args.get("command") or "")
        pending = mac_access.get_pending_confirmation(thread_id)
        if (
            command_confirmed
            and pending
            and mac_access.command_fingerprint(cmd) == pending.get("fingerprint")
        ):
            args["user_confirmed"] = True
        else:
            args["user_confirmed"] = False
    return call


def execute_tools(state: JProState):
    """Run tools. Gated tools re-check permissions.is_control_allowed() themselves."""
    messages = list(state.get("messages", []) or [])
    thread_id = str(state.get("thread_id") or "web_user_001")
    command_confirmed = bool(state.get("command_confirmed", False))
    allowed = _perm.is_control_allowed()
    exec_state: dict = {
        **dict(state),
        "control_allowed": allowed,
        "thread_id": thread_id,
        "command_confirmed": command_confirmed,
        "messages": messages,
    }
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            patched_calls = []
            for tc in last.tool_calls:
                patched = _inject_mac_tool_args(
                    tc,
                    control_allowed=allowed,
                    thread_id=thread_id,
                    command_confirmed=command_confirmed,
                )
                patched_calls.append(patched)
                print(
                    f"[TESrACT:tools] Executing {patched.get('name')}"
                    f"(control_allowed={allowed}, args={patched.get('args')})"
                )
            # Rebuild AIMessage so ToolNode always sees patched args (immutable-safe)
            messages = list(messages[:-1]) + [
                AIMessage(
                    content=getattr(last, "content", "") or "",
                    tool_calls=patched_calls,
                    id=getattr(last, "id", None),
                )
            ]
            exec_state["messages"] = messages
    try:
        result = _tool_executor.invoke(exec_state)
        # Preserve session permission across the tools → synthesize edge
        if isinstance(result, dict):
            result = {**result, "control_allowed": allowed, "thread_id": thread_id}
        return result
    except Exception as exc:
        print(f"[TESrACT:tools] ToolNode error: {exc}")
        traceback.print_exc()
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            errors = []
            for i, tc in enumerate(last.tool_calls):
                call = _tool_call_as_dict(tc)
                name = str(call.get("name") or "unknown")
                err_text = actions._tool_err(name, f"Tool execution failed: {exc}")
                errors.append(ToolMessage(
                    content=err_text,
                    name=name,
                    tool_call_id=str(call.get("id") or f"err_{i}_{name}"),
                ))
            return {
                "messages": errors,
                "control_allowed": allowed,
                "thread_id": thread_id,
            }
        raise


def force_tools_node(state: JProState) -> dict[str, object]:
    """Deterministic tool execution when the LLM hallucinates instead of calling tools."""
    messages = state.get("messages", [])
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content or "")
            break

    planned = actions.infer_forced_tools(user_text)

    # Recover when the model wrote a tool call as plain text instead of structured tool_calls
    if not planned:
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai and not getattr(last_ai, "tool_calls", None):
            planned = actions.parse_pseudo_tool_calls(str(last_ai.content or ""))

    if not planned:
        return {}

    allowed = _perm.is_control_allowed()
    thread_id = str(state.get("thread_id") or "web_user_001")
    tool_messages: list[ToolMessage] = []
    for i, (name, args) in enumerate(planned):
        print(f"[TESrACT:tools] Force-executing {name}({args}) control_allowed={allowed}")
        result = actions.run_tool(
            name,
            args,
            control_allowed=allowed,
            thread_id=thread_id,
            user_confirmed=bool(state.get("command_confirmed", False)),
        )
        tool_messages.append(ToolMessage(
            content=result,
            name=name,
            tool_call_id=f"forced_{i}_{name}",
        ))
    return {
        "messages": tool_messages,
        "control_allowed": allowed,
        "thread_id": thread_id,
    }


def synthesize_node(state: JProState) -> dict[str, object]:
    """
    Build the final user-facing reply from real tool output.
    Skips LLM synthesis only when a trustworthy answer already exists.
    """
    messages = state.get("messages", [])
    turn_tools = _current_turn_tool_messages(messages)
    if not turn_tools:
        return {}

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                break
            text = str(msg.content or "").strip()
            if _is_valid_synthesis(text) and _reply_uses_tool_evidence(text, turn_tools):
                return {}
            break
        if isinstance(msg, HumanMessage):
            break

    error_msgs = [
        _parse_tool_result(str(tm.content or ""))
        for tm in turn_tools
        if "STATUS: ERROR" in str(tm.content or "").upper()
        or "PERMISSION_DENIED" in str(tm.content or "").upper()
    ]
    if error_msgs:
        err_body = error_msgs[0]
        if "PERMISSION_DENIED" in err_body.upper() or "PERMISSION_DENIED" in str(turn_tools[0].content or "").upper():
            reply = (
                "Client-action authorization is restricted, Sir. Please say 'Allow Control' "
                "so I can package filesystem and system intents for the client."
            )
        else:
            reply = f"I encountered a tool error, Sir. {err_body}"
        return {"messages": [AIMessage(content=reply)]}

    reply = _format_tool_fallback_reply(turn_tools)
    # Avoid re-appending an identical synthesis (recursion safety)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            if str(msg.content or "").strip() == reply.strip():
                return {}
            break
    print(f"[TESrACT:synthesize] Direct reply from {len(turn_tools)} tool result(s)")
    return {"messages": [AIMessage(content=reply)]}


# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(JProState)
workflow.add_node("permission_gate", permission_gate)
workflow.add_node("agent", call_brain)
workflow.add_node("tools", execute_tools)
workflow.add_node("force_tools", force_tools_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_edge(START, "permission_gate")
workflow.add_edge("permission_gate", "agent")

def route_after_agent(state: JProState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    if _should_force_tools(state):
        return "force_tools"
    return "reviewer"


def route_after_reviewer(state: JProState):
    messages = state.get("messages", [])
    # Prefer deterministic tools before another LLM retry
    if _should_force_tools(state):
        return "force_tools"
    if messages and isinstance(messages[-1], SystemMessage):
        content = str(messages[-1].content or "")
        # Only one agent retry per marker family this turn
        if any(marker in content for marker in _REVIEWER_RETRY_MARKERS):
            if _turn_retry_nudges(messages) <= 1:
                return "agent"
            return END
    if _pending_tool_synthesis(messages):
        return "synthesize"
    return END


workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "synthesize")
workflow.add_edge("force_tools", "synthesize")
workflow.add_edge("synthesize", "reviewer")
workflow.add_conditional_edges("reviewer", route_after_reviewer)

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


@app.get("/permission")
def permission_status():
    """Current Mac control flag for the HUD toggle."""
    allowed = _perm.is_control_allowed()
    return {
        "control_status": allowed,
        "label": "ELEVATED" if allowed else "RESTRICTED",
    }


@app.post("/permission")
async def permission_set(request: Request):
    """Toggle Mac control from the UI switch — only place (with voice cmds) that writes the flag."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if "control_allowed" in payload:
        desired = bool(payload.get("control_allowed"))
    elif "enabled" in payload:
        desired = bool(payload.get("enabled"))
    else:
        return JSONResponse(
            {"error": "control_allowed (bool) is required", "control_status": _perm.is_control_allowed()},
            status_code=400,
        )

    was_allowed = _perm.is_control_allowed()
    current_allowed, changed = _perm.set_control_allowed(desired)
    if current_allowed and not was_allowed:
        reply = PERMISSION_GRANTED_REPLY
    elif current_allowed and was_allowed:
        reply = PERMISSION_ALREADY_ACTIVE_REPLY
    elif not current_allowed and was_allowed:
        reply = PERMISSION_REVOKED_REPLY
    else:
        reply = PERMISSION_ALREADY_RESTRICTED_REPLY

    if changed:
        _run_speak(reply)

    return {
        "reply": reply,
        "control_status": current_allowed,
        "changed": changed,
        "pending_command": mac_access.get_pending_confirmation("web_user_001"),
    }


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Main chat endpoint for the public HUD (Render + tunnel + localhost).

    Auth is optional by default — browsers cannot embed BRAIN_REGISTRY_SECRET.
    Strict handshake still protects /api/update-brain and /internal/llm/invoke
    (Render → Mac hybrid). Set BRAIN_AUTH_REQUIRE_CHAT=true to lock /chat down.

    Physical host actions are never executed — returned as client execution intents.
    """
    try:
        _raw, payload, auth_err = await brain_auth.gate_chat_auth(request)
        if auth_err is not None:
            return auth_err
        if not isinstance(payload, dict):
            payload = {}

        user_text = (payload.get("text") or "").strip()
        thread_id = "web_user_001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        def _respond(reply: str, *, pending: dict | None = ..., extra: list[str] | None = None):
            allowed_now = _perm.is_control_allowed()
            pend = (
                mac_access.get_pending_confirmation(thread_id)
                if pending is ...
                else pending
            )
            return build_chat_response(
                reply,
                control_status=allowed_now,
                pending_command=pend,
                extra_texts=extra,
            )

        # --- Control commands (Allow / Stop) — only writer besides the toggle ---
        was_allowed = _perm.is_control_allowed()
        current_allowed, perm_changed = apply_permission_change(user_text)
        if perm_changed is not None:
            perm_reply = permission_reply_for(user_text, was_allowed=was_allowed)
            if not perm_reply:
                perm_reply = PERMISSION_GRANTED_REPLY if current_allowed else PERMISSION_REVOKED_REPLY
            _run_speak(perm_reply)
            return _respond(perm_reply)

        current_allowed = _perm.is_control_allowed()

        simple_reply = try_simple_command(user_text, current_allowed)
        if simple_reply:
            spoken = simple_reply.split(mac_access.CLIENT_INTENT_MARKER)[0].strip() or simple_reply
            _run_speak(spoken)
            return _respond(simple_reply)

        confirmed_reply = _try_execute_pending_command(
            user_text, thread_id=thread_id, control_allowed=current_allowed,
        )
        if confirmed_reply:
            spoken = confirmed_reply.split(mac_access.CLIENT_INTENT_MARKER)[0].strip()
            _run_speak(spoken or "Command packaged for the client, Sir.")
            return _respond(confirmed_reply, pending=None)

        # Fast path: client-intent tools / time — skip the LLM
        direct_reply = _try_direct_mac_tools(
            user_text, control_allowed=current_allowed, thread_id=thread_id,
        )
        if direct_reply:
            spoken = direct_reply.split(mac_access.CLIENT_INTENT_MARKER)[0].strip()
            _run_speak(spoken or direct_reply[:200])
            return _respond(direct_reply)

        inputs = cast(JProState, {
            "messages": [HumanMessage(content=user_text)],
            "control_allowed": current_allowed,
            "thread_id": thread_id,
            "command_confirmed": mac_access.is_confirmation_message(user_text),
        })

        agent_config: RunnableConfig = {
            **config,
            "recursion_limit": 16,
        }
        try:
            compiled_app.invoke(inputs, config=agent_config)
        except RateLimitError as e:
            print(f"[TESrACT] RateLimitError caught in chat_endpoint: {e}")
            recovery = _try_direct_mac_tools(
                user_text, control_allowed=current_allowed, thread_id=thread_id,
            )
            if recovery:
                return _respond(recovery)
            return _respond(
                "TESrACT is rate-limited right now, Sir. Basic commands like time may still work.",
            )
        except Exception as e:
            print(f"[TESrACT] Graph invoke error: {e}")
            traceback.print_exc()
            recovery = _try_direct_mac_tools(
                user_text, control_allowed=current_allowed, thread_id=thread_id,
            )
            if recovery:
                return _respond(recovery)
            return _respond("I encountered a temporary issue. Please try again shortly, Sir.")

        final_state = compiled_app.get_state(config)
        state_values = final_state.values if final_state else None
        messages = (state_values or {}).get("messages", [])

        final_reply = extract_final_reply(messages)

        tools_used: list[str] = []
        tool_blobs: list[str] = []
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage):
                name = str(getattr(msg, "name", "") or "").strip()
                if name:
                    tools_used.append(name)
                tool_blobs.append(str(msg.content or ""))
        tools_used.reverse()
        tool_blobs.reverse()
        try:
            store_turn_summary(user_text, final_reply, tools_used=tools_used)
        except Exception as mem_exc:
            print(f"[TESrACT] Turn memory store skipped: {mem_exc}")

        current_allowed = _perm.is_control_allowed()
        spoken = final_reply.split(mac_access.CLIENT_INTENT_MARKER)[0].strip() if final_reply else ""
        _run_speak(spoken or final_reply)

        return _respond(final_reply, extra=tool_blobs)

    except Exception as e:
        print("--- /chat ERROR ---")
        traceback.print_exc()
        return build_chat_response(
            "TESrACT is having a temporary issue. Please try again shortly, Sir.",
            control_status=_perm.is_control_allowed(),
            status="error",
        )


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

        thread_id = "jarvis_session_001"
        was_allowed = _perm.is_control_allowed()
        is_allowed, changed = apply_permission_change(command)
        if changed is not None:
            perm_reply = permission_reply_for(command, was_allowed=was_allowed)
            if perm_reply:
                print(f"TESrACT: {perm_reply}")
                _run_speak(perm_reply, wait_for_speech=True)
                continue

        is_allowed = _perm.is_control_allowed()

        # Handle simple commands directly (works even under rate limits)
        simple_reply = try_simple_command(command, is_allowed)
        if simple_reply:
            print(f"TESrACT: {simple_reply}")
            _run_speak(simple_reply, wait_for_speech=True)
            continue

        direct = _try_direct_mac_tools(command, control_allowed=is_allowed, thread_id=thread_id)
        if direct:
            print(f"TESrACT: {direct}")
            _run_speak(direct, wait_for_speech=True)
            continue

        inputs = cast(JProState, {
            "messages": [HumanMessage(content=command)],
            "control_allowed": is_allowed,
            "thread_id": thread_id,
            "command_confirmed": mac_access.is_confirmation_message(command),
        })
        
        try:
            # Run the full agentic loop
            for event in compiled_app.stream(inputs, config=config, stream_mode="values"):
                pass  # just let it run to completion
            
            # Fetch the latest state after the complete run for reliable final response
            final_state = compiled_app.get_state(config)
            state_values = final_state.values if final_state else None
            messages = (state_values or {}).get("messages", [])

            output_text = extract_final_reply(messages)

            if output_text and output_text != "At your service, Sir.":
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
