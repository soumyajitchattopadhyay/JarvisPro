"""
TESrACT LLM router — Groq (fast + tools) vs Colab T4 (heavy inference).

Environment variables:
  GROQ_API_KEY          — required for Groq path
  COLAB_LLM_URL         — ngrok URL from Colab (e.g. https://abc123.ngrok-free.app)
  COLAB_LLM_API_KEY     — must match TESRACT_API_KEY in Colab
  LLM_ROUTING_MODE      — auto | groq | colab | colab_fallback (default: auto)
  COLAB_LLM_TIMEOUT     — seconds (default: 120)
  COLAB_HEALTH_TTL      — cache health check seconds (default: 60)
"""
from __future__ import annotations

import os
import time
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

import httpx
from groq import RateLimitError
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq

Route = Literal["groq", "colab"]

HEAVY_KEYWORDS = (
    "analyze", "analysis", "explain in detail", "research", "summarize", "summary",
    "compare", "comparison", "write code", "implement", "architecture", "design",
    "step by step", "comprehensive", "in depth", "detailed", "essay", "report",
    "debug", "refactor", "optimize", "long form", "think hard", "deep think",
    "multi-step", "break down", "pros and cons", "evaluate",
)

ROUTING_MODE = os.getenv("LLM_ROUTING_MODE", "auto").strip().lower()
COLAB_LLM_URL = (os.getenv("COLAB_LLM_URL") or "").rstrip("/")
COLAB_LLM_API_KEY = os.getenv("COLAB_LLM_API_KEY", "")
COLAB_TIMEOUT = float(os.getenv("COLAB_LLM_TIMEOUT", "120"))
COLAB_HEALTH_TTL = int(os.getenv("COLAB_HEALTH_TTL", "60"))

_health_cache: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}

# Set after every route_and_invoke call — useful for tests and debugging.
last_route_used: Route | None = None


def _log(msg: str) -> None:
    """Structured routing log — prefix makes backend switches easy to grep."""
    print(f"[TESrACT:router] {msg}")


def _log_switch(preferred: Route, actual: Route, reason: str) -> None:
    if preferred == actual:
        _log(f"→ {actual.upper()} ({reason})")
    else:
        _log(f"Preferred {preferred.upper()} → using {actual.upper()} ({reason})")


def _last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human" or msg.__class__.__name__ == "HumanMessage":
            return str(getattr(msg, "content", "") or "")
    return ""


def classify_task(text: str, history_len: int = 0) -> Route:
    """Return 'colab' for heavy tasks, 'groq' for light/tool-friendly tasks."""
    t = (text or "").strip().lower()
    if not t:
        return "groq"

    if any(kw in t for kw in ("use colab", "heavy model", "gpu model", "deep analysis")):
        return "colab"
    if len(text) > 500:
        return "colab"
    if history_len > 10:
        return "colab"
    if any(kw in t for kw in HEAVY_KEYWORDS):
        return "colab"
    return "groq"


def choose_route(user_text: str, history_len: int = 0, force_colab: bool = False) -> Route:
    if force_colab:
        return "colab"
    if ROUTING_MODE == "groq":
        return "groq"
    if ROUTING_MODE == "colab":
        return "colab"
    if ROUTING_MODE == "colab_fallback":
        return "groq"
    # auto
    return classify_task(user_text, history_len)


def colab_configured() -> bool:
    return bool(COLAB_LLM_URL)


def colab_is_healthy(force: bool = False, *, log_failure: bool = True) -> bool:
    if not colab_configured():
        return False
    now = time.time()
    if not force and now - float(_health_cache["checked_at"]) < COLAB_HEALTH_TTL:
        return bool(_health_cache["ok"])

    ok = False
    err_detail = ""
    try:
        headers = {"ngrok-skip-browser-warning": "true"}
        if COLAB_LLM_API_KEY:
            headers["Authorization"] = f"Bearer {COLAB_LLM_API_KEY}"
        with httpx.Client(timeout=8.0) as client:
            res = client.get(f"{COLAB_LLM_URL}/health", headers=headers)
            ok = res.status_code == 200
            if not ok:
                err_detail = f"HTTP {res.status_code}"
    except Exception as exc:
        ok = False
        err_detail = str(exc)

    _health_cache["ok"] = ok
    _health_cache["checked_at"] = now
    if log_failure and not ok:
        _log(f"Colab health check failed ({err_detail or 'unreachable'}) — {COLAB_LLM_URL}")
    return ok


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    out: list[dict[str, str]] = []
    for msg in messages:
        role = role_map.get(getattr(msg, "type", ""), "user")
        if isinstance(msg, SystemMessage):
            role = "system"
        content = str(getattr(msg, "content", "") or "")
        if content:
            out.append({"role": role, "content": content})
    return out


def colab_chat(messages: list[BaseMessage], temperature: float = 0.2, max_tokens: int = 768) -> str:
    if not colab_configured():
        raise RuntimeError("COLAB_LLM_URL is not set")

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if COLAB_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {COLAB_LLM_API_KEY}"

    payload = {
        "messages": _to_openai_messages(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    with httpx.Client(timeout=COLAB_TIMEOUT) as client:
        res = client.post(
            f"{COLAB_LLM_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        if res.status_code >= 400:
            raise RuntimeError(f"Colab LLM HTTP {res.status_code}: {res.text[:300]}")
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()


def build_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)


def invoke_groq_with_tools(
    llm_with_tools: ChatGroq,
    messages: list[BaseMessage],
    control_allowed: bool,
    max_retries: int = 3,
) -> AIMessage:
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            if response.tool_calls:
                for t in response.tool_calls:
                    t["args"]["control_allowed"] = control_allowed
            return response
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                _log(f"Groq rate limit — retry in {wait}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                _log(f"Groq error — retry ({attempt + 1}): {exc}")
                time.sleep(1)
                continue
            raise
    raise RuntimeError("Groq invoke failed after retries")


def invoke_colab(messages: list[BaseMessage]) -> AIMessage:
    if not colab_is_healthy():
        raise RuntimeError("Colab LLM endpoint is unreachable")
    text = colab_chat(messages)
    return AIMessage(content=text)


def _try_colab(
    messages: list[BaseMessage],
    *,
    reason: str,
) -> AIMessage | None:
    """Attempt Colab inference; return None (with log) when uplink is unavailable."""
    if not colab_configured():
        _log(f"Colab unavailable ({reason}): COLAB_LLM_URL not set")
        return None
    if not colab_is_healthy():
        _log(f"Colab unavailable ({reason}): GPU uplink unhealthy")
        return None
    try:
        _log(f"Invoking Colab T4 ({reason})…")
        result = invoke_colab(messages)
        _log("Colab response received ✓")
        return result
    except Exception as exc:
        _log(f"Colab invocation failed ({reason}): {exc}")
        return None


def _try_groq(
    llm_with_tools: ChatGroq,
    messages: list[BaseMessage],
    control_allowed: bool,
    *,
    reason: str,
) -> AIMessage:
    _log(f"Invoking Groq ({reason})…")
    result = invoke_groq_with_tools(llm_with_tools, messages, control_allowed)
    _log("Groq response received ✓")
    return result


def route_and_invoke(
    *,
    llm_with_tools: ChatGroq,
    sys_msg: SystemMessage,
    history: list[BaseMessage],
    control_allowed: bool,
    user_text: str,
    history_len: int,
) -> tuple[AIMessage, Route]:
    """
    Pick Groq or Colab, with automatic cross-backend fallback.
    Colab path does not use tool calling (text completion only).
    The agent keeps running as long as at least one backend is reachable.
    """
    global last_route_used

    preferred = choose_route(user_text, history_len)
    full_messages = [sys_msg] + history
    preview = (user_text[:60] + "…") if len(user_text) > 60 else (user_text or "(empty)")
    _log(f"Classified task → {preferred.upper()} | mode={ROUTING_MODE} | history={history_len} | \"{preview}\"")

    # --- Preferred Colab path (heavy tasks or forced mode) ---
    if preferred == "colab":
        colab_result = _try_colab(full_messages, reason="preferred route")
        if colab_result is not None:
            _log_switch(preferred, "colab", "heavy / forced task")
            last_route_used = "colab"
            return colab_result, "colab"
        _log_switch(preferred, "groq", "Colab unavailable — continuing on Groq")

    # --- Groq path (default, light tasks, or Colab unavailable) ---
    groq_reason = "light task" if preferred == "groq" else "Colab unavailable — fallback"
    try:
        result = _try_groq(llm_with_tools, full_messages, control_allowed, reason=groq_reason)
        _log_switch(preferred, "groq", groq_reason)
        last_route_used = "groq"
        return result, "groq"
    except RateLimitError as exc:
        _log(f"Groq rate limited ({exc})")
        if colab_configured() and colab_is_healthy(force=True):
            try:
                note = SystemMessage(
                    content=sys_msg.content
                    + "\n\nNote: Running on Colab GPU — tools unavailable this turn. Answer from knowledge only."
                )
                _log("Falling back to Colab after Groq rate limit…")
                result = invoke_colab([note] + history)
                _log_switch(preferred, "colab", "Groq rate limited — Colab fallback")
                last_route_used = "colab"
                return result, "colab"
            except Exception as colab_exc:
                _log(f"Colab fallback after rate limit failed ({colab_exc})")
        elif colab_configured():
            _log("Colab fallback skipped — GPU uplink unhealthy")
        raise
    except Exception as exc:
        _log(f"Groq error ({exc})")
        if ROUTING_MODE in ("colab_fallback", "auto"):
            colab_result = _try_colab(full_messages, reason="Groq error — fallback")
            if colab_result is not None:
                _log_switch(preferred, "colab", f"Groq failed — Colab fallback")
                last_route_used = "colab"
                return colab_result, "colab"
        raise