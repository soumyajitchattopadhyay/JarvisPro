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
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import Runnable

from dotenv import load_dotenv

load_dotenv()

import httpx
from groq import RateLimitError
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq

Route = Literal["groq", "colab"]

# Strong signals — one match is enough to route Colab.
STRONG_COLAB_PHRASES = (
    "use colab", "heavy model", "gpu model", "deep analysis",
    "analyze in detail", "comprehensive analysis", "step by step",
    "step-by-step", "in depth", "in-depth", "think hard", "deep think",
    "long form", "long-form", "pros and cons", "evaluate thoroughly",
    "research paper", "research summary", "literature review",
    "detailed report", "write an essay", "write a detailed",
    "write a comprehensive", "write a long", "multi-page",
    "system design", "architecture design", "design document",
    "design doc", "deep dive", "deep explanation",
    "complex explanation", "full implementation", "implement a complete",
    "build a complete", "code review", "technical report",
    "compare and contrast", "break it down", "walk me through",
)

# Moderate signals — contribute to heaviness score; combine with length/complexity.
MODERATE_COLAB_PHRASES = (
    "analyze", "analysis", "explain in detail", "architecture",
    "refactor", "multi-step", "multi step", "break down",
    "comprehensive", "thorough", "detailed", "essay", "report",
    "summary", "evaluate", "compare", "contrast", "reasoning",
    "white paper", "case study", "brainstorm", "outline",
    "draft a", "compose a", "write about", "long analysis",
    "complex system", "microservices", "distributed system",
)

# Regex patterns for output-size / writing intent (one match → Colab).
_HEAVY_OUTPUT_RE = re.compile(
    r"("
    r"\b\d{2,5}\s*-?\s*words?\b"  # "500-word", "1000 words"
    r"|\b(multi[- ]?page|several pages|page report)\b"
    r"|\bwrite\s+(a\s+)?(detailed|comprehensive|full|long|thorough)\b"
    r"|\b(draft|compose|create|produce)\s+(a\s+)?(detailed|comprehensive|full|long)\b"
    r"|\b(detailed|comprehensive|in[- ]depth|thorough)\s+"
    r"(analysis|essay|report|explanation|review|summary|guide|overview)\b"
    r"|\b(step[- ]by[- ]step|end[- ]to[- ]end)\s+"
    r"(guide|tutorial|explanation|analysis|implementation)\b"
    r")",
    re.IGNORECASE,
)

# Complexity hints in the query itself (word count, sentence count).
_MIN_HEAVY_QUERY_WORDS = 18

# Tool / action requests must stay on Groq (tool calling is Groq-only).
TOOL_FRIENDLY_PHRASES = (
    "what time", "current time", "tell time", "what's the time",
    "search for", "search the web", "look up", "web search",
    "open ", "browse", "go to", "navigate to", "visit ",
    "execute", "run code", "run python", "use the tool",
    "get_current_time", "open_url", "allow control",
)

# Short factual / conversational patterns → Groq.
_SIMPLE_QUERY_RE = re.compile(
    r"^("
    r"what (is|are|was|were)( the)?|who (is|was|are|were)|"
    r"when (did|was|is|were)|where (is|are|was|were)|"
    r"how (many|much|old|long|far|do|does|did|can|could|would|will)|"
    r"why (is|are|was|were|do|does|did)|"
    r"(hi|hello|hey|thanks|thank you|good morning|good evening|good night)|"
    r"tell me (the |a )?(time|date|weather|capital)|"
    r"define |meaning of |capital of "
    r")",
    re.IGNORECASE,
)

ROUTING_MODE = os.getenv("LLM_ROUTING_MODE", "auto").strip().lower()
COLAB_LLM_URL = (os.getenv("COLAB_LLM_URL") or "").rstrip("/")
COLAB_LLM_API_KEY = os.getenv("COLAB_LLM_API_KEY", "")
COLAB_TIMEOUT = float(os.getenv("COLAB_LLM_TIMEOUT", "120"))
COLAB_HEALTH_TTL = int(os.getenv("COLAB_HEALTH_TTL", "60"))

_health_cache: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}

# Set after every route_and_invoke call — useful for tests and debugging.
last_route_used: Route | None = None
last_route_reason: str = ""


@dataclass(frozen=True)
class Classification:
    route: Route
    reason: str
    score: int = 0
    signals: tuple[str, ...] = ()


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
    return classify_task_detailed(text, history_len).route


def _query_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def classify_task_detailed(text: str, history_len: int = 0) -> Classification:
    """
    Score-based routing — Groq for light/tool tasks; Colab for genuinely heavy work.

    Considers keywords, explicit output-size hints, query length, and complexity.
    Short factual queries, greetings, and tool-friendly requests always stay on Groq.
    """
    raw = (text or "").strip()
    t = raw.lower()
    if not t:
        return Classification("groq", "empty prompt → Groq", signals=("empty",))

    signals: list[str] = []
    query_words = _query_word_count(raw)

    # --- Groq: tool / action requests (tool calling is Groq-only) ---
    for phrase in TOOL_FRIENDLY_PHRASES:
        if phrase in t:
            reason = f"Groq: tool/action request ({phrase!r})"
            return Classification(
                "groq",
                reason,
                signals=(f"tool:{phrase}",),
            )

    # --- Colab: explicit heavy phrases ---
    for phrase in STRONG_COLAB_PHRASES:
        if phrase in t:
            signals.append(f"strong:{phrase}")
            reason = f"Colab: heavy phrase ({phrase!r})"
            return Classification(
                "colab",
                reason,
                score=10,
                signals=tuple(signals),
            )

    # --- Colab: writing / output-size patterns (e.g. "500-word essay") ---
    heavy_match = _HEAVY_OUTPUT_RE.search(raw)
    if heavy_match:
        matched = heavy_match.group(0)
        signals.append(f"pattern:{matched}")
        reason = f"Colab: long-form/output-size pattern ({matched!r})"
        return Classification(
            "colab",
            reason,
            score=10,
            signals=tuple(signals),
        )

    # --- Score accumulation for borderline cases ---
    score = 0
    moderate_hits = 0
    for phrase in MODERATE_COLAB_PHRASES:
        if phrase in t:
            score += 2
            moderate_hits += 1
            signals.append(f"moderate:{phrase}")

    # Query length (chars)
    char_len = len(raw)
    if char_len > 1200:
        score += 3
        signals.append("chars>1200")
    elif char_len > 600:
        score += 2
        signals.append("chars>600")
    elif char_len > 350:
        score += 1
        signals.append("chars>350")
    elif char_len > 200:
        score += 1
        signals.append("chars>200")

    # Query complexity (word count in the prompt itself)
    if query_words >= 45:
        score += 3
        signals.append(f"words>={query_words}")
    elif query_words >= _MIN_HEAVY_QUERY_WORDS:
        score += 2
        signals.append(f"words>={query_words}")
    elif query_words >= 12:
        score += 1
        signals.append(f"words>={query_words}")

    # Long threads nudge toward Colab when the task already looks analytical.
    if history_len > 40 and moderate_hits >= 1:
        score += 2
        signals.append("history>40+moderate")
    elif history_len > 40:
        score += 1
        signals.append("history>40")
    elif history_len > 25 and moderate_hits >= 1:
        score += 1
        signals.append("history>25+moderate")

    # Adaptive threshold: lower bar when analysis/writing keywords are present.
    colab_threshold = 3 if moderate_hits >= 1 else 5
    if score >= colab_threshold:
        reason = (
            f"Colab: heavy task (score {score}/{colab_threshold}, "
            f"{query_words} words, {char_len} chars)"
        )
        return Classification(
            "colab",
            reason,
            score=score,
            signals=tuple(signals),
        )

    # --- Groq: short factual / conversational ---
    if char_len <= 200 and _SIMPLE_QUERY_RE.match(t):
        return Classification(
            "groq",
            "Groq: short factual / conversational query",
            score=score,
            signals=("simple_pattern", *signals),
        )

    if char_len <= 120 and score == 0:
        return Classification(
            "groq",
            f"Groq: short message ({char_len} chars, {query_words} words)",
            score=score,
            signals=("short_length",),
        )

    signal_summary = ", ".join(signals) if signals else "none"
    reason = (
        f"Groq: light task (score {score}/{colab_threshold}, "
        f"{query_words} words, {char_len} chars) — signals: {signal_summary}"
    )
    return Classification("groq", reason, score=score, signals=tuple(signals))


def choose_route(
    user_text: str,
    history_len: int = 0,
    force_colab: bool = False,
) -> Route:
    return choose_route_detailed(user_text, history_len, force_colab).route


def choose_route_detailed(
    user_text: str,
    history_len: int = 0,
    force_colab: bool = False,
) -> Classification:
    if force_colab:
        return Classification("colab", "force_colab=True", signals=("forced",))
    if ROUTING_MODE == "groq":
        return Classification("groq", f"LLM_ROUTING_MODE={ROUTING_MODE}", signals=("mode_override",))
    if ROUTING_MODE == "colab":
        return Classification("colab", f"LLM_ROUTING_MODE={ROUTING_MODE}", signals=("mode_override",))
    if ROUTING_MODE == "colab_fallback":
        return Classification("groq", f"LLM_ROUTING_MODE={ROUTING_MODE} (prefer Groq)", signals=("mode_override",))
    return classify_task_detailed(user_text, history_len)


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
    llm_with_tools: Runnable[Any, AIMessage],
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
    llm_with_tools: Runnable[Any, AIMessage],
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
    llm_with_tools: Runnable[Any, AIMessage],
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
    global last_route_used, last_route_reason

    classification = choose_route_detailed(user_text, history_len)
    preferred = classification.route
    full_messages = [sys_msg] + history
    preview = (user_text[:60] + "…") if len(user_text) > 60 else (user_text or "(empty)")
    signal_str = ", ".join(classification.signals) if classification.signals else "none"
    _log(
        f"Route decision → {preferred.upper()} | reason: {classification.reason} | "
        f"score={classification.score} | signals=[{signal_str}] | "
        f"mode={ROUTING_MODE} | history={history_len} | \"{preview}\""
    )

    # --- Preferred Colab path (heavy tasks or forced mode) ---
    if preferred == "colab":
        colab_result = _try_colab(full_messages, reason=classification.reason)
        if colab_result is not None:
            switch_reason = f"{classification.reason} — Colab served"
            _log_switch(preferred, "colab", switch_reason)
            last_route_used = "colab"
            last_route_reason = switch_reason
            return colab_result, "colab"
        fallback_reason = f"{classification.reason} — Colab unavailable, falling back to Groq"
        _log_switch(preferred, "groq", fallback_reason)

    # --- Groq path (default, light tasks, or Colab unavailable) ---
    groq_reason = classification.reason if preferred == "groq" else f"{classification.reason} — Colab unavailable"
    try:
        result = _try_groq(llm_with_tools, full_messages, control_allowed, reason=groq_reason)
        _log_switch(preferred, "groq", groq_reason)
        last_route_used = "groq"
        last_route_reason = groq_reason
        return result, "groq"
    except RateLimitError as exc:
        _log(f"Groq rate limited ({exc})")
        if colab_configured() and colab_is_healthy(force=True):
            try:
                note = SystemMessage(
                    content=str(sys_msg.content or "")
                    + "\n\nNote: Running on Colab GPU — tools unavailable this turn. Answer from knowledge only."
                )
                _log("Falling back to Colab after Groq rate limit…")
                result = invoke_colab([note] + history)
                switch_reason = "Groq rate limited — Colab fallback"
                _log_switch(preferred, "colab", switch_reason)
                last_route_used = "colab"
                last_route_reason = switch_reason
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
                switch_reason = f"Groq failed — Colab fallback ({exc})"
                _log_switch(preferred, "colab", switch_reason)
                last_route_used = "colab"
                last_route_reason = switch_reason
                return colab_result, "colab"
        raise