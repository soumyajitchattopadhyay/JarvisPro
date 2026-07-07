"""
TESrACT LLM router — local-first (Ollama / MLX on Apple Silicon) with cloud fallback.

Priority (LLM_ROUTING_MODE=auto, default):
  1. Ollama on Mac unified RAM — light models for simple/tool tasks, heavy for analysis
  2. Groq — fast cloud + reliable tool calling when local is down or unsupported
  3. Colab T4 — deepest inference when local RAM is insufficient

Environment variables:
  OLLAMA_BASE_URL       — default http://localhost:11434
  OLLAMA_LIGHT_MODEL    — default llama3.2:3b (simple Q&A, web search, tools)
  OLLAMA_HEAVY_MODEL    — default gemma2:9b (analysis, long-form)
  OLLAMA_TIMEOUT        — seconds (default: 90)
  OLLAMA_HEALTH_TTL     — cache health check seconds (default: 30)
  MLX_ENABLED           — 1|true to enable Apple Silicon MLX text path (optional)
  MLX_MODEL_ID          — HuggingFace repo or local path for mlx-lm
  GROQ_API_KEY          — Groq cloud fallback
  COLAB_LLM_URL         — ngrok URL from Colab
  COLAB_LLM_API_KEY     — must match TESRACT_API_KEY in Colab
  LLM_ROUTING_MODE      — auto | local | groq | colab | colab_fallback
  COLAB_LLM_TIMEOUT     — seconds (default: 120)
  COLAB_HEALTH_TTL      — cache health check seconds (default: 60)

Future local media hooks (not implemented yet — structure only):
  LOCAL_IMAGE_GEN_BACKEND  — e.g. mlx-stable-diffusion
  LOCAL_VIDEO_GEN_BACKEND  — e.g. mlx-video / comfyui-local
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

Route = Literal["local", "groq", "colab"]
LocalTier = Literal["light", "heavy"]

# ---------------------------------------------------------------------------
# Future local media backends (stubs for later MLX / SD integration)
# ---------------------------------------------------------------------------
LOCAL_IMAGE_GEN_BACKEND = os.getenv("LOCAL_IMAGE_GEN_BACKEND", "")
LOCAL_VIDEO_GEN_BACKEND = os.getenv("LOCAL_VIDEO_GEN_BACKEND", "")

# Strong signals — route to heavy local tier (then Colab/Groq if local fails).
STRONG_HEAVY_PHRASES = (
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

MODERATE_HEAVY_PHRASES = (
    "analyze", "analysis", "explain in detail", "architecture",
    "refactor", "multi-step", "multi step", "break down",
    "comprehensive", "thorough", "detailed", "essay", "report",
    "summary", "evaluate", "compare", "contrast", "reasoning",
    "white paper", "case study", "brainstorm", "outline",
    "draft a", "compose a", "write about", "long analysis",
    "complex system", "microservices", "distributed system",
)

_HEAVY_OUTPUT_RE = re.compile(
    r"("
    r"\b\d{2,5}\s*-?\s*words?\b"
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

_MIN_HEAVY_QUERY_WORDS = 18

TOOL_FRIENDLY_PHRASES = (
    "what time", "current time", "tell time", "what's the time",
    "search for", "search the web", "look up", "web search",
    "open ", "browse", "go to", "navigate to", "visit ",
    "execute", "run code", "run python", "use the tool",
    "get_current_time", "open_url", "allow control",
)

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
OLLAMA_BASE_URL = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
OLLAMA_LIGHT_MODEL = os.getenv("OLLAMA_LIGHT_MODEL", "llama3.2:3b")
OLLAMA_HEAVY_MODEL = os.getenv("OLLAMA_HEAVY_MODEL", "gemma2:9b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "90"))
OLLAMA_HEALTH_TTL = int(os.getenv("OLLAMA_HEALTH_TTL", "30"))
MLX_ENABLED = os.getenv("MLX_ENABLED", "").strip().lower() in ("1", "true", "yes")
MLX_MODEL_ID = os.getenv("MLX_MODEL_ID", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

COLAB_LLM_URL = (os.getenv("COLAB_LLM_URL") or "").rstrip("/")
COLAB_LLM_API_KEY = os.getenv("COLAB_LLM_API_KEY", "")
COLAB_TIMEOUT = float(os.getenv("COLAB_LLM_TIMEOUT", "120"))
COLAB_HEALTH_TTL = int(os.getenv("COLAB_HEALTH_TTL", "60"))

_ollama_health_cache: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}
_colab_health_cache: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}
_ollama_llm_cache: dict[str, Any] = {}
_mlx_loaded: bool = False

last_route_used: Route | None = None
last_route_reason: str = ""
last_local_tier: LocalTier | None = None
last_local_model: str = ""


@dataclass(frozen=True)
class Classification:
    route: Route
    reason: str
    local_tier: LocalTier = "light"
    needs_tools: bool = False
    score: int = 0
    signals: tuple[str, ...] = ()


def _log(msg: str) -> None:
    print(f"[TESrACT:router] {msg}")


def _log_switch(preferred: Route, actual: Route, reason: str) -> None:
    if preferred == actual:
        _log(f"→ {actual.upper()} ({reason})")
    else:
        _log(f"Preferred {preferred.upper()} → using {actual.upper()} ({reason})")


def _log_local(tier: LocalTier, model: str, reason: str) -> None:
    _log(f"→ LOCAL [{tier.upper()}] model={model} ({reason})")


def _uses_local_first() -> bool:
    return ROUTING_MODE in ("auto", "local", "colab_fallback")


def _query_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def classify_task(text: str, history_len: int = 0) -> Route:
    return classify_task_detailed(text, history_len).route


def classify_task_detailed(text: str, history_len: int = 0) -> Classification:
    """
    Score-based routing — local light for simple/tool tasks; local heavy for deep work.

    Cloud routes (groq/colab) are chosen only via LLM_ROUTING_MODE overrides or when
    local backends are unavailable at invoke time.
    """
    raw = (text or "").strip()
    t = raw.lower()
    if not t:
        return Classification("local", "empty prompt → local light", local_tier="light")

    signals: list[str] = []
    query_words = _query_word_count(raw)
    needs_tools = False

    for phrase in TOOL_FRIENDLY_PHRASES:
        if phrase in t:
            needs_tools = True
            reason = f"Local light: tool/action request ({phrase!r})"
            return Classification(
                "local",
                reason,
                local_tier="light",
                needs_tools=True,
                signals=(f"tool:{phrase}",),
            )

    if "use colab" in t:
        return Classification(
            "colab",
            "Explicit Colab request",
            local_tier="heavy",
            signals=("explicit_colab",),
        )

    for phrase in STRONG_HEAVY_PHRASES:
        if phrase in t:
            signals.append(f"strong:{phrase}")
            reason = f"Local heavy: deep task ({phrase!r})"
            return Classification(
                "local",
                reason,
                local_tier="heavy",
                score=10,
                signals=tuple(signals),
            )

    heavy_match = _HEAVY_OUTPUT_RE.search(raw)
    if heavy_match:
        matched = heavy_match.group(0)
        signals.append(f"pattern:{matched}")
        reason = f"Local heavy: long-form/output-size ({matched!r})"
        return Classification(
            "local",
            reason,
            local_tier="heavy",
            score=10,
            signals=tuple(signals),
        )

    score = 0
    moderate_hits = 0
    for phrase in MODERATE_HEAVY_PHRASES:
        if phrase in t:
            score += 2
            moderate_hits += 1
            signals.append(f"moderate:{phrase}")

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

    if query_words >= 45:
        score += 3
        signals.append(f"words>={query_words}")
    elif query_words >= _MIN_HEAVY_QUERY_WORDS:
        score += 2
        signals.append(f"words>={query_words}")
    elif query_words >= 12:
        score += 1
        signals.append(f"words>={query_words}")

    if history_len > 40 and moderate_hits >= 1:
        score += 2
        signals.append("history>40+moderate")
    elif history_len > 40:
        score += 1
        signals.append("history>40")
    elif history_len > 25 and moderate_hits >= 1:
        score += 1
        signals.append("history>25+moderate")

    heavy_threshold = 3 if moderate_hits >= 1 else 5
    if score >= heavy_threshold:
        reason = (
            f"Local heavy: analytical task (score {score}/{heavy_threshold}, "
            f"{query_words} words, {char_len} chars)"
        )
        return Classification(
            "local",
            reason,
            local_tier="heavy",
            score=score,
            signals=tuple(signals),
        )

    if char_len <= 200 and _SIMPLE_QUERY_RE.match(t):
        return Classification(
            "local",
            "Local light: short factual / conversational query",
            local_tier="light",
            score=score,
            signals=("simple_pattern", *signals),
        )

    if char_len <= 120 and score == 0:
        return Classification(
            "local",
            f"Local light: short message ({char_len} chars, {query_words} words)",
            local_tier="light",
            score=score,
            signals=("short_length",),
        )

    signal_summary = ", ".join(signals) if signals else "none"
    reason = (
        f"Local light: general task (score {score}/{heavy_threshold}, "
        f"{query_words} words, {char_len} chars) — signals: {signal_summary}"
    )
    return Classification(
        "local",
        reason,
        local_tier="light",
        score=score,
        signals=tuple(signals),
    )


def choose_route(user_text: str, history_len: int = 0, force_colab: bool = False) -> Route:
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
    if ROUTING_MODE == "local":
        base = classify_task_detailed(user_text, history_len)
        return Classification(
            "local",
            f"LLM_ROUTING_MODE=local — {base.reason}",
            local_tier=base.local_tier,
            needs_tools=base.needs_tools,
            score=base.score,
            signals=("mode_override", *base.signals),
        )
    if ROUTING_MODE == "colab_fallback":
        base = classify_task_detailed(user_text, history_len)
        if base.local_tier == "heavy":
            return Classification(
                "colab",
                f"LLM_ROUTING_MODE=colab_fallback (heavy → prefer Colab)",
                local_tier="heavy",
                signals=("mode_override", *base.signals),
            )
        return Classification(
            "groq",
            f"LLM_ROUTING_MODE=colab_fallback (light → prefer Groq)",
            local_tier="light",
            signals=("mode_override", *base.signals),
        )
    return classify_task_detailed(user_text, history_len)


def ollama_configured() -> bool:
    return bool(OLLAMA_BASE_URL)


def ollama_is_healthy(force: bool = False, *, log_failure: bool = True) -> bool:
    if not ollama_configured():
        return False
    now = time.time()
    if not force and now - float(_ollama_health_cache["checked_at"]) < OLLAMA_HEALTH_TTL:
        return bool(_ollama_health_cache["ok"])

    ok = False
    err_detail = ""
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ok = res.status_code == 200
            if not ok:
                err_detail = f"HTTP {res.status_code}"
    except Exception as exc:
        ok = False
        err_detail = str(exc)

    _ollama_health_cache["ok"] = ok
    _ollama_health_cache["checked_at"] = now
    if log_failure and not ok:
        _log(f"Ollama health check failed ({err_detail or 'unreachable'}) — {OLLAMA_BASE_URL}")
    return ok


def colab_configured() -> bool:
    return bool(COLAB_LLM_URL)


def colab_is_healthy(force: bool = False, *, log_failure: bool = True) -> bool:
    if not colab_configured():
        return False
    now = time.time()
    if not force and now - float(_colab_health_cache["checked_at"]) < COLAB_HEALTH_TTL:
        return bool(_colab_health_cache["ok"])

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

    _colab_health_cache["ok"] = ok
    _colab_health_cache["checked_at"] = now
    if log_failure and not ok:
        _log(f"Colab health check failed ({err_detail or 'unreachable'}) — {COLAB_LLM_URL}")
    return ok


def local_status() -> dict[str, Any]:
    """Snapshot for startup logs and /health."""
    return {
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_healthy": ollama_is_healthy(log_failure=False),
        "light_model": OLLAMA_LIGHT_MODEL,
        "heavy_model": OLLAMA_HEAVY_MODEL,
        "mlx_enabled": MLX_ENABLED,
        "mlx_model": MLX_MODEL_ID if MLX_ENABLED else None,
        "image_gen_backend": LOCAL_IMAGE_GEN_BACKEND or None,
        "video_gen_backend": LOCAL_VIDEO_GEN_BACKEND or None,
    }


def _model_for_tier(tier: LocalTier) -> str:
    return OLLAMA_HEAVY_MODEL if tier == "heavy" else OLLAMA_LIGHT_MODEL


def build_ollama_llm(model: str | None = None, temperature: float = 0.2):
    from langchain_ollama import ChatOllama

    model_name = model or OLLAMA_LIGHT_MODEL
    if model_name in _ollama_llm_cache:
        return _ollama_llm_cache[model_name]

    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=8192,
    )
    _ollama_llm_cache[model_name] = llm
    return llm


def build_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
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


def invoke_ollama_with_tools(
    llm_with_tools: Runnable[Any, AIMessage],
    messages: list[BaseMessage],
    control_allowed: bool,
    *,
    model: str,
    max_retries: int = 2,
) -> AIMessage:
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            if response.tool_calls:
                for t in response.tool_calls:
                    t["args"]["control_allowed"] = control_allowed
            return response
        except Exception as exc:
            if attempt < max_retries - 1:
                _log(f"Ollama tool invoke retry ({attempt + 1}) model={model}: {exc}")
                time.sleep(1)
                continue
            raise


def invoke_colab(messages: list[BaseMessage]) -> AIMessage:
    if not colab_is_healthy():
        raise RuntimeError("Colab LLM endpoint is unreachable")
    text = colab_chat(messages)
    return AIMessage(content=text)


def _try_mlx(messages: list[BaseMessage], *, reason: str) -> AIMessage | None:
    """
    Optional Apple Silicon MLX text path (no tool calling).
    Install: pip install mlx-lm
    """
    global _mlx_loaded
    if not MLX_ENABLED:
        return None
    try:
        from mlx_lm import generate, load

        if not _mlx_loaded:
            _log(f"Loading MLX model {MLX_MODEL_ID} into unified RAM…")
            _try_mlx._model, _try_mlx._tokenizer = load(MLX_MODEL_ID)
            _mlx_loaded = True

        prompt_parts: list[str] = []
        for msg in messages:
            role = getattr(msg, "type", "user")
            content = str(getattr(msg, "content", "") or "")
            if content:
                prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts) + "\nassistant:"

        _log(f"Invoking MLX ({reason})…")
        text = generate(
            _try_mlx._model,
            _try_mlx._tokenizer,
            prompt=prompt,
            max_tokens=768,
            verbose=False,
        )
        _log("MLX response received ✓")
        return AIMessage(content=str(text).strip())
    except ImportError:
        _log("MLX_ENABLED but mlx-lm not installed — pip install mlx-lm")
        return None
    except Exception as exc:
        _log(f"MLX invocation failed ({reason}): {exc}")
        return None


def _try_local(
    tier: LocalTier,
    messages: list[BaseMessage],
    control_allowed: bool,
    *,
    reason: str,
    tools: list[Any] | None,
    needs_tools: bool,
) -> AIMessage | None:
    if not ollama_is_healthy():
        _log(f"Local Ollama unavailable ({reason})")
        return None

    model = _model_for_tier(tier)
    try:
        llm = build_ollama_llm(model)
        _log(f"Invoking Ollama [{tier}] model={model} ({reason})…")

        if needs_tools and tools:
            llm_tools = llm.bind_tools(tools)
            result = invoke_ollama_with_tools(
                llm_tools, messages, control_allowed, model=model,
            )
        else:
            result = llm.invoke(messages)
            if not isinstance(result, AIMessage):
                result = AIMessage(content=str(getattr(result, "content", result)))

        _log(f"Ollama [{tier}] response received ✓ (RAM-backed local)")
        return result
    except Exception as exc:
        _log(f"Ollama [{tier}] failed ({reason}): {exc}")
        if needs_tools:
            _log("Local tool calling failed — will try Groq for tool support")
        return None


def _try_colab(messages: list[BaseMessage], *, reason: str) -> AIMessage | None:
    if not colab_configured():
        _log(f"Colab unavailable ({reason}): COLAB_LLM_URL not set")
        return None
    if not colab_is_healthy():
        _log(f"Colab unavailable ({reason}): GPU uplink unhealthy")
        return None
    try:
        _log(f"Invoking Colab T4 ({reason})…")
        result = invoke_colab(messages)
        _log("Colab response received ✓ (cloud GPU)")
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
    _log("Groq response received ✓ (cloud)")
    return result


def _cloud_fallback_order(tier: LocalTier, needs_tools: bool) -> list[Route]:
    """Order of cloud backends after local/MLX failure."""
    if ROUTING_MODE == "colab_fallback" and tier == "heavy":
        return ["colab", "groq"]
    if needs_tools:
        return ["groq", "colab"]
    if tier == "heavy":
        return ["colab", "groq"]
    return ["groq", "colab"]


def route_and_invoke(
    *,
    llm_with_tools: Runnable[Any, AIMessage],
    sys_msg: SystemMessage,
    history: list[BaseMessage],
    control_allowed: bool,
    user_text: str,
    history_len: int,
    tools: list[Any] | None = None,
) -> tuple[AIMessage, Route]:
    """
    Local-first routing with automatic cloud fallback.

    Local path uses tiered Ollama models in unified RAM; MLX is optional for
    heavy text-only tasks. Groq retains tool calling when local models cannot
    bind tools. Colab serves deepest inference when local RAM is insufficient.
    """
    global last_route_used, last_route_reason, last_local_tier, last_local_model

    classification = choose_route_detailed(user_text, history_len)
    preferred = classification.route
    full_messages = [sys_msg] + history
    preview = (user_text[:60] + "…") if len(user_text) > 60 else (user_text or "(empty)")
    signal_str = ", ".join(classification.signals) if classification.signals else "none"
    _log(
        f"Route decision → {preferred.upper()} | tier={classification.local_tier} | "
        f"tools={classification.needs_tools} | reason: {classification.reason} | "
        f"score={classification.score} | signals=[{signal_str}] | "
        f"mode={ROUTING_MODE} | history={history_len} | \"{preview}\""
    )

    # --- Explicit cloud-only modes ---
    if preferred == "colab":
        colab_result = _try_colab(full_messages, reason=classification.reason)
        if colab_result is not None:
            _log_switch(preferred, "colab", classification.reason)
            last_route_used = "colab"
            last_route_reason = classification.reason
            last_local_tier = None
            last_local_model = ""
            return colab_result, "colab"
        _log("Colab unavailable — falling back to Groq")
        result = _try_groq(llm_with_tools, full_messages, control_allowed, reason="Colab unavailable")
        last_route_used = "groq"
        last_route_reason = "Colab unavailable — Groq fallback"
        return result, "groq"

    if preferred == "groq":
        try:
            result = _try_groq(llm_with_tools, full_messages, control_allowed, reason=classification.reason)
            last_route_used = "groq"
            last_route_reason = classification.reason
            last_local_tier = None
            last_local_model = ""
            return result, "groq"
        except RateLimitError:
            colab_result = _try_colab(full_messages, reason="Groq rate limited")
            if colab_result is not None:
                last_route_used = "colab"
                last_route_reason = "Groq rate limited — Colab fallback"
                return colab_result, "colab"
            raise
        except Exception as exc:
            colab_result = _try_colab(full_messages, reason=f"Groq error — {exc}")
            if colab_result is not None:
                last_route_used = "colab"
                last_route_reason = f"Groq failed — Colab fallback ({exc})"
                return colab_result, "colab"
            raise

    # --- Local-first path (auto / local mode) ---
    if _uses_local_first() and preferred == "local":
        tier = classification.local_tier
        local_result = _try_local(
            tier,
            full_messages,
            control_allowed,
            reason=classification.reason,
            tools=tools,
            needs_tools=classification.needs_tools,
        )
        if local_result is not None:
            model = _model_for_tier(tier)
            _log_local(tier, model, classification.reason)
            last_route_used = "local"
            last_route_reason = classification.reason
            last_local_tier = tier
            last_local_model = model
            return local_result, "local"

        # MLX secondary local path (text only, no tools)
        if not classification.needs_tools and tier == "heavy":
            mlx_result = _try_mlx(full_messages, reason=classification.reason)
            if mlx_result is not None:
                last_route_used = "local"
                last_route_reason = f"{classification.reason} — MLX fallback"
                last_local_tier = tier
                last_local_model = MLX_MODEL_ID
                return mlx_result, "local"

        fallback_chain = _cloud_fallback_order(tier, classification.needs_tools)
        for backend in fallback_chain:
            if backend == "groq":
                try:
                    fb_reason = f"{classification.reason} — local unavailable, Groq fallback"
                    result = _try_groq(llm_with_tools, full_messages, control_allowed, reason=fb_reason)
                    _log_switch("local", "groq", fb_reason)
                    last_route_used = "groq"
                    last_route_reason = fb_reason
                    last_local_tier = tier
                    last_local_model = ""
                    return result, "groq"
                except RateLimitError as exc:
                    _log(f"Groq rate limited during local fallback ({exc})")
                    continue
                except Exception as exc:
                    _log(f"Groq failed during local fallback ({exc})")
                    continue
            if backend == "colab":
                note = sys_msg
                if classification.needs_tools:
                    note = SystemMessage(
                        content=str(sys_msg.content or "")
                        + "\n\nNote: Cloud fallback — tools may be unavailable this turn."
                    )
                colab_result = _try_colab([note] + history, reason="local unavailable")
                if colab_result is not None:
                    fb_reason = f"{classification.reason} — local unavailable, Colab fallback"
                    _log_switch("local", "colab", fb_reason)
                    last_route_used = "colab"
                    last_route_reason = fb_reason
                    last_local_tier = tier
                    last_local_model = ""
                    return colab_result, "colab"

        raise RuntimeError("All backends exhausted (local, MLX, Groq, Colab)")

    # Safety net — should not reach here in normal configs
    result = _try_groq(llm_with_tools, full_messages, control_allowed, reason="default Groq")
    last_route_used = "groq"
    last_route_reason = "default Groq"
    return result, "groq"