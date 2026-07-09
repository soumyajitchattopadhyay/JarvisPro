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
  ENABLE_HYBRID_ROUTING — true to prefer a remote Mac instance (tunnel URL) before cloud
  LOCAL_INSTANCE_URL    — fallback Mac TESrACT URL (static). Live URL comes from brain_registry
                          (tunnel_manager auto-updates via POST /api/update-brain).
  LOCAL_INSTANCE_API_KEY — optional shared secret for /internal/llm/invoke
  LOCAL_INSTANCE_HEALTH_TTL — cache remote /health seconds (default: 30)
  LOCAL_INSTANCE_TIMEOUT    — remote invoke timeout seconds (default: 90)
  BRAIN_REGISTRY_SECRET     — shared secret for tunnel_manager ↔ Render registry
  TESRACT_RENDER_URL        — stable Render URL used by tunnel_manager to register

Future local media hooks (not implemented yet — structure only):
  LOCAL_IMAGE_GEN_BACKEND  — e.g. mlx-stable-diffusion
  LOCAL_VIDEO_GEN_BACKEND  — e.g. mlx-video / comfyui-local
"""
from __future__ import annotations

import json
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

Route = Literal["local", "remote_mac", "groq", "colab"]
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
    "search for", "search the web", "look up", "web search", "research ",
    "find out", "latest news", "summarize", "search_and_summarize",
    "open ", "browse", "go to", "navigate to", "visit ",
    "execute", "run code", "run python", "use the tool", "calculate",
    "generate image", "create image", "draw ", "make an image", "make a picture",
    "generate a picture", "image of", "picture of", "generate_image",
    "read file", "write file", "save file", "list files", "list directory",
    "create folder", "create directory", "make folder", "make a folder",
    "create a folder", "mkdir", "create_directory",
    "read_file", "write_file", "list_directory", "run_terminal_command",
    "terminal", "shell command", "run command", "execute command",
    "ls ", "cd ", "git ", "npm ", "pip ", "brew ",
    "recall memory", "remember", "what did we", "last time", "previous",
    "recall_memory", "save_memory", "manage_task", "task plan", "multi-step",
    "get_current_time", "open_url", "allow control",
    "do it", "perform", "take action", "can you create", "can you generate",
    "can you search", "can you run", "help me build", "help me create",
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

ENABLE_HYBRID_ROUTING = os.getenv("ENABLE_HYBRID_ROUTING", "").strip().lower() in ("1", "true", "yes")
# Static env fallback — live URL comes from brain_registry (tunnel_manager updates).
LOCAL_INSTANCE_URL = (os.getenv("LOCAL_INSTANCE_URL") or "").rstrip("/")
LOCAL_INSTANCE_API_KEY = os.getenv("LOCAL_INSTANCE_API_KEY", "")
LOCAL_INSTANCE_HEALTH_TTL = int(os.getenv("LOCAL_INSTANCE_HEALTH_TTL", "30"))
LOCAL_INSTANCE_TIMEOUT = float(os.getenv("LOCAL_INSTANCE_TIMEOUT", "90"))

_ollama_health_cache: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}
_ollama_models_cache: dict[str, Any] = {
    "models": set(),
    "capabilities": {},
    "checked_at": 0.0,
}
_colab_health_cache: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}
_remote_mac_health_cache: dict[str, Any] = {
    "ok": False,
    "checked_at": 0.0,
    "info": {},
    "url": "",
}
_ollama_llm_cache: dict[str, Any] = {}
_mlx_loaded: bool = False

last_route_used: Route | None = None
last_route_reason: str = ""
last_route_is_fallback: bool = False
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
    actual_label = _route_label(actual, fallback=preferred != actual)
    if preferred == actual:
        _log(f"→ {actual_label} ({reason})")
    else:
        preferred_label = _route_label(preferred, fallback=False)
        _log(f"Preferred {preferred_label} → using {actual_label} ({reason})")


def _route_label(route: Route, *, fallback: bool) -> str:
    if route == "local":
        return "LOCAL OLLAMA"
    if route == "remote_mac":
        return "LOCAL MAC (tunnel)"
    if route == "groq":
        return "GROQ (fallback)" if fallback else "GROQ"
    if route == "colab":
        return "COLAB (fallback)" if fallback else "COLAB"
    return route.upper()


def _log_local(tier: LocalTier, model: str, reason: str) -> None:
    _log(f"→ LOCAL OLLAMA [{tier.upper()}] model={model} ({reason})")


def _log_remote_mac(tier: LocalTier, model: str, reason: str) -> None:
    _log(f"→ LOCAL MAC (tunnel) [{tier.upper()}] model={model or 'unknown'} ({reason})")


def _log_groq(reason: str, *, fallback: bool) -> None:
    _log(f"→ {_route_label('groq', fallback=fallback)} ({reason})")


def _log_colab(reason: str, *, fallback: bool) -> None:
    _log(f"→ {_route_label('colab', fallback=fallback)} ({reason})")


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


def _normalize_ollama_model_name(name: str) -> str:
    return (name or "").strip().lower().split(":", 1)[0]


def _refresh_ollama_models(force: bool = False) -> None:
    if not ollama_is_healthy(log_failure=False):
        _ollama_models_cache["models"] = set()
        _ollama_models_cache["capabilities"] = {}
        _ollama_models_cache["checked_at"] = time.time()
        return

    now = time.time()
    if not force and now - float(_ollama_models_cache["checked_at"]) < OLLAMA_HEALTH_TTL:
        return

    models: set[str] = set()
    capabilities: dict[str, set[str]] = {}
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if res.status_code == 200:
                for entry in res.json().get("models", []):
                    name = str(entry.get("name") or entry.get("model") or "").strip()
                    if not name:
                        continue
                    models.add(name)
                    caps = entry.get("capabilities") or (entry.get("details") or {}).get(
                        "capabilities"
                    ) or []
                    capabilities[name.lower()] = {str(c).lower() for c in caps}
    except Exception:
        models = set()
        capabilities = {}

    _ollama_models_cache["models"] = models
    _ollama_models_cache["capabilities"] = capabilities
    _ollama_models_cache["checked_at"] = now


def ollama_installed_models(force: bool = False) -> set[str]:
    """Return model names reported by Ollama /api/tags (cached)."""
    _refresh_ollama_models(force=force)
    cached = _ollama_models_cache.get("models", set())
    return set(cached) if isinstance(cached, set) else set()


def ollama_model_supports_tools(model: str) -> bool:
    """Best-effort check whether an installed Ollama model advertises tool support."""
    _refresh_ollama_models()
    caps_map = _ollama_models_cache.get("capabilities", {})
    if not isinstance(caps_map, dict):
        return False

    target = (model or "").strip().lower()
    caps = caps_map.get(target)
    if caps is None:
        base = _normalize_ollama_model_name(target)
        for name, model_caps in caps_map.items():
            if name == base or name.startswith(f"{base}:"):
                caps = model_caps
                break
    if caps and "tools" in caps:
        return True

    # Conservative fallback for common tool-capable families when capabilities are absent.
    base = _normalize_ollama_model_name(target)
    return base.startswith(("llama3", "qwen2", "mistral", "mixtral", "phi3", "command-r"))


def ollama_model_available(model: str, *, force: bool = False) -> bool:
    """True when the requested Ollama model (or same base name) is pulled locally."""
    target = (model or "").strip().lower()
    if not target:
        return False
    installed = {m.lower() for m in ollama_installed_models(force=force)}
    if target in installed:
        return True
    base = _normalize_ollama_model_name(target)
    return any(
        m == base or m.startswith(f"{base}:")
        for m in installed
    )


def _resolve_local_model(tier: LocalTier, *, needs_tools: bool = False) -> tuple[str | None, str]:
    """
    Pick the best local Ollama model for a tier.
    Light tasks may reuse the heavy model when the light model is not pulled.
    Tool tasks prefer any installed model that advertises tool support.
    """
    preferred = _model_for_tier(tier)
    candidates: list[str] = []
    for name in (preferred, OLLAMA_LIGHT_MODEL, OLLAMA_HEAVY_MODEL):
        if name not in candidates:
            candidates.append(name)

    if needs_tools:
        for installed in sorted(ollama_installed_models()):
            if installed not in candidates:
                candidates.append(installed)

    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen or not ollama_model_available(candidate):
            continue
        seen.add(key)
        if needs_tools and not ollama_model_supports_tools(candidate):
            continue
        if candidate == preferred:
            return preferred, f"tier={tier}"
        return candidate, f"tier={tier} using {candidate!r} (preferred {preferred!r} unavailable)"

    if needs_tools:
        return None, (
            f"no tool-capable Ollama model installed for tier={tier} "
            f"(wanted {preferred!r})"
        )

    return None, f"no local model available for tier={tier} (wanted {preferred!r})"


def _is_local_capacity_error(exc: Exception) -> bool:
    """Detect Ollama failures that indicate the task exceeds local model/RAM limits."""
    msg = str(exc).lower()
    markers = (
        "memory", "oom", "out of memory", "context length", "context window",
        "too many tokens", "exceed", "timeout", "timed out", "resource",
        "server busy", "model requires", "not found",
    )
    return any(marker in msg for marker in markers)


def colab_configured() -> bool:
    return bool(COLAB_LLM_URL)


def hybrid_routing_enabled() -> bool:
    return ENABLE_HYBRID_ROUTING


def get_local_instance_url() -> str:
    """
    Live Mac tunnel URL for hybrid routing.

    Prefer brain_registry (updated automatically by tunnel_manager every time
    Cloudflare issues a new trycloudflare.com link). Fall back to LOCAL_INSTANCE_URL env.
    """
    try:
        import brain_registry

        live = brain_registry.get_brain_url()
        if live:
            return live.rstrip("/")
    except Exception:
        pass
    return (LOCAL_INSTANCE_URL or "").rstrip("/")


def on_local_instance_url_changed(url: str) -> None:
    """Called by brain_registry when tunnel_manager posts a new URL — bust health cache."""
    _remote_mac_health_cache["ok"] = False
    _remote_mac_health_cache["checked_at"] = 0.0
    _remote_mac_health_cache["info"] = {}
    _log(f"Remote Mac URL updated → {(url or '')[:64]}")


def remote_mac_configured() -> bool:
    return bool(get_local_instance_url())


def remote_mac_is_healthy(force: bool = False, *, log_failure: bool = True) -> bool:
    """Ping live Mac tunnel /health — True when the Mac TESrACT instance can serve locally."""
    if not hybrid_routing_enabled() or not remote_mac_configured():
        return False

    base = get_local_instance_url()
    if not base:
        return False

    now = time.time()
    cached_url = str(_remote_mac_health_cache.get("url") or "")
    if (
        not force
        and cached_url == base
        and now - float(_remote_mac_health_cache["checked_at"]) < LOCAL_INSTANCE_HEALTH_TTL
    ):
        return bool(_remote_mac_health_cache["ok"])

    ok = False
    err_detail = ""
    info: dict[str, Any] = {}
    try:
        headers = {"ngrok-skip-browser-warning": "true"}
        if LOCAL_INSTANCE_API_KEY:
            headers["Authorization"] = f"Bearer {LOCAL_INSTANCE_API_KEY}"
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{base}/health", headers=headers)
            if res.status_code == 200:
                data = res.json()
                ok = data.get("status") == "ok" and bool(data.get("local_llm_available"))
                info = {
                    "memory_available": data.get("memory_available"),
                    "ram_available_gb": (data.get("system") or {}).get("ram_available_gb"),
                }
            else:
                err_detail = f"HTTP {res.status_code}"
    except Exception as exc:
        ok = False
        err_detail = str(exc)

    _remote_mac_health_cache["ok"] = ok
    _remote_mac_health_cache["checked_at"] = now
    _remote_mac_health_cache["info"] = info
    _remote_mac_health_cache["url"] = base
    if log_failure and not ok:
        _log(
            f"Remote Mac health check failed ({err_detail or 'unreachable'}) — "
            f"{base}"
        )
    return ok


def hybrid_status() -> dict[str, Any]:
    """Snapshot for /health and startup logs."""
    enabled = hybrid_routing_enabled()
    live_url = get_local_instance_url()
    configured = bool(live_url)
    healthy = (
        remote_mac_is_healthy(log_failure=False)
        if enabled and configured
        else False
    )
    info = _remote_mac_health_cache.get("info", {})
    ram = info.get("ram_available_gb") if isinstance(info, dict) else None
    return {
        "enabled": enabled,
        "local_instance_url": live_url or None,
        "local_instance_url_env": LOCAL_INSTANCE_URL or None,
        "remote_mac_healthy": healthy,
        "remote_ram_available_gb": ram,
        "direct_ollama_healthy": ollama_is_healthy(log_failure=False),
        "brain_registry": True,
    }


def serialize_message(msg: BaseMessage) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": getattr(msg, "type", "human"),
        "content": str(getattr(msg, "content", "") or ""),
    }
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        out["tool_calls"] = msg.tool_calls
    if isinstance(msg, ToolMessage):
        out["name"] = str(getattr(msg, "name", "") or "")
        out["tool_call_id"] = str(getattr(msg, "tool_call_id", "") or "")
    return out


def deserialize_message(data: dict[str, Any]) -> BaseMessage:
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    if msg_type == "system":
        return SystemMessage(content=content)
    if msg_type == "tool":
        return ToolMessage(
            content=content,
            name=str(data.get("name") or ""),
            tool_call_id=str(data.get("tool_call_id") or ""),
        )
    if msg_type == "ai":
        tool_calls = data.get("tool_calls")
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=content)
    return HumanMessage(content=content)


def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [serialize_message(m) for m in messages]


def deserialize_messages(data: list[dict[str, Any]]) -> list[BaseMessage]:
    return [deserialize_message(d) for d in data]


def serialize_aimessage(msg: AIMessage) -> dict[str, Any]:
    return serialize_message(msg)


def deserialize_aimessage(data: dict[str, Any]) -> AIMessage:
    result = deserialize_message(data)
    if isinstance(result, AIMessage):
        return result
    return AIMessage(content=str(getattr(result, "content", result)))


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
    installed = sorted(ollama_installed_models()) if ollama_is_healthy(log_failure=False) else []
    return {
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_healthy": ollama_is_healthy(log_failure=False),
        "light_model": OLLAMA_LIGHT_MODEL,
        "heavy_model": OLLAMA_HEAVY_MODEL,
        "light_model_available": ollama_model_available(OLLAMA_LIGHT_MODEL),
        "heavy_model_available": ollama_model_available(OLLAMA_HEAVY_MODEL),
        "installed_models": installed,
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
        timeout=OLLAMA_TIMEOUT,
        validate_model_on_init=False,
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


def _stamp_control_allowed(response: AIMessage, control_allowed: bool) -> AIMessage:
    """Best-effort stamp of session permission onto tool call args (ToolNode also injects)."""
    if not getattr(response, "tool_calls", None):
        return response
    for t in response.tool_calls:
        try:
            if isinstance(t, dict):
                args = t.get("args")
                if not isinstance(args, dict):
                    t["args"] = {}
                    args = t["args"]
                args["control_allowed"] = bool(control_allowed)
            else:
                args = getattr(t, "args", None)
                if isinstance(args, dict):
                    args["control_allowed"] = bool(control_allowed)
        except Exception:
            continue
    return response


def invoke_groq_with_tools(
    llm_with_tools: Runnable[Any, AIMessage],
    messages: list[BaseMessage],
    control_allowed: bool,
    max_retries: int = 3,
) -> AIMessage:
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            return _stamp_control_allowed(response, control_allowed)
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
            return _stamp_control_allowed(response, control_allowed)
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
) -> tuple[AIMessage, str] | None:
    if not ollama_is_healthy():
        _log(f"Local Ollama unavailable ({reason})")
        return None

    model, model_note = _resolve_local_model(tier, needs_tools=needs_tools)
    if not model:
        _log(f"Local Ollama has no suitable model ({model_note})")
        if needs_tools:
            _log("No tool-capable local model — will try Groq (fallback)")
        return None

    try:
        llm = build_ollama_llm(model)
        _log(f"Invoking LOCAL OLLAMA [{tier}] model={model} ({model_note}; {reason})…")

        if needs_tools and tools:
            llm_tools = llm.bind_tools(tools)
            result = invoke_ollama_with_tools(
                llm_tools, messages, control_allowed, model=model,
            )
        else:
            result = llm.invoke(messages)
            if not isinstance(result, AIMessage):
                result = AIMessage(content=str(getattr(result, "content", result)))

        _log(f"LOCAL OLLAMA [{tier}] response received ✓ model={model}")
        return result, model
    except Exception as exc:
        capacity = _is_local_capacity_error(exc)
        detail = "task too heavy for local model" if capacity and tier == "heavy" else str(exc)
        _log(f"LOCAL OLLAMA [{tier}] failed ({detail})")
        if needs_tools:
            _log("Local tool calling failed — will try Groq (fallback) for tool support")
        elif capacity and tier == "heavy":
            _log("Heavy task exceeded local capacity — will try Colab/Groq (fallback)")
        return None


def _try_colab(messages: list[BaseMessage], *, reason: str, fallback: bool = True) -> AIMessage | None:
    if not colab_configured():
        _log(f"Colab unavailable ({reason}): COLAB_LLM_URL not set")
        return None
    if not colab_is_healthy():
        _log(f"Colab unavailable ({reason}): GPU uplink unhealthy")
        return None
    try:
        _log_colab(reason, fallback=fallback)
        result = invoke_colab(messages)
        _log("COLAB response received ✓")
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
    fallback: bool = True,
) -> AIMessage:
    _log_groq(reason, fallback=fallback)
    result = invoke_groq_with_tools(llm_with_tools, messages, control_allowed)
    _log("GROQ response received ✓")
    return result


def _try_remote_mac(
    messages: list[BaseMessage],
    control_allowed: bool,
    *,
    user_text: str,
    history_len: int,
    tier: LocalTier,
    needs_tools: bool,
    reason: str,
) -> AIMessage | None:
    global last_local_model
    if not hybrid_routing_enabled() or not remote_mac_configured():
        return None
    if not remote_mac_is_healthy():
        _log(f"Remote Mac unavailable ({reason}) — will try cloud fallback")
        return None

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    payload = {
        "messages": serialize_messages(messages),
        "control_allowed": control_allowed,
        "user_text": user_text,
        "history_len": history_len,
        "needs_tools": needs_tools,
        "local_tier": tier,
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    # Cryptographic handshake with the remote Brain
    try:
        import brain_auth

        headers.update(
            brain_auth.sign(
                method="POST",
                path="/internal/llm/invoke",
                body=body_bytes,
            )
        )
    except Exception:
        secret = (
            os.getenv("BRAIN_REGISTRY_SECRET")
            or os.getenv("LOCAL_INSTANCE_API_KEY")
            or LOCAL_INSTANCE_API_KEY
            or ""
        ).strip()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"

    base = get_local_instance_url()
    if not base:
        return None

    try:
        _log_remote_mac(tier, "", f"{reason} — {base}")
        with httpx.Client(timeout=LOCAL_INSTANCE_TIMEOUT) as client:
            res = client.post(
                f"{base}/internal/llm/invoke",
                content=body_bytes,
                headers=headers,
            )
        if res.status_code >= 400:
            _log(f"Remote Mac invoke failed HTTP {res.status_code}: {res.text[:300]}")
            return None
        data = res.json()
        msg = deserialize_aimessage(data.get("message", {}))
        model = data.get("model") or ""
        if model:
            last_local_model = model
        _log(f"Remote Mac response received ✓ model={model}")
        return msg
    except Exception as exc:
        _log(f"Remote Mac invoke failed ({reason}): {exc}")
        return None


def _cloud_fallback_order(tier: LocalTier, needs_tools: bool) -> list[Route]:
    """Order of cloud backends after local/MLX failure."""
    if ROUTING_MODE == "colab_fallback" and tier == "heavy":
        return ["colab", "groq"]
    if needs_tools:
        return ["groq", "colab"]
    if tier == "heavy":
        return ["colab", "groq"]
    return ["groq", "colab"]


def _tool_results_for_synthesis(history: list[BaseMessage]) -> str:
    """Extract RESULT bodies from recent ToolMessages for synthesis context."""
    blocks: list[str] = []
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            name = str(getattr(msg, "name", "") or "tool")
            content = str(getattr(msg, "content", "") or "")
            upper = content.upper()
            if "RESULT:" in upper:
                body = content[upper.index("RESULT:") + len("RESULT:"):].strip()
            else:
                body = content.strip()
            if body:
                blocks.append(f"### {name}\n{body}")
    blocks.reverse()
    return "\n\n".join(blocks)


def route_and_invoke_synthesis(
    *,
    llm: Runnable[Any, AIMessage],
    sys_msg: SystemMessage,
    history: list[BaseMessage],
    user_text: str,
    history_len: int,
) -> tuple[AIMessage, Route]:
    """
    Text-only synthesis pass after tools have run — never binds tools.
    Prefers local Ollama light, then Groq.
    """
    global last_route_used, last_route_reason, last_route_is_fallback, last_local_model, last_local_tier

    tool_block = _tool_results_for_synthesis(history)
    synthesis_sys = sys_msg
    if tool_block:
        synthesis_sys = SystemMessage(
            content=(
                f"{sys_msg.content}\n\n"
                "## Tool results (copy exact values into your reply — no placeholders)\n"
                f"{tool_block}"
            )
        )
    full_messages = [synthesis_sys] + history
    last_route_is_fallback = False
    preview = (user_text[:60] + "…") if len(user_text) > 60 else (user_text or "(empty)")
    _log(f"Synthesis pass (text-only) | history={history_len} | \"{preview}\"")

    if ollama_is_healthy(log_failure=False):
        model, _note = _resolve_local_model("light", needs_tools=False)
        if model:
            try:
                ollama_llm = build_ollama_llm(model)
                result = ollama_llm.invoke(full_messages)
                if not isinstance(result, AIMessage):
                    result = AIMessage(content=str(getattr(result, "content", result)))
                last_route_used = "local"
                last_route_reason = "tool result synthesis"
                last_local_tier = "light"
                last_local_model = model
                _log(f"Synthesis via LOCAL OLLAMA model={model} ✓")
                return result, "local"
            except Exception as exc:
                _log(f"Synthesis local failed ({exc}) — trying Groq")

    _log_groq("tool result synthesis", fallback=True)
    result = llm.invoke(full_messages)
    if not isinstance(result, AIMessage):
        result = AIMessage(content=str(getattr(result, "content", result)))
    last_route_used = "groq"
    last_route_reason = "tool result synthesis"
    last_route_is_fallback = True
    last_local_tier = None
    last_local_model = ""
    return result, "groq"


def route_and_invoke(
    *,
    llm_with_tools: Runnable[Any, AIMessage],
    sys_msg: SystemMessage,
    history: list[BaseMessage],
    control_allowed: bool,
    user_text: str,
    history_len: int,
    tools: list[Any] | None = None,
    allow_remote_mac: bool = True,
) -> tuple[AIMessage, Route]:
    """
    Local-first routing with automatic cloud fallback.

    Local path uses tiered Ollama models in unified RAM; MLX is optional for
    heavy text-only tasks. Groq retains tool calling when local models cannot
    bind tools. Colab serves deepest inference when local RAM is insufficient.
    """
    global last_route_used, last_route_reason, last_route_is_fallback
    global last_local_tier, last_local_model

    last_route_is_fallback = False
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
        colab_result = _try_colab(
            full_messages, reason=classification.reason, fallback=False,
        )
        if colab_result is not None:
            _log_switch(preferred, "colab", classification.reason)
            last_route_used = "colab"
            last_route_reason = classification.reason
            last_local_tier = None
            last_local_model = ""
            return colab_result, "colab"
        _log("Colab unavailable — falling back to Groq (fallback)")
        result = _try_groq(
            llm_with_tools, full_messages, control_allowed,
            reason="Colab unavailable", fallback=True,
        )
        last_route_used = "groq"
        last_route_reason = "Colab unavailable — Groq fallback"
        last_route_is_fallback = True
        return result, "groq"

    if preferred == "groq":
        try:
            result = _try_groq(
                llm_with_tools, full_messages, control_allowed,
                reason=classification.reason, fallback=False,
            )
            last_route_used = "groq"
            last_route_reason = classification.reason
            last_local_tier = None
            last_local_model = ""
            return result, "groq"
        except RateLimitError:
            colab_result = _try_colab(
                full_messages, reason="Groq rate limited", fallback=True,
            )
            if colab_result is not None:
                last_route_used = "colab"
                last_route_reason = "Groq rate limited — Colab fallback"
                last_route_is_fallback = True
                return colab_result, "colab"
            raise
        except Exception as exc:
            colab_result = _try_colab(
                full_messages, reason=f"Groq error — {exc}", fallback=True,
            )
            if colab_result is not None:
                last_route_used = "colab"
                last_route_reason = f"Groq failed — Colab fallback ({exc})"
                last_route_is_fallback = True
                return colab_result, "colab"
            raise

    # --- Local-first path (auto / local mode) ---
    if _uses_local_first() and preferred == "local":
        tier = classification.local_tier

        # 1) Direct Ollama on this host (Mac with unified RAM)
        local_result = None
        if ollama_is_healthy(log_failure=False):
            local_result = _try_local(
                tier,
                full_messages,
                control_allowed,
                reason=classification.reason,
                tools=tools,
                needs_tools=classification.needs_tools,
            )
        if local_result is not None:
            result, model = local_result
            _log_local(tier, model, classification.reason)
            last_route_used = "local"
            last_route_reason = classification.reason
            last_local_tier = tier
            last_local_model = model
            return result, "local"

        # 2) Remote Mac via tunnel (Render / cloud entry → home Mac instance)
        if allow_remote_mac and hybrid_routing_enabled():
            remote_result = _try_remote_mac(
                full_messages,
                control_allowed,
                user_text=user_text,
                history_len=history_len,
                tier=tier,
                needs_tools=classification.needs_tools,
                reason=classification.reason,
            )
            if remote_result is not None:
                model = last_local_model or _model_for_tier(tier)
                last_route_used = "remote_mac"
                last_route_reason = f"{classification.reason} — remote Mac tunnel"
                last_local_tier = tier
                if not last_local_model:
                    last_local_model = model
                return remote_result, "remote_mac"

        # 3) MLX secondary local path (text only, no tools)
        if not classification.needs_tools and tier == "heavy":
            mlx_result = _try_mlx(full_messages, reason=classification.reason)
            if mlx_result is not None:
                last_route_used = "local"
                last_route_reason = f"{classification.reason} — MLX fallback"
                last_local_tier = tier
                last_local_model = MLX_MODEL_ID
                return mlx_result, "local"

        # 4) Cloud fallbacks (Groq / Colab)
        if allow_remote_mac and hybrid_routing_enabled():
            _log("Local Mac unavailable — falling back to Groq/Colab")
        elif not ollama_is_healthy(log_failure=False):
            _log("Ollama not running — falling back to Groq/Colab")
        else:
            _log("Local Ollama could not serve task — falling back to Groq/Colab")

        fallback_chain = _cloud_fallback_order(tier, classification.needs_tools)
        for backend in fallback_chain:
            if backend == "groq":
                try:
                    fb_reason = f"{classification.reason} — local unavailable"
                    result = _try_groq(
                        llm_with_tools, full_messages, control_allowed,
                        reason=fb_reason, fallback=True,
                    )
                    _log_switch("local", "groq", fb_reason)
                    last_route_used = "groq"
                    last_route_reason = fb_reason
                    last_route_is_fallback = True
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
                colab_result = _try_colab(
                    [note] + history, reason="local unavailable", fallback=True,
                )
                if colab_result is not None:
                    fb_reason = f"{classification.reason} — local unavailable"
                    _log_switch("local", "colab", fb_reason)
                    last_route_used = "colab"
                    last_route_reason = fb_reason
                    last_route_is_fallback = True
                    last_local_tier = tier
                    last_local_model = ""
                    return colab_result, "colab"

        raise RuntimeError("All backends exhausted (local, MLX, Groq, Colab)")

    # Safety net — should not reach here in normal configs
    result = _try_groq(
        llm_with_tools, full_messages, control_allowed,
        reason="default Groq", fallback=False,
    )
    last_route_used = "groq"
    last_route_reason = "default Groq"
    return result, "groq"