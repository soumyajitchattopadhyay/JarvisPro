from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
import webbrowser
import datetime
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from duckduckgo_search import DDGS
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from local_search_memory import (
    format_recall_for_context,
    recall_memory,
    store_image_memory,
    store_note,
    store_search,
)
from mac_access import (
    classify_command,
    format_confirmation_request,
    list_path,
    read_text_file,
    resolve_mac_path,
    run_shell_command,
    set_pending_confirmation,
    write_text_file,
)

# Injected from graph state by ToolNode — hidden from the LLM tool schema.
ControlAllowed = Annotated[bool, InjectedState("control_allowed")]

repl = PythonREPL()
WORKSPACE_ROOT = Path(__file__).resolve().parent
GENERATED_DIR = WORKSPACE_ROOT / "generated_images"
AGENT_DATA_DIR = WORKSPACE_ROOT / "agent_data"
TASKS_FILE = AGENT_DATA_DIR / "active_task.json"

def _ensure_dirs() -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    AGENT_DATA_DIR.mkdir(exist_ok=True)


def _tool_ok(name: str, result: str) -> str:
    return f"TOOL: {name}\nSTATUS: SUCCESS\nRESULT:\n{result}"


def _tool_err(name: str, message: str) -> str:
    return f"TOOL: {name}\nSTATUS: ERROR\nRESULT:\n{message}"


def _permission_denied(tool_name: str, detail: str = "") -> str:
    msg = (
        "PERMISSION_DENIED: Mac access requires elevated control. "
        "Ask Sir to say 'Allow Control' once — permission stays active for the session."
    )
    if detail:
        msg = f"{msg} ({detail})"
    return _tool_err(tool_name, msg)


def _resolve_path(path: str) -> Path:
    """Resolve a Mac filesystem path (~, absolute, or relative to cwd)."""
    return resolve_mac_path(path)


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, str]]:
    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=max_results, backend="lite")
        return [
            {
                "title": r.get("title") or "Untitled",
                "body": r.get("body") or "",
                "url": r.get("href") or r.get("link") or "",
            }
            for r in hits
        ]


def _search_tavily(query: str, max_results: int = 5) -> tuple[str, list[dict[str, str]]]:
    """Optional Tavily API — set TAVILY_API_KEY in .env for richer results."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    with httpx.Client(timeout=25.0) as client:
        res = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
        res.raise_for_status()
        data = res.json()

    summary = (data.get("answer") or "").strip()
    hits: list[dict[str, str]] = []
    for hit in data.get("results") or []:
        hits.append({
            "title": hit.get("title") or "Untitled",
            "body": hit.get("content") or hit.get("snippet") or "",
            "url": hit.get("url") or "",
        })
    return summary, hits


def _format_search_results(
    query: str,
    hits: list[dict[str, str]],
    *,
    provider: str,
    summary: str = "",
    focus: str = "",
) -> str:
    lines: list[str] = [f"[Web search via {provider} — {len(hits)} source(s)]"]
    if focus:
        lines.append(f"Focus: {focus}")
    if summary:
        lines.append(f"\n## Summary\n{summary}")
    if hits:
        lines.append("\n## Sources")
        for i, hit in enumerate(hits, 1):
            title = hit.get("title") or "Untitled"
            body = (hit.get("body") or "").strip()
            url = hit.get("url") or ""
            snippet = body[:500] + ("…" if len(body) > 500 else "")
            lines.append(f"\n{i}. **{title}**")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   {snippet}")
    else:
        lines.append("\nNo results returned.")
    lines.append(
        "\n[Instruction: Synthesize the above into a concise answer for Sir. "
        "Cite sources by number when stating facts.]"
    )
    return "\n".join(lines)


def _run_web_search(query: str, *, max_results: int = 5, focus: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "Search Error: empty query."

    prior = format_recall_for_context(query)
    provider = "duckduckgo"
    summary = ""
    hits: list[dict[str, str]] = []

    try:
        if os.getenv("TAVILY_API_KEY", "").strip():
            provider = "tavily"
            summary, hits = _search_tavily(query, max_results=max_results)
        else:
            hits = _search_duckduckgo(query, max_results=max_results)
    except Exception as exc:
        if provider == "tavily":
            try:
                provider = "duckduckgo"
                hits = _search_duckduckgo(query, max_results=max_results)
            except Exception as fallback_exc:
                return f"Search Error: {fallback_exc}"
        else:
            return f"Search Error: {exc}"

    formatted = _format_search_results(
        query, hits, provider=provider, summary=summary, focus=focus,
    )
    store_search(query, formatted, source=provider)
    header = f"Query: {query}\n"
    return header + (prior + formatted if prior else formatted)


def _generate_via_pollinations(prompt: str, *, width: int = 1024, height: int = 1024) -> str:
    _ensure_dirs()
    encoded = urllib.parse.quote(prompt)
    remote_url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&nologo=true&enhance=true"
    )
    stamp = int(time.time())
    safe_slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()[:40]).strip("-") or "image"
    filename = f"img_{stamp}_{safe_slug}.png"
    local_path = GENERATED_DIR / filename

    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        res = client.get(remote_url)
        res.raise_for_status()
        local_path.write_bytes(res.content)

    rel = local_path.relative_to(WORKSPACE_ROOT).as_posix()
    result = (
        f"STATUS: SUCCESS\n"
        f"Image generated successfully.\n"
        f"Prompt: {prompt}\n"
        f"Saved to: {rel}\n"
        f"Local URL: /generated/{filename}\n"
        f"Remote URL: {remote_url}"
    )
    store_image_memory(prompt, result, backend="pollinations", image_ref=rel)
    return result


def _generate_via_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    _ensure_dirs()
    with httpx.Client(timeout=90.0) as client:
        res = client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1024x1024"},
        )
        res.raise_for_status()
        data = res.json()

    image_url = data["data"][0]["url"]
    stamp = int(time.time())
    filename = f"img_{stamp}_dalle.png"
    local_path = GENERATED_DIR / filename

    with httpx.Client(timeout=90.0) as client:
        img_res = client.get(image_url)
        img_res.raise_for_status()
        local_path.write_bytes(img_res.content)

    rel = local_path.relative_to(WORKSPACE_ROOT).as_posix()
    result = (
        f"STATUS: SUCCESS\n"
        f"Image generated via DALL-E 3.\n"
        f"Prompt: {prompt}\n"
        f"Saved to: {rel}\n"
        f"Local URL: /generated/{filename}"
    )
    store_image_memory(prompt, result, backend="openai", image_ref=rel)
    return result


@tool
def web_search(query: str):
    """Search the web for current information, facts, or news. Use for any question needing up-to-date data."""
    try:
        body = _run_web_search(query, max_results=5)
        return _tool_ok("web_search", body)
    except Exception as exc:
        return _tool_err("web_search", str(exc))


@tool
def search_and_summarize(query: str, focus: str = ""):
    """Deep web research with structured summary and numbered citations. Use for research, news, or 'find out about X'."""
    try:
        body = _run_web_search(query, max_results=6, focus=(focus or "").strip())
        return _tool_ok("search_and_summarize", body)
    except Exception as exc:
        return _tool_err("search_and_summarize", str(exc))


@tool
def generate_image(prompt: str, style: str = ""):
    """Generate an image from a text description. Always use this when Sir asks to create, draw, or generate an image."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "Image Error: prompt cannot be empty."

    full_prompt = f"{prompt}, {style}".strip(", ") if style else prompt
    backends: list[tuple[str, object]] = []

    if os.getenv("OPENAI_API_KEY", "").strip():
        backends.append(("openai", lambda: _generate_via_openai(full_prompt)))
    backends.append(("pollinations", lambda: _generate_via_pollinations(full_prompt)))

    errors: list[str] = []
    for name, fn in backends:
        try:
            return fn()
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    _ensure_dirs()
    prompt_path = GENERATED_DIR / f"prompt_{int(time.time())}.txt"
    prompt_path.write_text(full_prompt, encoding="utf-8")
    return (
        f"Image backends unavailable ({'; '.join(errors)}). "
        f"Saved prompt to {prompt_path.relative_to(WORKSPACE_ROOT).as_posix()} for manual generation."
    )


@tool
def execute_python_code(code: str, control_allowed: ControlAllowed = False):
    """Run Python code for calculations, data analysis, logic, or automation. Requires elevated control."""
    if not control_allowed:
        return _tool_err(
            "execute_python_code",
            "PERMISSION_DENIED: Code execution requires elevated access. "
            "Ask Sir to say 'Allow Control', then retry this tool.",
        )
    code = (code or "").strip()
    if not code:
        return _tool_err("execute_python_code", "code cannot be empty.")
    try:
        result = repl.run(code)
        output = str(result).strip() if result is not None else "(no output)"
        return _tool_ok("execute_python_code", f"Execution complete:\n{output}")
    except Exception as exc:
        return _tool_err("execute_python_code", str(exc))


@tool
def read_file(path: str, max_chars: int = 12000, control_allowed: ControlAllowed = False):
    """Read a text file anywhere on Sir's Mac (e.g. ~/Desktop/notes.txt, /Users/...). Requires Allow Control."""
    if not control_allowed:
        return _permission_denied("read_file")
    if not (path or "").strip():
        return _tool_err("read_file", "path cannot be empty.")
    try:
        target = _resolve_path(path)
        body = read_text_file(target, max_chars=max_chars)
        return _tool_ok("read_file", body)
    except Exception as exc:
        return _tool_err("read_file", str(exc))


@tool
def write_file(
    path: str,
    content: str,
    control_allowed: ControlAllowed = False,
    append: bool = False,
):
    """Create or edit a text file on Sir's Mac. Requires Allow Control for Mac-wide paths."""
    if not control_allowed:
        return _permission_denied("write_file")
    path = (path or "").strip()
    if not path:
        return _tool_err("write_file", "path cannot be empty.")
    try:
        target = _resolve_path(path)
        result = write_text_file(target, content or "", append=bool(append))
        return _tool_ok("write_file", result)
    except Exception as exc:
        return _tool_err("write_file", str(exc))


@tool
def list_directory(
    path: str = "~",
    max_entries: int = 200,
    control_allowed: ControlAllowed = False,
):
    """List files and folders anywhere on Sir's Mac (~, ~/Documents, /Applications, /Users/...). Requires Allow Control."""
    if not control_allowed:
        return _permission_denied("list_directory")
    try:
        raw = (path or "~").strip() or "~"
        target = _resolve_path(raw)
        # Applications folders are large — raise the cap automatically
        limit = int(max_entries or 200)
        if "applications" in str(target).lower():
            limit = max(limit, 400)
        body = list_path(target, max_entries=limit)
        return _tool_ok("list_directory", body)
    except Exception as exc:
        return _tool_err("list_directory", str(exc))


@tool
def run_terminal_command(
    command: str,
    working_directory: str = "",
    user_confirmed: bool = False,
    control_allowed: ControlAllowed = False,
    thread_id: str = "web_user_001",
):
    """
    Run a shell command on Sir's Mac. Safe commands (ls, pwd, git status) run immediately.
    Dangerous commands (rm, sudo, install) require Sir to confirm first.
    Blocked commands (rm -rf /, system edits) are never executed.
    """
    if not control_allowed:
        return _permission_denied("run_terminal_command")

    cmd = (command or "").strip()
    if not cmd:
        return _tool_err("run_terminal_command", "command cannot be empty.")

    level = classify_command(cmd)
    if level == "blocked":
        return _tool_err(
            "run_terminal_command",
            "BLOCKED: This command targets protected system resources and cannot be executed.",
        )
    if level == "confirm" and not user_confirmed:
        set_pending_confirmation(
            thread_id,
            tool="run_terminal_command",
            command=cmd,
            reason="potentially dangerous",
            working_directory=(working_directory or "").strip(),
        )
        return _tool_err(
            "run_terminal_command",
            format_confirmation_request(cmd, reason="potentially dangerous"),
        )

    try:
        output = run_shell_command(cmd, working_directory=working_directory)
        return _tool_ok("run_terminal_command", output)
    except subprocess.TimeoutExpired:
        return _tool_err("run_terminal_command", f"Command timed out after 120s: {cmd}")
    except Exception as exc:
        return _tool_err("run_terminal_command", str(exc))


@tool
def open_url_in_browser(url: str, control_allowed: ControlAllowed = False):
    """Open a URL in the system browser. Requires elevated control."""
    if not control_allowed:
        return _tool_err(
            "open_url_in_browser",
            "PERMISSION_DENIED: Browser access requires elevated access. "
            "Ask Sir to say 'Allow Control', then retry.",
        )
    url = (url or "").strip()
    if not url:
        return _tool_err("open_url_in_browser", "URL cannot be empty.")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    try:
        webbrowser.open(url)
        return _tool_ok("open_url_in_browser", f"Browser opened to {url}.")
    except Exception as exc:
        return _tool_err("open_url_in_browser", str(exc))


@tool
def get_current_time() -> str:
    """Returns the current local date and time. Use when Sir asks about time, date, or 'what day is it'. Always call this tool — never guess the time."""
    now = datetime.datetime.now()
    formatted = now.strftime("It is %A, %B %d %Y, %I:%M %p.")
    return _tool_ok("get_current_time", formatted)


def _load_task_plan() -> dict:
    _ensure_dirs()
    if not TASKS_FILE.exists():
        return {}
    try:
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_task_plan(plan: dict) -> None:
    _ensure_dirs()
    TASKS_FILE.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _format_task_plan(plan: dict) -> str:
    if not plan:
        return "STATUS: EMPTY\nNo active task plan."
    lines = [f"STATUS: ACTIVE\nTask: {plan.get('description', 'unknown')}", "Steps:"]
    for i, step in enumerate(plan.get("steps") or [], 1):
        lines.append(f"  {i}. [{step.get('status', 'pending')}] {step.get('text', '')}")
    return "\n".join(lines)


@tool("recall_memory")
def recall_memory_tool(query: str, limit: int = 4):
    """Search local memory for prior searches, notes, and conversation context. Use before repeating work or when Sir references past interactions."""
    query = (query or "").strip()
    if not query:
        return _tool_err("recall_memory", "query cannot be empty.")

    hits = recall_memory(query, n=min(max(limit, 1), 8))
    if not hits:
        return _tool_ok("recall_memory", f"No relevant memory found for: {query}")

    lines = [f"Found {len(hits)} memory hit(s) for: {query}", ""]
    for i, hit in enumerate(hits, 1):
        media = hit.get("media_type") or "search"
        topic = hit.get("topic") or "unknown"
        body = str(hit.get("content") or "")[:700]
        lines.append(f"{i}. [{media}] {topic}")
        lines.append(f"   {body}")
        if hit.get("tools_used"):
            lines.append(f"   Tools used: {hit['tools_used']}")
        lines.append("")
    return _tool_ok("recall_memory", "\n".join(lines).strip())


@tool
def save_memory_note(topic: str, content: str):
    """Save a fact, preference, or instruction Sir wants remembered across future turns."""
    topic = (topic or "").strip()
    content = (content or "").strip()
    if not topic or not content:
        return _tool_err("save_memory_note", "topic and content are required.")
    store_note(topic, content, source="agent")
    return _tool_ok("save_memory_note", f"Remembered under topic: {topic}")


@tool
def manage_task_plan(
    action: str,
    description: str = "",
    steps: str = "",
):
    """
    Track multi-step tasks across turns.
    action: create | view | update | complete | clear
    steps: newline-separated steps (for create), or 'step_number:done|pending|failed' (for update)
    """
    action = (action or "").strip().lower()
    plan = _load_task_plan()

    if action == "create":
        desc = (description or "").strip()
        if not desc:
            return "Task Error: description required for create."
        step_list = [s.strip() for s in (steps or "").splitlines() if s.strip()]
        if not step_list:
            step_list = ["Analyze request", "Execute with tools", "Report results to Sir"]
        plan = {
            "description": desc,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "active",
            "steps": [{"text": s, "status": "pending"} for s in step_list],
        }
        _save_task_plan(plan)
        lines = [f"STATUS: CREATED\nTask: {desc}", "Steps:"]
        for i, step in enumerate(plan["steps"], 1):
            lines.append(f"  {i}. [pending] {step['text']}")
        return "\n".join(lines)

    if action == "view":
        return _format_task_plan(plan)

    if action == "update":
        if not plan:
            return "Task Error: no active plan — use action=create first."
        updates = [s.strip() for s in (steps or "").splitlines() if s.strip()]
        for upd in updates:
            if ":" not in upd:
                continue
            idx_s, status = upd.split(":", 1)
            try:
                idx = int(idx_s.strip()) - 1
                step_list = plan.get("steps") or []
                if 0 <= idx < len(step_list):
                    step_list[idx]["status"] = status.strip().lower()
            except ValueError:
                continue
        _save_task_plan(plan)
        return _format_task_plan(plan)

    if action == "complete":
        if not plan:
            return "STATUS: EMPTY\nNo active task to complete."
        plan["status"] = "completed"
        plan["completed_at"] = datetime.datetime.now().isoformat()
        summary = _format_task_plan(plan)
        store_note(
            f"completed task: {plan.get('description', '')[:80]}",
            summary,
            source="task",
        )
        TASKS_FILE.unlink(missing_ok=True)
        return f"STATUS: COMPLETED\nTask archived to memory: {plan.get('description', '')}"

    if action == "clear":
        TASKS_FILE.unlink(missing_ok=True)
        return "STATUS: CLEARED\nActive task plan removed."

    return f"Task Error: unknown action '{action}'. Use create|view|update|complete|clear."


tools = [
    web_search,
    search_and_summarize,
    recall_memory_tool,
    save_memory_note,
    manage_task_plan,
    generate_image,
    execute_python_code,
    read_file,
    write_file,
    list_directory,
    run_terminal_command,
    open_url_in_browser,
    get_current_time,
]

TOOL_BY_NAME: dict[str, object] = {t.name: t for t in tools}

_CONTROL_GATED = frozenset({
    "execute_python_code",
    "read_file",
    "write_file",
    "list_directory",
    "run_terminal_command",
    "open_url_in_browser",
})

_CONFIRMATION_GATED = frozenset({"run_terminal_command"})

_TIME_QUERY_RE = re.compile(
    r"\b(what(?:'s| is) the time|what time|current time|tell(?: me)? the time|"
    r"what(?:'s| is) the date|what day is it)\b",
    re.IGNORECASE,
)
_SEARCH_QUERY_RE = re.compile(
    r"\b(search|look up|lookup|research|find out|latest|news about|web search)\b",
    re.IGNORECASE,
)
_RECALL_QUERY_RE = re.compile(
    r"\b(recall|remember|what did we|previous|last time|earlier|before)\b",
    re.IGNORECASE,
)
_TASK_VIEW_RE = re.compile(
    r"\b(task plan|current plan|plan status|show (my )?plan|view plan)\b",
    re.IGNORECASE,
)
_LIST_DIR_RE = re.compile(
    r"\b("
    r"list(?:\s+(?:the|my|all))?\s+(?:files?|folders?|directories|dir|contents?|apps?|applications)|"
    r"(?:show|display|see)\s+(?:(?:the|my|all)\s+)?(?:files?|folders?|directories|dir|contents?|apps?|applications)|"
    r"ls\b|"
    r"what(?:'s| is)\s+(?:on|in)\s+(?:my\s+)?(?:desktop|documents|downloads|folder|directory|apps?|applications|pc|mac|computer)|"
    r"files?\s+(?:on|in)\s+(?:my\s+)?(?:desktop|documents|downloads|folder|directory|home)|"
    r"list\s+(?:my\s+)?(?:desktop|documents|downloads|home|folder|directory|apps?|applications)|"
    r"(?:installed\s+)?(?:apps?|applications)\s+(?:on|in)\s+(?:my\s+)?(?:pc|mac|computer|system)|"
    r"(?:what|which)\s+apps?\b|"
    r"apps?\s+(?:installed|on\s+(?:my\s+)?(?:pc|mac|computer))"
    r")\b",
    re.IGNORECASE,
)
_LIST_APPS_RE = re.compile(
    r"\b("
    r"(?:list|show|display|see|get|find)\b.+\b(?:apps?|applications)\b|"
    r"\b(?:apps?|applications)\b.+\b(?:list|show|installed|on\s+my)\b|"
    r"installed\s+(?:apps?|applications)|"
    r"what(?:'s| is| are)\s+(?:the\s+)?(?:apps?|applications)\b"
    r")\b",
    re.IGNORECASE,
)
_PATH_HINTS = (
    (re.compile(r"\b(?:apps?|applications)\b", re.I), "/Applications"),
    (re.compile(r"\bdesktop\b", re.I), "~/Desktop"),
    (re.compile(r"\bdocuments\b", re.I), "~/Documents"),
    (re.compile(r"\bdownloads\b", re.I), "~/Downloads"),
    (re.compile(r"\bhome\b", re.I), "~"),
)

# Models without native tool-calling often emit this as plain text instead of tool_calls.
_PSEUDO_TOOL_CALL_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:tool[_ ]?call|function[_ ]?call|call(?:ing)?\s+tool)\s*[:\-]?\s*"
    r"(?P<name>[a-zA-Z_][\w]*)"
    r"(?:\s*\((?P<paren>[^)]*)\))?"
    r"(?:\s+(?P<rest>[^\n]+))?",
    re.IGNORECASE,
)
_KNOWN_TOOL_NAMES = frozenset({
    "web_search", "search_and_summarize", "recall_memory", "save_memory_note",
    "manage_task_plan", "generate_image", "execute_python_code",
    "read_file", "write_file", "list_directory", "run_terminal_command",
    "open_url_in_browser", "get_current_time",
})


def run_tool(
    name: str,
    args: Optional[dict] = None,
    *,
    control_allowed: bool = False,
    thread_id: str = "web_user_001",
    user_confirmed: bool = False,
) -> str:
    """Execute a tool programmatically (bypasses LLM tool-calling)."""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return _tool_err(name or "unknown", f"Unknown tool: {name}")

    payload = dict(args or {})
    if name in _CONTROL_GATED:
        payload["control_allowed"] = control_allowed
    if name in _CONFIRMATION_GATED:
        payload["thread_id"] = thread_id
        payload["user_confirmed"] = user_confirmed
    try:
        result = tool.invoke(payload)
        return str(result)
    except Exception as exc:
        return _tool_err(name, str(exc))


def _infer_list_path(user_text: str) -> str:
    """Best-effort Mac path for list_directory from natural language."""
    text = user_text or ""
    for pattern, path in _PATH_HINTS:
        if pattern.search(text):
            return path
    quoted = re.search(r'["\']([^"\']+)["\']', text)
    if quoted:
        return quoted.group(1).strip()
    bare = re.search(r"(~/[^\s,]+|/Users/[^\s,]+|/Applications\b|/System/[^\s,]+)", text)
    if bare:
        return bare.group(1).strip().rstrip(".,;:")
    return "~"


def _args_from_pseudo_rest(name: str, paren: str, rest: str) -> dict:
    """Map free-form pseudo tool-call text into tool args."""
    blob = " ".join(p for p in (paren or "", rest or "") if p).strip()
    if name == "list_directory":
        path = "~"
        pm = re.search(r"path\s*[=:]\s*['\"]?([^\s'\",}]+)", blob, re.I)
        if pm:
            path = pm.group(1)
        elif blob:
            token = blob.split()[0].strip("'\",;")
            if token.startswith(("~", "/", "./")) or token.lower() in {
                "desktop", "documents", "downloads", "applications", "home",
            }:
                path = {
                    "desktop": "~/Desktop",
                    "documents": "~/Documents",
                    "downloads": "~/Downloads",
                    "applications": "/Applications",
                    "home": "~",
                }.get(token.lower(), token)
        return {"path": path}
    if name == "read_file":
        pm = re.search(r"path\s*[=:]\s*['\"]?([^\s'\",}]+)", blob, re.I)
        path = pm.group(1) if pm else (blob.split()[0] if blob else "")
        return {"path": path}
    if name == "run_terminal_command":
        pm = re.search(r"command\s*[=:]\s*['\"](.+?)['\"]\s*$", blob, re.I)
        if pm:
            return {"command": pm.group(1)}
        return {"command": blob.strip("'\"") if blob else ""}
    if name == "open_url_in_browser":
        pm = re.search(r"url\s*[=:]\s*['\"]?(\S+)", blob, re.I)
        return {"url": pm.group(1) if pm else blob.split()[0] if blob else ""}
    if name in ("web_search", "search_and_summarize", "recall_memory"):
        pm = re.search(r"query\s*[=:]\s*['\"]?(.+)", blob, re.I)
        return {"query": (pm.group(1).strip().strip("'\"") if pm else blob) or ""}
    if name == "execute_python_code":
        pm = re.search(r"code\s*[=:]\s*['\"]?(.+)", blob, re.I | re.S)
        return {"code": (pm.group(1) if pm else blob) or ""}
    return {}


def looks_like_pseudo_tool_call(text: str) -> bool:
    """True when the model wrote a tool invocation as plain text (not structured tool_calls)."""
    t = (text or "").strip()
    if not t:
        return False
    if _PSEUDO_TOOL_CALL_RE.search(t):
        return True
    # Bare "list_directory /Applications" style lines
    lower = t.lower()
    for name in _KNOWN_TOOL_NAMES:
        if re.search(rf"(?:^|\n)\s*{re.escape(name)}\s*[\(/]", t, re.I):
            return True
        if lower.startswith(f"{name} ") or lower == name:
            return True
    return False


def parse_pseudo_tool_calls(text: str) -> list[tuple[str, dict]]:
    """
    Parse hallucinated plain-text tool invocations into real (name, args) pairs.
    Example: 'tool_call: list_directory /Applications'
    """
    text = (text or "").strip()
    if not text:
        return []

    planned: list[tuple[str, dict]] = []
    for m in _PSEUDO_TOOL_CALL_RE.finditer(text):
        name = (m.group("name") or "").strip()
        if name not in _KNOWN_TOOL_NAMES and name not in TOOL_BY_NAME:
            continue
        args = _args_from_pseudo_rest(name, m.group("paren") or "", m.group("rest") or "")
        planned.append((name, args))

    if planned:
        return _dedupe_planned(planned)

    # Fallback: single-line "list_directory /path" without tool_call: prefix
    for name in _KNOWN_TOOL_NAMES:
        m = re.match(rf"^\s*{re.escape(name)}\s*(?:\(([^)]*)\))?\s*(.*)$", text, re.I | re.S)
        if m:
            args = _args_from_pseudo_rest(name, m.group(1) or "", m.group(2) or "")
            return [(name, args)]
    return []


def _dedupe_planned(planned: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, args in planned:
        key = f"{name}:{sorted((args or {}).items())}"
        if key not in seen:
            seen.add(key)
            unique.append((name, args))
    return unique


def infer_forced_tools(user_text: str) -> list[tuple[str, dict]]:
    """
    Infer which tools should run deterministically when the LLM fails to call them.
    Returns list of (tool_name, args) in execution order.
    """
    text = (user_text or "").strip()
    if not text:
        return []

    inferred: list[tuple[str, dict]] = []
    if _TIME_QUERY_RE.search(text):
        inferred.append(("get_current_time", {}))

    # Apps on Mac → /Applications (and user Applications if present)
    if _LIST_APPS_RE.search(text) or (
        _LIST_DIR_RE.search(text) and re.search(r"\b(?:apps?|applications)\b", text, re.I)
    ):
        inferred.append(("list_directory", {"path": "/Applications", "max_entries": 400}))
        user_apps = str(Path.home() / "Applications")
        try:
            if Path(user_apps).is_dir() and any(Path(user_apps).iterdir()):
                inferred.append(("list_directory", {"path": "~/Applications", "max_entries": 200}))
        except OSError:
            pass
    elif _LIST_DIR_RE.search(text):
        inferred.append(("list_directory", {"path": _infer_list_path(text)}))

    if _SEARCH_QUERY_RE.search(text):
        inferred.append(("web_search", {"query": text}))
    if _RECALL_QUERY_RE.search(text):
        inferred.append(("recall_memory", {"query": text}))
    if _TASK_VIEW_RE.search(text):
        inferred.append(("manage_task_plan", {"action": "view"}))

    return _dedupe_planned(inferred)