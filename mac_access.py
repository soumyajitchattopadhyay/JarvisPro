"""
TESrACT host-boundary helpers — secure computational Brain mode.

Architectural contract
----------------------
The Mac process is a *cognitive engine*. It must never mutate the host OS on
behalf of public / tunnel traffic:

  • No shell execution (subprocess, os.system, Popen)
  • No unrestricted host filesystem writes
  • No host root / system path manipulation

Physical actions (file creation, shell, opening apps) are packaged as
**client-side execution intents**. The front-end (or a trusted local client)
is the only layer that may act on the user's machine.

This module provides:
  1. Safe path *string* normalization for labels in intents (no host writes)
  2. Client execution protocol builders / parsers
  3. Lightweight pending-intent bookkeeping under agent_data/ (brain-local only)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Brain-local data only (inside the project tree — never host root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
AGENT_DATA_DIR = _PROJECT_ROOT / "agent_data"
PENDING_FILE = AGENT_DATA_DIR / "pending_confirmations.json"

ActionType = Literal["DOWNLOAD_FILE", "DISPLAY_DATA", "LOGIC_ONLY"]
ExecutionTarget = Literal["client", "brain", "none"]

# Marker embedded in tool RESULT strings so main.py can harvest intents.
CLIENT_INTENT_MARKER = "CLIENT_EXECUTION_INTENT"


def _ensure_agent_data() -> None:
    """Create brain-local agent_data only (project-relative, never host root)."""
    AGENT_DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Path string helpers (pure string transforms — no host mutation)
# ---------------------------------------------------------------------------

_WELL_KNOWN_DIRS = {
    "desktop": "Desktop",
    "documents": "Documents",
    "downloads": "Downloads",
    "movies": "Movies",
    "music": "Music",
    "pictures": "Pictures",
    "public": "Public",
    "applications": "Applications",
    "home": "",
}


def real_home() -> Path:
    """Logical home path for intent labels only — never used to write."""
    home = Path.home().expanduser()
    env_home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if env_home:
        try:
            home = Path(os.path.expanduser(env_home))
        except Exception:
            pass
    return home


def sanitize_path_string(path: str) -> str:
    """
    Normalize LLM path placeholders to a clean logical path string.

    Does not create, write, or delete anything on disk.
    """
    raw = (path or "").strip().strip("'\"")
    if not raw:
        return raw

    home = real_home()
    home_str = str(home)
    real_user = home.name

    raw = raw.replace("\\", "/")
    raw = os.path.expandvars(raw)
    raw = re.sub(r"%USERPROFILE%", home_str, raw, flags=re.IGNORECASE)

    raw = re.sub(
        r"^(?:my|the|sir'?s|our)\s+(?=(?:desktop|documents|downloads|movies|music|pictures|home|applications)\b)",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    def _rewrite_users(m: re.Match) -> str:
        user = m.group(1)
        rest = m.group(2) or ""
        if user == real_user:
            return m.group(0)
        return home_str + rest

    raw = re.sub(
        r"^/(?:Users|home)/([^/]+)(/.*)?$",
        _rewrite_users,
        raw,
        flags=re.IGNORECASE,
    )

    lower = raw.lower().rstrip("/")
    if lower in _WELL_KNOWN_DIRS:
        sub = _WELL_KNOWN_DIRS[lower]
        return str(home / sub) if sub else home_str

    if raw == "~":
        return home_str
    if raw.startswith("~/"):
        return str(home / raw[2:])

    for key, folder in _WELL_KNOWN_DIRS.items():
        if not folder:
            continue
        m = re.match(rf"^{re.escape(key)}(/.*)?$", raw, re.IGNORECASE)
        if m:
            rest = (m.group(1) or "").lstrip("/")
            return str(home / folder / rest) if rest else str(home / folder)

    return raw


def resolve_logical_path(path: str) -> str:
    """Return a sanitized logical path string for intents (no disk I/O)."""
    cleaned = sanitize_path_string(path)
    if not cleaned:
        raise ValueError("Path cannot be empty.")
    return cleaned


# Back-compat aliases used by actions.py (no host FS resolution / existence checks)
def resolve_mac_path(path: str, *, must_exist: bool = False) -> Path:
    """
    Logical Path object for labeling only.

    must_exist is ignored — the brain does not inspect the host filesystem.
    """
    del must_exist  # intentionally unused
    return Path(resolve_logical_path(path))


def rewrite_paths_in_command(command: str) -> str:
    """Rewrite placeholder /Users/<name> segments inside a command string."""
    cmd = (command or "").strip()
    if not cmd:
        return cmd
    home = real_home()
    home_str = str(home)
    real_user = home.name

    def _sub_users(m: re.Match) -> str:
        user = m.group(1)
        rest = m.group(2) or ""
        if user == real_user:
            return m.group(0)
        return home_str + rest

    cmd = re.sub(
        r"/(?:Users|home)/([^/\s\"']+)(/[^\s\"']*)?",
        _sub_users,
        cmd,
        flags=re.IGNORECASE,
    )
    cmd = re.sub(r"(?<![\w/])~(?=/|\s|$)", home_str, cmd)
    return cmd


def find_named_path(name: str, *, search_roots: list[Path] | None = None) -> Path | None:
    """
    Host search is disabled in Brain mode.

    Returns a logical Desktop-relative Path for intent labels only.
    """
    del search_roots
    name = (name or "").strip().strip("/").strip("'\"")
    name = re.sub(r"^(?:my|the|sir'?s|our)\s+", "", name, flags=re.IGNORECASE).strip()
    if not name:
        return None
    lower = name.lower().rstrip("/")
    home = real_home()
    if lower in _WELL_KNOWN_DIRS:
        sub = _WELL_KNOWN_DIRS[lower]
        return home / sub if sub else home
    # Prefer logical Desktop placement for relative names
    return home / "Desktop" / Path(name).name


def last_remembered_path() -> Path | None:
    """Brain-local last-path hint (agent_data only)."""
    path_file = AGENT_DATA_DIR / "last_path.json"
    try:
        if not path_file.exists():
            return None
        data = json.loads(path_file.read_text(encoding="utf-8"))
        p = str(data.get("path") or "").strip()
        return Path(p) if p else None
    except Exception:
        return None


def remember_path(path: Path | str) -> None:
    """Store a logical path hint under agent_data/ (brain-local only)."""
    try:
        _ensure_agent_data()
        (AGENT_DATA_DIR / "last_path.json").write_text(
            json.dumps({"path": str(path)}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Client execution protocol
# ---------------------------------------------------------------------------

def build_execution_envelope(
    *,
    action: ActionType,
    payload: dict[str, Any],
    status: str = "success",
    reply: str = "",
    execution_target: ExecutionTarget = "client",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard API / tool response shape for client-side execution."""
    body: dict[str, Any] = {
        "status": status,
        "execution_target": execution_target,
        "action": action,
        "payload": payload or {},
    }
    if reply:
        body["reply"] = reply
    if extra:
        body.update(extra)
    return body


def client_intent(
    *,
    action: ActionType,
    intent: str,
    payload: dict[str, Any] | None = None,
    message: str = "",
) -> dict[str, Any]:
    """Build a single client execution intent block."""
    data = dict(payload or {})
    data.setdefault("intent", intent)
    if message:
        data.setdefault("message", message)
    return build_execution_envelope(
        action=action,
        payload=data,
        status="success",
        execution_target="client",
        reply=message,
    )


def format_client_intent_for_tool(
    *,
    action: ActionType,
    intent: str,
    payload: dict[str, Any] | None = None,
    message: str = "",
) -> str:
    """
    Serialize an intent for ToolMessage content.

    Prefix marker lets main.py harvest intents from tool results.
    """
    envelope = client_intent(
        action=action,
        intent=intent,
        payload=payload,
        message=message,
    )
    return (
        f"{CLIENT_INTENT_MARKER}\n"
        f"{json.dumps(envelope, ensure_ascii=False, indent=2)}"
    )


def extract_client_intents(text: str) -> list[dict[str, Any]]:
    """Parse zero or more CLIENT_EXECUTION_INTENT blocks from tool/output text."""
    raw = text or ""
    if CLIENT_INTENT_MARKER not in raw:
        # Also accept a bare JSON envelope with execution_target=client
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("execution_target") == "client":
                return [obj]
        except Exception:
            pass
        return []

    intents: list[dict[str, Any]] = []
    parts = raw.split(CLIENT_INTENT_MARKER)
    for part in parts[1:]:
        chunk = part.strip()
        if not chunk:
            continue
        # Take the first JSON object in the chunk
        start = chunk.find("{")
        if start < 0:
            continue
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(chunk[start:])
            if isinstance(obj, dict):
                intents.append(obj)
        except Exception:
            continue
    return intents


def logic_only_envelope(reply: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_execution_envelope(
        action="LOGIC_ONLY",
        payload=payload or {},
        status="success",
        reply=reply,
        execution_target="brain",
    )


# ---------------------------------------------------------------------------
# Intent factories for former OS-mutating tools
# ---------------------------------------------------------------------------

def intent_download_file(
    *,
    filename: str,
    content: str,
    suggested_path: str = "",
    append: bool = False,
    mime_type: str = "text/plain;charset=utf-8",
) -> str:
    name = Path(sanitize_path_string(filename) or "download.txt").name
    logical = sanitize_path_string(suggested_path) if suggested_path else ""
    msg = (
        f"Prepared file '{name}' for client-side download"
        + (f" (logical path: {logical})" if logical else "")
        + ". Host filesystem was not modified."
    )
    return format_client_intent_for_tool(
        action="DOWNLOAD_FILE",
        intent="write_file",
        message=msg,
        payload={
            "filename": name,
            "content": content or "",
            "suggested_path": logical or name,
            "append": bool(append),
            "mime_type": mime_type,
            "encoding": "utf-8",
            "bytes": len((content or "").encode("utf-8")),
        },
    )


def intent_create_directory(*, path: str) -> str:
    logical = sanitize_path_string(path)
    msg = (
        f"Directory creation deferred to client for path '{logical}'. "
        "Brain did not mutate the host filesystem."
    )
    return format_client_intent_for_tool(
        action="DISPLAY_DATA",
        intent="create_directory",
        message=msg,
        payload={
            "path": logical,
            "operation": "mkdir",
            "recursive": True,
        },
    )


def intent_list_directory(*, path: str, max_entries: int = 200) -> str:
    logical = sanitize_path_string(path or "~")
    msg = (
        f"Host directory listing is client-side only. "
        f"Request list of '{logical}' (max {max_entries})."
    )
    return format_client_intent_for_tool(
        action="DISPLAY_DATA",
        intent="list_directory",
        message=msg,
        payload={
            "path": logical,
            "max_entries": int(max_entries or 200),
            "operation": "list",
        },
    )


def intent_read_file(*, path: str, max_chars: int = 12000) -> str:
    logical = sanitize_path_string(path)
    msg = (
        f"Host file read is client-side only. "
        f"Request contents of '{logical}' (max {max_chars} chars)."
    )
    return format_client_intent_for_tool(
        action="DISPLAY_DATA",
        intent="read_file",
        message=msg,
        payload={
            "path": logical,
            "max_chars": int(max_chars or 12000),
            "operation": "read",
        },
    )


def intent_run_command(*, command: str, working_directory: str = "") -> str:
    cmd = rewrite_paths_in_command((command or "").strip())
    wd = sanitize_path_string(working_directory) if working_directory else ""
    msg = (
        "Shell execution is disabled on the Brain host. "
        f"Command packaged for client-side review/run: {cmd!r}"
    )
    return format_client_intent_for_tool(
        action="DISPLAY_DATA",
        intent="run_terminal_command",
        message=msg,
        payload={
            "command": cmd,
            "working_directory": wd,
            "operation": "shell",
            "host_executed": False,
        },
    )


def intent_open_url(*, url: str) -> str:
    u = (url or "").strip()
    if u and not u.startswith(("http://", "https://")):
        u = f"https://{u}"
    msg = f"Open URL on the client: {u}"
    return format_client_intent_for_tool(
        action="DISPLAY_DATA",
        intent="open_url_in_browser",
        message=msg,
        payload={
            "url": u,
            "operation": "open_url",
            "host_executed": False,
        },
    )


# ---------------------------------------------------------------------------
# Pending confirmation bookkeeping (brain-local JSON only)
# ---------------------------------------------------------------------------

_CONFIRM_PHRASES = (
    "confirm command",
    "confirm run",
    "yes run it",
    "yes, run it",
    "proceed with command",
    "execute the command",
    "run the command",
    "go ahead and run",
    "approved",
    "confirm execution",
)


def command_fingerprint(command: str) -> str:
    normalized = " ".join((command or "").strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_confirmation_message(text: str) -> bool:
    lower = (text or "").strip().lower()
    return any(phrase in lower for phrase in _CONFIRM_PHRASES)


def _load_pending() -> dict:
    _ensure_agent_data()
    if not PENDING_FILE.exists():
        return {}
    try:
        return json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_pending(data: dict) -> None:
    _ensure_agent_data()
    PENDING_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_pending_confirmation(
    thread_id: str,
    *,
    tool: str,
    command: str,
    reason: str,
    working_directory: str = "",
) -> str:
    """Record a client-side command intent awaiting user confirmation in the UI."""
    data = _load_pending()
    fp = command_fingerprint(command)
    data[thread_id] = {
        "tool": tool,
        "command": command,
        "fingerprint": fp,
        "reason": reason,
        "working_directory": working_directory,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_target": "client",
    }
    _save_pending(data)
    return fp


def get_pending_confirmation(thread_id: str) -> dict | None:
    entry = _load_pending().get(thread_id)
    return entry if isinstance(entry, dict) else None


def clear_pending_confirmation(thread_id: str) -> None:
    data = _load_pending()
    if thread_id in data:
        del data[thread_id]
        _save_pending(data)


def consume_confirmation(thread_id: str, command: str) -> bool:
    pending = get_pending_confirmation(thread_id)
    if not pending:
        return False
    if command_fingerprint(command) != pending.get("fingerprint"):
        return False
    clear_pending_confirmation(thread_id)
    return True


def format_confirmation_request(command: str, *, reason: str = "client-side") -> str:
    return (
        f"CONFIRMATION_REQUIRED ({reason})\n"
        f"Command for client execution:\n  {command}\n\n"
        "Reply with 'Confirm command' to package this intent for the client. "
        "The Brain will not execute shell commands on the host."
    )


# ---------------------------------------------------------------------------
# Compatibility stubs — former mutation APIs raise loudly if misused
# ---------------------------------------------------------------------------

class HostMutationDisabled(RuntimeError):
    """Raised when legacy code tries to mutate the host OS via the Brain."""


def run_shell_command(*_args: Any, **_kwargs: Any) -> str:
    raise HostMutationDisabled(
        "Shell execution is disabled on the TESrACT Brain. "
        "Use intent_run_command() and return a client execution payload."
    )


def write_text_file(*_args: Any, **_kwargs: Any) -> str:
    raise HostMutationDisabled(
        "Host filesystem writes are disabled on the TESrACT Brain. "
        "Use intent_download_file() for client-side DOWNLOAD_FILE."
    )


def ensure_directory(*_args: Any, **_kwargs: Any) -> str:
    raise HostMutationDisabled(
        "Host mkdir is disabled on the TESrACT Brain. "
        "Use intent_create_directory() for client-side DISPLAY_DATA."
    )


def read_text_file(*_args: Any, **_kwargs: Any) -> str:
    raise HostMutationDisabled(
        "Host file reads are disabled on the TESrACT Brain. "
        "Use intent_read_file() for client-side DISPLAY_DATA."
    )


def list_path(*_args: Any, **_kwargs: Any) -> str:
    raise HostMutationDisabled(
        "Host directory listing is disabled on the TESrACT Brain. "
        "Use intent_list_directory() for client-side DISPLAY_DATA."
    )


def classify_command(command: str) -> Literal["ok", "denied", "blocked", "confirm"]:
    """
    Classify a command for client-side policy hints only.

    The Brain never executes these commands.
    """
    cmd = (command or "").strip()
    if not cmd:
        return "denied"
    # Extremely destructive patterns — mark blocked for client UI
    blocked = re.compile(
        r"(?:"
        r"\brm\s+(-[^\s]*)*[rf]{1,2}[^\s]*\s+[/~]"
        r"|\bsudo\s+rm\b"
        r"|\bmkfs\b"
        r"|\bdd\s+if="
        r"|\bshutdown\b"
        r"|\breboot\b"
        r")",
        re.IGNORECASE,
    )
    if blocked.search(cmd):
        return "blocked"
    dangerous = re.compile(
        r"(?:\bsudo\b|\brm\b|\bchmod\b|\bchown\b|\bkill\b|\bpip\s+install\b|"
        r"\bnpm\s+install\b|\bbrew\s+install\b)",
        re.IGNORECASE,
    )
    if dangerous.search(cmd):
        return "confirm"
    return "ok"


def assess_write_path(_path: Path) -> Literal["ok", "denied", "blocked", "confirm"]:
    """All host writes are blocked at the Brain boundary."""
    return "blocked"
