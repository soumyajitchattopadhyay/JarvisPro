"""
Mac filesystem and terminal access helpers for TESrACT.

All Mac-wide operations require explicit Allow Control permission.
Dangerous shell commands require an additional per-command user confirmation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

AGENT_DATA_DIR = Path(__file__).resolve().parent / "agent_data"
PENDING_FILE = AGENT_DATA_DIR / "pending_confirmations.json"

PermissionLevel = Literal["ok", "denied", "blocked", "confirm"]

# Paths that must never be written/deleted via agent tools.
_BLOCKED_WRITE_PREFIXES = (
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
    "/private/etc",
    "/private/var/root",
    "/Library/Apple",
    "/Applications/Utilities",
)

_BLOCKED_WRITE_HOME_SUFFIXES = (
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".gnupg",
    "Library/Keychains",
)

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".html", ".css", ".js", ".ts", ".tsx", ".jsx", ".xml", ".toml", ".ini",
    ".sh", ".bash", ".zsh", ".env", ".cfg", ".conf", ".log", ".sql",
    ".rb", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".cpp", ".h",
    ".vue", ".svelte", ".dockerfile", ".gitignore", ".env.example",
}

_BLOCKED_COMMAND_RE = re.compile(
    r"(?:"
    r"\brm\s+(-[^\s]*)*[rf]{1,2}[^\s]*\s+[/~]"  # rm -rf /
    r"|\brm\s+(-[^\s]*)*[rf]{1,2}[^\s]*\s+\S*/\.\."  # rm -rf parent traversal
    r"|\bsudo\s+rm\b"
    r"|\bmkfs\b"
    r"|\bdd\s+if="
    r"|\b>\s*/etc/"
    r"|\b>\s*/System/"
    r"|\bchmod\s+(-R\s+)?777\s+/"
    r"|\bchown\s+.*\s+/"
    r"|\blaunchctl\s+(unload|remove)\s+system"
    r"|\bdefaults\s+write\s+com\.apple"
    r"|\bkillall\s+-9\s+(WindowServer|kernel|launchd)"
    r")",
    re.IGNORECASE,
)

_DANGEROUS_COMMAND_RE = re.compile(
    r"(?:"
    r"\bsudo\b"
    r"|\brm\b"
    r"|\bmv\b"
    r"|\bchmod\b"
    r"|\bchown\b"
    r"|\bkill\b"
    r"|\bpkill\b"
    r"|\bkillall\b"
    r"|\bcurl\b.*\|\s*(?:ba)?sh"
    r"|\bwget\b.*\|\s*(?:ba)?sh"
    r"|\bpip\s+install\b"
    r"|\bnpm\s+install\b"
    r"|\bbrew\s+install\b"
    r"|\bgit\s+push\b"
    r"|\bgit\s+reset\s+--hard\b"
    r"|\btruncate\b"
    r"|\bshutdown\b"
    r"|\breboot\b"
    r"|\blaunchctl\b"
    r"|\bdefaults\s+write\b"
    r"|\bopen\s+-a\b"
    r")",
    re.IGNORECASE,
)

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


def _ensure_agent_data() -> None:
    AGENT_DATA_DIR.mkdir(exist_ok=True)


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


def command_fingerprint(command: str) -> str:
    normalized = " ".join((command or "").strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_confirmation_message(text: str) -> bool:
    lower = (text or "").strip().lower()
    return any(phrase in lower for phrase in _CONFIRM_PHRASES)


def set_pending_confirmation(
    thread_id: str,
    *,
    tool: str,
    command: str,
    reason: str,
    working_directory: str = "",
) -> str:
    data = _load_pending()
    fp = command_fingerprint(command)
    data[thread_id] = {
        "tool": tool,
        "command": command,
        "fingerprint": fp,
        "reason": reason,
        "working_directory": working_directory,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
    """True when user confirmed and command matches the pending entry."""
    pending = get_pending_confirmation(thread_id)
    if not pending:
        return False
    if command_fingerprint(command) != pending.get("fingerprint"):
        return False
    clear_pending_confirmation(thread_id)
    return True


def resolve_mac_path(path: str, *, must_exist: bool = False) -> Path:
    raw = (path or "").strip()
    if not raw:
        raise ValueError("Path cannot be empty.")
    expanded = os.path.expanduser(os.path.expandvars(raw))
    resolved = Path(expanded).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return resolved


def _is_under_home(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path.home().resolve())
        return True
    except ValueError:
        return False


def assess_write_path(path: Path) -> PermissionLevel:
    resolved = str(path.resolve())
    for prefix in _BLOCKED_WRITE_PREFIXES:
        if resolved.startswith(prefix):
            return "blocked"
    if _is_under_home(path):
        rel = str(path.resolve().relative_to(Path.home().resolve()))
        for suffix in _BLOCKED_WRITE_HOME_SUFFIXES:
            if rel == suffix or rel.startswith(suffix + "/"):
                return "blocked"
    return "ok"


def classify_command(command: str) -> PermissionLevel:
    cmd = (command or "").strip()
    if not cmd:
        return "denied"
    if _BLOCKED_COMMAND_RE.search(cmd):
        return "blocked"
    if _DANGEROUS_COMMAND_RE.search(cmd):
        return "confirm"
    return "ok"


def format_confirmation_request(command: str, *, reason: str = "dangerous") -> str:
    return (
        f"CONFIRMATION_REQUIRED ({reason})\n"
        f"Command to run:\n  {command}\n\n"
        "Reply with 'Confirm command' or 'Yes, run it' to approve this exact command."
    )


def run_shell_command(command: str, *, working_directory: str = "", timeout: int = 120) -> str:
    cmd = (command or "").strip()
    if not cmd:
        raise ValueError("Command cannot be empty.")

    cwd: str | None = None
    if working_directory:
        cwd_path = resolve_mac_path(working_directory)
        if not cwd_path.is_dir():
            raise NotADirectoryError(f"Working directory not found: {working_directory}")
        cwd = str(cwd_path)

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "LANG": "C.UTF-8"},
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    lines = [f"Exit code: {proc.returncode}"]
    if stdout:
        lines.append(f"STDOUT:\n{stdout[:12000]}")
    if stderr:
        lines.append(f"STDERR:\n{stderr[:4000]}")
    if proc.returncode != 0 and not stdout and not stderr:
        lines.append("Command failed with no output.")
    return "\n".join(lines)


def is_probably_text_file(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_EXTENSIONS:
        return True
    if path.name in (".env.example", "Dockerfile", "Makefile", "LICENSE"):
        return True
    return path.suffix == "" and path.stat().st_size < 512_000


def read_text_file(path: Path, *, max_chars: int = 12000) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    if not is_probably_text_file(path):
        raise ValueError(
            f"Refusing to read likely binary file: {path.suffix or '(no extension)'} "
            f"({path.stat().st_size} bytes)"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return (
            f"File: {path} ({len(text)} chars, showing first {max_chars})\n\n"
            f"{text[:max_chars]}\n\n… [truncated]"
        )
    return f"File: {path}\n\n{text}"


def write_text_file(path: Path, content: str, *, append: bool = False) -> str:
    level = assess_write_path(path)
    if level == "blocked":
        raise PermissionError(
            f"Blocked: writes to protected system path are not allowed ({path})."
        )
    if path.suffix and path.suffix.lower() not in _TEXT_EXTENSIONS and path.name not in (
        "Dockerfile", "Makefile", "LICENSE",
    ):
        raise ValueError(f"Refusing to write unsupported file type: {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content or "")
        mode = "appended to"
    else:
        path.write_text(content or "", encoding="utf-8")
        mode = "written"
    return f"{mode.capitalize()} {len(content or '')} characters to {path}"


def list_path(path: Path, *, max_entries: int = 100) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"Contents of {path}:"]
    for entry in entries[:max_entries]:
        kind = "dir" if entry.is_dir() else "file"
        try:
            size = entry.stat().st_size if entry.is_file() else 0
            extra = f" ({size:,} bytes)" if entry.is_file() else ""
        except OSError:
            extra = ""
        lines.append(f"  [{kind}] {entry.name}{extra}")
    if len(entries) > max_entries:
        lines.append(f"  … and {len(entries) - max_entries} more")
    return "\n".join(lines)