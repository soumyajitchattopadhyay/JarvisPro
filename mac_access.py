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


# Placeholder home segments LLMs invent instead of the real username.
_FAKE_USER_SEGMENTS = frozenset({
    "yourusername", "username", "your_username", "user", "myuser",
    "myusername", "name", "homeuser", "macuser", "user_name",
    "yourname", "your-name", "example", "placeholder",
})

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
    """Absolute path to the real current user's home directory."""
    home = Path.home().expanduser().resolve()
    env_home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if env_home:
        try:
            home = Path(os.path.expanduser(env_home)).resolve()
        except Exception:
            pass
    return home


def sanitize_path_string(path: str) -> str:
    """
    Rewrite LLM hallucinations and expand ~ / $HOME to the real Mac home.

    Examples:
      /Users/yourusername/Desktop  →  /Users/<real>/Desktop
      /Users/fakeuser123/Desktop   →  /Users/<real>/Desktop
      ~/Desktop/TESrACT-1          →  /Users/<real>/Desktop/TESrACT-1
      Desktop                      →  /Users/<real>/Desktop
      my Desktop/foo               →  /Users/<real>/Desktop/foo
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

    # Strip leading "my "/"the "/"sir's " from location phrases
    raw = re.sub(
        r"^(?:my|the|sir'?s|our)\s+(?=(?:desktop|documents|downloads|movies|music|pictures|home|applications)\b)",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    # /Users/<anyone-not-me>/... or /home/<anyone>/... → real home
    # Personal TESrACT always operates as the logged-in user; LLMs invent wrong usernames.
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

    # Bare well-known folder names
    lower = raw.lower().rstrip("/")
    if lower in _WELL_KNOWN_DIRS:
        sub = _WELL_KNOWN_DIRS[lower]
        return str(home / sub) if sub else home_str

    # Leading ~/
    if raw == "~":
        return home_str
    if raw.startswith("~/"):
        return str(home / raw[2:])

    # Relative "Desktop/foo" or "Documents/bar" (after stripping my/the)
    for key, folder in _WELL_KNOWN_DIRS.items():
        if not folder:
            continue
        m = re.match(rf"^{re.escape(key)}(/.*)?$", raw, re.IGNORECASE)
        if m:
            rest = (m.group(1) or "").lstrip("/")
            return str(home / folder / rest) if rest else str(home / folder)

    return raw


def resolve_mac_path(path: str, *, must_exist: bool = False) -> Path:
    """
    Resolve a user/LLM path to an absolute real Mac path.

    Always expands ~ and $HOME to the current user's real home directory.
    Rewrites any non-matching /Users/<name> to the real home (LLM placeholders).
    Does not require intermediate parents to exist (needed for mkdir/write).
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError("Path cannot be empty.")

    cleaned = sanitize_path_string(raw)
    if not cleaned:
        raise ValueError(f"Path cannot be empty (original={path!r}).")

    try:
        expanded = os.path.expanduser(os.path.expandvars(cleaned))
        home = real_home()
        real_user = home.name
        parts = Path(expanded).parts
        # Path.parts on Unix: ('/', 'Users', 'name', ...)
        if len(parts) >= 3 and parts[1] in ("Users", "home"):
            if parts[2] != real_user:
                expanded = str(home.joinpath(*parts[3:])) if len(parts) > 3 else str(home)
        # strict=False: allow resolving paths whose parents do not exist yet
        resolved = Path(expanded).expanduser().resolve(strict=False)
    except TypeError:
        # Python < 3.9 resolve(strict=) not available — fall back
        resolved = Path(expanded).expanduser().resolve()
    except Exception as exc:
        raise ValueError(
            f"Could not resolve path original={path!r} sanitized={cleaned!r}: {exc}"
        ) from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Path not found: original={path!r} resolved={resolved}"
        )
    return resolved


def rewrite_paths_in_command(command: str) -> str:
    """Rewrite non-real /Users/<name> paths inside shell commands to real home."""
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
    # Expand bare ~ that is not part of a longer token
    cmd = re.sub(r"(?<![\w/])~(?=/|\s|$)", home_str, cmd)
    return cmd


# Session hint: last path successfully used (for "that folder" follow-ups)
_LAST_PATH_FILE = AGENT_DATA_DIR / "last_path.json"


def remember_path(path: Path | str) -> None:
    try:
        AGENT_DATA_DIR.mkdir(exist_ok=True)
        p = Path(path).expanduser().resolve()
        _LAST_PATH_FILE.write_text(
            json.dumps({"path": str(p)}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def last_remembered_path() -> Path | None:
    try:
        if not _LAST_PATH_FILE.exists():
            return None
        data = json.loads(_LAST_PATH_FILE.read_text(encoding="utf-8"))
        p = Path(str(data.get("path") or ""))
        return p if p.exists() else p  # return even if missing for mkdir parent
    except Exception:
        return None


def find_named_path(name: str, *, search_roots: list[Path] | None = None) -> Path | None:
    """Find an existing folder/file by name under Desktop/home (shallow search only)."""
    name = (name or "").strip().strip("/").strip("'\"")
    name = re.sub(r"^(?:my|the|sir'?s|our)\s+", "", name, flags=re.IGNORECASE).strip()
    if not name:
        return None
    home = real_home()
    # Well-known locations resolve directly
    lower = name.lower().rstrip("/")
    if lower in _WELL_KNOWN_DIRS:
        sub = _WELL_KNOWN_DIRS[lower]
        target = home / sub if sub else home
        return target if target.exists() else target

    roots = search_roots or [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home,
    ]
    # Exact path if name is already absolute-ish
    try:
        direct = resolve_mac_path(name)
        if direct.exists():
            return direct
    except Exception:
        pass

    # basename only for nested search
    base_name = Path(name).name
    for root in roots:
        if not root.is_dir():
            continue
        candidate = root / base_name
        if candidate.exists():
            try:
                return candidate.resolve(strict=False)
            except TypeError:
                return candidate.resolve()
        # One level deeper only (e.g. Desktop/TESrACT-1) — never recursive rglob
        try:
            for child in root.iterdir():
                if child.is_dir():
                    nested = child / base_name
                    if nested.exists():
                        try:
                            return nested.resolve(strict=False)
                        except TypeError:
                            return nested.resolve()
        except OSError:
            continue
    return None


def ensure_directory(path: Path) -> str:
    """Create a directory (and parents). Returns status string with real path."""
    try:
        resolved = path.expanduser().resolve(strict=False)
    except TypeError:
        resolved = path.expanduser().resolve()
    level = assess_write_path(resolved)
    if level == "blocked":
        raise PermissionError(
            f"Blocked: cannot create directory on protected path ({resolved})."
        )
    existed = resolved.exists() and resolved.is_dir()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"mkdir failed for resolved={resolved}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise OSError(f"Path exists but is not a directory: {resolved}")
    remember_path(resolved)
    if existed:
        return f"Directory already exists: {resolved}"
    return f"Created directory: {resolved}"


def _is_under_home(path: Path) -> bool:
    try:
        try:
            resolved = path.resolve(strict=False)
            home = real_home().resolve(strict=False)
        except TypeError:
            resolved = path.resolve()
            home = real_home().resolve()
        resolved.relative_to(home)
        return True
    except ValueError:
        return False


def assess_write_path(path: Path) -> PermissionLevel:
    try:
        resolved_path = path.resolve(strict=False)
    except TypeError:
        resolved_path = path.resolve()
    resolved = str(resolved_path)
    for prefix in _BLOCKED_WRITE_PREFIXES:
        if resolved.startswith(prefix):
            return "blocked"
    if _is_under_home(path):
        try:
            home = real_home().resolve(strict=False)
        except TypeError:
            home = real_home().resolve()
        rel = str(resolved_path.relative_to(home))
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
    cmd = rewrite_paths_in_command((command or "").strip())
    if not cmd:
        raise ValueError("Command cannot be empty.")

    cwd: str | None = None
    if working_directory:
        try:
            cwd_path = resolve_mac_path(working_directory)
        except Exception as exc:
            raise NotADirectoryError(
                f"Working directory invalid: original={working_directory!r} ({exc})"
            ) from exc
        if not cwd_path.is_dir():
            raise NotADirectoryError(
                f"Working directory not found: original={working_directory!r} resolved={cwd_path}"
            )
        cwd = str(cwd_path)

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "HOME": str(real_home()), "LANG": "C.UTF-8"},
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    lines = [f"Command: {cmd}", f"Exit code: {proc.returncode}"]
    if cwd:
        lines.insert(1, f"CWD: {cwd}")
    if stdout:
        lines.append(f"STDOUT:\n{stdout[:12000]}")
    if stderr:
        lines.append(f"STDERR:\n{stderr[:4000]}")
    if proc.returncode != 0 and not stdout and not stderr:
        lines.append("Command failed with no output.")
    # Remember mkdir targets so "that folder" follow-ups resolve correctly
    if proc.returncode == 0 and re.search(r"\bmkdir\b", cmd):
        for token in re.findall(r"(?:^|\s)(/[^\s\"']+)", cmd):
            try:
                p = Path(token)
                if p.is_dir():
                    remember_path(p)
            except Exception:
                pass
        # Also catch quoted paths: mkdir -p '/Users/.../Folder'
        for token in re.findall(r"""['"](/[^'"]+)['"]""", cmd):
            try:
                p = Path(token)
                if p.is_dir():
                    remember_path(p)
            except Exception:
                pass
    return "\n".join(lines)


def is_probably_text_file(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_EXTENSIONS:
        return True
    if path.name in (".env.example", "Dockerfile", "Makefile", "LICENSE"):
        return True
    return path.suffix == "" and path.stat().st_size < 512_000


def read_text_file(path: Path, *, max_chars: int = 12000) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: resolved={resolved}")
    if not resolved.is_file():
        raise IsADirectoryError(f"Not a file: resolved={resolved}")
    if not is_probably_text_file(resolved):
        raise ValueError(
            f"Refusing to read likely binary file: resolved={resolved} "
            f"suffix={resolved.suffix or '(none)'} ({resolved.stat().st_size} bytes)"
        )
    text = resolved.read_text(encoding="utf-8", errors="replace")
    remember_path(resolved.parent)
    if len(text) > max_chars:
        return (
            f"File: {resolved} ({len(text)} chars, showing first {max_chars})\n\n"
            f"{text[:max_chars]}\n\n… [truncated]"
        )
    return f"File: {resolved}\n\n{text}"


def write_text_file(path: Path, content: str, *, append: bool = False) -> str:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except TypeError:
        resolved = path.expanduser().resolve()
    level = assess_write_path(resolved)
    if level == "blocked":
        raise PermissionError(
            f"Blocked: writes to protected system path are not allowed (resolved={resolved})."
        )
    if resolved.suffix and resolved.suffix.lower() not in _TEXT_EXTENSIONS and resolved.name not in (
        "Dockerfile", "Makefile", "LICENSE",
    ):
        raise ValueError(
            f"Refusing to write unsupported file type: resolved={resolved} suffix={resolved.suffix}"
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if append and resolved.exists():
            with resolved.open("a", encoding="utf-8") as fh:
                fh.write(content or "")
            mode = "appended to"
        else:
            resolved.write_text(content or "", encoding="utf-8")
            mode = "written"
    except OSError as exc:
        raise OSError(
            f"Write failed for resolved={resolved}: {exc}"
        ) from exc
    remember_path(resolved.parent)
    return f"{mode.capitalize()} {len(content or '')} characters to {resolved}"


def list_path(path: Path, *, max_entries: int = 100) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: resolved={resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: resolved={resolved}")

    entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"Contents of {resolved}:"]
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
    # Do not overwrite last-path memory with broad listings (Desktop/home);
    # mkdir/write set the follow-up target for "that folder".
    return "\n".join(lines)