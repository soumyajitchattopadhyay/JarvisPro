"""
Simple Mac-control permission for TESrACT.

Single source of truth: user_settings.json
  { "control_allowed": true|false }

Toggle UI and "Allow Control" / "Stop Control" are the only writers.
Tools read this flag directly — no graph-state permission games.
"""
from __future__ import annotations

import json
import os
import re
import threading

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_BASE_DIR, "user_settings.json")
_LOCK = threading.Lock()

# Prefer a single global flag; keep legacy thread key for migration.
_FLAG_KEY = "control_allowed"
_LEGACY_THREAD = "web_user_001"

_GRANT_ONLY_RE = re.compile(
    r"^\s*(?:yes[,.]?\s+|please\s+|ok(?:ay)?[,.]?\s+|sure[,.]?\s+)*"
    r"(?:allow\s+control|grant\s+control|grant\s+access|enable\s+control|"
    r"elevate(?:d)?(?:\s+access)?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_REVOKE_ONLY_RE = re.compile(
    r"^\s*(?:please\s+|ok(?:ay)?[,.]?\s+)*"
    r"(?:stop\s+control|revoke\s+control|revoke\s+access|disable\s+control|"
    r"restrict\s+control)\s*[.!]?\s*$",
    re.IGNORECASE,
)

GRANTED_REPLY = (
    "Elevated Mac control is ON, Sir. "
    "I can read, write, and run commands until you turn it off or say 'Stop Control'."
)
REVOKED_REPLY = (
    "Mac control is OFF, Sir. Filesystem and terminal tools are locked "
    "until you turn the toggle on or say 'Allow Control'."
)
ALREADY_ON_REPLY = "Mac control is already ON, Sir."
ALREADY_OFF_REPLY = "Mac control is already OFF, Sir."


def _load() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def is_control_allowed() -> bool:
    """Authoritative read — used by tools and API."""
    with _LOCK:
        data = _load()
        if _FLAG_KEY in data:
            return bool(data[_FLAG_KEY])
        # Migrate legacy per-thread flag once
        if _LEGACY_THREAD in data:
            return bool(data[_LEGACY_THREAD])
        return False


def set_control_allowed(allowed: bool) -> tuple[bool, bool]:
    """
    Set control flag.
    Returns (new_value, changed).
    """
    allowed = bool(allowed)
    with _LOCK:
        data = _load()
        previous = bool(data.get(_FLAG_KEY, data.get(_LEGACY_THREAD, False)))
        data[_FLAG_KEY] = allowed
        # Keep legacy key in sync so older code paths stay consistent
        data[_LEGACY_THREAD] = allowed
        _save(data)
    changed = previous != allowed
    print(
        f"[TESrACT:perm] control={'ON' if allowed else 'OFF'}"
        f"{'' if changed else ' (unchanged)'}"
    )
    return allowed, changed


def parse_control_command(text: str) -> bool | None:
    """
    If the user message is only a control command, return True (grant) or False (revoke).
    Otherwise None — do not touch permission for normal requests.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if _REVOKE_ONLY_RE.match(raw):
        return False
    if _GRANT_ONLY_RE.match(raw):
        return True
    return None


def handle_control_command(text: str) -> tuple[str, bool] | None:
    """
    If text is a pure control command, apply it and return (reply, control_allowed).
    Otherwise return None.
    """
    intent = parse_control_command(text)
    if intent is None:
        return None
    was = is_control_allowed()
    now, _changed = set_control_allowed(intent)
    if intent:
        reply = ALREADY_ON_REPLY if was else GRANTED_REPLY
    else:
        reply = ALREADY_OFF_REPLY if not was else REVOKED_REPLY
    return reply, now
