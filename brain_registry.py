"""
Active Mac-brain tunnel registry for TESrACT hybrid routing.

Cloudflare quick tunnels rotate every ~24h (and on Wi-Fi/process drops).
The Mac runs tunnel_manager.py which posts each new trycloudflare.com URL here.
Render (or any host) stores it and exposes a stable "global brain" link that
always redirects / returns the live tunnel.

Persistence: agent_data/brain_url.json (survives process restarts on the same host).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "agent_data"
_REGISTRY_FILE = _DATA_DIR / "brain_url.json"
_LOCK = threading.RLock()

# In-memory cache (authoritative while process is up; file is the backup).
_STATE: dict[str, Any] = {
    "brain_url": None,
    "updated_at": None,
    "source": None,
    "last_heartbeat_at": None,
}

_URL_RE = re.compile(r"^https://[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$")

# Hosts we accept by default (ephemeral tunnels + common personal tunnels).
_ALLOWED_HOST_SUFFIXES = (
    ".trycloudflare.com",
    ".cfargotunnel.com",
    ".ngrok-free.app",
    ".ngrok-free.dev",
    ".ngrok.io",
    ".loca.lt",
    # Pinggy SSH tunnels (tunnel_manager.py)
    ".free.pinggy.net",
    ".run.pinggy-free.link",
    ".a.free.pinggy.link",
    ".pinggy.link",
    ".pinggy.io",
)


def _secret() -> str:
    return (
        os.getenv("BRAIN_REGISTRY_SECRET")
        or os.getenv("TESRACT_BRAIN_SECRET")
        or os.getenv("LOCAL_INSTANCE_API_KEY")
        or ""
    ).strip()


def secret_configured() -> bool:
    return bool(_secret())


def verify_secret(provided: str | None) -> bool:
    """Constant-time compare against the shared brain secret."""
    expected = _secret()
    if not expected:
        # Fail closed when no secret is configured — prevent open rewrite of the brain URL.
        return False
    try:
        import hmac

        return hmac.compare_digest((provided or "").strip(), expected)
    except Exception:
        return (provided or "").strip() == expected


def _allow_any_host() -> bool:
    return os.getenv("BRAIN_URL_ALLOW_ANY", "").strip().lower() in ("1", "true", "yes")


def validate_brain_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Only https public tunnel hosts by default."""
    raw = (url or "").strip().rstrip("/")
    if not raw:
        return False, "brain_url is empty"
    if not _URL_RE.match(raw):
        return False, "brain_url must be a valid https URL"
    try:
        parsed = urlparse(raw)
    except Exception as exc:
        return False, f"invalid URL: {exc}"
    if parsed.scheme != "https":
        return False, "brain_url must use https"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "brain_url missing host"
    # Block obvious SSRF targets
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local"):
        return False, "localhost / .local URLs are not allowed"
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("172."):
        return False, "private network URLs are not allowed"
    if _allow_any_host():
        return True, "ok"
    if any(host.endswith(suffix) or host == suffix.lstrip(".") for suffix in _ALLOWED_HOST_SUFFIXES):
        return True, "ok"
    return (
        False,
        "host not in allow-list (trycloudflare / ngrok / pinggy). "
        "Set BRAIN_URL_ALLOW_ANY=true to accept other https hosts.",
    )


def _load_file() -> dict[str, Any]:
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_file(state: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(_REGISTRY_FILE)


def _hydrate_from_disk() -> None:
    """Load file into memory once if memory is empty."""
    if _STATE.get("brain_url"):
        return
    data = _load_file()
    url = (data.get("brain_url") or "").strip().rstrip("/")
    if not url:
        # Fall back to static env so first boot still works before tunnel_manager posts.
        env_url = (os.getenv("LOCAL_INSTANCE_URL") or "").strip().rstrip("/")
        if env_url:
            _STATE["brain_url"] = env_url
            _STATE["updated_at"] = data.get("updated_at")
            _STATE["source"] = "env"
        return
    _STATE["brain_url"] = url
    _STATE["updated_at"] = data.get("updated_at")
    _STATE["source"] = data.get("source") or "disk"
    _STATE["last_heartbeat_at"] = data.get("last_heartbeat_at")


def get_brain_url() -> str | None:
    """Active Mac tunnel URL, or None if unknown."""
    with _LOCK:
        _hydrate_from_disk()
        url = _STATE.get("brain_url")
        return str(url).rstrip("/") if url else None


def set_brain_url(
    url: str,
    *,
    source: str = "tunnel_manager",
    heartbeat: bool = False,
) -> dict[str, Any]:
    """
    Register / refresh the active brain URL.
    Returns the public status dict.
    """
    ok, reason = validate_brain_url(url)
    if not ok:
        raise ValueError(reason)

    cleaned = url.strip().rstrip("/")
    now = datetime.now(timezone.utc).isoformat()

    with _LOCK:
        previous = _STATE.get("brain_url")
        changed = previous != cleaned
        _STATE["brain_url"] = cleaned
        _STATE["source"] = source
        _STATE["last_heartbeat_at"] = now
        if changed or not _STATE.get("updated_at"):
            _STATE["updated_at"] = now
        snapshot = {
            "brain_url": _STATE["brain_url"],
            "updated_at": _STATE["updated_at"],
            "last_heartbeat_at": _STATE["last_heartbeat_at"],
            "source": _STATE["source"],
            "changed": changed,
            "heartbeat": heartbeat and not changed,
        }
        try:
            _save_file({
                "brain_url": _STATE["brain_url"],
                "updated_at": _STATE["updated_at"],
                "last_heartbeat_at": _STATE["last_heartbeat_at"],
                "source": _STATE["source"],
            })
        except Exception as exc:
            snapshot["persist_error"] = str(exc)

    if changed:
        _notify_url_changed(cleaned)
    return snapshot


def clear_brain_url() -> None:
    with _LOCK:
        _STATE["brain_url"] = None
        _STATE["updated_at"] = None
        _STATE["source"] = None
        _STATE["last_heartbeat_at"] = None
        try:
            if _REGISTRY_FILE.exists():
                _REGISTRY_FILE.unlink()
        except Exception:
            pass


def status() -> dict[str, Any]:
    with _LOCK:
        _hydrate_from_disk()
        url = _STATE.get("brain_url")
        updated = _STATE.get("updated_at")
        age_s = None
        if updated:
            try:
                ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                age_s = max(0, int(time.time() - ts.timestamp()))
            except Exception:
                age_s = None
        return {
            "online": bool(url),
            "brain_url": url,
            "updated_at": updated,
            "last_heartbeat_at": _STATE.get("last_heartbeat_at"),
            "source": _STATE.get("source"),
            "age_seconds": age_s,
            "secret_configured": secret_configured(),
            "global_link_paths": ["/go", "/brain", "/api/global-brain"],
        }


def _notify_url_changed(url: str) -> None:
    """Invalidate hybrid-router health cache so the new tunnel is used immediately."""
    try:
        import llm_router

        llm_router.on_local_instance_url_changed(url)
    except Exception as exc:
        print(f"[TESrACT:brain] health-cache notify skipped: {exc}")
