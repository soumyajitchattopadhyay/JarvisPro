#!/usr/bin/env python3
"""
Self-healing Cloudflare quick-tunnel manager for TESrACT.

What it does:
  1. Launches `cloudflared tunnel --url http://localhost:<PORT>` against your local TESrACT.
  2. Scrapes the ephemeral https://*.trycloudflare.com URL from cloudflared logs.
  3. POSTs it to your stable Render app: POST /api/update-brain
  4. When the tunnel dies (~24h free limit, Wi-Fi blip, process crash), restarts and re-registers.

Usage (on your Mac, while TESrACT is running on the local port):
    python tunnel_manager.py

Env (see .env.example):
    TESRACT_LOCAL_PORT=8000
    TESRACT_RENDER_URL=https://your-app.onrender.com
    BRAIN_REGISTRY_SECRET=long-random-secret   # must match Render
    LOCAL_INSTANCE_API_KEY=...                # fallback secret if BRAIN_REGISTRY_SECRET unset
    CLOUDFLARED_BIN=cloudflared               # optional path override
    TUNNEL_HEARTBEAT_SECONDS=300              # re-post URL so Render restarts recover
    TUNNEL_RESTART_DELAY_SECONDS=5
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

# cloudflared prints the public URL on stdout/stderr — match common formats
_URL_RE = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com",
    re.IGNORECASE,
)

_running = True
_child: subprocess.Popen | None = None


def _log(msg: str) -> None:
    print(f"[tunnel_manager] {msg}", flush=True)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def local_port() -> int:
    return _env_int("TESRACT_LOCAL_PORT", _env_int("PORT", 8000))


def render_base_url() -> str:
    url = (
        os.getenv("TESRACT_RENDER_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or os.getenv("RENDER_PUBLIC_URL")
        or ""
    ).strip().rstrip("/")
    return url


def registry_secret() -> str:
    return (
        os.getenv("BRAIN_REGISTRY_SECRET")
        or os.getenv("TESRACT_BRAIN_SECRET")
        or os.getenv("LOCAL_INSTANCE_API_KEY")
        or ""
    ).strip()


def cloudflared_bin() -> str:
    override = (os.getenv("CLOUDFLARED_BIN") or "").strip()
    if override:
        return override
    found = shutil.which("cloudflared")
    if found:
        return found
    # common Homebrew locations
    for candidate in (
        "/opt/homebrew/bin/cloudflared",
        "/usr/local/bin/cloudflared",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "cloudflared"


def heartbeat_seconds() -> int:
    return max(60, _env_int("TUNNEL_HEARTBEAT_SECONDS", 300))


def restart_delay() -> float:
    return max(1.0, float(_env_int("TUNNEL_RESTART_DELAY_SECONDS", 5)))


def post_brain_url(tunnel_url: str, *, heartbeat: bool = False) -> bool:
    base = render_base_url()
    secret = registry_secret()
    if not base:
        _log("TESRACT_RENDER_URL is not set — cannot register tunnel with Render")
        return False
    if not secret:
        _log(
            "BRAIN_REGISTRY_SECRET (or LOCAL_INSTANCE_API_KEY) is not set — "
            "cannot authenticate with Render"
        )
        return False

    endpoint = f"{base}/api/update-brain"
    path = "/api/update-brain"
    payload = {
        "brain_url": tunnel_url.rstrip("/"),
        "secret_key": secret,  # legacy body field (HMAC headers preferred)
        "heartbeat": heartbeat,
        "source": "tunnel_manager",
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "TESrACT-tunnel-manager/1.0",
    }
    # Cryptographic handshake — HMAC-SHA256 + static token + Bearer
    try:
        import brain_auth

        headers.update(
            brain_auth.sign(
                method="POST",
                path=path,
                body=body_bytes,
                secret=secret,
            )
        )
    except Exception as exc:
        _log(f"HMAC sign failed ({exc}); falling back to Bearer secret only")
        headers["Authorization"] = f"Bearer {secret}"

    if httpx is None:
        _log("httpx not installed — pip install httpx")
        return False

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            res = client.post(endpoint, content=body_bytes, headers=headers)
        if res.status_code >= 400:
            _log(f"Register failed HTTP {res.status_code}: {res.text[:300]}")
            return False
        data = {}
        try:
            data = res.json()
        except Exception:
            pass
        kind = "heartbeat" if heartbeat else "register"
        _log(
            f"{kind} OK → {base}  "
            f"(global link: {base}/go)  "
            f"active={data.get('active_url') or tunnel_url}"
        )
        return True
    except Exception as exc:
        _log(f"Register failed: {exc}")
        return False


def wait_for_local_server(port: int, timeout: float = 60.0) -> bool:
    """Optional: wait until local TESrACT answers /health."""
    if httpx is None:
        return True
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline and _running:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(url)
            if res.status_code == 200:
                _log(f"Local TESrACT is up at {url}")
                return True
        except Exception:
            pass
        time.sleep(1.0)
    _log(f"Local server not ready at {url} (continuing anyway)")
    return False


def _handle_signal(signum: int, _frame: object) -> None:
    global _running
    _log(f"signal {signum} — shutting down")
    _running = False
    if _child and _child.poll() is None:
        try:
            _child.terminate()
        except Exception:
            pass


def run_one_tunnel(port: int) -> int:
    """
    Start cloudflared, register URL, heartbeat until process exits.
    Returns cloudflared exit code.
    """
    global _child
    import threading

    bin_path = cloudflared_bin()
    target = f"http://127.0.0.1:{port}"
    cmd = [bin_path, "tunnel", "--no-autoupdate", "--url", target]
    _log(f"Starting: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        _log(
            f"cloudflared not found ({bin_path}). "
            "Install: brew install cloudflare/cloudflare/cloudflared"
        )
        return 127

    _child = process
    tunnel_url: list[str | None] = [None]  # mutable cell for reader thread
    url_event = threading.Event()

    def _reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if not _running:
                break
            text = line.rstrip()
            if not text:
                continue
            lower = text.lower()
            if any(k in lower for k in ("err", "fail", "trycloudflare", "registered", "disconnected", "retry")):
                _log(f"cloudflared| {text}")
            match = _URL_RE.search(text)
            if match:
                found = match.group(0).rstrip("/")
                prev = tunnel_url[0]
                if found != prev:
                    tunnel_url[0] = found
                    if prev is None:
                        _log(f"Mac brain tunnel online: {found}")
                        post_brain_url(found, heartbeat=False)
                        url_event.set()
                    else:
                        _log(f"Tunnel URL changed: {found}")
                        post_brain_url(found, heartbeat=False)
                        url_event.set()
        url_event.set()  # unblock waiter on process exit

    reader = threading.Thread(target=_reader, daemon=True, name="cloudflared-reader")
    reader.start()

    try:
        # Wait up to 90s for first public URL
        if not url_event.wait(timeout=90.0) or not tunnel_url[0]:
            if process.poll() is not None:
                _log("No trycloudflare.com URL seen — tunnel process ended early")
                return int(process.returncode or 1)
            # Still running but no URL yet — keep waiting until process dies
            _log("Still waiting for trycloudflare.com URL…")
            while _running and process.poll() is None and not tunnel_url[0]:
                time.sleep(0.5)
            if not tunnel_url[0]:
                _log("Tunnel never published a public URL")
                return int(process.returncode or 1)

        # Heartbeat until cloudflared exits
        next_beat = time.time() + heartbeat_seconds()
        while _running and process.poll() is None:
            time.sleep(1.0)
            current = tunnel_url[0]
            if current and time.time() >= next_beat:
                post_brain_url(current, heartbeat=True)
                next_beat = time.time() + heartbeat_seconds()

        code = process.wait() if process.poll() is None else (process.returncode or 0)
        _log(f"cloudflared exited with code {code}")
        return int(code or 0)
    finally:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        try:
            reader.join(timeout=2)
        except Exception:
            pass
        _child = None


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    port = local_port()
    base = render_base_url()
    secret = registry_secret()

    _log("=== TESrACT Self-Healing Tunnel Manager ===")
    _log(f"local target : http://127.0.0.1:{port}")
    _log(f"render base  : {base or '(NOT SET — set TESRACT_RENDER_URL)'}")
    _log(f"secret set   : {'yes' if secret else 'NO — set BRAIN_REGISTRY_SECRET'}")
    _log(f"cloudflared  : {cloudflared_bin()}")
    if base:
        _log(f"global link  : {base}/go")

    if not base or not secret:
        _log(
            "WARNING: registration will fail until TESRACT_RENDER_URL and "
            "BRAIN_REGISTRY_SECRET are set in .env (matching Render env vars)."
        )

    wait_for_local_server(port, timeout=30.0)

    while _running:
        code = run_one_tunnel(port)
        if not _running:
            break
        delay = restart_delay()
        _log(f"Tunnel dropped (exit {code}). Restarting in {delay}s…")
        time.sleep(delay)

    _log("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
