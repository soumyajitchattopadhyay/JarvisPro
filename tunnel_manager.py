#!/usr/bin/env python3
"""
Self-healing Pinggy SSH quick-tunnel manager for TESrACT.

What it does:
  1. Generates a dedicated SSH key to bypass Pinggy password prompts automatically.
  2. Launches an SSH reverse tunnel to Pinggy (unlimited bandwidth).
  3. Scrapes the ephemeral https://*.pinggy.link URL from the terminal output.
  4. POSTs it to your stable Render app: POST /api/update-brain
  5. Proactively restarts the tunnel every 55 minutes to prevent Pinggy's 60m drop limit.
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
except ImportError: 
    httpx = None 

# Match Pinggy's secure URLs
_URL_RE = re.compile(
    r"https://[a-zA-Z0-9-]+\.(?:free\.pinggy\.net|run\.pinggy-free\.link)",
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
        _log("BRAIN_REGISTRY_SECRET is not set — cannot authenticate with Render")
        return False

    endpoint = f"{base}/api/update-brain"
    path = "/api/update-brain"
    payload = {
        "brain_url": tunnel_url.rstrip("/"),
        "secret_key": secret, 
        "heartbeat": heartbeat,
        "source": "tunnel_manager",
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "TESrACT-tunnel-manager/2.0-Pinggy",
    }
    
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
    global _child
    import threading

    # 1. Generate SSH Key Pair to bypass password prompt
    key_path = os.path.expanduser("~/.ssh/tesract_pinggy")
    if not os.path.exists(key_path):
        _log("Generating dedicated SSH key for Pinggy to bypass password prompts...")
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", ""],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # 2. Configure SSH Command
    target = f"localhost:{port}"
    cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        "-o", "BatchMode=yes",
        "-p", "443",
        f"-R0:{target}",
        "a.pinggy.io"
    ]
    _log(f"Starting SSH: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        _log("SSH binary not found. This is impossible on a Mac. Matrix is broken.")
        return 127

    _child = process
    tunnel_url: list[str | None] = [None]  
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
            
            # Catch errors but ignore massive ASCII art
            if any(k in lower for k in ("err", "fail", "expire", "refused")):
                clean_text = re.sub(r'\s+', ' ', text).strip()
                if clean_text:
                    _log(f"ssh| {clean_text}")
                    
            # Scrape the URL
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
        url_event.set() 

    reader = threading.Thread(target=_reader, daemon=True, name="ssh-reader")
    reader.start()

    try:
        if not url_event.wait(timeout=90.0) or not tunnel_url[0]:
            if process.poll() is not None:
                _log("No pinggy URL seen — tunnel process ended early")
                return int(process.returncode or 1)
            _log("Still waiting for pinggy URL…")
            while _running and process.poll() is None and not tunnel_url[0]:
                time.sleep(0.5)
            if not tunnel_url[0]:
                return int(process.returncode or 1)

        # 3. The 55-Minute Auto-Recycle Loop
        start_time = time.time()
        next_beat = time.time() + heartbeat_seconds()
        
        while _running and process.poll() is None:
            time.sleep(1.0)
            now = time.time()
            
            if now - start_time >= 3300: # 55 minutes
                _log("55-minute Pinggy limit reached. Proactively recycling tunnel...")
                break

            current = tunnel_url[0]
            if current and now >= next_beat:
                post_brain_url(current, heartbeat=True)
                next_beat = now + heartbeat_seconds()

        code = process.wait() if process.poll() is None else (process.returncode or 0)
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

    _log("=== TESrACT Self-Healing Tunnel Manager (Pinggy SSH) ===")
    _log(f"local target : http://127.0.0.1:{port}")
    if base:
        _log(f"global link  : {base}/go")

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
