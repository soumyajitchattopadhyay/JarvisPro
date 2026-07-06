"""
TESrACT Rendezvous Server — bridges Colab ngrok URLs to the main application.

Colab registers its current public ngrok URL after each runtime restart.
The desktop / Render app fetches the latest URL instead of editing .env manually.

Deploy on Render (free tier):
  Build:  pip install -r requirements.txt
  Start:  uvicorn rendezvous_server:app --host 0.0.0.0 --port $PORT
  Env:    RENDEZVOUS_API_KEY=<shared-secret>   (required)

Endpoints:
  POST /register    — Colab publishes ngrok URL + API key
  GET  /active-url  — Main app fetches the latest registered URL
  GET  /health      — Render health check
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Shared secret — must match TESRACT_API_KEY in Colab and RENDEZVOUS_API_KEY in .env
RENDEZVOUS_API_KEY = os.getenv("RENDEZVOUS_API_KEY", "").strip()

# How long before a registration is considered stale (informational only).
STALE_AFTER_SECONDS = int(os.getenv("RENDEZVOUS_STALE_SECONDS", "900"))  # 15 min

# In-memory store (single Colab instance; sufficient for personal use).
_store: dict[str, Any] = {
    "url": "",
    "registered_at": 0.0,
    "last_seen": 0.0,
}

_NGROK_URL_RE = re.compile(
    r"^https://[a-z0-9][-a-z0-9.]*\.(ngrok-free\.app|ngrok-free\.dev|ngrok\.io|ngrok\.app)(/.*)?$",
    re.IGNORECASE,
)

app = FastAPI(
    title="TESrACT Rendezvous Server",
    version="1.0",
    description="Registers Colab ngrok URLs for automatic discovery by TESrACT.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    """Payload sent by Colab after ngrok tunnel is established."""

    url: str = Field(..., description="Public ngrok HTTPS URL (no trailing slash required)")
    api_key: str = Field(..., min_length=1, description="Must match RENDEZVOUS_API_KEY on this server")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith("https://"):
            raise ValueError("URL must use HTTPS")
        if not _NGROK_URL_RE.match(url):
            raise ValueError("URL must be a valid ngrok HTTPS endpoint")
        return url


def _require_server_key() -> None:
    if not RENDEZVOUS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Rendezvous server is not configured — set RENDEZVOUS_API_KEY",
        )


def _check_api_key(provided: str | None, *, context: str) -> None:
    _require_server_key()
    if not provided or provided.strip() != RENDEZVOUS_API_KEY:
        raise HTTPException(status_code=401, detail=f"Invalid API key ({context})")


def _auth_from_bearer(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


def _iso(ts: float) -> str | None:
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _age_seconds(ts: float) -> float | None:
    if ts <= 0:
        return None
    return round(time.time() - ts, 1)


@app.get("/health")
def health() -> dict[str, Any]:
    """Render health check — does not expose the registered URL."""
    return {
        "status": "ok",
        "configured": bool(RENDEZVOUS_API_KEY),
        "has_active_url": bool(_store["url"]),
        "last_seen": _iso(float(_store["last_seen"])),
        "stale_after_seconds": STALE_AFTER_SECONDS,
    }


@app.post("/register")
def register(body: RegisterRequest) -> dict[str, Any]:
    """
    Colab calls this immediately after ngrok exposes a public URL.
    Updates the in-memory registry and refreshes last_seen.
    """
    _check_api_key(body.api_key, context="register")

    now = time.time()
    _store["url"] = body.url
    _store["registered_at"] = now
    _store["last_seen"] = now

    return {
        "status": "registered",
        "url": body.url,
        "registered_at": _iso(now),
        "last_seen": _iso(now),
    }


@app.get("/active-url")
def active_url(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """
    Main application fetches the latest Colab ngrok URL.
    Requires Authorization: Bearer <RENDEZVOUS_API_KEY> when the server is secured.
    """
    token = _auth_from_bearer(authorization)
    _check_api_key(token, context="active-url")

    url = str(_store["url"] or "")
    if not url:
        raise HTTPException(status_code=404, detail="No Colab URL registered yet")

    last_seen = float(_store["last_seen"])
    age = _age_seconds(last_seen)
    stale = age is not None and age > STALE_AFTER_SECONDS

    return {
        "url": url,
        "registered_at": _iso(float(_store["registered_at"])),
        "last_seen": _iso(last_seen),
        "age_seconds": age,
        "stale": stale,
        "stale_after_seconds": STALE_AFTER_SECONDS,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("rendezvous_server:app", host="0.0.0.0", port=port, reload=False)