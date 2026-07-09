"""
Cryptographic handshake for TESrACT Brain endpoints.

Protects public-facing brain routes from unauthenticated access over
tunnels / the internet. Uses HMAC-SHA256 over method, path, body hash,
and a skew-checked timestamp.

Headers (preferred):
  X-Brain-Timestamp: <unix seconds>
  X-Brain-Signature: <hex HMAC-SHA256 of canonical string>
  Authorization: Bearer <BRAIN_REGISTRY_SECRET>   # accepted alternative

Canonical string:
  {timestamp}.{METHOD}.{path}.{sha256_hex(body)}

Shared secret sources (first non-empty wins):
  BRAIN_REGISTRY_SECRET | TESRACT_BRAIN_SECRET | LOCAL_INSTANCE_API_KEY

Localhost bypass (GUI on the same machine only):
  BRAIN_AUTH_LOCALHOST_BYPASS=true  (default: true)
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

# Max accepted clock skew for signed requests (seconds).
_MAX_SKEW_SECONDS = int(os.getenv("BRAIN_AUTH_MAX_SKEW_SECONDS", "300") or "300")

# Header names
HDR_TIMESTAMP = "X-Brain-Timestamp"
HDR_SIGNATURE = "X-Brain-Signature"
HDR_TOKEN = "X-Brain-Token"  # optional: hex(sha256(secret)) constant token


def get_shared_secret() -> str:
    return (
        os.getenv("BRAIN_REGISTRY_SECRET")
        or os.getenv("TESRACT_BRAIN_SECRET")
        or os.getenv("LOCAL_INSTANCE_API_KEY")
        or ""
    ).strip()


def secret_configured() -> bool:
    return bool(get_shared_secret())


def _localhost_bypass_enabled() -> bool:
    raw = (os.getenv("BRAIN_AUTH_LOCALHOST_BYPASS") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def is_localhost_request(request: Request) -> bool:
    """True when the TCP peer is loopback (local HUD / CLI)."""
    client = request.client
    if client is None:
        return False
    host = (client.host or "").strip().lower()
    return host in ("127.0.0.1", "::1", "localhost")


def body_sha256_hex(body: bytes | str | None) -> str:
    if body is None:
        data = b""
    elif isinstance(body, str):
        data = body.encode("utf-8")
    else:
        data = body
    return hashlib.sha256(data).hexdigest()


def static_token_hex(secret: str | None = None) -> str:
    """SHA-256 of the shared secret — usable as a fixed header token."""
    sec = (secret if secret is not None else get_shared_secret()).encode("utf-8")
    return hashlib.sha256(sec).hexdigest()


def canonical_string(
    *,
    timestamp: str | int,
    method: str,
    path: str,
    body_hash: str,
) -> str:
    return f"{timestamp}.{method.upper()}.{path}.{body_hash}"


def sign(
    *,
    method: str,
    path: str,
    body: bytes | str | None = b"",
    timestamp: int | None = None,
    secret: str | None = None,
) -> dict[str, str]:
    """
    Build auth headers for an outbound request to a TESrACT brain endpoint.
    """
    sec = (secret if secret is not None else get_shared_secret()).strip()
    if not sec:
        raise ValueError("BRAIN_REGISTRY_SECRET is not configured — cannot sign request")
    ts = int(time.time() if timestamp is None else timestamp)
    bhash = body_sha256_hex(body)
    message = canonical_string(
        timestamp=ts,
        method=method,
        path=path,
        body_hash=bhash,
    )
    sig = hmac.new(
        sec.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        HDR_TIMESTAMP: str(ts),
        HDR_SIGNATURE: sig,
        HDR_TOKEN: static_token_hex(sec),
        "Authorization": f"Bearer {sec}",
    }


def _constant_time_eq(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False


def verify_bearer_or_raw_secret(provided: str | None, secret: str | None = None) -> bool:
    expected = (secret if secret is not None else get_shared_secret()).strip()
    if not expected:
        return False
    got = (provided or "").strip()
    if got.lower().startswith("bearer "):
        got = got[7:].strip()
    return _constant_time_eq(got, expected)


def verify_static_token(token: str | None, secret: str | None = None) -> bool:
    expected = static_token_hex(secret)
    got = (token or "").strip().lower()
    return bool(got) and _constant_time_eq(got, expected)


def verify_hmac_signature(
    *,
    method: str,
    path: str,
    body: bytes | str | None,
    timestamp: str | None,
    signature: str | None,
    secret: str | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason)."""
    sec = (secret if secret is not None else get_shared_secret()).strip()
    if not sec:
        return False, "brain secret not configured"
    if not timestamp or not signature:
        return False, "missing timestamp or signature"
    try:
        ts = int(str(timestamp).strip())
    except ValueError:
        return False, "invalid timestamp"
    now = int(time.time())
    if abs(now - ts) > max(30, _MAX_SKEW_SECONDS):
        return False, "timestamp outside allowed skew"

    bhash = body_sha256_hex(body)
    message = canonical_string(
        timestamp=ts,
        method=method,
        path=path,
        body_hash=bhash,
    )
    expected = hmac.new(
        sec.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    got = (signature or "").strip().lower()
    if not _constant_time_eq(got, expected):
        return False, "invalid signature"
    return True, "ok"


def extract_raw_secret_from_request(request: Request, payload: dict | None = None) -> str:
    """Pull a raw shared secret from headers or JSON body (legacy clients)."""
    # Preferred explicit headers
    for key in ("X-Brain-Secret", "X-TESrACT-Secret"):
        val = request.headers.get(key)
        if val:
            return val.strip()
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if payload and isinstance(payload, dict):
        for key in ("secret_key", "secret", "brain_secret"):
            if payload.get(key):
                return str(payload.get(key)).strip()
    return ""


async def read_body_bytes(request: Request) -> bytes:
    """Read and cache request body so downstream handlers can re-parse JSON."""
    # Starlette caches body after first read
    return await request.body()


def unauthorized_response(detail: str = "Unauthorized") -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "error": "unauthorized",
            "detail": detail,
            "execution_target": "none",
            "action": "LOGIC_ONLY",
            "payload": {},
        },
        status_code=401,
        headers={"WWW-Authenticate": "Bearer, HMAC-SHA256"},
    )


def verify_request(
    request: Request,
    body: bytes | str | None = b"",
    *,
    payload: dict | None = None,
    allow_localhost_bypass: bool | None = None,
) -> tuple[bool, str]:
    """
    Strict brain auth gate.

    Accepts (in order):
      1. Valid HMAC signature headers
      2. Valid X-Brain-Token (sha256 of secret)
      3. Valid raw secret via Authorization Bearer / body secret_key

    Localhost peers may bypass when BRAIN_AUTH_LOCALHOST_BYPASS is enabled
    (default true) so the on-machine HUD keeps working without embedding secrets.
    """
    bypass = (
        _localhost_bypass_enabled()
        if allow_localhost_bypass is None
        else allow_localhost_bypass
    )
    if bypass and is_localhost_request(request):
        return True, "localhost_bypass"

    secret = get_shared_secret()
    if not secret:
        # Fail closed for non-local public exposure when no secret is set.
        return False, "BRAIN_REGISTRY_SECRET not configured on this host"

    path = request.url.path
    method = request.method

    # 1) HMAC
    ts = request.headers.get(HDR_TIMESTAMP)
    sig = request.headers.get(HDR_SIGNATURE)
    if ts and sig:
        ok, reason = verify_hmac_signature(
            method=method,
            path=path,
            body=body,
            timestamp=ts,
            signature=sig,
            secret=secret,
        )
        if ok:
            return True, "hmac"
        return False, reason

    # 2) Static SHA-256 token header
    token = request.headers.get(HDR_TOKEN)
    if token and verify_static_token(token, secret):
        return True, "static_token"

    # 3) Raw shared secret (Bearer / body) — constant-time compare
    raw = extract_raw_secret_from_request(request, payload)
    if raw and verify_bearer_or_raw_secret(raw, secret):
        return True, "shared_secret"

    return False, "missing or invalid brain credentials"


async def gate_brain_auth(
    request: Request,
) -> tuple[bytes, dict[str, Any] | None, JSONResponse | None]:
    """Return (body, payload, error_response). error_response is None when authorized."""
    import json

    body = await read_body_bytes(request)
    payload: dict[str, Any] | None = None
    if body:
        try:
            data = json.loads(body.decode("utf-8"))
            if isinstance(data, dict):
                payload = data
        except Exception:
            payload = None
    ok, reason = verify_request(request, body, payload=payload)
    if not ok:
        print(f"[TESrACT:auth] REJECT {request.method} {request.url.path}: {reason}")
        return body, payload, unauthorized_response(reason)
    return body, payload, None
