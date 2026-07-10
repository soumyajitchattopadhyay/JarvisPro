"""
Zero-cost Email OTP authentication for TESrACT (smtplib).

In-memory OTP vault + session map (ephemeral per process). Chat *message*
history can still live in SQLite via session_store; graph checkpoints use
the authenticated email as LangGraph thread_id.
"""
from __future__ import annotations

import os
import random
import re
import smtplib
import ssl
import time
import uuid
from email.message import EmailMessage
from typing import Any

# Temporary OTP store: {email: {"otp": "123456", "expires": unix_ts}}
otp_vault: dict[str, dict[str, Any]] = {}

# Valid sessions: {session_token: email}
valid_sessions: dict[str, str] = {}

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_OTP_TTL_SECONDS = 5 * 60  # 5 minutes as specified


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(email)))


def email_allowed(email: str) -> bool:
    """Optional allow-list: AUTH_ALLOWED_EMAILS=a@x.com,b@y.com"""
    email = normalize_email(email)
    raw = (os.getenv("AUTH_ALLOWED_EMAILS") or "").strip()
    if not raw:
        return True
    allowed = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return email in allowed


def smtp_credentials() -> tuple[str, str, str, int, bool]:
    """Return (host, user, password, port, use_ssl)."""
    host = (os.getenv("AUTH_SMTP_HOST") or "smtp.gmail.com").strip()
    user = (os.getenv("AUTH_SMTP_USER") or os.getenv("AUTH_SMTP_FROM") or "").strip()
    password = (os.getenv("AUTH_SMTP_PASSWORD") or "").strip()
    port = int(os.getenv("AUTH_SMTP_PORT", "587") or "587")
    use_ssl = (os.getenv("AUTH_SMTP_SSL") or "").strip().lower() in ("1", "true", "yes")
    return host, user, password, port, use_ssl or port == 465


def smtp_configured() -> bool:
    _, user, password, _, _ = smtp_credentials()
    return bool(user and password)


def dev_echo_otp() -> bool:
    raw = (os.getenv("AUTH_DEV_ECHO_OTP") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def auth_enabled() -> bool:
    raw = (os.getenv("AUTH_ENABLED") or "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def auth_required() -> bool:
    """
    When true, /chat requires a session_token.
    Explicit AUTH_REQUIRED wins; otherwise auto-on if SMTP is configured.
    """
    if not auth_enabled():
        return False
    raw = (os.getenv("AUTH_REQUIRED") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return smtp_configured() or bool((os.getenv("AUTH_ALLOWED_EMAILS") or "").strip())


def generate_otp() -> str:
    return f"{random.SystemRandom().randint(0, 999_999):06d}"


def _purge_expired_otps() -> None:
    now = time.time()
    dead = [e for e, row in otp_vault.items() if float(row.get("expires") or 0) < now]
    for e in dead:
        otp_vault.pop(e, None)


def send_otp_email(
    target_email: str,
    sender_email: str | None = None,
    app_password: str | None = None,
) -> dict[str, Any]:
    """
    Generate a 6-digit OTP, store it for 5 minutes, and email it via smtplib.

    Returns a result dict: {ok, detail, ...}. Never raises to the route layer
    for expected validation errors; SMTP failures are returned as ok=False.
    """
    email = normalize_email(target_email)
    if not is_valid_email(email):
        return {"ok": False, "error": "invalid_email", "detail": "Enter a valid email address."}
    if not email_allowed(email):
        return {
            "ok": False,
            "error": "email_not_allowed",
            "detail": "This email is not on the operator allow-list.",
        }

    host, env_user, env_password, port, use_ssl = smtp_credentials()
    from_addr = (sender_email or env_user or os.getenv("AUTH_SMTP_FROM") or "").strip()
    password = (app_password or env_password or "").strip()
    user = (env_user or from_addr).strip()

    code = generate_otp()
    now = time.time()
    _purge_expired_otps()
    otp_vault[email] = {
        "otp": code,
        "expires": now + _OTP_TTL_SECONDS,
        "attempts": 0,
        "created": now,
    }

    echo = dev_echo_otp()
    skip_smtp = (os.getenv("AUTH_DEV_SKIP_SMTP") or "").strip().lower() in ("1", "true", "yes")
    mailed = False

    if echo:
        print(f"[TESrACT:auth] DEV OTP for {email}: {code}")

    if not user or not password:
        if echo:
            return {
                "ok": True,
                "email": email,
                "mailed": False,
                "expires_in": _OTP_TTL_SECONDS,
                "detail": "OTP generated (dev mode — check Mac logs).",
                "dev_code": code,
            }
        return {
            "ok": False,
            "error": "smtp_not_configured",
            "detail": (
                "SMTP not configured. Set AUTH_SMTP_USER and AUTH_SMTP_PASSWORD "
                "(Gmail App Password), or AUTH_DEV_ECHO_OTP=true for local testing."
            ),
        }

    if skip_smtp and echo:
        return {
            "ok": True,
            "email": email,
            "mailed": False,
            "expires_in": _OTP_TTL_SECONDS,
            "detail": "OTP generated (SMTP skipped in dev).",
            "dev_code": code,
        }

    msg = EmailMessage()
    msg["Subject"] = f"TESrACT access code: {code}"
    msg["From"] = from_addr or user
    msg["To"] = email
    msg.set_content(
        f"Your TESrACT one-time code is: {code}\n\n"
        f"It expires in 5 minutes.\n"
        f"If you did not request this, ignore this email.\n"
    )
    msg.add_alternative(
        f"""\
<html><body style="font-family:ui-monospace,Menlo,monospace;background:#050a0f;color:#d2f8ff;padding:24px">
  <div style="max-width:420px;margin:auto;border:1px solid rgba(0,255,255,.25);border-radius:12px;padding:24px;background:rgba(0,20,30,.9)">
    <div style="letter-spacing:3px;color:#00ffff;font-size:12px;margin-bottom:12px">TESrACT // CORE</div>
    <p style="margin:0 0 16px;opacity:.85">Operator access code</p>
    <div style="font-size:32px;letter-spacing:10px;color:#00fbff;font-weight:700">{code}</div>
    <p style="margin:18px 0 0;font-size:12px;opacity:.55">Expires in 5 minutes. Dispatched from your Mac via smtplib — no paid auth SaaS.</p>
  </div>
</body></html>
""",
        subtype="html",
    )

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
                smtp.login(user, password)
                smtp.send_message(msg)
        mailed = True
    except Exception as exc:
        print(f"[TESrACT:auth] SMTP send failed: {exc}")
        if not echo:
            # Roll back OTP if mail never left
            otp_vault.pop(email, None)
            return {
                "ok": False,
                "error": "smtp_failed",
                "detail": f"Could not send email: {exc}",
            }
        return {
            "ok": True,
            "email": email,
            "mailed": False,
            "expires_in": _OTP_TTL_SECONDS,
            "detail": f"SMTP failed ({exc}); dev OTP still valid.",
            "dev_code": code,
            "smtp_warning": str(exc),
        }

    out: dict[str, Any] = {
        "ok": True,
        "email": email,
        "mailed": mailed,
        "expires_in": _OTP_TTL_SECONDS,
        "detail": f"Code sent to {email}." if mailed else "Code generated.",
    }
    if echo:
        out["dev_code"] = code
    return out


def verify_otp(email: str, code: str) -> dict[str, Any]:
    """Validate OTP and issue a UUID session_token on success."""
    email = normalize_email(email)
    code = (code or "").strip().replace(" ", "")
    if not is_valid_email(email):
        return {"ok": False, "error": "invalid_email", "detail": "Invalid email."}
    if not re.fullmatch(r"\d{6}", code):
        return {"ok": False, "error": "invalid_code", "detail": "Enter the 6-digit code."}

    _purge_expired_otps()
    row = otp_vault.get(email)
    if not row:
        return {"ok": False, "error": "no_otp", "detail": "Request a code first."}
    if float(row.get("expires") or 0) < time.time():
        otp_vault.pop(email, None)
        return {"ok": False, "error": "expired", "detail": "Code expired — request a new one."}

    attempts = int(row.get("attempts") or 0)
    if attempts >= 8:
        otp_vault.pop(email, None)
        return {"ok": False, "error": "locked", "detail": "Too many attempts — request a new code."}

    if str(row.get("otp")) != code:
        row["attempts"] = attempts + 1
        left = max(0, 8 - row["attempts"])
        return {
            "ok": False,
            "error": "mismatch",
            "detail": f"Incorrect code ({left} attempts left).",
        }

    otp_vault.pop(email, None)
    token = str(uuid.uuid4())
    valid_sessions[token] = email
    return {
        "ok": True,
        "session_token": token,
        "token": token,  # alias for existing HUD clients
        "email": email,
        "detail": "Session unlocked.",
    }


def resolve_session_token(
    authorization: str | None = None,
    *,
    x_session_token: str | None = None,
    payload_token: str | None = None,
) -> str | None:
    """Extract session token from Bearer header, X-Session-Token, or body field."""
    if payload_token and str(payload_token).strip():
        return str(payload_token).strip()
    if x_session_token and str(x_session_token).strip():
        return str(x_session_token).strip()
    auth = (authorization or "").strip()
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        # Ignore brain shared secret masquerading as user session
        brain = (
            os.getenv("BRAIN_REGISTRY_SECRET")
            or os.getenv("LOCAL_INSTANCE_API_KEY")
            or ""
        ).strip()
        if brain and tok == brain:
            return None
        return tok or None
    return None


def email_for_token(token: str | None) -> str | None:
    if not token:
        return None
    return valid_sessions.get(token.strip())


def revoke_session(token: str | None) -> None:
    if not token:
        return
    valid_sessions.pop(token.strip(), None)


def status_public() -> dict[str, Any]:
    return {
        "auth_enabled": auth_enabled(),
        "auth_required": auth_required(),
        "smtp_configured": smtp_configured(),
        "dev_echo": dev_echo_otp(),
        "otp_ttl_seconds": _OTP_TTL_SECONDS,
        "engine": "smtplib+memory",
    }
