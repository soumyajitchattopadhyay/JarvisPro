"""
Zero-cost email OTP authentication for TESrACT.

The Mac brain:
  1. Generates a 6-digit OTP, stores a hash in local SQLite
  2. Dispatches it via smtplib + free Gmail/app-password (or any SMTP)
  3. Verifies the code and issues a long-lived session token

No Twilio / SendGrid / Supabase required.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import random
import re
import smtplib
import ssl
import time
from email.message import EmailMessage
from typing import Any

import session_store

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def auth_enabled() -> bool:
    """Auth is on unless explicitly disabled. SMTP optional in dev echo mode."""
    raw = (os.getenv("AUTH_ENABLED") or "true").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def auth_required() -> bool:
    """When true, /chat requires a valid session (unless localhost bypass)."""
    if not auth_enabled():
        return False
    raw = (os.getenv("AUTH_REQUIRED") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    # Auto: lock the HUD once SMTP (or an email allow-list) is configured.
    return smtp_configured() or bool((os.getenv("AUTH_ALLOWED_EMAILS") or "").strip())


def otp_ttl_seconds() -> int:
    return max(60, int(os.getenv("AUTH_OTP_TTL_SECONDS", "600") or "600"))


def session_ttl_days() -> int:
    return max(1, int(os.getenv("AUTH_SESSION_DAYS", "30") or "30"))


def max_otp_attempts() -> int:
    return max(3, int(os.getenv("AUTH_OTP_MAX_ATTEMPTS", "8") or "8"))


def dev_echo_otp() -> bool:
    """Log OTP and optionally return it in API (local testing without SMTP)."""
    raw = (os.getenv("AUTH_DEV_ECHO_OTP") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def smtp_configured() -> bool:
    return bool(
        (os.getenv("AUTH_SMTP_HOST") or "").strip()
        and (os.getenv("AUTH_SMTP_USER") or "").strip()
        and (os.getenv("AUTH_SMTP_PASSWORD") or "").strip()
    )


def _pepper() -> str:
    return (
        os.getenv("AUTH_OTP_PEPPER")
        or os.getenv("BRAIN_REGISTRY_SECRET")
        or os.getenv("TESRACT_BRAIN_SECRET")
        or "tesract-local-otp-pepper"
    ).strip()


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


def _hash_code(email: str, code: str) -> str:
    msg = f"{normalize_email(email)}:{code.strip()}".encode("utf-8")
    return hmac.new(_pepper().encode("utf-8"), msg, hashlib.sha256).hexdigest()


def generate_otp_code() -> str:
    return f"{random.SystemRandom().randint(0, 999999):06d}"


def send_otp_email(to_email: str, code: str) -> None:
    host = (os.getenv("AUTH_SMTP_HOST") or "smtp.gmail.com").strip()
    port = int(os.getenv("AUTH_SMTP_PORT", "587") or "587")
    user = (os.getenv("AUTH_SMTP_USER") or "").strip()
    password = (os.getenv("AUTH_SMTP_PASSWORD") or "").strip()
    from_addr = (os.getenv("AUTH_SMTP_FROM") or user).strip()
    use_ssl = (os.getenv("AUTH_SMTP_SSL") or "").strip().lower() in ("1", "true", "yes")

    if not user or not password:
        raise RuntimeError(
            "SMTP not configured. Set AUTH_SMTP_USER and AUTH_SMTP_PASSWORD "
            "(Gmail: create an App Password)."
        )

    msg = EmailMessage()
    msg["Subject"] = f"TESrACT access code: {code}"
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(
        f"Your TESrACT one-time code is: {code}\n\n"
        f"It expires in {otp_ttl_seconds() // 60} minutes.\n"
        f"If you did not request this, ignore this email.\n"
    )
    msg.add_alternative(
        f"""\
<html><body style="font-family:ui-monospace,Menlo,monospace;background:#050a0f;color:#d2f8ff;padding:24px">
  <div style="max-width:420px;margin:auto;border:1px solid rgba(0,255,255,.25);border-radius:12px;padding:24px;background:rgba(0,20,30,.9)">
    <div style="letter-spacing:3px;color:#00ffff;font-size:12px;margin-bottom:12px">TESrACT // CORE</div>
    <p style="margin:0 0 16px;opacity:.85">Operator access code</p>
    <div style="font-size:32px;letter-spacing:10px;color:#00fbff;font-weight:700">{code}</div>
    <p style="margin:18px 0 0;font-size:12px;opacity:.55">Expires in {otp_ttl_seconds() // 60} minutes. Local Mac dispatch — no third-party auth SaaS.</p>
  </div>
</body></html>
""",
        subtype="html",
    )

    if use_ssl or port == 465:
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


def request_otp(email: str) -> dict[str, Any]:
    email = normalize_email(email)
    if not is_valid_email(email):
        return {"ok": False, "error": "invalid_email", "detail": "Enter a valid email address."}
    if not email_allowed(email):
        return {
            "ok": False,
            "error": "email_not_allowed",
            "detail": "This email is not on the operator allow-list.",
        }

    code = generate_otp_code()
    session_store.save_otp(email, _hash_code(email, code), ttl_seconds=otp_ttl_seconds())

    echo = dev_echo_otp()
    mailed = False
    mail_error = ""
    if smtp_configured() and not (
        echo and (os.getenv("AUTH_DEV_SKIP_SMTP") or "").strip().lower() in ("1", "true", "yes")
    ):
        try:
            send_otp_email(email, code)
            mailed = True
        except Exception as exc:
            mail_error = str(exc)
            print(f"[TESrACT:auth] SMTP send failed: {exc}")
            if not echo:
                return {
                    "ok": False,
                    "error": "smtp_failed",
                    "detail": f"Could not send email: {exc}",
                }
    elif not smtp_configured() and not echo:
        return {
            "ok": False,
            "error": "smtp_not_configured",
            "detail": (
                "OTP email not configured on the Mac. Set AUTH_SMTP_* env vars "
                "or AUTH_DEV_ECHO_OTP=true for local testing."
            ),
        }

    if echo:
        print(f"[TESrACT:auth] DEV OTP for {email}: {code}")

    out: dict[str, Any] = {
        "ok": True,
        "email": email,
        "expires_in": otp_ttl_seconds(),
        "mailed": mailed,
        "detail": (
            f"Code sent to {email}."
            if mailed
            else "Code generated (dev mode — check Mac logs)."
        ),
    }
    if echo:
        out["dev_code"] = code
    if mail_error and echo:
        out["smtp_warning"] = mail_error
    return out


def verify_otp(email: str, code: str) -> dict[str, Any]:
    email = normalize_email(email)
    code = (code or "").strip().replace(" ", "")
    if not is_valid_email(email):
        return {"ok": False, "error": "invalid_email", "detail": "Invalid email."}
    if not re.fullmatch(r"\d{6}", code):
        return {"ok": False, "error": "invalid_code", "detail": "Enter the 6-digit code."}

    row = session_store.get_otp(email)
    if not row:
        return {"ok": False, "error": "no_otp", "detail": "Request a code first."}
    if float(row["expires_at"]) < time.time():
        session_store.clear_otp(email)
        return {"ok": False, "error": "expired", "detail": "Code expired — request a new one."}
    if int(row.get("attempts") or 0) >= max_otp_attempts():
        session_store.clear_otp(email)
        return {"ok": False, "error": "locked", "detail": "Too many attempts — request a new code."}

    expected = row["code_hash"]
    got = _hash_code(email, code)
    if not hmac.compare_digest(expected, got):
        attempts = session_store.bump_otp_attempts(email)
        left = max(0, max_otp_attempts() - attempts)
        return {
            "ok": False,
            "error": "mismatch",
            "detail": f"Incorrect code ({left} attempts left).",
        }

    session_store.clear_otp(email)
    token = session_store.create_session(email, ttl_days=session_ttl_days())
    conv_id = session_store.ensure_conversation(email)
    return {
        "ok": True,
        "token": token,
        "email": email,
        "conversation_id": conv_id,
        "expires_in_days": session_ttl_days(),
        "detail": "Session unlocked.",
    }


def resolve_bearer_token(authorization: str | None, *, alt_header: str | None = None) -> str | None:
    if alt_header and alt_header.strip():
        return alt_header.strip()
    auth = (authorization or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def session_from_request_headers(
    authorization: str | None = None,
    x_session_token: str | None = None,
) -> dict[str, Any] | None:
    token = resolve_bearer_token(authorization, alt_header=x_session_token)
    if not token:
        return None
    # Avoid treating brain HMAC shared secret as a user session
    brain_secret = (
        os.getenv("BRAIN_REGISTRY_SECRET")
        or os.getenv("LOCAL_INSTANCE_API_KEY")
        or ""
    ).strip()
    if brain_secret and hmac.compare_digest(token, brain_secret):
        return None
    return session_store.get_session(token)


def status_public() -> dict[str, Any]:
    return {
        "auth_enabled": auth_enabled(),
        "auth_required": auth_required(),
        "smtp_configured": smtp_configured(),
        "dev_echo": dev_echo_otp(),
        "otp_ttl_seconds": otp_ttl_seconds(),
        "session_days": session_ttl_days(),
    }
