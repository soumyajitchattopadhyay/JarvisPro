#!/usr/bin/env python3
"""
TESrACT SMTP connection tester (Gmail App Password / any AUTH_SMTP_* provider).

Loads project .env, logs into the SMTP server with smtplib, and sends a short
test message to the operator. Prints detailed TLS/SSL diagnostics on failure.

Usage (from repo root, with venv active if you use one):
    python test_smtp.py
    python test_smtp.py you@gmail.com          # override recipient
    python test_smtp.py --self                 # send only to AUTH_SMTP_USER
"""
from __future__ import annotations

import argparse
import os
import re
import smtplib
import ssl
import sys
import traceback
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
_ENV = _ROOT / ".env"

# Absolute-top load (project .env, then CWD)
load_dotenv(_ENV)
load_dotenv()


def _mask(value: str, keep: int = 2) -> str:
    v = value or ""
    if len(v) <= keep * 2:
        return "*" * len(v) if v else "(empty)"
    return f"{v[:keep]}…{v[-keep:]} ({len(v)} chars)"


def _credentials() -> tuple[str, str, str, int, bool, str]:
    host = (os.getenv("AUTH_SMTP_HOST") or "smtp.gmail.com").strip()
    user = (os.getenv("AUTH_SMTP_USER") or os.getenv("AUTH_SMTP_FROM") or "").strip()
    password = re.sub(r"\s+", "", (os.getenv("AUTH_SMTP_PASSWORD") or "").strip())
    port = int(os.getenv("AUTH_SMTP_PORT", "587") or "587")
    use_ssl_flag = (os.getenv("AUTH_SMTP_SSL") or "").strip().lower() in ("1", "true", "yes")
    use_ssl = use_ssl_flag or port == 465
    from_addr = (os.getenv("AUTH_SMTP_FROM") or user).strip()
    return host, user, password, port, use_ssl, from_addr


def _print_config(host: str, user: str, password: str, port: int, use_ssl: bool, from_addr: str) -> None:
    mode = "SMTP_SSL (implicit TLS)" if use_ssl else "SMTP + STARTTLS"
    lines = [
        "=== TESrACT SMTP test ===",
        f".env path     : {_ENV}  ({'found' if _ENV.is_file() else 'MISSING'})",
        f"AUTH_SMTP_HOST: {host}",
        f"AUTH_SMTP_PORT: {port}",
        f"AUTH_SMTP_SSL : {use_ssl}  → {mode}",
        f"AUTH_SMTP_USER: {user or '(empty)'}",
        f"AUTH_SMTP_FROM: {from_addr or '(empty)'}",
        f"PASSWORD      : {_mask(password)}",
        "",
    ]
    print("\n".join(lines), flush=True)


def _send(
    *,
    host: str,
    user: str,
    password: str,
    port: int,
    use_ssl: bool,
    from_addr: str,
    to_addr: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = "TESrACT SMTP test — connection OK"
    msg["From"] = from_addr or user
    msg["To"] = to_addr
    msg.set_content(
        "This is a TESrACT SMTP connectivity test.\n\n"
        f"Host={host}:{port}  mode={'SSL' if use_ssl else 'STARTTLS'}\n"
        "If you received this, Email OTP can deliver codes.\n"
    )

    context = ssl.create_default_context()
    print(f"Connecting to {host}:{port} …")
    if use_ssl:
        print("Using smtplib.SMTP_SSL (port 465 style)…")
        with smtplib.SMTP_SSL(host, port, context=context, timeout=45) as smtp:
            smtp.set_debuglevel(1)
            print("Logging in…")
            smtp.login(user, password)
            print("Sending message…")
            smtp.send_message(msg)
    else:
        print("Using smtplib.SMTP + starttls (port 587 style)…")
        with smtplib.SMTP(host, port, timeout=45) as smtp:
            smtp.set_debuglevel(1)
            smtp.ehlo()
            print("Starting TLS…")
            smtp.starttls(context=context)
            smtp.ehlo()
            print("Logging in…")
            smtp.login(user, password)
            print("Sending message…")
            smtp.send_message(msg)
    print()
    print(f"SUCCESS — test email sent to {to_addr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test TESrACT AUTH_SMTP_* settings")
    parser.add_argument(
        "recipient",
        nargs="?",
        default="",
        help="Destination email (default: AUTH_SMTP_USER / AUTH_ALLOWED_EMAILS first entry)",
    )
    parser.add_argument(
        "--self",
        action="store_true",
        help="Force recipient = AUTH_SMTP_USER",
    )
    args = parser.parse_args(argv)

    host, user, password, port, use_ssl, from_addr = _credentials()
    _print_config(host, user, password, port, use_ssl, from_addr)

    if not _ENV.is_file():
        print(
            "ERROR: project .env not found.\n"
            "Create it from .env.example and set AUTH_SMTP_USER + AUTH_SMTP_PASSWORD.\n"
            "See Gmail App Password steps in .env.example.",
            file=sys.stderr,
        )
        return 2

    if not user or not password:
        print(
            "ERROR: SMTP not configured.\n"
            "Set AUTH_SMTP_USER and AUTH_SMTP_PASSWORD in .env\n"
            "(Gmail: use a 16-character App Password, not your normal password).\n\n"
            "Example (STARTTLS / 587):\n"
            "  AUTH_SMTP_HOST=smtp.gmail.com\n"
            "  AUTH_SMTP_PORT=587\n"
            "  AUTH_SMTP_SSL=false\n"
            "  AUTH_SMTP_USER=you@gmail.com\n"
            "  AUTH_SMTP_PASSWORD=xxxxxxxxxxxxxxxx\n"
            "  AUTH_SMTP_FROM=you@gmail.com\n",
            file=sys.stderr,
        )
        return 2

    allowed = (os.getenv("AUTH_ALLOWED_EMAILS") or "").strip()
    first_allowed = ""
    if allowed:
        first_allowed = next((e.strip() for e in allowed.split(",") if e.strip()), "")

    if args.self:
        to_addr = user
    elif args.recipient:
        to_addr = args.recipient.strip()
    else:
        to_addr = first_allowed or user

    if not to_addr or "@" not in to_addr:
        print("ERROR: no valid recipient. Pass an email or set AUTH_SMTP_USER.", file=sys.stderr)
        return 2

    print(f"Recipient    : {to_addr}")
    print()

    try:
        _send(
            host=host,
            user=user,
            password=password,
            port=port,
            use_ssl=use_ssl,
            from_addr=from_addr or user,
            to_addr=to_addr,
        )
        return 0
    except smtplib.SMTPAuthenticationError as exc:
        print("\n=== SMTP AUTHENTICATION FAILED ===", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nGmail tips:\n"
            "  • Use an App Password (Google Account → Security → 2-Step Verification → App passwords)\n"
            "  • Do NOT use your normal Gmail password\n"
            "  • AUTH_SMTP_USER must be the full Gmail address that owns the app password\n"
            "  • If 2FA is off, App Passwords are unavailable — enable 2-Step Verification first\n",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1
    except ssl.SSLError as exc:
        print("\n=== TLS / SSL ERROR ===", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nTry the other port mode:\n"
            "  STARTTLS: AUTH_SMTP_PORT=587  AUTH_SMTP_SSL=false\n"
            "  Implicit SSL: AUTH_SMTP_PORT=465  AUTH_SMTP_SSL=true\n"
            "Also check system time / corporate TLS interception.\n",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1
    except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, TimeoutError, OSError) as exc:
        print("\n=== CONNECTION ERROR ===", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            f"\nCould not reach {host}:{port}. Check network / firewall / VPN.\n"
            "Gmail SMTP requires outbound TCP 587 (STARTTLS) or 465 (SSL).\n",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1
    except Exception as exc:
        print("\n=== SMTP FAILED ===", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
