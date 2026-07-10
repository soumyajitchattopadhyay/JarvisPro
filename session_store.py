"""
Local free chat history + auth state on the Mac brain (SQLite).

No cloud database: agent_data/tesract_sessions.db lives on the Mac SSD.
Render is stateless — when hybrid is up, auth/history routes are proxied here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_BASE = Path(__file__).resolve().parent
_DATA_DIR = _BASE / "agent_data"
_DB_PATH = _DATA_DIR / "tesract_sessions.db"
_LOCK = threading.RLock()


def _db_path() -> Path:
    override = (os.getenv("TESRACT_SESSION_DB") or "").strip()
    return Path(override) if override else _DB_PATH


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with _LOCK, _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS otp_codes (
                email       TEXT PRIMARY KEY,
                code_hash   TEXT NOT NULL,
                expires_at  REAL NOT NULL,
                attempts    INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT PRIMARY KEY,
                email       TEXT NOT NULL,
                created_at  REAL NOT NULL,
                expires_at  REAL NOT NULL,
                last_seen   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_email ON sessions(email);

            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                email       TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT 'Session',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_email ON conversations(email);

            CREATE TABLE IF NOT EXISTS messages (
                id               TEXT PRIMARY KEY,
                conversation_id  TEXT NOT NULL,
                email            TEXT NOT NULL,
                role             TEXT NOT NULL,
                content          TEXT NOT NULL,
                created_at       REAL NOT NULL,
                meta_json        TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_msg_email ON messages(email, created_at);
            """
        )


# ---------------------------------------------------------------------------
# OTP
# ---------------------------------------------------------------------------

def save_otp(email: str, code_hash: str, *, ttl_seconds: int) -> None:
    email = email.strip().lower()
    now = time.time()
    with _LOCK, _conn() as con:
        con.execute(
            """
            INSERT INTO otp_codes (email, code_hash, expires_at, attempts, created_at)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(email) DO UPDATE SET
                code_hash=excluded.code_hash,
                expires_at=excluded.expires_at,
                attempts=0,
                created_at=excluded.created_at
            """,
            (email, code_hash, now + max(60, ttl_seconds), now),
        )


def get_otp(email: str) -> dict[str, Any] | None:
    email = email.strip().lower()
    with _LOCK, _conn() as con:
        row = con.execute(
            "SELECT email, code_hash, expires_at, attempts, created_at FROM otp_codes WHERE email=?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def bump_otp_attempts(email: str) -> int:
    email = email.strip().lower()
    with _LOCK, _conn() as con:
        con.execute(
            "UPDATE otp_codes SET attempts = attempts + 1 WHERE email=?",
            (email,),
        )
        row = con.execute(
            "SELECT attempts FROM otp_codes WHERE email=?", (email,)
        ).fetchone()
    return int(row["attempts"]) if row else 0


def clear_otp(email: str) -> None:
    email = email.strip().lower()
    with _LOCK, _conn() as con:
        con.execute("DELETE FROM otp_codes WHERE email=?", (email,))


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(email: str, *, ttl_days: int = 30) -> str:
    email = email.strip().lower()
    token = uuid.uuid4().hex + uuid.uuid4().hex
    now = time.time()
    expires = now + max(1, ttl_days) * 86400
    with _LOCK, _conn() as con:
        con.execute(
            """
            INSERT INTO sessions (token, email, created_at, expires_at, last_seen)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token, email, now, expires, now),
        )
    return token


def get_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    token = token.strip()
    now = time.time()
    with _LOCK, _conn() as con:
        row = con.execute(
            "SELECT token, email, created_at, expires_at, last_seen FROM sessions WHERE token=?",
            (token,),
        ).fetchone()
        if not row:
            return None
        if float(row["expires_at"]) < now:
            con.execute("DELETE FROM sessions WHERE token=?", (token,))
            return None
        con.execute(
            "UPDATE sessions SET last_seen=? WHERE token=?",
            (now, token),
        )
        return {
            "token": row["token"],
            "email": row["email"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "last_seen": now,
        }


def revoke_session(token: str | None) -> None:
    if not token:
        return
    with _LOCK, _conn() as con:
        con.execute("DELETE FROM sessions WHERE token=?", (token.strip(),))


def revoke_sessions_for_email(email: str) -> None:
    email = email.strip().lower()
    with _LOCK, _conn() as con:
        con.execute("DELETE FROM sessions WHERE email=?", (email,))


# ---------------------------------------------------------------------------
# Conversations / messages
# ---------------------------------------------------------------------------

def ensure_conversation(email: str, conversation_id: str | None = None, *, title: str = "Session") -> str:
    email = email.strip().lower()
    now = time.time()
    with _LOCK, _conn() as con:
        if conversation_id:
            row = con.execute(
                "SELECT id FROM conversations WHERE id=? AND email=?",
                (conversation_id, email),
            ).fetchone()
            if row:
                return str(row["id"])
        # Reuse latest conversation if still “active” today
        row = con.execute(
            """
            SELECT id FROM conversations
            WHERE email=?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (email,),
        ).fetchone()
        if row and not conversation_id:
            cid = str(row["id"])
            con.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, cid),
            )
            return cid
        cid = conversation_id or uuid.uuid4().hex
        con.execute(
            """
            INSERT INTO conversations (id, email, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cid, email, (title or "Session")[:120], now, now),
        )
        return cid


def append_message(
    *,
    email: str,
    conversation_id: str,
    role: str,
    content: str,
    meta: dict[str, Any] | None = None,
) -> str:
    email = email.strip().lower()
    role = (role or "assistant").strip().lower()
    content = content or ""
    mid = uuid.uuid4().hex
    now = time.time()
    with _LOCK, _conn() as con:
        con.execute(
            """
            INSERT INTO messages (id, conversation_id, email, role, content, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mid,
                conversation_id,
                email,
                role,
                content,
                now,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (now, conversation_id),
        )
    return mid


def list_messages(
    email: str,
    *,
    conversation_id: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    email = email.strip().lower()
    limit = max(1, min(1000, int(limit)))
    with _LOCK, _conn() as con:
        if conversation_id:
            rows = con.execute(
                """
                SELECT id, conversation_id, email, role, content, created_at, meta_json
                FROM messages
                WHERE email=? AND conversation_id=?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (email, conversation_id, limit),
            ).fetchall()
        else:
            # Latest conversation's messages
            conv = con.execute(
                """
                SELECT id FROM conversations
                WHERE email=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (email,),
            ).fetchone()
            if not conv:
                return []
            rows = con.execute(
                """
                SELECT id, conversation_id, email, role, content, created_at, meta_json
                FROM messages
                WHERE email=? AND conversation_id=?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (email, conv["id"], limit),
            ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["meta_json"] or "{}")
        except Exception:
            meta = {}
        out.append(
            {
                "id": r["id"],
                "conversation_id": r["conversation_id"],
                "email": r["email"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
                "meta": meta,
            }
        )
    return out


def list_conversations(email: str, *, limit: int = 50) -> list[dict[str, Any]]:
    email = email.strip().lower()
    limit = max(1, min(200, int(limit)))
    with _LOCK, _conn() as con:
        rows = con.execute(
            """
            SELECT id, email, title, created_at, updated_at
            FROM conversations
            WHERE email=?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (email, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def new_conversation(email: str, *, title: str = "New session") -> str:
    email = email.strip().lower()
    cid = uuid.uuid4().hex
    now = time.time()
    with _LOCK, _conn() as con:
        con.execute(
            """
            INSERT INTO conversations (id, email, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cid, email, title[:120], now, now),
        )
    return cid


# Init on import so first request never races schema creation
try:
    init_db()
except Exception as exc:  # pragma: no cover
    print(f"[TESrACT:session] DB init deferred: {exc}")
