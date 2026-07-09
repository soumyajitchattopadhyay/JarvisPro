"""
Local in-RAM memory for TESrACT (optimized for MacBook unified RAM).

Uses ChromaDB EphemeralClient — fully in-process, no disk persistence.
Web search results are embedded and recalled across turns; image/video hooks
are ready for future local media backends (see LOCAL_*_GEN_BACKEND in .env).

Environment variables:
  MEMORY_MAX_ENTRIES       — max documents per collection (default: 500)
  MEMORY_TTL_SECONDS       — evict entries older than this (default: 604800 = 7 days)
  MEMORY_MAX_DOC_CHARS     — truncate stored document text (default: 8000)
  MEMORY_MAX_MB            — soft budget across all collections (default: 256)
  MEMORY_CLEANUP_INTERVAL  — minimum seconds between TTL sweeps (default: 60)
  MEMORY_ENABLE_IMAGE_HOOK — 1|true to activate image memory collection
  MEMORY_ENABLE_VIDEO_HOOK — 1|true to activate video memory collection
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

MemoryKind = Literal["search", "image", "video"]

_CLIENT: Any = None
_COLLECTIONS: dict[MemoryKind, Any] = {}

# Rough accounting — updated on store/delete for fast stats without full scans.
_estimated_bytes: dict[MemoryKind, int] = {"search": 0, "image": 0, "video": 0}
_store_count: dict[MemoryKind, int] = {"search": 0, "image": 0, "video": 0}

_last_cleanup_at: float = 0.0
_last_cleanup_removed: int = 0

MEMORY_MAX_ENTRIES = max(50, int(os.getenv("MEMORY_MAX_ENTRIES", "500")))
MEMORY_TTL_SECONDS = max(3600, int(os.getenv("MEMORY_TTL_SECONDS", str(7 * 86400))))
MEMORY_MAX_DOC_CHARS = max(500, int(os.getenv("MEMORY_MAX_DOC_CHARS", "8000")))
MEMORY_MAX_MB = max(32.0, float(os.getenv("MEMORY_MAX_MB", "256")))
MEMORY_CLEANUP_INTERVAL = max(10, int(os.getenv("MEMORY_CLEANUP_INTERVAL", "60")))
EMBEDDING_DIM = 384  # Chroma default embedding model output size
EMBEDDING_BYTES = EMBEDDING_DIM * 4

_COLLECTION_META: dict[MemoryKind, dict[str, Any]] = {
    "search": {"name": "tesract_web_searches", "space": "cosine"},
    "image": {"name": "tesract_image_memory", "space": "cosine"},
    "video": {"name": "tesract_video_memory", "space": "cosine"},
}


@dataclass(frozen=True)
class MemoryLimits:
    max_entries: int
    ttl_seconds: int
    max_doc_chars: int
    max_mb: float
    cleanup_interval: int


def _limits() -> MemoryLimits:
    return MemoryLimits(
        max_entries=MEMORY_MAX_ENTRIES,
        ttl_seconds=MEMORY_TTL_SECONDS,
        max_doc_chars=MEMORY_MAX_DOC_CHARS,
        max_mb=MEMORY_MAX_MB,
        cleanup_interval=MEMORY_CLEANUP_INTERVAL,
    )


def _log(msg: str) -> None:
    print(f"[TESrACT:memory] {msg}")


def _image_hook_enabled() -> bool:
    if os.getenv("MEMORY_ENABLE_IMAGE_HOOK", "").strip().lower() in ("1", "true", "yes"):
        return True
    return bool(os.getenv("LOCAL_IMAGE_GEN_BACKEND", "").strip())


def _video_hook_enabled() -> bool:
    if os.getenv("MEMORY_ENABLE_VIDEO_HOOK", "").strip().lower() in ("1", "true", "yes"):
        return True
    return bool(os.getenv("LOCAL_VIDEO_GEN_BACKEND", "").strip())


def _get_client():
    """Single EphemeralClient — all collections live in unified Mac RAM, no disk."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    import chromadb
    from chromadb.config import Settings

    _CLIENT = chromadb.EphemeralClient(
        Settings(
            anonymized_telemetry=False,
            is_persistent=False,
            allow_reset=True,
        )
    )
    return _CLIENT


def _get_collection(kind: MemoryKind):
    """Lazy-init a named in-memory collection."""
    if kind in _COLLECTIONS:
        return _COLLECTIONS[kind]

    if kind == "image" and not _image_hook_enabled():
        raise RuntimeError("Image memory hook is disabled")
    if kind == "video" and not _video_hook_enabled():
        raise RuntimeError("Video memory hook is disabled")

    meta = _COLLECTION_META[kind]
    client = _get_client()
    coll = client.get_or_create_collection(
        name=meta["name"],
        metadata={"hnsw:space": meta["space"], "kind": kind},
    )
    _COLLECTIONS[kind] = coll
    return coll


def _truncate(text: str, *, max_chars: int | None = None) -> str:
    limit = max_chars or MEMORY_MAX_DOC_CHARS
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _doc_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _estimate_entry_bytes(document: str, metadata: dict[str, Any]) -> int:
    doc_b = len(document.encode("utf-8"))
    meta_b = len(str(metadata).encode("utf-8"))
    return doc_b + meta_b + EMBEDDING_BYTES + 64  # small overhead per row


def _track_store(kind: MemoryKind, document: str, metadata: dict[str, Any]) -> None:
    _estimated_bytes[kind] += _estimate_entry_bytes(document, metadata)
    _store_count[kind] += 1


def _track_delete(kind: MemoryKind, removed: int, *, bytes_removed: int = 0) -> None:
    _store_count[kind] = max(0, _store_count[kind] - removed)
    if bytes_removed > 0:
        _estimated_bytes[kind] = max(0, _estimated_bytes[kind] - bytes_removed)
    elif removed > 0:
        # Fallback: re-sync estimate from collection on bulk delete.
        _resync_bytes_estimate(kind)


def _resync_bytes_estimate(kind: MemoryKind) -> None:
    if kind not in _COLLECTIONS:
        _estimated_bytes[kind] = 0
        return
    try:
        coll = _COLLECTIONS[kind]
        if coll.count() == 0:
            _estimated_bytes[kind] = 0
            return
        data = coll.get(include=["documents", "metadatas"])
        total = 0
        for doc, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
            total += _estimate_entry_bytes(str(doc or ""), meta or {})
        _estimated_bytes[kind] = total
    except Exception:
        pass


def _collection_stats(kind: MemoryKind, *, init_if_warm: bool = True) -> dict[str, Any]:
    enabled = True
    if kind == "image":
        enabled = _image_hook_enabled()
    elif kind == "video":
        enabled = _video_hook_enabled()

    out: dict[str, Any] = {
        "kind": kind,
        "collection": _COLLECTION_META[kind]["name"],
        "enabled": enabled,
        "initialized": kind in _COLLECTIONS,
        "entries": 0,
        "estimated_mb": round(_estimated_bytes[kind] / (1024 ** 2), 3),
    }

    if not enabled:
        out["status"] = "hook_disabled"
        return out

    if kind not in _COLLECTIONS:
        out["status"] = "ready"
        return out

    if not init_if_warm:
        out["status"] = "warm"
        out["entries"] = "unknown"
        return out

    try:
        coll = _COLLECTIONS[kind]
        out["entries"] = coll.count()
        out["status"] = "active"
    except Exception as exc:
        out["status"] = "error"
        out["error"] = str(exc)
    return out


def _evict_ids(coll: Any, ids: list[str], kind: MemoryKind) -> int:
    if not ids:
        return 0
    try:
        # Best-effort byte accounting before delete.
        data = coll.get(ids=ids, include=["documents", "metadatas"])
        bytes_removed = 0
        for doc, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
            bytes_removed += _estimate_entry_bytes(str(doc or ""), meta or {})
        coll.delete(ids=ids)
        _track_delete(kind, len(ids), bytes_removed=bytes_removed)
        return len(ids)
    except Exception as exc:
        _log(f"Eviction failed ({kind}): {exc}")
        return 0


def _cleanup_collection(kind: MemoryKind, *, force: bool = False) -> int:
    """TTL + capacity eviction for one collection. Returns number of docs removed."""
    global _last_cleanup_at, _last_cleanup_removed

    if kind not in _COLLECTIONS:
        return 0

    now = time.time()
    if not force and now - _last_cleanup_at < MEMORY_CLEANUP_INTERVAL:
        return 0

    coll = _COLLECTIONS[kind]
    try:
        count = coll.count()
    except Exception as exc:
        _log(f"Cleanup count failed ({kind}): {exc}")
        return 0

    if count == 0:
        return 0

    removed = 0
    limits = _limits()

    try:
        data = coll.get(include=["metadatas"])
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        cutoff = now - limits.ttl_seconds

        expired_ids = [
            doc_id
            for doc_id, meta in zip(ids, metas)
            if float((meta or {}).get("stored_at") or 0) < cutoff
        ]
        removed += _evict_ids(coll, expired_ids, kind)

        count = coll.count()
        if count > limits.max_entries:
            data = coll.get(include=["metadatas"])
            ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            ranked = sorted(
                zip(ids, metas),
                key=lambda row: float((row[1] or {}).get("stored_at") or 0),
            )
            overflow = count - limits.max_entries
            oldest_ids = [doc_id for doc_id, _ in ranked[:overflow]]
            removed += _evict_ids(coll, oldest_ids, kind)
    except Exception as exc:
        _log(f"Cleanup sweep failed ({kind}): {exc}")

    if removed:
        _log(f"Cleanup ({kind}): evicted {removed} entries")
    return removed


def _maybe_cleanup(kind: MemoryKind) -> None:
    """Run periodic cleanup across warm collections."""
    global _last_cleanup_at, _last_cleanup_removed

    now = time.time()
    should_run = now - _last_cleanup_at >= MEMORY_CLEANUP_INTERVAL

    coll = _COLLECTIONS.get(kind)
    if coll is not None:
        try:
            if coll.count() > MEMORY_MAX_ENTRIES:
                should_run = True
        except Exception:
            pass

    total_est_mb = sum(_estimated_bytes.values()) / (1024 ** 2)
    if total_est_mb > MEMORY_MAX_MB * 0.9:
        should_run = True

    if not should_run:
        return

    removed = 0
    for k in list(_COLLECTIONS.keys()):
        removed += _cleanup_collection(k, force=True)

    _last_cleanup_at = now
    _last_cleanup_removed = removed


def run_memory_cleanup(*, force: bool = False) -> dict[str, Any]:
    """Manual or scheduled cleanup across all warm collections."""
    global _last_cleanup_at, _last_cleanup_removed

    removed = 0
    if force:
        for kind in list(_COLLECTIONS.keys()):
            removed += _cleanup_collection(kind, force=True)
        _last_cleanup_at = time.time()
        _last_cleanup_removed = removed
    else:
        _maybe_cleanup("search")
        removed = _last_cleanup_removed

    return {
        "removed": removed,
        "last_cleanup_at": _last_cleanup_at,
        "collections": {
            kind: _collection_stats(kind) for kind in ("search", "image", "video")
        },
    }


def _upsert(
    kind: MemoryKind,
    *,
    doc_id: str,
    document: str,
    metadata: dict[str, Any],
) -> None:
    document = _truncate(document)
    if not document:
        return

    metadata = {**metadata, "stored_at": time.time(), "kind": kind}
    coll = _get_collection(kind)
    coll.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])
    _track_store(kind, document, metadata)
    _maybe_cleanup(kind)


# ---------------------------------------------------------------------------
# Web search memory (active)
# ---------------------------------------------------------------------------

def store_search(query: str, results: str, *, source: str = "duckduckgo") -> None:
    """Persist a search query and its result text in local vector memory."""
    query = (query or "").strip()
    results = (results or "").strip()
    if not query or not results:
        return

    try:
        _upsert(
            "search",
            doc_id=_doc_id("search", query, results),
            document=results,
            metadata={
                "query": query[:500],
                "source": source,
                "media_type": "text",
            },
        )
    except Exception as exc:
        _log(f"Failed to store search: {exc}")


def recall_similar(query: str, n: int = 3) -> list[dict[str, Any]]:
    """Return prior searches semantically similar to `query`."""
    query = (query or "").strip()
    if not query:
        return []

    try:
        coll = _get_collection("search")
        if coll.count() == 0:
            return []
        hits = coll.query(query_texts=[query], n_results=min(n, coll.count()))
        out: list[dict[str, Any]] = []
        docs = (hits.get("documents") or [[]])[0]
        metas = (hits.get("metadatas") or [[]])[0]
        dists = (hits.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            out.append({
                "query": (meta or {}).get("query", ""),
                "source": (meta or {}).get("source", "unknown"),
                "results": doc,
                "distance": dist,
                "media_type": (meta or {}).get("media_type", "text"),
            })
        return out
    except Exception as exc:
        _log(f"Recall failed: {exc}")
        return []


def format_recall_for_context(query: str, n: int = 2) -> str:
    """Format prior hits as context prefix for web_search tool output."""
    hits = recall_similar(query, n=n)
    if not hits:
        return ""

    # Cosine distance in Chroma: 0 = identical, higher = less similar.
    # Drop weak matches so unrelated prior searches (e.g. OpenAI vs Python) do not pollute results.
    max_distance = float(os.getenv("MEMORY_SEARCH_MAX_DISTANCE", "0.55") or "0.55")
    filtered = []
    for hit in hits:
        dist = hit.get("distance")
        if dist is not None:
            try:
                if float(dist) > max_distance:
                    continue
            except (TypeError, ValueError):
                pass
        # Also require a little lexical overlap as a safety net
        q_prev = str(hit.get("query") or "").lower()
        q_now = (query or "").lower()
        tokens_now = {t for t in re.findall(r"[a-z0-9]{3,}", q_now)}
        tokens_prev = {t for t in re.findall(r"[a-z0-9]{3,}", q_prev)}
        if tokens_now and tokens_prev and not (tokens_now & tokens_prev):
            continue
        filtered.append(hit)

    if not filtered:
        return ""

    lines = ["[Prior local search memory — may be stale]"]
    for i, hit in enumerate(filtered, 1):
        q = hit.get("query") or "unknown"
        src = hit.get("source") or "unknown"
        body = str(hit.get("results") or "")[:600]
        lines.append(f"{i}. Earlier query ({src}): {q!r}")
        lines.append(f"   {body}")
    lines.append("")
    return "\n".join(lines)


def store_note(topic: str, content: str, *, source: str = "agent") -> None:
    """Persist a fact, preference, or context note for later semantic recall."""
    topic = (topic or "").strip()
    content = (content or "").strip()
    if not topic or not content:
        return

    try:
        _upsert(
            "search",
            doc_id=_doc_id("note", topic, content, source),
            document=f"Topic: {topic}\n{content}",
            metadata={
                "query": topic[:500],
                "source": source,
                "media_type": "note",
            },
        )
    except Exception as exc:
        _log(f"Failed to store note: {exc}")


def store_turn_summary(
    user_message: str,
    assistant_reply: str,
    *,
    tools_used: list[str] | None = None,
) -> None:
    """Persist a compact summary of a completed turn for multi-turn recall."""
    user_message = (user_message or "").strip()
    assistant_reply = (assistant_reply or "").strip()
    if not user_message:
        return

    tools_used = tools_used or []
    summary = (
        f"User: {user_message[:400]}\n"
        f"Assistant: {assistant_reply[:600]}\n"
        f"Tools: {', '.join(tools_used) if tools_used else 'none'}"
    )
    try:
        _upsert(
            "search",
            doc_id=_doc_id("turn", user_message, assistant_reply, str(tools_used)),
            document=summary,
            metadata={
                "query": user_message[:500],
                "source": "conversation",
                "media_type": "turn",
                "tools_used": ",".join(tools_used)[:200],
            },
        )
    except Exception as exc:
        _log(f"Failed to store turn summary: {exc}")


def recall_memory(query: str, n: int = 4) -> list[dict[str, Any]]:
    """Unified semantic recall across searches, notes, and past turns."""
    query = (query or "").strip()
    if not query:
        return []

    try:
        coll = _get_collection("search")
        if coll.count() == 0:
            return []
        hits = coll.query(query_texts=[query], n_results=min(n, coll.count()))
        out: list[dict[str, Any]] = []
        docs = (hits.get("documents") or [[]])[0]
        metas = (hits.get("metadatas") or [[]])[0]
        dists = (hits.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            meta = meta or {}
            out.append({
                "topic": meta.get("query", ""),
                "content": doc,
                "source": meta.get("source", "unknown"),
                "media_type": meta.get("media_type", "search"),
                "tools_used": meta.get("tools_used", ""),
                "distance": dist,
            })
        return out
    except Exception as exc:
        _log(f"Unified recall failed: {exc}")
        return []


def format_memory_for_system(user_query: str, n: int = 3) -> str:
    """Build a brief memory block for injection into the agent system prompt."""
    hits = recall_memory(user_query, n=n)
    if not hits:
        return ""

    lines = ["Relevant prior context from local memory (may be stale — verify if critical):"]
    for i, hit in enumerate(hits, 1):
        media = hit.get("media_type") or "search"
        topic = hit.get("topic") or "unknown"
        body = str(hit.get("content") or "")[:450]
        lines.append(f"{i}. [{media}] {topic}")
        lines.append(f"   {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image / video memory hooks (future local media backends)
# ---------------------------------------------------------------------------

def store_image_memory(
    prompt: str,
    description: str,
    *,
    backend: str = "",
    image_ref: str = "",
    source: str = "local_image_gen",
) -> None:
    """
    Hook for future local image generation — stores prompt + result description.
    No-op when image memory hook is disabled.
    """
    if not _image_hook_enabled():
        return

    prompt = (prompt or "").strip()
    description = (description or "").strip()
    if not prompt or not description:
        return

    try:
        _upsert(
            "image",
            doc_id=_doc_id("image", prompt, description, image_ref),
            document=description,
            metadata={
                "prompt": prompt[:500],
                "image_ref": image_ref[:300],
                "backend": backend or os.getenv("LOCAL_IMAGE_GEN_BACKEND", ""),
                "source": source,
                "media_type": "image",
            },
        )
    except Exception as exc:
        _log(f"Failed to store image memory: {exc}")


def recall_image_similar(prompt: str, n: int = 3) -> list[dict[str, Any]]:
    """Recall prior image generations similar to `prompt`."""
    if not _image_hook_enabled():
        return []
    return _recall_kind("image", prompt, n=n)


def store_video_memory(
    prompt: str,
    transcript: str,
    *,
    backend: str = "",
    duration_sec: float = 0,
    source: str = "local_video_gen",
) -> None:
    """
    Hook for future local video generation — stores prompt + transcript text.
    No-op when video memory hook is disabled.
    """
    if not _video_hook_enabled():
        return

    prompt = (prompt or "").strip()
    transcript = (transcript or "").strip()
    if not prompt or not transcript:
        return

    try:
        _upsert(
            "video",
            doc_id=_doc_id("video", prompt, transcript),
            document=transcript,
            metadata={
                "prompt": prompt[:500],
                "duration_sec": duration_sec,
                "backend": backend or os.getenv("LOCAL_VIDEO_GEN_BACKEND", ""),
                "source": source,
                "media_type": "video",
            },
        )
    except Exception as exc:
        _log(f"Failed to store video memory: {exc}")


def recall_video_similar(prompt: str, n: int = 3) -> list[dict[str, Any]]:
    """Recall prior video transcripts similar to `prompt`."""
    if not _video_hook_enabled():
        return []
    return _recall_kind("video", prompt, n=n)


def _recall_kind(kind: MemoryKind, query: str, *, n: int = 3) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query or kind not in _COLLECTIONS:
        return []

    try:
        coll = _COLLECTIONS[kind]
        if coll.count() == 0:
            return []
        hits = coll.query(query_texts=[query], n_results=min(n, coll.count()))
        out: list[dict[str, Any]] = []
        docs = (hits.get("documents") or [[]])[0]
        metas = (hits.get("metadatas") or [[]])[0]
        dists = (hits.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            out.append({
                "prompt": (meta or {}).get("prompt", ""),
                "content": doc,
                "distance": dist,
                "media_type": (meta or {}).get("media_type", kind),
                "backend": (meta or {}).get("backend", ""),
            })
        return out
    except Exception as exc:
        _log(f"Recall failed ({kind}): {exc}")
        return []


# ---------------------------------------------------------------------------
# Health / stats
# ---------------------------------------------------------------------------

def _host_ram_snapshot() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        mem = proc.memory_info()
        return {
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "used_percent": vm.percent,
            "process_rss_mb": round(mem.rss / (1024 ** 2), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def memory_available() -> bool:
    """True if local memory backend is usable (does not force Chroma init)."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    if "search" not in _COLLECTIONS:
        return True
    try:
        _COLLECTIONS["search"].count()
        return True
    except Exception:
        return False


def search_memory_status() -> dict[str, Any]:
    """Lightweight status for /health — avoids Chroma init unless already warm."""
    available = memory_available()
    stats = _collection_stats("search", init_if_warm=True)
    return {
        "available": available,
        "backend": "chromadb-ephemeral",
        "persistent": False,
        "entries": stats.get("entries", 0),
        "estimated_mb": stats.get("estimated_mb", 0),
        "limits": {
            "max_entries": MEMORY_MAX_ENTRIES,
            "max_mb": MEMORY_MAX_MB,
        },
    }


def memory_stats() -> dict[str, Any]:
    """Backward-compatible stats dict."""
    detailed = get_memory_stats(detailed=False)
    return {
        "entries": detailed["collections"]["search"].get("entries", 0),
        "backend": detailed["backend"],
        "estimated_mb": detailed["estimated_memory_mb"],
        "persistent": False,
    }


def get_memory_stats(*, detailed: bool = True) -> dict[str, Any]:
    """Full memory subsystem stats for /memory/stats monitoring."""
    limits = _limits()
    collections = {
        kind: _collection_stats(kind, init_if_warm=detailed or kind in _COLLECTIONS)
        for kind in ("search", "image", "video")
    }

    total_entries = 0
    for info in collections.values():
        entries = info.get("entries")
        if isinstance(entries, int):
            total_entries += entries

    if detailed:
        for kind in _COLLECTIONS:
            _resync_bytes_estimate(kind)
    estimated_bytes = sum(_estimated_bytes.values())

    return {
        "timestamp": time.time(),
        "backend": "chromadb-ephemeral",
        "persistent": False,
        "client_initialized": _CLIENT is not None,
        "limits": {
            "max_entries_per_collection": limits.max_entries,
            "ttl_seconds": limits.ttl_seconds,
            "max_doc_chars": limits.max_doc_chars,
            "max_mb": limits.max_mb,
            "cleanup_interval_seconds": limits.cleanup_interval,
        },
        "hooks": {
            "image_enabled": _image_hook_enabled(),
            "video_enabled": _video_hook_enabled(),
        },
        "collections": collections,
        "total_entries": total_entries,
        "estimated_memory_mb": round(estimated_bytes / (1024 ** 2), 3),
        "store_operations": dict(_store_count),
        "last_cleanup_at": _last_cleanup_at or None,
        "last_cleanup_removed": _last_cleanup_removed,
        "host_ram": _host_ram_snapshot(),
    }


def reset_memory(kind: MemoryKind | None = None) -> dict[str, Any]:
    """Drop one or all in-memory collections (destructive; for debugging)."""
    global _CLIENT, _last_cleanup_at, _last_cleanup_removed

    kinds: tuple[MemoryKind, ...]
    if kind is None:
        kinds = ("search", "image", "video")
    else:
        kinds = (kind,)

    for k in kinds:
        _COLLECTIONS.pop(k, None)
        _estimated_bytes[k] = 0
        _store_count[k] = 0

    if _CLIENT is not None and kind is None:
        try:
            _CLIENT.reset()
        except Exception as exc:
            _log(f"Client reset failed: {exc}")
        _CLIENT = None

    _last_cleanup_at = 0.0
    _last_cleanup_removed = 0
    _log(f"Reset memory: {kind or 'all'}")
    return get_memory_stats(detailed=False)