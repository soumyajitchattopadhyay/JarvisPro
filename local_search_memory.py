"""
Local in-RAM search memory for TESrACT.

Stores web search queries + results in ChromaDB (in-process, no disk) so the
agent can recall prior searches without hitting the network again.

Future extensions (see llm_router.py LOCAL_MEDIA_* hooks):
  - image search result thumbnails
  - video transcript embeddings
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

_collection: Any = None


def _get_collection():
    """Lazy-init Chroma collection — keeps vectors in unified Mac RAM."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(
        Settings(
            anonymized_telemetry=False,
            is_persistent=False,
            allow_reset=True,
        )
    )
    _collection = client.get_or_create_collection(
        name="tesract_web_searches",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _doc_id(query: str, results: str) -> str:
    digest = hashlib.sha256(f"{query}\n{results}".encode()).hexdigest()[:24]
    return f"search_{digest}"


def store_search(query: str, results: str, *, source: str = "duckduckgo") -> None:
    """Persist a search query and its result text in local vector memory."""
    query = (query or "").strip()
    results = (results or "").strip()
    if not query or not results:
        return

    try:
        coll = _get_collection()
        coll.upsert(
            ids=[_doc_id(query, results)],
            documents=[results],
            metadatas=[{
                "query": query[:500],
                "source": source,
                "stored_at": time.time(),
            }],
        )
    except Exception as exc:
        print(f"[TESrACT:memory] Failed to store search: {exc}")


def recall_similar(query: str, n: int = 3) -> list[dict[str, Any]]:
    """Return prior searches semantically similar to `query`."""
    query = (query or "").strip()
    if not query:
        return []

    try:
        coll = _get_collection()
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
            })
        return out
    except Exception as exc:
        print(f"[TESrACT:memory] Recall failed: {exc}")
        return []


def format_recall_for_context(query: str, n: int = 2) -> str:
    """Format prior hits as context prefix for web_search tool output."""
    hits = recall_similar(query, n=n)
    if not hits:
        return ""

    lines = ["[Prior local search memory — may be stale]"]
    for i, hit in enumerate(hits, 1):
        q = hit.get("query") or "unknown"
        src = hit.get("source") or "unknown"
        body = str(hit.get("results") or "")[:600]
        lines.append(f"{i}. Earlier query ({src}): {q!r}")
        lines.append(f"   {body}")
    lines.append("")
    return "\n".join(lines)


def memory_stats() -> dict[str, Any]:
    """Lightweight stats for /health and debugging."""
    try:
        coll = _get_collection()
        return {"entries": coll.count(), "backend": "chromadb-inmemory"}
    except Exception as exc:
        return {"entries": 0, "backend": "chromadb-inmemory", "error": str(exc)}