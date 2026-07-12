from __future__ import annotations

import gc
import json
import os
import re
import time
import urllib.parse
import datetime
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# Package was renamed duckduckgo_search → ddgs; support both.
try:
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:  # pragma: no cover
        DDGS = None  # type: ignore

from local_search_memory import (
    format_recall_for_context,
    recall_memory,
    store_image_memory,
    store_note,
    store_search,
)
from mac_access import (
    classify_command,
    find_named_path,
    format_confirmation_request,
    generate_free_image as _mac_generate_free_image,
    intent_create_directory,
    intent_download_file,
    intent_list_directory,
    intent_open_url,
    intent_read_file,
    intent_run_command,
    last_remembered_path,
    real_home,
    remember_path,
    resolve_mac_path,
    rewrite_paths_in_command,
    sanitize_path_string,
    set_pending_confirmation,
)
import permissions

repl = PythonREPL()
WORKSPACE_ROOT = Path(__file__).resolve().parent
# Free Pollinations pipeline writes project-local generated/ (served at /generated)
GENERATED_DIR = WORKSPACE_ROOT / "generated"
AGENT_DATA_DIR = WORKSPACE_ROOT / "agent_data"
TASKS_FILE = AGENT_DATA_DIR / "active_task.json"


def _control_ok(control_allowed: bool = False) -> bool:
    """True when Mac control is ON (file flag) or an explicit override is passed."""
    return bool(control_allowed) or permissions.is_control_allowed()

def _ensure_dirs() -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    AGENT_DATA_DIR.mkdir(exist_ok=True)


def _tool_ok(name: str, result: str) -> str:
    return f"TOOL: {name}\nSTATUS: SUCCESS\nRESULT:\n{result}"


def _tool_err(name: str, message: str) -> str:
    return f"TOOL: {name}\nSTATUS: ERROR\nRESULT:\n{message}"


def _permission_denied(tool_name: str, detail: str = "") -> str:
    msg = (
        "PERMISSION_DENIED: Client-action authorization is OFF. "
        "Turn on the MAC CONTROL toggle or say 'Allow Control' "
        "to authorize client-side execution intents."
    )
    if detail:
        msg = f"{msg} ({detail})"
    return _tool_err(tool_name, msg)


def _client_intent_ok(name: str, intent_blob: str) -> str:
    """Wrap a client execution intent as a successful tool result."""
    return _tool_ok(name, intent_blob)


def _clean_path_arg(path: str) -> str:
    """Strip model junk like 'filename: requirements.txt' or path=... wrappers."""
    raw = (path or "").strip().strip("'\"")
    raw = re.sub(
        r"^(?:filename|file\s*name|path|file)\s*[:=]\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    # Kill hallucinated placeholder homes early
    raw = sanitize_path_string(raw)
    raw = raw.strip().strip("'\"")
    return raw


def _safe_resolve(path: Path) -> Path:
    """resolve() that allows missing parents (needed for create/write)."""
    try:
        return path.expanduser().resolve(strict=False)
    except TypeError:
        return path.expanduser().resolve()


def _resolve_path(path: str) -> Path:
    """
    Resolve a Mac path to the real absolute path on this machine.

    Order:
      1. sanitize placeholders (/Users/yourusername → real home)
      2. expand ~ / Desktop / Documents
      3. if relative name exists under Desktop/home, use that
      4. project workspace for project files
    Never requires the path to already exist (create/write targets).
    """
    raw = _clean_path_arg(path)
    if not raw:
        raise ValueError("Path cannot be empty.")

    try:
        primary = resolve_mac_path(raw)
        if primary.exists():
            return primary
    except Exception:
        primary = None

    norm = raw.replace("\\", "/").strip()
    # Strip leading ~/ for segment logic
    rel = norm[2:] if norm.startswith("~/") else norm.lstrip("/")

    # Multi-segment relative: TESrACT-1/Soumyajit or Desktop/TESrACT-1/x
    if not norm.startswith("/") and "/" in rel:
        parts = [p for p in rel.split("/") if p and p != "~"]
        if parts:
            root_name = parts[0]
            # Desktop/foo → under home/Desktop
            if root_name.lower() in ("desktop", "documents", "downloads"):
                base = real_home() / root_name.capitalize()
                return _safe_resolve(base.joinpath(*parts[1:]))
            found_root = find_named_path(root_name)
            if found_root:
                return _safe_resolve(found_root.joinpath(*parts[1:]))
            # Default: under Desktop
            return _safe_resolve((real_home() / "Desktop").joinpath(*parts))

    # Single name: TESrACT-1, notes.txt
    if not norm.startswith("/") and "/" not in rel:
        found = find_named_path(rel)
        if found:
            return found
        # Prefer Desktop for new folders/files
        desktop_cand = _safe_resolve(real_home() / "Desktop" / rel)
        ws_cand = _safe_resolve(WORKSPACE_ROOT / rel)
        if ws_cand.exists():
            return ws_cand
        return desktop_cand

    if primary is not None:
        return primary
    try:
        return resolve_mac_path(raw)
    except Exception as exc:
        raise ValueError(
            f"Could not resolve path original={path!r} cleaned={raw!r}: {exc}"
        ) from exc


def _normalize_ddg_hits(raw_hits) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in raw_hits or []:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "title": r.get("title") or "Untitled",
                "body": r.get("body") or r.get("snippet") or r.get("content") or "",
                "url": r.get("href") or r.get("link") or r.get("url") or "",
            }
        )
    return out


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    DuckDuckGo text search with backend fallbacks.

    `backend="lite"` (older code path) now often returns 0 hits. Prefer `auto` /
    `html`, and treat empty result sets as failures so we rotate backends.
    """
    if DDGS is None:
        raise RuntimeError(
            "No DuckDuckGo client installed. Run: pip install ddgs"
        )

    # Order matters: auto/html are currently reliable; lite is last-resort.
    backends: list[str | None] = ["auto", "html", None, "lite"]
    errors: list[str] = []

    for backend in backends:
        try:
            with DDGS() as ddgs:
                kwargs: dict = {"max_results": max_results}
                if backend is not None:
                    kwargs["backend"] = backend
                raw = ddgs.text(query, **kwargs)
                # Some versions return a generator; materialize fully.
                hits = _normalize_ddg_hits(list(raw) if raw is not None else [])
            if hits:
                return hits
            errors.append(f"backend={backend!r}: 0 hits")
        except TypeError:
            # Older/newer API without `backend=` kwarg
            try:
                with DDGS() as ddgs:
                    raw = ddgs.text(query, max_results=max_results)
                    hits = _normalize_ddg_hits(list(raw) if raw is not None else [])
                if hits:
                    return hits
                errors.append("backend=<default>: 0 hits")
            except Exception as exc:
                errors.append(f"backend=<default>: {exc}")
        except Exception as exc:
            errors.append(f"backend={backend!r}: {exc}")

    # News endpoint is a useful secondary path for current-events queries.
    try:
        with DDGS() as ddgs:
            raw_news = ddgs.news(query, max_results=max_results)
            news_hits = []
            for r in list(raw_news or []):
                if not isinstance(r, dict):
                    continue
                news_hits.append(
                    {
                        "title": r.get("title") or "Untitled",
                        "body": r.get("body") or r.get("excerpt") or "",
                        "url": r.get("url") or r.get("href") or "",
                    }
                )
            if news_hits:
                return news_hits
            errors.append("news: 0 hits")
    except Exception as exc:
        errors.append(f"news: {exc}")

    raise RuntimeError(
        "DuckDuckGo returned no results. " + "; ".join(errors[:6])
    )


def _search_tavily(query: str, max_results: int = 5) -> tuple[str, list[dict[str, str]]]:
    """Optional Tavily API — set TAVILY_API_KEY in .env for richer results."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    with httpx.Client(timeout=25.0) as client:
        res = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
        res.raise_for_status()
        data = res.json()

    summary = (data.get("answer") or "").strip()
    hits: list[dict[str, str]] = []
    for hit in data.get("results") or []:
        hits.append({
            "title": hit.get("title") or "Untitled",
            "body": hit.get("content") or hit.get("snippet") or "",
            "url": hit.get("url") or "",
        })
    return summary, hits


def _search_wikipedia(query: str, max_results: int = 5) -> tuple[str, list[dict[str, str]]]:
    """
    Free Wikipedia search + page summary. Much more reliable for factual
    questions (sports results, capitals, etc.) than flaky DDG scrapers.
    """
    headers = {
        "User-Agent": "TESrACT/1.0 (local assistant; contact: local)",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as client:
        search_res = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
        )
        search_res.raise_for_status()
        search_data = search_res.json()
        results = (search_data.get("query") or {}).get("search") or []
        if not results:
            raise RuntimeError("Wikipedia returned no search hits")

        hits: list[dict[str, str]] = []
        summary = ""
        for i, hit in enumerate(results[:max_results]):
            title = str(hit.get("title") or "Untitled")
            page_id = hit.get("pageid")
            snippet = re.sub(r"<[^>]+>", "", str(hit.get("snippet") or ""))
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            body = snippet
            # Enrich the top hit: plain extract + parsed lead HTML (infobox scores).
            if i == 0 and title:
                try:
                    ext_res = client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "prop": "extracts",
                            "explaintext": 1,
                            "exchars": 4500,
                            "redirects": 1,
                            "titles": title,
                            "format": "json",
                        },
                    )
                    extract = ""
                    if ext_res.status_code == 200:
                        pages = (ext_res.json().get("query") or {}).get("pages") or {}
                        for page in pages.values():
                            extract = (page.get("extract") or "").strip()
                            if extract:
                                body = extract
                                break

                    # Parse section 0 HTML — includes infobox lines like
                    # "Argentina won 4–2 on penalties" that extracts often drop.
                    parse_res = client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "parse",
                            "page": title,
                            "prop": "text",
                            "section": 0,
                            "format": "json",
                            "redirects": 1,
                        },
                    )
                    parsed_text = ""
                    if parse_res.status_code == 200:
                        try:
                            from html import unescape as _unescape

                            html = parse_res.json().get("parse", {}).get("text", {}).get("*", "")
                            parsed_text = re.sub(r"<[^>]+>", " ", html or "")
                            parsed_text = re.sub(r"\s+", " ", _unescape(parsed_text)).strip()
                        except Exception:
                            parsed_text = ""

                    outcome_bits: list[str] = []
                    for blob in (parsed_text, extract):
                        if not blob:
                            continue
                        for pat in (
                            r"([A-Z][A-Za-z .]{2,40}\s+won\s+\d[–-]\d\s+on\s+penalties)",
                            r"([A-Z][A-Za-z .]{2,40}\s+won\s+the\s+(?:match|final|tournament)[^.]*\.)",
                            r"((?:won|defeated|beat)\s+by\s+[A-Z][A-Za-z .]{2,40}[^.]*\.)",
                        ):
                            m = re.search(pat, blob, flags=re.I)
                            if m:
                                outcome_bits.append(m.group(1).strip())
                    # Dedup
                    seen = set()
                    outcomes = []
                    for bit in outcome_bits:
                        key = bit.lower()
                        if key not in seen:
                            seen.add(key)
                            outcomes.append(bit)

                    lead = (extract.split("\n\n")[0] if extract else snippet)[:700].strip()
                    if outcomes:
                        summary = (lead + " " + " ".join(outcomes[:2])).strip()
                        body = summary + ("\n\n" + extract if extract else "")
                    elif extract:
                        summary = extract[:1200]
                    elif parsed_text:
                        summary = parsed_text[:1200]
                        body = summary

                    sum_res = client.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
                    )
                    if sum_res.status_code == 200:
                        sum_data = sum_res.json()
                        if sum_data.get("content_urls", {}).get("desktop", {}).get("page"):
                            url = sum_data["content_urls"]["desktop"]["page"]
                        if not summary:
                            extract2 = (sum_data.get("extract") or "").strip()
                            if extract2:
                                body = extract2
                                summary = extract2
                except Exception:
                    pass
            hits.append({"title": title, "body": body, "url": url})

        if not summary and hits:
            summary = hits[0].get("body") or ""
        return summary, hits


def _format_search_results(
    query: str,
    hits: list[dict[str, str]],
    *,
    provider: str,
    summary: str = "",
    focus: str = "",
) -> str:
    lines: list[str] = [f"[Web search via {provider} — {len(hits)} source(s)]"]
    if focus:
        lines.append(f"Focus: {focus}")
    if summary:
        lines.append(f"\n## Summary\n{summary}")
    if hits:
        lines.append("\n## Sources")
        for i, hit in enumerate(hits, 1):
            title = hit.get("title") or "Untitled"
            body = (hit.get("body") or "").strip()
            url = hit.get("url") or ""
            snippet = body[:500] + ("…" if len(body) > 500 else "")
            lines.append(f"\n{i}. **{title}**")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   {snippet}")
    else:
        lines.append("\nNo results returned.")
    lines.append(
        "\n[Instruction: Synthesize the above into a concise answer for Sir. "
        "Cite sources by number when stating facts.]"
    )
    return "\n".join(lines)


def _run_web_search(query: str, *, max_results: int = 5, focus: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "Search Error: empty query."

    prior = format_recall_for_context(query)
    provider = "duckduckgo"
    summary = ""
    hits: list[dict[str, str]] = []
    errors: list[str] = []

    # Prefer Tavily when configured; then Wikipedia (reliable facts); then DuckDuckGo.
    if os.getenv("TAVILY_API_KEY", "").strip():
        try:
            provider = "tavily"
            summary, hits = _search_tavily(query, max_results=max_results)
        except Exception as exc:
            errors.append(f"tavily: {exc}")
            hits = []

    if not hits:
        try:
            provider = "wikipedia"
            summary, hits = _search_wikipedia(query, max_results=max_results)
        except Exception as exc:
            errors.append(f"wikipedia: {exc}")
            hits = []

    if not hits:
        try:
            provider = "duckduckgo"
            hits = _search_duckduckgo(query, max_results=max_results)
        except Exception as exc:
            errors.append(f"duckduckgo: {exc}")
            hits = []

    if not hits:
        detail = "; ".join(errors) if errors else "no providers returned results"
        return (
            f"Search Error: no web results for {query!r} ({detail}). "
            "Try a more specific query, or set TAVILY_API_KEY for a paid search API."
        )

    formatted = _format_search_results(
        query, hits, provider=provider, summary=summary, focus=focus,
    )
    try:
        store_search(query, formatted, source=provider)
    except Exception as mem_exc:
        # Memory is best-effort — never fail the tool because of Chroma/storage.
        print(f"[TESrACT:search] memory store skipped: {mem_exc}")
    header = f"Query: {query}\n"
    return header + (prior + formatted if prior else formatted)


# ---------------------------------------------------------------------------
# Image generation rubrics (preference / VLM training labels)
# Axes align with common T2I eval stacks: CLIPScore, LAION Aesthetics,
# ImageReward, PickScore, HPSv2, GenEval composition, safety filters.
# ---------------------------------------------------------------------------
IMAGE_RUBRIC_SCHEMA = "tesract.image_rubric.v1"

# (key, display name, weight for composite 0–100, short training note)
IMAGE_RUBRIC_AXES: list[tuple[str, str, float, str]] = [
    (
        "prompt_alignment",
        "Prompt alignment",
        0.22,
        "CLIPScore / text–image fidelity — primary signal for SFT & RLHF",
    ),
    (
        "aesthetic_quality",
        "Aesthetic quality",
        0.16,
        "LAION Aesthetics predictor style overall appeal",
    ),
    (
        "composition",
        "Composition",
        0.12,
        "GenEval / spatial layout, framing, subject placement",
    ),
    (
        "detail_sharpness",
        "Detail & sharpness",
        0.10,
        "Texture fidelity, edge clarity, fine structure",
    ),
    (
        "color_lighting",
        "Color & lighting",
        0.10,
        "Palette harmony, exposure, cinematic light",
    ),
    (
        "style_consistency",
        "Style consistency",
        0.10,
        "Requested medium/style held across the frame",
    ),
    (
        "artifact_freedom",
        "Artifact freedom",
        0.10,
        "Hands/faces/text glitches, distortion, compression",
    ),
    (
        "safety_suitability",
        "Safety suitability",
        0.10,
        "NSFW / violence / disallowed content (training filter)",
    ),
]


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read width/height from PNG/JPEG headers without heavy deps."""
    try:
        data = path.read_bytes()[:64]
    except Exception:
        return None
    # PNG: IHDR at offset 16 after 8-byte signature
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        if w > 0 and h > 0:
            return w, h
    # JPEG SOF0/SOF2 scan (best-effort)
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        try:
            raw = path.read_bytes()
            i = 2
            while i < len(raw) - 9:
                if raw[i] != 0xFF:
                    i += 1
                    continue
                marker = raw[i + 1]
                if marker in (0xC0, 0xC2):  # SOF0 / SOF2
                    h = int.from_bytes(raw[i + 5 : i + 7], "big")
                    w = int.from_bytes(raw[i + 7 : i + 9], "big")
                    if w > 0 and h > 0:
                        return w, h
                    break
                if marker == 0xD9:
                    break
                if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
                    i += 2
                    continue
                seglen = int.from_bytes(raw[i + 2 : i + 4], "big")
                i += 2 + seglen
        except Exception:
            return None
    return None


def _clamp_score(v: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, round(v, 2)))


def _heuristic_image_rubrics(
    prompt: str,
    image_path: Path | None,
    *,
    backend: str = "pollinations",
) -> dict:
    """
    Deterministic 1–5 scores usable as weak labels for preference datasets.
    Not a substitute for human/CLIP/ImageReward judges — export as `method=heuristic`.
    """
    p = (prompt or "").strip()
    words = re.findall(r"[A-Za-z0-9']+", p.lower())
    n_words = len(words)
    style_hits = sum(
        1
        for k in (
            "cinematic", "photorealistic", "photo", "oil", "watercolor", "anime",
            "pixel", "3d", "render", "studio", "bokeh", "neon", "cyberpunk",
            "illustration", "sketch", "macro", "portrait", "landscape",
        )
        if k in p.lower()
    )
    composition_hits = sum(
        1
        for k in (
            "wide", "close-up", "closeup", "rule of thirds", "centered",
            "overhead", "low angle", "symmetry", "depth of field", "foreground",
            "background", "panorama", "isometric",
        )
        if k in p.lower()
    )
    lighting_hits = sum(
        1
        for k in (
            "golden hour", "sunset", "sunrise", "rim light", "soft light",
            "dramatic", "volumetric", "neon", "moonlight", "studio lighting",
            "overcast", "hdr",
        )
        if k in p.lower()
    )
    safety_flags = sum(
        1
        for k in (
            "nsfw", "nude", "naked", "gore", "blood", "explicit", "porn",
            "violence", "bloody", "erotic",
        )
        if k in p.lower()
    )

    size_bytes = 0
    dims = None
    if image_path and image_path.is_file():
        try:
            size_bytes = image_path.stat().st_size
        except OSError:
            size_bytes = 0
        dims = _png_dimensions(image_path)

    # Base quality from bytes (empty / tiny = low)
    if size_bytes <= 0:
        size_factor = 1.5
    elif size_bytes < 20_000:
        size_factor = 2.5
    elif size_bytes < 80_000:
        size_factor = 3.3
    elif size_bytes < 250_000:
        size_factor = 4.0
    else:
        size_factor = 4.4

    res_factor = 3.5
    if dims:
        px = dims[0] * dims[1]
        if px >= 1024 * 1024:
            res_factor = 4.5
        elif px >= 512 * 512:
            res_factor = 4.0
        else:
            res_factor = 3.0

    prompt_specificity = _clamp_score(2.2 + min(n_words, 40) * 0.05 + style_hits * 0.15)
    alignment = _clamp_score(prompt_specificity + (0.2 if n_words >= 6 else -0.4))
    aesthetic = _clamp_score((size_factor + res_factor) / 2 + style_hits * 0.08)
    composition = _clamp_score(3.2 + composition_hits * 0.25 + (0.2 if n_words >= 8 else 0))
    detail = _clamp_score(size_factor * 0.55 + res_factor * 0.45)
    color = _clamp_score(3.3 + lighting_hits * 0.22 + style_hits * 0.05)
    style = _clamp_score(3.0 + style_hits * 0.35 + (0.3 if backend == "openai" else 0))
    artifacts = _clamp_score(size_factor - 0.1 + (0.2 if backend == "openai" else 0))
    safety = 1.5 if safety_flags else 4.8

    scores = {
        "prompt_alignment": alignment,
        "aesthetic_quality": aesthetic,
        "composition": composition,
        "detail_sharpness": detail,
        "color_lighting": color,
        "style_consistency": style,
        "artifact_freedom": artifacts,
        "safety_suitability": safety,
    }
    rationales = {
        "prompt_alignment": (
            f"Prompt specificity ~{n_words} tokens; style cues={style_hits}. "
            "Proxy for CLIP-style text–image alignment."
        ),
        "aesthetic_quality": (
            f"File {size_bytes} B"
            + (f", {dims[0]}×{dims[1]}" if dims else "")
            + "; aesthetic proxy from resolution/size + style keywords."
        ),
        "composition": f"Composition cues in prompt: {composition_hits}.",
        "detail_sharpness": f"Detail proxy from payload size={size_bytes} and resolution.",
        "color_lighting": f"Lighting/style keywords: {lighting_hits}/{style_hits}.",
        "style_consistency": f"Backend={backend}; style keywords={style_hits}.",
        "artifact_freedom": "Heuristic — true artifact checks need a vision judge.",
        "safety_suitability": (
            "Flagged risk terms in prompt." if safety_flags else "No hard safety keywords detected."
        ),
    }
    return {
        "scores": scores,
        "rationales": rationales,
        "dims": dims,
        "size_bytes": size_bytes,
    }


def build_image_rubric(
    prompt: str,
    image_path: Path | None,
    *,
    backend: str = "pollinations",
    public_url: str = "",
) -> dict:
    """
    Build a structured rubric payload for VLM / preference training pipelines.
    Writes ``*.rubric.json`` next to the image when possible.
    """
    heuristic = _heuristic_image_rubrics(prompt, image_path, backend=backend)
    scores: dict[str, float] = heuristic["scores"]
    rationales: dict[str, str] = heuristic["rationales"]

    dimensions: dict[str, dict] = {}
    weighted = 0.0
    for key, label, weight, note in IMAGE_RUBRIC_AXES:
        s = float(scores.get(key, 3.0))
        dimensions[key] = {
            "label": label,
            "score": s,
            "scale": "1-5",
            "weight": weight,
            "rationale": rationales.get(key, ""),
            "training_note": note,
        }
        weighted += s * weight

    # Map weighted 1–5 → 0–100 preference-style composite (PickScore/ImageReward-like)
    composite = round(max(0.0, min(100.0, (weighted - 1.0) / 4.0 * 100.0)), 2)

    image_rel = ""
    resolved_image: Path | None = None
    if image_path is not None:
        try:
            resolved_image = Path(image_path).expanduser().resolve()
        except Exception:
            resolved_image = Path(image_path)
        if resolved_image.is_file():
            try:
                image_rel = resolved_image.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
            except ValueError:
                image_rel = resolved_image.as_posix()

    payload = {
        "schema": IMAGE_RUBRIC_SCHEMA,
        "use": "preference_and_vlm_training_labels",
        "prompt": (prompt or "").strip(),
        "backend": backend,
        "image_path": image_rel,
        "public_url": public_url or "",
        "width": heuristic["dims"][0] if heuristic["dims"] else None,
        "height": heuristic["dims"][1] if heuristic["dims"] else None,
        "size_bytes": heuristic["size_bytes"],
        "method": "heuristic_v1",
        "dimensions": dimensions,
        "composite_score": composite,
        "composite_scale": "0-100",
        "preference_label": (
            "preferred" if composite >= 72 else "borderline" if composite >= 55 else "reject"
        ),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if resolved_image is not None and resolved_image.is_file():
        # Prefer clean stem.rubric.json next to image
        side = resolved_image.with_name(resolved_image.stem + ".rubric.json")
        try:
            side.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            try:
                payload["rubric_sidecar"] = side.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
            except ValueError:
                payload["rubric_sidecar"] = side.as_posix()
        except Exception as exc:
            print(f"[TESrACT:rubric] sidecar write skipped: {exc}")

    return payload


def format_image_rubric_markdown(payload: dict) -> str:
    """Human + LLM readable rubric table for chat / training export."""
    lines = [
        "### Image rubrics (LLM training labels)",
        "",
        "| Dimension | Score (1–5) | Training note |",
        "|---|---:|---|",
    ]
    dims = payload.get("dimensions") or {}
    for key, label, _w, note in IMAGE_RUBRIC_AXES:
        d = dims.get(key) or {}
        score = d.get("score", "—")
        lines.append(f"| {label} | **{score}** | {note} |")
    composite = payload.get("composite_score", "—")
    pref = payload.get("preference_label", "—")
    lines.append("")
    lines.append(
        f"**Composite preference score:** {composite}/100 "
        f"· label=`{pref}` · schema=`{payload.get('schema', IMAGE_RUBRIC_SCHEMA)}`"
    )
    if payload.get("rubric_sidecar"):
        lines.append(f"**Rubric JSON:** `/{payload['rubric_sidecar']}` (downloadable)")
    lines.append(
        "_Axes inspired by CLIPScore, LAION Aesthetics, ImageReward, PickScore, "
        "HPSv2, and GenEval — heuristic weak labels for dataset building._"
    )
    return "\n".join(lines)


def _attach_image_rubric(
    body: str,
    prompt: str,
    *,
    backend: str,
    public_url: str = "",
    image_path: Path | None = None,
) -> str:
    """Append rubric markdown + ensure sidecar exists."""
    path = image_path
    if path is None and public_url:
        # Map /generated/foo.png → project file
        rel = public_url.lstrip("/")
        candidate = WORKSPACE_ROOT / rel
        if candidate.is_file():
            path = candidate
    try:
        payload = build_image_rubric(
            prompt, path, backend=backend, public_url=public_url,
        )
        return body.rstrip() + "\n\n" + format_image_rubric_markdown(payload)
    except Exception as exc:
        print(f"[TESrACT:rubric] attach failed: {exc}")
        return body


def _generate_via_pollinations(prompt: str, *, width: int = 1024, height: int = 1024) -> str:
    """
    Free Pollinations path — delegates to mac_access.generate_free_image.
    width/height kept for API compat; free pipeline uses 1024×1024.
    """
    _ = width, height
    md = _mac_generate_free_image(prompt)
    if md.startswith("Image Error:") or md.startswith("Search Error:"):
        raise RuntimeError(md)
    # Keep memory + structured SUCCESS envelope while embedding the markdown
    # the HUD renders inline.
    m = re.search(r"\((/generated/[^)]+)\)", md)
    local_url = m.group(1) if m else ""
    try:
        store_image_memory(prompt, md, backend="pollinations", image_ref=local_url.lstrip("/"))
    except Exception as mem_exc:
        print(f"[TESrACT:image] memory store skipped: {mem_exc}")
    body = (
        f"STATUS: SUCCESS\n"
        f"Image generated successfully (free Pollinations pipeline).\n"
        f"Prompt: {prompt}\n"
        f"Local URL: {local_url}\n"
        f"Download: {local_url}\n"
        f"{md}"
    )
    return _attach_image_rubric(
        body, prompt, backend="pollinations", public_url=local_url,
    )


def _generate_via_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    _ensure_dirs()
    with httpx.Client(timeout=90.0) as client:
        res = client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1024x1024"},
        )
        res.raise_for_status()
        data = res.json()

    image_url = data["data"][0]["url"]
    stamp = int(time.time())
    filename = f"img_{stamp}_dalle.png"
    local_path = GENERATED_DIR / filename

    with httpx.Client(timeout=90.0) as client:
        img_res = client.get(image_url)
        img_res.raise_for_status()
        local_path.write_bytes(img_res.content)

    rel = local_path.relative_to(WORKSPACE_ROOT).as_posix()
    md = f"![Generated Image](/generated/{filename})"
    result = (
        f"STATUS: SUCCESS\n"
        f"Image generated via DALL-E 3.\n"
        f"Prompt: {prompt}\n"
        f"Saved to: {rel}\n"
        f"Local URL: /generated/{filename}\n"
        f"Download: /generated/{filename}\n"
        f"{md}"
    )
    store_image_memory(prompt, result, backend="openai", image_ref=rel)
    return _attach_image_rubric(
        result,
        prompt,
        backend="openai",
        public_url=f"/generated/{filename}",
        image_path=local_path,
    )


@tool
def web_search(query: str):
    """Search the web for current information, facts, or news. Use for any question needing up-to-date data."""
    try:
        body = _run_web_search(query, max_results=5)
        if body.startswith("Search Error:"):
            return _tool_err("web_search", body)
        return _tool_ok("web_search", body)
    except Exception as exc:
        return _tool_err("web_search", str(exc))


@tool
def search_and_summarize(query: str, focus: str = ""):
    """Deep web research with structured summary and numbered citations. Use for research, news, or 'find out about X'."""
    try:
        body = _run_web_search(query, max_results=6, focus=(focus or "").strip())
        if body.startswith("Search Error:"):
            return _tool_err("search_and_summarize", body)
        return _tool_ok("search_and_summarize", body)
    except Exception as exc:
        return _tool_err("search_and_summarize", str(exc))


@tool
def generate_free_image(prompt: str) -> str:
    """
    Free local image generation via Pollinations AI (no API key).
    Always use this (or generate_image) when Sir asks to create, draw, or generate an image.
    Returns markdown the HUD renders inline: ![Generated Image](/generated/img_….png)
    Also attaches multi-axis image rubrics (CLIP / aesthetics / ImageReward-style)
    and a downloadable .rubric.json sidecar for VLM preference training.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return _tool_err("generate_free_image", "prompt cannot be empty.")
    try:
        md = _mac_generate_free_image(prompt)
        if md.startswith("Image Error:"):
            return _tool_err("generate_free_image", md)
        local_url = ""
        try:
            m = re.search(r"\((/generated/[^)]+)\)", md)
            local_url = m.group(1) if m else ""
            store_image_memory(
                prompt,
                md,
                backend="pollinations",
                image_ref=(local_url.lstrip("/") if local_url else ""),
            )
        except Exception as mem_exc:
            print(f"[TESrACT:image] memory store skipped: {mem_exc}")
        body = (
            f"Image ready for Sir.\n"
            f"Download: {local_url or '/generated/'}\n"
            f"{md}"
        )
        body = _attach_image_rubric(
            body, prompt, backend="pollinations", public_url=local_url,
        )
        # Tool envelope + strict markdown for synthesis / HUD
        return _tool_ok("generate_free_image", body)
    except Exception as exc:
        return _tool_err("generate_free_image", str(exc))


@tool
def generate_image(prompt: str, style: str = ""):
    """Generate an image from a text description. Uses free Pollinations by default. Always call this when Sir asks to create, draw, or generate an image."""
    prompt = (prompt or "").strip()
    if not prompt:
        return _tool_err("generate_image", "prompt cannot be empty.")

    full_prompt = f"{prompt}, {style}".strip(", ") if style else prompt
    # Free pipeline first — no paid API required
    backends: list[tuple[str, object]] = [
        ("pollinations", lambda: _generate_via_pollinations(full_prompt)),
    ]
    if os.getenv("OPENAI_API_KEY", "").strip():
        backends.append(("openai", lambda: _generate_via_openai(full_prompt)))

    errors: list[str] = []
    for name, fn in backends:
        try:
            body = fn()
            return _tool_ok("generate_image", body)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    _ensure_dirs()
    prompt_path = GENERATED_DIR / f"prompt_{int(time.time())}.txt"
    prompt_path.write_text(full_prompt, encoding="utf-8")
    return _tool_err(
        "generate_image",
        f"Image backends unavailable ({'; '.join(errors)}). "
        f"Saved prompt to {prompt_path.relative_to(WORKSPACE_ROOT).as_posix()}.",
    )


def _resolve_project_file_path(file_path: str) -> Path:
    """
    Resolve upload / project-local paths for document / image tools.

    The HUD passes URL-style paths with a leading slash
    (e.g. ``/agent_data/uploads/uuid.jpeg``). Treating that as a filesystem
    absolute path looks under the server root and raises FileNotFoundError.
    Strip leading slashes and join against the process CWD (project root).
    Blocks path escape outside the project tree.
    """
    raw = (file_path or "").strip().strip("'\"")
    if not raw:
        raise FileNotFoundError("empty file_path")

    root = WORKSPACE_ROOT.resolve()
    cwd_root = Path(os.getcwd()).resolve()
    candidates: list[Path] = []

    # Prefer already-resolved absolute paths under project/CWD (no double-join)
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        try:
            exp_res = expanded.resolve(strict=False)
            under_project = False
            try:
                exp_res.relative_to(root)
                under_project = True
            except ValueError:
                try:
                    exp_res.relative_to(cwd_root)
                    under_project = True
                except ValueError:
                    under_project = False
            if under_project:
                candidates.append(expanded)
        except Exception:
            pass

    # Frontend: /agent_data/uploads/... → agent_data/uploads/... under CWD
    clean_path = raw.lstrip("/")
    actual_path = os.path.join(os.getcwd(), clean_path)
    for c in (Path(actual_path), WORKSPACE_ROOT / clean_path):
        if c not in candidates:
            candidates.append(c)

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            # Must stay under project root or current working directory
            try:
                resolved.relative_to(root)
            except ValueError:
                resolved.relative_to(cwd_root)
            if resolved.is_file():
                return resolved
            last_err = FileNotFoundError(str(resolved))
        except Exception as exc:
            last_err = exc
            continue

    raise FileNotFoundError(
        f"path not allowed or not found: {raw!r} (tried under CWD and project root)"
    ) from last_err


def _truncate_middle_for_memory(text: str, max_chars: int = 3000) -> tuple[str, bool]:
    """
    Keep head + tail of a large extraction so LangGraph/SQLite state stays small.
    """
    if max_chars < 64:
        max_chars = 64
    if len(text) <= max_chars:
        return text, False
    marker = "\n\n[...TRUNCATED FOR MEMORY...]\n\n"
    budget = max_chars - len(marker)
    if budget < 32:
        return text[:max_chars], True
    head_n = budget // 2
    tail_n = budget - head_n
    return text[:head_n] + marker + text[-tail_n:], True


@tool
def extract_pdf_context(file_path: str) -> str:
    """
    Extract full text from a local PDF into active memory.

    Use ONLY for .pdf paths under /agent_data/uploads/ (or other project-local PDFs).
    NEVER call this on images (.jpg/.jpeg/.png) — use analyze_uploaded_image instead.
    Large extracts are middle-truncated (~3k chars) to protect Render RAM / checkpointer.
    """
    # Hard gate on extension so image uploads never enter the PDF parser
    if not (file_path or "").strip().lower().endswith(".pdf"):
        return "Error: This tool is for PDFs only. Use the image tool for images."

    # HUD paths arrive as /agent_data/... (URL-absolute). Never open as FS root.
    clean_path = (file_path or "").lstrip("/")
    actual_path = os.path.join(os.getcwd(), clean_path)

    try:
        path = _resolve_project_file_path(actual_path)
    except FileNotFoundError as exc:
        return _tool_err(
            "extract_pdf_context",
            f"FileNotFoundError: could not open PDF at {file_path!r} "
            f"(resolved {actual_path!r}: {exc}).",
        )
    except Exception as exc:
        return _tool_err("extract_pdf_context", f"Invalid path: {exc}")

    if path.suffix.lower() != ".pdf":
        return "Error: This tool is for PDFs only. Use the image tool for images."

    try:
        from pypdf import PdfReader
    except ImportError:
        return _tool_err(
            "extract_pdf_context",
            "pypdf is not installed. Run: pip install pypdf",
        )

    # Default 3000 chars — keeps tool output + LangGraph SQLite checkpoints lean on 512MB Render
    max_chars = int(os.getenv("PDF_EXTRACT_MAX_CHARS", "3000") or "3000")
    reader = None
    pages_text: list[str] = []
    page_count = 0
    try:
        # Open via resolved absolute path under project/CWD (not server root)
        open_path = str(path)
        # strict=False reduces some validation overhead on damaged PDFs
        try:
            reader = PdfReader(open_path, strict=False)
        except TypeError:
            reader = PdfReader(open_path)
        page_count = len(reader.pages)
        # Gather only a bit more than max_chars so middle-truncate still has tail context
        gather_budget = max(max_chars * 3, max_chars + 512)
        total = 0
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as page_exc:
                text = f"[page {i + 1} extract error: {page_exc}]"
            text = text.strip()
            if text:
                block = f"--- Page {i + 1} ---\n{text}"
                pages_text.append(block)
                total += len(block)
            # Stop early — no need to parse the whole book into RAM on free tier
            if total >= gather_budget:
                pages_text.append(
                    f"\n[…stopped after page {i + 1}/{page_count} for memory limits…]"
                )
                break
            # Drop page reference promptly
            del page

        combined = "\n\n".join(pages_text).strip()
        pages_text.clear()
        del pages_text

        if not combined:
            body = (
                f"PDF opened ({path.name}, {page_count} page(s)) but no extractable text "
                "(may be scanned/image-only)."
            )
            return _tool_ok("extract_pdf_context", body)

        combined, truncated = _truncate_middle_for_memory(combined, max_chars=max_chars)
        body = (
            f"Source: {path.name}\n"
            f"Pages: {page_count}\n"
            f"Truncated: {truncated}\n\n"
            f"{combined}"
        )
        # Free large string intermediate before packaging tool result
        del combined
        return _tool_ok("extract_pdf_context", body)
    except FileNotFoundError as exc:
        return _tool_err(
            "extract_pdf_context",
            f"FileNotFoundError: {exc}",
        )
    except Exception as exc:
        return _tool_err("extract_pdf_context", f"Failed to read PDF: {exc}")
    finally:
        # Ruthlessly dump PDF parser / page objects (Render 512MB OOM guard)
        try:
            del reader
        except Exception:
            pass
        try:
            del pages_text
        except Exception:
            pass
        gc.collect()


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
# Prefer env override; try common local vision tags in order.
_VISION_MODEL_CANDIDATES = (
    os.getenv("OLLAMA_VISION_MODEL", "").strip(),
    "llama3.2-vision",
    "llava",
    "llava:latest",
    "moondream",
)


def _ollama_host() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")


def _vision_model_candidates() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in _VISION_MODEL_CANDIDATES:
        key = (name or "").strip()
        if not key:
            continue
        low = key.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(key)
    return out or ["llava", "llama3.2-vision"]


def _call_ollama_vision(image_path: Path, prompt: str) -> tuple[str, str]:
    """
    Send image + prompt to a local Ollama vision model.

    Returns (model_used, response_text). Raises RuntimeError with a user-facing
    message when Ollama is down or no vision model is pulled.
    """
    models = _vision_model_candidates()
    last_err: Exception | None = None
    missing_models: list[str] = []

    # Prefer official Ollama Python client (local daemon).
    try:
        import ollama  # type: ignore
    except ImportError:
        ollama = None  # type: ignore

    if ollama is not None:
        try:
            client = ollama.Client(host=_ollama_host())
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to local Ollama at {_ollama_host()}: {exc}. "
                "Start Ollama (e.g. `ollama serve`) and pull a vision model."
            ) from exc

        for model in models:
            try:
                response = client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [str(image_path)],
                        }
                    ],
                )
                # ollama>=0.4 returns ChatResponse; older returns dict
                if isinstance(response, dict):
                    content = (
                        (response.get("message") or {}).get("content")
                        or response.get("response")
                        or ""
                    )
                else:
                    msg = getattr(response, "message", None)
                    content = (
                        getattr(msg, "content", None)
                        if msg is not None
                        else str(response)
                    ) or ""
                content = str(content).strip()
                if not content:
                    raise RuntimeError(f"Empty response from vision model {model!r}")
                return model, content
            except Exception as exc:
                last_err = exc
                err_s = str(exc).lower()
                if (
                    "not found" in err_s
                    or "pull" in err_s
                    or "model" in err_s and "404" in err_s
                    or "try pulling" in err_s
                ):
                    missing_models.append(model)
                    continue
                # Connection refused / daemon down
                if any(
                    k in err_s
                    for k in ("connection", "refused", "connect", "unreachable", "timed out")
                ):
                    raise RuntimeError(
                        f"Local Ollama is not reachable at {_ollama_host()}: {exc}. "
                        "Start Ollama, then run: ollama pull llava  (or llama3.2-vision)"
                    ) from exc
                # Other model errors — try next candidate
                missing_models.append(model)
                continue

    # httpx fallback (same API the rest of TESrACT uses for health checks)
    try:
        import base64

        raw = image_path.read_bytes()
        # Soft cap ~8MB encoded payload to avoid OOM on free tier
        max_bytes = int(os.getenv("VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)) or str(8 * 1024 * 1024))
        if len(raw) > max_bytes:
            raise RuntimeError(
                f"Image too large for vision tool ({len(raw)} bytes; limit {max_bytes})."
            )
        b64 = base64.b64encode(raw).decode("ascii")
        del raw
        host = _ollama_host()
        for model in models:
            try:
                with httpx.Client(timeout=float(os.getenv("OLLAMA_TIMEOUT", "90") or "90")) as client:
                    res = client.post(
                        f"{host}/api/chat",
                        json={
                            "model": model,
                            "stream": False,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt,
                                    "images": [b64],
                                }
                            ],
                        },
                    )
                if res.status_code == 404 or (
                    res.status_code >= 400
                    and "not found" in (res.text or "").lower()
                ):
                    missing_models.append(model)
                    last_err = RuntimeError(res.text[:300])
                    continue
                res.raise_for_status()
                data = res.json()
                content = (
                    (data.get("message") or {}).get("content")
                    or data.get("response")
                    or ""
                ).strip()
                if not content:
                    raise RuntimeError(f"Empty response from vision model {model!r}")
                return model, content
            except RuntimeError:
                raise
            except Exception as exc:
                last_err = exc
                err_s = str(exc).lower()
                if any(k in err_s for k in ("connection", "refused", "connect", "unreachable")):
                    raise RuntimeError(
                        f"Local Ollama is not reachable at {host}: {exc}. "
                        "Start Ollama, then run: ollama pull llava  (or llama3.2-vision)"
                    ) from exc
                missing_models.append(model)
                continue
    except RuntimeError:
        raise
    except Exception as exc:
        last_err = exc

    tried = ", ".join(models) if models else "(none)"
    pull_hint = "ollama pull llava" if "llava" in tried else f"ollama pull {models[0]}"
    detail = f" Last error: {last_err}" if last_err else ""
    raise RuntimeError(
        f"No local vision model available (tried: {tried}). "
        f"Please pull a vision model on this Mac, e.g. `{pull_hint}` "
        f"or `ollama pull llama3.2-vision`, then retry.{detail}"
    )


@tool
def analyze_uploaded_image(file_path: str, prompt: str = "") -> str:
    """
    Analyze an uploaded image with a local Ollama vision model (llava / llama3.2-vision).

    Use when Sir uploads an image (.jpg, .jpeg, .png, …) under /agent_data/uploads/
    or asks what is in a picture. NEVER use extract_pdf_context on image files.
    `prompt` is the question about the image (defaults to a general describe request).
    """
    # HUD paths arrive as /agent_data/... (URL-absolute). Never open as FS root.
    clean_path = (file_path or "").lstrip("/")
    actual_path = os.path.join(os.getcwd(), clean_path)

    try:
        path = _resolve_project_file_path(actual_path)
    except FileNotFoundError as exc:
        return _tool_err(
            "analyze_uploaded_image",
            f"FileNotFoundError: could not open image at {file_path!r} "
            f"(resolved {actual_path!r}: {exc}).",
        )
    except Exception as exc:
        return _tool_err("analyze_uploaded_image", f"Invalid path: {exc}")

    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        return _tool_err(
            "analyze_uploaded_image",
            f"Not an image file: {path.name}. "
            "Use extract_pdf_context for PDFs, or provide .jpg/.jpeg/.png.",
        )

    question = (prompt or "").strip() or (
        "Describe this image in detail. Note any text, objects, people, and context."
    )

    try:
        # Pass resolved absolute path so Ollama never sees a leading-slash URL path
        model, content = _call_ollama_vision(path, question)
        # Bound tool output so LangGraph checkpoints stay lean
        max_chars = int(os.getenv("VISION_RESULT_MAX_CHARS", "4000") or "4000")
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[…truncated…]"
            truncated = True
        body = (
            f"Source: {path.name}\n"
            f"Model: {model}\n"
            f"Prompt: {question}\n"
            f"Truncated: {truncated}\n\n"
            f"{content}"
        )
        return _tool_ok("analyze_uploaded_image", body)
    except Exception as exc:
        return _tool_err("analyze_uploaded_image", str(exc))


@tool
def execute_python_code(code: str, control_allowed: bool = False):
    """
    Run process-local Python for math, logic, data transforms, or code experiments.
    Does NOT touch the host filesystem or shell. Requires Allow Control.
    """
    if not _control_ok(control_allowed):
        return _permission_denied("execute_python_code")
    code = (code or "").strip()
    if not code:
        return _tool_err("execute_python_code", "code cannot be empty.")
    # Soft guard: reject obvious OS-mutation / network escapes inside the REPL
    lowered = code.lower()
    banned = (
        "subprocess", "os.system", "os.popen", "os.remove", "os.unlink",
        "os.rmdir", "shutil.rmtree", "shutil.move", "shutil.copy",
        "webbrowser", "socket.", "pty.", "ctypes.",
        "__import__('subprocess')", '__import__("subprocess")',
    )
    if any(tok in lowered for tok in banned):
        return _tool_err(
            "execute_python_code",
            "BLOCKED: code attempts host I/O or process control. "
            "Use client-execution tools (write_file / run_terminal_command) instead.",
        )
    try:
        result = repl.run(code)
        output = str(result).strip() if result is not None else "(no output)"
        return _tool_ok(
            "execute_python_code",
            f"Brain-local execution complete (no host mutation):\n{output}",
        )
    except Exception as exc:
        return _tool_err("execute_python_code", str(exc))


@tool
def read_file(path: str, max_chars: int = 12000, control_allowed: bool = False):
    """
    Request a text-file read as a CLIENT-SIDE intent.
    The Brain does not read the host filesystem; the client must supply content.
    """
    if not _control_ok(control_allowed):
        return _permission_denied("read_file")
    if not (path or "").strip():
        return _tool_err("read_file", "path cannot be empty.")
    try:
        logical = str(_resolve_path(sanitize_path_string(path)))
        remember_path(logical)
        return _client_intent_ok(
            "read_file",
            intent_read_file(path=logical, max_chars=max_chars),
        )
    except Exception as exc:
        return _tool_err("read_file", f"{exc} | original_path={path!r}")


@tool
def write_file(
    path: str,
    content: str,
    control_allowed: bool = False,
    append: bool = False,
):
    """
    Package file content as a CLIENT-SIDE DOWNLOAD_FILE intent.
    The Brain never writes to the host filesystem.
    """
    if not _control_ok(control_allowed):
        return _permission_denied("write_file")
    path = (path or "").strip()
    if not path:
        return _tool_err("write_file", "path cannot be empty.")
    path = sanitize_path_string(path)
    try:
        logical = str(_resolve_path(path))
        remember_path(logical)
        filename = Path(logical).name or "download.txt"
        return _client_intent_ok(
            "write_file",
            intent_download_file(
                filename=filename,
                content=content or "",
                suggested_path=logical,
                append=bool(append),
            ),
        )
    except Exception as exc:
        return _tool_err("write_file", f"{exc} | original_path={path!r}")


@tool
def create_directory(path: str, control_allowed: bool = False):
    """
    Package a mkdir request as a CLIENT-SIDE DISPLAY_DATA intent.
    Prefer ~/Desktop/... logical paths. Brain does not create host folders.
    """
    if not _control_ok(control_allowed):
        return _permission_denied("create_directory")
    path = (path or "").strip()
    if not path:
        return _tool_err("create_directory", "path cannot be empty.")
    path = sanitize_path_string(path)
    try:
        logical = str(_resolve_path(path))
        remember_path(logical)
        return _client_intent_ok(
            "create_directory",
            intent_create_directory(path=logical),
        )
    except Exception as exc:
        return _tool_err("create_directory", f"{exc} | original_path={path!r}")


@tool
def list_directory(
    path: str = "~",
    max_entries: int = 200,
    control_allowed: bool = False,
):
    """
    Package a directory listing as a CLIENT-SIDE DISPLAY_DATA intent.
    The Brain does not enumerate the host filesystem.
    """
    if not _control_ok(control_allowed):
        return _permission_denied("list_directory")
    try:
        raw = (path or "~").strip() or "~"
        logical = str(_resolve_path(sanitize_path_string(raw)))
        return _client_intent_ok(
            "list_directory",
            intent_list_directory(path=logical, max_entries=max_entries),
        )
    except Exception as exc:
        return _tool_err("list_directory", f"{exc} | original_path={path!r}")


@tool
def run_terminal_command(
    command: str,
    working_directory: str = "",
    user_confirmed: bool = False,
    control_allowed: bool = False,
    thread_id: str = "web_user_001",
):
    """
    Package a shell command as a CLIENT-SIDE intent. Never executed on the Brain host.
    Dangerous commands require confirmation before packaging.
    Blocked commands are refused entirely.
    """
    if not _control_ok(control_allowed):
        return _permission_denied("run_terminal_command")

    cmd = rewrite_paths_in_command((command or "").strip())
    if not cmd:
        return _tool_err("run_terminal_command", "command cannot be empty.")

    level = classify_command(cmd)
    if level == "blocked":
        return _tool_err(
            "run_terminal_command",
            f"BLOCKED: destructive system command refused by Brain policy. command={cmd!r}",
        )
    if level == "confirm" and not user_confirmed:
        set_pending_confirmation(
            thread_id,
            tool="run_terminal_command",
            command=cmd,
            reason="client-side dangerous command",
            working_directory=(working_directory or "").strip(),
        )
        return _tool_err(
            "run_terminal_command",
            format_confirmation_request(cmd, reason="client-side dangerous command"),
        )

    wd = sanitize_path_string(working_directory) if working_directory else ""
    return _client_intent_ok(
        "run_terminal_command",
        intent_run_command(command=cmd, working_directory=wd),
    )


@tool
def open_url_in_browser(url: str, control_allowed: bool = False):
    """Package a URL open as a CLIENT-SIDE DISPLAY_DATA intent (no host browser launch)."""
    if not _control_ok(control_allowed):
        return _permission_denied("open_url_in_browser")
    url = (url or "").strip()
    if not url:
        return _tool_err("open_url_in_browser", "URL cannot be empty.")
    return _client_intent_ok("open_url_in_browser", intent_open_url(url=url))


def _resolve_display_timezone():
    """
    Timezone for user-facing clock answers.

    Priority:
      1. TESRACT_TIMEZONE (e.g. Asia/Kolkata)
      2. TZ
      3. System local timezone
    """
    name = (os.getenv("TESRACT_TIMEZONE") or os.getenv("TZ") or "").strip()
    if name:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(name), name
        except Exception:
            pass
    # Fall back to the host's local tz (Mac = IST for this project; Render = often UTC).
    return datetime.datetime.now().astimezone().tzinfo, (
        datetime.datetime.now().astimezone().tzname() or "local"
    )


def format_current_time() -> str:
    """Human-readable current time with explicit timezone (never bare UTC on Render)."""
    tz, label = _resolve_display_timezone()
    now = datetime.datetime.now(tz) if tz is not None else datetime.datetime.now()
    # %Z may be empty for some ZoneInfo builds — always append a clear label.
    zone = now.strftime("%Z") or label or "local"
    return now.strftime(f"It is %A, %B %d %Y, %I:%M %p {zone}.")


@tool
def get_current_time() -> str:
    """Returns the current local date and time. Use when Sir asks about time, date, or 'what day is it'. Always call this tool — never guess the time."""
    return _tool_ok("get_current_time", format_current_time())


def _load_task_plan() -> dict:
    _ensure_dirs()
    if not TASKS_FILE.exists():
        return {}
    try:
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_task_plan(plan: dict) -> None:
    _ensure_dirs()
    TASKS_FILE.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _format_task_plan(plan: dict) -> str:
    if not plan:
        return "STATUS: EMPTY\nNo active task plan."
    lines = [f"STATUS: ACTIVE\nTask: {plan.get('description', 'unknown')}", "Steps:"]
    for i, step in enumerate(plan.get("steps") or [], 1):
        lines.append(f"  {i}. [{step.get('status', 'pending')}] {step.get('text', '')}")
    return "\n".join(lines)


@tool("recall_memory")
def recall_memory_tool(query: str, limit: int = 4):
    """Search local memory for prior searches, notes, and conversation context. Use before repeating work or when Sir references past interactions."""
    query = (query or "").strip()
    if not query:
        return _tool_err("recall_memory", "query cannot be empty.")

    hits = recall_memory(query, n=min(max(limit, 1), 8))
    if not hits:
        return _tool_ok("recall_memory", f"No relevant memory found for: {query}")

    lines = [f"Found {len(hits)} memory hit(s) for: {query}", ""]
    for i, hit in enumerate(hits, 1):
        media = hit.get("media_type") or "search"
        topic = hit.get("topic") or "unknown"
        body = str(hit.get("content") or "")[:700]
        lines.append(f"{i}. [{media}] {topic}")
        lines.append(f"   {body}")
        if hit.get("tools_used"):
            lines.append(f"   Tools used: {hit['tools_used']}")
        lines.append("")
    return _tool_ok("recall_memory", "\n".join(lines).strip())


@tool
def save_memory_note(topic: str, content: str):
    """Save a fact, preference, or instruction Sir wants remembered across future turns."""
    topic = (topic or "").strip()
    content = (content or "").strip()
    if not topic or not content:
        return _tool_err("save_memory_note", "topic and content are required.")
    store_note(topic, content, source="agent")
    return _tool_ok("save_memory_note", f"Remembered under topic: {topic}")


@tool
def manage_task_plan(
    action: str,
    description: str = "",
    steps: str = "",
):
    """
    Track multi-step tasks across turns.
    action: create | view | update | complete | clear
    steps: newline-separated steps (for create), or 'step_number:done|pending|failed' (for update)
    """
    action = (action or "").strip().lower()
    plan = _load_task_plan()

    if action == "create":
        desc = (description or "").strip()
        if not desc:
            return "Task Error: description required for create."
        step_list = [s.strip() for s in (steps or "").splitlines() if s.strip()]
        if not step_list:
            step_list = ["Analyze request", "Execute with tools", "Report results to Sir"]
        plan = {
            "description": desc,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "active",
            "steps": [{"text": s, "status": "pending"} for s in step_list],
        }
        _save_task_plan(plan)
        lines = [f"STATUS: CREATED\nTask: {desc}", "Steps:"]
        for i, step in enumerate(plan["steps"], 1):
            lines.append(f"  {i}. [pending] {step['text']}")
        return "\n".join(lines)

    if action == "view":
        return _format_task_plan(plan)

    if action == "update":
        if not plan:
            return "Task Error: no active plan — use action=create first."
        updates = [s.strip() for s in (steps or "").splitlines() if s.strip()]
        for upd in updates:
            if ":" not in upd:
                continue
            idx_s, status = upd.split(":", 1)
            try:
                idx = int(idx_s.strip()) - 1
                step_list = plan.get("steps") or []
                if 0 <= idx < len(step_list):
                    step_list[idx]["status"] = status.strip().lower()
            except ValueError:
                continue
        _save_task_plan(plan)
        return _format_task_plan(plan)

    if action == "complete":
        if not plan:
            return "STATUS: EMPTY\nNo active task to complete."
        plan["status"] = "completed"
        plan["completed_at"] = datetime.datetime.now().isoformat()
        summary = _format_task_plan(plan)
        store_note(
            f"completed task: {plan.get('description', '')[:80]}",
            summary,
            source="task",
        )
        TASKS_FILE.unlink(missing_ok=True)
        return f"STATUS: COMPLETED\nTask archived to memory: {plan.get('description', '')}"

    if action == "clear":
        TASKS_FILE.unlink(missing_ok=True)
        return "STATUS: CLEARED\nActive task plan removed."

    return f"Task Error: unknown action '{action}'. Use create|view|update|complete|clear."


tools = [
    web_search,
    search_and_summarize,
    recall_memory_tool,
    save_memory_note,
    manage_task_plan,
    generate_free_image,
    generate_image,
    extract_pdf_context,
    analyze_uploaded_image,
    execute_python_code,
    read_file,
    write_file,
    create_directory,
    list_directory,
    run_terminal_command,
    open_url_in_browser,
    get_current_time,
]

TOOL_BY_NAME: dict[str, object] = {t.name: t for t in tools}

_CONTROL_GATED = frozenset({
    "execute_python_code",
    "read_file",
    "write_file",
    "create_directory",
    "list_directory",
    "run_terminal_command",
    "open_url_in_browser",
})

_CONFIRMATION_GATED = frozenset({"run_terminal_command"})

_TIME_QUERY_RE = re.compile(
    r"\b(what(?:'s| is) the time|what time|current time|tell(?: me)? the time|"
    r"what(?:'s| is) the date|what day is it)\b",
    re.IGNORECASE,
)
_SEARCH_QUERY_RE = re.compile(
    r"\b(search|look up|lookup|research|find out|latest|news about|web search)\b",
    re.IGNORECASE,
)
# Factual / current-events questions that MUST hit the web, not model memory.
_FACT_SEARCH_RE = re.compile(
    r"\b("
    r"who won|who is the|who was the|who are the|"
    r"what is the (?:current|latest|score)|what happened|"
    r"when did|when was|when is the|"
    r"world cup|fifa|olympics|election|prime minister|president of|"
    r"stock price|weather in|score of|final of|"
    r"which (?:team|country|player)|how many (?:goals|votes|seats)"
    r")\b",
    re.IGNORECASE,
)
_SEARCH_DEEP_RE = re.compile(
    r"\b(research|summarize|summary|deep dive|find out about|comprehensive)\b",
    re.IGNORECASE,
)
_IMAGE_GEN_RE = re.compile(
    r"\b("
    r"generate(?:\s+an?)?\s+image|create(?:\s+an?)?\s+image|draw(?:\s+an?)?\s+(?:image|picture)|"
    r"make(?:\s+an?)?\s+(?:image|picture)|image\s+of|picture\s+of|"
    r"generate_free_image|generate_image"
    r")\b",
    re.IGNORECASE,
)
# Strip chat-command wrappers so DDG gets a real topic, not "search the web for …".
_SEARCH_WRAPPER_RE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+)?"
    r"(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s*up|lookup|research|find\s+out(?:\s+about)?|"
    r"web\s+search(?:\s+for)?|google)\s+",
    re.IGNORECASE,
)
_SEARCH_TRAILING_RE = re.compile(
    r"\s+(?:and\s+summarize(?:\s+it)?|please|for me|thanks|thank you)\.?$",
    re.IGNORECASE,
)


def _clean_search_query(text: str) -> str:
    """Extract the topical query from a natural-language search request."""
    q = (text or "").strip().strip("\"'")
    if not q:
        return q
    cleaned = _SEARCH_WRAPPER_RE.sub("", q).strip()
    cleaned = _SEARCH_TRAILING_RE.sub("", cleaned).strip(" .,?!")

    # Rewrite factoid phrasings so DDG does not latch onto "won" (currency) etc.
    who_won = re.match(
        r"^(?:who\s+won)\s+(?:the\s+)?(.+?)\??$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if who_won:
        topic = who_won.group(1).strip()
        if topic:
            cleaned = f"{topic} winner"
    else:
        who_is = re.match(
            r"^(?:who\s+is|who\s+was)\s+(?:the\s+)?(.+?)\??$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if who_is:
            topic = who_is.group(1).strip()
            if topic:
                cleaned = topic

    # Prefer keyword form for major tournaments — "final" ranks better on Wikipedia
    # than "winner" (which often returns a generic tournament overview page).
    if re.search(r"\bfifa\b.*\bworld\s*cup\b|\bworld\s*cup\b.*\bfifa\b", cleaned, re.I):
        year = re.search(r"\b(19|20)\d{2}\b", cleaned)
        y = year.group(0) if year else ""
        cleaned = f"{y} FIFA World Cup final".strip()

    # Avoid empty / ultra-short leftovers after stripping wrappers
    if len(cleaned) >= 2:
        return cleaned
    return q
_RECALL_QUERY_RE = re.compile(
    r"\b(recall|remember|what did we|previous|last time|earlier|before)\b",
    re.IGNORECASE,
)
_TASK_VIEW_RE = re.compile(
    r"\b(task plan|current plan|plan status|show (my )?plan|view plan)\b",
    re.IGNORECASE,
)
_LIST_DIR_RE = re.compile(
    r"\b("
    r"list(?:\s+(?:the|my|all))?\s+(?:files?|folders?|directories|dir|contents?|apps?|applications)|"
    r"(?:show|display|see)\s+(?:(?:the|my|all)\s+)?(?:files?|folders?|directories|dir|contents?|apps?|applications)|"
    r"ls\b|"
    r"what(?:'s| is)\s+(?:on|in)\s+(?:my\s+)?(?:desktop|documents|downloads|folder|directory|apps?|applications|pc|mac|computer)|"
    r"files?\s+(?:on|in)\s+(?:my\s+)?(?:desktop|documents|downloads|folder|directory|home)|"
    r"list\s+(?:my\s+)?(?:desktop|documents|downloads|home|folder|directory|apps?|applications)|"
    r"(?:installed\s+)?(?:apps?|applications)\s+(?:on|in)\s+(?:my\s+)?(?:pc|mac|computer|system)|"
    r"(?:what|which)\s+apps?\b|"
    r"apps?\s+(?:installed|on\s+(?:my\s+)?(?:pc|mac|computer))"
    r")\b",
    re.IGNORECASE,
)
_LIST_APPS_RE = re.compile(
    r"\b("
    r"(?:list|show|display|see|get|find)\b.+\b(?:apps?|applications)\b|"
    r"\b(?:apps?|applications)\b.+\b(?:list|show|installed|on\s+my)\b|"
    r"installed\s+(?:apps?|applications)|"
    r"what(?:'s| is| are)\s+(?:the\s+)?(?:apps?|applications)\b"
    r")\b",
    re.IGNORECASE,
)
_READ_FILE_RE = re.compile(
    r"\b(?:"
    r"(?:read|open|show|cat|display|view|print)\s+(?:the\s+)?(?:contents?\s+of\s+)?(?:the\s+)?file\s+"
    r"|read\s+(?:the\s+)?"
    r"|what(?:'s| is)\s+in\s+(?:the\s+)?(?:file\s+)?"
    r"|show\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?"
    r")"
    r"['\"]?([~/]?[\w./\\-]+\.[\w]+)['\"]?",
    re.IGNORECASE,
)
# Explicit write WITH content only — never force empty writes (would wipe files).
_WRITE_FILE_CONTENT_RE = re.compile(
    r"\b(?:write|create|save|put)\s+(?:to\s+)?(?:a\s+|the\s+)?file\s+"
    r"['\"]?([~/]?[\w./\\-]+\.[\w]+)['\"]?\s+"
    r"(?:with\s+(?:the\s+)?(?:content|text|data)\s*[:\-]?\s*|\s*[:\-]\s*)(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_WRITE_FILE_CONTENT_RE2 = re.compile(
    r"\b(?:write|save)\s+(?:this\s+)?(?:to\s+)?['\"]?([~/]?[\w./\\-]+\.[\w]+)['\"]?\s*"
    r"(?:"
    r"\s*[:\-]\s*"
    r"|\s+with\s+(?:the\s+)?(?:content|text|data)\s*[:\-]?\s*"
    r")(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SHELL_CMD_RE = re.compile(
    r"\b(?:"
    r"(?:run|execute)\s+(?:this\s+)?(?:shell\s+|terminal\s+)?command\s*[:\-]?\s*`([^`]+)`"
    r"|(?:run|execute)\s+`([^`]+)`"
    r"|(?:run|execute)\s+(?:in\s+)?(?:the\s+)?(?:shell|terminal)\s*[:\-]?\s*(.+)$"
    r"|shell\s*:\s*(.+)$"
    r")",
    re.IGNORECASE,
)
# create/make folder named X inside/on/in Y
_MKDIR_NAMED_IN_RE = re.compile(
    r"\b(?:make|create|add)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:folder|directory|dir)\s+"
    r"(?:named|called)\s+['\"]?([^'\"\n,]+?)['\"]?\s+"
    r"(?:inside|in|under|within|on|at|onto|to)\s+(?:my\s+|the\s+|sir'?s\s+)?"
    r"['\"]?([^'\"\n]+?)['\"]?\s*$",
    re.IGNORECASE,
)
# create/make folder X inside/on/in Y  (no named/called)
_MKDIR_BARE_IN_RE = re.compile(
    r"\b(?:make|create|add)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:folder|directory|dir)\s+"
    r"['\"]?([^'\"\n,]+?)['\"]?\s+"
    r"(?:inside|in|under|within|on|at|onto|to)\s+(?:my\s+|the\s+|sir'?s\s+)?"
    r"['\"]?([^'\"\n]+?)['\"]?\s*$",
    re.IGNORECASE,
)
# mkdir [-p] path  OR  create folder at path  OR  create folder ~/Desktop/Foo
_MKDIR_PATH_RE = re.compile(
    r"\b(?:"
    r"mkdir\s+(?:-p\s+)?['\"]?([~/][^\s'\"]+|/(?:Users|home)/[^\s'\"]+|Desktop/[^\s'\"]+|Documents/[^\s'\"]+|Downloads/[^\s'\"]+)['\"]?"
    r"|(?:make|create|add)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:folder|directory|dir)\s+(?:at\s+)?"
    r"['\"]?([~/][^\s'\"]+|/(?:Users|home)/[^\s'\"]+|Desktop/[^\s'\"]+|Documents/[^\s'\"]+|Downloads/[^\s'\"]+)['\"]?\s*$"
    r")",
    re.IGNORECASE,
)
# create a folder named X  (default parent = Desktop)
_MKDIR_NAMED_ONLY_RE = re.compile(
    r"\b(?:make|create|add)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:folder|directory|dir)\s+"
    r"(?:named|called)\s+['\"]?([^'\"\n]+?)['\"]?\s*$",
    re.IGNORECASE,
)
# create a file named X on/in Y with content: ...
_CREATE_FILE_CONTENT_RE = re.compile(
    r"\b(?:create|write|make|save)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:text\s+)?file\s+"
    r"(?:(?:named|called)\s+)?['\"]?([^'\"\n]+?\.\w+|[^'\"\n]+?)['\"]?\s+"
    r"(?:inside|in|under|within|on|at|to)\s+(?:my\s+|the\s+|sir'?s\s+)?"
    r"(?:that\s+folder|the\s+folder|it|['\"]?([^'\"\n]+?)['\"]?)\s*"
    r"(?:with\s+(?:the\s+)?(?:content|text|data)\s*[:\-]?\s*|\s*[:\-]\s*)(.+)$",
    re.IGNORECASE | re.DOTALL,
)
# create a file inside that folder with content: ...
_CREATE_FILE_SIMPLE_RE = re.compile(
    r"\b(?:create|write|make)\s+(?:a\s+)?(?:text\s+)?file\s+"
    r"(?:inside|in)\s+(?:that\s+folder|the\s+folder|it)\s+"
    r"(?:with\s+(?:the\s+)?(?:content|text|data)\s*[:\-]?\s*)(.+)$",
    re.IGNORECASE | re.DOTALL,
)
# create empty/new file path (no content)
_CREATE_EMPTY_FILE_RE = re.compile(
    r"\b(?:create|make)\s+(?:a\s+|an\s+)?(?:new\s+|empty\s+)?(?:text\s+)?file\s+"
    r"(?:(?:named|called)\s+)?['\"]?([~/]?[\w./\\-]+\.[\w]+)['\"]?"
    r"(?:\s+(?:inside|in|on|at|under)\s+(?:my\s+|the\s+)?['\"]?([^'\"\n]+?)['\"]?)?\s*$",
    re.IGNORECASE,
)
_PATH_HINTS = (
    (re.compile(r"\b(?:apps?|applications)\b", re.I), "/Applications"),
    (re.compile(r"\bdesktop\b", re.I), "~/Desktop"),
    (re.compile(r"\bdocuments\b", re.I), "~/Documents"),
    (re.compile(r"\bdownloads\b", re.I), "~/Downloads"),
    (re.compile(r"\bhome\b", re.I), "~"),
)

# Models without native tool-calling often emit this as plain text instead of tool_calls.
_PSEUDO_TOOL_CALL_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:tool[_ ]?call|function[_ ]?call|call(?:ing)?\s+tool)\s*[:\-]?\s*"
    r"(?P<name>[a-zA-Z_][\w]*)"
    r"(?:\s*\((?P<paren>[^)]*)\))?"
    r"(?:\s+(?P<rest>[^\n]+))?",
    re.IGNORECASE,
)
_KNOWN_TOOL_NAMES = frozenset({
    "web_search", "search_and_summarize", "recall_memory", "save_memory_note",
    "manage_task_plan", "generate_free_image", "generate_image", "extract_pdf_context",
    "analyze_uploaded_image",
    "execute_python_code",
    "read_file", "write_file", "create_directory", "list_directory",
    "run_terminal_command", "open_url_in_browser", "get_current_time",
})


def run_tool(
    name: str,
    args: Optional[dict] = None,
    *,
    control_allowed: bool = False,
    thread_id: str = "web_user_001",
    user_confirmed: bool = False,
) -> str:
    """Execute a tool programmatically (bypasses LLM tool-calling)."""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return _tool_err(name or "unknown", f"Unknown tool: {name}")

    payload = dict(args or {})
    if name in _CONTROL_GATED:
        # Always pass the authoritative file flag (or explicit override)
        payload["control_allowed"] = bool(control_allowed) or permissions.is_control_allowed()
    if name in _CONFIRMATION_GATED:
        payload["thread_id"] = thread_id
        payload["user_confirmed"] = user_confirmed
    try:
        result = tool.invoke(payload)
        return str(result)
    except Exception as exc:
        return _tool_err(name, str(exc))


def _infer_list_path(user_text: str) -> str:
    """Best-effort Mac path for list_directory from natural language."""
    text = user_text or ""
    for pattern, path in _PATH_HINTS:
        if pattern.search(text):
            return path
    quoted = re.search(r'["\']([^"\']+)["\']', text)
    if quoted:
        return quoted.group(1).strip()
    bare = re.search(r"(~/[^\s,]+|/Users/[^\s,]+|/Applications\b|/System/[^\s,]+)", text)
    if bare:
        return bare.group(1).strip().rstrip(".,;:")
    return "~"


def _args_from_pseudo_rest(name: str, paren: str, rest: str) -> dict:
    """Map free-form pseudo tool-call text into tool args."""
    blob = " ".join(p for p in (paren or "", rest or "") if p).strip()
    if name in ("list_directory", "create_directory", "read_file"):
        path = "~" if name == "list_directory" else ""
        pm = re.search(r"path\s*[=:]\s*['\"]?([^\s'\",}]+)", blob, re.I)
        if pm:
            path = _clean_path_arg(pm.group(1))
        elif blob:
            token = _clean_path_arg(blob.split()[0].strip("'\",;"))
            if token.startswith(("~", "/", "./")) or token.lower() in {
                "desktop", "documents", "downloads", "applications", "home",
            }:
                path = {
                    "desktop": "~/Desktop",
                    "documents": "~/Documents",
                    "downloads": "~/Downloads",
                    "applications": "/Applications",
                    "home": "~",
                }.get(token.lower(), token)
            else:
                path = token
        if name == "read_file" and not path and blob:
            file_tok = re.search(r"([~/]?[\w./\\-]+\.[\w]+)", blob)
            path = file_tok.group(1) if file_tok else blob.split()[0]
            path = _clean_path_arg(path)
        return {"path": path or "~"}
    if name == "write_file":
        pm = re.search(r"path\s*[=:]\s*['\"]?([^\s'\",}]+)", blob, re.I)
        cm = re.search(r"content\s*[=:]\s*['\"]?(.+)", blob, re.I | re.S)
        path = pm.group(1) if pm else ""
        if not path and blob:
            file_tok = re.search(r"([~/]?[\w./\\-]+\.[\w]+)", blob)
            path = file_tok.group(1) if file_tok else ""
        return {
            "path": _clean_path_arg(path),
            "content": (cm.group(1).strip().strip("'\"") if cm else ""),
        }
    if name == "run_terminal_command":
        pm = re.search(r"command\s*[=:]\s*['\"](.+?)['\"]\s*$", blob, re.I)
        if pm:
            return {"command": pm.group(1)}
        return {"command": blob.strip("'\"") if blob else ""}
    if name == "open_url_in_browser":
        pm = re.search(r"url\s*[=:]\s*['\"]?(\S+)", blob, re.I)
        return {"url": pm.group(1) if pm else blob.split()[0] if blob else ""}
    if name in ("web_search", "search_and_summarize", "recall_memory"):
        pm = re.search(r"query\s*[=:]\s*['\"]?(.+)", blob, re.I)
        return {"query": (pm.group(1).strip().strip("'\"") if pm else blob) or ""}
    if name == "execute_python_code":
        pm = re.search(r"code\s*[=:]\s*['\"]?(.+)", blob, re.I | re.S)
        return {"code": (pm.group(1) if pm else blob) or ""}
    return {}


def looks_like_pseudo_tool_call(text: str) -> bool:
    """True when the model wrote a tool invocation as plain text (not structured tool_calls)."""
    t = (text or "").strip()
    if not t:
        return False
    if _PSEUDO_TOOL_CALL_RE.search(t):
        return True
    # Bare "list_directory /Applications" style lines
    lower = t.lower()
    for name in _KNOWN_TOOL_NAMES:
        if re.search(rf"(?:^|\n)\s*{re.escape(name)}\s*[\(/]", t, re.I):
            return True
        if lower.startswith(f"{name} ") or lower == name:
            return True
    return False


def parse_pseudo_tool_calls(text: str) -> list[tuple[str, dict]]:
    """
    Parse hallucinated plain-text tool invocations into real (name, args) pairs.
    Example: 'tool_call: list_directory /Applications'
    """
    text = (text or "").strip()
    if not text:
        return []

    planned: list[tuple[str, dict]] = []
    for m in _PSEUDO_TOOL_CALL_RE.finditer(text):
        name = (m.group("name") or "").strip()
        if name not in _KNOWN_TOOL_NAMES and name not in TOOL_BY_NAME:
            continue
        args = _args_from_pseudo_rest(name, m.group("paren") or "", m.group("rest") or "")
        planned.append((name, args))

    if planned:
        return _dedupe_planned(planned)

    # Fallback: single-line "list_directory /path" without tool_call: prefix
    for name in _KNOWN_TOOL_NAMES:
        m = re.match(rf"^\s*{re.escape(name)}\s*(?:\(([^)]*)\))?\s*(.*)$", text, re.I | re.S)
        if m:
            args = _args_from_pseudo_rest(name, m.group(1) or "", m.group(2) or "")
            return [(name, args)]
    return []


def _dedupe_planned(planned: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, args in planned:
        key = f"{name}:{sorted((args or {}).items())}"
        if key not in seen:
            seen.add(key)
            unique.append((name, args))
    return unique


def _normalize_location_phrase(name: str) -> str:
    """Strip my/the/sir's and map well-known locations to real paths."""
    raw = (name or "").strip().strip("'\"")
    raw = re.sub(r"^(?:my|the|sir'?s|our)\s+", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s+$", "", raw)
    # Drop trailing filler words
    raw = re.sub(r"\s+(?:please|now|thanks|thank you)\s*$", "", raw, flags=re.IGNORECASE)
    return sanitize_path_string(raw) if raw else raw


def _resolve_parent_folder(parent_name: str) -> str:
    """Resolve a parent folder name to a real absolute path string (no deep rglob)."""
    parent_name = (parent_name or "").strip().strip("'\"")
    lower = parent_name.lower().strip()
    lower = re.sub(r"^(?:my|the|sir'?s|our)\s+", "", lower).strip()

    if lower in ("that folder", "the folder", "it", "there", "same folder", "same place"):
        last = last_remembered_path()
        if last is not None:
            target = last if last.is_dir() else last.parent
            try:
                return str(target.resolve(strict=False))
            except TypeError:
                return str(target.resolve())
        return str(real_home() / "Desktop")

    # Well-known locations: Desktop, Documents, ...
    cleaned = _normalize_location_phrase(parent_name)
    if cleaned:
        try:
            primary = resolve_mac_path(cleaned)
            # If it's an existing dir, use it; if well-known (Desktop etc.), use even if empty
            if primary.exists() and primary.is_dir():
                return str(primary)
            # Bare Desktop/Documents path even if somehow missing
            for key, folder in (
                ("desktop", "Desktop"),
                ("documents", "Documents"),
                ("downloads", "Downloads"),
            ):
                if lower == key or lower == folder.lower():
                    return str(real_home() / folder)
            # Parent path for a new subfolder under a known root
            if any(
                str(primary).startswith(str(real_home() / f))
                for f in ("Desktop", "Documents", "Downloads")
            ):
                return str(primary)
        except Exception:
            pass

    found = find_named_path(parent_name if not cleaned else Path(cleaned).name)
    if found and found.is_dir():
        return str(found)

    # Default: under Desktop (real home — never invent usernames)
    name_only = Path(cleaned or parent_name).name
    try:
        return str((real_home() / "Desktop" / name_only).resolve(strict=False))
    except TypeError:
        return str((real_home() / "Desktop" / name_only).resolve())


def _infer_mkdir(text: str) -> tuple[str, dict] | None:
    """Infer create_directory with a real absolute path (not shell mkdir)."""
    text = (text or "").strip()
    if not text:
        return None

    m = _MKDIR_NAMED_IN_RE.search(text) or _MKDIR_BARE_IN_RE.search(text)
    if m:
        folder_name = m.group(1).strip().strip("'\"")
        parent_name = m.group(2).strip().strip("'\"")
        # Guard: folder name should not be a location word alone
        if folder_name.lower() in ("folder", "directory", "dir", "new"):
            return None
        parent = _resolve_parent_folder(parent_name)
        target = str(Path(parent) / folder_name)
        return ("create_directory", {"path": target})

    m_path = _MKDIR_PATH_RE.search(text)
    if m_path:
        raw_path = next((g for g in m_path.groups() if g), "").strip().strip("'\"")
        if raw_path:
            try:
                target = str(resolve_mac_path(raw_path))
            except Exception:
                target = sanitize_path_string(raw_path)
            return ("create_directory", {"path": target})

    m_named = _MKDIR_NAMED_ONLY_RE.search(text)
    if m_named:
        folder_name = m_named.group(1).strip().strip("'\"")
        if folder_name.lower() in ("folder", "directory", "dir"):
            return None
        # If name itself is a path, use it; else put on Desktop
        if folder_name.startswith(("~", "/", "Desktop/", "Documents/", "Downloads/")):
            try:
                target = str(resolve_mac_path(folder_name))
            except Exception:
                target = str(real_home() / "Desktop" / Path(folder_name).name)
        else:
            target = str(real_home() / "Desktop" / folder_name)
        return ("create_directory", {"path": target})

    return None


def _ensure_text_filename(name: str) -> str:
    name = (name or "note.txt").strip().strip("'\"")
    if not name:
        return "note.txt"
    if "." not in Path(name).name:
        return f"{name}.txt"
    return name


def _infer_create_file(text: str) -> tuple[str, dict] | None:
    """Infer write_file with content (or empty create) into a named / last folder."""
    m = _CREATE_FILE_CONTENT_RE.search(text)
    if m:
        filename = _ensure_text_filename(m.group(1) or "note.txt")
        parent_name = (m.group(2) or "that folder").strip().strip("'\"")
        content = (m.group(3) or "").strip()
        # If parent_name looks empty because group was optional path
        if not parent_name or parent_name.lower() in ("with", "content", "text"):
            parent_name = "that folder"
        parent = _resolve_parent_folder(parent_name)
        path = str(Path(parent) / Path(filename).name)
        return ("write_file", {"path": path, "content": content})

    m2 = _CREATE_FILE_SIMPLE_RE.search(text)
    if m2:
        content = (m2.group(1) or "").strip()
        parent = _resolve_parent_folder("that folder")
        path = str(Path(parent) / "note.txt")
        return ("write_file", {"path": path, "content": content})

    # write/save file X with content / write file X: content
    for pattern in (_WRITE_FILE_CONTENT_RE, _WRITE_FILE_CONTENT_RE2):
        wm = pattern.search(text)
        if wm:
            path = _clean_path_arg(wm.group(1))
            content = (wm.group(2) or "").strip()
            if path and content:
                return ("write_file", {"path": path, "content": content})

    # create empty/new file (explicit create only — never wipe via "edit")
    m3 = _CREATE_EMPTY_FILE_RE.search(text)
    if m3:
        filename = _ensure_text_filename(m3.group(1))
        parent_name = (m3.group(2) or "").strip() if m3.lastindex and m3.lastindex >= 2 else ""
        if parent_name:
            parent = _resolve_parent_folder(parent_name)
            path = str(Path(parent) / Path(filename).name)
        else:
            # Absolute-ish path in filename itself
            try:
                path = str(resolve_mac_path(filename)) if filename.startswith(("~", "/")) or "/" in filename else str(
                    real_home() / "Desktop" / Path(filename).name
                )
            except Exception:
                path = str(real_home() / "Desktop" / Path(filename).name)
        return ("write_file", {"path": path, "content": ""})

    return None


def infer_forced_tools(user_text: str) -> list[tuple[str, dict]]:
    """
    Infer which tools should run deterministically when the LLM fails to call them.
    Returns list of (tool_name, args) in execution order.
    """
    text = (user_text or "").strip()
    if not text:
        return []

    inferred: list[tuple[str, dict]] = []
    if _TIME_QUERY_RE.search(text):
        inferred.append(("get_current_time", {}))

    # Create folder (before generic list/read to avoid false matches)
    mkdir = _infer_mkdir(text)
    if mkdir:
        inferred.append(mkdir)

    create_file = _infer_create_file(text)
    if create_file:
        inferred.append(create_file)

    # Apps on Mac → /Applications (and user Applications if present)
    if _LIST_APPS_RE.search(text) or (
        _LIST_DIR_RE.search(text) and re.search(r"\b(?:apps?|applications)\b", text, re.I)
    ):
        inferred.append(("list_directory", {"path": "/Applications", "max_entries": 400}))
        user_apps = real_home() / "Applications"
        try:
            if user_apps.is_dir() and any(user_apps.iterdir()):
                inferred.append(("list_directory", {"path": str(user_apps), "max_entries": 200}))
        except OSError:
            pass
    elif _LIST_DIR_RE.search(text) and not mkdir and not create_file:
        inferred.append(("list_directory", {"path": _infer_list_path(text)}))

    read_m = _READ_FILE_RE.search(text)
    if read_m and not create_file:
        inferred.append(("read_file", {"path": _clean_path_arg(read_m.group(1))}))
    elif not create_file:
        loose = re.search(
            r"\b(?:read|open|cat|view)\s+['\"]?([~/]?[\w./\\-]+\.[\w]+)['\"]?",
            text,
            re.I,
        )
        if loose:
            inferred.append(("read_file", {"path": _clean_path_arg(loose.group(1))}))

    # NOTE: never force write_file with empty content from "edit file" — that wipes files.
    # Contentful writes are handled by _infer_create_file above.

    shell_m = _SHELL_CMD_RE.search(text)
    if shell_m and not mkdir:
        cmd = next((g for g in shell_m.groups() if g), "").strip()
        if cmd:
            inferred.append(("run_terminal_command", {"command": rewrite_paths_in_command(cmd)}))

    if _SEARCH_QUERY_RE.search(text) or _FACT_SEARCH_RE.search(text):
        query = _clean_search_query(text) or text
        if _SEARCH_DEEP_RE.search(text):
            inferred.append(("search_and_summarize", {"query": query, "focus": ""}))
        else:
            inferred.append(("web_search", {"query": query}))
    if _IMAGE_GEN_RE.search(text):
        # Strip the command wrapper so Pollinations gets a clean subject prompt.
        img_prompt = re.sub(
            r"^(?:please\s+|can you\s+|could you\s+)?"
            r"(?:generate|create|draw|make)(?:\s+an?)?\s+(?:image|picture)(?:\s+of)?\s*",
            "",
            text.strip(),
            flags=re.I,
        ).strip(" .,?!") or text.strip()
        inferred.append(("generate_free_image", {"prompt": img_prompt}))
    # Uploaded / mentioned PDF → force extract before answering
    # Prefer the canonical upload URL path over bare filenames in the same line.
    pdf_match = (
        re.search(r"(/agent_data/uploads/[A-Za-z0-9._/-]+\.pdf)", text, re.I)
        or re.search(r"(agent_data/uploads/[A-Za-z0-9._/-]+\.pdf)", text, re.I)
        or re.search(r"\b([A-Za-z0-9._-]+\.pdf)\b", text, re.I)
    )
    if pdf_match and (
        re.search(r"\b(pdf|document|upload|file|summarize|read|extract|what does)\b", text, re.I)
        or "/agent_data/uploads/" in text
        or "ATTACHED UPLOADS" in text
    ):
        path = pdf_match.group(1)
        if path.startswith("agent_data/"):
            path = "/" + path
        elif not path.startswith("/") and ("uploads" in text or "ATTACHED" in text):
            path = f"/agent_data/uploads/{Path(path).name}"
        inferred.append(("extract_pdf_context", {"file_path": path}))

    # Uploaded / mentioned image → vision tool (never extract_pdf_context)
    img_match = (
        re.search(
            r"(/agent_data/uploads/[A-Za-z0-9._/-]+\.(?:png|jpe?g|gif|webp|bmp))",
            text,
            re.I,
        )
        or re.search(
            r"(agent_data/uploads/[A-Za-z0-9._/-]+\.(?:png|jpe?g|gif|webp|bmp))",
            text,
            re.I,
        )
        or re.search(r"\b([A-Za-z0-9._-]+\.(?:png|jpe?g|gif|webp|bmp))\b", text, re.I)
    )
    if img_match and (
        re.search(
            r"\b(image|picture|photo|png|jpe?g|upload|file|describe|analyze|what(?:'s| is)|see|look)\b",
            text,
            re.I,
        )
        or "/agent_data/uploads/" in text
        or "ATTACHED UPLOADS" in text
        or "(image)" in text.lower()
    ):
        path = img_match.group(1)
        if path.startswith("agent_data/"):
            path = "/" + path
        elif not path.startswith("/") and ("uploads" in text or "ATTACHED" in text):
            path = f"/agent_data/uploads/{Path(path).name}"
        # Strip attachment boilerplate for a cleaner vision prompt
        vision_prompt = re.sub(
            r"\n*\[ATTACHED UPLOADS[^\]]*\][\s\S]*$",
            "",
            text,
            flags=re.I,
        ).strip()
        if not vision_prompt or vision_prompt.lower() in {"analyze", "describe", "what is this"}:
            vision_prompt = (
                "Describe this image in detail. Note any text, objects, people, and context."
            )
        inferred.append(
            ("analyze_uploaded_image", {"file_path": path, "prompt": vision_prompt})
        )

    if _RECALL_QUERY_RE.search(text):
        inferred.append(("recall_memory", {"query": _clean_search_query(text) or text}))
    if _TASK_VIEW_RE.search(text):
        inferred.append(("manage_task_plan", {"action": "view"}))

    return _dedupe_planned(inferred)