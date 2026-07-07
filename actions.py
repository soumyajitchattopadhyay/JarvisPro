import os
import webbrowser
import datetime
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from duckduckgo_search import DDGS

from local_search_memory import format_recall_for_context, store_search

repl = PythonREPL()


def _search_duckduckgo(query: str, max_results: int = 3) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results, backend="lite")
        return "\n".join([f"Source: {r['title']} - {r['body']}" for r in results])


def _search_tavily(query: str, max_results: int = 3) -> str:
    """Optional Tavily API — set TAVILY_API_KEY in .env for richer results."""
    import httpx

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    with httpx.Client(timeout=20.0) as client:
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
        lines: list[str] = []
        if data.get("answer"):
            lines.append(f"Summary: {data['answer']}")
        for hit in data.get("results") or []:
            title = hit.get("title") or "Untitled"
            body = hit.get("content") or hit.get("snippet") or ""
            url = hit.get("url") or ""
            lines.append(f"Source: {title} ({url}) - {body}")
        return "\n".join(lines) if lines else "No Tavily results."


@tool
def web_search(query: str):
    """Search the web for current information, facts, or news. Recalls prior local searches when relevant."""
    query = (query or "").strip()
    if not query:
        return "Search Error: empty query."

    prior = format_recall_for_context(query)
    provider = "duckduckgo"
    try:
        if os.getenv("TAVILY_API_KEY", "").strip():
            provider = "tavily"
            fresh = _search_tavily(query)
        else:
            fresh = _search_duckduckgo(query)
    except Exception as e:
        if provider == "tavily":
            try:
                provider = "duckduckgo"
                fresh = _search_duckduckgo(query)
            except Exception as fallback_exc:
                return f"Search Error: {fallback_exc}"
        else:
            return f"Search Error: {str(e)}"

    store_search(query, fresh, source=provider)
    header = f"[Web search via {provider} — stored in local RAM memory]\n"
    return header + (prior + fresh if prior else fresh)


@tool
def execute_python_code(code: str, control_allowed: bool = False):
    """Executes Python code for calculations, logic, data work or automation. Requires control_allowed=True."""
    if not control_allowed:
        return "PERMISSION_DENIED: Elevated access is RESTRICTED. The user must say 'Allow Control' to proceed."
    try:
        result = repl.run(code)
        return f"Execution complete:\n{result}"
    except Exception as e:
        return f"Execution failed: {str(e)}"


@tool
def open_url_in_browser(url: str, control_allowed: bool = False):
    """Opens a URL. Requires 'control_allowed=True'."""
    if not control_allowed:
        return "ERROR: Browser access is RESTRICTED."
    webbrowser.open(url)
    return f"Browser opened to {url}."


@tool
def get_current_time() -> str:
    """Returns the current local date and time. Use this when the user asks about time, date, or 'what day is it'."""
    now = datetime.datetime.now()
    return now.strftime("It is %A, %B %d %Y, %I:%M %p.")


tools = [web_search, execute_python_code, open_url_in_browser, get_current_time]