import webbrowser
import datetime
import os
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from duckduckgo_search import DDGS

repl = PythonREPL()

@tool
def web_search(query: str):
    """Search for information without opening a browser window."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3, backend="lite")
            return "\n".join([f"Source: {r['title']} - {r['body']}" for r in results])
    except Exception as e:
        return f"Search Error: {str(e)}"

@tool
def execute_python_code(code: str, control_allowed: bool = False):
    """Executes Python code. Requires 'control_allowed=True'."""
    if not control_allowed:
        return "PERMISSION_DENIED: System control is RESTRICTED. Please ask the user for 'Allow Control'."
    try:
        result = repl.run(code)
        return f"Execution Result:\n{result}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

@tool
def open_url_in_browser(url: str, control_allowed: bool = False):
    """Opens a URL. Requires 'control_allowed=True'."""
    if not control_allowed:
        return "ERROR: Browser access is RESTRICTED."
    webbrowser.open(url)
    return f"Browser opened to {url}."

tools = [execute_python_code, web_search, open_url_in_browser]