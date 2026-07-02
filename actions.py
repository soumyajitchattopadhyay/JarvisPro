import webbrowser
import datetime
import os
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from duckduckgo_search import DDGS

repl = PythonREPL()

@tool
def web_search(query: str):
    """Search the web for current information, facts, or news."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3, backend="lite")
            return "\n".join([f"Source: {r['title']} - {r['body']}" for r in results])
    except Exception as e:
        return f"Search Error: {str(e)}"

@tool
def execute_python_code(code: str, control_allowed: bool = False):
    """Executes Python code for calculations, logic, data work or automation. Requires control_allowed=True."""
    if not control_allowed:
        return "PERMISSION_DENIED: LATTICE control is RESTRICTED. The user must grant access by saying 'Allow Control'."
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