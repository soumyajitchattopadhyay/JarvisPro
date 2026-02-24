from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig # Added for type safety
from fastapi.responses import HTMLResponse
import uvicorn
import json
import os
import webbrowser
import threading
import time
import traceback

# 1. IMPORT THE CORRECTED NAME
try:
    from main import compiled_graph as langgraph_graph 
except ImportError as e:
    print(f"CRITICAL: Could not import 'compiled_graph' from main.py. \nError: {e}")
    exit(1)

try:
    from speak import speak 
except ImportError:
    speak = None

server = FastAPI(title="TESrACT Core")
server.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SETTINGS_FILE = "user_settings.json"

def get_permission(thread_id="web_user_001"):
    if not os.path.exists(SETTINGS_FILE): return False
    try:
        with open(SETTINGS_FILE, "r") as f: 
            return json.load(f).get(thread_id, False)
    except: return False

def set_permission(status: bool, thread_id="web_user_001"):
    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: data = json.load(f)
        except: pass
    data[thread_id] = status
    with open(SETTINGS_FILE, "w") as f: json.dump(data, f)

@server.get("/", response_class=HTMLResponse)
async def serve_interface():
    try:
        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, "index.html"), "r") as f: 
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)

@server.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_text = data.get("text", "")
        thread_id = "web_user_001"

        if "allow control" in user_text.lower(): set_permission(True)
        elif "stop control" in user_text.lower(): set_permission(False)

        current_allowed = get_permission()
        
        # FIX: Define the config strictly as a RunnableConfig to satisfy LangGraph
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        
        # 2. CALL THE GRAPH
        result = langgraph_graph.invoke(
            {"messages": [HumanMessage(content=user_text)], "control_allowed": current_allowed}, 
            config=config
        )
        
        final_reply = result["messages"][-1].content
        
        if speak:
            threading.Thread(target=speak, args=(final_reply, False), daemon=True).start()
        
        return {"reply": final_reply, "control_status": current_allowed}
    except Exception as e:
        print("--- API ERROR TRACEBACK ---")
        traceback.print_exc()
        return {"reply": f"Logic Error: {str(e)}"}

def launch_tesseract():
    time.sleep(2) 
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=launch_tesseract, daemon=True).start()
    uvicorn.run(server, host="0.0.0.0", port=8000)
