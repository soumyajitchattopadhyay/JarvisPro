from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from fastapi.responses import HTMLResponse
import uvicorn
import json
import os
import webbrowser
import threading
import time
import traceback

# Ensure main is imported carefully
try:
    from main import app as langgraph_app 
except ImportError as e:
    print(f"CRITICAL: Could not import 'app' from main.py. Check for syntax errors in main.py. \nError: {e}")
    exit(1)

from speak import speak 

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
        # Uses absolute path to ensure the file is found
        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, "index.html"), "r") as f: 
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found in script directory.</h1>", status_code=404)

@server.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_text = data.get("text", "").lower()
        thread_id = "web_user_001"

        if "allow control" in user_text: set_permission(True)
        elif "stop control" in user_text: set_permission(False)

        current_allowed = get_permission()
        
        # NOTE: Using a threadpool for the sync 'invoke' call prevents blocking the API
        def run_graph():
            return langgraph_app.invoke(
                {"messages": [HumanMessage(content=user_text)], "control_allowed": current_allowed}, 
                config={"configurable": {"thread_id": thread_id}}
            )

        # In a real production app, use anyio.to_thread.run_sync
        # For this setup, we call it directly but we've wrapped the error handling
        result = run_graph()
        
        final_reply = result["messages"][-1].content
        
        # Call speak in a background thread so the API returns immediately
        threading.Thread(target=speak, args=(final_reply, False), daemon=True).start()
        
        return {"reply": final_reply, "control_status": current_allowed}
    except Exception as e:
        # Prints the full error to your console so you can see what actually failed
        print("--- API ERROR TRACEBACK ---")
        traceback.print_exc()
        return {"reply": f"Logic Error: {str(e)}"}

def launch_tesseract():
    time.sleep(2) 
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. The AI will not respond.")
    
    threading.Thread(target=launch_tesseract, daemon=True).start()
    uvicorn.run(server, host="0.0.0.0", port=8000)