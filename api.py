"""
DEPRECATED for new deployments.

The full Three.js + /chat server now lives in main.py.
Use:
    uvicorn main:app --host 0.0.0.0 --port $PORT

or locally:
    python main.py web

This file now just re-exports the ready app for compatibility.
"""
from main import app as server

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    def launch():
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    threading.Thread(target=launch, daemon=True).start()
    uvicorn.run(server, host="0.0.0.0", port=8000)