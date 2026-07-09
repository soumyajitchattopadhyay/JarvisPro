# TESrACT

A proactive, agentic AI assistant with a JARVIS-inspired personality.

- Powered by LangGraph + Groq (Llama 3.3)
- Three.js cyberpunk HUD frontend
- Voice + text + GUI modes
- Tool use, access control, and memory
- **Self-healing Cloudflare tunnel** → stable global link on Render

Run locally:
```
python main.py
```

Web / Render:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Self-healing Mac tunnel (24h Cloudflare quick tunnels)

Cloudflare `trycloudflare.com` links expire ~every 24 hours. TESrACT auto-registers
each new URL with your Render app so one stable link always opens the live Mac.

### One-time setup

1. On **Render**, set:
   - `BRAIN_REGISTRY_SECRET` = long random string
   - `ENABLE_HYBRID_ROUTING=true`
2. On **Mac** `.env`, set the same secret plus your Render URL:
   ```
   BRAIN_REGISTRY_SECRET=same-as-render
   TESRACT_RENDER_URL=https://YOUR-APP.onrender.com
   TESRACT_LOCAL_PORT=8000
   ENABLE_HYBRID_ROUTING=true
   ```
3. Install cloudflared: `brew install cloudflare/cloudflare/cloudflared`

### Every session on the Mac

```bash
# Terminal A — local TESrACT (Ollama + tools)
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal B — tunnel + auto-register with Render
python tunnel_manager.py
```

### Global link (bookmark this)

```
https://YOUR-APP.onrender.com/go
```

- Always **302-redirects** to the current `*.trycloudflare.com` tunnel
- JSON API: `GET /api/global-brain` → `{ "url": "https://….trycloudflare.com" }`
- Mac posts updates: `POST /api/update-brain` (done by `tunnel_manager.py`)

When the free tunnel drops, `tunnel_manager` restarts cloudflared, scrapes the new
URL, and updates Render again — no manual `.env` edits.
