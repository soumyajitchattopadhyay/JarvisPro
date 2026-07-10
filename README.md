# TESrACT

A proactive, agentic AI assistant with a JARVIS-inspired personality.

- Powered by LangGraph + Groq (Llama 3.3)
- Three.js cyberpunk HUD frontend
- Voice + text + GUI modes
- Tool use, access control, and memory
- **Self-healing Cloudflare tunnel** → stable global link on Render
- **Secure Brain mode** — host never mutates the OS for public requests; physical actions return client-side intents
- **HMAC handshake** on `/api/update-brain` and `/internal/llm/invoke` (`BRAIN_REGISTRY_SECRET`)
- **Public `/chat` HUD** by default (no browser secrets); optional `BRAIN_AUTH_REQUIRE_CHAT=true`

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

## Free operator auth + chat history (Mac SSD)

No paid identity provider and no cloud database:

| Piece | Where it runs | Cost |
|--------|----------------|------|
| Email OTP | Mac `smtplib` → Gmail App Password | Free |
| Session tokens | SQLite `agent_data/tesract_sessions.db` on Mac | Free |
| Chat history | Same SQLite (conversations + messages) | Free |
| Second-laptop UI | Render → Cloudflare tunnel → Mac | Free tunnel |

### Setup (Mac `.env`)

```bash
AUTH_SMTP_HOST=smtp.gmail.com
AUTH_SMTP_PORT=587
AUTH_SMTP_USER=you@gmail.com
AUTH_SMTP_PASSWORD=your-app-password
AUTH_SMTP_FROM=you@gmail.com
AUTH_ALLOWED_EMAILS=you@gmail.com   # optional allow-list
# Local testing without mail:
# AUTH_DEV_ECHO_OTP=true
```

### API

- `GET /auth/status` — auth/smtp flags  
- `POST /request-auth` `{ "email" }` — smtplib OTP (Mac); aliases: `/auth/request-otp`  
- `POST /verify-auth` `{ "email", "otp" }` → `{ session_token, email }`; aliases: `/auth/verify-otp`  
- `GET /auth/me` — Bearer / `X-Session-Token`  
- `GET /chat/history` — restore messages (SQLite on Mac)  
- `POST /chat` — requires `session_token`; LangGraph `thread_id` = operator email  

On Render, these routes are **proxied to the Mac** whenever the tunnel is healthy.

## Hybrid: Render edge → Mac brain

When `ENABLE_HYBRID_ROUTING=true` on Render and `tunnel_manager.py` is healthy:

1. The **browser talks to Render** (stable URL).
2. Render **proxies full `/chat`** to the live Mac tunnel (tools + LLM run on the Mac).
3. Fallback: if the Mac is offline, Render answers locally (Groq + edge tools).

Check `/health` → `hybrid.proxy_chat_to_brain` / `hybrid.remote_mac_healthy`.
Chat responses may include `"brain_route": "mac_proxy"` when the Mac handled the turn.

Set on Render: `TESRACT_ROLE=edge`, `TESRACT_TIMEZONE=Asia/Kolkata` (or your zone).
Set on Mac: `TESRACT_TIMEZONE=Asia/Kolkata` (host local is usually enough).

## Secure Brain architecture

The Mac process is a **cognitive engine** only:

| Concern | Behavior |
|--------|----------|
| Shell / `subprocess` / host writes | **Disabled** on the Brain |
| File creation | Returns `action: DOWNLOAD_FILE` for the client |
| List/read/mkdir/open/shell | Returns `action: DISPLAY_DATA` client intents |
| Pure Q&A / math / search | `action: LOGIC_ONLY` (Brain-local) |
| Public `/chat` (HUD) | Open by default — browsers cannot embed secrets. Set `BRAIN_AUTH_REQUIRE_CHAT=true` to lock |
| `/api/update-brain`, `/internal/llm/invoke` | Require HMAC or Bearer `BRAIN_REGISTRY_SECRET` |
| Localhost HUD | Always works; optional strict chat auth still allows loopback when `BRAIN_AUTH_LOCALHOST_BYPASS=true` |

Example authenticated `/chat` response when packaging a file:

```json
{
  "status": "success",
  "execution_target": "client",
  "action": "DOWNLOAD_FILE",
  "payload": { "filename": "notes.txt", "content": "..." },
  "reply": "Prepared file for client-side download...",
  "executions": [ ... ]
}
```

Sign outbound requests (tunnel_manager / hybrid Render→Mac) with headers from `brain_auth.sign()`:
`X-Brain-Timestamp`, `X-Brain-Signature`, optional `X-Brain-Token`, or `Authorization: Bearer <secret>`.
