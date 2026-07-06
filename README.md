# TESrACT

A proactive, agentic AI assistant with a JARVIS-inspired personality.

- Powered by LangGraph + Groq (Llama 3.3)
- Three.js cyberpunk HUD frontend
- Voice + text + GUI modes
- Tool use, access control, and memory

Run locally:
```
python main.py
```

Web / Render:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Colab GPU + Rendezvous Server

Heavy inference runs on a Colab T4 GPU exposed via ngrok. Instead of copying a new ngrok URL into `.env` after every Colab restart, TESrACT can discover the URL automatically through a small **Rendezvous Server**.

**How it works**

1. Deploy `rendezvous_server.py` on Render (see `render.yaml` → `tesract-rendezvous`).
2. Set `RENDEZVOUS_SERVER_URL` and `RENDEZVOUS_API_KEY` in your local `.env` (same key as `TESRACT_API_KEY` in Colab Secrets).
3. When Colab starts, it registers its ngrok URL via `POST /register`.
4. `llm_router.py` fetches the latest URL from `GET /active-url` (cached ~75s) and routes heavy tasks to Colab.

**Colab setup:** paste cells from `colab/tesract_llm_server.py` into your notebook. Add Colab Secrets: `TESRACT_API_KEY`, `RENDEZVOUS_SERVER_URL`, `NGROK_AUTHTOKEN`, `HF_TOKEN`.

**Fallback:** if `RENDEZVOUS_SERVER_URL` is unset, set `COLAB_LLM_URL` manually in `.env` (legacy mode).