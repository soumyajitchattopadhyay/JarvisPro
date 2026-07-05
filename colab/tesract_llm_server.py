# =============================================================================
# TESrACT Colab LLM Server — paste each "CELL" block into a separate Colab cell
# Notebook: https://colab.research.google.com/drive/1BQv2dWYN-8QdJNQth0ukNow6lbOmBacd
# GPU: Runtime → Change runtime type → T4 GPU
# =============================================================================

# %% CELL 1 — Install dependencies (restart runtime if Colab asks)
# !pip install -q torch transformers accelerate bitsandbytes fastapi uvicorn pyngrok huggingface_hub sentencepiece

# %% CELL 2 — Secrets (Colab: left sidebar → Secrets, or paste manually)
import os
from google.colab import userdata  # type: ignore

def _secret(name: str, default: str = "") -> str:
    try:
        return userdata.get(name)
    except Exception:
        return os.environ.get(name, default)

HF_TOKEN = _secret("HF_TOKEN")          # huggingface.co/settings/tokens (for Llama)
NGROK_AUTHTOKEN = _secret("NGROK_AUTHTOKEN")  # dashboard.ngrok.com/get-started/your-authtoken
TESRACT_API_KEY = _secret("TESRACT_API_KEY", "change-me-in-production")

MODEL_ID = os.environ.get("TESRACT_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
# Fallback if Llama gated access is unavailable:
# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

PORT = int(os.environ.get("TESRACT_PORT", "8000"))
MAX_NEW_TOKENS = int(os.environ.get("TESRACT_MAX_NEW_TOKENS", "1024"))

print("Model:", MODEL_ID)
print("HF token set:", bool(HF_TOKEN))
print("Ngrok token set:", bool(NGROK_AUTHTOKEN))

# %% CELL 3 — Load model on T4 (4-bit quantization)
import torch
import threading
import time
import uuid
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

assert torch.cuda.is_available(), "Enable T4 GPU: Runtime → Change runtime type → T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    torch_dtype=torch.float16,
)
model.eval()

print("Model loaded on", next(model.parameters()).device)

# %% CELL 4 — FastAPI server (OpenAI-compatible + health)
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

_inference_lock = threading.Lock()
_server_started = threading.Event()
_stats = {"requests": 0, "errors": 0, "last_request_at": None}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = Field(default=512, ge=16, le=MAX_NEW_TOKENS)
    stream: bool = False


app = FastAPI(title="TESrACT Colab LLM", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(authorization: str | None) -> None:
    if not TESRACT_API_KEY or TESRACT_API_KEY == "change-me-in-production":
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.split(" ", 1)[1].strip() != TESRACT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Use the model chat template when available."""
    payload = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
    # Fallback prompt format
    parts = []
    for m in messages:
        parts.append(f"{m.role.upper()}: {m.content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def _generate(messages: list[ChatMessage], temperature: float, max_tokens: int) -> str:
    prompt = _messages_to_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        with _inference_lock:  # one request at a time on T4
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "online",
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "stats": _stats,
    }


@app.post("/v1/chat/completions")
def chat_completions(
    body: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    if body.stream:
        raise HTTPException(status_code=400, detail="stream=false only")

    _stats["requests"] += 1
    _stats["last_request_at"] = time.time()

    try:
        content = _generate(body.messages, body.temperature, min(body.max_tokens, MAX_NEW_TOKENS))
    except Exception as exc:
        _stats["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": body.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/generate")
def generate_alias(
    body: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    """Simple alias used by TESrACT llm_router.py."""
    result = chat_completions(body, authorization)
    return {"reply": result["choices"][0]["message"]["content"]}


def _run_uvicorn() -> None:
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    _server_started.set()


threading.Thread(target=_run_uvicorn, daemon=True).start()
for _ in range(40):
    time.sleep(0.5)
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
        break
    except Exception:
        pass

print(f"Local server ready: http://127.0.0.1:{PORT}/health")

# %% CELL 5 — ngrok public URL (copy this into your .env as COLAB_LLM_URL)
from pyngrok import ngrok

ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(PORT, "http")
PUBLIC_URL = tunnel.public_url
print("\n" + "=" * 60)
print("TESrACT Colab LLM is LIVE")
print("COLAB_LLM_URL =", PUBLIC_URL)
print("Set in local .env:")
print(f"  COLAB_LLM_URL={PUBLIC_URL}")
print(f"  COLAB_LLM_API_KEY={TESRACT_API_KEY}")
print("=" * 60)

# %% CELL 6 — Keep-alive loop (run this last; keeps the session busy)
import urllib.request

print("Keep-alive running. Do not stop this cell.")
while True:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10)
        print(f"[keepalive] {time.strftime('%H:%M:%S')} OK")
    except Exception as exc:
        print(f"[keepalive] health check failed: {exc}")
    time.sleep(300)  # ping every 5 minutes