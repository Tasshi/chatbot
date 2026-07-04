import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = [
    "openrouter/free",  # auto-router: always picks a currently-available free model
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
]

if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    raise RuntimeError(
        "Set at least one of GEMINI_API_KEY or OPENROUTER_API_KEY in your .env file."
    )

BASE_DIR = Path(__file__).resolve().parent
CHATBOT_HTML_PATH = BASE_DIR / "chatbot.html"

app = FastAPI(title="LLM Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 1000
    temperature: float = 1.0


class ChatResponse(BaseModel):
    reply: str
    input_tokens: int
    output_tokens: int
    provider: str


@app.get("/")
def serve_chatbot():
    if not CHATBOT_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="chatbot.html not found next to server.py")
    return FileResponse(CHATBOT_HTML_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


async def call_gemini(req: ChatRequest) -> ChatResponse:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    contents = [
        {
            "role": "model" if m.role == "assistant" else "user",
            "parts": [{"text": m.content}],
        }
        for m in req.messages
    ]

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": req.max_tokens,
            "temperature": req.temperature,
        },
    }

    url = GEMINI_URL.format(model=GEMINI_MODEL)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, params={"key": GEMINI_API_KEY}, json=payload)

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Gemini error {resp.status_code}: {detail}")

    data = resp.json()
    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        reply = "\n".join(p.get("text", "") for p in parts) or "(no response)"
    except (KeyError, IndexError):
        raise RuntimeError("Gemini returned no candidates")

    usage = data.get("usageMetadata", {})
    return ChatResponse(
        reply=reply,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        provider=f"gemini:{GEMINI_MODEL}",
    )


async def call_openrouter(req: ChatRequest) -> ChatResponse:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    last_error = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in OPENROUTER_MODELS:
            try:
                resp = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://dpl.bt",
                        "X-Title": "DPL Chatbot",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                    },
                )
                if resp.status_code != 200:
                    last_error = f"{model} -> {resp.status_code}: {resp.text}"
                    continue

                data = resp.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content")
                if not reply:
                    last_error = f"{model} -> empty response"
                    continue

                usage = data.get("usage", {})
                return ChatResponse(
                    reply=reply,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    provider=f"openrouter:{model}",
                )
            except httpx.RequestError as exc:
                last_error = f"{model} -> {exc}"
                continue

    raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    errors = []

    if OPENROUTER_API_KEY:
        try:
            return await call_openrouter(req)
        except Exception as exc:
            errors.append(str(exc))

    if GEMINI_API_KEY:
        try:
            return await call_gemini(req)
        except Exception as exc:
            errors.append(str(exc))

    raise HTTPException(status_code=502, detail="All providers failed: " + " | ".join(errors))