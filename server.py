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
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{{model}}:generateContent"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file (see .env.example) "
        "with a free API key from https://aistudio.google.com/apikey."
    )

BASE_DIR = Path(__file__).resolve().parent
CHATBOT_HTML_PATH = BASE_DIR / "chatbot.html"

app = FastAPI(title="LLM Chatbot API")

# Allow the local HTML frontend (opened via file:// or a local dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (style.css, future .js/image files, etc.) at /static/*
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


@app.get("/")
def serve_chatbot():
    if not CHATBOT_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="chatbot.html not found next to server.py")
    return FileResponse(CHATBOT_HTML_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Gemini uses "model"/"user" roles and a "contents" list instead of Anthropic's "messages" format.
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
        try:
            resp = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=payload,
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}")

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()

    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        reply = "\n".join(p.get("text", "") for p in parts) or "(no response)"
    except (KeyError, IndexError):
        reply = "(no response)"

    usage = data.get("usageMetadata", {})

    return ChatResponse(
        reply=reply,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
    )
