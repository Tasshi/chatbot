# LLM chatbot (FastAPI + HTML)

A working chatbot: a FastAPI backend that holds your API key and calls Claude,
and a plain HTML frontend that talks to that backend. This avoids the CORS
error you get trying to call the Anthropic API directly from a browser.

## Setup

1. **Get an API key** from https://console.anthropic.com (Settings → API Keys).

2. **Create your `.env` file** in this folder:
   ```
   cp .env.example .env
   ```
   Then open `.env` and paste in your real key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Create a virtual environment and install dependencies:**
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux

   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```
   uvicorn server:app --reload --port 8000
   ```
   You should see it running at http://localhost:8000. Visit
   http://localhost:8000/health in a browser to confirm — it should return
   `{"status":"ok"}`.

5. **Open `chatbot.html`** directly in your browser (double-click it, or
   right-click → Open with → your browser). The "Server" field at the top
   should already say `http://localhost:8000` — leave it as is.

6. **Chat.** Your messages go: browser → FastAPI server → Anthropic API →
   back to the browser. Your API key never leaves the server.

## Why this fixes the CORS error

The Anthropic API doesn't allow direct requests from an arbitrary browser
origin (like `file://` or a random localhost port) for security reasons —
API keys aren't meant to sit in frontend JavaScript where anyone could copy
them from dev tools. The FastAPI server is a small proxy: it holds the key
server-side, accepts requests from your frontend, forwards them to
Anthropic, and returns the reply. `CORSMiddleware` in `server.py` is what
allows your HTML file to call this local server.

## Troubleshooting

- **"Failed to fetch" / "Is the server running?"** — the FastAPI server
  isn't running, or is running on a different port. Check the terminal
  where you ran `uvicorn`.
- **401 error** — your API key in `.env` is missing or invalid.
- **"ANTHROPIC_API_KEY is not set"** on startup — you haven't created `.env`,
  or it's not in the same folder as `server.py`.

## Next steps

- Add streaming (server-sent events) so responses appear token-by-token.
- Add rate limiting (`slowapi`) and request logging.
- Swap the frontend for a proper framework, or deploy the backend (Render,
  Railway, Fly.io) so you can reach it from anywhere, not just localhost.
