import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ESCO Mapper Frontend")

# --- PATH LOGIC ---
# This finds the directory where app.py lives, then looks for 'templates' inside it.
# This makes your app-dir src command much more stable.
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # CRITICAL: Use keywords (request=, name=, context=) to avoid the 500 error
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"backend_url": BACKEND_URL}
    )