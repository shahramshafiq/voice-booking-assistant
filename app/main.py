"""The FastAPI application: the web page, a health check and the voice WebSocket."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import build_instructions
from .booking.calendar_client import CalendarClient
from .booking.service import BookingService
from .config import settings
from .guardrails.input_rail import InputRail
from .guardrails.output_rail import OutputRail
from .llm import LLMClient
from .logging_setup import setup_logging
from .session import VoiceSession

setup_logging(settings.log_level)
log = logging.getLogger("main")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app):
    # everything below is created once and shared by every browser session
    llm = LLMClient(settings)
    llm.start_warmup()
    calendar = CalendarClient(settings)
    instructions = build_instructions(settings)

    app.state.llm = llm
    app.state.calendar = calendar
    app.state.service = BookingService(settings, calendar)
    app.state.input_rail = InputRail(settings)
    app.state.output_rail = OutputRail(settings, instructions)
    app.state.instructions = instructions

    log.info("ready: LLM=%s STT=%s TTS=%s calendar=%s", settings.openai_model, settings.deepgram_model,
             settings.elevenlabs_model, settings.google_calendar_id)
    yield
    await llm.close()
    await calendar.close()
    log.info("shut down")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    log.info("browser connected")
    state = websocket.app.state
    session = VoiceSession(websocket, settings, state.llm, state.service, state.input_rail, state.output_rail, state.instructions)
    await session.run()
    log.info("browser disconnected")
