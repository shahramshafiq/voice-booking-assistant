<div align="center">

# Voice Booking Assistant

A voice receptionist that books, looks up, moves and cancels appointments in Google Calendar, built for low latency.

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4.1--mini-412991?style=flat-square&logo=openai&logoColor=white)
![Google Calendar](https://img.shields.io/badge/Google%20Calendar-API%20v3-4285F4?style=flat-square&logo=googlecalendar&logoColor=white)
![Tests](https://img.shields.io/badge/tests-87%20passing-brightgreen?style=flat-square)

</div>

## Overview

You open a web page, press Start and talk. The assistant listens, decides what you need, checks one Google Calendar through a set of tools, offers free times, and only changes the calendar after you say yes to a read-back of the exact details. Speech goes in through Deepgram, the reply is written by an OpenAI model that can call tools, and ElevenLabs speaks it, with the work overlapped so the first sound starts as early as possible.

## Key Features

- **Eight booking tools:** services, business config, calendar availability, slot search, create, look up, reschedule and cancel. All rules (opening hours, breaks, buffers, notice, durations, cancellation policy) live in backend code and `.env`, never in the model.
- **Confirm before any change:** a booking, move or cancellation happens only after the assistant reads the details back and you say yes in the next turn. The question is written by the code, so what you hear is exactly what gets booked.
- **Early draft for speed:** the model starts working when Deepgram thinks you are nearly done. Read-only checks and the calendar re-check run early, and the final change waits until you have really finished speaking.
- **Safe against retries and races:** a repeated request returns the existing booking, a lost reply is checked before reporting failure, a reschedule updates the same calendar event and keeps the original if it fails, and two callers cannot take one slot.
- **Identity checks:** lookup, reschedule and cancel need the booking reference and the phone number, and wrong tries are limited per call and per phone number.
- **Input and output guardrails:** the input rail blocks instruction-override attempts and removes card numbers. The output rail stops secrets and instruction leaks, blocks "booked it" claims that no tool confirmed, and stops made-up booking references.
- **Short-term memory:** the conversation plus a state block written by the code (name, phone, offered times, looked-up booking, pending question). It is rebuilt empty on every call.
- **Honest failures:** an expired or invalid calendar token shows a clear warning and the assistant says it cannot reach the calendar, instead of pretending to book.
- **Natural voice behavior:** barge-in (interrupt it while it talks), short filler phrases while a slow tool works, and replies cleaned for speech.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI, Uvicorn, WebSockets |
| Speech to text | Deepgram Flux (streaming, with turn detection) |
| Language model | OpenAI Responses API with tool calling (gpt-4.1-mini) |
| Text to speech | ElevenLabs Flash v2.5 (streaming WebSocket) |
| Calendar | Google Calendar API v3 over httpx |
| Validation | Pydantic |
| Browser audio | AudioWorklet (microphone and player) |
| Testing | pytest |

## Project Structure

```text
voice-booking-assistant/
  app/
    main.py              FastAPI app, page, health check and the voice WebSocket
    config.py            Reads and validates every setting from .env
    session.py           One browser call: events, drafts, turns, barge-in
    turn.py              One spoken reply: guardrail, phrases, voice, filler
    agent.py             The loop: ask the model, run tools, ask again
    llm.py               OpenAI streaming client (kept warm)
    tools.py             Tool list, input checks and the confirmation gate
    memory.py            Short-term memory for one call
    stt.py / tts.py      Deepgram and ElevenLabs streaming clients
    chunker.py           Cuts the reply into phrases for the voice
    timing.py            Per-reply latency stopwatch
    booking/             Scheduling rules and the eight tools
    guardrails/          Input and output guardrails
  data/services.json     The services offered
  prompts/               System prompt template
  scripts/               check_calendar.py (free Google Calendar check)
  static/                Web page, audio worklets, styles
  tests/                 90 tests with fake services
```

## Getting Started

### Prerequisites

- Python 3.14
- API keys for OpenAI, Deepgram and ElevenLabs
- A Google Calendar (a separate test calendar is recommended) and its Calendar ID
- A modern desktop browser and a microphone (headphones recommended)

### Installation

```bash
git clone https://github.com/shahramshafiq/voice-booking-assistant.git
cd voice-booking-assistant
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

- `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`
- `GOOGLE_CALENDAR_ID` and `GOOGLE_CALENDAR_API_TOKEN`
- Business rules: `BUSINESS_NAME`, `BUSINESS_STAFF_NAME`, `BUSINESS_TIMEZONE`, `BUSINESS_HOURS`, `BUSINESS_BREAKS`, `BOOKING_BUFFER_MINUTES`, `MIN_NOTICE_MINUTES`, `MAX_ADVANCE_DAYS`, `CANCELLATION_NOTICE_HOURS`
- Services are listed in `data/services.json`

### Getting a Google Calendar token

The app uses a manually pasted access token (no login flow). It lasts about one hour.

1. Open the [Google OAuth 2.0 Playground](https://developers.google.com/oauthplayground/).
2. In "Input your own scopes" enter `https://www.googleapis.com/auth/calendar.events` and click Authorize APIs.
3. Sign in with the account that owns the calendar and allow access.
4. Click "Exchange authorization code for tokens" and copy the access token.
5. Paste it after `GOOGLE_CALENDAR_API_TOKEN=` in `.env` and restart the server.

Check it for free (no OpenAI, Deepgram or ElevenLabs calls):

```bash
python -m scripts.check_calendar
python -m scripts.check_calendar --write
```

### Run

```bash
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, press Start conversation and allow the microphone.

## Usage

Try a full conversation:

1. "I'd like to book a consultation tomorrow afternoon."
2. Pick one of the offered times, then give your name and phone number.
3. The assistant reads back the details and asks to confirm. Say "yes".
4. It books the slot and says a booking reference. Use the reference and the same phone number to look up, move or cancel the booking.

The page shows the live transcript, the tools that were used and a latency table for every reply.

## Testing

```bash
python -m pytest
```

The 90 tests use a fake Google Calendar, fake Deepgram and ElevenLabs connections and a scripted model, so they need no keys and cost nothing. They cover date parsing, slot maths, the booking rules, the confirmation gate, the guardrails and whole conversations.

## Limitations and Roadmap

- The Google token is pasted by hand and expires after about an hour.
- One staff member and one calendar. Locks and the race guard work inside one server process.
- The guardrails are rule based. They catch common attacks but are not a language-model judge.
- Speech recognition is configured for English and the app runs in the browser only, with no phone line.

## Author

**Shahram Shafiq** - [GitHub](https://github.com/shahramshafiq) | [LinkedIn](https://www.linkedin.com/in/shahramshafiq/)
