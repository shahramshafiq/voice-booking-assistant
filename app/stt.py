"""Speech to text with Deepgram Flux over a WebSocket.

Flux does the turn detection itself. It sends one "TurnInfo" message for each step:
  StartOfTurn     the user started talking (we use it for barge-in)
  Update          the words so far (we show them live)
  EagerEndOfTurn  the user PROBABLY finished (we start the LLM early)
  TurnResumed     they kept talking, throw the early answer away
  EndOfTurn       the user DEFINITELY finished (we speak the answer)
"""
import asyncio
import json
import logging
from contextlib import suppress
from urllib.parse import urlencode

import websockets

log = logging.getLogger("stt")


class DeepgramFluxStream:
    def __init__(self, settings, on_event, on_closed):
        """on_event(dict) is called for every message. on_closed(reason) if the socket ends by itself."""
        self.settings = settings
        self.on_event = on_event
        self.on_closed = on_closed
        self.websocket = None
        self.reader_task = None
        self.bytes_sent = 0
        self._closing = False

    @property
    def audio_seconds_sent(self):
        """How many seconds of audio we have streamed so far (16-bit mono = 2 bytes per sample)."""
        return self.bytes_sent / (self.settings.deepgram_sample_rate * 2)

    async def connect(self):
        s = self.settings
        params = {
            "model": s.deepgram_model,
            "encoding": s.deepgram_encoding,
            "sample_rate": s.deepgram_sample_rate,
            "eot_threshold": s.deepgram_eot_threshold,
            "eot_timeout_ms": s.deepgram_eot_timeout_ms,
        }
        if s.deepgram_eager_eot_threshold is not None:
            params["eager_eot_threshold"] = s.deepgram_eager_eot_threshold

        url = f"{s.deepgram_ws_url}?{urlencode(params)}"
        log.info("connecting to Deepgram (%s, eager_eot=%s)", s.deepgram_model, s.deepgram_eager_eot_threshold)

        self.websocket = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {s.deepgram_api_key}"},
            open_timeout=15,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        )
        self.reader_task = asyncio.create_task(self._read_loop())

    async def send_audio(self, data):
        websocket = self.websocket
        if websocket is None or websocket.close_code is not None:
            return
        try:
            await websocket.send(data)
            self.bytes_sent += len(data)
        except Exception as exc:
            log.warning("could not send audio to Deepgram: %s", type(exc).__name__)

    async def _read_loop(self):
        reason = "Deepgram closed the connection"
        try:
            async for raw_message in self.websocket:
                if isinstance(raw_message, bytes):
                    continue
                try:
                    event = json.loads(raw_message)
                except json.JSONDecodeError:
                    log.warning("Deepgram sent something that is not JSON")
                    continue
                try:
                    await self.on_event(event)
                except Exception:
                    log.exception("error while handling a Deepgram event")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = f"Deepgram connection failed: {type(exc).__name__}: {exc}"
            log.error(reason)

        if not self._closing:
            await self.on_closed(reason)

    async def close(self):
        self._closing = True
        websocket = self.websocket
        if websocket is not None and websocket.close_code is None:
            with suppress(Exception):
                await websocket.send(json.dumps({"type": "CloseStream"}))
            with suppress(Exception):
                await websocket.close()
        if self.reader_task is not None and not self.reader_task.done():
            self.reader_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self.reader_task
        self.websocket = None
        self.reader_task = None