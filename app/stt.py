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
import time
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
        self.window_end = 0.0  # seconds of audio Deepgram says it has listened to so far
        self._closing = False
        # Audio waits in this outbox and a separate task sends it, so a slow upload never holds up the browser's messages.
        self._outbox = asyncio.Queue()
        self._queued_bytes = 0
        self._sender_task = None
        self._last_lag_warning = 0.0

    def is_open(self):
        """True while the connection to Deepgram is usable."""
        return (
            self.websocket is not None
            and self.websocket.close_code is None
            and self.reader_task is not None
            and not self.reader_task.done()
        )

    @property
    def audio_seconds_sent(self):
        """How many seconds of audio we have streamed so far (16-bit mono = 2 bytes per sample)."""
        return self.bytes_sent / (self.settings.deepgram_sample_rate * 2)

    @property
    def behind_ms(self):
        """How far Deepgram is behind the audio we have sent (normally 150 to 400 ms)."""
        return max(0, int((self.audio_seconds_sent - self.window_end) * 1000))

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
        self._sender_task = asyncio.create_task(self._send_loop())

    async def send_audio(self, data):
        """Never waits: the audio is put in the outbox and sent in the background."""
        websocket = self.websocket
        if websocket is None or websocket.close_code is not None:
            return
        self._queued_bytes += len(data)
        self._outbox.put_nowait(data)

    async def _send_loop(self):
        while True:
            data = await self._outbox.get()
            try:
                await self.websocket.send(data)
            except Exception as exc:
                log.warning("could not send audio to Deepgram: %s", type(exc).__name__)
                return  # the connection is gone; the reader reports it
            self._queued_bytes -= len(data)
            self.bytes_sent += len(data)

    def _check_lag(self, event):
        """Deepgram tells us how much audio it has listened to. If that is far behind what we sent, the words will
        arrive late (a slow internet upload, or Deepgram itself is slow). Say so in the log, at most every 5 s."""
        window_end = event.get("audio_window_end")
        if window_end is None:
            return
        self.window_end = float(window_end)
        now = time.monotonic()
        if self.behind_ms > self.settings.stt_lag_warning_ms and now - self._last_lag_warning > 5:
            self._last_lag_warning = now
            waiting = self._queued_bytes / (self.settings.deepgram_sample_rate * 2)
            log.warning("Deepgram is %.1f s behind (%.1f s of it still waiting here to be sent): your words will "
                        "arrive late. Usually a slow internet upload.", self.behind_ms / 1000, waiting)

    async def configure(self, eot_threshold, eot_timeout_ms):
        """Change how quickly Flux decides that a turn is finished, without reconnecting.
        Deepgram answers with ConfigureSuccess (or ConfigureFailure, and the old values stay)."""
        websocket = self.websocket
        if websocket is None or websocket.close_code is not None:
            return
        message = {"type": "Configure", "thresholds": {"eot_threshold": eot_threshold, "eot_timeout_ms": eot_timeout_ms}}
        try:
            await websocket.send(json.dumps(message))
        except Exception as exc:
            log.warning("could not send the new turn settings to Deepgram: %s", type(exc).__name__)

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
                self._check_lag(event)
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
        if self._sender_task is not None:
            self._sender_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._sender_task
            self._sender_task = None
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