"""Text to speech with ElevenLabs over ONE persistent WebSocket per browser session.

Each assistant reply gets its own "context" on that socket, so the connection is opened
once and reused for every turn. This is the same design as the earlier LLM + TTS project.
"""
import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import suppress
from urllib.parse import urlencode

import websockets

log = logging.getLogger("tts")


class ElevenLabsSession:
    """One persistent ElevenLabs socket for one browser session."""

    def __init__(self, settings):
        self.settings = settings
        self.websocket = None
        self.reader_task = None
        self.connection_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.contexts = {}
        self.closed = False

    # ---------------------------------------------------------------- connecting
    async def connect(self):
        """Open the socket if needed. Returns (websocket, newly_connected, connection_ms)."""
        async with self.connection_lock:
            if (
                self.websocket is not None
                and self.websocket.close_code is None
                and self.reader_task is not None
                and not self.reader_task.done()
            ):
                return self.websocket, False, 0.0

            if self.closed:
                raise RuntimeError("ElevenLabs session is already closed.")

            s = self.settings
            params = {
                "model_id": s.elevenlabs_model,
                "output_format": s.elevenlabs_output_format,
                "inactivity_timeout": str(s.elevenlabs_inactivity_timeout),
                "auto_mode": str(s.elevenlabs_auto_mode).lower(),
            }
            url = (
                f"{s.elevenlabs_ws_base_url}/v1/text-to-speech/"
                f"{s.elevenlabs_voice_id}/multi-stream-input?{urlencode(params)}"
            )

            started_at = time.perf_counter()
            self.websocket = await websockets.connect(
                url,
                open_timeout=s.tts_connect_timeout_seconds,
                ping_interval=20,
                ping_timeout=20,
                max_size=16 * 1024 * 1024,
            )
            connection_ms = (time.perf_counter() - started_at) * 1000
            log.info("new ElevenLabs connection opened in %.0f ms", connection_ms)
            self.reader_task = asyncio.create_task(self._read_messages())
            return self.websocket, True, connection_ms

    # ---------------------------------------------------------------- one reply = one context
    async def start_context(self, context_id, on_audio, on_first_audio):
        websocket, newly_connected, connection_ms = await self.connect()

        self.contexts[context_id] = {
            "done": asyncio.get_running_loop().create_future(),
            "on_audio": on_audio,
            "on_first_audio": on_first_audio,
            "first_audio_seen": False,
        }

        s = self.settings
        init_message = {
            "context_id": context_id,
            "text": " ",
            "xi_api_key": s.elevenlabs_api_key,
            "voice_settings": {
                "stability": s.elevenlabs_stability,
                "similarity_boost": s.elevenlabs_similarity,
                "use_speaker_boost": s.elevenlabs_speaker_boost,
            },
            "generation_config": {
                "chunk_length_schedule": s.elevenlabs_chunk_schedule,
            },
        }

        try:
            async with self.send_lock:
                await websocket.send(json.dumps(init_message))
        except Exception:
            self.contexts.pop(context_id, None)
            raise

        return {"new_connection": newly_connected, "connection_ms": connection_ms}

    async def check_voice(self, text="Hi."):
        """Speak one tiny word and throw the sound away. Returns how many bytes of audio came back.

        0 means the voice is not working. When the account is out of credits ElevenLabs does not send an error:
        it simply answers every text with "no audio, finished".
        """
        total = 0

        async def count(audio):
            nonlocal total
            total += len(audio)

        async def nothing():
            pass

        context_id = f"check-{uuid.uuid4()}"
        await self.start_context(context_id, on_audio=count, on_first_audio=nothing)
        await self.send_text(context_id, text + " ", flush=True)
        try:
            await asyncio.wait_for(self.finish_context(context_id), self.settings.tts_check_timeout_seconds)
        except asyncio.TimeoutError:
            log.warning("the voice check timed out after %.0f s", self.settings.tts_check_timeout_seconds)
        return total

    async def send_text(self, context_id, text, flush=False):
        websocket = self.websocket
        if websocket is None or websocket.close_code is not None:
            raise RuntimeError("ElevenLabs WebSocket is not connected.")

        message = {"context_id": context_id, "text": text}
        if flush:
            message["flush"] = True

        if text.strip():
            log.debug("sending %d characters to TTS (flush=%s): %r", len(text.strip()), flush, text.strip()[:60])

        async with self.send_lock:
            await websocket.send(json.dumps(message))

    async def finish_context(self, context_id):
        """Normal end of a reply: flush what is left, close ONLY this context, wait for is_final."""
        websocket = self.websocket
        context = self.contexts.get(context_id)
        if websocket is None or websocket.close_code is not None:
            raise RuntimeError("ElevenLabs WebSocket is not connected.")
        if context is None:
            raise RuntimeError(f"Unknown ElevenLabs context: {context_id}")

        async with self.send_lock:
            await websocket.send(json.dumps({"context_id": context_id, "flush": True}))
            await websocket.send(json.dumps({"context_id": context_id, "close_context": True}))

        try:
            await asyncio.wait_for(
                asyncio.shield(context["done"]),
                timeout=self.settings.turn_timeout_seconds,
            )
        finally:
            self.contexts.pop(context_id, None)

    async def cancel_context(self, context_id):
        """Interruption: stop this reply right now. Late audio for it is ignored."""
        context = self.contexts.pop(context_id, None)
        if context is not None and not context["done"].done():
            context["done"].cancel()

        websocket = self.websocket
        if websocket is None or websocket.close_code is not None:
            return
        with suppress(Exception):
            async with self.send_lock:
                await websocket.send(json.dumps({"context_id": context_id, "close_context": True}))

    # ---------------------------------------------------------------- listening to ElevenLabs
    async def _read_messages(self):
        """Continuously read all audio from the shared socket and route it to the right context."""
        try:
            async for raw_message in self.websocket:
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")

                payload = json.loads(raw_message)
                context_id = payload.get("context_id", payload.get("contextId"))

                if payload.get("error"):
                    log.error("ElevenLabs error: %s", payload["error"])
                    error = RuntimeError(f"ElevenLabs error: {payload['error']}")
                    targets = [self.contexts[context_id]] if context_id in self.contexts else list(self.contexts.values())
                    for context in targets:
                        if not context["done"].done():
                            context["done"].set_exception(error)
                    continue

                context = self.contexts.get(context_id)
                if context is None:
                    continue  # a reply that already finished or was interrupted

                encoded_audio = payload.get("audio")
                if encoded_audio:
                    audio_bytes = base64.b64decode(encoded_audio)
                    if audio_bytes:
                        if not context["first_audio_seen"]:
                            context["first_audio_seen"] = True
                            await context["on_first_audio"]()
                        await context["on_audio"](audio_bytes)

                if payload.get("is_final", payload.get("isFinal", False)):
                    if not context["done"].done():
                        context["done"].set_result(True)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("ElevenLabs reader failed: %s: %s", type(exc).__name__, exc)
            self._fail_waiting(f"ElevenLabs connection failed: {type(exc).__name__}: {exc}")
            return

        if not self.closed:
            log.warning("ElevenLabs closed the connection")
            self._fail_waiting("ElevenLabs closed the connection")

    def _fail_waiting(self, message):
        for context in list(self.contexts.values()):
            if not context["done"].done():
                context["done"].set_exception(RuntimeError(message))

    async def close(self):
        self.closed = True
        websocket = self.websocket

        if websocket is not None and websocket.close_code is None:
            with suppress(Exception):
                async with self.send_lock:
                    await websocket.send(json.dumps({"close_socket": True}))
            with suppress(Exception):
                await websocket.close()

        if self.reader_task is not None and not self.reader_task.done():
            self.reader_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self.reader_task

        self.websocket = None
        self.reader_task = None

        for context in self.contexts.values():
            if not context["done"].done():
                context["done"].cancel()
        self.contexts.clear()