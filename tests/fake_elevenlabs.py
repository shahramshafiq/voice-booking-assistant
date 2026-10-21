"""A fake ElevenLabs multi-context WebSocket server, for the tests (the real service is never contacted).

mode "ok":    text is spoken. Audio is made when 50 characters are waiting or on "flush", like the real service.
mode "quota": behaves like an account with no credits left. Checked against the real service: it answers every
              text that is flushed with {"audio": null, "isFinal": true}, with no error message at all.
"""
import asyncio
import base64
import json
import math
import struct

import websockets


class FakeElevenLabs:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.connections = 0
        self.received = []  # every message the app sent
        self.server = None

    async def start(self):
        self.server = await websockets.serve(self._handle, "127.0.0.1", 0)
        return f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()

    async def _handle(self, websocket):
        self.connections += 1
        contexts = {}
        async for raw in websocket:
            message = json.loads(raw)
            self.received.append(message)
            if message.get("close_socket"):
                await websocket.close()
                return
            context_id = message.get("context_id")
            if context_id is None:
                continue
            context = contexts.setdefault(context_id, {"buffer": ""})
            if message.get("close_context"):
                await self._speak(websocket, context_id, context)
                await websocket.send(json.dumps({"audio": None, "isFinal": True, "contextId": context_id}))
                contexts.pop(context_id, None)
                continue
            text = message.get("text")
            if text and text.strip():
                context["buffer"] += text
            if message.get("flush") or len(context["buffer"]) >= 50:
                await self._speak(websocket, context_id, context)
                if self.mode == "quota":
                    await websocket.send(json.dumps({"audio": None, "isFinal": True, "contextId": context_id}))

    async def _speak(self, websocket, context_id, context):
        text, context["buffer"] = context["buffer"].strip(), ""
        if not text or self.mode != "ok":
            return
        samples = int(16000 * 0.06 * len(text))  # 60 ms of tone per character
        pcm = b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(samples))
        half = len(pcm) // 2
        for piece in (pcm[:half], pcm[half:]):
            await asyncio.sleep(0.02)
            await websocket.send(json.dumps({"audio": base64.b64encode(piece).decode(), "isFinal": False, "contextId": context_id}))
