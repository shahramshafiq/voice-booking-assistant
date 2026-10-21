"""Tests for the Deepgram client's outbox and lag check (no network)."""
import asyncio
import logging

from app.config import settings
from app.stt import DeepgramFluxStream


class SlowSocket:
    """Like a connection on a slow upload: every send takes a while."""
    close_code = None

    def __init__(self):
        self.sent = []

    async def send(self, data):
        await asyncio.sleep(0.05)
        self.sent.append(data)


def test_slow_upload_never_blocks_and_late_words_are_reported(caplog):
    async def scenario():
        async def nothing(*_):
            pass

        stt = DeepgramFluxStream(settings, nothing, nothing)
        stt.websocket = SlowSocket()
        stt._sender_task = asyncio.create_task(stt._send_loop())
        chunk = b"\x00\x00" * int(settings.deepgram_sample_rate * 0.08)  # 80 ms of audio

        started = asyncio.get_running_loop().time()
        for _ in range(10):
            await stt.send_audio(chunk)
        assert asyncio.get_running_loop().time() - started < 0.01  # the browser's messages never wait for the upload
        assert stt.bytes_sent == 0  # nothing has really gone out yet
        await asyncio.sleep(0.7)
        assert stt.bytes_sent == len(chunk) * 10 and stt.audio_seconds_sent == 0.8

        stt._check_lag({"type": "TurnInfo", "event": "Update", "audio_window_end": 0.6})
        assert stt.behind_ms == 200 and "behind" not in caplog.text  # normal
        stt.bytes_sent += len(chunk) * 30  # 2.4 s more sent, but Deepgram has not caught up
        stt._check_lag({"type": "TurnInfo", "event": "Update", "audio_window_end": 0.6})
        assert stt.behind_ms == 2600
        stt._sender_task.cancel()

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert "Deepgram is 2.6 s behind" in caplog.text
