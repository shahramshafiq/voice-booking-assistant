"""Tests for the ElevenLabs client against a fake server that speaks the real protocol (no key, no credits used)."""
import asyncio
from dataclasses import replace

from app.config import settings as base_settings
from app.tts import ElevenLabsSession
from tests.fake_elevenlabs import FakeElevenLabs


def run(coroutine):
    return asyncio.run(coroutine)


async def with_server(mode, scenario):
    fake = FakeElevenLabs(mode)
    url = await fake.start()
    tts = ElevenLabsSession(replace(base_settings, elevenlabs_ws_base_url=url, turn_timeout_seconds=5, tts_check_timeout_seconds=3))
    try:
        return await scenario(tts, fake)
    finally:
        await tts.close()
        await fake.stop()


async def say(tts, context_id, steps):
    """Run the same calls a spoken reply makes. Returns (bytes of audio, how many times the first-audio callback ran)."""
    got = {"bytes": 0, "first": 0}

    async def on_audio(data):
        got["bytes"] += len(data)

    async def on_first():
        got["first"] += 1

    await tts.start_context(context_id, on_audio=on_audio, on_first_audio=on_first)
    for text, flush in steps:
        await tts.send_text(context_id, text + " ", flush=flush)
    await tts.finish_context(context_id)
    return got["bytes"], got["first"]


def test_a_reply_is_spoken_and_the_connection_is_reused():
    async def scenario(tts, fake):
        first = await say(tts, "a", [("We offer a consultation and a follow up.", True), ("Which one?", False)])
        second = await say(tts, "b", [("Okay.", True)])
        assert first[0] > 40000 and first[1] == 1  # about 3 seconds of sound, announced once
        assert second[0] > 0 and fake.connections == 1

    run(with_server("ok", scenario))


def test_the_filler_then_answer_pattern_gives_sound_for_both():
    async def scenario(tts, fake):
        total, first = await say(tts, "c", [("Sure, let me check the calendar.", True), ("I have two PM or three PM.", True), ("Which one?", False)])
        expected = 16000 * 2 * 0.06 * len("Sure, let me check the calendar. I have two PM or three PM. Which one?")
        assert total > expected * 0.9

    run(with_server("ok", scenario))


def test_short_text_waits_for_a_flush_like_the_real_service():
    async def scenario(tts, fake):
        got = []

        async def on_audio(data):
            got.append(len(data))

        async def on_first():
            pass

        await tts.start_context("d", on_audio=on_audio, on_first_audio=on_first)
        await tts.send_text("d", "Hello there ", flush=False)
        await asyncio.sleep(0.2)
        assert got == []  # fewer than 50 characters and no flush: nothing is spoken yet
        await tts.send_text("d", "friend ", flush=True)
        await asyncio.sleep(0.3)
        assert sum(got) > 0
        await tts.finish_context("d")

    run(with_server("ok", scenario))


def test_the_voice_check_hears_sound_when_the_account_works():
    async def scenario(tts, fake):
        assert await tts.check_voice() > 0

    run(with_server("ok", scenario))


def test_an_account_without_credits_is_detected_by_the_voice_check_and_by_a_reply():
    async def scenario(tts, fake):
        assert await tts.check_voice() == 0  # no error message, just silence: that is what the real service does
        total, first = await say(tts, "e", [("We offer a consultation and a follow up.", True)])
        assert total == 0 and first == 0

    run(with_server("quota", scenario))


def test_a_cancelled_reply_ignores_late_sound():
    async def scenario(tts, fake):
        got = []

        async def on_audio(data):
            got.append(len(data))

        async def on_first():
            pass

        await tts.start_context("f", on_audio=on_audio, on_first_audio=on_first)
        await tts.send_text("f", "A long answer that is still being spoken when the caller interrupts. ", flush=True)
        await tts.cancel_context("f")
        await asyncio.sleep(0.3)
        assert sum(got) <= 60000  # at most what had already arrived; nothing after the cancel is delivered
        assert "f" not in tts.contexts

    run(with_server("ok", scenario))
