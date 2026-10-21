"""One assistant reply: agent text -> output guardrail -> chunks -> TTS audio -> browser.

A Turn can start in "draft" mode: the model starts working as soon as Deepgram says the user
PROBABLY finished (EagerEndOfTurn). The Turn then waits at a gate. If the user really finished
(EndOfTurn) the gate opens and the text that is already written is spoken immediately.
If the user kept talking (TurnResumed) the draft is simply thrown away. Tools that only READ may
already have run during the draft; tools that CHANGE something wait at the same gate (see tools.py).
"""
import asyncio
import logging
import uuid
from contextlib import suppress

from .agent import run_agent
from .booking.speech import tidy_numbers
from .chunker import extract_speakable_chunks
from .tools import ToolContext

log = logging.getLogger("turn")


class Turn:
    def __init__(self, user_text, history, settings, llm, tts, timer, send_event, send_audio, on_speaking,
                 memory, runner, rail, instructions, scripted_reply=None):
        self.turn_id = 0  # the real number is given when the turn is committed
        self.user_text = user_text
        self.history = history  # a copy of the conversation so far
        self.settings = settings
        self.llm = llm
        self.tts = tts
        self.timer = timer
        self.send_event = send_event
        self.send_audio = send_audio
        self.on_speaking = on_speaking  # called once, when the first audio arrives
        self.memory = memory
        self.runner = runner
        self.rail = rail  # the output guardrail
        self.instructions = instructions
        self.scripted_reply = scripted_reply  # when set, this is said instead of asking the model

        self.gate = asyncio.Event()  # opened by commit(); a draft waits here
        self.ctx = ToolContext(gate=self.gate, user_text=user_text, emit=send_event)
        self.task = None
        self.cancelled = False  # set the moment the reply is stopped, so late audio is dropped
        self.draft_hit = False  # True if the early (draft) answer was the one we used
        self.eot_lag_ms = None

        self.model_chunks = []  # the approved phrases that make up the answer (not the filler)
        self.sent_chunks = []  # everything sent to TTS, in order (used to estimate what was heard)
        self._display = []  # the same, for the page: shown with plain numbers
        self.audio_bytes = 0  # audio received from TTS so far
        self.finished = False  # True once ALL audio has been generated

        self._context_id = None
        self._context_open = False
        self._buffer = ""  # text written but not yet cut into phrases
        self._first_sent = False  # the first phrase after a pause is small and flushed at once
        self._after_filler = False  # a filler is playing, so the next phrase can be a natural size (still flushed)
        self._tail = ""  # the end of the previous phrase, for the output guardrail
        self._ready = 0  # how many pieces of speakable text are waiting in the queue

    # ---------------------------------------------------------------- control
    def start(self):
        self.task = asyncio.create_task(self._run())
        return self.task

    def commit(self, turn_no):
        """The user really finished: allow this turn to start speaking."""
        self.turn_id = turn_no
        self.ctx.turn_no = turn_no
        for name, args, result in self.ctx.deferred:  # what the early draft learned now counts
            self.runner.record(name, args, result)
        self.ctx.deferred.clear()
        self.timer.mark("committed")
        self.gate.set()

    def stop(self):
        """Ask the reply to stop right now. Returns at once; the clean-up (closing the model's stream, closing
        the voice context) finishes in the background."""
        self.cancelled = True
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def cancel(self):
        """Stop the reply and wait until its clean-up is finished."""
        self.stop()
        if self.task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await self.task

    @property
    def answer(self):
        """What the assistant said, as plain text with plain numbers (this goes into the conversation history)."""
        return tidy_numbers(" ".join(self.model_chunks)).strip()

    @property
    def audio_ms(self):
        # 16-bit mono PCM: 2 bytes per sample
        return self.audio_bytes / (self.settings.tts_sample_rate * 2) * 1000

    def heard_text(self, played_ms):
        """Best guess of the words the user actually heard before interrupting.

        It is an estimate: the share of audio that was played is applied to the text that was sent.
        """
        sent = " ".join(self.sent_chunks)
        if not sent or self.audio_bytes == 0 or played_ms <= 0:
            return ""
        ratio = min(1.0, played_ms / max(self.audio_ms, 1.0))
        cut = int(len(sent) * ratio)
        if cut >= len(sent):
            return tidy_numbers(sent)
        text = sent[:cut]
        if not sent[cut].isspace():  # do not end in the middle of a word
            space = text.rfind(" ")
            text = text[:space] if space > 0 else ""
        return tidy_numbers(text.strip())

    # ---------------------------------------------------------------- the work
    async def _run(self):
        queue = asyncio.Queue()
        producer = asyncio.create_task(self._produce(queue))
        try:
            await self.gate.wait()  # a draft pauses here until the user has really finished

            self._context_id = str(uuid.uuid4())
            await self.tts.start_context(
                self._context_id,
                on_audio=self._on_audio,
                on_first_audio=self._on_first_audio,
            )
            self._context_open = True

            still_going = True
            while still_going:
                item = await queue.get()
                if item is None:
                    still_going = await self._speak("", force=True)  # whatever text is left
                    break
                if isinstance(item, Exception):
                    raise item
                kind, data = item
                if kind in ("text", "scripted"):
                    self._ready -= 1
                if kind == "text":
                    still_going = await self._speak(data)
                elif kind == "scripted":
                    still_going = await self._speak(data, force=True, trusted=True)
                elif kind == "tool_start":
                    await self._early_filler(data, answer_is_ready=self._ready > 0)
                elif kind == "tools":
                    still_going = await self._speak("", force=True)  # say what is written before the pause
                    if still_going:
                        await self._maybe_say_filler(data)
                    self._first_sent = False  # the first phrase after the pause is flushed at once

            if not self.model_chunks:
                raise RuntimeError("The model returned no text.")

            await self.tts.finish_context(self._context_id)
            self._context_open = False
            self.finished = True
            await self._report_if_the_voice_was_silent()
            log.info("turn %s: all audio generated (%.1f s of sound for %d characters)", self.turn_id, self.audio_ms / 1000,
                     sum(len(chunk) for chunk in self.sent_chunks))
            log.info("turn %s said: %s", self.turn_id, tidy_numbers(" ".join(self._display)))

        finally:
            producer.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await producer
            if self._context_open and not self.finished:
                with suppress(Exception):
                    await self.tts.cancel_context(self._context_id)

    async def _report_if_the_voice_was_silent(self):
        """The voice service can answer "finished" without any sound (for example when the account is out of
        credits). The reply text is still on the page; tell the user why they could not hear it."""
        characters = sum(len(chunk.strip()) for chunk in self.sent_chunks)
        if characters and self.audio_ms < characters * self.settings.tts_min_ms_per_character:
            log.error("turn %s: the voice returned %.1f s of sound for %d characters; the account may be out of credits",
                      self.turn_id, self.audio_ms / 1000, characters)
            await self.send_event({"type": "warning", "message": self.settings.no_audio_warning})

    async def _produce(self, queue):
        """Runs in the background: put everything the assistant wants to say on the queue."""

        async def emit(kind, data):
            if kind == "text":
                self.timer.mark("llm_first_token")
            if kind in ("text", "scripted"):
                self._ready += 1
            await queue.put((kind, data))

        try:
            if self.scripted_reply is not None:
                await emit("scripted", self.scripted_reply)
            else:
                self.timer.mark("llm_request")
                await run_agent(
                    llm=self.llm, runner=self.runner, memory=self.memory, ctx=self.ctx, settings=self.settings,
                    instructions=self.instructions, history=self.history, user_text=self.user_text,
                    emit=emit, timer=self.timer,
                )
            await queue.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("turn %s: the assistant failed: %s: %s", self.turn_id, type(exc).__name__, exc)
            await queue.put(RuntimeError(f"The language model failed: {type(exc).__name__}: {exc}"))

    # ---------------------------------------------------------------- speaking
    async def _speak(self, text, force=False, trusted=False):
        """Cut the text into phrases and send each one. Returns False when the output guardrail stopped the answer."""
        self._buffer += text
        chunks, self._buffer = extract_speakable_chunks(
            self._buffer,
            force=force,
            first_chunk=not self._first_sent and not self._after_filler,
            first_chunk_chars=self.settings.first_chunk_chars,
            target_chars=self.settings.chunk_target_chars,
            max_extra_chars=self.settings.chunk_max_extra_chars,
        )
        for chunk in chunks:
            if not await self._send_chunk(chunk, trusted):
                self._buffer = ""
                return False
        return True

    async def _send_chunk(self, chunk, trusted):
        if not trusted:
            verdict = self.rail.check(chunk, self._tail, self.memory.achieved, self.memory.known_refs)
            if not verdict.ok:
                # say the safe replacement instead, and stop the rest of the answer
                await self._emit_chunk(verdict.text, flush=True, model=True)
                return False
            chunk = verdict.text
        self._tail = f"{self._tail} {chunk}"[-80:]
        await self._emit_chunk(chunk, flush=not self._first_sent, model=True)
        self._first_sent = True
        return True

    async def _emit_chunk(self, text, flush, model):
        self.timer.mark("first_chunk_sent")
        self.sent_chunks.append(text)
        if model:
            self.model_chunks.append(text)
        # The page always gets the whole reply so far with plain numbers (a number can be cut between two phrases).
        self._display.append(text)
        await self.send_event({"type": "assistant_text", "text": tidy_numbers(" ".join(self._display)), "replace": True})
        await self.tts.send_text(self._context_id, text + " ", flush=flush)

    async def _early_filler(self, name, answer_is_ready):
        """The model has just begun a tool request. Writing its arguments takes another second or so, and a
        second model call follows the tool, so say the filler phrase right now. Skipped when something was
        already said, or when the answer is already waiting in the queue (the early draft finished everything
        before the caller did)."""
        if self.sent_chunks or self._buffer.strip() or answer_is_ready:
            return
        phrases = self._filler_phrases(name)
        if not phrases:
            return
        self.timer.mark("filler_sent")
        await self._emit_chunk(phrases[self.turn_id % len(phrases)], flush=True, model=False)
        self._first_sent = False  # the first real phrase after the filler is flushed at once
        self._after_filler = True

    def _filler_phrases(self, name):
        """Which phrases fit this tool right now? A change that is only being prepared (the confirmation question
        comes next) gets a different sentence from a change the caller has just said yes to."""
        if self.runner.is_write(name) and not self.runner.expects_confirmed_change(name, self.ctx):
            return self.settings.tool_filler_prepare
        return self.settings.tool_fillers.get(name, [])

    async def _maybe_say_filler(self, calls):
        """While a tool works, say a short phrase so the line never goes silent. Skipped when something was
        already said, or when the tools are already finished (for example they ran during the early draft)."""
        if self.sent_chunks:
            return
        tasks = [call["task"] for call in calls]
        if any(self.runner.is_write(call["name"]) for call in calls):
            # a change that only needs a confirmation question finishes at once and needs no filler
            await asyncio.wait(tasks, timeout=self.settings.tool_filler_write_delay_seconds)
        phrases = self._filler_phrases(calls[0]["name"])
        if all(task.done() for task in tasks) or not phrases:
            return
        self.timer.mark("filler_sent")
        await self._emit_chunk(phrases[self.turn_id % len(phrases)], flush=True, model=False)
        self._after_filler = True

    async def _on_first_audio(self):
        self.timer.mark("tts_first_audio")
        await self.on_speaking(self)

    async def _on_audio(self, audio_bytes):
        if self.cancelled:
            return  # audio that arrives after an interruption must never reach the speakers
        self.audio_bytes += len(audio_bytes)
        self.timer.mark("audio_to_browser")
        await self.send_audio(audio_bytes)