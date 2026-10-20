"""Everything about talking to OpenAI.

One LLMClient is created when the server starts and shared by every browser session.
It keeps its HTTPS connection warm so a turn never pays for a cold connection.
"""
import asyncio
import logging
import time
from contextlib import suppress
from typing import NamedTuple, Optional

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

log = logging.getLogger("llm")


class LLMEvent(NamedTuple):
    """Something the model produced: a piece of text, or a complete request to use a tool."""
    kind: str  # "text" or "tool_call"
    text: str = ""
    call: Optional[dict] = None  # {"id", "call_id", "name", "arguments"}


class LLMClient:
    def __init__(self, settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=20,
                    # the library default is 5 seconds, which hangs up between turns
                    keepalive_expiry=settings.openai_keepalive_expiry_seconds,
                ),
            ),
        )
        self._warm_task = None

    # ---------------------------------------------------------------- keep the connection warm
    def start_warmup(self):
        self._warm_task = asyncio.create_task(self._warm_loop())

    async def _warm_loop(self):
        """Ping OpenAI every few seconds so the HTTPS connection never goes cold."""
        while True:
            started_at = time.perf_counter()
            try:
                await self.client.models.retrieve(self.settings.openai_model)
                log.debug("warm-up ping ok: %.0f ms", (time.perf_counter() - started_at) * 1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("warm-up ping failed: %s", type(exc).__name__)
            await asyncio.sleep(self.settings.openai_warm_interval_seconds)

    # ---------------------------------------------------------------- the actual reply
    async def stream(self, instructions, input_items, tools, allow_tools=True):
        """Yield LLMEvents as OpenAI writes the answer.

        A tool request is yielded the moment it is complete, so the tool can start working
        while the model is still finishing its message.
        """
        options = {
            "model": self.settings.openai_model,
            "instructions": instructions,
            "input": input_items,
            "stream": True,
            "max_output_tokens": self.settings.openai_max_output_tokens,
        }
        if tools:
            options.update(tools=tools, tool_choice="auto" if allow_tools else "none", parallel_tool_calls=False)

        stream = await self.client.responses.create(**options)
        try:
            async for event in stream:
                kind = event.type
                if kind == "response.output_text.delta":
                    if event.delta:
                        yield LLMEvent("text", text=event.delta)
                elif kind == "response.output_item.done" and getattr(event.item, "type", None) == "function_call":
                    item = event.item
                    yield LLMEvent("tool_call", call={"id": item.id, "call_id": item.call_id, "name": item.name, "arguments": item.arguments})
                elif kind in ("error", "response.failed"):
                    raise RuntimeError(f"OpenAI stream failed: {event}")
                elif kind == "response.incomplete":
                    # normal when the answer hits OPENAI_MAX_OUTPUT_TOKENS: just stop here
                    log.warning("OpenAI answer was cut off at the output token limit")
                    return
        finally:
            with suppress(Exception):
                await stream.close()

    async def close(self):
        if self._warm_task is not None:
            self._warm_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._warm_task
        await self.client.close()