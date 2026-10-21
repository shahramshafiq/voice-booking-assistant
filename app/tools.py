"""The tools the model may call, and the gate every call must pass.

The model never touches the calendar. It asks, and ToolRunner decides:
  - the input is checked against the schema (bad input comes back as a clear error);
  - READ tools run at once, even during an early "draft" answer, because they change nothing;
  - WRITE tools (create, reschedule, cancel) wait until the caller has really finished speaking, and
    only run if the caller said "yes" to this exact action in an EARLIER turn. Otherwise the action is
    parked as "pending" and the system itself reads it back and asks for confirmation.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ValidationError

from .booking.calendar_client import CalendarError
from .booking.models import (
    BookingLookupArgs, CheckAvailabilityArgs, CreateBookingArgs, FindSlotsArgs, NoArgs, RescheduleArgs,
)
from .booking.service import BookingError

log = logging.getLogger("tools")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    model: type
    description: str
    write: bool = False


TOOLS = {spec.name: spec for spec in [
    ToolSpec("get_services", NoArgs, "List the services with id, name and duration. The prompt already has them; rarely needed."),
    ToolSpec("get_business_config", NoArgs, "The staff member, calendar, hours, breaks, buffer and booking rules. Rarely needed."),
    ToolSpec("check_calendar_availability", CheckAvailabilityArgs,
             "Low level: free times on ONE day for a length in minutes. Prefer find_available_slots."),
    ToolSpec("find_available_slots", FindSlotsArgs,
             "Find free appointment times for a service over one or more days. Use this to offer times."),
    ToolSpec("create_booking", CreateBookingArgs,
             "Book an appointment. The system reads the details back and asks the caller to confirm; "
             "call it again with the same details after the caller says yes.", write=True),
    ToolSpec("get_booking", BookingLookupArgs, "Look up a booking by reference and phone number."),
    ToolSpec("reschedule_booking", RescheduleArgs,
             "Move a booking to a new day and time. The system asks the caller to confirm first.", write=True),
    ToolSpec("cancel_booking", BookingLookupArgs,
             "Cancel a booking. The system asks the caller to confirm first.", write=True),
]}

NEEDS_VERIFICATION = {"get_booking", "reschedule_booking", "cancel_booking"}


@dataclass
class ToolContext:
    """What a tool call needs to know about the turn it belongs to."""
    gate: asyncio.Event  # opens when the caller has really finished speaking
    user_text: str
    emit: Callable  # async function that sends a small event to the browser
    turn_no: int = 0  # set at the moment the turn is committed
    deferred: list = field(default_factory=list)  # results of tools that finished before the caller did


def _inline_refs(schema, definitions):
    """Pydantic puts nested models under $defs; write them in place and drop the noisy titles."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            return _inline_refs(definitions[schema["$ref"].split("/")[-1]], definitions)
        return {k: _inline_refs(v, definitions) for k, v in schema.items() if k not in ("title", "$defs")}
    if isinstance(schema, list):
        return [_inline_refs(item, definitions) for item in schema]
    return schema


def build_definitions():
    """The tool list in the format OpenAI expects, generated from the same classes that validate the input."""
    definitions = []
    for spec in TOOLS.values():
        schema = spec.model.model_json_schema()
        definitions.append({
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": _inline_refs(schema, schema.get("$defs", {})),
            "strict": False,
        })
    return definitions


def for_model(result):
    """The compact JSON text the model gets back. Internal fields stay out."""
    return json.dumps({k: v for k, v in result.items() if k not in ("event_id", "say")}, separators=(",", ":"))


def _word_pattern(words):
    escaped = "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![a-z'])(?:{escaped})(?![a-z'])") if words else None


class ToolRunner:
    """One per conversation: it knows the conversation's memory."""

    def __init__(self, settings, service, memory):
        self.settings = settings
        self.service = service
        self.memory = memory
        self.definitions = build_definitions()
        self._yes = _word_pattern(settings.confirm_words)
        self._no = _word_pattern(settings.refuse_words)

    @staticmethod
    def is_write(name):
        return name in TOOLS and TOOLS[name].write

    def is_affirmative(self, text):
        """Did the caller say yes? A "no" or "wait" in the same sentence cancels it."""
        cleaned = re.sub(r"[^a-z' ]", " ", text.lower().replace("\u2019", "'"))
        if self._no and self._no.search(cleaned):
            return False
        return bool(self._yes and self._yes.search(cleaned))

    def expects_confirmed_change(self, name, ctx):
        """Will this change tool probably run for real, because the caller is saying yes to the question just asked?
        Used before the model has finished writing the arguments, so it can only look at what the caller said."""
        pending = self.memory.pending
        upcoming_turn = ctx.turn_no or self.memory.turn_no + 1
        return (
            pending is not None
            and pending.tool == name
            and pending.turn_no == upcoming_turn - 1
            and self.is_affirmative(ctx.user_text)
        )

    # ================================================================ the one entry point
    async def run(self, name, raw_arguments, ctx):
        started = time.perf_counter()
        args = None
        recorded = False  # True when a finished change already updated the memory itself
        try:
            spec = TOOLS.get(name)
            if spec is None:
                result = {"ok": False, "error": "UNKNOWN_TOOL", "message": f"There is no tool called {name}."}
            else:
                args = self._parse(spec, raw_arguments)
                result, recorded = await self._dispatch(spec, args, ctx)
        except ValueError as exc:  # bad JSON or bad fields
            result = {"ok": False, "error": "INVALID_INPUT", "message": str(exc)}
        except BookingError as exc:
            result = exc.as_result()
        except CalendarError as exc:
            result = {"ok": False, "error": exc.code, "message": exc.message}
        except asyncio.TimeoutError:
            result = {"ok": False, "error": "TOOL_TIMEOUT", "message": "That took too long. Ask the caller to try again in a moment."}
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("tool %s crashed", name)  # details go to the log only, never to the model
            result = {"ok": False, "error": "INTERNAL_ERROR", "message": "Something went wrong on our side."}

        if not recorded:
            if ctx.gate.is_set():
                self.record(name, args, result)
            else:
                # an early draft must not change the memory unless the draft is really used (see Turn.commit)
                ctx.deferred.append((name, args, result))
        elapsed = (time.perf_counter() - started) * 1000
        log.info("tool %s -> %s (%.0f ms)", name, result.get("error", "ok"), elapsed)
        try:
            await ctx.emit({"type": "tool", "name": name, "ok": bool(result.get("ok")), "error": result.get("error"), "ms": round(elapsed)})
        except Exception:
            pass
        return result

    # ================================================================ steps
    @staticmethod
    def _parse(spec, raw_arguments):
        try:
            data = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
        except json.JSONDecodeError:
            raise ValueError("The arguments were not valid JSON.")
        try:
            return spec.model.model_validate(data)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            raise ValueError(f"Invalid arguments ({problems}). Ask the caller for what is missing.")

    def record(self, name, args, result):
        """Write what a finished tool call taught us into the memory."""
        if args is not None:
            self.memory.note_result(name, args, result)
        if result.get("error") == "BOOKING_NOT_FOUND":
            self.memory.verify_failures += 1
        if result.get("ok"):
            self.memory.achieved.add(name)

    async def _dispatch(self, spec, args, ctx):
        """Returns (result, recorded)."""
        if spec.name in NEEDS_VERIFICATION and self.memory.verify_failures >= self.settings.max_verify_failures:
            return {"ok": False, "error": "TOO_MANY_ATTEMPTS",
                    "message": f"Too many wrong attempts. Ask the caller to contact {self.settings.business_staff_name} directly."}, False
        if not spec.write:
            method = getattr(self.service, spec.name)
            return await asyncio.wait_for(method(args), self.settings.tool_timeout_seconds), False
        return await self._run_write(spec, args, ctx)

    async def _run_write(self, spec, args, ctx):
        # Step 1, which may happen during the early draft: check everything against the live calendar.
        # Nothing is changed here, so a discarded draft costs nothing.
        # The confirmation QUESTION is checked against the in-memory calendar window; the check before the real
        # change (the caller is saying yes) always asks Google.
        plan = await self.service.prepare(spec.name, args, self.memory.verified_booking, fresh=self.expects_confirmed_change(spec.name, ctx))
        if plan.result is not None:  # an error, or a retry of something already done: nothing to confirm
            return plan.result, False

        upcoming_turn = ctx.turn_no or self.memory.turn_no + 1
        confirmed = self.memory.pending_is_confirmable(plan.key, upcoming_turn) and self.is_affirmative(ctx.user_text)

        await ctx.gate.wait()  # Step 2, the real change, waits until the caller has really finished speaking
        if not confirmed:
            self.memory.set_pending(plan.key, spec.name, plan.question)
            log.info("%s is waiting for the caller's yes", spec.name)
            return {"ok": False, "error": "CONFIRMATION_REQUIRED", "say": plan.question,
                    "message": "Nothing has been done yet. The caller is being asked to confirm."}, False
        self.memory.pending = None

        # From here on the change must finish even if the caller interrupts. The memory is therefore
        # updated by a callback of the change itself, which runs even when nobody is waiting any more.
        change = asyncio.ensure_future(plan.apply())

        def remember(done):
            if not done.cancelled() and done.exception() is None:
                self.record(spec.name, args, done.result())

        change.add_done_callback(remember)
        return await asyncio.shield(change), True