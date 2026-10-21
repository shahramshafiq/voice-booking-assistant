"""Tests for the tool runner: validation, the confirmation gate, and the short-term memory it feeds."""
import asyncio
import json
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from app.booking.calendar_client import CalendarClient
from app.booking.service import BookingService
from app.config import settings as base_settings
from app.memory import SessionMemory
from app.tools import TOOLS, ToolContext, ToolRunner, build_definitions, for_model
from tests.fake_google import FakeGoogleCalendar

TZ = ZoneInfo("Asia/Karachi")
NOW = datetime(2026, 10, 19, 8, 0, tzinfo=TZ)
SETTINGS = replace(base_settings, calendar_cache_seconds=30)


def setup():
    fake = FakeGoogleCalendar()
    service = BookingService(SETTINGS, CalendarClient(SETTINGS, transport=fake.transport()), clock=lambda: NOW)
    memory = SessionMemory(SETTINGS)
    return fake, service, memory, ToolRunner(SETTINGS, service, memory)


async def no_events(event):
    pass


def turn(memory, text, committed=True):
    """A new conversation turn. committed=False is an early draft that the caller has not finished yet."""
    memory.turn_no += 1
    gate = asyncio.Event()
    if committed:
        gate.set()
    return ToolContext(gate=gate, user_text=text, emit=no_events, turn_no=memory.turn_no)


BOOK = {
    "customer_details": {"name": "Ali Raza", "phone": "03001234567"},
    "service_id": "consultation",
    "start_time": "2026-10-20T09:00:00+05:00",
    "end_time": "2026-10-20T10:00:00+05:00",
}


def call(runner, name, args, ctx):
    return asyncio.run(runner.run(name, json.dumps(args), ctx))


def test_definitions_match_the_models_and_have_no_leftover_refs():
    definitions = build_definitions()
    assert [d["name"] for d in definitions] == list(TOOLS)
    text = json.dumps(definitions)
    assert "$ref" not in text and "$defs" not in text
    create = next(d for d in definitions if d["name"] == "create_booking")
    assert create["parameters"]["properties"]["customer_details"]["properties"]["phone"]["type"] == "string"
    assert set(create["parameters"]["required"]) == {"customer_details", "service_id", "start_time", "end_time"}


def test_bad_arguments_come_back_as_a_clear_error():
    _, _, memory, runner = setup()
    ctx = turn(memory, "hi")
    result = asyncio.run(runner.run("find_available_slots", "{not json", ctx))
    assert result["error"] == "INVALID_INPUT"
    result = call(runner, "find_available_slots", {"service_id": "consultation"}, ctx)
    assert result["error"] == "INVALID_INPUT" and "date_range" in result["message"]
    assert call(runner, "make_coffee", {}, ctx)["error"] == "UNKNOWN_TOOL"


def test_reading_tools_run_even_in_an_early_draft():
    fake, _, memory, runner = setup()
    ctx = turn(memory, "slots tomorrow", committed=False)  # the caller has not finished speaking
    result = call(runner, "find_available_slots", {"service_id": "consultation", "date_range": {"start_date": "tomorrow"}}, ctx)
    assert result["ok"] and result["slots"]
    assert memory.offered_slots == []  # a draft does not touch the memory...
    for item in ctx.deferred:  # ...until the draft is really used (Turn.commit does this)
        runner.record(*item)
    assert memory.offered_slots and memory.service_name == "Consultation"


def test_a_draft_never_changes_anything():
    fake, _, memory, runner = setup()
    ctx = turn(memory, "book it", committed=False)

    async def attempt():
        task = asyncio.ensure_future(runner.run("create_booking", json.dumps(BOOK), ctx))
        await asyncio.sleep(0.2)
        assert not task.done()  # it is waiting for the caller to finish speaking
        task.cancel()  # the caller kept talking: the draft is thrown away
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(attempt())
    assert fake.events == {} and memory.pending is None  # nothing happened, nothing was parked


def test_a_draft_prepares_the_change_and_only_the_final_step_waits():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", BOOK, turn(memory, "book it"))  # the caller is asked to confirm
    ctx = ToolContext(gate=asyncio.Event(), user_text="Yes please", emit=no_events)  # a draft: not numbered yet

    async def scenario():
        task = asyncio.ensure_future(runner.run("create_booking", json.dumps(BOOK), ctx))
        await asyncio.sleep(0.3)
        methods = [m for m, _ in fake.requests]
        assert "GET" in methods and "POST" not in methods and not task.done()  # checked early, nothing changed yet
        memory.turn_no += 1  # the caller finishes: the turn is committed
        ctx.turn_no = memory.turn_no
        ctx.gate.set()
        return await task

    result = asyncio.run(scenario())
    assert result["ok"] and len(fake.events) == 1


def test_a_yes_to_an_older_question_does_not_count():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    turn(memory, "what is your address")  # an unrelated turn in between
    result = call(runner, "create_booking", BOOK, turn(memory, "yes, and can you also check my other booking"))
    assert result["error"] == "CONFIRMATION_REQUIRED" and fake.events == {}


def test_first_call_only_asks_for_confirmation():
    fake, _, memory, runner = setup()
    result = call(runner, "create_booking", BOOK, turn(memory, "Book it for Ali, number 0300 1234567"))
    assert result["error"] == "CONFIRMATION_REQUIRED"
    assert "Consultation tomorrow at 9 AM" in result["say"] and "Shall I book it?" in result["say"]
    assert fake.events == {} and memory.pending is not None
    assert memory.customer_name == "Ali Raza"


def test_yes_in_the_next_turn_makes_the_booking():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    ctx = turn(memory, "Yes, please go ahead")
    result = call(runner, "create_booking", BOOK, ctx)
    assert result["ok"] and len(fake.events) == 1
    assert result["say"].startswith("Done. Your Consultation is booked for tomorrow at 9 AM. Your booking reference is ")
    assert "say" not in json.loads(for_model(result))  # the model is never shown the sentence
    assert "create_booking" in memory.achieved
    assert memory.pending is None and result["booking_reference"] in memory.known_refs
    assert memory.recent_actions and "booked Consultation" in memory.recent_actions[0]


def test_a_yes_in_the_same_turn_is_not_enough():
    fake, _, memory, runner = setup()
    ctx = turn(memory, "Book it, yes, confirm, go ahead")
    first = call(runner, "create_booking", BOOK, ctx)
    second = call(runner, "create_booking", BOOK, ctx)  # the model tries again straight away
    assert first["error"] == second["error"] == "CONFIRMATION_REQUIRED"
    assert fake.events == {}


def test_no_or_unclear_answers_do_not_confirm():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    for answer in ("No, wait", "hmm let me think", "don't do it", "yes but not tomorrow"):
        assert call(runner, "create_booking", BOOK, turn(memory, answer))["error"] == "CONFIRMATION_REQUIRED", answer
    assert fake.events == {}


def test_a_changed_booking_needs_a_fresh_confirmation():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    changed = dict(BOOK, start_time="2026-10-20T10:00:00+05:00", end_time="2026-10-20T11:00:00+05:00")
    result = call(runner, "create_booking", changed, turn(memory, "yes but make it 10"))
    assert result["error"] == "CONFIRMATION_REQUIRED" and "10 AM" in result["say"]
    assert fake.events == {}
    assert call(runner, "create_booking", changed, turn(memory, "yes"))["ok"]
    assert len(fake.events) == 1


def test_an_old_question_expires():
    fake, _, memory, runner = setup()
    memory.settings = replace(SETTINGS, confirm_ttl_seconds=0)  # every question is already too old
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    assert call(runner, "create_booking", BOOK, turn(memory, "yes"))["error"] == "CONFIRMATION_REQUIRED"
    assert fake.events == {}


def test_plain_yes_and_no_are_recognised_but_only_right_after_the_question():
    _, _, memory, runner = setup()
    assert runner.quick_decision(turn(memory, "yes")) is None  # nothing was asked yet
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    decisions = {
        "Yes": "yes", "Yes please go ahead": "yes", "okay": "yes", "Yeah, that's right": "yes", "Sure, go ahead.": "yes",
        "Uh, Yeah. Of course.": "yes", "Of course.": "yes", "Of course not": None,
        "No": "no", "No thanks": "no", "Nope, don't do it": "no", "wait": "no", "never mind": "no",
        "yes but not tomorrow": None, "Yes, and what are your hours": None, "hmm let me think": None,
        "thanks": None, "make it three pm": None, "": None,
    }
    for words, expected in decisions.items():
        _, _, memory2, runner2 = setup()
        call(runner2, "create_booking", BOOK, turn(memory2, "book it"))
        assert runner2.quick_decision(turn(memory2, words)) == expected, words
    memory.turn_no += 1  # an unrelated turn in between
    assert runner.quick_decision(turn(memory, "yes")) is None  # the question is no longer the previous turn


def test_cancel_flow_uses_the_looked_up_details():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", dict(BOOK, start_time="2026-10-23T09:00:00+05:00", end_time="2026-10-23T10:00:00+05:00"), turn(memory, "book"))
    booked = call(runner, "create_booking", dict(BOOK, start_time="2026-10-23T09:00:00+05:00", end_time="2026-10-23T10:00:00+05:00"), turn(memory, "yes"))
    lookup = {"booking_reference": booked["booking_reference"], "customer_phone": "03001234567"}
    found = call(runner, "get_booking", lookup, turn(memory, "cancel my booking, the reference is that one"))
    assert found["ok"] and memory.verified_booking["service"] == "Consultation"
    asked = call(runner, "cancel_booking", lookup, turn(memory, "please cancel it"))
    assert asked["error"] == "CONFIRMATION_REQUIRED" and "your Consultation" in asked["say"]
    assert len(fake.events) == 1
    done = call(runner, "cancel_booking", lookup, turn(memory, "yes"))
    assert done["ok"] and fake.events == {} and memory.verified_booking is None


def test_wrong_details_lock_the_lookup_after_a_few_tries():
    _, _, memory, runner = setup()
    for _ in range(SETTINGS.max_verify_failures):
        result = call(runner, "get_booking", {"booking_reference": "000000", "customer_phone": "03001234567"}, turn(memory, "check"))
        assert result["error"] == "BOOKING_NOT_FOUND"
    locked = call(runner, "get_booking", {"booking_reference": "000000", "customer_phone": "03001234567"}, turn(memory, "check"))
    assert locked["error"] == "TOO_MANY_ATTEMPTS"


def test_a_change_that_is_interrupted_still_finishes_and_is_remembered():
    fake, _, memory, runner = setup()
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    ctx = turn(memory, "yes")

    async def interrupted():
        task = asyncio.ensure_future(runner.run("create_booking", json.dumps(BOOK), ctx))
        await asyncio.sleep(0)  # the change has started
        task.cancel()  # the caller interrupts
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)  # give the protected change time to finish

    asyncio.run(interrupted())
    assert len(fake.events) == 1  # the booking was made exactly once
    assert memory.recent_actions and "create_booking" in memory.achieved  # and the memory knows about it


def test_what_the_model_sees_hides_internal_fields_and_secrets():
    text = for_model({"ok": True, "event_id": "abc", "say": "hello", "booking_reference": "123456"})
    assert json.loads(text) == {"ok": True, "booking_reference": "123456"}


def test_state_block_shows_the_facts_the_model_needs():
    _, _, memory, runner = setup()
    call(runner, "find_available_slots", {"service_id": "consultation", "date_range": {"start_date": "tomorrow"}}, turn(memory, "slots"))
    call(runner, "create_booking", BOOK, turn(memory, "book it"))
    state = memory.render_state(NOW, "Asia/Karachi")
    assert "Monday 19 October 2026, 8 AM" in state
    assert "name Ali Raza, phone 03001234567" in state
    assert "start 2026-10-20T09:00:00+05:00" in state
    assert "Waiting for the caller to say yes or no" in state
