"""Tests for the in-memory calendar window: searches and lookups need no trip to Google, changes always do."""
import asyncio
import json
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from app.booking.calendar_client import CalendarClient
from app.booking.models import BookingLookupArgs, CreateBookingArgs, CustomerDetails, DateRange, FindSlotsArgs
from app.booking.service import BookingService
from app.config import settings as base_settings
from tests.fake_google import FakeGoogleCalendar

TZ = ZoneInfo("Asia/Karachi")
NOW = datetime(2026, 10, 19, 8, 0, tzinfo=TZ)


def make():
    fake = FakeGoogleCalendar()
    config = replace(base_settings, calendar_cache_seconds=30)
    return fake, BookingService(config, CalendarClient(config, transport=fake.transport()), clock=lambda: NOW)


def gets(fake):
    return [m for m, _ in fake.requests].count("GET")


def find(service, day="tomorrow"):
    return service.find_available_slots(FindSlotsArgs(service_id="consultation", date_range=DateRange(start_date=day)))


def book_args(slot):
    return CreateBookingArgs(customer_details=CustomerDetails(name="Ali Raza", phone="0300 1234567"),
                             service_id="consultation", start_time=slot["start"], end_time=slot["end"])


def test_searches_for_any_day_need_only_one_request_for_the_whole_window():
    fake, service = make()

    async def scenario():
        await service.warm()
        before = gets(fake)
        for day in ("tomorrow", "friday", "next week", "2026-11-05"):
            assert (await find(service, day))["ok"]
        assert gets(fake) == before  # nothing went to Google: the window answered every search

    asyncio.run(scenario())


def test_a_booking_can_be_looked_up_from_memory_but_the_change_itself_is_live():
    fake, service = make()

    async def scenario():
        await service.warm()
        slot = (await find(service))["slots"][0]
        created = await service.create_booking(book_args(slot))
        await asyncio.sleep(0.2)  # the background refresh after the change
        before = gets(fake)
        lookup = BookingLookupArgs(booking_reference=created["booking_reference"], customer_phone="03001234567")
        assert (await service.get_booking(lookup))["ok"]
        assert gets(fake) == before  # found in memory
        plan = await service.prepare("cancel_booking", lookup)  # default fresh=True: asks Google
        assert plan.apply is not None and gets(fake) > before

    asyncio.run(scenario())


def test_the_confirmation_question_uses_memory_and_the_real_change_is_checked_live():
    fake, service = make()

    async def scenario():
        await service.warm()
        slot = (await find(service))["slots"][0]
        before = gets(fake)
        question = await service.prepare("create_booking", book_args(slot), fresh=False)
        assert question.apply is not None and gets(fake) == before  # no request for the question
        live = await service.prepare("create_booking", book_args(slot), fresh=True)
        assert gets(fake) == before + 1  # the check before the real change asks Google

    asyncio.run(scenario())


def test_a_change_made_elsewhere_is_caught_by_the_live_check():
    fake, service = make()

    async def scenario():
        await service.warm()
        slot = (await find(service))["slots"][0]
        fake.add_event(slot["start"], slot["end"])  # somebody books the slot directly in Google Calendar
        stale = await service.prepare("create_booking", book_args(slot), fresh=False)
        assert stale.apply is not None  # the memory has not noticed yet...
        live = await service.prepare("create_booking", book_args(slot), fresh=True)
        assert live.result is not None and live.result["error"] == "SLOT_UNAVAILABLE"  # ...but the live check does

    asyncio.run(scenario())


def test_the_memory_is_refilled_after_a_change_so_the_next_search_is_instant_and_correct():
    fake, service = make()

    async def scenario():
        await service.warm()
        slot = (await find(service))["slots"][0]
        await service.create_booking(book_args(slot))
        await asyncio.sleep(0.2)
        before = gets(fake)
        again = await find(service)
        assert gets(fake) == before
        assert all(s["start"] != slot["start"] for s in again["slots"])  # the new booking is already in the window

    asyncio.run(scenario())
