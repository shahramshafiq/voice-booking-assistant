"""Tests for the slot maths: hours, breaks, buffers, existing events (no network, no API keys)."""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from app.booking.scheduling import (
    Rules, busy_intervals, day_windows, free_slots, parse_breaks, parse_business_hours, pick_slots, slot_problem,
)
from app.booking.timeparse import Preference

TZ = ZoneInfo("Asia/Karachi")
RULES = Rules(
    tz=TZ,
    hours=parse_business_hours("mon-fri 09:00-17:00;sat 10:00-14:00"),
    breaks=parse_breaks("13:00-14:00"),
    buffer_minutes=10,
    step_minutes=30,
    max_advance_days=30,
)
NOW = datetime(2026, 10, 19, 8, 0, tzinfo=TZ)  # Monday 8 AM
TUESDAY = date(2026, 10, 20)


def at(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def event(day, start_hour, end_hour, event_id="e1", **extra):
    return {
        "id": event_id,
        "start": {"dateTime": at(day, start_hour).isoformat()},
        "end": {"dateTime": at(day, end_hour).isoformat()},
        **extra,
    }


def test_parse_hours_and_breaks():
    assert RULES.hours[0] == (time(9, 0), time(17, 0))
    assert RULES.hours[5] == (time(10, 0), time(14, 0))
    assert 6 not in RULES.hours  # Sunday is closed
    assert parse_breaks("none") == []
    for bad in ("funday 09:00-17:00", "mon 17:00-09:00"):
        with pytest.raises(ValueError):
            parse_business_hours(bad)


def test_windows_skip_the_break():
    windows = day_windows(TUESDAY, RULES)
    assert [(s.hour, e.hour) for s, e in windows] == [(9, 13), (14, 17)]
    assert day_windows(date(2026, 10, 25), RULES) == []  # Sunday


def test_empty_day_gives_every_slot_that_fits():
    slots = free_slots(TUESDAY, 60, [], RULES, NOW, 60)
    starts = [s.strftime("%H:%M") for s, _ in slots]
    assert starts[0] == "09:00"
    assert "12:00" in starts and "12:30" not in starts  # a 60 minute slot at 12:30 would run into the break
    assert starts[-1] == "16:00"


def test_buffer_pushes_the_next_slot_away():
    busy = busy_intervals([event(20, 10, 11)], RULES)
    starts = [s.strftime("%H:%M") for s, _ in free_slots(TUESDAY, 30, busy, RULES, NOW, 60)]
    assert "09:00" in starts       # ends 09:30, the busy time starts 09:50
    assert "09:30" not in starts   # ends 10:00, which is inside the buffer that starts at 09:50
    assert "10:00" not in starts and "10:30" not in starts
    assert "11:00" not in starts   # starts inside the buffer after 11:00
    assert "11:30" in starts


def test_cancelled_free_and_excluded_events_do_not_block():
    events = [
        event(20, 10, 11, event_id="a", status="cancelled"),
        event(20, 11, 12, event_id="b", transparency="transparent"),
        event(20, 14, 15, event_id="mine"),
    ]
    busy = busy_intervals(events, RULES, exclude_id="mine")
    assert busy == []


def test_all_day_event_blocks_the_day():
    holiday = {"id": "h", "start": {"date": "2026-10-20"}, "end": {"date": "2026-10-21"}}
    assert free_slots(TUESDAY, 30, busy_intervals([holiday], RULES), RULES, NOW, 60) == []


def test_slot_problems():
    busy = busy_intervals([event(20, 10, 11)], RULES)
    assert slot_problem(at(20, 15), at(20, 16), busy, RULES, NOW, 60) is None
    assert slot_problem(at(20, 10, 30), at(20, 11, 30), busy, RULES, NOW, 60) == "CONFLICT"
    assert slot_problem(at(20, 12, 30), at(20, 13, 30), busy, RULES, NOW, 60) == "OUTSIDE_HOURS"  # runs into the break
    assert slot_problem(at(20, 8), at(20, 9), busy, RULES, NOW, 60) == "OUTSIDE_HOURS"
    assert slot_problem(at(19, 8, 30), at(19, 9, 30), [], RULES, NOW, 60) == "TOO_SOON"
    assert slot_problem(datetime(2026, 12, 15, 10, tzinfo=TZ), datetime(2026, 12, 15, 11, tzinfo=TZ), [], RULES, NOW, 60) == "TOO_FAR"


def test_minimum_notice_is_per_call():
    starts = [s.strftime("%H:%M") for s, _ in free_slots(date(2026, 10, 19), 60, [], RULES, NOW, 180)]
    assert starts[0] == "11:00"  # 8:00 now + 3 hours notice


def test_preferences():
    afternoon = Preference(time(12, 0), time(16, 59), None)
    starts = [s.strftime("%H:%M") for s, _ in free_slots(TUESDAY, 60, [], RULES, NOW, 60, afternoon)]
    assert starts[0] == "12:00" and starts[-1] == "16:00"  # 12:00 to 13:00 fits just before the break
    assert "12:30" not in starts and "13:00" not in starts
    after_three = Preference(time(15, 0), None, None)
    starts = [s.strftime("%H:%M") for s, _ in free_slots(TUESDAY, 60, [], RULES, NOW, 60, after_three)]
    assert starts == ["15:00", "15:30", "16:00"]
    near_noon = Preference(None, None, time(12, 0))
    first = free_slots(TUESDAY, 60, [], RULES, NOW, 60, near_noon)[0][0]
    assert first.strftime("%H:%M") == "12:00"


def test_pick_slots_spreads_over_days():
    one_day = [(TUESDAY, free_slots(TUESDAY, 60, [], RULES, NOW, 60))]
    assert len(pick_slots(one_day, 4)) == 4
    wednesday = date(2026, 10, 21)
    two_days = one_day + [(wednesday, free_slots(wednesday, 60, [], RULES, NOW, 60))]
    picked = pick_slots(two_days, 4)
    assert [s.day for s, _ in picked] == [20, 20, 21, 21]
