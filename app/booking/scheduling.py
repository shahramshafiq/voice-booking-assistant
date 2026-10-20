"""The scheduling maths: working hours, breaks, buffers, existing events -> free appointment slots.

Everything here is a plain function with no network calls, so it is easy to test.
The rules (hours, breaks, buffer...) come in as one Rules object built from .env.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .timeparse import Preference

DAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass(frozen=True)
class Rules:
    tz: ZoneInfo
    hours: dict  # weekday number (0 = Monday) -> (opens, closes)
    breaks: list  # [(starts, ends)] every working day
    buffer_minutes: int  # free gap kept around every appointment
    step_minutes: int  # slots start on this grid, e.g. every 30 minutes
    max_advance_days: int


# ---------------------------------------------------------------------- reading the .env text
def _hhmm(text):
    hour, _, minute = text.strip().partition(":")
    return time(int(hour), int(minute))


def _range(text):
    start, _, end = text.strip().partition("-")
    first, last = _hhmm(start), _hhmm(end)
    if last <= first:
        raise ValueError(f"{text!r}: the end must be after the start")
    return first, last


def parse_business_hours(text):
    """'mon-fri 09:00-17:00;sat 10:00-14:00' -> {0: (9:00, 17:00), ..., 5: (10:00, 14:00)}."""
    hours = {}
    for rule in text.split(";"):
        days_text, _, times_text = rule.strip().partition(" ")
        first, _, last = days_text.lower().partition("-")
        last = last or first
        if first not in DAY_CODES or last not in DAY_CODES:
            raise ValueError(f"unknown day in {rule!r} (use mon, tue, wed, thu, fri, sat, sun)")
        opens, closes = _range(times_text)
        for day in range(DAY_CODES.index(first), DAY_CODES.index(last) + 1):
            hours[day] = (opens, closes)
    if not hours:
        raise ValueError("no working hours found")
    return hours


def parse_breaks(text):
    """'13:00-14:00,16:00-16:15' -> [(13:00, 14:00), (16:00, 16:15)]. 'none' -> []."""
    if text.strip().lower() == "none":
        return []
    return [_range(part) for part in text.split(",")]


# ---------------------------------------------------------------------- one day
def day_windows(day, rules):
    """The stretches of a day when appointments can happen: opening hours minus breaks."""
    hours = rules.hours.get(day.weekday())
    if hours is None:
        return []
    windows = [(datetime.combine(day, hours[0], tzinfo=rules.tz), datetime.combine(day, hours[1], tzinfo=rules.tz))]
    for break_start, break_end in sorted(rules.breaks):
        cut_start = datetime.combine(day, break_start, tzinfo=rules.tz)
        cut_end = datetime.combine(day, break_end, tzinfo=rules.tz)
        remaining = []
        for start, end in windows:
            if cut_end <= start or cut_start >= end:
                remaining.append((start, end))
                continue
            if start < cut_start:
                remaining.append((start, cut_start))
            if cut_end < end:
                remaining.append((cut_end, end))
        windows = remaining
    return windows


# ---------------------------------------------------------------------- existing events
def event_bounds(event, tz):
    """(start, end) of a Google Calendar event as aware datetimes, or None if it has no usable time."""
    start, end = event.get("start", {}), event.get("end", {})
    try:
        if "dateTime" in start and "dateTime" in end:
            return (datetime.fromisoformat(start["dateTime"]).astimezone(tz),
                    datetime.fromisoformat(end["dateTime"]).astimezone(tz))
        if "date" in start and "date" in end:  # an all-day event blocks the whole day
            return (datetime.combine(date.fromisoformat(start["date"]), time.min, tzinfo=tz),
                    datetime.combine(date.fromisoformat(end["date"]), time.min, tzinfo=tz))
    except ValueError:
        return None
    return None


def busy_intervals(events, rules, exclude_id=None):
    """Every existing event, made longer by the buffer on both sides."""
    pad = timedelta(minutes=rules.buffer_minutes)
    busy = []
    for event in events:
        if event.get("status") == "cancelled" or event.get("transparency") == "transparent":
            continue
        if exclude_id is not None and event.get("id") == exclude_id:
            continue
        bounds = event_bounds(event, rules.tz)
        if bounds is not None:
            busy.append((bounds[0] - pad, bounds[1] + pad))
    return busy


# ---------------------------------------------------------------------- slots
def slot_problem(start, end, busy, rules, now, min_notice_minutes):
    """None when the slot is bookable, otherwise a short reason code."""
    if start < now + timedelta(minutes=min_notice_minutes):
        return "TOO_SOON"
    if start > now + timedelta(days=rules.max_advance_days):
        return "TOO_FAR"
    if not any(w_start <= start and end <= w_end for w_start, w_end in day_windows(start.date(), rules)):
        return "OUTSIDE_HOURS"
    if any(start < b_end and end > b_start for b_start, b_end in busy):
        return "CONFLICT"
    return None


def _minutes(moment):
    return moment.hour * 60 + moment.minute


def _matches(start_time, pref):
    if pref is None:
        return True
    if pref.earliest is not None and start_time < pref.earliest:
        return False
    if pref.latest is not None and start_time > pref.latest:
        return False
    return True


def free_slots(day, duration_minutes, busy, rules, now, min_notice_minutes, pref: Optional[Preference] = None):
    """All free (start, end) pairs on one day, earliest first (or closest to pref.target first)."""
    length = timedelta(minutes=duration_minutes)
    step = timedelta(minutes=rules.step_minutes)
    slots = []
    for window_start, window_end in day_windows(day, rules):
        start = window_start
        while start + length <= window_end:
            end = start + length
            if _matches(start.timetz().replace(tzinfo=None), pref) and \
                    slot_problem(start, end, busy, rules, now, min_notice_minutes) is None:
                slots.append((start, end))
            start += step
    if pref is not None and pref.target is not None:
        slots.sort(key=lambda s: abs(_minutes(s[0]) - _minutes(pref.target)))
    return slots


def pick_slots(days_with_slots, limit):
    """Choose up to `limit` slots. Over several days, at most two per day so the options are spread out."""
    days_with_slots = [(day, slots) for day, slots in days_with_slots if slots]
    per_day = limit if len(days_with_slots) <= 1 else 2
    chosen = []
    for _, slots in days_with_slots:
        chosen.extend(slots[:per_day])
    return sorted(chosen[:limit], key=lambda slot: slot[0])