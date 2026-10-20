"""Understands the dates and times people say: "tomorrow", "next friday", "after 3 PM", "afternoon".

The model passes the caller's own words to the tools. This file turns them into real dates and
times, using the business clock, so the date arithmetic never depends on the model.
Every function raises ValueError with a plain message when it cannot understand the text.
"""
import re
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Optional

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
WEEKDAY_WORDS = "|".join(WEEKDAYS)
MONTH_WORDS = "|".join(MONTHS)

CLOCK = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)"


@dataclass(frozen=True)
class Preference:
    """Limits on when an appointment may START. None means 'no limit'."""
    earliest: Optional[time] = None
    latest: Optional[time] = None
    target: Optional[time] = None  # the time the caller is aiming for: closest slots first


# ---------------------------------------------------------------------- dates
def parse_date(text, today):
    """'2026-10-21', 'tomorrow', 'next friday', '21 october', 'october 21st' -> a date."""
    t = re.sub(r"[,.]", " ", str(text).lower())
    t = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", t)
    t = re.sub(r"\b(the|of|on)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # "tuesday 21 october" and "friday the 23rd": the weekday is only decoration when a date follows
    t = re.sub(rf"^(?:{WEEKDAY_WORDS}) (?=\d|(?:{MONTH_WORDS})\b)", "", t)

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return date.fromisoformat(t)
        except ValueError:
            raise ValueError(f"{text!r} is not a real date")
    if t == "today":
        return today
    if t == "tomorrow":
        return today + timedelta(days=1)
    if t in ("day after tomorrow", "the day after tomorrow"):
        return today + timedelta(days=2)

    m = re.fullmatch(rf"(?:(next|this|coming) )?({WEEKDAY_WORDS})", t)
    if m:
        target = WEEKDAYS[m.group(2)]
        ahead = (target - today.weekday()) % 7
        if ahead == 0 and m.group(1) != "this":
            ahead = 7  # "monday" said on a Monday means the next one
        return today + timedelta(days=ahead)

    if re.fullmatch(r"\d{1,2}", t):  # "the 21st": the next 21st
        for months_ahead in range(3):
            month_index = today.month - 1 + months_ahead
            try:
                result = date(today.year + month_index // 12, month_index % 12 + 1, int(t))
            except ValueError:
                continue
            if result >= today:
                return result
        raise ValueError(f"{text!r} is not a real date")

    m = re.fullmatch(rf"(\d{{1,2}}) ({MONTH_WORDS})(?: (\d{{4}}))?", t)
    if m:
        day, month, year = int(m.group(1)), MONTHS[m.group(2)], m.group(3)
    else:
        m = re.fullmatch(rf"({MONTH_WORDS}) (\d{{1,2}})(?: (\d{{4}}))?", t)
        if not m:
            raise ValueError(f"I could not understand the date {text!r}")
        month, day, year = MONTHS[m.group(1)], int(m.group(2)), m.group(3)
    try:
        result = date(int(year) if year else today.year, month, day)
        if not year and result < today:
            result = date(today.year + 1, month, day)  # "5 january" said in October means next year
        return result
    except ValueError:
        raise ValueError(f"{text!r} is not a real date")


def parse_date_range(start_text, end_text, today):
    """Returns (first_day, last_day). Also understands 'this week' and 'next week'."""
    start_clean = str(start_text).strip().lower()
    if start_clean == "this week":
        return today, today + timedelta(days=6 - today.weekday())
    if start_clean == "next week":
        monday = today + timedelta(days=7 - today.weekday())
        return monday, monday + timedelta(days=6)
    first = parse_date(start_text, today)
    last = parse_date(end_text, today) if end_text else first
    if last < first:
        raise ValueError("the end date is before the start date")
    return first, last


# ---------------------------------------------------------------------- clock times
def parse_clock(text):
    """'3 pm', '3:30pm', '15:00', 'noon' -> a time. A bare 1 to 7 is taken as the afternoon."""
    t = str(text).strip().lower().replace(".", "")
    if t == "noon":
        return time(12, 0)
    if t == "midnight":
        return time(0, 0)
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if not m:
        raise ValueError(f"I could not understand the time {text!r}")
    hour, minute, suffix = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if minute > 59 or hour > 23 or (suffix and not 1 <= hour <= 12):
        raise ValueError(f"{text!r} is not a real time")
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif suffix is None and 1 <= hour <= 7:
        # ponytail: a bare "3" is guessed as 3 PM because nobody books at 3 AM; the model also confirms the time
        hour += 12
    return time(hour, minute)


def _minus_one_minute(moment):
    total = moment.hour * 60 + moment.minute - 1
    return time(max(total, 0) // 60, max(total, 0) % 60)


def parse_time_preference(text):
    """'afternoon', 'after 3 PM', 'before 11', 'around 2 pm' -> Preference. Empty text -> None."""
    if not text or not str(text).strip():
        return None
    t = str(text).lower().replace(".", "")
    earliest = latest = target = None
    found = False

    if "morning" in t:
        earliest, latest, found = time(0, 0), time(11, 59), True
    elif "afternoon" in t:
        earliest, latest, found = time(12, 0), time(16, 59), True
    elif "evening" in t:
        earliest, latest, found = time(17, 0), time(23, 59), True

    after = re.search(rf"\b(?:after|from|past|later than)\s+{CLOCK}", t)
    before = re.search(rf"\b(?:before|earlier than|until|by)\s+{CLOCK}", t)
    if after:
        clock = parse_clock(after.group(1))
        earliest = max(earliest, clock) if earliest else clock
        found = True
    if before:
        clock = _minus_one_minute(parse_clock(before.group(1)))
        latest = min(latest, clock) if latest else clock
        found = True
    if not after and not before:
        around = re.search(rf"\b(?:around|about|at|near)?\s*{CLOCK}\b", t)
        if around and re.search(r"\d|noon|midnight", around.group(1)):
            target = parse_clock(around.group(1))
            found = True

    if not found:
        raise ValueError(f"I could not understand the time preference {text!r}")
    return Preference(earliest, latest, target)