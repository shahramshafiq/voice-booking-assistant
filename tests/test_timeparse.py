"""Tests for the date and time understanding (no network, no API keys)."""
from datetime import date, time

import pytest

from app.booking.timeparse import Preference, parse_clock, parse_date, parse_date_range, parse_time_preference

TODAY = date(2026, 10, 19)  # a Monday


def test_relative_dates():
    assert parse_date("today", TODAY) == TODAY
    assert parse_date("Tomorrow", TODAY) == date(2026, 10, 20)
    assert parse_date("day after tomorrow", TODAY) == date(2026, 10, 21)


def test_weekdays():
    assert parse_date("friday", TODAY) == date(2026, 10, 23)
    assert parse_date("next friday", TODAY) == date(2026, 10, 23)
    assert parse_date("monday", TODAY) == date(2026, 10, 26)  # a Monday said on a Monday means next week
    assert parse_date("this monday", TODAY) == TODAY


def test_written_dates():
    assert parse_date("2026-11-02", TODAY) == date(2026, 11, 2)
    assert parse_date("21 October", TODAY) == date(2026, 10, 21)
    assert parse_date("October 21st", TODAY) == date(2026, 10, 21)
    assert parse_date("the 5th of january", TODAY) == date(2027, 1, 5)  # already past this year
    assert parse_date("Tuesday 21 October", TODAY) == date(2026, 10, 21)  # exactly how the assistant says dates
    assert parse_date("Friday the 23rd", TODAY) == date(2026, 10, 23)
    assert parse_date("the 21st", TODAY) == date(2026, 10, 21)
    assert parse_date("the 5th", TODAY) == date(2026, 11, 5)  # the 5th of this month has passed


def test_bad_dates():
    for text in ("someday", "2026-02-30", "31 february"):
        with pytest.raises(ValueError):
            parse_date(text, TODAY)


def test_date_ranges():
    assert parse_date_range("this week", None, TODAY) == (TODAY, date(2026, 10, 25))
    assert parse_date_range("next week", None, TODAY) == (date(2026, 10, 26), date(2026, 11, 1))
    assert parse_date_range("tomorrow", "friday", TODAY) == (date(2026, 10, 20), date(2026, 10, 23))
    with pytest.raises(ValueError):
        parse_date_range("friday", "tuesday", TODAY)


def test_clock_times():
    assert parse_clock("3 pm") == time(15, 0)
    assert parse_clock("3:30PM") == time(15, 30)
    assert parse_clock("15:00") == time(15, 0)
    assert parse_clock("9 am") == time(9, 0)
    assert parse_clock("12 am") == time(0, 0)
    assert parse_clock("noon") == time(12, 0)
    assert parse_clock("3") == time(15, 0)  # nobody books at 3 AM
    assert parse_clock("10") == time(10, 0)
    for text in ("25:00", "13 pm", "banana"):
        with pytest.raises(ValueError):
            parse_clock(text)


def test_time_preferences():
    assert parse_time_preference("") is None
    assert parse_time_preference("morning") == Preference(time(0, 0), time(11, 59), None)
    assert parse_time_preference("tomorrow afternoon") == Preference(time(12, 0), time(16, 59), None)
    assert parse_time_preference("after 3 PM") == Preference(time(15, 0), None, None)
    assert parse_time_preference("before 11") == Preference(None, time(10, 59), None)
    assert parse_time_preference("afternoon after 3") == Preference(time(15, 0), time(16, 59), None)
    assert parse_time_preference("around 2 pm") == Preference(None, None, time(14, 0))
    assert parse_time_preference("3 pm") == Preference(None, None, time(15, 0))
    with pytest.raises(ValueError):
        parse_time_preference("whenever")
