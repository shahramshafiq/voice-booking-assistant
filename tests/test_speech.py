"""Tests for the spoken form of dates and numbers and the plain form shown on the page."""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from app.booking.speech import say_day, say_slot, say_time, spell_digits, tidy_numbers


def test_times_and_days_are_said_the_way_people_say_them():
    assert [say_time(time(h, m)) for h, m in [(15, 0), (15, 30), (9, 5), (12, 0), (0, 0)]] == ["3 PM", "3:30 PM", "9:05 AM", "12 PM", "12 AM"]
    today = date(2026, 10, 19)
    assert say_day(today, today) == "today" and say_day(date(2026, 10, 20), today) == "tomorrow"
    assert say_day(date(2026, 10, 23), today) == "Friday 23 October"
    assert say_slot(datetime(2026, 10, 20, 15, 0, tzinfo=ZoneInfo("Asia/Karachi")), today) == "tomorrow at 3 PM"


def test_digits_are_spelled_in_groups_for_the_voice():
    assert spell_digits("03001234567") == "0 3 0, 0 1 2, 3 4 5, 6 7"
    assert spell_digits("550781") == "5 5 0, 7 8 1"


def test_spoken_digits_become_plain_numbers_for_the_page():
    assert tidy_numbers("phone number 0 3 0, 0 1 2, 3 4 5, 6 7. Shall I book it?") == "phone number 0300 1234567. Shall I book it?"
    assert tidy_numbers("Your booking reference is 5 5 0, 7 8 1.") == "Your booking reference is 550781."
    assert tidy_numbers("Your reference is 5 5 0 7 8 1 today") == "Your reference is 550781 today"
    assert tidy_numbers(spell_digits("1234567890")) == "1234567890"
    assert tidy_numbers(spell_digits("0092300123456")) == "0092300123456"


def test_ordinary_numbers_are_left_alone():
    for text in ("I have 3 PM or 4 PM tomorrow.", "Options 2, 3, 4, 5 are free.", "On the 21st at 9 30.", "We open at 9 and close at 5.",
                 "A 60 minute consultation costs 5000 rupees.", "Pick 1, 2 or 3."):
        assert tidy_numbers(text) == text
