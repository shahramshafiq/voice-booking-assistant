"""Turns dates, times and numbers into words that sound natural when a voice reads them aloud."""


def say_time(moment):
    """15:00 -> '3 PM', 15:30 -> '3:30 PM'."""
    hour = moment.hour % 12 or 12
    suffix = "AM" if moment.hour < 12 else "PM"
    if moment.minute == 0:
        return f"{hour} {suffix}"
    return f"{hour}:{moment.minute:02d} {suffix}"


def say_day(day, today):
    """Today, tomorrow, or 'Tuesday 21 October'."""
    if day == today:
        return "today"
    if (day - today).days == 1:
        return "tomorrow"
    return f"{day.strftime('%A')} {day.day} {day.strftime('%B')}"


def say_slot(start, today):
    """'tomorrow at 3 PM'."""
    return f"{say_day(start.date(), today)} at {say_time(start)}"


def spell_digits(digits):
    """'03001234567' -> '0 3 0, 0 1 2, 3 4 5, 6 7' (groups of three, one digit at a time)."""
    groups = [digits[i:i + 3] for i in range(0, len(digits), 3)]
    return ", ".join(" ".join(group) for group in groups)