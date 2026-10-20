"""A stopwatch for one conversation turn.

Every event is stored with the time it first happened. The summary reports each event
in milliseconds relative to the moment the turn was confirmed ("committed"), so
negative numbers mean "this happened earlier, while the system was guessing ahead".
"""
import time


class TurnTimer:
    def __init__(self):
        self.marks = {}

    def mark(self, name):
        """Record an event. Only the FIRST time counts."""
        if name not in self.marks:
            self.marks[name] = time.perf_counter()

    def has(self, name):
        return name in self.marks

    def summary(self):
        start = self.marks.get("committed")
        if start is None:
            return {}
        return {
            f"{name}_ms": round((at - start) * 1000)
            for name, at in sorted(self.marks.items(), key=lambda item: item[1])
            if name != "committed"
        }