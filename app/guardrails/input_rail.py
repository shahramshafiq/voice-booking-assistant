"""The input guardrail: checks what the caller said BEFORE the model sees it.

It is plain code (no extra model call), so it adds no noticeable delay. It can:
  - block a message that tries to change the assistant's rules or asks for secrets;
  - block a message that is far too long;
  - remove a credit card number, so it never reaches the model, the logs or the history.
A blocked message gets a fixed spoken reply and the model is not called at all.
"""
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("guardrail.input")

CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


@dataclass
class InputVerdict:
    action: str  # "allow", "block" or "redact"
    text: str  # what to give the model (already cleaned)
    reply: str = ""  # what to say instead, when blocked
    reason: str = ""


def luhn_ok(digits):
    """The check that real card numbers pass; it separates them from phone numbers."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number = number * 2 - 9 if number > 4 else number * 2
        total += number
    return total % 10 == 0


class InputRail:
    def __init__(self, settings):
        self.settings = settings
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in settings.input_block_patterns]

    def check(self, text):
        text = text.strip()
        if len(text) > self.settings.input_max_chars:
            log.warning("input blocked: too long (%d characters)", len(text))
            return InputVerdict("block", "", self.settings.input_too_long_reply, "too_long")

        for pattern in self.patterns:
            if pattern.search(text):
                log.warning("input blocked: matched %r", pattern.pattern[:40])
                return InputVerdict("block", "", self.settings.input_block_reply, "injection")

        redacted = CARD_NUMBER.sub(self._remove_if_card, text)
        if redacted != text:
            log.warning("input changed: a card number was removed")
            return InputVerdict("redact", redacted, reason="card_number")
        return InputVerdict("allow", text)

    @staticmethod
    def _remove_if_card(match):
        digits = re.sub(r"\D", "", match.group(0))
        return "[card number removed]" if 13 <= len(digits) <= 19 and luhn_ok(digits) else match.group(0)