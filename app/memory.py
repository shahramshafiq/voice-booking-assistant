"""Short-term memory for ONE conversation. It starts empty and is thrown away when the call ends.

Two kinds of memory live here:
  1. the conversation itself: the last few things the caller said and the assistant answered;
  2. the facts of the booking being worked on: name, phone, times that were offered, a booking
     that was looked up, an action that is waiting for the caller's "yes".
The second kind is written by the CODE (from tool calls and results), never by the model, and it is
shown to the model again at every question, so nothing important is forgotten between turns.
"""
import time
from dataclasses import dataclass

from .booking.speech import say_time


@dataclass
class Pending:
    """An action the caller has been asked to confirm."""
    key: str  # identifies exactly this action
    tool: str
    question: str  # what was read back to the caller
    turn_no: int  # the conversation turn in which it was asked
    created: float


class SessionMemory:
    def __init__(self, settings):
        self.settings = settings
        self.messages = []  # [{"role": "user"/"assistant", "content": "..."}]
        self.turn_no = 0  # counts the turns that were really committed
        self.pending = None
        self.verify_failures = 0  # wrong reference/phone attempts in this conversation
        self.known_refs = set()  # booking references that really appeared in this conversation
        self.achieved = set()  # tools that really succeeded in this conversation (the output guardrail checks claims against it)
        self.customer_name = None
        self.customer_phone = None
        self.service_name = None
        self.offered_slots = []
        self.verified_booking = None
        self.recent_actions = []
        self.calendar_problem = None  # set when the calendar could not be reached at the start

    # ---------------------------------------------------------------- the conversation
    def add_exchange(self, user_text, assistant_text):
        self.messages.append({"role": "user", "content": user_text})
        if assistant_text.strip():
            self.messages.append({"role": "assistant", "content": assistant_text.strip()})
        self.messages = self.messages[-self.settings.history_max_messages:]
        while self.messages and self.messages[0]["role"] != "user":
            self.messages.pop(0)  # never start with an orphan assistant message

    # ---------------------------------------------------------------- the confirmation step
    def set_pending(self, key, tool, question):
        self.pending = Pending(key, tool, question, self.turn_no, time.monotonic())

    def pending_is_confirmable(self, key, current_turn):
        """True when `key` is the action asked about in the turn right before this one, and the question is not too old."""
        p = self.pending
        return (
            p is not None
            and p.key == key
            and p.turn_no == current_turn - 1
            and time.monotonic() - p.created < self.settings.confirm_ttl_seconds
        )

    # ---------------------------------------------------------------- facts from tool calls
    def note_result(self, tool, args, result):
        """Remember what a tool call taught us."""
        reference = result.get("booking_reference") or getattr(args, "booking_reference", None)
        if reference:
            self.known_refs.add("".join(ch for ch in str(reference) if ch.isdigit()))

        if tool == "create_booking":
            self.customer_name = args.customer_details.name.strip()
            self.customer_phone = args.customer_details.phone.strip()

        slots = result.get("slots") or result.get("alternatives")
        if slots and tool != "get_services":
            self.offered_slots = slots[:6]
        if tool == "find_available_slots" and result.get("ok"):
            self.service_name = result["service"]["name"]

        if not result.get("ok"):
            return
        if tool in ("get_booking", "create_booking", "reschedule_booking"):
            self.verified_booking = {key: result.get(key) for key in ("booking_reference", "service", "label", "customer_name")}
        if tool in ("create_booking", "reschedule_booking", "cancel_booking"):
            verb = {"create_booking": "booked", "reschedule_booking": "rescheduled", "cancel_booking": "cancelled"}[tool]
            self.recent_actions.append(f"{verb} {result.get('service')} {result.get('label')}, reference {result.get('booking_reference')}")
            self.recent_actions = self.recent_actions[-3:]
        if tool == "cancel_booking":
            self.verified_booking = None

    # ---------------------------------------------------------------- shown to the model at every question
    def render_state(self, now, timezone_name):
        lines = [
            "SESSION STATE (facts kept by the system during this call; trust them):",
            f"Now: {now.strftime('%A')} {now.day} {now.strftime('%B %Y')}, {say_time(now)} ({timezone_name}).",
        ]
        if self.calendar_problem:
            lines.append(f"Calendar: NOT REACHABLE ({self.calendar_problem}). Tell the caller you cannot use the calendar right now.")
        if self.customer_name or self.customer_phone:
            lines.append(f"Customer so far: name {self.customer_name or 'unknown'}, phone {self.customer_phone or 'unknown'}.")
        if self.service_name:
            lines.append(f"Service being discussed: {self.service_name}.")
        if self.offered_slots:
            lines.append("Times offered most recently (use these exact start and end values when booking):")
            for number, slot in enumerate(self.offered_slots, start=1):
                lines.append(f" {number}. {slot['label']} (start {slot['start']}, end {slot['end']})")
        if self.verified_booking:
            b = self.verified_booking
            lines.append(f"Verified booking: reference {b['booking_reference']}, {b['service']}, {b['label']}, for {b['customer_name']}.")
        if self.pending:
            lines.append(f"Waiting for the caller to say yes or no to: {self.pending.question}")
        if self.recent_actions:
            lines.append("Actions completed in this call: " + "; ".join(self.recent_actions) + ".")
        return "\n".join(lines)