"""A tiny fake Google Calendar for the tests (the real internet is never used).

It understands just what the app uses: list, insert, patch and delete events, the access token
check, the privateExtendedProperty filter, and a way to make the next requests fail.
"""
import json
import re
from datetime import datetime
from itertools import count
from urllib.parse import unquote

import httpx


class FakeGoogleCalendar:
    def __init__(self, token="ya29.fake-test-token"):
        self.token = token
        self.events = {}  # id -> event dict
        self.requests = []  # (method, path) of every request, to count calls
        self.queries = []  # the query parameters of every request
        self.fail_next = []  # list of (status, lose_answer) to inject, used one per request
        self.fail_always = {}  # HTTP method -> status that every request of that kind gets
        self._ids = count(1)

    # ---------------------------------------------------------------- helpers for the tests
    def add_event(self, start_iso, end_iso, event_id=None, **extra):
        event_id = event_id or f"seed{next(self._ids)}"
        self.events[event_id] = {"id": event_id, "status": "confirmed", "start": {"dateTime": start_iso}, "end": {"dateTime": end_iso}, **extra}
        return event_id

    def transport(self):
        return httpx.MockTransport(self._handle)

    # ---------------------------------------------------------------- the fake server
    def _handle(self, request):
        path = unquote(request.url.path)
        self.requests.append((request.method, path))
        self.queries.append(dict(request.url.params))
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return httpx.Response(401, json={"error": {"code": 401, "errors": [{"reason": "authError"}]}})
        if request.method in self.fail_always:  # e.g. {"PATCH": 500}: every update fails
            status = self.fail_always[request.method]
            return httpx.Response(status, json={"error": {"code": status, "errors": [{"reason": "backendError"}]}})
        if self.fail_next:
            status, lose_answer = self.fail_next.pop(0)
            if lose_answer:  # the change is made, but the answer never arrives
                self._apply(request, path)
                raise httpx.ReadTimeout("answer lost", request=request)
            return httpx.Response(status, json={"error": {"code": status, "errors": [{"reason": "backendError"}]}})
        return self._apply(request, path)

    def _apply(self, request, path):
        match = re.search(r"/events(?:/([^/]+))?$", path)
        if not match:
            return httpx.Response(404, json={"error": {"code": 404, "errors": [{"reason": "notFound"}]}})
        event_id = match.group(1)
        body = json.loads(request.content) if request.content else {}

        if request.method == "GET" and event_id is None:
            return httpx.Response(200, json={"items": self._list(request.url.params)})
        if request.method == "POST" and event_id is None:
            new_id = f"ev{next(self._ids)}"
            self.events[new_id] = {"id": new_id, "status": "confirmed", **body}
            return httpx.Response(200, json=self.events[new_id])
        if event_id not in self.events:
            return httpx.Response(404, json={"error": {"code": 404, "errors": [{"reason": "notFound"}]}})
        if request.method == "PATCH":
            self.events[event_id].update(body)
            return httpx.Response(200, json=self.events[event_id])
        if request.method == "DELETE":
            del self.events[event_id]
            return httpx.Response(204)
        return httpx.Response(400, json={"error": {"code": 400, "errors": [{"reason": "badRequest"}]}})

    def _list(self, params):
        time_min = datetime.fromisoformat(params["timeMin"]) if "timeMin" in params else None
        time_max = datetime.fromisoformat(params["timeMax"]) if "timeMax" in params else None
        wanted = dict(item.split("=", 1) for item in params.get_list("privateExtendedProperty"))
        found = []
        for event in self.events.values():
            if wanted:
                private = event.get("extendedProperties", {}).get("private", {})
                if any(private.get(name) != value for name, value in wanted.items()):
                    continue
            start = datetime.fromisoformat(event["start"]["dateTime"])
            end = datetime.fromisoformat(event["end"]["dateTime"])
            if (time_min and end <= time_min) or (time_max and start >= time_max):
                continue
            found.append(event)
        return sorted(found, key=lambda e: e["start"]["dateTime"])
