"""The only file that talks to Google Calendar.

It uses the access token pasted into .env. The token is put in one request header and is
never logged, never returned to the model and never put in an error message.
Every failure becomes a small, clear CalendarError so the rest of the app never sees raw HTTP.
"""
import asyncio
import logging
import time
from contextlib import suppress
from urllib.parse import quote

import httpx

log = logging.getLogger("calendar")

# Only ask Google for the fields we use: smaller answers arrive faster.
EVENT_FIELDS = "nextPageToken,items(id,status,transparency,start,end,extendedProperties/private)"


class CalendarError(Exception):
    code = "CALENDAR_ERROR"
    retryable = False

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CalendarAuthError(CalendarError):
    code = "CALENDAR_AUTH_FAILED"
    hint = "Paste a fresh token into .env and restart the server."  # advice for you, never shown to the model or the caller


class CalendarNotFound(CalendarError):
    code = "CALENDAR_NOT_FOUND"


class CalendarUnavailable(CalendarError):
    code = "CALENDAR_UNAVAILABLE"
    retryable = True


class CalendarRejected(CalendarError):
    code = "CALENDAR_REJECTED"


class CalendarClient:
    def __init__(self, settings, transport=None):
        """`transport` is only used by the tests to replace the real internet with a fake Google."""
        self.settings = settings
        self.calendar_path = f"/calendars/{quote(settings.google_calendar_id, safe='')}/events"
        self.http = httpx.AsyncClient(
            base_url=settings.google_calendar_base_url,
            headers={"Authorization": f"Bearer {settings.google_calendar_api_token}"},
            timeout=settings.calendar_timeout_seconds,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=settings.calendar_keepalive_expiry_seconds),
            transport=transport,
        )
        self._cache = {}  # query -> (time, events)

    # ---------------------------------------------------------------- the one place that sends requests
    async def _request(self, method, path, params=None, body=None, retry=True):
        attempts = 1 + (self.settings.calendar_max_retries if retry else 0)
        last_error = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                response = await self.http.request(method, path, params=params, json=body)
            except httpx.TimeoutException:
                last_error = CalendarUnavailable("Google Calendar did not answer in time.")
            except httpx.TransportError as exc:
                last_error = CalendarUnavailable(f"Could not reach Google Calendar ({type(exc).__name__}).")
            else:
                log.info("calendar %s -> %d (%.0f ms)", method, response.status_code, (time.perf_counter() - started) * 1000)
                if response.status_code < 300:
                    return response.json() if response.content else {}
                last_error = self._error_for(response)
            if not last_error.retryable or attempt == attempts:
                break
            log.warning("calendar request failed (%s), retrying", last_error.code)
            await asyncio.sleep(0.3 * attempt)
        raise last_error

    @staticmethod
    def _error_for(response):
        """Turn an HTTP error into the right CalendarError. Only the status and reason are used, never the headers."""
        status = response.status_code
        reason = ""
        with suppress(Exception):
            reason = response.json()["error"]["errors"][0]["reason"]
        if status == 401:
            return CalendarAuthError("The Google Calendar token is expired or invalid.")
        if status == 403 and reason in ("rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"):
            return CalendarUnavailable("Google Calendar is busy, too many requests.")
        if status == 403:
            return CalendarAuthError("The Google Calendar token is not allowed to use this calendar (missing permission or wrong calendar).")
        if status in (404, 410):
            return CalendarNotFound("Google Calendar could not find that calendar or event.")
        if status == 429 or status >= 500:
            return CalendarUnavailable(f"Google Calendar had a problem (status {status}).")
        return CalendarRejected(f"Google Calendar rejected the request (status {status}, {reason or 'no reason given'}).")

    # ---------------------------------------------------------------- reading
    async def list_events(self, time_min=None, time_max=None, private_props=None, fresh=False):
        """Events in a time window and/or with matching private properties.

        Answers are kept for CALENDAR_CACHE_SECONDS unless fresh=True (used right before any change).
        """
        params = [("singleEvents", "true"), ("orderBy", "startTime"), ("maxResults", "250"), ("fields", EVENT_FIELDS)]
        if time_min is not None:
            params.append(("timeMin", time_min.isoformat()))
        if time_max is not None:
            params.append(("timeMax", time_max.isoformat()))
        for name, value in sorted((private_props or {}).items()):
            params.append(("privateExtendedProperty", f"{name}={value}"))

        key = tuple(params)
        cached = self._cache.get(key)
        if not fresh and cached and time.monotonic() - cached[0] < self.settings.calendar_cache_seconds:
            log.debug("calendar answer taken from memory")
            return cached[1]

        events = []
        page_token = None
        for _ in range(5):  # at most five pages, far more than one day or one booking needs
            page_params = params + ([("pageToken", page_token)] if page_token else [])
            data = await self._request("GET", self.calendar_path, params=page_params)
            events.extend(data.get("items", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        now = time.monotonic()
        self._cache = {k: v for k, v in self._cache.items() if now - v[0] < self.settings.calendar_cache_seconds}
        self._cache[key] = (now, events)
        return events

    async def check_access(self):
        """A tiny request that proves the token and the calendar ID work. Also warms the connection."""
        params = [("maxResults", "1"), ("fields", "items(id)")]
        await self._request("GET", self.calendar_path, params=params, retry=False)

    # ---------------------------------------------------------------- changing
    async def insert_event(self, body):
        # Not retried automatically: a retry after a timeout could create the event twice.
        # The booking service checks whether it was created before it reports a failure.
        event = await self._request("POST", self.calendar_path, body=body, retry=False)
        self._cache.clear()
        return event

    async def patch_event(self, event_id, body):
        event = await self._request("PATCH", f"{self.calendar_path}/{quote(event_id, safe='')}", body=body)
        self._cache.clear()
        return event

    async def delete_event(self, event_id):
        try:
            await self._request("DELETE", f"{self.calendar_path}/{quote(event_id, safe='')}")
        except CalendarNotFound:
            pass  # already gone: the goal (no such appointment) is reached
        self._cache.clear()

    async def close(self):
        await self.http.aclose()