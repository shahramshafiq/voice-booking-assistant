"""Tests that LLMClient reads OpenAI's streaming events correctly, using the real OpenAI library
against a fake server (no internet, no API key)."""
import asyncio
import json

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from app.config import settings
from app.llm import LLMClient
from app.tools import build_definitions

RESPONSE = {"id": "resp_1", "object": "response", "created_at": 1, "status": "in_progress", "model": "gpt-4.1-mini",
            "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
CALL = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "find_available_slots",
        "arguments": "{\"service_id\":\"consultation\"}", "status": "completed"}
MESSAGE = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
           "content": [{"type": "output_text", "text": "Let me look. ", "annotations": []}]}


def server_sent(events):
    body = ""
    for number, event in enumerate(events):
        event = dict(event, sequence_number=number)
        body += f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
    return body.encode()


def make_client(events):
    """An LLMClient whose network is replaced by a fake that answers with `events` and remembers the request."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=server_sent(events))

    llm = LLMClient(settings)
    llm.client = AsyncOpenAI(api_key="test", base_url="http://fake/v1",
                             http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return llm, seen


async def collect(llm, **options):
    return [event async for event in llm.stream("be nice", [{"role": "user", "content": "hi"}], build_definitions(), **options)]


def test_text_and_a_tool_request_are_read_in_order():
    events = [
        {"type": "response.created", "response": RESPONSE},
        {"type": "response.output_item.added", "output_index": 0, "item": dict(MESSAGE, content=[], status="in_progress")},
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Let me "},
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "look. "},
        {"type": "response.output_item.done", "output_index": 0, "item": MESSAGE},
        {"type": "response.output_item.added", "output_index": 1, "item": dict(CALL, arguments="", status="in_progress")},
        {"type": "response.output_item.done", "output_index": 1, "item": CALL},
        {"type": "response.completed", "response": dict(RESPONSE, status="completed", output=[MESSAGE, CALL])},
    ]
    llm, seen = make_client(events)
    result = asyncio.run(collect(llm))
    assert [(e.kind, e.text) for e in result[:2]] == [("text", "Let me "), ("text", "look. ")]
    assert result[2].kind == "tool_start" and result[2].call == {"name": "find_available_slots"}  # the name arrives first
    assert result[3].kind == "tool_call"
    assert result[3].call == {"id": "fc_1", "call_id": "call_1", "name": "find_available_slots", "arguments": "{\"service_id\":\"consultation\"}"}
    assert len(result) == 4  # the message item is not mistaken for a tool request

    body = seen["body"]
    assert body["stream"] is True and body["instructions"] == "be nice" and body["model"] == settings.openai_model
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert [t["name"] for t in body["tools"]][:2] == ["get_services", "get_business_config"]
    assert body["max_output_tokens"] == settings.openai_max_output_tokens


def test_the_last_round_forbids_tools():
    llm, seen = make_client([{"type": "response.completed", "response": dict(RESPONSE, status="completed")}])
    asyncio.run(collect(llm, allow_tools=False))
    assert seen["body"]["tool_choice"] == "none"


def test_a_failed_response_raises():
    llm, _ = make_client([{"type": "response.failed", "response": dict(RESPONSE, status="failed")}])
    try:
        asyncio.run(collect(llm))
        assert False, "should have raised"
    except RuntimeError as error:
        assert "failed" in str(error)


def test_a_cut_off_answer_ends_quietly():
    events = [
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Half an ans"},
        {"type": "response.incomplete", "response": dict(RESPONSE, status="incomplete")},
    ]
    llm, _ = make_client(events)
    assert [e.text for e in asyncio.run(collect(llm))] == ["Half an ans"]
