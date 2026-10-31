import json

import pytest

from context_gateway.core.types import Scope
from tests.api.conftest import parse_sse, running_app
from tests.helpers import user

SCOPE = Scope("default", "anonymous", "default")


async def stored(h):
    session = await h.store.find_session(SCOPE)
    return None if session is None else await h.store.list_messages(session.id, limit=100)


def assembled_text(events):
    return "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events
        if isinstance(e, dict) and "choices" in e
    )


# spec 40
async def test_streaming_sse_events_done_and_recording():
    async with running_app() as h:
        h.llm.queue_text("The quick brown fox jumps over the lazy dog.")
        response = await h.chat(user("tell me"), model="claude-alias", stream=True)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["X-Accel-Buffering"] == "no"
        assert response.headers["X-Memory-Mode"] == "stateless"
        assert response.headers["X-Memory-History-Turns"] == "0"
        assert response.headers["X-Request-Id"]

        events = parse_sse(response.text)
        assert events[-1] == "[DONE]"
        chunks = events[:-1]
        assert len(chunks) > 3
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert {c["model"] for c in chunks} == {"claude-alias"}
        assert assembled_text(chunks) == "The quick brown fox jumps over the lazy dog."
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

        # Turn is recorded after the stream completes.
        messages = await stored(h)
        assert [(m.role, m.message["content"]) for m in messages] == [
            ("user", "tell me"),
            ("assistant", "The quick brown fox jumps over the lazy dog."),
        ]

        h.llm.queue_text("again")
        await h.chat(user("second"))
        assert [m["content"] for m in h.llm.calls[1].messages] == [
            "tell me",
            "The quick brown fox jumps over the lazy dog.",
            "second",
        ]


async def test_streaming_passes_stream_options_but_not_user():
    async with running_app() as h:
        await h.chat(user("hi"), stream=True, stream_options={"include_usage": True}, user="alice")
        assert h.llm.calls[0].stream is True
        assert h.llm.calls[0].params == {"stream_options": {"include_usage": True}}


async def test_streaming_in_off_mode_records_nothing():
    async with running_app() as h:
        response = await h.chat(user("hi"), stream=True, headers={"X-Memory-Mode": "off"})
        assert response.headers["X-Memory-Mode"] == "off"
        assert parse_sse(response.text)[-1] == "[DONE]"
        assert await stored(h) is None


# spec 41
async def test_streaming_tool_calls_are_assembled_and_recorded():
    async with running_app() as h:
        h.llm.queue_tool_call("get_weather", {"city": "Paris", "unit": "celsius"})
        response = await h.chat(user("weather in Paris?"), stream=True)

        events = parse_sse(response.text)
        assert events[-1] == "[DONE]"
        assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"

        messages = await stored(h)
        assert [m.role for m in messages] == ["user", "assistant"]
        recorded = messages[1].message
        assert recorded["content"] is None
        assert len(recorded["tool_calls"]) == 1
        call = recorded["tool_calls"][0]
        assert call["id"] == "call_1"
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "unit": "celsius"}


# spec 42
@pytest.mark.parametrize("when", ["call", "first_chunk"])
@pytest.mark.parametrize(("upstream", "expected"), [(429, 429), (401, 401), (500, 502)])
async def test_upstream_error_before_first_chunk_is_a_json_error(when, upstream, expected):
    async with running_app() as h:
        h.llm.queue_error(upstream, "denied", when=when)
        response = await h.chat(user("hi"), stream=True)
        assert response.status_code == expected
        assert response.headers["content-type"].startswith("application/json")
        error = response.json()["error"]
        assert error["type"] == "upstream_error"
        assert error["message"] == "denied"
        assert await stored(h) == []


# spec 43
async def test_mid_stream_error_emits_error_event_and_records_nothing():
    async with running_app() as h:
        h.llm.queue_text("this reply will be cut off midway", fail_after_chunks=3)
        response = await h.chat(user("hi"), stream=True)

        assert response.status_code == 200
        events = parse_sse(response.text)
        assert "[DONE]" not in events
        assert events[-1]["error"]["type"] == "upstream_error"
        assert events[-1]["error"]["code"] == "upstream_error"
        assert any(isinstance(e, dict) and "choices" in e for e in events[:-1]), (
            "partial chunks arrive before the error"
        )
        assert await stored(h) == []


async def test_stream_that_ends_without_finish_reason_is_still_recorded():
    from tests.fakes import FakeLLM

    class NoFinishLLM(FakeLLM):
        async def _chunk_iter(self, model, reply):
            async for chunk in super()._chunk_iter(model, reply):
                if chunk["choices"][0]["finish_reason"] is None:
                    yield chunk

    async with running_app(llm=NoFinishLLM()) as h:
        h.llm.queue_text("complete anyway")
        response = await h.chat(user("hi"), stream=True)
        assert parse_sse(response.text)[-1] == "[DONE]"
        messages = await stored(h)
        assert messages[1].message["content"] == "complete anyway"


async def test_stream_headers_report_history_on_second_call():
    async with running_app() as h:
        await h.chat(user("one"), stream=True)
        second = await h.chat(user("two"), stream=True)
        assert second.headers["X-Memory-History-Turns"] == "1"
