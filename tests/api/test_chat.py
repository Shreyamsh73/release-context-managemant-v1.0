import json

import pytest

from context_gateway.core.types import Scope
from tests.api.conftest import running_app
from tests.fakes import FakeLLM
from tests.helpers import assistant, system, user

SCOPE = Scope("default", "anonymous", "default")


async def stored(h, scope=SCOPE):
    session = await h.store.find_session(scope)
    if session is None:
        return None
    return await h.store.list_messages(session.id, limit=500)


# spec 36
async def test_stateless_conversation_gets_stored_history():
    async with running_app() as h:
        h.llm.queue_text("Nice to meet you, Shreyamsh.")
        first = await h.chat(user("my name is Shreyamsh"))
        assert first.status_code == 200
        assert first.headers["X-Memory-Mode"] == "stateless"
        assert first.headers["X-Memory-History-Turns"] == "0"

        h.llm.queue_text("Your name is Shreyamsh.")
        second = await h.chat(user("what is my name"))
        assert second.status_code == 200
        assert second.json()["choices"][0]["message"]["content"] == "Your name is Shreyamsh."

        assert h.llm.calls[1].messages == [
            user("my name is Shreyamsh"),
            assistant("Nice to meet you, Shreyamsh."),
            user("what is my name"),
        ]
        assert second.headers["X-Memory-Mode"] == "stateless"
        assert second.headers["X-Memory-History-Turns"] == "1"
        assert second.headers["X-Memory-Dropped-Turns"] == "0"
        assert second.headers["X-Memory-Summary"] == "none"
        assert int(second.headers["X-Memory-Prompt-Tokens"]) > 0
        assert second.headers["X-Memory-App"] == "default"
        assert second.headers["X-Memory-User"] == "anonymous"
        assert second.headers["X-Memory-Session"] == "default"
        assert second.headers["X-Request-Id"]


async def test_response_is_openai_shaped_and_model_is_the_clients_name():
    async with running_app(model_aliases={"claude": "bedrock/converse/real-id"}) as h:
        response = await h.chat(user("hi"), model="claude")
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "claude"
        assert body["choices"][0]["message"] == {"role": "assistant", "content": "fake reply"}
        assert h.llm.calls[0].model == "bedrock/converse/real-id"


async def test_missing_model_uses_default_model():
    async with running_app(default_model="bedrock/converse/default-id") as h:
        response = await h.client.post("/v1/chat/completions", json={"messages": [user("hi")]})
        assert response.status_code == 200
        assert h.llm.calls[0].model == "bedrock/converse/default-id"


async def test_unlisted_model_rejected_when_not_allowed():
    async with running_app(allow_unlisted_models=False, model_aliases={"a": "x/y"}) as h:
        response = await h.chat(user("hi"), model="other")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert h.llm.calls == []


async def test_params_pass_through_and_user_is_not_forwarded():
    async with running_app() as h:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        await h.chat(
            user("hi"),
            temperature=0.2,
            max_tokens=50,
            tools=tools,
            tool_choice="auto",
            stop=["x"],
            user="alice",
            stream_options={"include_usage": True},
        )
        params = h.llm.calls[0].params
        assert params == {
            "temperature": 0.2,
            "max_tokens": 50,
            "tools": tools,
            "tool_choice": "auto",
            "stop": ["x"],
        }


async def test_body_user_field_sets_scope_user():
    async with running_app() as h:
        response = await h.chat(user("hi"), user="alice")
        assert response.headers["X-Memory-User"] == "alice"
        assert await stored(h, Scope("default", "alice", "default")) is not None


# spec 37
async def test_stateful_client_history_is_not_duplicated():
    async with running_app() as h:
        headers = {"X-Memory-Mode": "stateful", "X-Memory-Session": "chat1"}
        scope = Scope("default", "anonymous", "chat1")

        h.llm.queue_text("r1")
        c1 = [system("sys"), user("q1")]
        first = await h.chat(*c1, headers=headers)
        h.llm.queue_text("r2")
        c2 = [system("sys"), user("q1"), assistant("r1"), user("q2")]
        await h.chat(*c2, headers=headers)
        h.llm.queue_text("r3")
        c3 = [*c2, assistant("r2"), user("q3")]
        last = await h.chat(*c3, headers=headers)

        assert first.headers["X-Memory-Mode"] == "stateful"
        assert last.headers["X-Memory-Summary"] == "none"
        assert [c.messages for c in h.llm.calls] == [c1, c2, c3]
        messages = await stored(h, scope)
        assert [(m.role, m.message["content"]) for m in messages] == [
            ("user", "q1"),
            ("assistant", "r1"),
            ("user", "q2"),
            ("assistant", "r2"),
            ("user", "q3"),
            ("assistant", "r3"),
        ]


async def test_auto_detected_stateful_chat_gets_a_stable_fingerprint_session():
    async with running_app() as h:
        c2 = [user("q1"), assistant("r1"), user("q2")]
        first = await h.chat(*c2)
        c3 = [*c2, assistant("fake reply"), user("q3")]
        second = await h.chat(*c3)

        key = first.headers["X-Memory-Session"]
        assert key.startswith("fp-")
        assert second.headers["X-Memory-Session"] == key
        assert first.headers["X-Memory-Mode"] == "stateful"
        messages = await stored(h, Scope("default", "anonymous", key))
        # Call 1 backfills the whole history (3 + reply). Call 2 adds only the tail.
        assert [m.message["content"] for m in messages] == [
            "q1",
            "r1",
            "q2",
            "fake reply",
            "q3",
            "fake reply",
        ]


async def test_stateful_over_budget_trims_clients_oldest_turns():
    async with running_app(context_budget_tokens=40) as h:
        long = "x" * 60
        messages = [user(long), assistant(long), user("q2"), assistant("a2"), user("q3")]
        response = await h.chat(*messages)
        assert response.headers["X-Memory-Mode"] == "stateful"
        assert int(response.headers["X-Memory-Dropped-Turns"]) == 1
        assert h.llm.calls[0].messages == [user("q2"), assistant("a2"), user("q3")]


# spec 38
async def test_scoped_route_resolves_scope_from_path_and_header_overrides():
    async with running_app() as h:
        path = "/scope/ide/alice/s1/v1/chat/completions"
        response = await h.chat(user("hi"), path=path)
        assert response.status_code == 200
        assert response.headers["X-Memory-App"] == "ide"
        assert response.headers["X-Memory-User"] == "alice"
        assert response.headers["X-Memory-Session"] == "s1"
        assert await stored(h, Scope("ide", "alice", "s1")) is not None

        overridden = await h.chat(user("hi"), path=path, headers={"X-Memory-User": "bob"})
        assert overridden.headers["X-Memory-User"] == "bob"
        assert overridden.headers["X-Memory-App"] == "ide"
        assert await stored(h, Scope("ide", "bob", "s1")) is not None


async def test_scoped_route_with_auto_session_uses_default_when_stateless():
    async with running_app() as h:
        response = await h.chat(user("hi"), path="/scope/ide/alice/auto/v1/chat/completions")
        assert response.headers["X-Memory-Session"] == "default"


async def test_scoped_route_with_auto_session_fingerprints_when_stateful():
    async with running_app() as h:
        path = "/scope/ide/alice/auto/v1/chat/completions"
        response = await h.chat(user("q"), assistant("a"), user("q2"), path=path)
        assert response.headers["X-Memory-Session"].startswith("fp-")


# spec 39
async def test_users_and_sessions_are_isolated():
    async with running_app() as h:
        alice = {"X-Memory-User": "alice"}
        await h.chat(user("alice secret"), headers=alice)
        await h.chat(user("bob question"), headers={"X-Memory-User": "bob"})
        await h.chat(user("alice other"), headers={**alice, "X-Memory-Session": "s2"})

        assert h.llm.calls[1].messages == [user("bob question")]
        assert h.llm.calls[2].messages == [user("alice other")]
        await h.chat(user("alice again"), headers=alice)
        contents = [m["content"] for m in h.llm.calls[3].messages]
        assert contents == ["alice secret", "fake reply", "alice again"]


# spec 45
async def test_mode_off_forwards_untouched_and_stores_nothing():
    async with running_app() as h:
        messages = [system("s"), user("q"), assistant("a"), user("q2")]
        response = await h.chat(*messages, headers={"X-Memory-Mode": "off"})
        assert response.status_code == 200
        assert response.headers["X-Memory-Mode"] == "off"
        assert response.headers["X-Memory-History-Turns"] == "0"
        assert response.headers["X-Memory-Prompt-Tokens"] == "0"
        assert response.headers["X-Memory-Summary"] == "none"
        assert "X-Memory-App" not in response.headers
        assert "X-Memory-Session" not in response.headers
        assert h.llm.calls[0].messages == messages
        assert await h.store.find_session(SCOPE) is None


# spec 46
class BrokenStore:
    """Every method raises, like a database that is down."""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise RuntimeError(f"{name} is down")

        return fail


async def test_broken_store_degrades_to_passthrough():
    async with running_app(store=BrokenStore()) as h:
        messages = [system("s"), user("hello")]
        response = await h.chat(*messages)
        assert response.status_code == 200
        assert response.headers["X-Memory-Mode"] == "degraded"
        assert response.headers["X-Memory-History-Turns"] == "0"
        assert response.headers["X-Memory-Session"] == "default"
        assert response.json()["choices"][0]["message"]["content"] == "fake reply"
        assert h.llm.calls[0].messages == messages


async def test_broken_store_degrades_streaming_too():
    async with running_app(store=BrokenStore()) as h:
        response = await h.chat(user("hello"), stream=True)
        assert response.status_code == 200
        assert response.headers["X-Memory-Mode"] == "degraded"
        assert response.text.endswith("data: [DONE]\n\n")


async def test_recording_failure_does_not_fail_the_request():
    class FailingAppend:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def append_messages(self, *args, **kwargs):
            raise RuntimeError("disk full")

    from context_gateway.store.memory import InMemorySessionStore

    async with running_app(store=FailingAppend(InMemorySessionStore())) as h:
        response = await h.chat(user("hi"))
        assert response.status_code == 200
        assert response.headers["X-Memory-Mode"] == "stateless"


# spec 44
@pytest.mark.parametrize(
    ("upstream", "expected"),
    [(429, 429), (400, 400), (401, 401), (500, 502), (503, 502), (None, 502)],
)
async def test_upstream_errors_are_mapped(upstream, expected):
    async with running_app() as h:
        h.llm.queue_error(upstream, "provider said no")
        response = await h.chat(user("hi"))
        assert response.status_code == expected
        assert response.json() == {
            "error": {
                "message": "provider said no",
                "type": "upstream_error",
                "param": None,
                "code": "upstream_error",
            }
        }
        # Nothing is recorded when the provider call fails.
        messages = await stored(h)
        assert messages == []


async def test_long_upstream_messages_are_truncated():
    async with running_app() as h:
        h.llm.queue_error(500, "x" * 2000)
        response = await h.chat(user("hi"))
        assert len(response.json()["error"]["message"]) <= 503


async def test_unexpected_exception_returns_500_without_stack_trace():
    class ExplodingLLM(FakeLLM):
        async def complete(self, **kwargs):
            raise ZeroDivisionError("secret internals")

    async with running_app(llm=ExplodingLLM()) as h:
        response = await h.chat(user("hi"))
        assert response.status_code == 500
        assert response.json() == {
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        }
        assert "secret" not in response.text
        assert response.headers["X-Request-Id"]


# spec 51
@pytest.mark.parametrize(
    "body",
    [
        {"model": "m", "messages": []},
        {"model": "m"},
        {"model": "m", "messages": "hello"},
        {"model": "m", "messages": ["not a dict"]},
        {"model": "m", "messages": [{"content": "no role"}]},
        {"model": "m", "messages": [{"role": 5, "content": "x"}]},
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": "maybe"},
        {"model": 7, "messages": [{"role": "user", "content": "x"}]},
        {"model": "m", "messages": [{"role": "system", "content": "only system"}]},
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "user": 5},
    ],
)
async def test_validation_errors_are_openai_shaped_400(body):
    async with running_app() as h:
        response = await h.client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_request"
        assert error["message"]
        assert set(error) == {"message", "type", "param", "code"}
        assert h.llm.calls == []


@pytest.mark.parametrize("raw", [b"{not json", b"", b"[1, 2]", b'"text"'])
async def test_bad_json_is_openai_shaped_400(raw):
    async with running_app() as h:
        response = await h.client.post(
            "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


async def test_invalid_scope_and_mode_headers_are_400():
    async with running_app() as h:
        bad_scope = await h.chat(user("hi"), headers={"X-Memory-User": "has space"})
        assert bad_scope.status_code == 400
        assert bad_scope.json()["error"]["code"] == "invalid_scope"
        assert "user" in bad_scope.json()["error"]["message"]

        bad_mode = await h.chat(user("hi"), headers={"X-Memory-Mode": "sometimes"})
        assert bad_mode.status_code == 400
        assert bad_mode.json()["error"]["code"] == "invalid_memory_mode"


async def test_unknown_route_and_wrong_method_use_error_shape():
    async with running_app() as h:
        missing = await h.client.get("/nope")
        assert missing.status_code == 404
        assert missing.json()["error"]["type"] == "not_found_error"
        wrong = await h.client.get("/v1/chat/completions")
        assert wrong.status_code == 405
        assert wrong.json()["error"]["type"] == "invalid_request_error"


async def test_request_id_is_reused_when_sane_and_generated_otherwise():
    async with running_app() as h:
        reused = await h.chat(user("hi"), headers={"X-Request-Id": "req-123"})
        assert reused.headers["X-Request-Id"] == "req-123"
        generated = await h.chat(user("hi"), headers={"X-Request-Id": "bad id with spaces"})
        assert generated.headers["X-Request-Id"] != "bad id with spaces"
        assert len(generated.headers["X-Request-Id"]) == 32


async def test_tight_budget_reports_dropped_turns_in_headers():
    async with running_app(context_budget_tokens=60) as h:
        for i in range(4):
            await h.chat(user(f"question number {i} " + "y" * 30))
        last = await h.chat(user("final"))
        assert int(last.headers["X-Memory-Dropped-Turns"]) >= 1
        assert int(last.headers["X-Memory-Prompt-Tokens"]) <= 60


async def test_tool_call_response_is_recorded_whole():
    async with running_app() as h:
        h.llm.queue_tool_call("get_weather", {"city": "Paris"})
        response = await h.chat(user("weather?"))
        message = response.json()["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        messages = await stored(h)
        assert messages[1].message["content"] is None
        assert json.loads(messages[1].message["tool_calls"][0]["function"]["arguments"]) == {
            "city": "Paris"
        }
