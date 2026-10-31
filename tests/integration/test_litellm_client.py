"""The real LiteLLM client, run offline with LiteLLM's built-in mock responses.

These tests import LiteLLM (slow), so they live outside the unit and API suites. They need no
network, no credentials and no database.
"""

import httpx
import pytest

from context_gateway.app import create_app
from context_gateway.core.stream import StreamAccumulator
from context_gateway.core.tokens import HeuristicTokenCounter, LiteLLMTokenCounter
from context_gateway.llm.base import LLMError
from context_gateway.llm.litellm_client import LiteLLMClient
from context_gateway.store.memory import InMemorySessionStore
from tests.api.conftest import parse_sse
from tests.helpers import make_settings, user

MODEL = "bedrock/converse/us.anthropic.claude-sonnet-4-6"


@pytest.fixture(scope="module")
def litellm_module():
    import litellm

    return litellm


@pytest.fixture
def client(litellm_module):
    return LiteLLMClient()


async def test_complete_returns_an_openai_dict(client):
    response = await client.complete(
        model=MODEL, messages=[user("hi")], params={"mock_response": "hello there"}
    )
    assert isinstance(response, dict)
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "hello there"
    assert response["choices"][0]["finish_reason"] == "stop"


async def test_stream_yields_chunk_dicts_that_assemble(client):
    iterator = await client.stream(
        model="openai/gpt-4o-mini",
        messages=[user("hi")],
        params={"mock_response": "hello there friend"},
    )
    chunks = [chunk async for chunk in iterator]
    assert len(chunks) > 2
    assert all(isinstance(c, dict) for c in chunks)
    acc = StreamAccumulator()
    for chunk in chunks:
        acc.add(chunk)
    assert acc.text == "hello there friend"
    assert acc.completed is True
    assert acc.message() == {"role": "assistant", "content": "hello there friend"}


async def test_exception_status_is_mapped_to_llm_error(client, litellm_module):
    rate_limit = litellm_module.RateLimitError(
        message="slow down", llm_provider="openai", model="m"
    )
    with pytest.raises(LLMError) as exc:
        await client.complete(
            model="openai/gpt-4o-mini", messages=[user("hi")], params={"mock_response": rate_limit}
        )
    assert exc.value.status_code == 429
    assert "slow down" in exc.value.message


async def test_unknown_provider_becomes_a_400_llm_error(client):
    with pytest.raises(LLMError) as exc:
        await client.complete(model="no-such-model-xyz", messages=[user("hi")], params={})
    assert exc.value.status_code == 400


async def test_stream_errors_before_the_first_chunk_are_llm_errors(client):
    with pytest.raises(LLMError) as exc:
        await client.stream(model="no-such-model-xyz", messages=[user("hi")], params={})
    assert exc.value.status_code == 400


async def test_calls_pass_drop_params_true(client, monkeypatch):
    seen = {}

    async def spy(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(client._litellm, "acompletion", spy)
    with pytest.raises(LLMError):
        await client.complete(model=MODEL, messages=[user("hi")], params={"temperature": 0.1})
    assert seen["drop_params"] is True
    assert seen["temperature"] == 0.1
    with pytest.raises(LLMError):
        await client.stream(model=MODEL, messages=[user("hi")], params={})
    assert seen["stream"] is True
    assert seen["drop_params"] is True


async def test_gateway_end_to_end_with_real_litellm_client_and_counter(litellm_module):
    """Chat through the gateway with LiteLLM doing the (mocked) provider call and counting."""
    settings = make_settings(llm_backend="litellm", token_counter="litellm")
    app = create_app(settings, store=InMemorySessionStore())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            first = await http.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [user("my name is Shreyamsh")],
                    "mock_response": "Hi Shreyamsh",
                    "user": "alice",
                },
            )
            assert first.status_code == 200, first.text
            assert first.json()["choices"][0]["message"]["content"] == "Hi Shreyamsh"
            assert first.json()["model"] == MODEL
            second = await http.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [user("what is my name")],
                    "mock_response": "Shreyamsh",
                    "user": "alice",
                },
            )
            assert second.headers["X-Memory-History-Turns"] == "1"
            assert int(second.headers["X-Memory-Prompt-Tokens"]) > 0

            streamed = await http.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [user("stream please")],
                    "mock_response": "streamed words here",
                    "stream": True,
                    "user": "alice",
                },
            )
            assert streamed.status_code == 200
            events = parse_sse(streamed.text)
            assert events[-1] == "[DONE]"
            text = "".join((e["choices"][0]["delta"].get("content") or "") for e in events[:-1])
            assert text == "streamed words here"
            assert {e["model"] for e in events[:-1]} == {MODEL}

            failing = await http.post(
                "/v1/chat/completions",
                json={"model": "no-such-model-xyz", "messages": [user("hi")], "user": "alice"},
            )
            assert failing.status_code == 400
            assert failing.json()["error"]["type"] == "upstream_error"


def test_litellm_token_counter_counts_offline_and_grows_with_text(litellm_module):
    counter = LiteLLMTokenCounter()
    short = counter.count([user("hi")], MODEL)
    long = counter.count([user("hi " * 200)], MODEL)
    assert isinstance(short, int)
    assert 0 < short < long
    assert counter.count([user("hi")], None) > 0


def test_litellm_token_counter_falls_back_to_heuristic_on_error(litellm_module, monkeypatch):
    counter = LiteLLMTokenCounter()

    def boom(**kwargs):
        raise ValueError("tokenizer unavailable")

    monkeypatch.setattr(counter._litellm, "token_counter", boom)
    messages = [user("hello world")]
    assert counter.count(messages, MODEL) == HeuristicTokenCounter().count(messages, MODEL)
