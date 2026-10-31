import logging

import httpx
import pytest
from redis.asyncio import Redis

from context_gateway.app import create_app
from context_gateway.store.cache import CachedSessionStore
from context_gateway.store.postgres import PostgresSessionStore
from tests.fakes import FakeLLM
from tests.helpers import make_settings, user
from tests.store_contract import SCOPE, StoreContract, conversation

CLOSED_PORT_URL = "redis://127.0.0.1:1/0"


class SpyPostgresStore(PostgresSessionStore):
    """Counts load_window calls to tell cache hits from misses."""

    window_loads = 0

    async def load_window(self, session_id, *, max_turns):
        self.window_loads += 1
        return await super().load_window(session_id, max_turns=max_turns)


@pytest.fixture
async def redis(redis_url):
    client = Redis.from_url(redis_url, decode_responses=True)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
async def inner(pg_async_url):
    store = SpyPostgresStore(pg_async_url)
    yield store
    await store.close()


@pytest.fixture
async def store(inner, redis):
    return CachedSessionStore(inner, redis, 600, cache_max_turns=50)


class TestCachedPostgresContract(StoreContract):
    """The shared contract suite through the Redis cache over real Postgres."""


# spec 56
async def test_second_load_window_is_served_from_cache(store, inner, redis):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(3))

    first = await store.load_window(session.id, max_turns=50)
    second = await store.load_window(session.id, max_turns=50)

    assert inner.window_loads == 1
    assert second == first
    assert await redis.exists(f"cg:window:{session.id}:50") == 1
    assert 0 < await redis.ttl(f"cg:window:{session.id}:50") <= 600


async def test_cache_is_invalidated_after_append_messages(store, inner, redis):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(1))
    await store.load_window(session.id, max_turns=50)
    await store.load_window(session.id, max_turns=50)
    assert inner.window_loads == 1

    await store.append_messages(session.id, conversation(1))
    assert await redis.exists(f"cg:window:{session.id}:50") == 0
    window = await store.load_window(session.id, max_turns=50)
    assert len(window.turns) == 2
    assert inner.window_loads == 2


async def test_cache_is_invalidated_after_summary_and_delete(store, inner, redis):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(4))
    await store.load_window(session.id, max_turns=50)
    await store.apply_summary(session.id, summary="S", through_turn=2, expected_through_turn=0)
    window = await store.load_window(session.id, max_turns=50)
    assert window.summary == "S"
    assert [t.index for t in window.turns] == [3, 4]

    await store.delete_session(session.id)
    assert await redis.keys("cg:window:*") == []


async def test_redis_down_still_returns_correct_data(inner, caplog):
    dead = Redis.from_url(
        CLOSED_PORT_URL, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5
    )
    store = CachedSessionStore(inner, dead, 600, cache_max_turns=50)
    session = await store.get_or_create_session(SCOPE)
    with caplog.at_level(logging.WARNING):
        await store.append_messages(session.id, conversation(2))
        window = await store.load_window(session.id, max_turns=50)
        again = await store.load_window(session.id, max_turns=50)
        assert await store.apply_summary(
            session.id, summary="S", through_turn=1, expected_through_turn=0
        )
        assert (await store.load_window(session.id, max_turns=50)).summary == "S"
        assert await store.delete_session(session.id) is True
    assert [t.index for t in window.turns] == [1, 2]
    assert again == window
    assert "redis" in caplog.text
    await dead.aclose()


async def test_readyz_checks_real_redis_and_reports_when_it_is_down(redis_url):
    llm = FakeLLM()
    for url, healthy in ((redis_url, True), (CLOSED_PORT_URL, False)):
        settings = make_settings(redis_url=url)
        app = create_app(settings, llm=llm)  # store is built from settings, so Redis is wired in
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                response = await client.get("/readyz")
                body = response.json()
                assert body["checks"]["store"] == "ok"
                if healthy:
                    assert response.status_code == 200
                    assert body["checks"]["redis"] == "ok"
                else:
                    assert response.status_code == 503
                    assert body["checks"]["redis"].startswith("error")


async def test_chat_flow_through_the_cache_with_the_postgres_store(pg_async_url, pg_url, redis_url):
    """Whole request path with Redis in front of Postgres: history is correct on every call."""
    settings = make_settings(
        store="postgres",
        database_url=pg_url,
        redis_url=redis_url,
        window_max_turns=50,
    )
    llm = FakeLLM()
    app = create_app(settings, llm=llm)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            for i in range(1, 5):
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [user(f"question {i}")]},
                )
                assert response.status_code == 200
                assert response.headers["X-Memory-History-Turns"] == str(i - 1)
            last = [m["content"] for m in llm.calls[-1].messages]
            assert last == [
                "question 1", "fake reply", "question 2", "fake reply",
                "question 3", "fake reply", "question 4",
            ]  # fmt: skip
            # Each chat call ends with an append, which drops the key. A read fills it again.
            preview = {"model": "m", "messages": [user("peek")]}
            await client.post("/memory/v1/context/preview", json=preview)
    check = Redis.from_url(redis_url, decode_responses=True)
    try:
        assert len(await check.keys("cg:window:*")) == 1
    finally:
        await check.flushdb()
        await check.aclose()
