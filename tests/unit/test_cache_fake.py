"""CachedSessionStore logic with fakeredis, including Redis failure injection."""

import logging
import time

import fakeredis
import pytest

from context_gateway.core.types import Scope
from context_gateway.store.cache import CachedSessionStore
from context_gateway.store.memory import InMemorySessionStore
from tests.store_contract import SCOPE, StoreContract, conversation


class SpyStore(InMemorySessionStore):
    """Counts load_window calls so tests can tell cache hits from misses."""

    def __init__(self):
        super().__init__()
        self.window_loads = 0

    async def load_window(self, session_id, *, max_turns):
        self.window_loads += 1
        return await super().load_window(session_id, max_turns=max_turns)


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def inner():
    return SpyStore()


@pytest.fixture
def store(inner, redis):
    return CachedSessionStore(inner, redis, 600, cache_max_turns=50)


class TestCachedStoreContract(StoreContract):
    """The cache decorator must not change any store behavior.

    The contract suite calls load_window with several max_turns values, so this also
    covers the pass-through path for non-default values.
    """


async def test_second_load_is_served_from_cache(store, inner, redis):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(2))

    first = await store.load_window(session.id, max_turns=50)
    second = await store.load_window(session.id, max_turns=50)

    assert inner.window_loads == 1
    assert second == first
    assert await redis.exists(f"cg:window:{session.id}:50") == 1
    ttl = await redis.ttl(f"cg:window:{session.id}:50")
    assert 0 < ttl <= 600


async def test_cache_is_invalidated_by_append_summary_and_delete(store, inner, redis):
    session = await store.get_or_create_session(SCOPE)
    key = f"cg:window:{session.id}:50"
    await store.append_messages(session.id, conversation(3))

    await store.load_window(session.id, max_turns=50)
    assert await redis.exists(key) == 1
    await store.append_messages(session.id, conversation(1))
    assert await redis.exists(key) == 0
    window = await store.load_window(session.id, max_turns=50)
    assert len(window.turns) == 4
    assert inner.window_loads == 2

    assert await store.apply_summary(
        session.id, summary="S", through_turn=2, expected_through_turn=0
    )
    assert await redis.exists(key) == 0
    window = await store.load_window(session.id, max_turns=50)
    assert window.summary == "S"
    assert [t.index for t in window.turns] == [3, 4]

    await store.delete_session(session.id)
    assert await redis.exists(key) == 0
    assert await store.load_window(session.id, max_turns=50) is None


async def test_only_the_default_max_turns_is_cached(store, inner, redis):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(1))
    await store.load_window(session.id, max_turns=5)
    await store.load_window(session.id, max_turns=5)
    await store.load_window(session.id, max_turns=None)
    assert inner.window_loads == 3
    assert await redis.keys("cg:window:*") == []


async def test_missing_session_is_not_cached(store, inner, redis):
    import uuid

    missing = uuid.uuid4()
    assert await store.load_window(missing, max_turns=50) is None
    assert await store.load_window(missing, max_turns=50) is None
    assert inner.window_loads == 2
    assert await redis.keys("cg:window:*") == []


async def test_failed_summary_cas_still_returns_false(store):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(2))
    assert await store.apply_summary(
        session.id, summary="a", through_turn=1, expected_through_turn=0
    )
    assert not await store.apply_summary(
        session.id, summary="b", through_turn=2, expected_through_turn=0
    )


async def test_unreadable_cache_entry_is_replaced(store, inner, redis, caplog):
    session = await store.get_or_create_session(SCOPE)
    await store.append_messages(session.id, conversation(1))
    key = f"cg:window:{session.id}:50"
    await redis.set(key, "{not json")
    with caplog.at_level(logging.WARNING):
        window = await store.load_window(session.id, max_turns=50)
    assert len(window.turns) == 1
    assert "unreadable" in caplog.text
    await store.load_window(session.id, max_turns=50)
    assert inner.window_loads == 1


class ExplodingRedis:
    """Every command raises, like a Redis that is down."""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise ConnectionError("redis is down")

        return fail


async def test_redis_failure_never_fails_a_request(inner, caplog):
    store = CachedSessionStore(inner, ExplodingRedis(), 600, cache_max_turns=50)
    session = await store.get_or_create_session(SCOPE)
    with caplog.at_level(logging.WARNING):
        await store.append_messages(session.id, conversation(2))
        window = await store.load_window(session.id, max_turns=50)
        assert len(window.turns) == 2
        assert await store.apply_summary(
            session.id, summary="S", through_turn=1, expected_through_turn=0
        )
        assert (await store.load_window(session.id, max_turns=50)).summary == "S"
        assert await store.delete_session(session.id) is True
    assert "redis" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_redis_that_hangs_is_bounded_by_the_client_timeout(inner):
    """The real client is built with short timeouts, so a dead host cannot stall requests."""
    from context_gateway.deps import build_redis
    from tests.helpers import make_settings

    redis = build_redis(make_settings(redis_url="redis://127.0.0.1:1/0"))
    store = CachedSessionStore(inner, redis, 600, cache_max_turns=50)
    session = await store.get_or_create_session(Scope("a", "b", "c"))
    await store.append_messages(session.id, conversation(1))
    started = time.perf_counter()
    window = await store.load_window(session.id, max_turns=50)
    assert len(window.turns) == 1
    assert time.perf_counter() - started < 5
    await redis.aclose()
