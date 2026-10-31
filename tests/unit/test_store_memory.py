import asyncio

import pytest

from context_gateway.store.memory import InMemorySessionStore
from tests.store_contract import SCOPE, StoreContract, conversation


@pytest.fixture
def store():
    return InMemorySessionStore()


class TestInMemoryStore(StoreContract):
    """Runs the shared contract suite against the in-memory store."""

    async def test_concurrent_appends_get_unique_gapless_seq(self, store):
        session = await store.get_or_create_session(SCOPE)
        await asyncio.gather(
            *(store.append_messages(session.id, conversation(1)) for _ in range(20))
        )
        stored = await store.list_messages(session.id, limit=500)
        assert [m.seq for m in stored] == list(range(1, 41))
