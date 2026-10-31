import asyncio

import pytest

from context_gateway.core.types import Scope
from context_gateway.store.base import NewMessage
from context_gateway.store.postgres import PostgresSessionStore
from tests.store_contract import SCOPE, StoreContract, conversation


@pytest.fixture
async def store(pg_async_url):
    postgres = PostgresSessionStore(pg_async_url)
    yield postgres
    await postgres.close()


# spec 52
class TestPostgresStore(StoreContract):
    """Runs the shared contract suite against real Postgres."""

    # spec 53
    async def test_concurrent_appends_get_unique_gapless_seq(self, store):
        session = await store.get_or_create_session(SCOPE)
        results = await asyncio.gather(
            *(store.append_messages(session.id, conversation(1)) for _ in range(20))
        )
        stored = await store.list_messages(session.id, limit=500)
        assert [m.seq for m in stored] == list(range(1, 41))
        assert [m.turn_index for m in stored] == [i for t in range(1, 21) for i in (t, t)]
        refreshed = await store.get_session(session.id)
        assert (refreshed.last_seq, refreshed.last_turn) == (40, 20)
        assert sorted(r.last_seq for r in results) == list(range(2, 42, 2))

    async def test_concurrent_get_or_create_returns_one_session(self, store):
        sessions = await asyncio.gather(*(store.get_or_create_session(SCOPE) for _ in range(15)))
        assert len({s.id for s in sessions}) == 1

    # spec 54
    async def test_apply_summary_cas_against_real_db(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(5))
        outcomes = await asyncio.gather(
            *(
                store.apply_summary(
                    session.id, summary=f"S{i}", through_turn=2, expected_through_turn=0
                )
                for i in range(8)
            )
        )
        assert outcomes.count(True) == 1
        refreshed = await store.get_session(session.id)
        assert refreshed.summarized_through_turn == 2
        assert refreshed.summary in {f"S{i}" for i in range(8)}
        assert (
            await store.apply_summary(
                session.id, summary="late", through_turn=3, expected_through_turn=0
            )
            is False
        )

    async def test_delete_cascades_messages_in_the_database(self, store, pg_url):
        import psycopg

        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(2))
        await store.delete_session(session.id)
        with psycopg.connect(pg_url) as conn:
            count = conn.execute(
                "SELECT count(*) FROM cg_messages WHERE session_id = %s", (session.id,)
            ).fetchone()[0]
        assert count == 0

    async def test_messages_are_stored_as_jsonb(self, store, pg_url):
        import psycopg

        session = await store.get_or_create_session(Scope("a", "b", "c"))
        message = {"role": "user", "content": "snow ☃", "extra": {"n": [1, 2]}}
        await store.append_messages(session.id, [NewMessage(message, "user", 3, "m")])
        with psycopg.connect(pg_url) as conn:
            row = conn.execute(
                "SELECT jsonb_typeof(message), message->>'content', token_count, model "
                "FROM cg_messages WHERE session_id = %s",
                (session.id,),
            ).fetchone()
        assert row == ("object", "snow ☃", 3, "m")

    async def test_duplicate_seq_is_rejected_by_the_database(self, pg_url, store):
        import psycopg

        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(1))
        with psycopg.connect(pg_url) as conn, pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO cg_messages (session_id, seq, turn_index, role, message, token_count)"
                " VALUES (%s, 1, 1, 'user', '{}', 1)",
                (session.id,),
            )

    async def test_ping_fails_when_database_is_unreachable(self):
        broken = PostgresSessionStore("postgresql+psycopg://cg:cg@127.0.0.1:1/none")
        with pytest.raises(Exception, match=r".+"):
            await broken.ping()
        await broken.close()
