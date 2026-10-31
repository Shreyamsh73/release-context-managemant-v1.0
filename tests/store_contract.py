"""Behavior tests every `SessionStore` implementation must pass.

Subclass `StoreContract` and provide a `store` fixture. The same suite runs against the
in-memory store (unit tests) and the Postgres store (integration tests).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from context_gateway.core.types import Scope
from context_gateway.store.base import NewMessage, SessionNotFoundError

SCOPE = Scope("app", "alice", "default")


def new(role: str, text: str | None, tokens: int = 10, **extra: Any) -> NewMessage:
    message: dict[str, Any] = {"role": role, "content": text, **extra}
    return NewMessage(message=message, role=role, token_count=tokens, model="m")


def conversation(turns: int, tokens: int = 10) -> list[NewMessage]:
    out: list[NewMessage] = []
    for i in range(1, turns + 1):
        out.append(new("user", f"q{i}", tokens))
        out.append(new("assistant", f"a{i}", tokens))
    return out


class StoreContract:
    """Parametrized by the `store` fixture of the subclass."""

    async def test_get_or_create_is_idempotent_per_scope(self, store):
        first = await store.get_or_create_session(SCOPE)
        second = await store.get_or_create_session(SCOPE)
        assert first.id == second.id
        assert (first.app, first.user, first.session_key) == ("app", "alice", "default")
        assert first.summary is None
        assert first.summarized_through_turn == 0
        assert first.last_seq == 0
        assert first.last_turn == 0
        assert second.updated_at >= first.updated_at

    async def test_different_scopes_are_isolated(self, store):
        base = await store.get_or_create_session(SCOPE)
        other_user = await store.get_or_create_session(Scope("app", "bob", "default"))
        other_app = await store.get_or_create_session(Scope("other", "alice", "default"))
        other_session = await store.get_or_create_session(Scope("app", "alice", "s2"))
        assert len({base.id, other_user.id, other_app.id, other_session.id}) == 4

    async def test_find_session_never_creates(self, store):
        assert await store.find_session(SCOPE) is None
        assert await store.find_session(SCOPE) is None
        created = await store.get_or_create_session(SCOPE)
        found = await store.find_session(SCOPE)
        assert found is not None
        assert found.id == created.id

    async def test_get_session_by_id(self, store):
        assert await store.get_session(uuid4()) is None
        created = await store.get_or_create_session(SCOPE)
        fetched = await store.get_session(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_append_assigns_seq_and_turn_indexes(self, store):
        session = await store.get_or_create_session(SCOPE)
        result = await store.append_messages(session.id, conversation(2))
        assert (result.last_seq, result.last_turn) == (4, 2)
        stored = await store.list_messages(session.id, limit=50)
        assert [(m.seq, m.turn_index, m.role) for m in stored] == [
            (1, 1, "user"),
            (2, 1, "assistant"),
            (3, 2, "user"),
            (4, 2, "assistant"),
        ]
        refreshed = await store.get_session(session.id)
        assert (refreshed.last_seq, refreshed.last_turn) == (4, 2)

    async def test_tool_messages_stay_in_their_turn(self, store):
        session = await store.get_or_create_session(SCOPE)
        call = {
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]
        }
        await store.append_messages(
            session.id,
            [
                new("user", "q"),
                new("assistant", None, **call),
                new("tool", "42", tool_call_id="c1"),
                new("assistant", "done"),
                new("user", "q2"),
            ],
        )
        stored = await store.list_messages(session.id, limit=50)
        assert [m.turn_index for m in stored] == [1, 1, 1, 1, 2]

    async def test_assistant_first_message_gets_turn_one(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, [new("assistant", "hello")])
        await store.append_messages(session.id, [new("user", "hi"), new("assistant", "yo")])
        stored = await store.list_messages(session.id, limit=50)
        assert [m.turn_index for m in stored] == [1, 2, 2]

    async def test_turn_numbering_continues_across_appends(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(1))
        result = await store.append_messages(session.id, [new("assistant", "again")])
        assert result.last_turn == 1
        result = await store.append_messages(session.id, conversation(1))
        assert (result.last_seq, result.last_turn) == (5, 2)

    async def test_append_returns_unsummarized_tokens(self, store):
        session = await store.get_or_create_session(SCOPE)
        result = await store.append_messages(session.id, conversation(2, tokens=7))
        assert result.unsummarized_tokens == 28

    async def test_append_empty_list_changes_nothing(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(1))
        result = await store.append_messages(session.id, [])
        assert (result.last_seq, result.last_turn, result.unsummarized_tokens) == (2, 1, 20)

    async def test_append_to_missing_session_raises(self, store):
        with pytest.raises(SessionNotFoundError):
            await store.append_messages(uuid4(), conversation(1))

    async def test_messages_round_trip_verbatim(self, store):
        session = await store.get_or_create_session(SCOPE)
        tool_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        }
        rich = {
            "role": "user",
            "content": [
                {"type": "text", "text": "héllo ☃ 日本語"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
        await store.append_messages(
            session.id,
            [
                NewMessage(rich, "user", 5, "provider/model"),
                NewMessage(tool_call, "assistant", 6, None),
            ],
        )
        stored = await store.list_messages(session.id, limit=10)
        assert stored[0].message == rich
        assert stored[0].model == "provider/model"
        assert stored[0].token_count == 5
        assert stored[1].message == tool_call
        assert stored[1].model is None
        assert stored[0].created_at is not None

    async def test_stored_messages_are_isolated_from_caller_mutation(self, store):
        session = await store.get_or_create_session(SCOPE)
        message = {"role": "user", "content": "original"}
        await store.append_messages(session.id, [NewMessage(message, "user", 1, None)])
        message["content"] = "mutated"
        stored = await store.list_messages(session.id, limit=10)
        assert stored[0].message["content"] == "original"

    async def test_append_updates_updated_at(self, store):
        session = await store.get_or_create_session(SCOPE)
        await asyncio.sleep(0.01)
        await store.append_messages(session.id, conversation(1))
        refreshed = await store.get_session(session.id)
        assert refreshed.updated_at > session.updated_at

    async def test_load_window_missing_session_is_none(self, store):
        assert await store.load_window(uuid4(), max_turns=10) is None

    async def test_load_window_groups_turns_in_order(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(3, tokens=10))
        window = await store.load_window(session.id, max_turns=None)
        assert window.session_id == session.id
        assert window.summary is None
        assert window.summarized_through_turn == 0
        assert [t.index for t in window.turns] == [1, 2, 3]
        assert [t.tokens for t in window.turns] == [20, 20, 20]
        assert [m["content"] for m in window.turns[1].messages] == ["q2", "a2"]
        assert window.unsummarized_tokens == 60
        assert window.truncated is False

    async def test_load_window_empty_session(self, store):
        session = await store.get_or_create_session(SCOPE)
        window = await store.load_window(session.id, max_turns=5)
        assert window.turns == []
        assert window.unsummarized_tokens == 0
        assert window.truncated is False

    async def test_load_window_max_turns_keeps_newest_and_flags_truncation(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(5, tokens=10))
        window = await store.load_window(session.id, max_turns=2)
        assert [t.index for t in window.turns] == [4, 5]
        assert window.truncated is True
        # Tokens cover all unsummarized messages, not only the returned turns.
        assert window.unsummarized_tokens == 100

    async def test_load_window_max_turns_exactly_equal_is_not_truncated(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(3))
        window = await store.load_window(session.id, max_turns=3)
        assert [t.index for t in window.turns] == [1, 2, 3]
        assert window.truncated is False

    async def test_apply_summary_moves_the_window_floor(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(4, tokens=10))
        ok = await store.apply_summary(
            session.id, summary="S1", through_turn=2, expected_through_turn=0
        )
        assert ok is True
        window = await store.load_window(session.id, max_turns=None)
        assert window.summary == "S1"
        assert window.summarized_through_turn == 2
        assert [t.index for t in window.turns] == [3, 4]
        assert window.unsummarized_tokens == 40
        refreshed = await store.get_session(session.id)
        assert refreshed.summary == "S1"
        assert refreshed.summarized_through_turn == 2

    async def test_append_after_summary_counts_only_unsummarized_tokens(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(4, tokens=10))
        await store.apply_summary(session.id, summary="S", through_turn=3, expected_through_turn=0)
        result = await store.append_messages(session.id, conversation(1, tokens=10))
        assert result.unsummarized_tokens == 40
        assert result.last_turn == 5

    async def test_apply_summary_is_compare_and_set(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(4))
        first = await store.apply_summary(
            session.id, summary="S1", through_turn=2, expected_through_turn=0
        )
        stale = await store.apply_summary(
            session.id, summary="S-stale", through_turn=3, expected_through_turn=0
        )
        assert (first, stale) == (True, False)
        refreshed = await store.get_session(session.id)
        assert refreshed.summary == "S1"
        assert refreshed.summarized_through_turn == 2
        second = await store.apply_summary(
            session.id, summary="S2", through_turn=3, expected_through_turn=2
        )
        assert second is True

    async def test_apply_summary_on_missing_session_returns_false(self, store):
        ok = await store.apply_summary(
            uuid4(), summary="x", through_turn=1, expected_through_turn=0
        )
        assert ok is False

    async def test_list_messages_returns_most_recent_in_ascending_order(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(5))
        recent = await store.list_messages(session.id, limit=3)
        assert [m.seq for m in recent] == [8, 9, 10]
        everything = await store.list_messages(session.id, limit=500)
        assert [m.seq for m in everything] == list(range(1, 11))

    async def test_list_messages_unknown_session_is_empty(self, store):
        assert await store.list_messages(uuid4(), limit=5) == []

    async def test_list_messages_includes_summarized_messages(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(3))
        await store.apply_summary(session.id, summary="S", through_turn=2, expected_through_turn=0)
        everything = await store.list_messages(session.id, limit=50)
        assert len(everything) == 6

    async def test_delete_session_removes_messages_and_allows_recreate(self, store):
        session = await store.get_or_create_session(SCOPE)
        await store.append_messages(session.id, conversation(2))
        assert await store.delete_session(session.id) is True
        assert await store.delete_session(session.id) is False
        assert await store.get_session(session.id) is None
        assert await store.find_session(SCOPE) is None
        assert await store.list_messages(session.id, limit=10) == []
        assert await store.load_window(session.id, max_turns=None) is None

        fresh = await store.get_or_create_session(SCOPE)
        assert fresh.id != session.id
        assert (fresh.last_seq, fresh.last_turn) == (0, 0)

    async def test_delete_does_not_touch_other_sessions(self, store):
        keep = await store.get_or_create_session(Scope("app", "bob", "default"))
        gone = await store.get_or_create_session(SCOPE)
        await store.append_messages(keep.id, conversation(1))
        await store.append_messages(gone.id, conversation(1))
        await store.delete_session(gone.id)
        assert len(await store.list_messages(keep.id, limit=10)) == 2

    async def test_ping_succeeds(self, store):
        await store.ping()
