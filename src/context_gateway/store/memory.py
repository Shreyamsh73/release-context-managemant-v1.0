"""In-memory session store. Same semantics as Postgres, for tests and quick local runs."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from context_gateway.core.types import Scope, Turn, Window
from context_gateway.store.base import (
    AppendResult,
    NewMessage,
    SessionNotFoundError,
    SessionRecord,
    StoredMessage,
)


def _now() -> datetime:
    return datetime.now(UTC)


class InMemorySessionStore:
    """Dict-backed `SessionStore`. Data is lost when the process exits."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[UUID, SessionRecord] = {}
        self._by_scope: dict[tuple[str, str, str], UUID] = {}
        self._messages: dict[UUID, list[StoredMessage]] = {}

    async def get_or_create_session(self, scope: Scope) -> SessionRecord:
        """Return the session for the scope, creating it when missing."""
        key = (scope.app, scope.user, scope.session)
        async with self._lock:
            session_id = self._by_scope.get(key)
            if session_id is not None:
                record = replace(self._sessions[session_id], updated_at=_now())
            else:
                now = _now()
                session_id = uuid4()
                record = SessionRecord(
                    id=session_id,
                    app=scope.app,
                    user=scope.user,
                    session_key=scope.session,
                    summary=None,
                    summarized_through_turn=0,
                    last_seq=0,
                    last_turn=0,
                    created_at=now,
                    updated_at=now,
                )
                self._by_scope[key] = session_id
                self._messages[session_id] = []
            self._sessions[session_id] = record
            return record

    async def find_session(self, scope: Scope) -> SessionRecord | None:
        """Return the session for the scope, or None."""
        async with self._lock:
            session_id = self._by_scope.get((scope.app, scope.user, scope.session))
            return self._sessions.get(session_id) if session_id else None

    async def get_session(self, session_id: UUID) -> SessionRecord | None:
        """Return the session with this id, or None."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def load_window(self, session_id: UUID, *, max_turns: int | None) -> Window | None:
        """Return the summary and unsummarized turns, or None if the session is missing."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            floor = session.summarized_through_turn
            unsummarized = [m for m in self._messages[session_id] if m.turn_index > floor]
            total_tokens = sum(m.token_count for m in unsummarized)
            cut = 0
            if max_turns is not None:
                cut = session.last_turn - max_turns
            selected = [m for m in unsummarized if m.turn_index > cut]
            grouped: dict[int, list[StoredMessage]] = {}
            for stored in selected:
                grouped.setdefault(stored.turn_index, []).append(stored)
            turns = [
                Turn(
                    index=index,
                    messages=[copy.deepcopy(m.message) for m in group],
                    tokens=sum(m.token_count for m in group),
                )
                for index, group in sorted(grouped.items())
            ]
            return Window(
                session_id=session_id,
                summary=session.summary,
                summarized_through_turn=floor,
                turns=turns,
                unsummarized_tokens=total_tokens,
                truncated=max_turns is not None and cut > floor,
            )

    async def append_messages(
        self, session_id: UUID, messages: Sequence[NewMessage]
    ) -> AppendResult:
        """Append messages, assigning seq and turn index."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(str(session_id))
            last_seq, last_turn = session.last_seq, session.last_turn
            now = _now()
            for new in messages:
                if new.role == "user" or last_turn == 0:
                    last_turn += 1
                last_seq += 1
                self._messages[session_id].append(
                    StoredMessage(
                        seq=last_seq,
                        turn_index=last_turn,
                        role=new.role,
                        message=copy.deepcopy(new.message),
                        token_count=new.token_count,
                        model=new.model,
                        created_at=now,
                    )
                )
            if messages:
                session = replace(session, last_seq=last_seq, last_turn=last_turn, updated_at=now)
                self._sessions[session_id] = session
            unsummarized = sum(
                m.token_count
                for m in self._messages[session_id]
                if m.turn_index > session.summarized_through_turn
            )
            return AppendResult(
                last_seq=session.last_seq,
                last_turn=session.last_turn,
                unsummarized_tokens=unsummarized,
            )

    async def apply_summary(
        self, session_id: UUID, *, summary: str, through_turn: int, expected_through_turn: int
    ) -> bool:
        """Compare-and-set the summary. Returns True when the session changed."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.summarized_through_turn != expected_through_turn:
                return False
            self._sessions[session_id] = replace(
                session,
                summary=summary,
                summarized_through_turn=through_turn,
                updated_at=_now(),
            )
            return True

    async def list_messages(self, session_id: UUID, *, limit: int) -> list[StoredMessage]:
        """Return the most recent stored messages in ascending order."""
        async with self._lock:
            stored = self._messages.get(session_id, [])
            tail = stored[-limit:] if limit > 0 else []
            return [replace(m, message=copy.deepcopy(m.message)) for m in tail]

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete the session and its messages. Returns True if it existed."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            self._by_scope.pop((session.app, session.user, session.session_key), None)
            self._messages.pop(session_id, None)
            return True

    async def ping(self) -> None:
        """Always healthy."""

    async def close(self) -> None:
        """Nothing to release."""
