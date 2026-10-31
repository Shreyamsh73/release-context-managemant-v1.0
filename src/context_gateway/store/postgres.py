"""Postgres session store built on SQLAlchemy Core and psycopg 3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from context_gateway.core.types import Scope, Turn, Window
from context_gateway.store.base import (
    AppendResult,
    NewMessage,
    SessionNotFoundError,
    SessionRecord,
    StoredMessage,
)
from context_gateway.store.tables import messages, sessions


def _record(row: Any) -> SessionRecord:
    data = row._mapping
    return SessionRecord(
        id=data["id"],
        app=data["app_id"],
        user=data["user_id"],
        session_key=data["session_key"],
        summary=data["summary"],
        summarized_through_turn=data["summarized_through_turn"],
        last_seq=data["last_seq"],
        last_turn=data["last_turn"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


class PostgresSessionStore:
    """`SessionStore` backed by Postgres. Takes an async SQLAlchemy URL."""

    def __init__(
        self,
        url: str,
        *,
        pool_size: int = 10,
        max_overflow: int = 10,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._engine = engine or create_async_engine(
            url, pool_size=pool_size, max_overflow=max_overflow, pool_pre_ping=True
        )

    async def get_or_create_session(self, scope: Scope) -> SessionRecord:
        """Insert the session or touch `updated_at` when it exists, in one statement."""
        statement = (
            pg_insert(sessions)
            .values(
                id=uuid4(),
                app_id=scope.app,
                user_id=scope.user,
                session_key=scope.session,
            )
            .on_conflict_do_update(
                constraint="uq_cg_sessions_scope", set_={"updated_at": func.now()}
            )
            .returning(*sessions.c)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(statement)).one()
        return _record(row)

    async def find_session(self, scope: Scope) -> SessionRecord | None:
        """Return the session for the scope, or None."""
        statement = select(sessions).where(
            sessions.c.app_id == scope.app,
            sessions.c.user_id == scope.user,
            sessions.c.session_key == scope.session,
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(statement)).first()
        return _record(row) if row else None

    async def get_session(self, session_id: UUID) -> SessionRecord | None:
        """Return the session with this id, or None."""
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(sessions).where(sessions.c.id == session_id))).first()
        return _record(row) if row else None

    async def load_window(self, session_id: UUID, *, max_turns: int | None) -> Window | None:
        """Return the summary and unsummarized turns, newest `max_turns` at most."""
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(sessions).where(sessions.c.id == session_id))).first()
            if row is None:
                return None
            session = _record(row)
            floor = session.summarized_through_turn
            cut = session.last_turn - max_turns if max_turns is not None else 0
            lower = max(floor, cut)
            rows = (
                await conn.execute(
                    select(messages.c.turn_index, messages.c.message, messages.c.token_count)
                    .where(messages.c.session_id == session_id, messages.c.turn_index > lower)
                    .order_by(messages.c.seq)
                )
            ).all()
            total = (
                await conn.execute(
                    select(func.coalesce(func.sum(messages.c.token_count), 0)).where(
                        messages.c.session_id == session_id, messages.c.turn_index > floor
                    )
                )
            ).scalar_one()

        grouped: dict[int, list[Any]] = {}
        tokens: dict[int, int] = {}
        for turn_index, message, token_count in rows:
            grouped.setdefault(turn_index, []).append(message)
            tokens[turn_index] = tokens.get(turn_index, 0) + token_count
        turns = [Turn(index=i, messages=grouped[i], tokens=tokens[i]) for i in sorted(grouped)]
        return Window(
            session_id=session_id,
            summary=session.summary,
            summarized_through_turn=floor,
            turns=turns,
            unsummarized_tokens=int(total),
            truncated=max_turns is not None and cut > floor,
        )

    async def append_messages(
        self, session_id: UUID, messages_in: Sequence[NewMessage]
    ) -> AppendResult:
        """Append messages in one transaction. The session row lock serializes writers."""
        async with self._engine.begin() as conn:
            locked = (
                await conn.execute(
                    select(
                        sessions.c.last_seq,
                        sessions.c.last_turn,
                        sessions.c.summarized_through_turn,
                    )
                    .where(sessions.c.id == session_id)
                    .with_for_update()
                )
            ).first()
            if locked is None:
                raise SessionNotFoundError(str(session_id))
            last_seq, last_turn, summarized = locked

            if messages_in:
                rows = []
                for new in messages_in:
                    if new.role == "user" or last_turn == 0:
                        last_turn += 1
                    last_seq += 1
                    rows.append(
                        {
                            "session_id": session_id,
                            "seq": last_seq,
                            "turn_index": last_turn,
                            "role": new.role,
                            "message": new.message,
                            "token_count": new.token_count,
                            "model": new.model,
                        }
                    )
                await conn.execute(insert(messages), rows)
                await conn.execute(
                    update(sessions)
                    .where(sessions.c.id == session_id)
                    .values(last_seq=last_seq, last_turn=last_turn, updated_at=func.now())
                )

            total = (
                await conn.execute(
                    select(func.coalesce(func.sum(messages.c.token_count), 0)).where(
                        messages.c.session_id == session_id, messages.c.turn_index > summarized
                    )
                )
            ).scalar_one()
        return AppendResult(last_seq=last_seq, last_turn=last_turn, unsummarized_tokens=int(total))

    async def apply_summary(
        self, session_id: UUID, *, summary: str, through_turn: int, expected_through_turn: int
    ) -> bool:
        """Compare-and-set: only update when the summary has not moved since it was read."""
        statement = (
            update(sessions)
            .where(
                sessions.c.id == session_id,
                sessions.c.summarized_through_turn == expected_through_turn,
            )
            .values(summary=summary, summarized_through_turn=through_turn, updated_at=func.now())
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(statement)
        return result.rowcount == 1

    async def list_messages(self, session_id: UUID, *, limit: int) -> list[StoredMessage]:
        """Return the most recent stored messages in ascending order."""
        statement = (
            select(messages)
            .where(messages.c.session_id == session_id)
            .order_by(messages.c.seq.desc())
            .limit(limit)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement)).all()
        return [
            StoredMessage(
                seq=r.seq,
                turn_index=r.turn_index,
                role=r.role,
                message=r.message,
                token_count=r.token_count,
                model=r.model,
                created_at=r.created_at,
            )
            for r in reversed(rows)
        ]

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete the session row. Messages go with it through the foreign key cascade."""
        async with self._engine.begin() as conn:
            result = await conn.execute(delete(sessions).where(sessions.c.id == session_id))
        return result.rowcount > 0

    async def ping(self) -> None:
        """Raise if the database is unreachable."""
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def close(self) -> None:
        """Dispose the connection pool."""
        await self._engine.dispose()
