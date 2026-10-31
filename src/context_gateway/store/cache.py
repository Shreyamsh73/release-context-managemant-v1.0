"""Redis cache for session windows. A decorator around any `SessionStore`."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

from context_gateway.core.types import Scope, Window
from context_gateway.store.base import (
    AppendResult,
    NewMessage,
    SessionRecord,
    SessionStore,
    StoredMessage,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CachedSessionStore:
    """Caches the hot-path `load_window` call in Redis. Redis errors never fail a request."""

    def __init__(
        self,
        inner: SessionStore,
        redis: Redis,
        ttl_seconds: int,
        *,
        cache_max_turns: int = 200,
    ) -> None:
        self._inner = inner
        self._redis = redis
        self._ttl = ttl_seconds
        self._cache_max_turns = cache_max_turns

    def _key(self, session_id: UUID) -> str:
        return f"cg:window:{session_id}:{self._cache_max_turns}"

    async def _invalidate(self, session_id: UUID) -> None:
        try:
            await self._redis.delete(self._key(session_id))
        except Exception:
            logger.warning("redis cache invalidate failed", exc_info=True)

    # Cached read -----------------------------------------------------------------------
    async def load_window(self, session_id: UUID, *, max_turns: int | None) -> Window | None:
        """Serve the default window from Redis when present; fill the cache on a miss."""
        if max_turns != self._cache_max_turns:
            return await self._inner.load_window(session_id, max_turns=max_turns)
        key = self._key(session_id)
        try:
            raw = await self._redis.get(key)
        except Exception:
            logger.warning("redis cache read failed", exc_info=True)
            return await self._inner.load_window(session_id, max_turns=max_turns)
        if raw is not None:
            try:
                return Window.from_dict(json.loads(raw))
            except Exception:
                logger.warning("redis cache entry unreadable, reloading", exc_info=True)
        window = await self._inner.load_window(session_id, max_turns=max_turns)
        if window is not None:
            try:
                payload = json.dumps(window.to_dict(), ensure_ascii=False)
                await self._redis.set(key, payload, ex=self._ttl)
            except Exception:
                logger.warning("redis cache write failed", exc_info=True)
        return window

    # Writes that invalidate ------------------------------------------------------------
    async def append_messages(
        self, session_id: UUID, messages: Sequence[NewMessage]
    ) -> AppendResult:
        """Append, then drop the cached window."""
        try:
            return await self._inner.append_messages(session_id, messages)
        finally:
            await self._invalidate(session_id)

    async def apply_summary(
        self, session_id: UUID, *, summary: str, through_turn: int, expected_through_turn: int
    ) -> bool:
        """Compare-and-set the summary, then drop the cached window."""
        try:
            return await self._inner.apply_summary(
                session_id,
                summary=summary,
                through_turn=through_turn,
                expected_through_turn=expected_through_turn,
            )
        finally:
            await self._invalidate(session_id)

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete the session, then drop the cached window."""
        try:
            return await self._inner.delete_session(session_id)
        finally:
            await self._invalidate(session_id)

    # Plain delegation ------------------------------------------------------------------
    async def get_or_create_session(self, scope: Scope) -> SessionRecord:
        """Delegate to the inner store."""
        return await self._inner.get_or_create_session(scope)

    async def find_session(self, scope: Scope) -> SessionRecord | None:
        """Delegate to the inner store."""
        return await self._inner.find_session(scope)

    async def get_session(self, session_id: UUID) -> SessionRecord | None:
        """Delegate to the inner store."""
        return await self._inner.get_session(session_id)

    async def list_messages(self, session_id: UUID, *, limit: int) -> list[StoredMessage]:
        """Delegate to the inner store."""
        return await self._inner.list_messages(session_id, limit=limit)

    async def ping(self) -> None:
        """Ping the inner store. Redis health is reported separately by `/readyz`."""
        await self._inner.ping()

    async def close(self) -> None:
        """Close the inner store. The Redis client is closed by its owner."""
        await self._inner.close()
