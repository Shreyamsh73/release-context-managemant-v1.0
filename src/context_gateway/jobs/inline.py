"""Inline job queue: runs summaries as asyncio tasks inside the gateway process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

from context_gateway.core.summary import SummaryOutcome
from context_gateway.logging import kv

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_SECONDS = 10.0


class InlineJobQueue:
    """One asyncio task per session at a time. Good for single-process deployments."""

    def __init__(self, runner: Callable[[UUID], Awaitable[SummaryOutcome]]) -> None:
        self._runner = runner
        self._tasks: set[asyncio.Task[None]] = set()
        self._running: dict[UUID, asyncio.Task[None]] = {}
        self._closing = False

    async def start(self) -> None:
        """Nothing to open."""

    async def enqueue_summarize(self, session_id: UUID) -> None:
        """Start a summary task unless one is already running for this session."""
        if self._closing:
            return
        running = self._running.get(session_id)
        if running is not None and not running.done():
            return
        task = asyncio.create_task(self._run(session_id), name=f"summarize-{session_id}")
        self._tasks.add(task)
        self._running[session_id] = task
        task.add_done_callback(lambda done, sid=session_id: self._forget(sid, done))

    def _forget(self, session_id: UUID, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if self._running.get(session_id) is task:
            del self._running[session_id]

    async def _run(self, session_id: UUID) -> None:
        try:
            outcome = await self._runner(session_id)
            if outcome is SummaryOutcome.PARTIAL:
                outcome = await self._runner(session_id)
            logger.debug("summary job done %s", kv(session=session_id, outcome=outcome))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("summary job failed %s", kv(session=session_id))

    async def drain(self) -> None:
        """Wait for all running tasks. Used by tests and by `close`."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        """Wait up to 10 seconds for pending tasks, then cancel the rest."""
        self._closing = True
        try:
            await asyncio.wait_for(self.drain(), timeout=CLOSE_TIMEOUT_SECONDS)
        except TimeoutError:
            for task in list(self._tasks):
                task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
