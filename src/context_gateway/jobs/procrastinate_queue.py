"""Job queue backed by procrastinate (Postgres). Jobs run in a separate worker process."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from procrastinate.exceptions import AlreadyEnqueued

from context_gateway.logging import kv

if TYPE_CHECKING:
    from procrastinate import App

logger = logging.getLogger(__name__)

TASK_NAME = "summarize_session"


class ProcrastinateJobQueue:
    """Defers `summarize_session` jobs. One waiting job per session at most."""

    def __init__(self, app: App) -> None:
        self._app = app

    async def start(self) -> None:
        """Open the procrastinate connection pool."""
        await self._app.open_async()

    async def enqueue_summarize(self, session_id: UUID) -> None:
        """Defer a job. A second request for a session that already has one waiting is a no-op."""
        key = str(session_id)
        try:
            await self._app.configure_task(TASK_NAME, queueing_lock=key, lock=key).defer_async(
                session_id=key
            )
        except AlreadyEnqueued:
            logger.debug("summary job already queued %s", kv(session=key))

    async def close(self) -> None:
        """Close the connection pool."""
        await self._app.close_async()
