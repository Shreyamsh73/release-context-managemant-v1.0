"""Procrastinate tasks."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from uuid import UUID

from procrastinate.exceptions import AlreadyEnqueued

from context_gateway.config import Settings, get_settings
from context_gateway.core.summary import Summarizer, SummaryOutcome
from context_gateway.core.tokens import TokenCounter
from context_gateway.deps import build_llm, build_redis, build_store, make_counter
from context_gateway.llm.base import LLMClient
from context_gateway.logging import kv
from context_gateway.store.base import SessionStore
from context_gateway.worker.app import app

logger = logging.getLogger(__name__)

TASK_NAME = "summarize_session"
QUEUE_NAME = "summaries"


@dataclass
class Runtime:
    """Long-lived objects of the worker process, built on first use."""

    settings: Settings
    store: SessionStore
    llm: LLMClient
    counter: TokenCounter
    summarizer: Summarizer


_runtime: Runtime | None = None


def build_runtime(settings: Settings) -> Runtime:
    """Build the store (with the Redis cache when configured), LLM client and counter."""
    store = build_store(settings, build_redis(settings))
    llm = build_llm(settings)
    counter = make_counter(settings)
    return Runtime(settings, store, llm, counter, Summarizer(store, llm, counter, settings))


def get_runtime() -> Runtime:
    """The per-process runtime. Created lazily so importing this module stays cheap."""
    global _runtime
    if _runtime is None:
        _runtime = build_runtime(get_settings())
    return _runtime


def set_runtime(runtime: Runtime | None) -> None:
    """Replace the runtime. Used by tests; pass None to reset."""
    global _runtime
    _runtime = runtime


@app.task(name=TASK_NAME, queue=QUEUE_NAME, retry=3)
async def summarize_session(session_id: str) -> None:
    """Fold old turns of a session into its rolling summary."""
    runtime = get_runtime()
    outcome = await runtime.summarizer.run(UUID(session_id))
    logger.info("summary job finished %s", kv(session=session_id, outcome=outcome.value))
    if outcome is SummaryOutcome.PARTIAL:
        # More old turns remain. Queue another pass; a waiting job for the session is enough.
        with contextlib.suppress(AlreadyEnqueued):
            await summarize_session.configure(
                queueing_lock=session_id, lock=session_id
            ).defer_async(session_id=session_id)
