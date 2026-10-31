"""Dependency container and FastAPI dependency getters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

from context_gateway.config import Settings
from context_gateway.core.context import ContextBuilder
from context_gateway.core.summary import Summarizer
from context_gateway.core.tokens import TokenCounter, make_token_counter
from context_gateway.jobs.base import JobQueue
from context_gateway.llm.base import LLMClient
from context_gateway.store.base import SessionStore

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class Deps:
    """Everything a request handler needs. Built once in the app lifespan."""

    settings: Settings
    llm: LLMClient
    store: SessionStore
    queue: JobQueue
    counter: TokenCounter
    builder: ContextBuilder
    summarizer: Summarizer
    redis: Redis | None = None

    async def close(self) -> None:
        """Close the queue first so running jobs can finish, then the clients."""
        for name, closer in (
            ("queue", self.queue.close),
            ("llm", self.llm.close),
            ("store", self.store.close),
            ("redis", self._close_redis),
        ):
            try:
                await closer()
            except Exception:
                logger.warning("error closing %s", name, exc_info=True)

    async def _close_redis(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()


def get_deps(request: Request) -> Deps:
    """FastAPI dependency returning the app's `Deps`."""
    return request.app.state.deps


def build_llm(settings: Settings) -> LLMClient:
    """Build the LLM client named by `CTX_LLM_BACKEND`."""
    if settings.llm_backend == "echo":
        from context_gateway.llm.echo import EchoLLMClient

        return EchoLLMClient()
    from context_gateway.llm.litellm_client import LiteLLMClient

    return LiteLLMClient()


def build_redis(settings: Settings) -> Redis | None:
    """Build the Redis client, or None when `CTX_REDIS_URL` is empty."""
    if not settings.redis_url:
        return None
    from redis.asyncio import Redis

    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        health_check_interval=0,
    )


def build_store(settings: Settings, redis: Redis | None = None) -> SessionStore:
    """Build the session store, wrapped in the Redis cache when a client is given."""
    store: SessionStore
    if settings.store == "memory":
        from context_gateway.store.memory import InMemorySessionStore

        store = InMemorySessionStore()
    else:
        from context_gateway.store.postgres import PostgresSessionStore

        store = PostgresSessionStore(settings.async_database_url)
    if redis is not None:
        from context_gateway.store.cache import CachedSessionStore

        store = CachedSessionStore(
            store,
            redis,
            ttl_seconds=settings.cache_ttl_seconds,
            cache_max_turns=settings.window_max_turns,
        )
    return store


def build_queue(settings: Settings, summarizer: Summarizer) -> JobQueue:
    """Build the job queue named by `CTX_JOB_BACKEND`."""
    if settings.job_backend == "procrastinate":
        from context_gateway.jobs.procrastinate_queue import ProcrastinateJobQueue
        from context_gateway.worker.app import app

        return ProcrastinateJobQueue(app)
    from context_gateway.jobs.inline import InlineJobQueue

    return InlineJobQueue(summarizer.run)


def make_counter(settings: Settings) -> TokenCounter:
    """Build the token counter named by `CTX_TOKEN_COUNTER`."""
    return make_token_counter(settings.token_counter)
