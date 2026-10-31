"""Shared fixtures. Integration fixtures for Postgres and Redis live in this file too."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Keep unit and API tests free of LiteLLM and of any developer machine config.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

PG_IMAGE = "pgvector/pgvector:pg16"
REDIS_IMAGE = "redis:7-alpine"

_TRUNCATE = (
    "TRUNCATE cg_messages, cg_sessions, procrastinate_jobs, procrastinate_workers, "
    "procrastinate_periodic_defers, procrastinate_events CASCADE"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark everything under tests/integration with the `integration` marker."""
    integration_dir = Path(__file__).parent / "integration"
    for item in items:
        if integration_dir in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _clean_ctx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove CTX_* variables so settings in tests only come from explicit arguments."""
    for key in list(os.environ):
        if key.startswith("CTX_"):
            monkeypatch.delenv(key, raising=False)


def _docker_container(factory):
    """Start a testcontainers container, or skip with a clear reason."""
    try:
        container = factory()
        container.start()
    except Exception as exc:  # Docker missing, no daemon, no registry access
        pytest.skip(f"no usable Docker for testcontainers ({type(exc).__name__}: {exc})")
    return container


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """libpq URL of a Postgres for tests: TEST_DATABASE_URL, else a container, else skip."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    from testcontainers.postgres import PostgresContainer

    container = _docker_container(
        lambda: PostgresContainer(
            PG_IMAGE, username="cg", password="cg", dbname="cg_test", driver=None
        )
    )
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    """The test database with all migrations applied, once per session."""
    from context_gateway.cli import upgrade_database
    from context_gateway.config import normalize_database_url

    libpq = normalize_database_url(database_url)
    upgrade_database("postgresql+psycopg://" + libpq.split("://", 1)[1])
    return libpq


@pytest.fixture
def pg_url(migrated_database_url: str) -> str:
    """libpq URL of the migrated test database, with all gateway tables emptied."""
    import psycopg

    with psycopg.connect(migrated_database_url, autocommit=True) as conn:
        conn.execute(_TRUNCATE)
    return migrated_database_url


@pytest.fixture
def pg_async_url(pg_url: str) -> str:
    """SQLAlchemy async URL for the same database."""
    return "postgresql+psycopg://" + pg_url.split("://", 1)[1]


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """URL of a Redis for tests: TEST_REDIS_URL, else a container, else skip."""
    url = os.environ.get("TEST_REDIS_URL")
    if url:
        yield url
        return
    from testcontainers.redis import RedisContainer

    container = _docker_container(lambda: RedisContainer(REDIS_IMAGE))
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"
    finally:
        container.stop()
