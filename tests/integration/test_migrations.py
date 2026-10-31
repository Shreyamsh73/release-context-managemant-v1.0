import uuid

import psycopg
import pytest
from psycopg import sql

from context_gateway.cli import build_alembic_config, upgrade_database


@pytest.fixture
def scratch_database(database_url):
    """A brand-new empty database so migrations run from zero."""
    name = f"cg_mig_{uuid.uuid4().hex[:10]}"
    base, _, _ = database_url.rpartition("/")
    admin = psycopg.connect(database_url, autocommit=True)
    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = f"{base}/{name}"
    try:
        yield url
    finally:
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )
        admin.close()


def tables(url):
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


def sync(url):
    return "postgresql+psycopg://" + url.split("://", 1)[1]


# spec 55
def test_upgrade_head_creates_gateway_and_procrastinate_tables(scratch_database):
    assert tables(scratch_database) == set()
    upgrade_database(sync(scratch_database))
    found = tables(scratch_database)
    assert {"cg_sessions", "cg_messages", "alembic_version"} <= found
    assert {
        "procrastinate_jobs",
        "procrastinate_workers",
        "procrastinate_events",
        "procrastinate_periodic_defers",
    } <= found
    with psycopg.connect(scratch_database) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchall() == [("0002",)]


def test_running_upgrade_twice_is_a_no_op(scratch_database):
    upgrade_database(sync(scratch_database))
    before = tables(scratch_database)
    upgrade_database(sync(scratch_database))
    assert tables(scratch_database) == before
    with psycopg.connect(scratch_database) as conn:
        assert conn.execute("SELECT count(*) FROM alembic_version").fetchone()[0] == 1


def test_schema_matches_the_spec(scratch_database):
    upgrade_database(sync(scratch_database))
    with psycopg.connect(scratch_database) as conn:
        columns = conn.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'cg_messages' ORDER BY ordinal_position"
        ).fetchall()
        constraints = {
            r[0]
            for r in conn.execute(
                "SELECT conname FROM pg_constraint WHERE conrelid IN "
                "('cg_sessions'::regclass, 'cg_messages'::regclass)"
            ).fetchall()
        }
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'cg_messages'"
            ).fetchall()
        }
    assert columns == [
        ("id", "bigint", "NO"),
        ("session_id", "uuid", "NO"),
        ("seq", "integer", "NO"),
        ("turn_index", "integer", "NO"),
        ("role", "text", "NO"),
        ("message", "jsonb", "NO"),
        ("token_count", "integer", "NO"),
        ("model", "text", "YES"),
        ("created_at", "timestamp with time zone", "NO"),
    ]
    assert {"uq_cg_sessions_scope", "uq_cg_messages_seq"} <= constraints
    assert "ix_cg_messages_session_turn" in indexes


def test_alembic_config_escapes_percent_signs():
    config = build_alembic_config("postgresql+psycopg://u:p%40ss@h/db")
    assert config.get_main_option("sqlalchemy.url") == "postgresql+psycopg://u:p%40ss@h/db"
    assert config.get_main_option("script_location").endswith("migrations")
