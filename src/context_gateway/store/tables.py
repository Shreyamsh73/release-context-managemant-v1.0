"""SQLAlchemy Core table definitions. All names are prefixed `cg_`."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

sessions = Table(
    "cg_sessions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("app_id", Text, nullable=False),
    Column("user_id", Text, nullable=False),
    Column("session_key", Text, nullable=False),
    Column("summary", Text, nullable=True),
    Column("summarized_through_turn", Integer, nullable=False, server_default=text("0")),
    Column("last_seq", Integer, nullable=False, server_default=text("0")),
    Column("last_turn", Integer, nullable=False, server_default=text("0")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("app_id", "user_id", "session_key", name="uq_cg_sessions_scope"),
)

messages = Table(
    "cg_messages",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "session_id",
        UUID(as_uuid=True),
        ForeignKey("cg_sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("seq", Integer, nullable=False),
    Column("turn_index", Integer, nullable=False),
    Column("role", Text, nullable=False),
    Column("message", JSONB, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("model", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("session_id", "seq", name="uq_cg_messages_seq"),
    Index("ix_cg_messages_session_turn", "session_id", "turn_index"),
)
