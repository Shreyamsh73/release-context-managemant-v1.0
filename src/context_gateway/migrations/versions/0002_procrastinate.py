"""Apply the procrastinate job queue schema.

Upgrading procrastinate later needs its own migration files applied in a new revision.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-31
"""

from __future__ import annotations

import procrastinate.schema
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # exec_driver_sql keeps SQLAlchemy from parsing ":" in the SQL. no_parameters keeps psycopg
    # from parsing "%", which procrastinate uses in RAISE messages.
    op.get_bind().exec_driver_sql(
        procrastinate.schema.SchemaManager.get_schema(),
        execution_options={"no_parameters": True},
    )


def downgrade() -> None:
    raise NotImplementedError("Dropping the procrastinate schema is not supported")
