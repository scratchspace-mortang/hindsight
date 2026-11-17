"""Add memory_change_events table for graph-staleness change records.

Revision ID: d4e5f6a7b8c9
Revises: 300fe58170e9
Create Date: 2026-11-17

Durable, timestamped change records (not a bare boolean): every observation
text change, source fold-in, and source-memory edit/refine/supersede/invalidate
writes one row here transactionally with the write it describes, so a page's
provenance baseline (a seq value) can answer "did anything change since I was
built" per page. seq is a monotonically increasing identity — the baseline is
a sequence number, never a boolean — and event_id is the stable operation
identity used to deduplicate retries.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "300fe58170e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_change_events (
            seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            event_id TEXT NOT NULL,
            bank_id TEXT NOT NULL,
            subject_id UUID NOT NULL,
            event_type TEXT NOT NULL,
            before_text TEXT,
            after_text TEXT,
            detail JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_change_events_bank_event "
        f"ON {schema}memory_change_events (bank_id, event_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_change_events_subject_seq "
        f"ON {schema}memory_change_events (bank_id, subject_id, seq)"
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_change_events_subject_seq")
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_memory_change_events_bank_event")
    op.execute(f"DROP TABLE IF EXISTS {schema}memory_change_events")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
