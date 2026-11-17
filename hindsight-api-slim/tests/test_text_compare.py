"""The consolidation worker compares observation text before and
after the model call on the server and writes durable, timestamped change
records — TEXT_CHANGE for a rewrite, FOLD_IN for a source addition, SRC_EDIT
for a source-memory edit/refine/supersede/invalidate.

``compare_text`` is server-side and deterministic; the model only
produces text and is never asked whether text changed. A fold-in that adds no
new source id and changes no text records nothing. Retries are deduplicated by
stable event identity, never by before/after values.
"""

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.pg import consolidation as pg_con
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.response_models import MemoryFact
from tests.consolidation_actions import execute_create_action, execute_update_action
from tests.test_retain_same_document_concurrency import _StubEmbeddings

OLD_HEAD = "300fe58170e9"  # alembic head before the change-record migration


@pytest_asyncio.fixture
async def memory_engine(pg0_db_url: str) -> AsyncIterator[MemoryEngine]:
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
    )
    await memory.initialize()
    try:
        yield memory
    finally:
        await memory.close()


async def _change_events_since(*, conn, fq_table, bank_id: str, subject_id: uuid.UUID, baseline_seq: int) -> list:
    """Test helper The change records for ``subject_id`` above
    ``baseline_seq``, oldest first. Lives here, not in product code: the
    product graph rule answers through the provenance batch read, and this
    seq-window query exists only for the baseline assertions."""
    return await conn.fetch(
        f"""
        SELECT seq, event_id, bank_id, subject_id, event_type, before_text, after_text, detail, created_at
        FROM {fq_table("memory_change_events")}
        WHERE bank_id = $1 AND subject_id = $2 AND seq > $3
        ORDER BY seq
        """,
        bank_id,
        uuid.UUID(str(subject_id)),
        baseline_seq,
    )


async def _make_bank(memory: MemoryEngine) -> str:
    bank_id = f"test-text-compare-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    return bank_id


async def _drop_bank(memory: MemoryEngine, bank_id: str) -> None:
    await memory.delete_bank(bank_id, request_context=RequestContext())


async def _seed_memory_unit(memory: MemoryEngine, bank_id: str, text: str, fact_type: str = "world") -> uuid.UUID:
    unit_id = uuid.uuid4()
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags) VALUES ($1, $2, $3, $4, $5)",
            unit_id,
            bank_id,
            text,
            fact_type,
            ["source:new"],
        )
    return unit_id


async def _create_observation(memory: MemoryEngine, bank_id: str, source_id: uuid.UUID, text: str) -> str:
    await execute_create_action(
        pool=await memory._get_backend(),
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[source_id],
        text=text,
        source_fact_tags=["source:new"],
    )
    result = await memory.list_memory_units(bank_id=bank_id, fact_type="observation", request_context=RequestContext())
    assert result["total"] == 1
    return result["items"][0]["id"]


async def _update_observation(
    memory: MemoryEngine,
    bank_id: str,
    source_id: uuid.UUID,
    observation_id: str,
    before_text: str,
    new_text: str,
) -> None:
    snapshot = MemoryFact(id=observation_id, text=before_text, fact_type="observation", tags=["source:new"])
    await execute_update_action(
        pool=await memory._get_backend(),
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[source_id],
        observation_id=observation_id,
        new_text=new_text,
        observations=[snapshot],
        source_fact_tags=["source:new"],
    )


async def _events(memory: MemoryEngine, bank_id: str) -> list:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT seq, event_id, subject_id, event_type, before_text, after_text, detail, created_at "
            f"FROM {fq_table('memory_change_events')} WHERE bank_id = $1 ORDER BY seq",
            bank_id,
        )
    # The pool registers no JSONB codec, so detail arrives as raw JSON text.
    return [dict(r, detail=json.loads(r["detail"]) if isinstance(r["detail"], str) else r["detail"]) for r in rows]


# --------------------------------------------------------------------- pure compare


def test_text_compare_pure_changed_and_unchanged() -> None:
    """The verdict is a deterministic string compare on the server."""
    assert C.compare_text("Alice plays cello.", "Alice plays cello.") is False
    assert C.compare_text("Alice plays cello.", "Alice still plays cello.") is True
    assert C.compare_text("", "") is False
    assert C.compare_text("", "x") is True
    # exact compare: a whitespace-only difference is a change (fail-safe stale)
    assert C.compare_text("a b", "a  b") is True


def test_text_compare_pure_model_never_asked() -> None:
    """The consolidation model only produces text: no action schema carries a
    changed/not-changed verdict field for the server to read."""
    for action in (C._CreateAction, C._UpdateAction, C._DeleteAction):
        for field_name in action.model_fields:
            assert "changed" not in field_name, f"{action.__name__} carries verdict field {field_name!r}"


# --------------------------------------------------------------------- rewrite path


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_rewrite_changed_writes_durable_record(memory_engine: MemoryEngine) -> None:
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        await _update_observation(
            memory_engine, bank_id, source_id, observation_id, "Alice plays cello.", "Alice sings."
        )
        rows = await _events(memory_engine, bank_id)
        assert len(rows) == 1
        (row,) = rows
        assert row["event_type"] == "TEXT_CHANGE"
        assert str(row["subject_id"]) == observation_id
        assert row["before_text"] == "Alice plays cello."
        assert row["after_text"] == "Alice sings."
        assert row["created_at"] is not None
        assert row["event_id"]
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_rewrite_unchanged_writes_no_record(memory_engine: MemoryEngine) -> None:
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        await _update_observation(
            memory_engine, bank_id, source_id, observation_id, "Alice plays cello.", "Alice plays cello."
        )
        assert await _events(memory_engine, bank_id) == []
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_a_to_b_to_a_to_b_records_each_transition(memory_engine: MemoryEngine) -> None:
    """Distinct committed transitions are recorded even when values repeat:
    A->B, B->A, A->B is three records, not two."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        await _update_observation(memory_engine, bank_id, source_id, observation_id, "Alice plays cello.", "B")
        first_seq = (await _events(memory_engine, bank_id))[0]["seq"]
        await _update_observation(memory_engine, bank_id, source_id, observation_id, "B", "Alice plays cello.")
        await _update_observation(memory_engine, bank_id, source_id, observation_id, "Alice plays cello.", "B")
        rows = await _events(memory_engine, bank_id)
        assert [r["event_type"] for r in rows] == ["TEXT_CHANGE", "TEXT_CHANGE", "TEXT_CHANGE"]
        assert [(r["before_text"], r["after_text"]) for r in rows] == [
            ("Alice plays cello.", "B"),
            ("B", "Alice plays cello."),
            ("Alice plays cello.", "B"),
        ]
        assert rows[0]["seq"] < rows[1]["seq"] < rows[2]["seq"]

        # A refresh between the first and last transitions stamps a baseline;
        # the test helper answers against that explicit baseline.
        async with memory_engine._pool.acquire() as conn:
            above = await _change_events_since(
                conn=conn, fq_table=fq_table, bank_id=bank_id, subject_id=observation_id, baseline_seq=first_seq
            )
        assert [r["seq"] for r in above] == [rows[1]["seq"], rows[2]["seq"]]
        async with memory_engine._pool.acquire() as conn:
            none_above = await _change_events_since(
                conn=conn, fq_table=fq_table, bank_id=bank_id, subject_id=observation_id, baseline_seq=rows[2]["seq"]
            )
        assert none_above == []
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_rollback_leaves_no_record(memory_engine: MemoryEngine) -> None:
    """The record commits transactionally with the observation write: a
    rolled-back rewrite leaves neither the new text nor a record."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        store = get_memories()
        with pytest.raises(RuntimeError):
            async with C.acquire_with_retry(await memory_engine._get_backend()) as conn:
                async with conn.transaction():
                    await store.rewrite_observation(
                        conn=conn,
                        fq_table=fq_table,
                        bank_id=bank_id,
                        observation_id=observation_id,
                        text="Alice sings.",
                        embedding=None,
                        source_memory_ids=[source_id],
                        tags=["source:new"],
                        bounds=C._TemporalBounds(),
                        previous=MemoryFact(
                            id=observation_id, text="Alice plays cello.", fact_type="observation", tags=[]
                        ),
                    )
                    raise RuntimeError("abort the batch")
        rows = await _events(memory_engine, bank_id)
        assert rows == []
        updated = await memory_engine.get_memory_unit(
            bank_id=bank_id, memory_id=observation_id, request_context=RequestContext()
        )
        assert updated["text"] == "Alice plays cello."
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_retry_deduplicated_by_event_identity(memory_engine: MemoryEngine) -> None:
    """Same stable event identity replays to one record; the same values under
    a different identity are a distinct transition and record again."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        event_id = f"consolidation:{observation_id}:op-1"
        async with memory_engine._pool.acquire() as conn:
            async with conn.transaction():
                await pg_con.record_change_event(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    subject_id=observation_id,
                    event_type="TEXT_CHANGE",
                    event_id=event_id,
                    before_text="x",
                    after_text="y",
                    detail=None,
                )
                await pg_con.record_change_event(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    subject_id=observation_id,
                    event_type="TEXT_CHANGE",
                    event_id=event_id,
                    before_text="x",
                    after_text="y",
                    detail=None,
                )
                await pg_con.record_change_event(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    subject_id=observation_id,
                    event_type="TEXT_CHANGE",
                    event_id=f"consolidation:{observation_id}:op-2",
                    before_text="x",
                    after_text="y",
                    detail=None,
                )
        rows = await _events(memory_engine, bank_id)
        assert len(rows) == 2
    finally:
        await _drop_bank(memory_engine, bank_id)


# --------------------------------------------------------------------- fold-in path


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_fold_in_no_text_change_writes_source_addition(memory_engine: MemoryEngine) -> None:
    """A fold-in that adds a source id records a source-addition record
    even when text is unchanged, and no text record."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        new_source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice practices piano.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        store = get_memories()
        async with memory_engine._pool.acquire() as conn:
            async with conn.transaction():
                folded = await store.fold_sources_into_observation(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    observation_id=observation_id,
                    expected_text="Alice plays cello.",
                    merged_text="Alice plays cello.",
                    source_memory_ids=[new_source_id],
                    bounds=C._TemporalBounds(),
                    embeddings=memory_engine.embeddings,
                )
        assert folded is True
        rows = await _events(memory_engine, bank_id)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "FOLD_IN"
        assert rows[0]["before_text"] == rows[0]["after_text"] == "Alice plays cello."
        assert rows[0]["detail"]["added_source_ids"] == [str(new_source_id)]
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_fold_in_no_new_source_no_change_writes_nothing(memory_engine: MemoryEngine) -> None:
    """A fold-in that adds no new source id and changes no text records nothing."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        store = get_memories()
        async with memory_engine._pool.acquire() as conn:
            async with conn.transaction():
                folded = await store.fold_sources_into_observation(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    observation_id=observation_id,
                    expected_text="Alice plays cello.",
                    merged_text="Alice plays cello.",
                    source_memory_ids=[source_id],
                    bounds=C._TemporalBounds(),
                    embeddings=memory_engine.embeddings,
                )
        assert folded is True
        assert await _events(memory_engine, bank_id) == []
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_fold_in_text_change_writes_both_records(memory_engine: MemoryEngine) -> None:
    """A fold that changes text and adds a source records the text change and
    the source addition."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        new_source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice practices piano.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        store = get_memories()
        async with memory_engine._pool.acquire() as conn:
            async with conn.transaction():
                folded = await store.fold_sources_into_observation(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    observation_id=observation_id,
                    expected_text="Alice plays cello.",
                    merged_text="Alice plays cello and piano.",
                    source_memory_ids=[new_source_id],
                    bounds=C._TemporalBounds(),
                    embeddings=memory_engine.embeddings,
                )
        assert folded is True
        rows = await _events(memory_engine, bank_id)
        assert sorted(r["event_type"] for r in rows) == ["FOLD_IN", "TEXT_CHANGE"]
        text_change = next(r for r in rows if r["event_type"] == "TEXT_CHANGE")
        assert (text_change["before_text"], text_change["after_text"]) == (
            "Alice plays cello.",
            "Alice plays cello and piano.",
        )
    finally:
        await _drop_bank(memory_engine, bank_id)


# --------------------------------------------------------------------- source writes


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_src_edit_writes_record(memory_engine: MemoryEngine) -> None:
    """A source-memory edit (the apply_edit write path behind edit, refine and
    supersede) records a durable SRC_EDIT record transactionally."""
    bank_id = await _make_bank(memory_engine)
    try:
        unit_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        store = get_memories()
        async with memory_engine._pool.acquire() as conn:
            async with conn.transaction():
                await store.apply_edit(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    unit_id=str(unit_id),
                    text="Alice sings.",
                    context=None,
                    fact_type="world",
                    occurred_start=None,
                    occurred_end=None,
                    event_date=None,
                    mentioned_at=None,
                    entity_ids=None,
                )
        rows = await _events(memory_engine, bank_id)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "SRC_EDIT"
        assert str(rows[0]["subject_id"]) == str(unit_id)
        assert (rows[0]["before_text"], rows[0]["after_text"]) == ("Alice plays cello.", "Alice sings.")
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_invalidate_writes_record(memory_engine: MemoryEngine) -> None:
    """An invalidate records a durable SRC_EDIT record (model SRC_EDIT covers
    edit/refine/supersede/invalidate) transactionally with the archive move."""
    bank_id = await _make_bank(memory_engine)
    try:
        unit_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        store = get_memories()
        async with memory_engine._pool.acquire() as conn:
            async with conn.transaction():
                assert (
                    await store.invalidate_memory(
                        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=str(unit_id), reason="wrong fact"
                    )
                    is True
                )
        rows = await _events(memory_engine, bank_id)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "SRC_EDIT"
        assert str(rows[0]["subject_id"]) == str(unit_id)
        assert rows[0]["detail"]["op"] == "invalidate"
    finally:
        await _drop_bank(memory_engine, bank_id)


# --------------------------------------------------------------------- migration


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_text_compare_migration_runs_against_pre_change_bank(pg0_db_url: str) -> None:
    """The change-record migration creates its table on a bank that predates
    it, without losing the bank's existing rows."""
    import asyncpg
    from alembic import command as alembic_command
    from alembic.config import Config

    schema = f"text_compare_mig_{uuid.uuid4().hex[:8]}"
    script_location = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

    def _cfg() -> Config:
        cfg = Config()
        cfg.set_main_option("script_location", script_location)
        cfg.set_main_option("sqlalchemy.url", pg0_db_url)
        cfg.set_main_option("prepend_sys_path", ".")
        cfg.set_main_option("path_separator", "os")
        cfg.set_main_option("target_schema", schema)
        return cfg

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        # Bring the schema to the head that predates the change-record migration.
        alembic_command.upgrade(_cfg(), OLD_HEAD)
        unit_id = uuid.uuid4()
        await conn.execute(
            f'INSERT INTO "{schema}".memory_units (id, bank_id, text, fact_type, tags) VALUES ($1, $2, $3, $4, $5)',
            unit_id,
            "pre-change-bank",
            "Alice plays cello.",
            "world",
            [],
        )
        missing = await conn.fetchval(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = $1 AND table_name = 'memory_change_events'",
            schema,
        )
        assert missing == 0
        # Upgrade to head: the migration creates the table.
        alembic_command.upgrade(_cfg(), "head")
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = $1 AND table_name = 'memory_change_events'",
                schema,
            )
            == 1
        )
        assert (
            await conn.fetchval(f'SELECT text FROM "{schema}".memory_units WHERE id = $1', unit_id)
            == "Alice plays cello."
        )
        cols = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'memory_change_events'",
                schema,
            )
        }
        assert {
            "seq",
            "event_id",
            "bank_id",
            "subject_id",
            "event_type",
            "before_text",
            "after_text",
            "detail",
            "created_at",
        } <= cols
    finally:
        await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await conn.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_rewrite_unchanged_text_new_source_records_fold_in(memory_engine: MemoryEngine) -> None:
    """A consolidation UPDATE (rewrite) that adds a source id
    with unchanged text records a FOLD_IN, exactly like the fold path."""
    from hindsight_api.engine.memories.pg.consolidation import rewrite_observation
    from hindsight_api.engine.schema import fq_store_table

    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        new_source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice practices piano.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        backend = await memory_engine._get_backend()
        async with backend.acquire() as conn:
            async with conn.transaction():
                ok = await rewrite_observation(
                    conn=conn,
                    fq_table=fq_store_table,
                    observation_id=observation_id,
                    text="Alice plays cello.",
                    embedding=None,
                    source_memory_ids=[source_id, new_source_id],
                    tags=["source:new"],
                    bounds=C._TemporalBounds(),
                )
        assert ok is True
        rows = await _events(memory_engine, bank_id)
        fold_ins = [r for r in rows if r["event_type"] == "FOLD_IN"]
        assert len(fold_ins) == 1, rows
        assert fold_ins[0]["detail"]["added_source_ids"] == [str(new_source_id)]
        assert not [r for r in rows if r["event_type"] == "TEXT_CHANGE"]
    finally:
        await _drop_bank(memory_engine, bank_id)


@pytest.mark.asyncio
async def test_record_change_event_is_noop_on_oracle(memory_engine: MemoryEngine, monkeypatch) -> None:
    """The change-record table exists only on Postgres, so recording on
    Oracle must not issue the INSERT (it would fail on the missing table)."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr("hindsight_api.engine.schema._is_oracle", lambda: True)
    conn = AsyncMock()
    await pg_con.record_change_event(
        conn=conn,
        fq_table=fq_table,
        bank_id="b",
        subject_id=uuid.uuid4(),
        event_type="TEXT_CHANGE",
        before_text="a",
        after_text="b",
    )
    conn.execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_replayed_record_call_deduplicates_without_explicit_id(memory_engine: MemoryEngine) -> None:
    """A retried record write with the same arguments replays to one
    record even when the caller supplied no event id (the identity is
    derived from the transition, not minted per retry)."""
    bank_id = await _make_bank(memory_engine)
    try:
        source_id = await _seed_memory_unit(memory_engine, bank_id, "Alice plays cello.")
        observation_id = await _create_observation(memory_engine, bank_id, source_id, "Alice plays cello.")
        async with memory_engine._pool.acquire() as conn:
            for _ in range(2):
                async with conn.transaction():
                    await pg_con.record_change_event(
                        conn=conn,
                        fq_table=fq_table,
                        bank_id=bank_id,
                        subject_id=observation_id,
                        event_type="TEXT_CHANGE",
                        before_text="x",
                        after_text="y",
                        detail=None,
                    )
        rows = await _events(memory_engine, bank_id)
        assert len(rows) == 1, rows
    finally:
        await _drop_bank(memory_engine, bank_id)
