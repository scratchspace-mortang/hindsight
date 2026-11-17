"""Batch provenance read and batch/single agreement.

``provenance_batch_read`` returns, per page, the
cited ids, the read ids, the observation sources expanded one hop, and the
stored text-change signal, in one batch — never a query per page — and never
traverses memory-to-memory edges.

The batch decision
(``compute_mental_models_are_stale``) answers the same rule as the single
path (``compute_mental_model_is_stale``) per page, including the deletion
test the batch used to skip and excluding the bank-watermark shortcut from
any page the shortcut cannot settle (deletion raises no watermark, and a
graph page has no scope question to settle).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.base import ProvenanceBatch
from hindsight_api.engine.memories.pg import consolidation as pg_con
from hindsight_api.engine.memories.pg.reads import provenance_batch_read
from hindsight_api.engine.memory_engine import (
    MemoryEngine,
    _mental_model_stale_scope_from_row,
    fq_table,
)
from hindsight_api.engine.schema import fq_store_table

PAGE_CONTENT = "## Conventions\n\nAutocommit is disabled so changes can be reviewed.\n\nTests run on every push.\n"


def _fact(fact_id: str, text: str = "a fact", fact_type: str = "observation") -> dict[str, Any]:
    return {"id": fact_id, "text": text, "type": fact_type, "context": None}


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_memory(memory: MemoryEngine, conn, bank_id: str, text: str) -> uuid.UUID:
    """Seed one world memory through the store (like the retraction tests)."""
    store = get_memories()
    fact = SimpleNamespace(
        fact_text=text,
        embedding=(await memory.embeddings.encode([text]))[0],
        fact_type="world",
        tags=[],
        context=None,
        document_id=None,
        chunk_id=None,
        metadata=None,
        observation_scopes=None,
        entities=[],
        causal_relations=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        attachment_ids=[],
    )
    unit_ids = await store.insert_facts(
        conn=conn, ops=memory._backend.ops, bank_id=bank_id, facts=[fact], document_id=None
    )
    await store.mark_consolidated(
        conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc)
    )
    return uuid.UUID(unit_ids[0])


async def _insert_observation(conn, bank_id: str, text: str, source_ids: list[uuid.UUID] | None = None) -> uuid.UUID:
    obs_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, source_memory_ids) "
        "VALUES ($1, $2, $3, 'observation', $4, $5)",
        obs_id,
        bank_id,
        text,
        [],
        source_ids or [],
    )
    return obs_id


async def _record_event(conn, bank_id: str, subject_id: uuid.UUID, event_type: str) -> int:
    """Commit one durable change record; return its seq."""
    async with conn.transaction():
        await pg_con.record_change_event(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            subject_id=subject_id,
            event_type=event_type,
            before_text="before",
            after_text="after",
        )
    return await conn.fetchval(
        f"SELECT max(seq) FROM {fq_table('memory_change_events')} WHERE bank_id = $1 AND subject_id = $2",
        bank_id,
        subject_id,
    )


async def _max_seq(conn, bank_id: str) -> int:
    value = await conn.fetchval(
        f"SELECT COALESCE(max(seq), 0) FROM {fq_table('memory_change_events')} WHERE bank_id = $1",
        bank_id,
    )
    return int(value or 0)


async def _create_page(
    memory: MemoryEngine,
    bank_id: str,
    request_context: RequestContext,
    *,
    staleness_check: str = "graph",
    graph_stale: str = "cited_memories",
) -> str:
    model = await memory.create_mental_model(
        bank_id=bank_id,
        name="Conventions",
        source_query="What are the project's conventions?",
        content=PAGE_CONTENT,
        request_context=request_context,
        trigger={
            "mode": "full",
            "refresh_after_consolidation": True,
            "staleness_check": staleness_check,
            "graph_stale": graph_stale,
        },
    )
    return model["id"]


async def _stamp_provenance(
    conn,
    bank_id: str,
    mm_id: str,
    *,
    based_on: dict[str, Any] | None,
    graph_provenance: dict[str, Any] | None,
    since_offset: timedelta = timedelta(0),
) -> None:
    """Write the reflect_response a successful refresh would have left."""
    reflect_response: dict[str, Any] = {"text": PAGE_CONTENT, "mental_models": []}
    if based_on is not None:
        reflect_response["based_on"] = based_on
    if graph_provenance is not None:
        reflect_response["graph_provenance"] = graph_provenance
    await conn.execute(
        f"UPDATE {fq_store_table('mental_models')} SET reflect_response = $3::jsonb, "
        f"last_memory_seen_at = now() + $4::interval, last_refreshed_at = now() WHERE bank_id = $1 AND id = $2",
        bank_id,
        mm_id,
        json.dumps(reflect_response),
        since_offset,
    )


def _prov(read_ids: list[str], baseline: int) -> dict[str, Any]:
    return {"read_memory_ids": read_ids, "captured_empty": False, "baseline_seq": baseline}


async def _mm_row(conn, bank_id: str, mm_id: str):
    row = await conn.fetchrow(
        f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
        f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
        bank_id,
        mm_id,
    )
    assert row is not None
    return row


async def _single(memory: MemoryEngine, conn, bank_id: str, mm_id: str) -> bool:
    row = await _mm_row(conn, bank_id, mm_id)
    return await memory.compute_mental_model_is_stale(conn, bank_id, row)


async def _batch(memory: MemoryEngine, conn, bank_id: str, mm_ids: list[str]) -> dict[str, bool]:
    rows = {mm_id: await _mm_row(conn, bank_id, mm_id) for mm_id in mm_ids}
    return await memory.compute_mental_models_are_stale(
        conn,
        bank_id,
        {mm_id: _mental_model_stale_scope_from_row(row, key=mm_id) for mm_id, row in rows.items()},
    )


class _CountingConn:
    """A real connection wrapper that records every statement it issues."""

    def __init__(self, conn):
        self._conn = conn
        self.statements: list[str] = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._conn, name)

    async def fetch(self, sql, *args):
        self.statements.append(str(sql))
        return await self._conn.fetch(sql, *args)

    async def fetchrow(self, sql, *args):
        self.statements.append(str(sql))
        return await self._conn.fetchrow(sql, *args)

    async def fetchval(self, sql, *args):
        self.statements.append(str(sql))
        return await self._conn.fetchval(sql, *args)

    async def execute(self, sql, *args):
        self.statements.append(str(sql))
        return await self._conn.execute(sql, *args)


# ---------------------------------------------------------------------------
# Batch read shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provenance_batch_read_returns_cited_read_sources_and_change_signal(memory, request_context):
    """One read returns cited ids, read ids, observation sources
    expanded one hop, and the stored change signal per page."""
    bank_id = f"test-pb-shape-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        source = await _insert_memory(memory, conn, bank_id, "Alice plays cello.")
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.", [source])
        other = await _insert_memory(memory, conn, bank_id, "Bob plays drums.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs), str(other)], baseline),
        )
        seq = await _record_event(conn, bank_id, obs, "TEXT_CHANGE")

        batch = await provenance_batch_read(conn=conn, fq_table=fq_store_table, bank_id=bank_id, page_ids=[mm_id])
        entry = batch[mm_id]
        assert isinstance(entry, ProvenanceBatch)
        assert entry.cited_ids == [str(obs)]
        assert entry.read_ids == [str(obs), str(other)]
        assert entry.source_ids == [str(source)]
        assert entry.change_seqs == {str(obs): seq}
        assert entry.baseline_seq == baseline
        assert entry.captured_empty is False
        assert entry.based_on_present is True
        assert entry.live_cited_ids == [str(obs)]
        trigger = entry.trigger if isinstance(entry.trigger, dict) else json.loads(entry.trigger)
        assert trigger["staleness_check"] == "graph"
        assert trigger["graph_stale"] == "cited_memories"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_provenance_batch_read_expands_one_hop_through_observations_only(memory, request_context):
    """An observation's own sources are provenance; sources of
    those sources (memory-to-memory edges) are never traversed."""
    bank_id = f"test-pb-noedge-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        source = await _insert_memory(memory, conn, bank_id, "Deep source.")
        inner = await _insert_observation(conn, bank_id, "Inner observation.", [source])
        outer = await _insert_observation(conn, bank_id, "Outer observation.", [inner])
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(outer))]},
            graph_provenance=_prov([str(outer)], baseline),
        )
        batch = await provenance_batch_read(conn=conn, fq_table=fq_store_table, bank_id=bank_id, page_ids=[mm_id])
        entry = batch[mm_id]
        assert entry.source_ids == [str(inner)]
        assert str(source) not in entry.source_ids
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_provenance_batch_read_missing_and_captured_empty_shape(memory, request_context):
    """Missing provenance is distinct from a stamped captured-empty set."""
    bank_id = f"test-pb-mshape-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_nothing = await _create_page(memory, bank_id, request_context)
        mm_empty = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_empty,
            based_on={},
            graph_provenance={"read_memory_ids": [], "captured_empty": True, "baseline_seq": baseline},
        )
        batch = await provenance_batch_read(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, page_ids=[mm_nothing, mm_empty]
        )
        nothing = batch[mm_nothing]
        assert nothing.baseline_seq is None
        assert nothing.read_ids is None
        assert nothing.cited_ids == []
        assert nothing.based_on_present is False
        empty = batch[mm_empty]
        assert empty.read_ids == []
        assert empty.captured_empty is True
        assert empty.baseline_seq == baseline
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_provenance_batch_read_is_one_batch_not_n_plus_one(memory, request_context):
    """The read issues a constant number of statements per
    batch, whatever the page count (no N+1 per page)."""
    bank_id = f"test-pb-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_ids = []
        for _ in range(3):
            mm_id = await _create_page(memory, bank_id, request_context)
            baseline = await _max_seq(conn, bank_id)
            await _stamp_provenance(
                conn,
                bank_id,
                mm_id,
                based_on={"observation": [_fact(str(obs))]},
                graph_provenance=_prov([str(obs)], baseline),
            )
            mm_ids.append(mm_id)
        for count in (1, 3):
            counting = _CountingConn(conn)
            batch = await provenance_batch_read(
                conn=counting, fq_table=fq_store_table, bank_id=bank_id, page_ids=mm_ids[:count]
            )
            assert len(batch) == count
            assert len(counting.statements) <= 5, counting.statements
            if count == 1:
                single_count = len(counting.statements)
            else:
                assert len(counting.statements) <= single_count, (
                    f"{count} pages cost {len(counting.statements)} statements vs {single_count} for one"
                )
        assert any("memory_change_events" in s for s in counting.statements)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_provenance_batch_read_empty_batch_is_empty(memory, request_context):
    """An empty batch needs no round-trip and answers {} (documented contract)."""
    bank_id = f"test-pb-empty-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        counting = _CountingConn(conn)
        batch = await provenance_batch_read(conn=counting, fq_table=fq_store_table, bank_id=bank_id, page_ids=[])
        assert batch == {}
        assert counting.statements == []
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Batch and single agree per page
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["SRC_EDIT", "TEXT_CHANGE", "FOLD_IN"])
async def test_batch_agrees_with_single_graph_provenance_events(memory, request_context, event_type):
    """In graph mode each provenance event marks batch and single stale; an
    event at or below the baseline marks neither."""
    bank_id = f"test-pb-agree-{event_type}-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        assert await _single(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, obs, event_type)
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: True}
        assert await _single(memory, conn, bank_id, mm_id) is True
        new_baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], new_baseline),
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        assert await _single(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_agrees_with_single_new_unlinked_and_bare_bump(memory, request_context):
    """A brand-new memory and a bare update-time bump mark
    neither the batch nor the single graph verdict stale."""
    bank_id = f"test-pb-unlinked-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        source = await _insert_memory(memory, conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"world": [_fact(str(source), "Alice plays cello.", "world")]},
            graph_provenance=_prov([str(source)], baseline),
        )
        await _insert_memory(memory, conn, bank_id, "A brand-new unrelated memory.")
        await conn.execute(
            f"UPDATE {fq_store_table('memory_units')} SET updated_at = now() WHERE bank_id = $1 AND id = $2",
            bank_id,
            source,
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        assert await _single(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_agrees_with_single_missing_and_captured_empty(memory, request_context):
    """Both paths fail closed on missing provenance and read fresh on
    a stamped captured-empty set."""
    bank_id = f"test-pb-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_missing = await _create_page(memory, bank_id, request_context)
        mm_empty = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_empty,
            based_on={},
            graph_provenance={"read_memory_ids": [], "captured_empty": True, "baseline_seq": baseline},
        )
        assert await _batch(memory, conn, bank_id, [mm_missing, mm_empty]) == {
            mm_missing: True,
            mm_empty: False,
        }
        assert await _single(memory, conn, bank_id, mm_missing) is True
        assert await _single(memory, conn, bank_id, mm_empty) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_agrees_with_single_read_vs_cited_traversal(memory, request_context):
    """read_memories sees a read-but-not-cited change; cited_memories does not."""
    bank_id = f"test-pb-read-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        cited = await _insert_memory(memory, conn, bank_id, "Cited fact.")
        read_only = await _insert_memory(memory, conn, bank_id, "Read but not cited.")
        mm_read = await _create_page(memory, bank_id, request_context, graph_stale="read_memories")
        mm_cited = await _create_page(memory, bank_id, request_context, graph_stale="cited_memories")
        baseline = await _max_seq(conn, bank_id)
        for mm_id in (mm_read, mm_cited):
            await _stamp_provenance(
                conn,
                bank_id,
                mm_id,
                based_on={"world": [_fact(str(cited), "Cited fact.", "world")]},
                graph_provenance=_prov([str(cited), str(read_only)], baseline),
            )
        await _record_event(conn, bank_id, read_only, "SRC_EDIT")
        assert await _batch(memory, conn, bank_id, [mm_read, mm_cited]) == {
            mm_read: True,
            mm_cited: False,
        }
        assert await _single(memory, conn, bank_id, mm_read) is True
        assert await _single(memory, conn, bank_id, mm_cited) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tagscope", "graph"])
async def test_batch_deletion_marks_stale_even_when_bank_quiet(memory, request_context, mode):
    """Deletion raises no watermark, so the batch must ask the
    liveness question even when the bank-wide shortcut would settle fresh."""
    bank_id = f"test-pb-del-{mode}-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        source = await _insert_memory(memory, conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check=mode)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"world": [_fact(str(source), "Alice plays cello.", "world")]},
            graph_provenance=_prov([str(source)], baseline),
            # A watermark in the future: the shortcut settles fresh outright.
            since_offset=timedelta(hours=1),
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, source
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: True}
        assert await _single(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_tagscope_ignores_provenance_event(memory, request_context):
    """In tagscope mode a provenance record alone
    marks neither the batch nor the single stale, away from the scope rule."""
    bank_id = f"test-pb-tagignore-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="tagscope")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
            since_offset=timedelta(hours=1),
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        assert await _single(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_watermark_shortcut_still_skips_scope_query_for_tagscope(memory, request_context, monkeypatch):
    """The bank-watermark shortcut remains, for the tagscope scope question:
    a quiet page is settled with no scoped batch query."""
    from hindsight_api.engine.memories.pg import reads

    bank_id = f"test-pb-shortcut-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="tagscope")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
            since_offset=timedelta(hours=1),
        )
        calls: list[int] = []
        original = reads.any_memory_updated_since_batch

        async def counting_batch(**kwargs):
            calls.append(1)
            return await original(**kwargs)

        monkeypatch.setattr(reads, "any_memory_updated_since_batch", counting_batch)
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        assert calls == [], "a watermark-settled page must not pay for the scoped batch query"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_graph_page_never_pays_for_scope_query(memory, request_context, monkeypatch):
    """A graph page has no scope question: an event above its baseline marks
    it stale even with a quiet bank, and the scope batch is never called."""
    from hindsight_api.engine.memories.pg import reads

    bank_id = f"test-pb-graphscope-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="graph")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
            since_offset=timedelta(hours=1),
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        calls: list[int] = []
        original = reads.any_memory_updated_since_batch

        async def counting_batch(**kwargs):
            calls.append(1)
            return await original(**kwargs)

        monkeypatch.setattr(reads, "any_memory_updated_since_batch", counting_batch)
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: True}
        assert calls == [], "a graph page must not be answered by the scope rule"
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Surfaces: is_stale reflects the graph rule without a surface change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_stale_surfaces_reflect_graph_rule(memory, request_context):
    """The list and the tree carry the graph verdict on is_stale with no
    schema change: the engine batch answers both."""
    bank_id = f"test-pb-surfaces-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
    listed = await memory.list_mental_models(bank_id, with_staleness=True, request_context=request_context)
    item = next(i for i in listed.items if i["id"] == mm_id)
    assert item["is_stale"] is True, "the list must reflect the graph rule"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_is_stale_tree_resolves_knowledge_page_id_to_provenance(memory, request_context):
    """The tree keys its answers by knowledge-page id, not mental-model id:
    the batch provenance read resolves the hop and the verdict still
    reflects the graph rule."""
    bank_id = f"test-pb-tree-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        page_id = str(uuid.uuid4())
        await conn.execute(
            f"INSERT INTO {fq_store_table('knowledge_pages')} (id, bank_id, parent_id, kind, name, mental_model_id) "
            f"VALUES ($1, $2, NULL, 'page', 'Conventions', $3)",
            page_id,
            bank_id,
            mm_id,
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
    nodes = await memory.list_page_nodes(bank_id, with_staleness=True, request_context=request_context)
    node = next(n for n in nodes if n["id"] == page_id)
    assert node["is_stale"] is True, "the tree must reflect the graph rule via the page id"
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_cited_empty_with_read_set_reads_fresh(memory, request_context):
    """Cited-empty with a non-empty read set reads fresh in
    cited_memories mode, exactly as the single path does."""
    bank_id = f"test-pb-cited-empty-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance={
                "read_memory_ids": [str(obs)],
                "captured_empty": False,
                "cited_empty": True,
                "baseline_seq": baseline,
            },
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        assert await _single(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_read_mode_deletion_of_read_only_memory_marks_stale(memory, request_context):
    """In read_memories mode, deleting a read-but-not-cited
    memory marks the page stale on the batch path too."""
    bank_id = f"test-pb-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        cited = await _insert_memory(memory, conn, bank_id, "Cited fact.")
        read_only = await _insert_memory(memory, conn, bank_id, "Read but not cited.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"world": [_fact(str(cited), "Cited fact.", "world")]},
            graph_provenance=_prov([str(cited), str(read_only)], baseline),
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: False}
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, read_only
        )
        assert await _batch(memory, conn, bank_id, [mm_id]) == {mm_id: True}
        assert await _single(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_null_watermark_batch_and_single_agree_on_graph_page(memory, request_context):
    """A graph page with fresh provenance but a NULL watermark
    (scope None) must get the same answer on batch and single: the batch asks
    provenance instead of answering stale without it."""
    bank_id = f"test-pb-null-watermark-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        # last_refreshed_at is NOT NULL in the schema, so a NULL scope is
        # exercised at the batch API level directly: the caller passes None
        # for a page it could not stamp, and the batch must still ask
        # provenance for a graph page, as the single path does.
        assert await _single(memory, conn, bank_id, mm_id) is False
        batch = await memory.compute_mental_models_are_stale(conn, bank_id, {mm_id: None})
        assert batch == {mm_id: False}
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_tagscope_only_batch_skips_change_record_query(memory, request_context):
    """A batch with no graph pages must not pay for the change-record
    query — tagscope pages need only liveness, which the unit read already
    answers. A graph page in the batch still gets its change signal."""
    bank_id = f"test-pb-no-graph-pages-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_tag = await _create_page(memory, bank_id, request_context, staleness_check="tagscope")
        mm_graph = await _create_page(memory, bank_id, request_context, staleness_check="graph")
        baseline = await _max_seq(conn, bank_id)
        for mm_id in (mm_tag, mm_graph):
            await _stamp_provenance(
                conn,
                bank_id,
                mm_id,
                based_on={"observation": [_fact(str(obs))]},
                graph_provenance=_prov([str(obs)], baseline),
            )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        counting = _CountingConn(conn)
        batch = await provenance_batch_read(conn=counting, fq_table=fq_store_table, bank_id=bank_id, page_ids=[mm_tag])
        assert batch[mm_tag].change_seqs == {}
        assert not any("memory_change_events" in s for s in counting.statements)
        counting.statements.clear()
        batch = await provenance_batch_read(
            conn=counting, fq_table=fq_store_table, bank_id=bank_id, page_ids=[mm_tag, mm_graph]
        )
        assert any("memory_change_events" in s for s in counting.statements)
        assert batch[mm_graph].change_seqs
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_mcp_tree_twin_reflects_graph_rule(memory, request_context):
    """The MCP twin of the knowledge tree carries the graph
    verdict on is_stale (the batch answers both surfaces)."""
    from hindsight_api.mcp_tools import _do_get_page_tree

    bank_id = f"test-pb-skip-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        page_id = str(uuid.uuid4())
        await conn.execute(
            f"INSERT INTO {fq_store_table('knowledge_pages')} (id, bank_id, parent_id, kind, name, mental_model_id) "
            f"VALUES ($1, $2, NULL, 'page', 'Conventions', $3)",
            page_id,
            bank_id,
            mm_id,
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
    tree = await _do_get_page_tree(memory, bank_id, request_context)
    page = tree["roots"][0]
    assert page["id"] == page_id
    assert page["is_stale"] is True
    await memory.delete_bank(bank_id, request_context=request_context)

    # Both get_page_tree docstrings mirror the HTTP graph
    # blind-spot sentence; neither promises unconditional freshness.
    import ast
    import pathlib

    from hindsight_api import mcp_tools

    tree = ast.parse(pathlib.Path(mcp_tools.__file__).read_text())
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_page_tree":
            first = node.body[0]
            assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            docstrings.append(first.value.value)
    assert len(docstrings) == 2
    for doc in docstrings:
        assert "provably up to date" not in doc
        assert "a brand-new memory outside every provenance observation never raises this flag" in " ".join(doc.split())


@pytest.mark.asyncio
async def test_vanished_page_row_batch_fails_closed(memory, request_context):
    """A page row deleted between the caller's scope read
    and the batch's provenance read fails closed (True), exactly as the
    single path answers a missing row; it must not fall through to the
    tagscope question and answer fresh. It lives here so
    `pytest -k provenance` collects it."""
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(conn, bank_id, mm_id, based_on={}, graph_provenance=_prov([], baseline))
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        assert row is not None
        scopes = {str(mm_id): _mental_model_stale_scope_from_row(row, key=str(mm_id))}
        assert scopes[str(mm_id)] is not None
        # The page row vanishes before the batch's provenance read.
        await conn.execute(
            f"DELETE FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2", bank_id, mm_id
        )
        answers = await memory.compute_mental_models_are_stale(conn, bank_id, scopes)
        assert answers[str(mm_id)] is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_vanished_bank_template_slug_batch_fails_closed(memory, request_context):
    """A bank-template knowledge-page id is a slug
    ("sentiment-overview"), not an mm-/kp-/kf- prefixed id or UUID. A
    stamped scope under that key whose row resolved nothing must still
    fail closed, not fall through to the tagscope question."""
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(conn, bank_id, mm_id, based_on={}, graph_provenance=_prov([], baseline))
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        assert row is not None
        scope = _mental_model_stale_scope_from_row(row, key="sentiment-overview")
        assert scope is not None
        answers = await memory.compute_mental_models_are_stale(conn, bank_id, {"sentiment-overview": scope})
        assert answers["sentiment-overview"] is True
    await memory.delete_bank(bank_id, request_context=request_context)
