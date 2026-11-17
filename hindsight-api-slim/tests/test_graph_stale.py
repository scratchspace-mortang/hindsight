"""Graph staleness rule beside the scope rule.

In graph mode a page is stale exactly when a durable change record newer than
its provenance baseline touches its provenance: a source memory changed
(SRC_EDIT), a read observation's text changed (TEXT_CHANGE), a new source id
was folded in (FOLD_IN), a cited memory was deleted, or provenance is missing
(fail closed). NEW_UNLINKED memories and bare update-time bumps never mark
a graph page stale. Tagscope keeps the scope rule and the legacy
retracted-grounding test: deletion marks stale, other provenance events alone
do not.

These tests cover the staleness rule, the first-refresh path from
NEVER_BUILT, and the stamp shapes that must read stale. They exercise the
engine seams ``compute_mental_model_is_stale`` and the refresh digest
(reflect_response).
"""

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.pg import consolidation as pg_con
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.reflect.retractions import based_on_fact_ids
from hindsight_api.engine.response_models import ReflectResult, ToolCallTrace
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
    mode: str = "full",
) -> str:
    model = await memory.create_mental_model(
        bank_id=bank_id,
        name="Conventions",
        source_query="What are the project's conventions?",
        content=PAGE_CONTENT,
        request_context=request_context,
        trigger={
            "mode": mode,
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
) -> None:
    """Write the reflect_response a successful refresh would have left."""
    reflect_response: dict[str, Any] = {"text": PAGE_CONTENT, "mental_models": []}
    if based_on is not None:
        reflect_response["based_on"] = based_on
    if graph_provenance is not None:
        reflect_response["graph_provenance"] = graph_provenance
    await conn.execute(
        f"UPDATE {fq_store_table('mental_models')} SET reflect_response = $3::jsonb, "
        f"last_memory_seen_at = now(), last_refreshed_at = now() WHERE bank_id = $1 AND id = $2",
        bank_id,
        mm_id,
        json.dumps(reflect_response),
    )


async def _is_stale(memory: MemoryEngine, conn, bank_id: str, mm_id: str) -> bool:
    row = await conn.fetchrow(
        f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
        f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
        bank_id,
        mm_id,
    )
    assert row is not None
    return await memory.compute_mental_model_is_stale(conn, bank_id, row)


def _prov(read_ids: list[str], baseline: int) -> dict[str, Any]:
    return {"read_memory_ids": read_ids, "captured_empty": False, "baseline_seq": baseline}


# ---------------------------------------------------------------------------
# Provenance events mark stale in graph mode; NEW_UNLINKED never does
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_src_edit_on_cited_source_marks_stale(memory, request_context):
    """SRC_EDIT on a cited memory's source marks a graph page stale."""
    bank_id = f"test-gs-src-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        source = await _insert_memory(memory, conn, bank_id, "Alice plays cello.")
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.", [source])
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, source, "SRC_EDIT")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_graph_text_change_on_cited_observation_marks_stale(memory, request_context):
    """TEXT_CHANGE on a cited read observation marks a graph page stale."""
    bank_id = f"test-gs-text-{uuid.uuid4().hex[:8]}"
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
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_graph_fold_in_on_cited_observation_marks_stale(memory, request_context):
    """FOLD_IN (new source id, text may be same) marks a graph page stale."""
    bank_id = f"test-gs-fold-{uuid.uuid4().hex[:8]}"
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
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, obs, "FOLD_IN")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_graph_new_unlinked_memory_never_marks_stale(memory, request_context):
    """A brand-new memory folded into no provenance observation is a blind
    spot by design: it must not mark the page stale."""
    bank_id = f"test-gs-unlinked-{uuid.uuid4().hex[:8]}"
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
        await _insert_memory(memory, conn, bank_id, "A brand-new unrelated memory.")
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# A bare update-time bump never marks stale in graph mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_bare_updated_at_bump_never_marks_stale(memory, request_context):
    """Bumping updated_at with no provenance signal keeps a graph page fresh."""
    bank_id = f"test-gs-bump-{uuid.uuid4().hex[:8]}"
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
        await conn.execute(
            f"UPDATE {fq_store_table('memory_units')} SET updated_at = now() WHERE bank_id = $1 AND id = $2",
            bank_id,
            source,
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Entering FRESH only via a refresh commit (baseline stamp)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_refresh_baseline_clears_stale(memory, request_context):
    """After a provenance event marks the page stale, stamping a baseline
    at/after the event (what a refresh commit does) makes it fresh again —
    an incorporated signal never keeps the page stale."""
    bank_id = f"test-gs-baseline-{uuid.uuid4().hex[:8]}"
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
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        new_baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], new_baseline),
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tagscope ignores provenance-only events; DELETE marks stale in both modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tagscope_ignores_provenance_events(memory, request_context):
    """In tagscope mode a TEXT_CHANGE record alone does not mark stale."""
    bank_id = f"test-gs-tag-{uuid.uuid4().hex[:8]}"
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
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tagscope", "graph"])
async def test_delete_of_cited_memory_marks_stale_in_both_modes(memory, request_context, mode):
    """Deletion of a cited memory marks stale in both modes."""
    bank_id = f"test-gs-delete-{mode}-{uuid.uuid4().hex[:8]}"
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
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, source
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Captured-empty versus missing provenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_missing_provenance_fails_closed(memory, request_context):
    """A graph page with no provenance stamped reads stale (fail closed)."""
    bank_id = f"test-gs-missing-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(memory, bank_id, request_context)
        # Watermark stamped (so the scope question has an answer) but no
        # graph_provenance and no based_on: provenance is missing.
        await conn.execute(
            f"UPDATE {fq_store_table('mental_models')} SET last_memory_seen_at = now() WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_graph_captured_empty_reads_fresh(memory, request_context):
    """Captured-empty is not missing: a refresh that read nothing stamps an
    explicit empty set, and the page reads fresh against it (never stale-forever)."""
    bank_id = f"test-gs-empty-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(memory, bank_id, request_context)
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance={"read_memory_ids": [], "captured_empty": True, "baseline_seq": baseline},
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_graph_empty_without_capture_marker_fails_closed(memory, request_context):
    """An empty based_on with no captured-empty marker is missing provenance,
    not a stamped empty set: it must fail closed."""
    bank_id = f"test-gs-nomark-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(memory, bank_id, request_context)
        await _stamp_provenance(conn, bank_id, mm_id, based_on={}, graph_provenance=None)
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# cited vs read traversal; observation-source expansion; no memory-to-memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_memories_traverses_read_set_beyond_cited(memory, request_context):
    """graph_stale=read_memories: an SRC_EDIT on a read-but-not-cited memory
    marks stale; cited_memories must not see it."""
    bank_id = f"test-gs-read-{uuid.uuid4().hex[:8]}"
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
        assert await _is_stale(memory, conn, bank_id, mm_read) is True
        assert await _is_stale(memory, conn, bank_id, mm_cited) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_graph_never_traverses_memory_to_memory_edges(memory, request_context):
    """Expansion goes one hop through observations: O1 -> O2 -> S. An event on
    S (two hops from O1) must not mark a page citing O1 stale, while an
    event on O2 (one hop) must."""
    bank_id = f"test-gs-notraverse-{uuid.uuid4().hex[:8]}"
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
        await _record_event(conn, bank_id, source, "SRC_EDIT")
        assert await _is_stale(memory, conn, bank_id, mm_id) is False, "two-hop source is not provenance"
        await _record_event(conn, bank_id, inner, "TEXT_CHANGE")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True, "one-hop observation is provenance"
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Two-page baseline 
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_pages_opposite_sides_of_one_fold(memory, request_context):
    """Two pages with different baselines on opposite sides of one fold:
    the page built before the fold reads stale, the page built after reads fresh."""
    bank_id = f"test-gs-twopage-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_before = await _create_page(memory, bank_id, request_context)
        mm_after = await _create_page(memory, bank_id, request_context)
        baseline_before = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_before,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline_before),
        )
        await _record_event(conn, bank_id, obs, "FOLD_IN")
        baseline_after = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_after,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline_after),
        )
        assert await _is_stale(memory, conn, bank_id, mm_before) is True
        assert await _is_stale(memory, conn, bank_id, mm_after) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Refresh digest: read set persisted from the reflect tool trace
# ---------------------------------------------------------------------------


def _install_reflect(memory, monkeypatch, *, text: str, based_on=None, tool_trace=None, side_effect=None):
    async def fake_reflect_async(**kwargs):
        if side_effect is not None:
            await side_effect()
        return ReflectResult.model_validate(
            {
                "text": text,
                "based_on": based_on
                or {"observation": [], "world": [], "experience": [], "mental-models": [], "directives": []},
                "tool_trace": tool_trace or [],
            }
        )

    monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)


@pytest.mark.asyncio
async def test_refresh_persists_read_set_from_tool_trace(memory, request_context, monkeypatch):
    """The digest persists ids from the reflect tool trace (recursively
    deduped), so read_memories can see a memory the agent read but did not cite."""
    bank_id = f"test-gs-trace-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        other = await _insert_observation(conn, bank_id, "Bob plays drums.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories")
    trace = [
        ToolCallTrace(
            tool="recall",
            reason="need facts",
            input={},
            output={"memories": [{"id": str(obs), "text": "Alice plays cello.", "fact_type": "observation"}]},
            duration_ms=1,
            iteration=1,
        ),
        ToolCallTrace(
            tool="search_observations",
            reason="need observations",
            input={},
            output={"observations": [{"id": str(other)}]},
            duration_ms=1,
            iteration=2,
        ),
        # A repeat: the persisted set is a set, not a log.
        ToolCallTrace(
            tool="recall",
            reason="again",
            input={},
            output={"memories": [{"id": str(obs), "text": "Alice plays cello.", "fact_type": "observation"}]},
            duration_ms=1,
            iteration=3,
        ),
    ]
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice plays cello.\n",
        based_on={
            "observation": [_fact(str(obs))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
        tool_trace=trace,
    )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert sorted(prov["read_memory_ids"]) == sorted([str(obs), str(other)])
    assert prov["captured_empty"] is False
    assert isinstance(prov["baseline_seq"], int)
    # An SRC_EDIT on the read-but-not-cited observation now marks it stale.
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, other, "SRC_EDIT")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_first_refresh_from_never_built_establishes_a_fresh_baseline(memory, request_context, monkeypatch):
    """FIRST: a first refresh from NEVER_BUILT (no watermark, no provenance)
    establishes the baseline on commit; before it the page reads stale
    (a never-built bank plus a refresh commit), after it reads fresh."""
    bank_id = f"test-gs-first-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        assert row["last_memory_seen_at"] is None
        assert await memory.compute_mental_model_is_stale(conn, bank_id, row) is True
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice plays cello.\n",
        based_on={
            "observation": [_fact(str(obs))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
    )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_first_refresh_captured_empty_reads_fresh(memory, request_context, monkeypatch):
    """FIRST + captured-empty: a first refresh that read nothing stamps an
    explicit empty set and reads fresh, instead of staying stale-forever."""
    bank_id = f"test-gs-first-empty-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        # One observation in scope so the refresh runs (has_sources).
        await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
    _install_reflect(memory, monkeypatch, text="## Conventions\n\nNothing yet.\n", based_on={"observation": []})
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert prov["read_memory_ids"] == []
    assert prov["captured_empty"] is True
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Delta gate: a gate-rejected memory appears in the persisted read set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_rejected_memory_appears_in_persisted_read_set(memory, request_context, monkeypatch):
    """The delta gate lists memories and the decision
    model rejects one; the rejected id is still in the persisted read set,
    so a later change to it can mark the page stale."""
    bank_id = f"test-gs-gate-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
    pool = await memory._get_pool()
    kept_id = uuid.uuid4()
    rejected_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        for unit_id, text in (
            (kept_id, "The user prefers bullet points."),
            (rejected_id, "The build server runs Linux."),
        ):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
                "VALUES ($1, $2, $3, 'observation', ARRAY[]::varchar[], NOW() - INTERVAL '2 minutes', "
                "NOW() - INTERVAL '2 minutes')",
                unit_id,
                bank_id,
                text,
            )
    memory._reflect_decision_model_client = SimpleNamespace(  # type: ignore[attr-defined]
        could_change_page=AsyncMock(side_effect=lambda topic, page, text: "prefers" in text)
    )

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Bullet points."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    monkeypatch.setattr(memory, "reflect_async", AsyncMock(side_effect=AssertionError("gate replaces reflect")))

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(kept_id) in prov["read_memory_ids"]
    assert str(rejected_id) in prov["read_memory_ids"], "gate-rejected id must persist in the read set"
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await _record_event(conn, bank_id, rejected_id, "SRC_EDIT")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_read_mode_deleted_read_id_pruned_at_next_delta_refresh(memory, request_context, monkeypatch):
    """A read id that became dead must not poison the verdict forever.

    Read mode stamps the previous read set unioned with the refresh's own
    reads (the delta page is grounded on everything it has ever read). A
    read-but-not-cited memory that is then deleted makes the page stale
    (read-mode liveness) — correct once. But if the delta refresh re-stamps
    the dead id, the page is stale forever: the baseline is above every
    record, yet the liveness check keeps firing, and the persisted read set
    grows monotonically with every refresh. The unioned set is intersected
    with live ids at stamp time, so the dead id is dropped and the page
    reads fresh after the refresh that consumed the deletion.
    """
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
    pool = await memory._get_pool()
    first_id = uuid.uuid4()
    dead_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        for unit_id, text in (
            (first_id, "The user prefers bullet points."),
            (dead_id, "The user prefers tabs."),
        ):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
                "VALUES ($1, $2, $3, 'observation', ARRAY[]::varchar[], NOW() - INTERVAL '2 minutes', "
                "NOW() - INTERVAL '2 minutes')",
                unit_id,
                bank_id,
                text,
            )
    memory._reflect_decision_model_client = SimpleNamespace(  # type: ignore[attr-defined]
        could_change_page=AsyncMock(side_effect=lambda topic, page, text: "prefers" in text)
    )

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Bullet points."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    monkeypatch.setattr(memory, "reflect_async", AsyncMock(side_effect=AssertionError("gate replaces reflect")))

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(first_id) in prov["read_memory_ids"]
    assert str(dead_id) in prov["read_memory_ids"]
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        # The read-but-not-cited memory dies; no change record is raised for
        # it, so only the read-mode liveness check can see it.
        await conn.execute("DELETE FROM memory_units WHERE bank_id = $1 AND id = $2", bank_id, dead_id)
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        # A new memory in the delta window so the next refresh has sources
        # and actually re-stamps (a no-sources refresh restores the old
        # stamp by design).
        new_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
            "VALUES ($1, $2, $3, 'observation', ARRAY[]::varchar[], NOW() - INTERVAL '1 minute', "
            "NOW() - INTERVAL '1 minute')",
            new_id,
            bank_id,
            "The user prefers YAML.",
        )
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(dead_id) not in prov["read_memory_ids"], "the dead id must be pruned at the next refresh start"
    assert str(first_id) in prov["read_memory_ids"]
    assert str(new_id) in prov["read_memory_ids"]
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_mid_flight_read_deletion_reads_stale_then_pruned_next_refresh(memory, request_context, monkeypatch):
    """A memory read during a delta refresh and deleted before the
    stamp must not be pruned from the persisted read set (model
    READ-UND-FLIGHT/DEL-FLIGHT want STALE). The previous set is pruned at
    This snapshot and this refresh's own reads are stamped unpruned, so the
    mid-flight deletion keeps the page stale after commit; the next
    refresh prunes it at its refresh start and reads fresh.
    """
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        cited = await _insert_observation(conn, bank_id, "Alice plays cello.")
        victim = await _insert_observation(conn, bank_id, "Bob plays drums.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
    # Force the reflect path: the delta gate declines (no decision model).
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))

    async def die_during_reflect():
        async with pool.acquire() as conn:
            async with conn.transaction():
                await get_memories().invalidate_memory(
                    conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=str(victim), reason="superseded"
                )

    trace = [
        ToolCallTrace(
            tool="recall",
            reason="need facts",
            input={},
            output={
                "memories": [
                    {"id": str(cited), "text": "Alice plays cello.", "fact_type": "observation"},
                    {"id": str(victim), "text": "Bob plays drums.", "fact_type": "observation"},
                ]
            },
            duration_ms=1,
            iteration=1,
        )
    ]
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice plays cello. Bob plays drums.\n",
        based_on={
            "observation": [_fact(str(cited))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
        tool_trace=trace,
        side_effect=die_during_reflect,
    )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(victim) in prov["read_memory_ids"], "a mid-flight deletion must stay in the read set"
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        # A live memory in the next delta window so the second refresh re-stamps.
        new_id = await _insert_observation(conn, bank_id, "Carol plays piano.")
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
    trace2 = [
        ToolCallTrace(
            tool="recall",
            reason="need facts",
            input={},
            output={"memories": [{"id": str(new_id), "text": "Carol plays piano.", "fact_type": "observation"}]},
            duration_ms=1,
            iteration=1,
        )
    ]
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice plays cello. Carol plays piano.\n",
        based_on={
            "observation": [_fact(str(cited))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
        tool_trace=trace2,
    )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(victim) not in prov["read_memory_ids"], "the next refresh start must prune the dead id"
    assert str(new_id) in prov["read_memory_ids"]
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_prune_to_empty_reads_fresh_with_captured_empty(memory, request_context, monkeypatch):
    """Empty-refresh policy: a delta refresh whose whole
    previous read set died before refresh start, and that itself reads nothing,
    stamps captured_empty=True over the empty live set and reads fresh.
    The refresh consumed the deletion state at refresh start, so a stale
    verdict becomes fresh at refresh commit; the dead id does not poison the
    verdict forever. A no-sources refresh would instead restore the old
    stamp; this refresh has a source in scope but reads nothing.
    """
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        only = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance=_prov([str(only)], baseline),
        )
        await conn.execute("DELETE FROM memory_units WHERE bank_id = $1 AND id = $2", bank_id, only)
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        # A source in the delta window so the refresh runs (has_sources).
        await _insert_observation(conn, bank_id, "Bob plays drums.")
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))
    _install_reflect(memory, monkeypatch, text="## Conventions\n\nNothing new.\n", based_on={"observation": []})

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Nothing new."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert prov["read_memory_ids"] == []
    assert prov["captured_empty"] is True
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_cited_prune_to_empty_restamps_fresh(memory, request_context):
    """A cited-mode refresh that prunes its only citation must restamp
    ``cited_empty`` from the post-prune based_on, so the single and batch
    readers both answer fresh instead of staying stale on every later tick.
    """
    bank_id = f"test-gs-cited-prune-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        only = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="cited_memories", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(only))]},
            graph_provenance=_prov([str(only)], baseline),
        )
        await conn.execute(f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, only)
        assert await _is_stale(memory, conn, bank_id, mm_id) is True

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    rr = refreshed["reflect_response"]
    assert based_on_fact_ids(rr["based_on"]) == []
    assert rr["graph_provenance"]["cited_empty"] is True

    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        answers = await memory.compute_mental_models_are_stale(conn, bank_id, {mm_id: row})
        assert answers[mm_id] is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_store_delete_observations_purges_change_records(memory, request_context):
    """``writes.delete_observations`` (via the store) purges the change
    records of the observations it deletes, so a record cannot outlive the
    observation row it describes (store-owned writer)."""
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    store = get_memories()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 1
        async with conn.transaction():
            await store.delete_observations(conn=conn, fq_table=fq_store_table, bank_id=bank_id)
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 0
        assert (
            await conn.fetchval("SELECT count(*) FROM memory_units WHERE bank_id = $1 AND id = $2", bank_id, obs) == 0
        )
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_tagscope_delta_refresh_needs_no_live_memory_ids(memory, request_context, monkeypatch):
    """Refresh-start pruning is graph-mode only. A tagscope delta page with
    a persisted read set refreshes through a store that does not implement
    ``live_memory_ids``: tagscope never consumes the read set, so no
    liveness question is asked and the refresh succeeds.
    """
    import hindsight_api.engine.memories as memories_pkg

    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="tagscope", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance=_prov([str(obs)], baseline),
        )
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )

    class _NoLiveStore:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def live_memory_ids(self, **kwargs):
            raise NotImplementedError("live_memory_ids not implemented")

    monkeypatch.setattr(memories_pkg, "get_memories", lambda: _NoLiveStore(get_memories()))
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))
    trace = [
        ToolCallTrace(
            tool="recall",
            reason="need facts",
            input={},
            output={"memories": [{"id": str(obs), "text": "Alice plays cello.", "fact_type": "observation"}]},
            duration_ms=1,
            iteration=1,
        )
    ]
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice plays cello.\n",
        based_on={"observation": []},
        tool_trace=trace,
    )

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Alice plays cello."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(obs) in prov["read_memory_ids"]
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Race: a mid-flight event must read stale after commit; never stamp the latest seq
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_race_mid_flight_event_reads_stale_after_commit(memory, request_context, monkeypatch):
    """The refresh reads observation A, event B
    commits mid-flight, the refresh stamps no version newer than A — so the
    page reads stale, never silently fresh."""
    bank_id = f"test-gs-race-{uuid.uuid4().hex[:8]}"
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

    async def commit_event_mid_flight():
        async with pool.acquire() as conn:
            await _record_event(conn, bank_id, obs, "TEXT_CHANGE")

    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice sings.\n",
        based_on={
            "observation": [_fact(str(obs))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
        side_effect=commit_event_mid_flight,
    )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    async with pool.acquire() as conn:
        event_seq = await _max_seq(conn, bank_id)
        stamped = refreshed["reflect_response"]["graph_provenance"]["baseline_seq"]
        assert stamped < event_seq, "baseline must never be newer than the consumed version"
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Crash: no record and no baseline before commit; a committed record remains
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crash_before_event_commits_leaves_no_record(memory, request_context):
    """A crash before the event commits leaves no record, so a fresh
    page stays fresh (the rolled-back transaction records nothing)."""
    bank_id = f"test-gs-noevent-{uuid.uuid4().hex[:8]}"
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
        with pytest.raises(RuntimeError):
            async with conn.transaction():
                await pg_con.record_change_event(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    subject_id=obs,
                    event_type="TEXT_CHANGE",
                    before_text="a",
                    after_text="b",
                )
                raise RuntimeError("crash before commit")
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_failed_refresh_stamps_no_baseline_and_event_remains(memory, request_context, monkeypatch):
    """A refresh that fails before commit stamps no new baseline,
    while the already-committed event record remains — the page reads stale
    and a retry still sees the event."""
    bank_id = f"test-gs-crash-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_source_query = source_query, "
            "last_memory_seen_at = NOW() - INTERVAL '1 day' WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True

    # A delta refresh whose op call fails: the refresh refuses to write.
    async def boom(*, messages, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", boom)
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nAlice sings.\n",
        based_on={
            "observation": [_fact(str(obs))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
    )
    from hindsight_api.engine.memory_engine import MentalModelRefreshError

    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=request_context)
    stored = await memory.get_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, detail="full", request_context=request_context
    )
    assert stored["reflect_response"]["graph_provenance"]["baseline_seq"] == baseline, (
        "crash/failure stamps no baseline"
    )
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is True, "committed record remains above the baseline"
    await memory.delete_bank(bank_id, request_context=request_context)


def test_freshness_rules_for_unbuilt_bank_and_fresh_commit():
    """No code path stamps FRESH
    without a flight — the only baseline stamp lives in the refresh commit.
    This pure test pins the persisted keys the stamp must carry, so a stamp
    without a consumed baseline cannot be constructed by the digest."""
    prov = _prov(["x"], 7)
    assert prov["baseline_seq"] == 7
    assert prov["captured_empty"] is False
    # based_on alone (legacy shape) carries no baseline: the engine must
    # treat it as missing provenance in graph mode, which
    # test_graph_missing_provenance_fails_closed exercises end to end.
    assert based_on_fact_ids({"observation": [_fact("x")]}) == ["x"]


@pytest.mark.asyncio
async def test_invalid_stamp_shapes_read_stale_through_product_code(memory, request_context):
    """Invalid stamp shapes are exercised through product code, not
    constants: a baseline-less stamp (a commit without a flight) and a
    never-built page both read stale via compute_mental_model_is_stale,
    and _batch_graph_is_stale agrees on the baseline-less shape."""
    from hindsight_api.engine.memories.base import ProvenanceBatch

    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
        # A never-built page with no watermark or provenance reads stale.
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        assert await memory.compute_mental_model_is_stale(conn, bank_id, row) is True
        # A based_on-only stamp (a commit without a flight: no consumed
        # baseline) must not read fresh through either product path.
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=None,
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        prov = ProvenanceBatch(
            page_id=mm_id,
            trigger={"staleness_check": "graph", "graph_stale": "cited_memories"},
            cited_ids=[str(obs)],
            based_on_present=True,
            read_ids=[str(obs)],
            captured_empty=False,
            baseline_seq=None,
            cited_source_ids=[],
            read_source_ids=[],
            live_cited_ids=[str(obs)],
            live_read_ids=[str(obs)],
            change_seqs={},
        )
        assert memory._batch_graph_is_stale(prov, graph_stale="cited_memories") is True
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_late_committing_earlier_seq_is_not_covered_by_baseline(memory, request_context):
    """Identity seq does not follow commit order. A change record that
    allocates seq N but commits after a later seq commits must not be covered
    by a baseline captured while it was still uncommitted: the capture must
    wait for the in-flight record, so the stamped baseline is in commit order."""
    import asyncio

    bank_id = f"test-gs-late-seq-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
    recorded = asyncio.Event()
    release = asyncio.Event()
    late_seq: list[int] = []

    async def late_commit():
        async with pool.acquire() as conn:
            async with conn.transaction():
                await pg_con.record_change_event(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    subject_id=obs,
                    event_type="TEXT_CHANGE",
                    before_text="a",
                    after_text="b",
                )
                late_seq.append(
                    await conn.fetchval(
                        f"SELECT max(seq) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
                    )
                )
                recorded.set()
                await release.wait()

    task = asyncio.create_task(late_commit())
    await recorded.wait()
    # While the late record's transaction is still open, a baseline capture
    # must not return a value that drops it: it waits for the commit.
    capture = asyncio.create_task(memory._mental_model_event_version(bank_id, mm_id))
    # Prove the capture task really started and is blocked on the advisory
    # lock before timing it: without this, the timeout below also fires when
    # the task merely never got scheduled.
    async with pool.acquire() as probe:
        async with asyncio.timeout(10):
            while (
                await probe.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                    "AND query LIKE '%advisory_xact_lock%'"
                )
                == 0
            ):
                await asyncio.sleep(0.01)
    assert not capture.done(), "capture must be started and blocked on the lock"
    try:
        await asyncio.wait_for(asyncio.shield(capture), timeout=0.5)
        captured_early = True
    except asyncio.TimeoutError:
        captured_early = False
    release.set()
    captured, _prev_read_ids = await capture
    await task
    assert not captured_early, "baseline capture must wait for the in-flight record to commit"
    assert captured >= late_seq[0], "captured baseline must include the late-committing record"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delete_bank_purges_change_events(memory, request_context):
    """Deleting a bank must not leave its memory_change_events behind."""
    bank_id = f"test-gs-bank-delete-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 1
    await memory.delete_bank(bank_id, request_context=request_context)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 0


@pytest.mark.asyncio
async def test_cited_empty_with_read_set_reads_fresh(memory, request_context):
    """A page that read memories but cited none is captured-empty for the
    cited set, not stale-forever: read set non-empty must not fail the
    cited-mode empty check."""
    bank_id = f"test-gs-cited-empty-{uuid.uuid4().hex[:8]}"
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
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_read_mode_deletion_of_read_only_memory_marks_stale(memory, request_context):
    """In read_memories mode the read set is the provenance, so
    deleting a read-but-not-cited memory marks the page stale (it is a
    cited-equivalent deletion, not an uncited UND)."""
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
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
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, read_only
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_cited_mode_deletion_of_read_only_memory_stays_fresh_single(memory, request_context):
    """In cited_memories mode a memory the page read but
    did not cite is outside the provenance, so deleting it leaves the page
    fresh — the cited-deletion boundary from the model (UND stays clean in
    cited mode)."""
    bank_id = f"test-gs-cited-single-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        cited = await _insert_memory(memory, conn, bank_id, "Cited fact.")
        read_only = await _insert_memory(memory, conn, bank_id, "Read but not cited.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="cited_memories")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"world": [_fact(str(cited), "Cited fact.", "world")]},
            graph_provenance=_prov([str(cited), str(read_only)], baseline),
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, read_only
        )
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_cited_mode_deletion_of_read_only_memory_stays_fresh_batch(memory, request_context):
    """The same cited-mode boundary through the batch
    staleness answer — deleting a read-but-not-cited memory leaves the page
    fresh."""
    bank_id = f"test-gs-cited-batch-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        cited = await _insert_memory(memory, conn, bank_id, "Cited fact.")
        read_only = await _insert_memory(memory, conn, bank_id, "Read but not cited.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="cited_memories")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"world": [_fact(str(cited), "Cited fact.", "world")]},
            graph_provenance=_prov([str(cited), str(read_only)], baseline),
        )
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, read_only
        )
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        assert row is not None
        answers = await memory.compute_mental_models_are_stale(conn, bank_id, {mm_id: row})
        assert answers[mm_id] is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_skipped_refresh_stamps_no_new_baseline(memory, request_context, monkeypatch):
    """A refresh skipped for no sources stamps no baseline, so a
    provenance event above the old baseline keeps the page stale."""
    bank_id = f"test-gs-skip-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Conventions",
            source_query="What are the conventions?",
            content=PAGE_CONTENT,
            tags=["zzz-nomatch-xyz"],
            request_context=request_context,
            trigger={
                "mode": "full",
                "refresh_after_consolidation": True,
                "staleness_check": "graph",
                "graph_stale": "cited_memories",
            },
        )
        mm_id = model["id"]
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(obs))]},
            graph_provenance=_prov([str(obs)], baseline),
        )
        await _record_event(conn, bank_id, obs, "TEXT_CHANGE")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    _install_reflect(memory, monkeypatch, text="## Conventions\n\nNothing.\n", based_on={"observation": []})
    await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=request_context)
    stored = await memory.get_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, detail="full", request_context=request_context
    )
    prov = (stored.get("reflect_response") or {}).get("graph_provenance") or {}
    assert prov.get("baseline_seq") == baseline, "skipped refresh must not advance the baseline"
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("non_memory_key", ["mental-models", "directives"])
async def test_non_memory_based_on_stamps_cited_empty(memory, request_context, monkeypatch, non_memory_key):
    """A refresh whose based_on carries only non-memory ids (mental
    models or directives) stamps cited_empty, because the cited-mode reader
    reads those ids as an empty set. Counted by the old any-id predicate the
    stamp said cited_empty False and the page read stale forever."""
    bank_id = f"test-gs-{non_memory_key}-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        # One observation in scope so the refresh runs (has_sources).
        await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context)
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nNothing yet.\n",
        based_on={non_memory_key: [{"id": "sibling-1", "text": "a sibling", "type": non_memory_key}]},
    )
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert prov["cited_empty"] is True
    # Through the digest: the batch reader's own cited_empty agrees, and
    # both product paths read the page fresh.
    async with pool.acquire() as conn:
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
        batch = await get_memories().provenance_batch_read(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, page_ids=[mm_id]
        )
    assert batch[mm_id].cited_empty is True
    assert batch[mm_id].cited_ids == []
    assert memory._batch_graph_is_stale(batch[mm_id], graph_stale="cited_memories") is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delete_document_purges_change_events(memory, request_context):
    """Change records must not outlive their subject on delete_document.
    A document whose fact was edited (an SRC_EDIT record on the fact) then
    deleted leaves zero records, not an orphan above every baseline."""
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO documents(id, bank_id) VALUES($1, $2)", "doc-target", bank_id)
        await conn.execute(
            "INSERT INTO memory_units(id, bank_id, document_id, text, fact_type) "
            "VALUES($1, $2, $3, 'Alice plays cello.', 'world')",
            uuid.uuid4(),
            bank_id,
            "doc-target",
        )
        fact_id = await conn.fetchval(
            "SELECT id FROM memory_units WHERE bank_id = $1 AND document_id = $2", bank_id, "doc-target"
        )
        await _record_event(conn, bank_id, fact_id, "SRC_EDIT")
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 1
    await memory.delete_document("doc-target", bank_id, request_context=request_context)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delete_document_purges_archived_change_events(memory, request_context):
    """A document-derived memory invalidated before the delete leaves
    its change record in the curation archive subject set, not in
    memory_units, so both the engine per-document purge (unit_ids from
    document_source_units) and writes.delete_document must name archived
    subjects too — otherwise the record outlives the archived row the
    document cascade removes."""
    bank_id = f"test-gs-doc-delete-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    store = get_memories()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO documents(id, bank_id) VALUES($1, $2)", "doc-derived", bank_id)
        fact_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units(id, bank_id, document_id, text, fact_type) "
            "VALUES($1, $2, $3, 'Alice plays cello.', 'world')",
            fact_id,
            bank_id,
            "doc-derived",
        )
        async with conn.transaction():
            assert await store.invalidate_memory(
                conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=str(fact_id), reason="superseded"
            )
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 1
    await memory.delete_document("doc-derived", bank_id, request_context=request_context)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_store_delete_document_purges_archived_change_events(memory, request_context):
    """The store-level writer is exercised on its own.

    ``writes.delete_document`` (via ``get_memories().delete_document``) must
    purge the change records of a document's invalidated (archived) memories
    without the engine purge ever running: it is the purge on the retain
    replace path (pg/retain.delete_document_for_replace) and the default
    store's delete_document_rows, so a regression here would orphan records
    above every page baseline on those paths while ``MemoryEngine.delete_document``
    stayed green on its own purge.
    """
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    store = get_memories()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO documents(id, bank_id) VALUES($1, $2)", "doc-store", bank_id)
        fact_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units(id, bank_id, document_id, text, fact_type) "
            "VALUES($1, $2, $3, 'Alice plays cello.', 'world')",
            fact_id,
            bank_id,
            "doc-store",
        )
        async with conn.transaction():
            assert await store.invalidate_memory(
                conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=str(fact_id), reason="superseded"
            )
        # The invalidation itself records the change event for the archived
        # subject — the archived arm writes.delete_document must purge.
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 1
        # Store-level only: no engine delete_document and no
        # purge_change_events_for_subjects call anywhere in this test.
        async with conn.transaction():
            await store.delete_document(conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id="doc-store")
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id
        )
        assert count == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oracle_all_unscoped_batch_issues_no_provenance_read(monkeypatch):
    """On a non-Postgres backend, a batch with every scope NULL answers
    every page stale without any provenance read — the change-record table
    does not exist there, so nothing may query it."""
    import hindsight_api.engine.memories as memories_pkg
    from hindsight_api.engine.memory_engine import MemoryEngine

    class RecordingConn:
        backend_type = "oracle"

        def __init__(self):
            self.statements: list[str] = []

        async def fetchval(self, sql, *args, **kwargs):
            self.statements.append(sql)
            return None

        async def fetchrow(self, sql, *args, **kwargs):
            self.statements.append(sql)
            return None

        async def fetch(self, sql, *args, **kwargs):
            self.statements.append(sql)
            return []

        async def execute(self, sql, *args, **kwargs):
            self.statements.append(sql)

    store_calls: list[str] = []

    class RecordingStore:
        async def latest_memory_write_at(self, *, conn, fq_table, bank_id):
            # The bank watermark is safe on Oracle (it reads memory_units
            # only); the branch may ask it even with no stamped scopes.
            return None

        async def provenance_batch_read(self, *, conn, fq_table, bank_id, page_ids):
            store_calls.append("provenance_batch_read")
            raise AssertionError("provenance read on a non-Postgres batch")

        async def any_memory_updated_since_batch(self, *, conn, fq_table, bank_id, scopes):
            store_calls.append("any_memory_updated_since_batch")
            raise AssertionError("scope batch on an all-unscoped batch")

    monkeypatch.setattr(memories_pkg, "get_memories", lambda: RecordingStore())
    conn = RecordingConn()
    engine = MemoryEngine.__new__(MemoryEngine)
    answers = await engine.compute_mental_models_are_stale(conn, "bank", {"page-1": None, "graph-1": None})
    assert answers == {"page-1": True, "graph-1": True}
    assert store_calls == []
    assert not any("memory_change_events" in statement for statement in conn.statements)


@pytest.mark.asyncio
async def test_record_without_open_transaction_is_asserted():
    """A change-record writer called with no open transaction fails
    loud: the xact advisory lock serializing record commits would otherwise
    autocommit per statement and the seq ordering guarantee would vanish.
    Driven through the real :class:`PostgresConnection` wrapper around an
    asyncpg-shaped stub: the stub's
    ``is_in_transaction`` is what the wrapper reads."""
    from hindsight_api.engine.db.postgresql import PostgresConnection

    class StubAsyncpgConn:
        backend_type = "postgresql"

        def __init__(self, in_txn: bool) -> None:
            self._in_txn = in_txn
            self.executed: list[str] = []

        def is_in_transaction(self) -> bool:
            return self._in_txn

        async def execute(self, query: str, *args, **kwargs) -> str:
            self.executed.append(query)
            return "OK"

    outside = PostgresConnection(StubAsyncpgConn(False))
    with pytest.raises(RuntimeError, match="requires an open transaction"):
        await pg_con.record_change_event(
            conn=outside,
            fq_table=fq_table,
            bank_id="bank",
            subject_id=uuid.uuid4(),
            event_type="TEXT_CHANGE",
        )
    assert outside._conn.executed == [], "no statement may run without a transaction"

    inside = PostgresConnection(StubAsyncpgConn(True))
    await pg_con.record_change_event(
        conn=inside,
        fq_table=fq_table,
        bank_id="bank",
        subject_id=uuid.uuid4(),
        event_type="TEXT_CHANGE",
        before_text="before",
        after_text="after",
    )
    assert any("pg_advisory_xact_lock" in q for q in inside._conn.executed)
    assert any("memory_change_events" in q for q in inside._conn.executed)


@pytest.mark.asyncio
async def test_two_folds_with_different_detail_record_separately(memory, request_context):
    """The FOLD_IN event identity includes the
    audit detail. Two folds of one observation in one transaction with the
    same version token and before/after values but different
    ``added_source_ids`` record two rows (each fold's detail survives);
    a retry replaying the same detail still deduplicates to one."""
    bank_id = f"test-gs-second-fold-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Alice plays cello.")
        async with conn.transaction():
            base = dict(
                bank_id=bank_id,
                subject_id=obs,
                event_type="FOLD_IN",
                version_token="token-1",
                before_text="Alice plays cello.",
                after_text="Alice plays cello.",
            )
            rec = dict(bank_id=bank_id, subject_id=obs, event_type="FOLD_IN")
            await pg_con.record_change_event(
                conn=conn,
                fq_table=fq_table,
                **rec,
                event_id=pg_con.stable_event_id(**base, detail={"added_source_ids": ["src-1"]}),
                detail={"added_source_ids": ["src-1"]},
            )
            await pg_con.record_change_event(
                conn=conn,
                fq_table=fq_table,
                **rec,
                event_id=pg_con.stable_event_id(**base, detail={"added_source_ids": ["src-2"]}),
                detail={"added_source_ids": ["src-2"]},
            )
            # Retry of the first fold: same detail, same identity, still dedupes.
            await pg_con.record_change_event(
                conn=conn,
                fq_table=fq_table,
                **rec,
                event_id=pg_con.stable_event_id(**base, detail={"added_source_ids": ["src-1"]}),
                detail={"added_source_ids": ["src-1"]},
            )
        rows = await conn.fetch(
            f"SELECT seq, detail FROM {fq_table('memory_change_events')} "
            f"WHERE bank_id = $1 AND subject_id = $2 ORDER BY seq",
            bank_id,
            obs,
        )
        assert len(rows) == 2
        details = [json.loads(r["detail"]) if isinstance(r["detail"], str) else r["detail"] for r in rows]
        assert details == [{"added_source_ids": ["src-1"]}, {"added_source_ids": ["src-2"]}]
        assert rows[0]["seq"] < rows[1]["seq"]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_chunk_delete_purges_outgoing_change_events(memory, request_context):
    """A delta re-ingest cascade that deletes
    the outgoing facts must purge their change records too, or the records
    outlive the rows they describe."""
    from hindsight_api.engine.retain import chunk_storage

    bank_id = f"test-gs-chunk-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    doc_id = str(uuid.uuid4())
    chunk_id = f"{doc_id}_0"
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at) "
            "VALUES ($1, $2, 'doomed text', 'hash-0', NOW(), NOW())",
            doc_id,
            bank_id,
        )
        await conn.execute(
            "INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text, content_hash) "
            "VALUES ($1, $2, $3, 0, 'doomed text', 'hash-0')",
            chunk_id,
            doc_id,
            bank_id,
        )
        unit_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, document_id, chunk_id, created_at, updated_at) "
            "VALUES ($1, $2, 'doomed fact', 'experience', $3, $4, NOW(), NOW())",
            unit_id,
            bank_id,
            doc_id,
            chunk_id,
        )
        await _record_event(conn, bank_id, unit_id, "SRC_EDIT")
        assert (
            await conn.fetchval(f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id)
            == 1
        )
    async with pool.acquire() as conn:
        async with conn.transaction():
            await chunk_storage.delete_chunks_by_ids(conn, [chunk_id], bank_id, ops=memory._backend.ops)
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id)
            == 0
        )
        assert await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", unit_id) == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_type_delete_purges_archived_change_events(memory, request_context):
    """The purge predicate must cover
    both memory_units and invalidated_memory_units — a record naming an
    archived id outlives the archived row otherwise."""
    bank_id = f"test-gs-type-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    store = get_memories()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "live world fact")
        archived = await _insert_memory(memory, conn, bank_id, "archived world fact")
        async with conn.transaction():
            assert await store.invalidate_memory(
                conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=str(archived), reason="dup"
            )
        await _record_event(conn, bank_id, live, "SRC_EDIT")
        assert (
            await conn.fetchval(f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id)
            == 2
        )
    async with pool.acquire() as conn:
        async with conn.transaction():
            await store.delete_bank_memories_of_type(
                conn=conn,
                ops=memory._backend.ops,
                fq_table=fq_store_table,
                bank_id=bank_id,
                fact_type="world",
                unit_ids=[str(live)],
            )
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(f"SELECT count(*) FROM {fq_table('memory_change_events')} WHERE bank_id = $1", bank_id)
            == 0
        )
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_uncited_read_text_survives_deletion_with_fresh_verdict(memory, request_context, monkeypatch):
    """A read-mode delta page whose
    text rests on observation O where O is uncited and deleted reads
    fresh after a delta refresh that runs no gate and reads nothing,
    with the text intact. The retraction pass is keyed to cited ids
    (O has no based_on entry, so there is nothing to unsay) and the
    read-mode liveness check only sees the persisted read set, which
    Refresh start prunes against liveness  — so a read-but-not-cited
    deletion is invisible to both after one refresh. Stated in
    model/record.md assumptions and spec.md Further Notes beside the
    NEW_UNLINKED blind spot."""
    bank_id = f"test-gs-probe-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        only = await _insert_observation(conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance=_prov([str(only)], baseline),
        )
        await conn.execute("DELETE FROM memory_units WHERE bank_id = $1 AND id = $2", bank_id, only)
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        # A source in the delta window so the refresh runs (has_sources).
        await _insert_observation(conn, bank_id, "Bob plays drums.")
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day', "
            "content = $3 WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            "## Conventions\n\nAlice plays cello.\n",
        )
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))
    _install_reflect(memory, monkeypatch, text="## Conventions\n\nNothing new.\n", based_on={"observation": []})

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Nothing new."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    async with pool.acquire() as conn:
        stale = await _is_stale(memory, conn, bank_id, mm_id)
        row = await conn.fetchrow(
            f"SELECT content FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
    text = row["content"] or ""
    assert prov["read_memory_ids"] == []
    assert prov["captured_empty"] is True
    assert stale is False, "fresh verdict over text resting on deleted O"
    assert "Alice plays cello." in text, "the page text keeps the deleted observation's words"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_archive_restore_blind_spot_reads_fresh_with_text_intact(memory, request_context, monkeypatch):
    """Archive/restore blind spot: the weekly schedule only refreshes pages
    the rule already flagged. Memory O is archived before refresh start, so
    the refresh prunes it from the previous read set; refresh start prunes O
    from the read set, so the page's provenance no longer names it and no later record
    about O can reach it; restore is not a read and re-adds nothing.
    Supporting detail: restore writes no change record
    (pg/writes.py:523-592). The page then reads FRESH with its text
    intact. This test pins the FRESH verdict on the prune; the assertion
    it pins is read_memory_ids == []. Stated in model/record.md
    assumptions and spec.md Further Notes."""
    bank_id = f"test-gs-archive-restore-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    store = get_memories()
    async with pool.acquire() as conn:
        only = await _insert_memory(memory, conn, bank_id, "Alice plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance=_prov([str(only)], baseline),
        )
        # Archive O before refresh start: invalidate writes a durable SRC_EDIT
        # record above the stamped baseline, so the page reads stale.
        async with conn.transaction():
            assert await store.invalidate_memory(
                conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=str(only), reason="dup"
            )
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        # A source in the delta window so the refresh runs (has_sources).
        await _insert_observation(conn, bank_id, "Bob plays drums.")
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day', "
            "content = $3 WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            "## Conventions\n\nAlice plays cello.\n",
        )
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))
    _install_reflect(memory, monkeypatch, text="## Conventions\n\nNothing new.\n", based_on={"observation": []})

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Nothing new."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    # Restore O after the refresh via the engine revert path (the store
    # restore needs the engine's connection wrapper): restore writes no
    # change record, and the FRESH verdict comes from the refresh start prune
    # that removed O from the read set before this refresh read anything.
    restored = await memory.update_memory_unit(bank_id, str(only), state="valid", request_context=request_context)
    assert restored is not None
    async with pool.acquire() as conn:
        stale = await _is_stale(memory, conn, bank_id, mm_id)
        row = await conn.fetchrow(
            f"SELECT content FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
    text = row["content"] or ""
    assert prov["read_memory_ids"] == []
    assert prov["captured_empty"] is True
    assert stale is False, "fresh verdict over text resting on restored O"
    assert "Alice plays cello." in text, "the page text keeps the restored memory's words"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_non_postgres_prev_read_fallback_stays_unpruned(memory, request_context, monkeypatch):
    """On a non-Postgres backend the refresh-start capture takes no
    snapshot (``_mental_model_event_version`` returns ``(0, None)``), so a
    graph delta refresh falls back to the stored stamp, unpruned: a dead
    id in the previous read set stays in the persisted read set. Driven
    with a fake backend whose connections report ``backend_type !=
    "postgresql"`` while delegating every statement to the real
    connection."""
    from contextlib import asynccontextmanager

    bank_id = f"test-gs-non-postgres-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        dead = await _insert_observation(conn, bank_id, "Alice plays cello.")
        live = await _insert_observation(conn, bank_id, "Bob plays drums.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={},
            graph_provenance=_prov([str(dead), str(live)], baseline),
        )
        await conn.execute("DELETE FROM memory_units WHERE bank_id = $1 AND id = $2", bank_id, dead)
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day' "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )

    class _OracleConn:
        backend_type = "oracle"

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _OracleBackend:
        _wraps_backend = True

        def __init__(self, real):
            self._real = real

        @asynccontextmanager
        async def acquire(self):
            async with self._real.acquire() as conn:
                yield _OracleConn(conn)

    real_backend = await memory._get_backend()

    async def fake_get_backend():
        return _OracleBackend(real_backend)

    monkeypatch.setattr(memory, "_get_backend", fake_get_backend)
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))
    trace = [
        ToolCallTrace(
            tool="recall",
            reason="need facts",
            input={},
            output={"memories": [{"id": str(live), "text": "Bob plays drums.", "fact_type": "observation"}]},
            duration_ms=1,
            iteration=1,
        )
    ]
    _install_reflect(
        memory,
        monkeypatch,
        text="## Conventions\n\nBob plays drums.\n",
        based_on={
            "observation": [_fact(str(live))],
            "world": [],
            "experience": [],
            "mental-models": [],
            "directives": [],
        },
        tool_trace=trace,
    )

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Nothing new."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    prov = refreshed["reflect_response"]["graph_provenance"]
    assert str(dead) in prov["read_memory_ids"], "the prev_read None fallback must keep the dead id (unpruned)"
    assert str(live) in prov["read_memory_ids"]
    assert prov["baseline_seq"] == 0, "no change-record table: the capture reports 0"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_precapture_citation_enters_read_stamp(memory, request_context, monkeypatch):
    """A page whose
    based_on carries M from a pre-capture build (read ids [], no captures
    before that refresh) must read STALE after an SRC_EDIT on M in
    read_memories mode. The superset guarantee now lives at the verdict
    (both paths union live cited ids into the subject set), not in the
    stamp, so this test asserts the verdict and not the stamped ids."""
    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        m = await _insert_observation(conn, bank_id, "M plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories", mode="delta")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(m))]},
            graph_provenance={
                "read_memory_ids": [],
                "captured_empty": True,
                "cited_empty": False,
                "baseline_seq": baseline,
            },
        )
        # A source in the delta window so the refresh runs (has_sources).
        await _insert_observation(conn, bank_id, "Bob plays drums.")
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query, last_memory_seen_at = NOW() - INTERVAL '1 day', "
            "content = $3 WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            "## Conventions\n\nM plays cello.\n",
        )
    monkeypatch.setattr(memory, "_gate_delta_refresh", AsyncMock(return_value=None))
    _install_reflect(memory, monkeypatch, text="## Conventions\n\nNothing new.\n", based_on={"observation": []})

    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

    async def fake_call(*, messages, **kwargs):
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "conventions", "text": "Nothing new."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    async with pool.acquire() as conn:
        await _record_event(conn, bank_id, m, "SRC_EDIT")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_lossy_stamp_reads_stale_without_refresh(memory, request_context):
    """A lossy stamp (based_on {M}, read ids [],
    captured_empty) must read STALE on both paths after SRC_EDIT(M), with
    no refresh in between. The verdict-side cited union is the guarantee;
    the stamp was written by a build that never unioned cited ids."""
    from hindsight_api.engine.memory_engine import _mental_model_stale_scope_from_row

    bank_id = f"test-gs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        m = await _insert_observation(conn, bank_id, "M plays cello.")
        mm_id = await _create_page(memory, bank_id, request_context, graph_stale="read_memories")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_provenance(
            conn,
            bank_id,
            mm_id,
            based_on={"observation": [_fact(str(m))]},
            graph_provenance={
                "read_memory_ids": [],
                "captured_empty": True,
                "cited_empty": False,
                "baseline_seq": baseline,
            },
        )
        await _record_event(conn, bank_id, m, "SRC_EDIT")
        assert await _is_stale(memory, conn, bank_id, mm_id) is True
        row = await conn.fetchrow(
            f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
            f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
        )
        batch = await memory.compute_mental_models_are_stale(
            conn, bank_id, {str(mm_id): _mental_model_stale_scope_from_row(row, key=str(mm_id))}
        )
        assert batch[str(mm_id)] is True
    await memory.delete_bank(bank_id, request_context=request_context)
