"""40cr acceptance: consolidation invalidates only model-judged, in-batch, curatable facts.

The model expresses judgment by emitting a fact in the response's invalidations
list; the reason is per-entry audit metadata carried to the writer, not a signal
the service reads to decide what is current. The eligibility helper checks that
emitted facts are in the batch and curatable. Observation and mental-model ids,
and ids outside the batch, are refused here rather than reaching the writer.

The invalidations also ride the batch's all-or-nothing rule: a response dropped for
a source edit since read applies none of them, and a cancellation arriving while a
failed invalidation rolls back still cancels the job.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import ExitStack, asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine import graph_maintenance
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _BatchLLMResult,
    _FactBatchResponse,
    _Invalidation,
    _apply_invalidations_in_transaction,
    filter_invalidations,
    run_consolidation_job,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM


def _response(*invalidations: dict) -> _FactBatchResponse:
    return _FactBatchResponse(
        facts=[],
        deletes=[],
        invalidations=list(invalidations),
    )


def test_invalidations_carry_model_judgment_from_response() -> None:
    """The response carries the model's invalidation judgment as a first-class action."""
    response = _response({"fact_id": "f-outdated", "reason": "superseded by a later decision"})
    assert [(item.fact_id, item.reason) for item in response.invalidations] == [
        ("f-outdated", "superseded by a later decision")
    ]


def test_only_in_batch_curatable_facts_are_eligible() -> None:
    """Only emitted, in-batch, curatable facts are eligible; each keeps its own reason."""
    batch_ids = {"f-outdated", "f-current"}
    curatable = {"f-outdated", "f-current"}
    # The model judges by emission. f-current is still true, so the model does not
    # emit it and no reason-text scan is needed to keep it out of the eligible set.
    response = _response(
        {"fact_id": "f-outdated", "reason": "superseded"},
        {"fact_id": "f-outside", "reason": "not in this batch"},
        {"fact_id": "obs-1", "reason": "an observation, not a raw fact"},
        {"fact_id": "mm-1", "reason": "a mental model"},
    )
    entries = [(item.fact_id, item.reason) for item in response.invalidations]

    eligible = filter_invalidations(
        entries,
        batch_fact_ids=batch_ids,
        curatable_ids=curatable,
    )

    assert eligible == [("f-outdated", "superseded")], "only the emitted in-batch curatable fact is eligible"
    assert "f-current" not in [fact_id for fact_id, _ in eligible], "a batch fact the model did not emit stays untouched"


def test_in_batch_fact_outside_the_curatable_set_is_refused() -> None:
    """P4: in-batch is not enough — the eligibility helper refuses a non-curatable type.

    ``batch_fact_ids`` and ``curatable_ids`` were always the same set in the pinned cases,
    so the second half of :func:`filter_invalidations` never decided anything. A summary row
    of this batch is in the batch and must still be refused while a raw sibling lands.
    """
    eligible = filter_invalidations(
        [("f-summary", "a derived summary, not a raw fact"), ("f-outdated", "superseded")],
        batch_fact_ids={"f-summary", "f-outdated"},
        curatable_ids={"f-outdated"},
    )
    assert eligible == [("f-outdated", "superseded")], "only the curatable in-batch fact is eligible"


def test_duplicate_invalidations_keep_the_first_reason() -> None:
    """A repeated id resolves to its first reason, per C5."""
    eligible = filter_invalidations(
        [("f-outdated", "first"), ("f-outdated", "second")],
        batch_fact_ids={"f-outdated"},
        curatable_ids={"f-outdated"},
    )
    assert eligible == [("f-outdated", "first")]


def test_invalidation_applies_without_partial_state() -> None:
    """Applying an invalidation archives the fact with its reason and sweeps dependents."""
    from hindsight_api.engine.consolidation.consolidator import apply_invalidation_actions

    calls: list[tuple[str, str]] = []

    def writer(fact_id: str, reason: str) -> bool:
        calls.append((fact_id, reason))
        return True

    swept: list[str] = []

    def sweeper(fact_id: str) -> None:
        swept.append(fact_id)

    apply_invalidation_actions(
        [("f-outdated", "superseded")],
        invalidate=writer,
        sweep=sweeper,
    )

    assert calls == [("f-outdated", "superseded")]
    assert swept == ["f-outdated"], "every invalidated fact has its dependents swept in the same call"


@pytest.mark.parametrize("eligible", [[], [("f-1", "superseded")]])
def test_apply_handles_each_eligible_entry(eligible: list[tuple[str, str]]) -> None:
    """The apply seam writes exactly the eligible entries, and none when empty."""
    from hindsight_api.engine.consolidation.consolidator import apply_invalidation_actions

    calls: list[str] = []

    def writer(fact_id: str, reason: str) -> bool:
        calls.append(fact_id)
        return True

    apply_invalidation_actions(
        eligible,
        invalidate=writer,
        sweep=lambda fact_id: None,
    )
    assert calls == [fact_id for fact_id, _ in eligible]


def test_absent_row_is_a_skip_with_no_sweep() -> None:
    """P6: the documented skip rule — a writer that found no live row cancels the sweep.

    An absent row is not a rollback: the batch moves on to the next entry. Nothing was
    archived, so nothing derived from it may be swept.
    """
    from hindsight_api.engine.consolidation.consolidator import apply_invalidation_actions

    written: list[str] = []
    swept: list[str] = []

    def writer(fact_id: str, reason: str) -> bool:
        written.append(fact_id)
        return fact_id == "f-live"

    apply_invalidation_actions(
        [("f-absent", "superseded"), ("f-live", "superseded")],
        invalidate=writer,
        sweep=swept.append,
    )

    assert written == ["f-absent", "f-live"], "the writer is still asked about every entry"
    assert swept == ["f-live"], "the absent row is skipped, the live sibling still sweeps"


def test_explicit_null_reason_keeps_the_model_judgment() -> None:
    """C4/D6/R8: emission is the judgment; ``reason`` is optional audit metadata.

    A provider that emits an explicit JSON ``null`` for an optional string must not
    lose the whole invalidation: ``_validate_items`` would otherwise drop a valid
    judgment over metadata the writer accepts as ``str | None``.
    """
    response = _response({"fact_id": "f-1", "reason": None})
    assert [(item.fact_id, item.reason) for item in response.invalidations] == [("f-1", "")], (
        "an explicit null reason is empty metadata, not a reason to drop the judgment"
    )
    eligible = filter_invalidations(
        [(item.fact_id, item.reason) for item in response.invalidations],
        batch_fact_ids={"f-1"},
        curatable_ids={"f-1"},
    )
    assert eligible == [("f-1", "")], "the judgment survives and stays eligible"


def _paragraph(system: str, opening: str) -> str:
    """The one paragraph a prompt section starts with, up to its blank line."""
    start = system.index(opening)
    return system[start : system.index("\n\n", start)]


def test_invalidation_rule_separates_wrong_facts_from_superseded_ones() -> None:
    """P3: the invalidation rule must not sentence history the TIME rule keeps.

    The rule used to say only that an earlier in-batch fact is "outdated" when a later fact
    shows it, which is exactly the in-batch case ``_TIME_RULE`` keeps as the earlier state:
    two rules, one input, no stated priority. They now agree — invalidation is for a fact
    this batch shows to be WRONG, while a state merely superseded by a later one stays
    recorded as history.
    """
    system = build_consolidation_system_prompt()
    rule = _paragraph(system, "`invalidations` (rare)")
    time_rule = system[system.index("## TIME") : system.index("## WHAT AN OBSERVATION IS")]

    assert "outdated" not in rule, "the superseded-history reading is what the TIME rule contradicts"
    assert "wrong" in rule.lower(), "the rule must name the WRONG-fact criterion"
    assert "superseded" in rule, "and must say that merely being superseded is not wrong"
    assert "invalidat" in time_rule.lower(), "the TIME rule must point at the invalidation rule"


# ── the production path's canonical sequence, without a database ───────────────


class _FakeStore:
    """The store reads/writes the invalidation step makes, recorded in call order."""

    def __init__(self, order: list[str], *, live: bool, moved: bool) -> None:
        self.order = order
        self.live = live
        self.moved = moved

    async def lock_live_memory_ids(self, *, conn, fq_table, bank_id, unit_ids):
        self.order.append("live")
        return {str(unit) for unit in unit_ids} if self.live else set()

    async def invalidate_memory(self, *, conn, fq_table, bank_id, unit_id, reason):
        self.order.append("move")
        return self.moved


class _FakeEngine:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def sweep_invalidation_dependents(self, conn, bank_id: str, fact_ids: list[str]) -> None:
        self.order.append("sweep")


def _recorded_sequence_patches(order: list[str]) -> ExitStack:
    """Record the graph-maintenance enqueues the step owes, in the order they run."""

    async def relink(conn, bank_id: str, unit_ids: list[str]) -> int:
        order.append("relink")
        return 0

    async def prune(conn, bank_id: str, unit_ids: list[str]) -> int:
        order.append("prune")
        return 0

    stack = ExitStack()
    stack.enter_context(patch.object(graph_maintenance, "enqueue_relink_victims", new=relink))
    stack.enter_context(patch.object(graph_maintenance, "enqueue_entity_prune_candidates", new=prune))
    return stack


async def _drive_invalidation(store: _FakeStore, engine: _FakeEngine, entries: list[tuple[str, str]]) -> None:
    with ExitStack() as stack:
        stack.enter_context(patch.object(consolidator_module, "get_memories", return_value=store))
        stack.enter_context(_recorded_sequence_patches(store.order))
        await _apply_invalidations_in_transaction(
            conn=object(), bank_id="bank", entries=entries, memory_engine=engine
        )


@pytest.mark.asyncio
async def test_invalidation_mirrors_the_canonical_sequence() -> None:
    """P2: sweep, capture the relink victims and prune candidates, move, then sweep again."""
    order: list[str] = []
    store = _FakeStore(order, live=True, moved=True)
    fact_id = str(uuid.uuid4())

    await _drive_invalidation(store, _FakeEngine(order), [(fact_id, "superseded")])

    assert order == ["live", "sweep", "relink", "prune", "move", "sweep"], (
        "the canonical invalidate sequence: the captures run before the row (and its links "
        "and postings) go, and the second sweep catches a racing observation insert"
    )


@pytest.mark.asyncio
async def test_invalidation_skips_an_absent_live_row() -> None:
    """P6/P2: an absent row is a skip with no sweep and no enqueue (absence is read first)."""
    order: list[str] = []
    store = _FakeStore(order, live=False, moved=False)

    await _drive_invalidation(store, _FakeEngine(order), [(str(uuid.uuid4()), "superseded")])

    assert order == ["live"], "the row is gone: no sweep, no enqueue, no move"


@pytest.mark.asyncio
async def test_invalidation_writer_false_skips_the_post_move_sweep() -> None:
    """P5: a writer that found no row to move does not get the post-move sweep.

    Through the SQL store this cannot happen — the share-locked liveness read settles the
    absence before the sweep — so a store whose liveness read and write disagree is the only
    way to reach the guard. It still must not sweep after a move that did not happen.
    """
    order: list[str] = []
    store = _FakeStore(order, live=True, moved=False)

    await _drive_invalidation(store, _FakeEngine(order), [(str(uuid.uuid4()), "superseded")])

    assert order == ["live", "sweep", "relink", "prune", "move"], "no second sweep after an unmoved row"


# ── the batch's all-or-nothing rule, on a real database ────────────────────────


def _llm(callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


def _override_config(memory: MemoryEngine, **overrides):
    raw = _get_raw_config()
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})
    return patch.object(memory._config_resolver, "resolve_full_config", return_value=fake)


async def _insert_fact(conn, bank_id: str, text: str, tags: list[str]) -> uuid.UUID:
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, observation_scopes, created_at)
        VALUES ($1, $2, $3, 'experience', $4, $5::jsonb, now())
        """,
        mem_id,
        bank_id,
        text,
        tags,
        json.dumps(None),
    )
    return mem_id


async def _run_job(memory: MemoryEngine, bank_id: str, request_context, callback, *extra):
    """One ``run_consolidation_job`` with a mock LLM and a single-threaded batch."""
    original = memory._consolidation_llm_config
    memory._consolidation_llm_config = _llm(callback)
    try:
        with ExitStack() as stack:
            for ctx in (
                *extra,
                _override_config(memory, consolidation_llm_batch_size=4, consolidation_llm_parallelism=1),
                patch.object(memory, "submit_async_consolidation"),
            ):
                stack.enter_context(ctx)
            return await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
    finally:
        memory._consolidation_llm_config = original


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_changed_source_discard_applies_no_invalidation(memory: MemoryEngine, request_context):
    """#4831's discard covers invalidations: a stale-text response archives nothing.

    ``_sources_changed_since_read`` drops the whole response when a source fact was
    edited after the batch read it, stamps included. Invalidations rode along outside
    that reset, so a discarded response still archived a fact the edit deliberately
    kept live and swept its dependent observations.
    """
    bank_id = f"inval-stale-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    sweeps: list[list[str]] = []
    original_sweep = memory.sweep_invalidation_dependents

    async def record_sweep(conn, bank_id: str, fact_ids: list[str]) -> None:
        sweeps.append([str(fact) for fact in fact_ids])
        await original_sweep(conn, bank_id, fact_ids)

    async def stale_answer(**kwargs):
        # The batch read this fact; a curation edit then stamped ``updated_at``, so the
        # judgment is about text the bank no longer holds.
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_units SET updated_at = updated_at + interval '1 minute' WHERE id = $1",
                fact_id,
            )
        return _BatchLLMResult(invalidations=[_Invalidation(fact_id=str(fact_id), reason="superseded")])

    try:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_fact(conn, bank_id, "Alice lives in Berlin", ["user:alice"])

        await _run_job(
            memory,
            bank_id,
            request_context,
            lambda messages, scope: _FactBatchResponse(),
            patch.object(consolidator_module, "_consolidate_batch_with_llm", new=stale_answer),
            patch.object(memory, "sweep_invalidation_dependents", new=record_sweep),
        )

        async with memory._pool.acquire() as conn:
            live = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND fact_type = 'experience'", bank_id
            )
            archived = await conn.fetchval(
                "SELECT count(*) FROM invalidated_memory_units WHERE bank_id = $1", bank_id
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", fact_id)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert live == 1, "the discarded response must leave the fact live"
    assert archived == 0, "the discarded response must archive nothing"
    assert stamped is None, "the discarded response must not stamp its sources"
    assert sweeps == [], "the discarded response must not sweep dependent observations"


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_invalidation_applies_through_the_store_and_records_the_reason(memory: MemoryEngine, request_context):
    """A model-judged invalidation archives the fact through the store, reason and all.

    The archive move must go through the memories store that owns ``memory_units``: the
    schema resolver refuses a store-owned table named from outside the store (#4969), so
    a direct Postgres-writer call fails every batch that emits an invalidation.
    """
    bank_id = f"inval-apply-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    sweeps: list[list[str]] = []
    original_sweep = memory.sweep_invalidation_dependents

    async def record_sweep(conn, bank_id: str, fact_ids: list[str]) -> None:
        sweeps.append([str(fact) for fact in fact_ids])
        await original_sweep(conn, bank_id, fact_ids)

    async def judged_answer(**kwargs):
        return _BatchLLMResult(
            invalidations=[_Invalidation(fact_id=str(fact_id), reason="superseded by a later decision")]
        )

    try:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_fact(conn, bank_id, "Alice lives in Berlin", ["user:alice"])

        result = await _run_job(
            memory,
            bank_id,
            request_context,
            lambda messages, scope: _FactBatchResponse(),
            patch.object(consolidator_module, "_consolidate_batch_with_llm", new=judged_answer),
            patch.object(memory, "sweep_invalidation_dependents", new=record_sweep),
        )

        async with memory._pool.acquire() as conn:
            live = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", fact_id)
            archived = await conn.fetchval(
                "SELECT invalidation_reason FROM invalidated_memory_units WHERE id = $1", fact_id
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert result["memories_failed"] == 0, "an applied invalidation is a judgment, not a failed batch"
    assert live == 0, "the judged fact is archived, out of every recall surface"
    assert archived == "superseded by a later decision", "the writer records the model's own reason"
    assert sweeps == [[str(fact_id)], [str(fact_id)]], (
        "the archived fact's dependents are swept before the move and again after it, "
        "both inside the same transaction"
    )


class _TransactionCancelledOnExit:
    """The real transaction, whose failing exit delivers a cancellation instead."""

    def __init__(self, inner) -> None:
        self._inner = inner

    async def __aenter__(self):
        return await self._inner.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        # The real ROLLBACK runs first, then the cancellation arrives — the shape a
        # worker shutdown or operation timeout takes while the transaction unwinds.
        await self._inner.__aexit__(exc_type, exc, tb)
        if exc_type is not None:
            raise asyncio.CancelledError()
        return False


class _ConnectionProxy:
    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def transaction(self, *args, **kwargs):
        return _TransactionCancelledOnExit(self._inner.transaction(*args, **kwargs))


@asynccontextmanager
async def _proxy_connections(backend):
    async with acquire_with_retry(backend) as conn:
        yield _ConnectionProxy(conn)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_cancellation_during_invalidation_rollback_escapes(memory: MemoryEngine, request_context):
    """A cancellation while a failed invalidation rolls back must still cancel.

    The outer handler exists to turn an ordinary invalidation error into a ``failed``
    result the job can bisect. Once ``invalidation_failed`` is set, a ``CancelledError``
    delivered as the transaction unwinds reached that same branch and came back as a
    failed batch; asyncio never re-delivers a swallowed cancellation, so a job that was
    being shut down kept processing later batches.
    """
    bank_id = f"inval-cancel-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_fact(conn, bank_id, "Alice lives in Berlin", ["user:alice"])

        async def stale_answer(**kwargs):
            return _BatchLLMResult(invalidations=[_Invalidation(fact_id=str(fact_id), reason="superseded")])

        with pytest.raises(asyncio.CancelledError):
            await _run_job(
                memory,
                bank_id,
                request_context,
                lambda messages, scope: _FactBatchResponse(),
                patch.object(consolidator_module, "_consolidate_batch_with_llm", new=stale_answer),
                patch.object(
                    consolidator_module,
                    "_apply_invalidations_in_transaction",
                    new=AsyncMock(side_effect=RuntimeError("invalidate writer failed")),
                ),
                patch.object(consolidator_module, "acquire_with_retry", new=_proxy_connections),
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_a_failing_sweep_rolls_the_archive_move_back(memory: MemoryEngine, request_context):
    """P2/P5: the sweep after the move shares the move's transaction, so its failure rolls back.

    ``_apply_invalidations_in_transaction`` is the only production path for an invalidation;
    an exception from its sweep must escape into the batch transaction, not be turned into a
    completed batch. The move, its change record and the stamps roll back with it, and the
    fact stays live and pending for the next round.
    """
    bank_id = f"inval-sweep-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    sweeps: list[list[str]] = []

    async def fail_on_the_second_sweep(conn, bank_id: str, fact_ids: list[str]) -> None:
        sweeps.append([str(fact) for fact in fact_ids])
        if len(sweeps) == 2:
            raise RuntimeError("dependent sweep failed")

    try:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_fact(conn, bank_id, "Alice lives in Berlin", ["user:alice"])

        async def judged_answer(**kwargs):
            return _BatchLLMResult(invalidations=[_Invalidation(fact_id=str(fact_id), reason="superseded")])

        result = await _run_job(
            memory,
            bank_id,
            request_context,
            lambda messages, scope: _FactBatchResponse(),
            patch.object(consolidator_module, "_consolidate_batch_with_llm", new=judged_answer),
            patch.object(memory, "sweep_invalidation_dependents", new=fail_on_the_second_sweep),
        )

        async with memory._pool.acquire() as conn:
            live = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", fact_id)
            archived = await conn.fetchval(
                "SELECT count(*) FROM invalidated_memory_units WHERE id = $1", fact_id
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", fact_id)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert sweeps == [[str(fact_id)], [str(fact_id)]], "the dependent sweep runs before and after the move"
    assert result["status"] == "completed"
    assert result["memories_failed"] == 1, "a failed invalidation step is a failed batch, not a completion"
    assert live == 1, "the archive move rolled back with the sweep that failed after it"
    assert archived == 0, "nothing reached the archive"
    assert stamped is None, "a failed batch does not stamp its fact as consolidated"
