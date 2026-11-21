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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine import graph_maintenance
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _apply_invalidations_in_transaction,
    _BatchLLMResult,
    _FactBatchResponse,
    _Invalidation,
    filter_invalidations,
    run_consolidation_job,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage


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


async def _insert_fact(conn, bank_id: str, text: str, tags: list[str], observation_scopes: Any = None) -> uuid.UUID:
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
        json.dumps(observation_scopes),
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


# ── C1: the schema-correction re-ask must not drop the judgment ───────────────


class _ScriptedLLM:
    """A model double: scripted replies, recording every request it is sent."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def call(self, **kwargs: Any) -> LLMCallResult:
        self.requests.append(kwargs)
        return LLMCallResult(content=self.replies.pop(0), usage=TokenUsage())


@pytest.mark.asyncio
async def test_schema_correction_names_the_invalidations_field() -> None:
    """C1: a re-answer is taught every top-level field, or the judgment is lost.

    A reply whose top-level shape is wrong (a list instead of an object) gets one
    re-ask carrying the errors. The correction used to name only ``facts`` and
    ``deletes``: a re-answer taught that object omits ``invalidations``, the field
    defaults to ``[]``, validation passes, and the batch stamps the facts while the
    model's judgment is silently discarded.
    """
    llm = _ScriptedLLM([[{"facts": []}], _FactBatchResponse()])
    call_kwargs = {
        "messages": [
            {"role": "system", "content": "consolidate these facts"},
            {"role": "user", "content": "the facts"},
        ],
        "response_format": _FactBatchResponse,
    }

    answer = await consolidator_module._call_with_one_reask(llm, call_kwargs, _FactBatchResponse)

    assert isinstance(answer, _FactBatchResponse), "the second reply is returned as the answer"
    assert len(llm.requests) == 2, "exactly one re-ask after a rejected top-level shape"
    correction = llm.requests[1]["messages"][-1]["content"]
    assert "invalidations" in correction, "the correction must name the field an object re-answer must carry"
    assert "Not a list" in correction, "the shape instruction survives"


# ── C2: an invalidated fact is a judgment, not an unaddressed fact ────────────


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_invalidated_fact_is_not_reasked_as_unaddressed(memory: MemoryEngine, request_context):
    """C2: the archived fact must leave the re-ask loop and the skipped count.

    A fact the same response invalidated is absent from the created/updated sets and
    from the model's own skips, so the per-memory result used to call it
    ``no_durable_knowledge``. The caller reads that as an oversight and asks it again,
    up to two more times; the re-ask's creates are dropped as non-live, so the dead
    fact burns recall+embed+LLM while the stats report it as never addressed.
    """
    bank_id = f"inval-reask-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    calls: list[str] = []
    invalidated: uuid.UUID | None = None
    skipped: uuid.UUID | None = None

    def answer(messages, scope):
        if scope == "consolidation":
            calls.append("\n".join(str(message.get("content")) for message in messages))
        return _FactBatchResponse(
            facts=[{"fact_id": str(skipped), "kind": "momentary"}],
            invalidations=[{"fact_id": str(invalidated), "reason": "contradicted by a later fact"}],
        )

    try:
        async with memory._pool.acquire() as conn:
            invalidated = await _insert_fact(conn, bank_id, "Alice lives in Berlin", ["user:alice"])
            skipped = await _insert_fact(conn, bank_id, "The cat sat on the mat", ["user:alice"])

        result = await _run_job(memory, bank_id, request_context, answer)

        async with memory._pool.acquire() as conn:
            archived = await conn.fetchval("SELECT count(*) FROM invalidated_memory_units WHERE id = $1", invalidated)
            skipped_live = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", skipped)
            skipped_stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", skipped)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert len(calls) == 1, (
        "a model-judged invalidation is a judgment, not an unaddressed fact: it must not be re-asked"
        + (f"\nsecond consolidation request:\n{calls[1]}" if len(calls) > 1 else "")
    )
    assert result["memories_processed"] == 2, "each fact of the batch is processed once"
    assert result["skipped"] == 1, "only the model's own skip counts as skipped; the archived fact was judged"
    assert archived == 1, "the judged fact is archived"
    assert skipped_live == 1, "the model's skip settles its own fact"
    assert skipped_stamped is not None, "a settled fact is stamped consolidated"


# ── C4: eligibility is the set the model was shown, not the whole batch ───────


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_triage_settled_fact_cannot_be_invalidated(memory: MemoryEngine, request_context):
    """C4: a triage-settled fact was never in the prompt, so its id cannot be eligible.

    ``llm_memories`` removes triage-settled facts before the call. The eligibility
    filter used the whole ``memories`` batch instead, so a hallucinated id of a
    settled fact passed and the batch archived a fact the model never saw, on a
    judgment it never made.
    """
    from hindsight_api.engine.consolidation.triage import TriageResult

    bank_id = f"inval-triage-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    prompts: list[str] = []
    triaged: uuid.UUID | None = None
    shown: uuid.UUID | None = None

    async def settle(*args, **kwargs):
        return TriageResult(chatter={str(triaged)})

    def answer(messages, scope):
        if scope == "consolidation":
            prompts.append("\n".join(str(message.get("content")) for message in messages))
        return _FactBatchResponse(
            facts=[{"fact_id": str(shown), "kind": "momentary"}],
            invalidations=[{"fact_id": str(triaged), "reason": "hallucinated id of a settled fact"}],
        )

    try:
        async with memory._pool.acquire() as conn:
            triaged = await _insert_fact(conn, bank_id, "Triage settled this fact", ["user:alice"])
            shown = await _insert_fact(conn, bank_id, "This fact was shown to the model", ["user:alice"])

        await _run_job(
            memory,
            bank_id,
            request_context,
            answer,
            patch.object(consolidator_module, "triage_model", new=lambda config: object()),
            patch.object(consolidator_module, "triage_facts", new=settle),
        )

        async with memory._pool.acquire() as conn:
            triaged_live = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", triaged)
            triaged_archived = await conn.fetchval(
                "SELECT count(*) FROM invalidated_memory_units WHERE id = $1", triaged
            )
            shown_stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", shown)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert prompts, "the fact the model was shown must still reach the LLM"
    assert "Triage settled this fact" not in prompts[0], "the settled fact was never in the prompt"
    assert triaged_live == 1, "a fact the model never saw cannot be archived on its judgment"
    assert triaged_archived == 0, "nothing reached the archive"
    assert shown_stamped is not None, "the shown fact is settled by its own skip"


# ── C7: the stamp must not overwrite the sweep's requeue of a co-source ───────


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_in_batch_skipped_co_source_stays_pending_after_invalidation(memory: MemoryEngine, request_context):
    """C7: invalidated A and skipped B share one observation; the next round rebuilds it from B.

    The invalidation step sweeps the archived observation's dependents, and that sweep
    resets ``consolidated_at`` on the observation's surviving co-sources so the next
    round rebuilds it from them. The batch stamped every sub-batch id *after* that sweep,
    so an in-batch co-source B was requeued and immediately stamped: the next fetch saw
    nothing pending, the observation was never rebuilt, and B's contribution was lost.
    """
    bank_id = f"inval-stamp-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    archived_fact: uuid.UUID | None = None
    skipped_fact: uuid.UUID | None = None
    observation: uuid.UUID | None = None
    requests: list[str] = []
    replies = [
        # Round 1: the model judges A wrong and settles B as chatter. Both share one observation.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(skipped_fact), "kind": "momentary"}],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 2: the requeued survivor, consolidated again, rebuilds its observation.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(skipped_fact),
                    "kind": "durable",
                    "about": "Alice's home",
                    "adds": "yes",
                    "new": ["Alice lives in Potsdam"],
                }
            ]
        ),
    ]

    def answer(messages, scope):
        if scope == "consolidation":
            requests.append("\n".join(str(message.get("content")) for message in messages))
        return replies.pop(0)() if replies else _FactBatchResponse()

    try:
        async with memory._pool.acquire() as conn:
            archived_fact = await _insert_fact(conn, bank_id, "Alice lives in Berlin", ["user:alice"])
            skipped_fact = await _insert_fact(conn, bank_id, "Alice lives in Potsdam", ["user:alice"])
            observation = uuid.uuid4()
            await conn.execute(
                "INSERT INTO memory_units(id, bank_id, text, fact_type, source_memory_ids) "
                "VALUES($1,$2,'Alice lives in Berlin. Alice lives in Potsdam.','observation',$3)",
                observation,
                bank_id,
                [archived_fact, skipped_fact],
            )

        result = await _run_job(memory, bank_id, request_context, answer)

        async with memory._pool.acquire() as conn:
            archived = await conn.fetchval("SELECT count(*) FROM invalidated_memory_units WHERE id = $1", archived_fact)
            original_left = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", observation)
            rebuilt = await conn.fetchrow(
                "SELECT text, source_memory_ids FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
                bank_id,
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", skipped_fact)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert archived == 1, "the judged fact is archived"
    assert original_left == 0, "the sweep drops the observation built on the archived fact"
    assert len(requests) == 2, "the requeued survivor is fetched again in the same job, as the next round"
    assert "Alice lives in Potsdam" in requests[1], "the next round consolidates the survivor B"
    assert rebuilt is not None, "the next round rebuilds the observation from the surviving co-source"
    assert rebuilt["text"] == "Alice lives in Potsdam", "the rebuild carries B's knowledge, not the old text"
    assert list(rebuilt["source_memory_ids"]) == [skipped_fact], "the rebuilt observation stands on B alone"
    assert stamped is not None, "B is stamped only after the round that rebuilt it"


# C7 across passes: an earlier scope's requeue must survive the final stamp


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_earlier_scope_requeue_survives_the_final_pass_stamp(memory: MemoryEngine, request_context):
    """C7 across passes: the multi-scope caller stamps A and B only on the final pass.

    A memory consolidated at several tag scopes runs one pass per scope over the same
    sub-batch, and the consolidated_at stamp lands in the last pass's transaction. When
    the first pass invalidates A, its sweep deletes the observation A and B share and resets
    consolidated_at on the surviving co-source B. The final pass then stamped every id in
    the sub-batch, B included, after that sweep: the requeue was cancelled, the next fetch saw
    nothing pending, and the observation the sweep deleted was never rebuilt from B. Re-emitting
    the invalidation in the final pass does not rescue it - A is no longer live, so the step
    skips without a sweep. The requeue has to land after the final stamp.
    """
    bank_id = f"inval-multiscope-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    archived_fact: uuid.UUID | None = None
    skipped_fact: uuid.UUID | None = None
    observation: uuid.UUID | None = None
    requests: list[str] = []
    replies = [
        # Round 1, first scope: the model judges A wrong and settles B as chatter. A and B share
        # one observation at this scope, so the sweep below requeues B.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(skipped_fact), "kind": "momentary"}],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 1, final scope: A is archived already, so the repeated judgment skips at the
        # liveness read; the stamp this pass writes is what must not cancel the requeue.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(skipped_fact), "kind": "momentary"}],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 2, first scope: the requeued survivor rebuilds the observation there.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(skipped_fact),
                    "kind": "durable",
                    "about": "Alice's home",
                    "adds": "yes",
                    "new": ["Alice lives in Potsdam"],
                }
            ]
        ),
        # Round 2, final scope: B is settled again, so the rebuilt round stamps it.
        lambda: _FactBatchResponse(facts=[{"fact_id": str(skipped_fact), "kind": "momentary"}]),
    ]

    def answer(messages, scope):
        if scope == "consolidation":
            requests.append("\n".join(str(message.get("content")) for message in messages))
        return replies.pop(0)() if replies else _FactBatchResponse()

    try:
        async with memory._pool.acquire() as conn:
            # Same tags and the per_tag scope spec on both facts: one sub-batch, one LLM call
            # per tag, and the user:alice pass is the first of the two.
            archived_fact = await _insert_fact(
                conn, bank_id, "Alice lives in Berlin", ["user:alice", "session:s1"], "per_tag"
            )
            skipped_fact = await _insert_fact(
                conn, bank_id, "Alice lives in Potsdam", ["user:alice", "session:s1"], "per_tag"
            )
            observation = uuid.uuid4()
            await conn.execute(
                "INSERT INTO memory_units(id, bank_id, text, fact_type, tags, source_memory_ids) "
                "VALUES($1,$2,'Alice lives in Berlin. Alice lives in Potsdam.','observation',$3::text[],$4)",
                observation,
                bank_id,
                ["user:alice"],
                [archived_fact, skipped_fact],
            )

        result = await _run_job(memory, bank_id, request_context, answer)

        async with memory._pool.acquire() as conn:
            archived = await conn.fetchval("SELECT count(*) FROM invalidated_memory_units WHERE id = $1", archived_fact)
            original_left = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", observation)
            rebuilt = await conn.fetchrow(
                "SELECT text, source_memory_ids FROM memory_units "
                "WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
                bank_id,
                ["user:alice"],
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", skipped_fact)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert archived == 1, "the judged fact is archived"
    assert original_left == 0, "the earlier scope's sweep drops the shared observation"
    assert len(requests) == 4, (
        "the earlier scope's requeue is fetched again after the final pass; the final stamp "
        "cancelled it, so the deleted observation never came back"
    )
    assert "Alice lives in Potsdam" in requests[2], "the next round consolidates the survivor at the earlier scope"
    assert rebuilt is not None, "the earlier scope's observation is rebuilt from the surviving co-source"
    assert rebuilt["text"] == "Alice lives in Potsdam", "the rebuild carries B's knowledge, not the old text"
    assert list(rebuilt["source_memory_ids"]) == [skipped_fact], "the rebuilt observation stands on B alone"
    assert stamped is not None, "B is stamped only after the round that rebuilt it"


# C7 across passes through the action loop: a requeue the pre-read cannot see


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_update_widened_requeue_survives_the_final_pass_stamp(memory: MemoryEngine, request_context):
    """C7 across passes: the response's own UPDATE, not the pre-read, names the survivor.

    The first scope's response invalidates A and updates an observation that cited only B,
    citing [A, B]. ``_apply_update_action`` extends the observation's sources with A before
    the invalidation sweeps, so the sweep deletes a row the ``observations_for_sources`` read
    taken before the action loop never returned, and requeues its surviving co-source B.
    Reporting the requeue from that pre-read left B out of the final pass's exclusion set: the
    final stamp cancelled the requeue and the deleted observation was never rebuilt from B.
    """
    bank_id = f"inval-update-requeue-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    archived_fact: uuid.UUID | None = None
    surviving_fact: uuid.UUID | None = None
    observation: uuid.UUID | None = None
    requests: list[str] = []
    replies = [
        # Round 1, first scope: the model judges A wrong and lets B repeat the observation. Both
        # entries cite the same observation, so the response's UPDATE cites [A, B] onto a row
        # that cited only B; the invalidation sweep below deletes it and requeues B.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(archived_fact),
                    "existing": [
                        {
                            "observation_id": str(observation),
                            "observation_about": "Alice's home",
                            "effect": "repeats",
                        }
                    ],
                    "adds": "no",
                },
                {
                    "fact_id": str(surviving_fact),
                    "existing": [
                        {
                            "observation_id": str(observation),
                            "observation_about": "Alice's home",
                            "effect": "changes",
                            "text": "Alice lives in Potsdam",
                        }
                    ],
                    "adds": "no",
                },
            ],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 1, final scope: A's repeated judgment skips at the liveness read and B is settled
        # again, so the stamp this pass writes is what must not cancel the first scope's requeue.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 2, first scope: the requeued survivor rebuilds the observation there.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(surviving_fact),
                    "kind": "durable",
                    "about": "Alice's home",
                    "adds": "yes",
                    "new": ["Alice lives in Potsdam"],
                }
            ]
        ),
        # Round 2, final scope: B is settled again, so the rebuilt round stamps it.
        lambda: _FactBatchResponse(facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}]),
    ]

    def answer(messages, scope):
        if scope == "consolidation":
            requests.append("\n".join(str(message.get("content")) for message in messages))
        return replies.pop(0)() if replies else _FactBatchResponse()

    # The earlier scope's observation is built by the product from B alone, so the recall pool
    # really contains it: a manually inserted row carries no embedding and consolidation's
    # observation recall never returns it. One warm-up create per per_tag scope.
    def warm_create() -> _FactBatchResponse:
        return _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(surviving_fact),
                    "kind": "durable",
                    "about": "Alice's home",
                    "adds": "yes",
                    "new": ["Alice lives in Potsdam"],
                }
            ]
        )

    warm_replies = [warm_create, warm_create]

    def warm_answer(messages, scope):
        return warm_replies.pop(0)() if warm_replies else _FactBatchResponse()

    try:
        async with memory._pool.acquire() as conn:
            # Same tags and the per_tag scope spec on both facts: one sub-batch, one LLM call per
            # tag, and the user:alice pass is the first of the two.
            surviving_fact = await _insert_fact(
                conn, bank_id, "Alice lives in Potsdam", ["user:alice", "session:s1"], "per_tag"
            )

        warm = await _run_job(memory, bank_id, request_context, warm_answer)
        assert warm["status"] == "completed"

        async with memory._pool.acquire() as conn:
            observation = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation' "
                "AND tags @> $2::varchar[]",
                bank_id,
                ["user:alice"],
            )
            assert observation is not None, "the warm-up builds the earlier scope's observation"
            # The observation cites B only: the pre-action read for A does not return it.
            await conn.execute("UPDATE memory_units SET consolidated_at = NULL WHERE id = $1", surviving_fact)
            archived_fact = await _insert_fact(
                conn, bank_id, "Alice lives in Berlin", ["user:alice", "session:s1"], "per_tag"
            )

        # Only the measured run's calls count.
        requests.clear()
        result = await _run_job(memory, bank_id, request_context, answer)

        async with memory._pool.acquire() as conn:
            archived = await conn.fetchval("SELECT count(*) FROM invalidated_memory_units WHERE id = $1", archived_fact)
            original_left = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", observation)
            rebuilt = await conn.fetchrow(
                "SELECT text, source_memory_ids FROM memory_units "
                "WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
                bank_id,
                ["user:alice"],
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", surviving_fact)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert archived == 1, "the judged fact is archived"
    assert original_left == 0, "the update-widened observation is swept by the invalidation"
    assert len(requests) == 4, (
        "the sweep's requeue of B is fetched again after the final pass; reading the requeue "
        "from before the action loop left B out of the exclusion set, the stamp cancelled it, "
        "and the deleted observation never came back"
    )
    assert "Alice lives in Potsdam" in requests[2], "the next round consolidates the survivor at the earlier scope"
    assert rebuilt is not None, "the earlier scope's observation is rebuilt from the surviving co-source"
    assert rebuilt["text"] == "Alice lives in Potsdam", "the rebuild carries B's knowledge, not the old text"
    assert list(rebuilt["source_memory_ids"]) == [surviving_fact], "the rebuilt observation stands on B alone"
    assert stamped is not None, "B is stamped only after the round that rebuilt it"


# C7 across passes through the merge fold: a requeue only the post-fold read can see


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_fold_widened_requeue_survives_the_final_pass_stamp(memory: MemoryEngine, request_context):
    """C7 across passes: the response's own MERGE moves A onto the target after the pre-read.

    The first scope's response deletes an observation citing A and folds it into a target
    citing only B. ``_fold_deleted_evidence`` extends that target with A after the
    ``observations_for_sources`` read taken before the action loop, so the invalidation sweep
    deletes a row the pre-read never returned and requeues its surviving co-source B. Reporting
    the requeue from that pre-read left B out of the final pass's exclusion set: the final stamp
    cancelled the requeue and the merged-away observation was never rebuilt from B.
    """
    bank_id = f"inval-fold-requeue-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    archived_fact: uuid.UUID | None = None
    surviving_fact: uuid.UUID | None = None
    merged_away: uuid.UUID | None = None
    target: uuid.UUID | None = None
    requests: list[str] = []
    warm_replies = [
        # The measured DELETE/MERGE can only name rows the consolidation recall returned, so the
        # product builds one observation per fact first: a manually inserted row carries no
        # embedding and the recall never returns it. One warm-up call per per_tag scope.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(archived_fact),
                    "kind": "durable",
                    "about": "Alice's Berlin flat",
                    "adds": "yes",
                    "new": ["Alice's Berlin flat lease ends in March"],
                },
                {
                    "fact_id": str(surviving_fact),
                    "kind": "durable",
                    "about": "Alice's plants",
                    "adds": "yes",
                    "new": ["Alice moved her plants to the Potsdam flat"],
                },
            ]
        ),
        # The final scope settles both facts; only the first scope's observations are needed.
        lambda: _FactBatchResponse(
            facts=[
                {"fact_id": str(archived_fact), "kind": "momentary"},
                {"fact_id": str(surviving_fact), "kind": "momentary"},
            ]
        ),
    ]

    def warm_answer(messages, scope):
        return warm_replies.pop(0)() if scope == "consolidation" and warm_replies else _FactBatchResponse()

    # Round 1, first scope: merge the A-only observation into the B-only target, judging A wrong.
    # The fold widens the target with A, the sweep deletes it, and B is its surviving co-source.
    replies = [
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}],
            deletes=[
                {
                    "observation_id": str(merged_away),
                    "merged_into": str(target),
                    "reason": "both observations describe Alice's flat",
                }
            ],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 1, final scope: A's repeated judgment skips at the liveness read; the stamp this
        # pass writes is what must not cancel the first scope's requeue of B.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 2, first scope: the requeued survivor B rebuilds the observation.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(surviving_fact),
                    "kind": "durable",
                    "about": "Alice's plants",
                    "adds": "yes",
                    "new": ["Alice moved her plants to the Potsdam flat"],
                }
            ]
        ),
        # Round 2, final scope: B is settled again, so the rebuilt round stamps it.
        lambda: _FactBatchResponse(facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}]),
    ]

    def answer(messages, scope):
        if scope == "consolidation":
            requests.append("\n".join(str(message.get("content")) for message in messages))
        return replies.pop(0)() if replies else _FactBatchResponse()

    try:
        async with memory._pool.acquire() as conn:
            # Same tags and the per_tag scope spec on both facts: one sub-batch, one LLM call per
            # tag, and the user:alice pass is the first of the two.
            archived_fact = await _insert_fact(
                conn, bank_id, "Alice's Berlin flat lease ends in March", ["user:alice", "session:s1"], "per_tag"
            )
            surviving_fact = await _insert_fact(
                conn, bank_id, "Alice moved her plants to the Potsdam flat", ["user:alice", "session:s1"], "per_tag"
            )

        warm = await _run_job(memory, bank_id, request_context, warm_answer)
        assert warm["status"] == "completed"

        async with memory._pool.acquire() as conn:
            merged_away = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation' AND text = $2",
                bank_id,
                "Alice's Berlin flat lease ends in March",
            )
            target = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation' AND text = $2",
                bank_id,
                "Alice moved her plants to the Potsdam flat",
            )
            assert merged_away is not None, "the warm-up builds the observation that cites A"
            assert target is not None, "the warm-up builds the observation that cites B"
            await conn.execute(
                "UPDATE memory_units SET consolidated_at = NULL WHERE id = ANY($1::uuid[])",
                [archived_fact, surviving_fact],
            )

        # Only the measured run's calls count.
        requests.clear()
        result = await _run_job(memory, bank_id, request_context, answer)

        async with memory._pool.acquire() as conn:
            archived = await conn.fetchval("SELECT count(*) FROM invalidated_memory_units WHERE id = $1", archived_fact)
            merged_away_left = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", merged_away)
            target_left = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", target)
            rebuilt = await conn.fetchrow(
                "SELECT text, source_memory_ids FROM memory_units "
                "WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
                bank_id,
                ["user:alice"],
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", surviving_fact)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert archived == 1, "the judged fact is archived"
    assert merged_away_left == 0, "the response's merge deletes the observation it names"
    assert target_left == 0, "the invalidation sweep deletes the fold-widened target"
    assert len(requests) == 4, (
        "the fold-widened target's requeue of B is fetched again after the final pass; reading "
        "the requeue from before the fold left B out of the exclusion set, the stamp cancelled "
        "it, and the merged-away observation never came back"
    )
    assert "Alice moved her plants to the Potsdam flat" in requests[2], "the next round consolidates the survivor B"
    assert rebuilt is not None, "the next round rebuilds the observation from the surviving co-source"
    assert rebuilt["text"] == "Alice moved her plants to the Potsdam flat", "the rebuild carries B's knowledge"
    assert list(rebuilt["source_memory_ids"]) == [surviving_fact], "the rebuilt observation stands on B alone"
    assert stamped is not None, "B is stamped only after the round that rebuilt it"


# C7 across passes through the batch-delete loop: a deleted row must not be over-reported


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_batch_delete_co_source_is_not_over_reported(memory: MemoryEngine, request_context):
    """C7 across passes: the row this response deletes must not add its co-source to the requeue.

    The first scope's response deletes the observation citing A and B and judges A wrong. The
    DELETE runs before the sweep-time read, so the read never sees the row: B survives the
    sweep and belongs in the final pass's stamp. Reporting the requeue from the pre-read (taken
    before the delete loop) named B as a requeued co-source, left B out of the final stamp, and
    spent a whole extra consolidation round on a survivor whose observation the model had just
    retired.
    """
    bank_id = f"inval-delete-overreport-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    archived_fact: uuid.UUID | None = None
    surviving_fact: uuid.UUID | None = None
    observation: uuid.UUID | None = None
    requests: list[str] = []
    warm_replies = [
        # One observation from B, built by the product so it carries the embedding a delete
        # target must have to be recalled. The second pass only settles B.
        lambda: _FactBatchResponse(
            facts=[
                {
                    "fact_id": str(surviving_fact),
                    "kind": "durable",
                    "about": "Alice's home",
                    "adds": "yes",
                    "new": ["Alice lives in Potsdam"],
                }
            ]
        ),
        lambda: _FactBatchResponse(facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}]),
    ]

    def warm_answer(messages, scope):
        return warm_replies.pop(0)() if scope == "consolidation" and warm_replies else _FactBatchResponse()

    replies = [
        # Round 1, first scope: the response retires the observation and judges A wrong. The row
        # is gone before the sweep-time read, so B is never a surviving co-source of that sweep.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}],
            deletes=[
                {
                    "observation_id": str(observation),
                    "reason": "the observation is retired, not replaced",
                }
            ],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 1, final scope: A's repeated judgment skips at the liveness read; this pass is
        # where B's stamp lands, so a spurious requeue would leave B pending for another round.
        lambda: _FactBatchResponse(
            facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}],
            invalidations=[{"fact_id": str(archived_fact), "reason": "contradicted by a later fact"}],
        ),
        # Round 2 exists only when the pre-read over-reported B; these settle it if it runs.
        lambda: _FactBatchResponse(facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}]),
        lambda: _FactBatchResponse(facts=[{"fact_id": str(surviving_fact), "kind": "momentary"}]),
    ]

    def answer(messages, scope):
        if scope == "consolidation":
            requests.append("\n".join(str(message.get("content")) for message in messages))
        return replies.pop(0)() if replies else _FactBatchResponse()

    try:
        async with memory._pool.acquire() as conn:
            # Same tags and the per_tag scope spec on both facts: one sub-batch, one LLM call per
            # tag, and the user:alice pass is the first of the two.
            surviving_fact = await _insert_fact(
                conn, bank_id, "Alice lives in Potsdam", ["user:alice", "session:s1"], "per_tag"
            )

        warm = await _run_job(memory, bank_id, request_context, warm_answer)
        assert warm["status"] == "completed"

        async with memory._pool.acquire() as conn:
            observation = await conn.fetchval(
                "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation' "
                "AND tags @> $2::varchar[]",
                bank_id,
                ["user:alice"],
            )
            assert observation is not None, "the warm-up builds the observation the delete names"
            archived_fact = await _insert_fact(
                conn, bank_id, "Alice lives in Berlin", ["user:alice", "session:s1"], "per_tag"
            )
            # Both facts cite the row before the batch starts: a read taken before the delete
            # loop returns it and names B as a surviving co-source, which is the over-report.
            await conn.execute(
                "UPDATE memory_units SET source_memory_ids = ARRAY[$1::uuid, $2::uuid] WHERE id = $3",
                archived_fact,
                surviving_fact,
                observation,
            )
            await conn.execute("UPDATE memory_units SET consolidated_at = NULL WHERE id = $1", surviving_fact)

        # Only the measured run's calls count.
        requests.clear()
        result = await _run_job(memory, bank_id, request_context, answer)

        async with memory._pool.acquire() as conn:
            archived = await conn.fetchval("SELECT count(*) FROM invalidated_memory_units WHERE id = $1", archived_fact)
            observation_left = await conn.fetchval("SELECT count(*) FROM memory_units WHERE id = $1", observation)
            observations = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
                bank_id,
            )
            stamped = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", surviving_fact)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    assert result["status"] == "completed"
    assert archived == 1, "the judged fact is archived"
    assert observation_left == 0, "the response's batch DELETE removes the row before the sweep reads"
    assert observations == 0, "the retired observation is not rebuilt: no round is spent on it"
    assert len(requests) == 2, (
        "a deleted row is not a sweep victim: B survives the sweep and is stamped by the final "
        "pass; reporting it from the pre-read left it out of the stamp and added a round"
    )
    assert stamped is not None, "B is stamped by the final pass, with no requeue to cancel it"
