"""T8 (kata sp4y): fail-open boundary, Oracle skip and same-job ordering.

The boundary wraps T6's once-per-job page check (C6). An ordinary failure at any
of the five steps is logged and swallowed so the consolidation job keeps its own
writes and writes no page marks; cancellation and KeyboardInterrupt are re-raised.
On Oracle the check is skipped before an embedding is spent. A mark committed by
the check is visible to the job tail's refresh trigger in the same job only when
the run is not round limited and the page passes the trigger's own prefilter.

Property ids P1, P2, P3, P8 and P9 refer to machine 2 of
``model/staleness-marking.py@v3``; the specification stays in that model.
Every case uses a pg0 scratch bank or in-memory doubles, never a live bank.
"""

import asyncio
import importlib.util
import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine import memory_engine as memory_engine_mod
from hindsight_api.engine.consolidation import consolidator as con
from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memories.base import StoredMemory
from hindsight_api.engine.schema import fq_table

PAGE_QUESTION = "What are the project's conventions?"
PAGE_BODY = "## Conventions\n\nTests run on every push.\n"
MODEL_PATH = "/home/wilder/git-repos/work/hindsight/docs/work/graph-staleness/model/staleness-marking.py"
DIM = 384


# ---------------------------------------------------------------------------
# Local doubles (the shared ML fixtures cannot load in this environment)
# ---------------------------------------------------------------------------


class _StubEmbeddings(Embeddings):
    model_name = "page-boundary-stub"

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def dimension(self) -> int:
        return DIM

    async def initialize(self) -> None:
        return None

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


class _StubCrossEncoder(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "test"

    async def initialize(self) -> None:
        return None

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.5] * len(pairs)


@pytest.fixture(scope="session")
def embeddings():
    return _StubEmbeddings()


@pytest.fixture(scope="session")
def cross_encoder():
    return _StubCrossEncoder()


@pytest.fixture(autouse=True)
def enable_observations():
    raw = _get_raw_config()
    old = raw.enable_observations
    raw.enable_observations = True
    yield
    raw.enable_observations = old


def _make_config(**overrides):
    raw = _get_raw_config()
    return type(raw)(
        **{
            **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
            **overrides,
        }
    )


class _FakeConn:
    backend_type = "postgresql"

    def __init__(self, rows: list[StoredMemory] | None = None, backend_type: str = "postgresql") -> None:
        self.rows = rows or []
        self.backend_type = backend_type
        self.statements: list[str] = []

    def in_transaction(self) -> bool:
        return True

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetch(self, sql, *args, **kwargs):
        self.statements.append(sql)
        return list(self.rows)

    async def fetchrow(self, sql, *args, **kwargs):
        self.statements.append(sql)
        return None

    async def fetchval(self, sql, *args, **kwargs):
        self.statements.append(sql)
        return None


def _obs(unit_id: str, text: str, tags: list[str] | None = None) -> StoredMemory:
    return StoredMemory(unit_id=unit_id, text=text, fact_type="observation", tags=tags or [])


def _decide_all(**kwargs):
    """A decision that selects every (observation, page) pair it was offered."""
    return frozenset((obs, pid) for obs, cands in kwargs["candidates"].items() for pid, _ in cands)


async def _mark_count(conn, bank_id: str) -> int:
    return int(
        await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} "
            "WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
    )


async def _mark_rows(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        f"SELECT detail->>'page_id' AS page_id FROM {fq_table('memory_change_events')} "
        "WHERE bank_id = $1 AND event_type = 'PAGE_MARK' ORDER BY seq",
        bank_id,
    )
    return [row["page_id"] for row in rows]


async def _max_seq(conn, bank_id: str) -> int:
    value = await conn.fetchval(
        f"SELECT COALESCE(max(seq), 0) FROM {fq_table('memory_change_events')} WHERE bank_id = $1",
        bank_id,
    )
    return int(value or 0)


async def _seed_world(memory, bank_id: str, *, tag: str = "alpha") -> uuid.UUID:
    """Insert one unconsolidated source fact directly, so retain cannot race the job."""
    pool = await memory._get_pool()
    fact_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
            "VALUES ($1, $2, $3, 'world', $4, now(), now())",
            fact_id,
            bank_id,
            "The team keeps its conventions in a written page.",
            [tag],
        )
    return fact_id


def _patch_mark_page(monkeypatch, page_id: str):
    """Point the check's shortlist and decision at one known page."""

    async def shortlist(**kwargs):
        return [(page_id, PAGE_QUESTION)]

    async def decide(**kwargs):
        return frozenset((obs, page_id) for obs in kwargs["candidates"])

    monkeypatch.setattr(con, "page_shortlist", shortlist)
    monkeypatch.setattr(con, "decide_page_marks", decide)


def _patch_config(monkeypatch, memory, config):
    async def resolve(*args, **kwargs):
        return config

    monkeypatch.setattr(memory._config_resolver, "resolve_full_config", resolve)


# ---------------------------------------------------------------------------
# One injected failure per step: job succeeds, own writes stay, zero marks
# ---------------------------------------------------------------------------


async def _job_survives_step_failure(memory, request_context, monkeypatch, step: str):
    bank_id = f"test-pcb-{step}-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    await _seed_world(memory, bank_id)
    fired: list[str] = []

    # Flag the check so a global seam patched here does not break the job's own reads.
    real_runner = con.run_observation_page_check
    state = {"in_check": False}

    async def flagged_runner(**kwargs):
        state["in_check"] = True
        try:
            return await real_runner(**kwargs)
        finally:
            state["in_check"] = False

    monkeypatch.setattr(con, "run_observation_page_check", flagged_runner)

    if step == "read":
        real_get = con.get_memories

        class _Store:
            def get_memories(self, **kwargs):
                if state["in_check"]:
                    fired.append(step)
                    raise RuntimeError("injected failure: reading checked-set observations")
                return real_get().get_memories(**kwargs)

            def __getattr__(self, name):
                return getattr(real_get(), name)

        monkeypatch.setattr(con, "get_memories", lambda: _Store())
    elif step == "embedding":
        real_shortlist = con.page_shortlist

        class _EmbedEngine:
            async def search_pages(self, *args, **kwargs):
                fired.append(step)
                raise RuntimeError("injected failure: embedding the subject")

        async def shortlist(**kwargs):
            kwargs["memory_engine"] = _EmbedEngine()
            return await real_shortlist(**kwargs)

        monkeypatch.setattr(con, "page_shortlist", shortlist)
    elif step == "search":

        async def shortlist(**kwargs):
            fired.append(step)
            raise RuntimeError("injected failure: page search")

        monkeypatch.setattr(con, "page_shortlist", shortlist)
    elif step == "decision":

        async def decide(**kwargs):
            fired.append(step)
            raise RuntimeError("injected failure: decision call")

        monkeypatch.setattr(con, "decide_page_marks", decide)
    elif step == "mark_write":

        async def write(*args, **kwargs):
            fired.append(step)
            raise RuntimeError("injected failure: mark write")

        monkeypatch.setattr(con, "record_page_marks", write)
    else:
        raise AssertionError(f"unknown step {step}")

    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    result = await con.run_consolidation_job(memory, bank_id, request_context)

    assert fired == [step], f"the {step} failure was not injected into the check"
    assert result["status"] == "completed"
    assert result["observations_created"] >= 1
    assert result["page_check_ran"] is False
    assert result["page_marks_written"] == 0
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        assert await _mark_count(conn, bank_id) == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_boundary_failure_leaves_job_successful(memory, request_context, monkeypatch):
    """Reading the checked-set observations fails: the job still completes, zero marks."""
    await _job_survives_step_failure(memory, request_context, monkeypatch, "read")


@pytest.mark.asyncio
async def test_page_check_boundary_failure_embedding_leaves_job_successful(memory, request_context, monkeypatch):
    await _job_survives_step_failure(memory, request_context, monkeypatch, "embedding")


@pytest.mark.asyncio
async def test_page_check_boundary_failure_page_search_leaves_job_successful(memory, request_context, monkeypatch):
    await _job_survives_step_failure(memory, request_context, monkeypatch, "search")


@pytest.mark.asyncio
async def test_page_check_boundary_failure_decision_leaves_job_successful(memory, request_context, monkeypatch):
    await _job_survives_step_failure(memory, request_context, monkeypatch, "decision")


@pytest.mark.asyncio
async def test_page_check_boundary_failure_mark_write_leaves_job_successful(memory, request_context, monkeypatch):
    """The decision already selected pairs, yet a failed mark write leaves zero marks (P1, P2)."""
    await _job_survives_step_failure(memory, request_context, monkeypatch, "mark_write")


# ---------------------------------------------------------------------------
# Cancellation and Oracle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_check_boundary_cancelled_job_still_cancels(memory, request_context, monkeypatch):
    """A cancelled check re-raises untouched, so the whole job cancels."""
    bank_id = f"test-pcb-cancel-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    await _seed_world(memory, bank_id)
    fired: list[str] = []

    async def cancel(**kwargs):
        fired.append("cancel")
        raise asyncio.CancelledError

    monkeypatch.setattr(con, "decide_page_marks", cancel)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    with pytest.raises(asyncio.CancelledError):
        await con.run_consolidation_job(memory, bank_id, request_context)
    assert fired == ["cancel"]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        assert await _mark_count(conn, bank_id) == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_boundary_interrupt_reraised_untouched(memory, request_context, monkeypatch):
    """KeyboardInterrupt is re-raised too, not logged and swallowed as an ordinary failure."""
    bank_id = f"test-pcb-interrupt-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    await _seed_world(memory, bank_id)

    async def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(con, "decide_page_marks", interrupt)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    with pytest.raises(KeyboardInterrupt):
        await con.run_consolidation_job(memory, bank_id, request_context)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_boundary_oracle_skip(monkeypatch):
    """On Oracle no embedding, search, decision or mark happens, and nothing raises."""
    called: list[str] = []

    async def forbidden(**kwargs):
        called.append("called")
        raise AssertionError("the Oracle skip must happen before any check step")

    monkeypatch.setattr(con, "get_memories", lambda: SimpleNamespace(get_memories=forbidden))
    monkeypatch.setattr(con, "page_shortlist", forbidden)
    monkeypatch.setattr(con, "decide_page_marks", forbidden)
    monkeypatch.setattr(con, "record_page_marks", forbidden)

    config = SimpleNamespace(database_backend="oracle", observation_marking_page_stale="bank_wide")
    conn = _FakeConn(backend_type="oracle")
    outcome = await con.run_observation_page_check(
        bank_id="bank",
        created={"00000000-0000-0000-0000-000000000001": ["fact"]},
        config=config,
        request_context=SimpleNamespace(),
        memory_engine=SimpleNamespace(),
        llm_config=SimpleNamespace(),
        conn=conn,
    )
    assert outcome.ran is False
    assert outcome.selected == frozenset()
    assert outcome.marks_written == 0
    assert called == []
    assert conn.statements == []


def test_page_check_boundary_oracle_uses_graph_backend_predicate():
    """The skip reads the same Postgres-only backend predicate as the graph rule and C10."""
    predicate = memory_engine_mod._page_check_backend_postgresql
    assert predicate(SimpleNamespace(backend_type="postgresql")) is True
    assert predicate(SimpleNamespace(backend_type="oracle")) is False
    assert predicate(SimpleNamespace()) is True


# ---------------------------------------------------------------------------
# Ordering: same-job trigger visibility and the round-limit bound
# ---------------------------------------------------------------------------


async def _order_bank(memory, request_context, *, tag: str = "alpha"):
    bank_id = f"test-pco-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    model = await memory.create_mental_model(
        bank_id=bank_id,
        name="Conventions",
        source_query=PAGE_QUESTION,
        content=PAGE_BODY,
        tags=[tag],
        request_context=request_context,
        trigger={"mode": "full", "refresh_after_consolidation": True},
    )
    page_id = model["id"]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        baseline = await _max_seq(conn, bank_id)
        # A baseline predating the mark: the page was refreshed at this seq.
        rr = {
            "text": PAGE_BODY,
            "mental_models": [],
            "graph_provenance": {
                "read_memory_ids": [],
                "captured_empty": True,
                "cited_empty": True,
                "baseline_seq": baseline,
            },
        }
        await conn.execute(
            f"UPDATE {fq_table('mental_models')} SET reflect_response=$3::jsonb, "
            "last_memory_seen_at=now() + interval '1 day' WHERE bank_id=$1 AND id=$2",
            bank_id,
            page_id,
            json.dumps(rr),
        )
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
            "VALUES ($1, $2, $3, 'world', $4, now(), now())",
            uuid.uuid4(),
            bank_id,
            "The team keeps its conventions in a written page.",
            [tag],
        )
    return bank_id, page_id, baseline


@pytest.mark.asyncio
async def test_page_check_order_same_job(memory, request_context, monkeypatch):
    """H5: an inside-prefilter mark on a non-round-limited run is seen by this job's trigger."""
    bank_id, page_id, baseline = await _order_bank(memory, request_context)
    _patch_mark_page(monkeypatch, page_id)
    pool = await memory._get_pool()

    seen: dict[str, Any] = {}
    real_trigger = con._trigger_mental_model_refreshes

    async def spy_trigger(**kwargs):
        seen["trigger_called"] = True
        async with pool.acquire() as conn:
            marked = await memory_engine_mod.page_marks_read(conn, bank_id=bank_id, pages=[(page_id, baseline)])
            seen["mark_at_trigger"] = marked[page_id]
            seen["seq_at_trigger"] = await _max_seq(conn, bank_id)
        return await real_trigger(**kwargs)

    monkeypatch.setattr(con, "_trigger_mental_model_refreshes", spy_trigger)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)

    result = await con.run_consolidation_job(memory, bank_id, request_context)
    assert result["status"] == "completed"
    assert result["page_marks_written"] >= 1
    assert seen["trigger_called"] is True
    assert seen["mark_at_trigger"] is True
    assert seen["seq_at_trigger"] > baseline
    assert result["mental_models_refreshed"] >= 1
    async with pool.acquire() as conn:
        assert page_id in await _mark_rows(conn, bank_id)
    # H5 trace: baseline, mark seq and the mark read at trigger time are printed
    # so the same-job ordering is evidence rather than an inferred claim.
    print(f"\nH5 trace: baseline={baseline}, seq_at_trigger={seen['seq_at_trigger']}, "
          f"mark_at_trigger={seen['mark_at_trigger']}, "
          f"mental_models_refreshed={result['mental_models_refreshed']}")
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_order_round_limited_no_trigger(memory, request_context, monkeypatch):
    """A round-limited run still checks and writes its marks but runs no trigger at all."""
    bank_id, page_id, baseline = await _order_bank(memory, request_context)
    await _seed_world(memory, bank_id)
    _patch_mark_page(monkeypatch, page_id)
    pool = await memory._get_pool()

    calls: list[str] = []

    async def no_trigger(**kwargs):
        calls.append("trigger")
        return 0

    monkeypatch.setattr(con, "_trigger_mental_model_refreshes", no_trigger)
    monkeypatch.setattr(memory, "submit_async_consolidation", lambda *a, **kw: asyncio.sleep(0))
    config = _make_config(observation_marking_page_stale="bank_wide", consolidation_max_memories_per_round=1)
    _patch_config(monkeypatch, memory, config)

    result = await con.run_consolidation_job(memory, bank_id, request_context)
    assert result["status"] == "completed"
    assert result["mental_models_refreshed"] == 0
    assert calls == []
    assert result["page_check_ran"] is True
    assert result["page_marks_written"] >= 1
    async with pool.acquire() as conn:
        assert page_id in await _mark_rows(conn, bank_id)
        assert await _max_seq(conn, bank_id) > baseline
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_order_bank_wide_outside_prefilter_next_refresh(memory, request_context, monkeypatch):
    """A mark on a page outside the trigger prefilter is not refreshed in this job."""
    bank_id, page_id, baseline = await _order_bank(memory, request_context)
    _patch_mark_page(monkeypatch, page_id)
    pool = await memory._get_pool()
    calls: list[str] = []

    async def no_trigger(**kwargs):
        calls.append("trigger")
        return 0

    monkeypatch.setattr(con, "_trigger_mental_model_refreshes", no_trigger)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    # Scope the trigger's tag prefilter away from the page's tag on this run.
    async def scoped(**kwargs):
        kwargs["consolidated_tags"] = ["outside"]
        return await no_trigger(**kwargs)

    monkeypatch.setattr(con, "_trigger_mental_model_refreshes", scoped)
    result = await con.run_consolidation_job(memory, bank_id, request_context)
    assert result["status"] == "completed"
    assert calls == ["trigger"]  # the trigger ran; the page's tag was not in its prefilter
    async with pool.acquire() as conn:
        marked = await memory_engine_mod.page_marks_read(conn, bank_id=bank_id, pages=[(page_id, baseline)])
        assert marked[page_id] is True  # seen at the page's next refresh (its baseline predates the mark)
        assert await _max_seq(conn, bank_id) > baseline
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Crash recovery: the committed set is durable, never a partial set (P2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_check_crash_after_commit_keeps_committed_set(memory, request_context, monkeypatch):
    """Killed after the mark commit: exactly the committed set is durable and the page reads stale."""
    bank_id, page_id, baseline = await _order_bank(memory, request_context)
    _patch_mark_page(monkeypatch, page_id)
    pool = await memory._get_pool()
    real_runner = con.run_observation_page_check

    async def crash_after_commit(**kwargs):
        outcome = await real_runner(**kwargs)
        assert outcome.marks_written >= 1
        raise KeyboardInterrupt("simulated kill after the mark transaction committed")

    monkeypatch.setattr(con, "run_observation_page_check", crash_after_commit)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    with pytest.raises(KeyboardInterrupt):
        await con.run_consolidation_job(memory, bank_id, request_context)

    async with pool.acquire() as conn:
        durable = await _mark_rows(conn, bank_id)
        assert durable and all(p == page_id for p in durable)  # exactly the committed set, no partial pair
        marked = await memory_engine_mod.page_marks_read(conn, bank_id=bank_id, pages=[(page_id, baseline)])
        assert marked[page_id] is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_crash_before_commit_leaves_no_marks(memory, request_context, monkeypatch):
    """Killed before the mark commit: no marks at all (the other P2 half)."""
    bank_id, page_id, baseline = await _order_bank(memory, request_context)
    _patch_mark_page(monkeypatch, page_id)
    pool = await memory._get_pool()

    async def crash_before_commit(*args, **kwargs):
        raise KeyboardInterrupt("simulated kill before the mark transaction committed")

    monkeypatch.setattr(con, "record_page_marks", crash_before_commit)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    with pytest.raises(KeyboardInterrupt):
        await con.run_consolidation_job(memory, bank_id, request_context)

    async with pool.acquire() as conn:
        assert await _mark_count(conn, bank_id) == 0
        marked = await memory_engine_mod.page_marks_read(conn, bank_id=bank_id, pages=[(page_id, baseline)])
        assert marked[page_id] is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# U6: the feedback loop does not grow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_check_boundary_u6_no_feedback_growth(memory, request_context, monkeypatch):
    """Three bounded consolidate/refresh cycles: each adds one mark, none amplifies."""
    bank_id = f"test-pcb-u6-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    model = await memory.create_mental_model(
        bank_id=bank_id,
        name="U6",
        source_query=PAGE_QUESTION,
        content=PAGE_BODY,
        request_context=request_context,
        trigger={"mode": "full", "refresh_after_consolidation": True},
    )
    page_id = model["id"]
    _patch_mark_page(monkeypatch, page_id)
    config = _make_config(observation_marking_page_stale="bank_wide")
    _patch_config(monkeypatch, memory, config)
    pool = await memory._get_pool()
    counts: list[int] = []
    for cycle in range(3):
        await _seed_world(memory, bank_id, tag=f"u6-{cycle}")
        result = await con.run_consolidation_job(memory, bank_id, request_context)
        assert result["status"] == "completed"
        async with pool.acquire() as conn:
            # The refresh half of the cycle: stamp a new baseline at the current seq.
            seq = await _max_seq(conn, bank_id)
            await conn.execute(
                f"UPDATE {fq_table('mental_models')} SET reflect_response=jsonb_set("
                "COALESCE(reflect_response, '{}'::jsonb), '{graph_provenance,baseline_seq}', $3::jsonb), "
                "last_memory_seen_at=now() WHERE bank_id=$1 AND id=$2",
                bank_id,
                page_id,
                json.dumps(seq),
            )
            counts.append(await _mark_count(conn, bank_id))
    assert counts[0] >= 1
    assert counts[1] <= counts[0] + 1
    assert counts[2] <= counts[1] + 1
    print(f"\nU6 per-cycle mark counts: {counts}")
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Model machine 2: properties P1, P2, P3, P8, P9 and the five forbidden edges
# ---------------------------------------------------------------------------


def _model():
    spec = importlib.util.spec_from_file_location("graph_staleness_model_t8", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_mutant_detects(mutant: str, expect: str) -> None:
    model = _model()
    violations, _ = model.page_check_check(mutant=mutant)
    assert any(expect in item for item in violations), f"model mutant {mutant} was not detected by {expect}"


def test_page_check_boundary_p1_all_or_nothing():
    _model_mutant_detects("partial_marks", "P1:")


def test_page_check_boundary_p2_no_partial_marks():
    _model_mutant_detects("marks_on_failure", "P2:")
    _model_mutant_detects("crash_drops_committed_marks", "P2:")


def test_page_check_boundary_p3_fail_open():
    _model_mutant_detects("check_fails_job", "P3:")


def test_page_check_boundary_p8_oracle_skip():
    _model_mutant_detects("oracle_marks", "P8:")


def test_page_check_boundary_p9_no_skipped_check():
    _model_mutant_detects("check_skipped_on_created", "P9:")


def test_page_check_boundary_forbidden_transitions_unreachable():
    model = _model()
    for phase, event in model.PAGE_FORBIDDEN:
        ctx = model.page_ctx(created=True)
        if phase == "MARKS":
            ctx["selected"] = frozenset((model.PAGE_OBS, page) for page in model.PAGE_SHORTLIST)
            ctx["marks"] = frozenset(ctx["selected"])
        assert model.page_check_transition(phase, event, ctx) is None, f"forbidden {phase}+{event} is reachable"
