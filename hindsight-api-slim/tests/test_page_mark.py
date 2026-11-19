"""T7 (kata 2pz5): the durable page mark and its read in both staleness modes.

The mark is a ``PAGE_MARK`` row in ``memory_change_events`` whose subject is a
checked-set observation and whose detail names one page. The read sits above
the ``staleness_check`` branch, so a tagscope page and a graph page both answer
it. Model property ids in test names refer to machine 2 of
``model/staleness-marking.py@v3``; the specification stays in that model.

Every case runs on the pg0 scratch bank through the ``memory`` fixture. No live
bank and no migration is used.
"""

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine import memory_engine as memory_engine_mod
from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.pg import consolidation as pg_con
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table, page_marks_read
from hindsight_api.engine.schema import fq_store_table

PAGE_CONTENT = "## Conventions\n\nTests run on every push.\n"
DIM = 384


class _StubEmbeddings(Embeddings):
    """Deterministic vectors at the configured dimension, no model download.

    These cases never rank or embed for meaning; the bank only needs a vector
    of the right width when a page row is written. The shared local-model
    fixture cannot load in this environment (tokenizers/transformers version
    mismatch), so the file supplies its own no-download doubles.
    """

    model_name = "page-mark-stub"

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


@pytest.fixture(scope="session")
def embeddings():
    return _StubEmbeddings()


@pytest.fixture(scope="session")
def cross_encoder():
    return _StubCrossEncoder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact(fact_id: str, text: str = "a fact") -> dict[str, Any]:
    return {"id": fact_id, "text": text, "type": "observation", "context": None}


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_observation(conn, bank_id: str, text: str) -> uuid.UUID:
    """Seed an observation directly, so it can be a surviving twin this run did not create."""
    obs_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, tags) "
        "VALUES ($1, $2, $3, 'observation', $4)",
        obs_id,
        bank_id,
        text,
        [],
    )
    return obs_id


async def _max_seq(conn, bank_id: str) -> int:
    value = await conn.fetchval(
        f"SELECT COALESCE(max(seq), 0) FROM {fq_table('memory_change_events')} WHERE bank_id = $1",
        bank_id,
    )
    return int(value or 0)


async def _write_marks(conn, bank_id: str, marks: frozenset[tuple[str, str]]) -> int:
    """One mark transaction, the way C6 will call it."""
    async with conn.transaction():
        return await pg_con.record_page_marks(conn=conn, bank_id=bank_id, marks=marks)


async def _create_page(
    memory: MemoryEngine,
    bank_id: str,
    request_context: RequestContext,
    *,
    staleness_check: str = "tagscope",
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
            "graph_stale": "cited_memories",
        },
    )
    return model["id"]


async def _stamp_baseline(conn, bank_id: str, mm_id: str, baseline: int | None) -> None:
    """Write the refresh stamp, including a baseline or deliberately without one."""
    prov: dict[str, Any] = {
        "read_memory_ids": [],
        "captured_empty": True,
        "cited_empty": True,
    }
    if baseline is not None:
        prov["baseline_seq"] = baseline
    reflect_response = {"text": PAGE_CONTENT, "mental_models": [], "graph_provenance": prov}
    await conn.execute(
        f"UPDATE {fq_store_table('mental_models')} SET reflect_response = $3::jsonb, "
        f"last_memory_seen_at = now(), last_refreshed_at = now() WHERE bank_id = $1 AND id = $2",
        bank_id,
        mm_id,
        json.dumps(reflect_response),
    )


async def _is_stale(memory: MemoryEngine, conn, bank_id: str, mm_id: str) -> bool:
    row = await conn.fetchrow(
        f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at, reflect_response "
        f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
        bank_id,
        mm_id,
    )
    assert row is not None
    return await memory.compute_mental_model_is_stale(conn, bank_id, row)


class _CountingConn:
    """A Postgres-shaped connection that records statements and returns preset rows."""

    backend_type = "postgresql"

    def __init__(self, rows: list[dict] | None = None, in_transaction: bool = True) -> None:
        self.statements: list[str] = []
        self.rows = rows or []
        self._in_transaction = in_transaction

    def in_transaction(self) -> bool:
        return self._in_transaction

    async def execute(self, sql, *args, **kwargs) -> str:
        self.statements.append(sql)
        return "OK"

    async def fetchval(self, sql, *args, **kwargs):
        self.statements.append(sql)
        return None

    async def fetch(self, sql, *args, **kwargs):
        self.statements.append(sql)
        return list(self.rows)

    async def fetchrow(self, sql, *args, **kwargs):
        self.statements.append(sql)
        return None


# ---------------------------------------------------------------------------
# C9, P4: the writer, its subject and its identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_mark_p4_subject_is_checked_observation_and_twin(memory, request_context):
    """P4: every mark names a checked-set observation, which may be a
    surviving twin this run did not create. The page id lives in detail, never
    in subject_id."""
    bank_id = f"test-pm-p4-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        twin = await _insert_observation(conn, bank_id, "A twin from an earlier run.")
        page_id = "page-about-twin"
        inserted = await _write_marks(conn, bank_id, frozenset({(str(twin), page_id)}))
        assert inserted == 1
        row = await conn.fetchrow(
            f"SELECT subject_id, event_type, detail FROM {fq_table('memory_change_events')} "
            f"WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
        assert row is not None
        assert row["subject_id"] == twin
        detail = row["detail"] if isinstance(row["detail"], dict) else json.loads(row["detail"])
        assert detail == {"page_id": page_id}
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_mark_write_page_id_subject_raises():
    """A page id is text; subject_id is cast through uuid.UUID, so naming a
    page as the subject fails at the writer instead of storing a wrong kind."""
    conn = _CountingConn()
    with pytest.raises((ValueError, AttributeError)):
        await pg_con.record_page_marks(
            conn=conn,
            bank_id="bank",
            marks=frozenset({("plain-text-page-id", "page-1")}),
        )


@pytest.mark.asyncio
async def test_page_mark_write_one_row_per_pair_under_bank_lock():
    """C9: one row per (observation, page) pair, inside the existing per-bank
    advisory lock and an open transaction, with the inserted count returned."""
    obs = uuid.uuid4()
    conn = _CountingConn()
    inserted = await pg_con.record_page_marks(
        conn=conn,
        bank_id="bank",
        marks=frozenset({(str(obs), "p1"), (str(obs), "p2")}),
    )
    assert inserted == 2
    assert sum("pg_advisory_xact_lock" in s for s in conn.statements) >= 1
    assert sum("memory_change_events" in s and "INSERT" in s for s in conn.statements) == 2
    assert all("ON CONFLICT (bank_id, event_id) DO NOTHING" in s for s in conn.statements if "INSERT" in s)


@pytest.mark.asyncio
async def test_page_mark_write_two_pages_of_one_observation_are_two_rows(memory, request_context):
    """P5: the stable identity is the (observation, page) pair, so two pages
    named by one observation are two distinct marks, not one collapsed row."""
    bank_id = f"test-pm-pairs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Pair subject.")
        marks = frozenset({(str(obs), "page-a"), (str(obs), "page-b")})
        assert await _write_marks(conn, bank_id, marks) == 2
        rows = await conn.fetch(
            f"SELECT detail FROM {fq_table('memory_change_events')} "
            f"WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
        pages = {
            (row["detail"] if isinstance(row["detail"], dict) else json.loads(row["detail"]))["page_id"]
            for row in rows
        }
        assert pages == {"page-a", "page-b"}
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_mark_write_requires_open_transaction():
    """C9 errors: no transaction means the xact advisory lock would vanish."""
    conn = _CountingConn(in_transaction=False)
    with pytest.raises(RuntimeError, match="requires an open transaction"):
        await pg_con.record_page_marks(conn=conn, bank_id="bank", marks=frozenset())
    assert conn.statements == []


# ---------------------------------------------------------------------------
# C9, P1 and P5: all-or-nothing and idempotent retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_mark_write_abort_leaves_zero_rows(memory, request_context):
    """P1: an abort inside the mark transaction commits no marks."""
    bank_id = f"test-pm-abort-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "P1 subject.")
        with pytest.raises(RuntimeError, match="injected abort"):
            async with conn.transaction():
                await pg_con.record_page_marks(conn=conn, bank_id=bank_id, marks=frozenset({(str(obs), "p1")}))
                raise RuntimeError("injected abort")
        count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} "
            f"WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
        assert count == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_mark_p5_retry_committed_is_noop_then_abort_inserts(memory, request_context):
    """P5: a retry after a committed attempt inserts nothing and reports 0;
    a retry after an aborted attempt has no durable row to conflict with and
    inserts, reporting 1. The identity is the (observation, page) pair, not
    the ON CONFLICT clause."""
    bank_id = f"test-pm-p5-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "P5 subject.")
        marks = frozenset({(str(obs), "p1")})

        assert await _write_marks(conn, bank_id, marks) == 1
        assert await _write_marks(conn, bank_id, marks) == 0

        obs2 = await _insert_observation(conn, bank_id, "P5 abort subject.")
        marks2 = frozenset({(str(obs2), "p2")})
        with pytest.raises(RuntimeError, match="injected abort"):
            async with conn.transaction():
                await pg_con.record_page_marks(conn=conn, bank_id=bank_id, marks=marks2)
                raise RuntimeError("injected abort")
        assert await _write_marks(conn, bank_id, marks2) == 1

        rows = await conn.fetch(
            f"SELECT event_type, detail FROM {fq_table('memory_change_events')} WHERE bank_id = $1",
            bank_id,
        )
        assert len(rows) == 2
        assert all(row["event_type"] == "PAGE_MARK" for row in rows)
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# C10, P6: the read in both staleness modes, baseline ordering and no baseline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_mark_p6_mark_newer_than_baseline_reads_stale_in_both_modes(memory, request_context):
    """P6 and R13: a mark whose seq is newer than the page baseline makes the
    page stale in tagscope mode and in graph mode. Before the mark, both read
    fresh."""
    for staleness_check in ("tagscope", "graph"):
        bank_id = f"test-pm-p6-{staleness_check}-{uuid.uuid4().hex[:6]}"
        await _ensure_bank(memory, bank_id, request_context)
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            obs = await _insert_observation(conn, bank_id, "Marked subject.")
            mm_id = await _create_page(memory, bank_id, request_context, staleness_check=staleness_check)
            baseline = await _max_seq(conn, bank_id)
            await _stamp_baseline(conn, bank_id, mm_id, baseline)
            assert await _is_stale(memory, conn, bank_id, mm_id) is False
            assert await _write_marks(conn, bank_id, frozenset({(str(obs), mm_id)})) == 1
            assert await _is_stale(memory, conn, bank_id, mm_id) is True
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_mark_read_newer_baseline_is_fresh(memory, request_context):
    """A mark already below the page's baseline does not make it stale."""
    bank_id = f"test-pm-newer-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Already-ordered subject.")
        assert await _write_marks(conn, bank_id, frozenset({(str(obs), "page-a")})) == 1
        new_baseline = await _max_seq(conn, bank_id)
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="tagscope")
        await _stamp_baseline(conn, bank_id, mm_id, new_baseline)
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_mark_read_no_baseline_is_fresh(memory, request_context):
    """The accepted ordering limitation: a mark written before a page's first
    refresh cannot be ordered against it, so that page reads fresh until its
    refresh stamps a baseline. Named so it cannot be read as a defect."""
    bank_id = f"test-pm-nobase-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Unordered subject.")
        await _write_marks(conn, bank_id, frozenset({(str(obs), "page-nobase")}))
        answers = await page_marks_read(conn, bank_id=bank_id, pages=[("page-nobase", None)])
        assert answers == {"page-nobase": False}
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="tagscope")
        await _stamp_baseline(conn, bank_id, mm_id, None)
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# C10, P7: clearing after refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_mark_p7_refresh_after_mark_reads_fresh(memory, request_context):
    """P7: refreshing a page after a mark commits stamps a newer baseline,
    and the page reads fresh again while the mark is still on disk."""
    bank_id = f"test-pm-p7-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "P7 subject.")
        mm_id = await _create_page(memory, bank_id, request_context, staleness_check="tagscope")
        baseline = await _max_seq(conn, bank_id)
        await _stamp_baseline(conn, bank_id, mm_id, baseline)
        assert await _write_marks(conn, bank_id, frozenset({(str(obs), mm_id)})) == 1
        assert await _is_stale(memory, conn, bank_id, mm_id) is True

        mark_count = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_change_events')} "
            f"WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
        assert mark_count == 1  # the mark is still on disk
        refreshed_baseline = await _max_seq(conn, bank_id)
        await _stamp_baseline(conn, bank_id, mm_id, refreshed_baseline)
        assert await _is_stale(memory, conn, bank_id, mm_id) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# C10: Oracle, statement counts, single/batch agreement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_mark_read_oracle_answers_fresh_without_a_statement(monkeypatch):
    """On Oracle change records do not exist, so the read issues no statement
    and answers False for every page."""
    monkeypatch.setattr(memory_engine_mod, "_is_oracle", lambda: True)
    conn = _CountingConn()
    answers = await page_marks_read(conn, bank_id="bank", pages=[("p1", 3), ("p2", None)])
    assert answers == {"p1": False, "p2": False}
    assert conn.statements == []


@pytest.mark.asyncio
async def test_page_mark_read_single_uses_one_statement():
    """The single path issues one statement for its one page."""
    conn = _CountingConn()
    answers = await page_marks_read(conn, bank_id="bank", pages=[("p1", 5)])
    assert answers == {"p1": False}
    assert len(conn.statements) == 1


@pytest.mark.asyncio
async def test_page_mark_read_batch_uses_one_statement():
    """The batch path issues one statement for the whole batch, never one per page."""
    pages = [(f"page-{i}", i) for i in range(12)]
    conn = _CountingConn()
    answers = await page_marks_read(conn, bank_id="bank", pages=pages)
    assert answers == {page: False for page, _ in pages}
    assert len(conn.statements) == 1


@pytest.mark.asyncio
async def test_page_mark_read_batch_agrees_with_single(memory, request_context):
    """The batch answer equals the single answer on the same pairs."""
    bank_id = f"test-pm-agree-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        obs = await _insert_observation(conn, bank_id, "Agreement subject.")
        await _write_marks(conn, bank_id, frozenset({(str(obs), "page-1"), (str(obs), "page-3")}))
        baseline = 0
        pages = [("page-1", baseline), ("page-2", baseline), ("page-3", baseline)]
        batch = await page_marks_read(conn, bank_id=bank_id, pages=pages)
        singles = {}
        for page, seq in pages:
            singles.update(await page_marks_read(conn, bank_id=bank_id, pages=[(page, seq)]))
        assert batch == singles
        assert batch == {"page-1": True, "page-2": False, "page-3": True}
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# U5: cost of the page-addressed read over a seeded 10k-row event table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_mark_read_u5_explain_cost(memory, request_context):
    """U5: EXPLAIN the exact C10 query over a seeded 10k-row event table on a
    scratch bank. Record the plan, the rows it reads and the one statement the
    read issues. No migration and no index on detail is added."""
    bank_id = f"test-pm-u5-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        seed_rows = [
            (f"seed-{i}", bank_id, f"seed-page-{i % 100}")
            for i in range(10_000)
        ]
        async with conn.transaction():
            await conn.executemany(
                f"INSERT INTO {fq_table('memory_change_events')} "
                "(event_id, bank_id, subject_id, event_type, detail) "
                "VALUES ($1, $2, '00000000-0000-0000-0000-000000000001', 'PAGE_MARK', "
                "jsonb_build_object('page_id', $3::text))",
                seed_rows,
            )

        class _Capture:
            backend_type = "postgresql"

            def __init__(self, delegate):
                self.delegate = delegate
                self.sql = None
                self.args = None
                self.calls = 0

            async def fetch(self, sql, *args, **kwargs):
                self.calls += 1
                self.sql = sql
                self.args = args
                return await self.delegate.fetch(sql, *args, **kwargs)

        capture = _Capture(conn)
        answers = await page_marks_read(capture, bank_id=bank_id, pages=[("seed-page-1", 0)])
        assert answers == {"seed-page-1": True}
        assert capture.calls == 1

        plan_rows = await conn.fetch("EXPLAIN (ANALYZE, FORMAT JSON) " + capture.sql, *capture.args)
        plan = json.loads(plan_rows[0][0])[0]
        top = plan["Plan"]

        def nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from nodes(child)

        scan = next(
            (
                node
                for node in nodes(top)
                if node.get("Relation Name") == "memory_change_events"
                and node.get("Node Type") in ("Seq Scan", "Bitmap Heap Scan")
            ),
            None,
        )
        print("\nU5 EXPLAIN (10k event rows):")
        print(json.dumps(top, indent=2, sort_keys=True, default=str))
        print(f"U5 statement count for the read: {capture.calls}")
        print(f"U5 rows returned by the read: {top.get('Actual Rows')}")
        if scan is not None:
            print(
                "U5 memory_change_events scan: "
                f"{scan.get('Node Type')}, rows {scan.get('Actual Rows')}, "
                f"cost {scan.get('Total Cost')}, filter {scan.get('Filter')}"
            )
        # U5's accepted cost is one statement that scans the bank's events:
        # nothing indexes detail, so page matching happens in the join filter.
        # Postgres may drive that scan from the bank_id-prefixed subject/seq
        # index (a bitmap scan) rather than a bare sequential scan. Either
        # shape reads all 10k rows of one bank, which is what is costed here.
        assert scan is not None, "U5 expects a scan of the bank's event rows"
        assert not any("detail" in (node.get("Index Cond") or "") for node in nodes(top))
        assert capture.calls == 1
        assert top["Actual Rows"] >= 1
        if scan["Node Type"] == "Bitmap Heap Scan":
            index_rows = [
                node["Actual Rows"]
                for node in nodes(scan)
                if node.get("Node Type") == "Bitmap Index Scan"
            ]
            assert index_rows and index_rows[0] >= 10_000, "the bank's whole event set is scanned"
    await memory.delete_bank(bank_id, request_context=request_context)
