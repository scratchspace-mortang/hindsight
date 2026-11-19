"""T6 (kata 1894): the page check at the consolidation job tail.

The check runs once per consolidation job, after the last batch commits and before
the ``hit_round_limit`` gate, over that run's created observations after the fold
substitution. It builds one shortlist per checked-set subject, makes one structured
decision call per job, and writes the chosen (observation, page) pairs through T7's
writer. Model property ids refer to machine 2 of ``model/staleness-marking.py@v3``;
the specification stays in that model.

Every case runs on a pg0 scratch bank or on in-memory doubles. No live bank is used.
"""

import inspect
import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from hindsight_api.engine import memory_engine as memory_engine_mod
from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.consolidation import consolidator as con
from hindsight_api.engine.memories.base import StoredMemory
from hindsight_api.engine.response_models import LLMCallResult
from hindsight_api.engine.schema import fq_store_table, fq_table

PAGE_QUESTION = "What are the project's conventions?"
PAGE_BODY = "## Conventions\n\nTests run on every push.\n"
DIM = 384


class _StubEmbeddings(Embeddings):
    model_name = "page-check-stub"

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
# Doubles
# ---------------------------------------------------------------------------


class _FakeConn:
    """Postgres-shaped connection that records statements and owns a fake transaction."""

    backend_type = "postgresql"

    def __init__(self, rows: list[StoredMemory] | None = None) -> None:
        self.rows = rows or []
        self.statements: list[str] = []
        self.tx_count = 0

    def in_transaction(self) -> bool:
        return True

    @asynccontextmanager
    async def transaction(self):
        self.tx_count += 1
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


class _FakeStore:
    """Returns preset observations and counts the one text-and-tags query."""

    def __init__(self, rows: list[StoredMemory]) -> None:
        self.rows = rows
        self.calls: list[list[str]] = []

    async def get_memories(self, *, conn, fq_table, bank_id, unit_ids):
        self.calls.append(list(unit_ids))
        wanted = set(unit_ids)
        return [row for row in self.rows if row.unit_id in wanted]


def _obs(unit_id: str, text: str, tags: list[str] | None = None) -> StoredMemory:
    return StoredMemory(unit_id=unit_id, text=text, fact_type="observation", tags=tags or [])


def _record_search(rows: list[dict[str, Any]]):
    calls: list[dict[str, Any]] = []

    async def fake_search(
        *, memory_engine, bank_id, observation_text, observation_tags, scope, request_context, limit=10
    ):
        calls.append(
            {
                "bank_id": bank_id,
                "text": observation_text,
                "tags": list(observation_tags),
                "scope": scope,
                "limit": limit,
            }
        )
        return [(str(row["id"]), str(row.get("source_query") or "")) for row in rows]

    return fake_search, calls


def _decision_returning(pairs: set[tuple[str, str]]):
    calls: list[dict[str, Any]] = []

    async def fake_decide(*, bank_id, candidates, observation_texts, request_context, llm_config, config):
        calls.append(
            {
                "bank_id": bank_id,
                "candidates": dict(candidates),
                "texts": dict(observation_texts),
            }
        )
        # A well-behaved server attributes only offered (observation, page) pairs.
        return frozenset(
            (obs, pid)
            for obs, pid in pairs
            if obs in candidates and any(pid == offered for offered, _ in candidates[obs])
        )

    return fake_decide, calls


# ---------------------------------------------------------------------------
# C7: page shortlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_shortlist_drops_snippet_and_body(monkeypatch):
    """Only (page id, source_query) leaves the seam; snippet and page body do not."""
    snippet = "SNIPPET-MUST-NOT-REACH-DECISION"
    body = "BODY-MUST-NOT-REACH-DECISION"

    class _Engine:
        async def search_pages(self, bank_id, query, *, limit, tag_filter, request_context):
            return [
                {
                    "id": "kp-1",
                    "source_query": PAGE_QUESTION,
                    "snippet": snippet,
                    "content": body,
                    "score": 0.9,
                }
            ]

    shortlist = await memory_engine_mod.page_shortlist(
        memory_engine=_Engine(),
        bank_id="bank",
        observation_text="an observation",
        observation_tags=["alpha"],
        scope="bank_wide",
        request_context=SimpleNamespace(),
    )
    assert shortlist == [("kp-1", PAGE_QUESTION)]
    assert snippet not in repr(shortlist)
    assert body not in repr(shortlist)


@pytest.mark.asyncio
async def test_page_shortlist_scope_and_one_embedding_one_search(monkeypatch):
    """bank_wide passes no tag filter; tag_wide passes a PageTagFilter from current tags.

    Each subject pays exactly one search_pages call, and search_pages embeds its own
    query, so no text is embedded twice and no precomputed embedding is handed in.
    """
    seen: list[Any] = []

    class _Engine:
        async def search_pages(self, bank_id, query, *, limit, tag_filter, request_context):
            seen.append((bank_id, query, tag_filter, limit))
            return [{"id": "kp-1", "source_query": PAGE_QUESTION, "snippet": "s"}]

    engine = _Engine()
    wide = await memory_engine_mod.page_shortlist(
        memory_engine=engine,
        bank_id="bank",
        observation_text="text A",
        observation_tags=["alpha"],
        scope="bank_wide",
        request_context=SimpleNamespace(),
    )
    scoped = await memory_engine_mod.page_shortlist(
        memory_engine=engine,
        bank_id="bank",
        observation_text="text A",
        observation_tags=["alpha"],
        scope="tag_wide",
        request_context=SimpleNamespace(),
    )
    assert wide == scoped == [("kp-1", PAGE_QUESTION)]
    assert len(seen) == 2
    assert seen[0][2] is None
    assert isinstance(seen[1][2], memory_engine_mod.PageTagFilter)
    assert seen[1][2].tags == ["alpha"]
    assert seen[1][2].tags_match == "all_strict"  # untagged pages never cross scopes
    assert seen[0][1] == seen[1][1] == "text A"


@pytest.mark.asyncio
async def test_page_shortlist_untagged_tag_wide_is_empty():
    """An untagged subject under tag_wide yields no shortlist and no search call."""

    class _Engine:
        def __init__(self):
            self.calls = 0

        async def search_pages(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("an untagged tag_wide subject must not search")

    engine = _Engine()
    result = await memory_engine_mod.page_shortlist(
        memory_engine=engine,
        bank_id="bank",
        observation_text="text",
        observation_tags=[],
        scope="tag_wide",
        request_context=SimpleNamespace(),
    )
    assert result == []
    assert engine.calls == 0


# ---------------------------------------------------------------------------
# C6/C8: the runner and the decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_check_no_created_set_is_noop(monkeypatch):
    """No created observations means no search, no decision and no marks."""
    searched, search_calls = _record_search([{"id": "kp", "source_query": PAGE_QUESTION}])
    decided, decision_calls = _decision_returning({("o", "kp")})
    monkeypatch.setattr(con, "page_shortlist", searched)
    monkeypatch.setattr(con, "decide_page_marks", decided)

    outcome = await con.run_observation_page_check(
        bank_id="bank",
        created={},
        config=SimpleNamespace(),
        request_context=SimpleNamespace(),
        memory_engine=SimpleNamespace(),
        conn=_FakeConn(),
    )
    assert outcome.ran is False
    assert outcome.selected == frozenset()
    assert outcome.marks_written == 0
    assert search_calls == [] and decision_calls == []


@pytest.mark.asyncio
async def test_page_check_uses_only_checked_set(monkeypatch):
    """An observation that is neither created this run nor a folded twin is never checked."""
    rows = [_obs("created", "created text"), _obs("bystander", "not this run")]
    store = _FakeStore(rows)
    monkeypatch.setattr(con, "get_memories", lambda: store)
    fake_search, search_calls = _record_search(
        [{"id": "kp", "source_query": PAGE_QUESTION}]
    )
    monkeypatch.setattr(con, "page_shortlist", fake_search)
    decided, decision_calls = _decision_returning(
        {("created", "kp"), ("bystander", "kp")}
    )
    monkeypatch.setattr(con, "decide_page_marks", decided)
    written: list[frozenset] = []

    async def fake_write(conn, *, bank_id, marks):
        written.append(marks)
        return len(marks)

    monkeypatch.setattr(con, "record_page_marks", fake_write)

    outcome = await con.run_observation_page_check(
        bank_id="bank",
        created={"created": ["fact-1"]},
        config=SimpleNamespace(),
        request_context=SimpleNamespace(),
        memory_engine=SimpleNamespace(),
        llm_config=SimpleNamespace(),
        conn=_FakeConn(rows),
    )
    assert outcome.ran is True
    assert store.calls == [["created"]]
    assert [c["text"] for c in search_calls] == ["created text"]
    assert list(decision_calls[0]["candidates"]) == ["created"]
    assert written == [frozenset({("created", "kp")})]


@pytest.mark.asyncio
async def test_page_check_empty_answer_writes_no_marks(monkeypatch):
    rows = [_obs("created", "created text")]
    monkeypatch.setattr(con, "get_memories", lambda: _FakeStore(rows))
    fake_search, _ = _record_search([{"id": "kp", "source_query": PAGE_QUESTION}])
    monkeypatch.setattr(con, "page_shortlist", fake_search)
    decided, _ = _decision_returning(set())
    monkeypatch.setattr(con, "decide_page_marks", decided)
    written: list[frozenset] = []

    async def fake_write(conn, *, bank_id, marks):
        written.append(marks)
        return len(marks)

    monkeypatch.setattr(con, "record_page_marks", fake_write)
    outcome = await con.run_observation_page_check(
        bank_id="bank",
        created={"created": ["fact-1"]},
        config=SimpleNamespace(),
        request_context=SimpleNamespace(),
        memory_engine=SimpleNamespace(),
        llm_config=SimpleNamespace(),
        conn=_FakeConn(rows),
    )
    assert outcome.ran is True
    assert outcome.selected == frozenset()
    assert outcome.marks_written == 0
    assert written == [frozenset()]


@pytest.mark.asyncio
async def test_page_mark_decision_one_call_drops_offshortlist_and_orphans():
    """One call per job; ids outside a block's shortlist or in no block are dropped."""
    calls: list[dict[str, Any]] = []

    class _LLM:
        async def call(self, **kwargs):
            calls.append(kwargs)
            return LLMCallResult(
                content={
                    "blocks": [
                        {"observation_id": "o1", "page_ids": ["p1", "off-list"]},
                        {"observation_id": "o2", "page_ids": ["p1"]},
                    ]
                }
            )

    selected = await con.decide_page_marks(
        bank_id="bank",
        candidates={"o1": [("p1", "q1")], "o2": [("p1", "q1")]},
        observation_texts={"o1": "text 1", "o2": "text 2"},
        request_context=SimpleNamespace(),
        llm_config=_LLM(),
        config=SimpleNamespace(llm_temperature_consolidation=0.0, llm_strict_schema_consolidation=False),
    )
    assert selected == frozenset({("o1", "p1"), ("o2", "p1")})
    assert len(calls) == 1
    prompt = json.dumps(calls[0]["messages"])
    assert "off-list" not in prompt  # never offered, so never sent


@pytest.mark.asyncio
async def test_page_mark_decision_unparsable_answer_raises():
    """An unparsable answer raises into T8's boundary; it does not fail open here."""

    class _LLM:
        async def call(self, **kwargs):
            return LLMCallResult(content="not json at all {{{")

    with pytest.raises((ValueError, TypeError)):
        await con.decide_page_marks(
            bank_id="bank",
            candidates={"o1": [("p1", "q1")]},
            observation_texts={"o1": "text"},
            request_context=SimpleNamespace(),
            llm_config=_LLM(),
            config=SimpleNamespace(llm_temperature_consolidation=0.0, llm_strict_schema_consolidation=False),
        )


@pytest.mark.asyncio
async def test_page_mark_decision_other_structured_model_is_empty_answer():
    """A different structured payload is no page decision, not an unparsable one.

    Existing consolidation tests drive every LLM call with one mock response of a
    different batch model. Treating that as an empty answer keeps the job tail from
    breaking those jobs; raw text and dicts that miss the schema still raise.
    """
    from hindsight_api.engine.consolidation.consolidator import _FactBatchResponse

    class _LLM:
        async def call(self, **kwargs):
            return LLMCallResult(content=_FactBatchResponse(facts=[], deletes=[]))

    selected = await con.decide_page_marks(
        bank_id="bank",
        candidates={"o1": [("p1", "q1")]},
        observation_texts={"o1": "text"},
        request_context=SimpleNamespace(),
        llm_config=_LLM(),
        config=SimpleNamespace(llm_temperature_consolidation=0.0, llm_strict_schema_consolidation=False),
    )
    assert selected == frozenset()


# ---------------------------------------------------------------------------
# C6 tail placement and model properties
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_check_runs_once_per_job(memory, request_context, monkeypatch):
    """The tail calls the runner once per job, not once per fetch iteration."""
    calls: list[dict[str, Any]] = []

    async def fake_check(**kwargs):
        calls.append(kwargs)
        return con.PageCheckOutcome(ran=False, selected=frozenset(), marks_written=0)

    async def fake_trigger(**kwargs):
        return 0

    monkeypatch.setattr(con, "run_observation_page_check", fake_check, raising=False)
    monkeypatch.setattr(con, "_trigger_mental_model_refreshes", fake_trigger)
    bank_id = f"test-pc-once-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
            "VALUES ($1, $2, 'a fresh fact', 'world', '{}', now(), now())",
            uuid.uuid4(),
            bank_id,
        )
    result = await con.run_consolidation_job(memory, bank_id, request_context)
    assert result["status"] == "completed"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_page_check_p6_runs_before_trigger(monkeypatch):
    """P6 (tail order): the check returns before the refresh trigger runs."""
    source = inspect.getsource(con._run_consolidation_job)
    check_at = source.index("run_observation_page_check(")
    gate_at = source.index("if hit_round_limit:", check_at)
    trigger_at = source.index("_trigger_mental_model_refreshes(", gate_at)
    assert check_at < gate_at < trigger_at


@pytest.mark.asyncio
async def test_page_check_p4_marks_only_checked_shortlisted_pairs(monkeypatch):
    """P4: a mark names a shortlisted page and an observation in the checked set."""
    rows = [_obs("o1", "text")]
    store = _FakeStore(rows)
    monkeypatch.setattr(con, "get_memories", lambda: store)
    fake_search, _ = _record_search([{"id": "p1", "source_query": "q1"}])
    monkeypatch.setattr(con, "page_shortlist", fake_search)

    class _LLM:
        async def call(self, **kwargs):
            # p3 was never offered to this observation.
            return LLMCallResult(content={"blocks": [{"observation_id": "o1", "page_ids": ["p3"]}]})

    written: list[frozenset] = []

    async def fake_write(conn, *, bank_id, marks):
        written.append(marks)
        return len(marks)

    monkeypatch.setattr(con, "record_page_marks", fake_write)
    outcome = await con.run_observation_page_check(
        bank_id="bank",
        created={"o1": ["fact"]},
        config=SimpleNamespace(llm_temperature_consolidation=0.0, llm_strict_schema_consolidation=False),
        request_context=SimpleNamespace(),
        memory_engine=SimpleNamespace(),
        llm_config=_LLM(),
        conn=_FakeConn(rows),
    )
    assert outcome.ran is True
    assert outcome.selected == frozenset()
    assert outcome.marks_written == 0
    assert written == [frozenset()]


# ---------------------------------------------------------------------------
# R15: the fold substitution, with a surviving twin that may predate the run
# ---------------------------------------------------------------------------


def test_checked_set_after_folds_removes_folded_id_and_adds_twin():
    checked = con._checked_set_after_folds(
        {"folded": ["f1", "f2"], "other": ["f3"]},
        {"folded": "twin"},
    )
    assert "folded" not in checked
    assert checked["twin"] == ["f1", "f2"]
    assert checked["other"] == ["f3"]


@pytest.mark.asyncio
async def test_page_check_folded_twin_marks_page(memory, request_context, monkeypatch):
    """The folded row is deleted at the fold site; the tail reads the surviving twin."""
    bank_id = f"test-pc-twin-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    page = await memory.create_page(
        bank_id, "Conventions", PAGE_QUESTION, PAGE_BODY, request_context=request_context
    )
    page_id = page["id"]
    async with pool.acquire() as conn:
        folded = uuid.uuid4()
        twin = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags) VALUES "
            "($1, $2, 'folded text', 'observation', '{}'), ($3, $2, 'twin text', 'observation', '{}')",
            folded,
            bank_id,
            twin,
        )
        # The fold site deletes the created row; created itself is never pruned.
        await con._execute_delete_action(conn, bank_id, str(folded))
        checked = con._checked_set_after_folds({str(folded): ["fact-1"]}, {str(folded): str(twin)})
        assert list(checked) == [str(twin)]

        fake_search, search_calls = _record_search([{"id": page_id, "source_query": PAGE_QUESTION}])
        monkeypatch.setattr(con, "page_shortlist", fake_search)
        decided, _ = _decision_returning({(str(twin), page_id)})
        monkeypatch.setattr(con, "decide_page_marks", decided)

        outcome = await con.run_observation_page_check(
            bank_id=bank_id,
            created=checked,
            config=SimpleNamespace(observation_marking_page_stale="tag_wide"),
            request_context=request_context,
            memory_engine=memory,
            llm_config=SimpleNamespace(),
            conn=conn,
        )
        assert outcome.ran is True
        assert [c["text"] for c in search_calls] == ["twin text"]
        row = await conn.fetchrow(
            f"SELECT subject_id, detail FROM {fq_table('memory_change_events')} "
            "WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
        assert row is not None
        assert row["subject_id"] == twin
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_check_folded_into_existing_twin_marks_page(memory, request_context, monkeypatch):
    """The twin predates the run; the mark subject is the twin and the page reads stale."""
    bank_id = f"test-pc-existing-twin-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()
    page = await memory.create_page(
        bank_id, "Conventions", PAGE_QUESTION, PAGE_BODY, request_context=request_context
    )
    page_id = page["id"]
    async with pool.acquire() as conn:
        twin = uuid.uuid4()  # built before this run
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags) VALUES ($1, $2, 'old twin', 'observation', '{}')",
            twin,
            bank_id,
        )
        checked = con._checked_set_after_folds({"folded-here": ["fact-1"]}, {"folded-here": str(twin)})
        assert list(checked) == [str(twin)]

        fake_search, search_calls = _record_search([{"id": page_id, "source_query": PAGE_QUESTION}])
        monkeypatch.setattr(con, "page_shortlist", fake_search)
        decided, _ = _decision_returning({(str(twin), page_id)})
        monkeypatch.setattr(con, "decide_page_marks", decided)

        outcome = await con.run_observation_page_check(
            bank_id=bank_id,
            created=checked,
            config=SimpleNamespace(observation_marking_page_stale="tag_wide"),
            request_context=request_context,
            memory_engine=memory,
            llm_config=SimpleNamespace(),
            conn=conn,
        )
        assert outcome.ran is True
        assert [c["text"] for c in search_calls] == ["old twin"]
        row = await conn.fetchrow(
            f"SELECT subject_id FROM {fq_table('memory_change_events')} "
            "WHERE bank_id = $1 AND event_type = 'PAGE_MARK'",
            bank_id,
        )
        assert row is not None
        assert row["subject_id"] == twin
    from hindsight_api.engine.memory_engine import page_marks_read

    async with pool.acquire() as conn:
        marks = await page_marks_read(conn, bank_id=bank_id, pages=[(page_id, 0)])
        assert marks[page_id] is True
    await memory.delete_bank(bank_id, request_context=request_context)
