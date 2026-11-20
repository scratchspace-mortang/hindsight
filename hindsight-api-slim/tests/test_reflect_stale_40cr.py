"""40cr acceptance: stale graph-mode pages are dropped from reflect search.

The producer is the real staleness computation stubbed through the engine's
batched answer; the consumer is tool_search_mental_models. A test that built its
own page dict and re-filtered it would verify the fixture, not this seam.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.reflect.tools import tool_search_mental_models
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.schema import fq_table


def _row(page_id: str, name: str, trigger: dict, content: str = "body") -> dict:
    return {
        "id": page_id,
        "name": name,
        "content": content,
        "tags": None,
        "created_at": None,
        "last_refreshed_at": None,
        "last_memory_seen_at": None,
        "trigger": json.dumps(trigger),
        "relevance": 0.9,
    }


@pytest.mark.asyncio
async def test_stale_graph_page_dropped_tagscope_kept_fresh_kept() -> None:
    """A stale graph page is absent; a stale tagscope page and a fresh graph page remain."""
    rows = [
        _row("graph-stale", "GraphStale", {"staleness_check": "graph"}),
        _row("graph-fresh", "GraphFresh", {"staleness_check": "graph"}),
        _row("tag-stale", "TagStale", {"staleness_check": "tagscope"}),
    ]
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    engine = AsyncMock()
    engine.compute_mental_models_are_stale = AsyncMock(
        return_value={"graph-stale": True, "graph-fresh": False, "tag-stale": True}
    )

    result = await tool_search_mental_models(
        memory_engine=engine,
        conn=conn,
        bank_id="40cr-acceptance",
        query="current state",
        query_embedding=[0.1, 0.2],
    )

    ids = [page["id"] for page in result["mental_models"]]
    assert "graph-stale" not in ids, "a stale graph page must not be offered to the model"
    assert "graph-fresh" in ids, "a fresh graph page must remain"
    assert "tag-stale" in ids, "tagscope behaviour must not change"
    tag = next(page for page in result["mental_models"] if page["id"] == "tag-stale")
    assert tag["is_stale"] is True, "a kept tagscope page keeps its stale label"


@pytest.mark.asyncio
async def test_dropped_page_can_still_leave_lower_layers_available() -> None:
    """Dropping the only page does not itself raise; reflect may still search lower layers."""
    rows = [_row("graph-stale", "GraphStale", {"staleness_check": "graph"})]
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    engine = AsyncMock()
    engine.compute_mental_models_are_stale = AsyncMock(return_value={"graph-stale": True})

    result = await tool_search_mental_models(
        memory_engine=engine,
        conn=conn,
        bank_id="40cr-acceptance-2",
        query="current state",
        query_embedding=[0.1, 0.2],
    )

    assert result["mental_models"] == []
    assert result["count"] == 0


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_a_fresh_page_below_the_stale_top_k_is_still_served(memory: MemoryEngine, request_context) -> None:
    """C5: the drop runs after the LIMIT, so the fetch must leave room for it.

    Five stale graph pages rank above one fresh page. The SQL path used to fetch exactly
    ``max_results`` and then drop the stale ones, so the fresh page ranked below the cut
    was never fetched and the tool returned ``count: 0``. The ranking here is the real
    query against real rows; the staleness verdicts are the engine's batched answer
    stubbed, as in the other cases in this file.
    """
    bank_id = f"40cr-search-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    stale_ids: list[str] = []
    try:
        for index in range(5):
            page = await memory.create_mental_model(
                bank_id=bank_id,
                name=f"stale graph page {index}",
                source_query=f"stale page {index}",
                content=f"stale body {index}",
                trigger={"staleness_check": "graph"},
                request_context=request_context,
            )
            stale_ids.append(page["id"])
        fresh = await memory.create_mental_model(
            bank_id=bank_id,
            name="fresh graph page",
            source_query="fresh page",
            content="fresh body",
            trigger={"staleness_check": "graph"},
            request_context=request_context,
        )
        pool = await memory._get_pool()
        embedding = (await embedding_utils.generate_embeddings_batch(memory.embeddings, ["q"]))[0]
        # Deterministic rank: the stale pages sit on the query vector, the fresh one far
        # from it, so any top-k below five is all stale.
        near = "[" + ",".join(repr(value) for value in embedding) + "]"
        far = "[" + ",".join(repr(-value) for value in embedding) + "]"
        async with pool.acquire() as conn:
            for page_id in stale_ids:
                await conn.execute(
                    f"UPDATE {fq_table('mental_models')} SET embedding = $1::vector WHERE id = $2",
                    near,
                    page_id,
                )
            await conn.execute(
                f"UPDATE {fq_table('mental_models')} SET embedding = $1::vector WHERE id = $2",
                far,
                fresh["id"],
            )
            staleness = {page_id: True for page_id in stale_ids} | {fresh["id"]: False}
            with patch.object(memory, "compute_mental_models_are_stale", new=AsyncMock(return_value=staleness)):
                result = await tool_search_mental_models(memory, conn, bank_id, "q", embedding, max_results=5)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

    returned = [page["id"] for page in result["mental_models"]]
    assert returned == [fresh["id"]], "the stale top-k is dropped and the fresh page ranked below it is still served"
    assert result["count"] == 1, "the result count is the truncated set, not the fetched one"


@pytest.mark.asyncio
async def test_store_ranked_search_also_overfetches_before_the_stale_drop(monkeypatch) -> None:
    """C5 store path: a store that ranks pages must be asked for the same candidate set.

    The store path passes its own ``limit``; asking it for exactly ``max_results`` repeats
    the SQL path's defect for a deployment whose pages live in the store: the stale top-k is
    all it returns, and the drop leaves nothing.
    """
    stale_ids = [f"store-stale-{index}" for index in range(5)]
    fresh_id = "store-fresh"
    rows = [_row(page_id, page_id, {"staleness_check": "graph"}) for page_id in stale_ids]
    rows.append(_row(fresh_id, "FreshPage", {"staleness_check": "graph"}))
    ranked = [
        SimpleNamespace(page_id=page_id, score=1.0 - index / 10) for index, page_id in enumerate([*stale_ids, fresh_id])
    ]
    asked_limits: list[int] = []

    async def rank(bank_id, *, embedding, limit, **kwargs):
        asked_limits.append(limit)
        return ranked[:limit]

    store = SimpleNamespace(store_owned_for=lambda bank_id: True, search_pages_semantic=rank)
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)

    async def hydrate(query, bank_id, ids):
        return [row for row in rows if row["id"] in ids]

    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=hydrate)
    engine = AsyncMock()
    engine.compute_mental_models_are_stale = AsyncMock(
        return_value={**{page_id: True for page_id in stale_ids}, fresh_id: False}
    )

    result = await tool_search_mental_models(engine, conn, "40cr-store", "current state", [0.1, 0.2])

    assert asked_limits == [15], "the store is asked for the same 15-candidate set the SQL path ranks (5 x 3)"
    assert [page["id"] for page in result["mental_models"]] == [fresh_id], (
        "the stale store-ranked top-k is dropped and the fresh page below it is still served"
    )
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_store_ranked_search_truncates_fresh_survivors_to_max_results(monkeypatch) -> None:
    """C5 truncation: more fresh pages survive the stale drop than ``max_results``.

    Nothing is dropped here -- all six candidates are fresh graph pages -- so only the
    slice back to ``max_results`` can hold the result to five. The other C5 scenarios end
    with one survivor, which a stale drop alone produces: a regression that returned the
    whole fetched candidate set (up to 3x the requested pages) would still satisfy their
    count of one. This case fails by the length assertion when that slice is gone.
    """
    fresh_ids = [f"store-fresh-{index}" for index in range(6)]
    rows = [_row(page_id, page_id, {"staleness_check": "graph"}) for page_id in fresh_ids]
    ranked = [SimpleNamespace(page_id=page_id, score=1.0 - index / 10) for index, page_id in enumerate(fresh_ids)]
    asked_limits: list[int] = []

    async def rank(bank_id, *, embedding, limit, **kwargs):
        asked_limits.append(limit)
        return ranked[:limit]

    store = SimpleNamespace(store_owned_for=lambda bank_id: True, search_pages_semantic=rank)
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)

    async def hydrate(query, bank_id, ids):
        return [row for row in rows if row["id"] in ids]

    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=hydrate)
    engine = AsyncMock()
    engine.compute_mental_models_are_stale = AsyncMock(return_value={page_id: False for page_id in fresh_ids})

    result = await tool_search_mental_models(engine, conn, "40cr-truncate", "current state", [0.1, 0.2], max_results=5)

    assert len(result["mental_models"]) == 5, "reflect gets the requested five, not the fetched candidate set"
    assert result["count"] == 5, "the result count is the truncated set, not the fetched one"
    assert [page["id"] for page in result["mental_models"]] == fresh_ids[:5], (
        "the truncation keeps the top-ranked five in rank order"
    )
    assert asked_limits == [15], "the store is asked for the 15-candidate set the SQL path ranks"
