"""Staleness settings on the mental-model trigger.

``staleness_check`` and ``graph_stale`` are per-page settings, not global
config: they ride the ``MentalModelTrigger`` schema and the engine trigger
merge. Absent means the ``tagscope`` / ``cited_memories`` defaults, and any
other literal — including combined ones like ``tagscope_or_graph`` — is
rejected at write time. Fast, deterministic unit tests: no DB, no LLM.
"""

import pytest
from pydantic import ValidationError

from hindsight_api.api.http import MentalModelTrigger
from hindsight_api.engine.memory_engine import MemoryEngine

DEFAULTS = {"staleness_check": "tagscope", "graph_stale": "cited_memories"}


def _merge(trigger=None, base=None):
    """Run the engine trigger merge without a database or tenant setup.

    ``_merge_trigger`` only reads the class-level page default, so an
    uninitialised instance is enough to exercise it.
    """
    return MemoryEngine.__new__(MemoryEngine)._merge_trigger(trigger, base=base)


class TestStalenessCheckSchema:
    def test_defaults_when_unset(self):
        trigger = MentalModelTrigger()
        assert trigger.staleness_check == "tagscope"
        assert trigger.graph_stale == "cited_memories"

    def test_defaults_when_fields_absent_from_a_stored_trigger(self):
        """A legacy row stored before these fields existed reads back as defaults."""
        trigger = MentalModelTrigger.model_validate({"mode": "delta", "refresh_cron": "0 3 * * *"})
        assert trigger.staleness_check == "tagscope"
        assert trigger.graph_stale == "cited_memories"
        assert trigger.mode == "delta"
        assert trigger.refresh_cron == "0 3 * * *"

    def test_explicit_values_round_trip(self):
        trigger = MentalModelTrigger.model_validate({"staleness_check": "graph", "graph_stale": "read_memories"})
        assert trigger.staleness_check == "graph"
        assert trigger.graph_stale == "read_memories"
        dumped = trigger.model_dump()
        assert dumped["staleness_check"] == "graph"
        assert dumped["graph_stale"] == "read_memories"

    @pytest.mark.parametrize(
        "value",
        ["tagscope_or_graph", "tagscope_and_graph", "graph_or_tagscope", "graph_and_tagscope", "all", ""],
    )
    def test_staleness_check_rejects_combined_and_unknown_literals(self, value):
        with pytest.raises(ValidationError):
            MentalModelTrigger.model_validate({"staleness_check": value})

    @pytest.mark.parametrize(
        "value",
        ["cited_or_read", "cited_and_read", "both", "all", ""],
    )
    def test_graph_stale_rejects_combined_and_unknown_literals(self, value):
        with pytest.raises(ValidationError):
            MentalModelTrigger.model_validate({"graph_stale": value})


class TestStalenessCheckMerge:
    def test_page_create_without_fields_stores_defaults(self):
        """The page default carries the settings, so a page built without
        either field has them once merged for storage."""
        merged = _merge(None)
        for key, value in DEFAULTS.items():
            assert merged[key] == value, f"{key} missing from merged page trigger"

    def test_explicit_values_win_over_the_page_default(self):
        merged = _merge({"staleness_check": "graph", "graph_stale": "read_memories"})
        assert merged["staleness_check"] == "graph"
        assert merged["graph_stale"] == "read_memories"

    def test_one_explicit_value_keeps_the_other_default(self):
        merged = _merge({"staleness_check": "graph"})
        assert merged["staleness_check"] == "graph"
        assert merged["graph_stale"] == "cited_memories"

    def test_patch_preserves_a_stored_explicit_value(self):
        """Patching one trigger field must not reset the staleness settings."""
        stored = _merge({"staleness_check": "graph", "graph_stale": "read_memories"})
        patched = _merge({"refresh_cron": "0 3 * * *"}, base=stored)
        assert patched["refresh_cron"] == "0 3 * * *"
        assert patched["staleness_check"] == "graph"
        assert patched["graph_stale"] == "read_memories"

    def test_legacy_base_without_fields_validates_to_defaults(self):
        """A legacy stored trigger has no keys; through the read surface the
        defaults apply instead of stale-forever."""
        legacy = {"mode": "delta", "fact_types": ["observation"]}
        parsed = MentalModelTrigger.model_validate(legacy)
        assert parsed.staleness_check == "tagscope"
        assert parsed.graph_stale == "cited_memories"

    def test_bank_default_layer_carries_explicit_values(self):
        """``knowledge_page_default_trigger`` merges over the built-in page
        default, so an explicit bank default survives the layering."""
        page_default = _merge({"staleness_check": "graph"})
        merged = _merge(None, base=page_default)
        assert merged["staleness_check"] == "graph"
        assert merged["graph_stale"] == "cited_memories"


class TestOracleSingleBatchAgreement:
    @pytest.mark.asyncio
    async def test_single_graph_page_on_oracle_answers_like_batch_scope_rule(self):
        """On a non-Postgres backend the change-record table does not
        exist, so the single graph path must not query it — it answers by
        the same scope rule the batch uses there."""
        import uuid as _uuid
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        engine = MemoryEngine.__new__(MemoryEngine)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        cited = str(_uuid.uuid4())
        conn = SimpleNamespace(
            backend_type="oracle",
            fetchrow=AsyncMock(return_value=None),
            fetchval=AsyncMock(side_effect=AssertionError("change-events query on Oracle")),
            fetch=AsyncMock(side_effect=AssertionError("change-events query on Oracle")),
        )
        engine._bank_write_watermark = AsyncMock(return_value=datetime.now(timezone.utc))

        from hindsight_api.engine.memories.base import StoredMemory

        store = SimpleNamespace(
            live_memory_ids=AsyncMock(return_value={cited}),
            get_memories=AsyncMock(
                return_value=[StoredMemory(unit_id=cited, text="t", fact_type="world", context=None)]
            ),
            mental_model_is_stale=AsyncMock(return_value=False),
        )
        row = {
            "id": str(_uuid.uuid4()),
            "tags": [],
            "trigger": {"staleness_check": "graph", "graph_stale": "cited_memories"},
            "last_memory_seen_at": future,
            "last_refreshed_at": future,
            "reflect_response": {
                "based_on": {"world": [{"id": cited, "text": "t", "type": "world"}]},
                "graph_provenance": {"read_memory_ids": [cited], "captured_empty": False, "baseline_seq": 3},
            },
        }
        import hindsight_api.engine.memories as memories_mod

        original = memories_mod.get_memories
        memories_mod.get_memories = lambda: store  # type: ignore[assignment]
        try:
            assert await engine.compute_mental_model_is_stale(conn, "bank", row) is False
        finally:
            memories_mod.get_memories = original  # type: ignore[assignment]


class TestMcpNullTrigger:
    def test_explicit_null_staleness_fields_keep_stored_values(self):
        """An explicit null for the staleness fields means
        "keep the stored setting" (their description says so), so the MCP
        trigger patch must drop them instead of overwriting with null."""
        from hindsight_api.mcp_tools import MentalModelTriggerInput, _mental_model_trigger_patch

        trigger = MentalModelTriggerInput.model_validate(
            {"staleness_check": None, "graph_stale": None, "refresh_cron": "0 3 * * *"}
        )
        patch = _mental_model_trigger_patch(trigger)
        assert patch is not None
        assert "staleness_check" not in patch
        assert "graph_stale" not in patch
        assert patch["refresh_cron"] == "0 3 * * *"
