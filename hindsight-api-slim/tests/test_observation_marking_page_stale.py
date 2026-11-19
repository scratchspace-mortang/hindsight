"""Bank-level scope setting for the observation page check (kata p3z1, contract C5).

The setting decides which pages the page-check candidate search may draw from.
It is a value on HindsightConfig, resolved per bank through ConfigResolver, and
never a field on the per-page trigger schema.
"""

from dataclasses import fields
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.config_resolver import ConfigResolver
from hindsight_api.engine.response_models import MentalModelTrigger
from hindsight_api.mcp_tools import MentalModelTriggerInput

from .test_hierarchical_config import FakeBankConfigBackend, FakeBankConfigConnection

ENV_NAME = "HINDSIGHT_API_OBSERVATION_MARKING_PAGE_STALE"
FIELD_NAME = "observation_marking_page_stale"


def test_observation_marking_page_stale_defaults_to_tag_wide():
    """The upstream default is tag_wide and the declared type is the two-literal contract."""
    config = HindsightConfig.from_env()  # AttributeError on the pre-change tree
    assert config.observation_marking_page_stale == "tag_wide"

    hints = get_type_hints(HindsightConfig)
    declared = hints[FIELD_NAME]
    assert get_origin(declared) is Literal
    assert set(get_args(declared)) == {"bank_wide", "tag_wide"}

    field = next(f for f in fields(HindsightConfig) if f.name == FIELD_NAME)
    assert field.default == "tag_wide"


def test_observation_marking_page_stale_rejects_unknown_value(monkeypatch):
    """Config load rejects any value outside the two literals, it does not coerce or default it."""
    config = HindsightConfig.from_env()  # AttributeError on the pre-change tree
    assert config.observation_marking_page_stale == "tag_wide"

    monkeypatch.setenv(ENV_NAME, "sideways")
    with pytest.raises(ValueError, match=FIELD_NAME):
        HindsightConfig.from_env()


class _PerBankBackend(FakeBankConfigBackend):
    """A fake whose stored config differs per bank, so resolution cannot look global."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def acquire(self):
        return _PerBankConnection(self)


class _PerBankConnection(FakeBankConfigConnection):
    async def fetchrow(self, query, bank_id):
        return {"config": self.backend.configs.get(bank_id, {})}


@pytest.mark.asyncio
async def test_observation_marking_page_stale_resolves_per_bank():
    """A bank override wins in its bank while a second bank keeps the default."""
    assert FIELD_NAME in HindsightConfig.get_configurable_fields()

    backend = _PerBankBackend(
        {"bank-with-override": {FIELD_NAME: "bank_wide"}, "bank-without-override": {}}
    )
    resolver = ConfigResolver(backend=backend)

    overridden = await resolver.resolve_full_config("bank-with-override", cached=False)
    defaulted = await resolver.resolve_full_config("bank-without-override", cached=False)

    assert overridden.observation_marking_page_stale == "bank_wide"
    assert defaulted.observation_marking_page_stale == "tag_wide"


def test_observation_marking_page_stale_absent_from_page_trigger_schema():
    """The value is bank-scoped, so no page trigger schema or MCP trigger input carries it."""
    assert FIELD_NAME not in MentalModelTrigger.model_fields
    assert FIELD_NAME not in MentalModelTriggerInput.model_fields
