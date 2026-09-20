"""Explicit fixture adapter; replace run() when a real adapter is available."""

from sdk.fixture_skill import FixtureSkill


class GrowthRiskSkill(FixtureSkill):
    """One narrowly scoped, contract-validated capability."""
