"""A fixture is bound to one explicit case, never an inference fallback."""

from copy import deepcopy

from sdk.base_skill import BaseSkill
from sdk.exceptions import FixtureMismatchError, UnsupportedModeError
from sdk.schema import package_file, read_json


class FixtureSkill(BaseSkill):
    def run(self, payload: dict, *, mode: str) -> dict:
        if mode != "fixture":
            raise UnsupportedModeError("This adapter only implements fixture mode")
        fixture = read_json(package_file(self.package_dir, "fixture.json"))
        if payload != fixture["input"]:
            raise FixtureMismatchError("Input does not match the declared synthetic fixture; configure a live adapter for real data")
        return deepcopy(fixture["output"])
