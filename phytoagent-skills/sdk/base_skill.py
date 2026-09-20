"""The single contract all fixture and future live adapters implement."""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

from sdk.exceptions import ManifestError, UnsupportedModeError
from sdk.manifest import verify_manifest
from sdk.schema import load_schema, validate_payload


class BaseSkill(ABC):
    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir).absolute()
        self._manifest = verify_manifest(self.package_dir)
        self.name = self._manifest["name"]
        self.version = self._manifest["version"]
        self._input_schema = load_schema(self.package_dir, self._manifest["input_schema"])
        self._output_schema = load_schema(self.package_dir, self._manifest["output_schema"])

    @property
    def manifest(self) -> dict:
        return deepcopy(self._manifest)

    @property
    def metadata(self) -> dict:
        return {"skill_name": self.name, "version": self.version,
                "description": self._manifest["description"], "author": self._manifest["author"],
                "runtime": self._manifest["runtime"],
                "input_schema": deepcopy(self._input_schema), "output_schema": deepcopy(self._output_schema)}

    def validate_input(self, payload: Any) -> None:
        validate_payload(payload, self._input_schema, label=f"{self.name} input")

    def validate_output(self, payload: Any) -> None:
        validate_payload(payload, self._output_schema, label=f"{self.name} output")

    def execute(self, payload: dict, *, mode: str = "fixture") -> dict:
        current = verify_manifest(self.package_dir)
        if current["manifest_sha256"] != self._manifest["manifest_sha256"]:
            raise ManifestError("Package changed after skill initialization; reload it explicitly")
        if mode not in self._manifest["supported_modes"]:
            raise UnsupportedModeError(f"{self.name} does not support mode {mode!r}")
        self.validate_input(payload)
        result = self.run(deepcopy(payload), mode=mode)
        self.validate_output(result)
        return result

    @abstractmethod
    def run(self, payload: dict, *, mode: str) -> dict:
        """Implement only this method; execute() owns input and output validation."""
