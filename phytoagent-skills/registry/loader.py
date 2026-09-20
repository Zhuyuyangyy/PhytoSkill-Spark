"""Atomic, metadata-only discovery and Tool Schema export for StepFun."""

from copy import deepcopy
from pathlib import Path

from registry.validator import read_instructions, validate_package
from sdk.exceptions import RegistryError, SkillError
from sdk.schema import read_json, validate_payload

CONFIG_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["version", "skills_dir", "require_signature", "trusted_public_key"],
    "properties": {
        "version": {"const": 1},
        "skills_dir": {"type": "string", "minLength": 1},
        "require_signature": {"type": "boolean"},
        "trusted_public_key": {"type": ["string", "null"], "minLength": 1},
    },
}


class SkillRegistry:
    def __init__(self, skills_dir: str | Path, *, trusted_public_key: str | Path | None = None,
                 require_signature: bool = True):
        self.skills_dir = Path(skills_dir).absolute()
        self.trusted_public_key = Path(trusted_public_key).absolute() if trusted_public_key is not None else None
        self.require_signature = require_signature
        self._records: dict[str, dict] = {}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SkillRegistry":
        config_path = Path(config_path).resolve()
        config = read_json(config_path)
        validate_payload(config, CONFIG_SCHEMA, label="Registry configuration")
        key = config["trusted_public_key"]
        return cls(config_path.parent / config["skills_dir"],
                   trusted_public_key=config_path.parent / key if key is not None else None,
                   require_signature=config["require_signature"])

    def discover(self) -> list[dict]:
        root = self.skills_dir
        if (not root.is_dir() or root.is_symlink()
                or (hasattr(root, "is_junction") and root.is_junction())):
            raise RegistryError("Skills root must be an existing directory, not a link")
        if self.require_signature and self.trusted_public_key is None:
            raise RegistryError("Signed discovery requires a pinned public key")
        if self.trusted_public_key is not None and self.trusted_public_key.resolve().is_relative_to(root.resolve()):
            raise RegistryError("Trusted public key must be outside the scanned skills directory")
        pending = {}
        package_dir = root
        try:
            for package_dir in sorted(root.iterdir()):
                if package_dir.name.startswith(".") or package_dir.name == "__pycache__":
                    continue
                if not package_dir.is_dir() and not package_dir.is_symlink():
                    continue
                record = validate_package(package_dir, trusted_public_key=self.trusted_public_key,
                                          require_signature=self.require_signature)
                name = record["manifest"]["name"]
                if name in pending:
                    raise RegistryError(f"Duplicate Skill: {name}")
                pending[name] = record
        except (SkillError, OSError) as exc:
            raise RegistryError(f"Discovery failed for {package_dir.name}: {exc}") from exc
        # Publish a complete snapshot only after every package passes.
        self._records = pending
        return self.records

    def register(self, package_dir: str | Path) -> dict:
        """Register one installed package without inspecting unrelated siblings."""
        package_dir = Path(package_dir).absolute()
        if package_dir.parent.resolve() != self.skills_dir.resolve():
            raise RegistryError("Skill must be a direct child of the configured skills directory")
        if self.trusted_public_key is not None and self.trusted_public_key.resolve().is_relative_to(self.skills_dir.resolve()):
            raise RegistryError("Trusted public key must be outside the scanned skills directory")
        record = validate_package(package_dir, trusted_public_key=self.trusted_public_key,
                                  require_signature=self.require_signature)
        name = record["manifest"]["name"]
        if name in self._records:
            raise RegistryError(f"Duplicate Skill: {name}")
        self._records = {**self._records, name: record}
        return self.get(name)

    @property
    def records(self) -> list[dict]:
        return deepcopy(list(self._records.values()))

    @property
    def catalog(self) -> list[dict]:
        """Small routing context; no instructions, schemas or local paths."""
        return [{"name": name, "description": record["manifest"]["description"]}
                for name, record in self._records.items()]

    def catalog_names(self) -> list[str]:
        """Names only, for registry bookkeeping in tests and reports."""
        return sorted(self._records)

    def load_skill(self, name: str) -> dict:
        """Second disclosure level: expand only the selected Skill."""
        record = self.verify_entry(name)
        return {"instructions": read_instructions(Path(record["package_dir"]), record["manifest"]),
                "tool": {"type": "function", "function": {
                    "name": name, "description": record["manifest"]["description"],
                    "parameters": deepcopy(record["input_schema"])}},
                "output_schema": deepcopy(record["output_schema"]),
                "manifest_sha256": record["manifest"]["manifest_sha256"],
                "verification": deepcopy(record["verification"])}

    @property
    def available_tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": name, "description": record["manifest"]["description"],
                "parameters": deepcopy(record["input_schema"]),
            }} for name, record in self._records.items()
        ]

    def get(self, name: str) -> dict:
        try:
            return deepcopy(self._records[name])
        except KeyError as exc:
            raise RegistryError(f"Unknown Skill: {name}") from exc

    def verify_entry(self, name: str) -> dict:
        """Call immediately before consuming package content or loading its code."""
        previous = self.get(name)
        current = validate_package(previous["package_dir"], trusted_public_key=self.trusted_public_key,
                                   require_signature=self.require_signature)
        if current["manifest"]["manifest_sha256"] != previous["manifest"]["manifest_sha256"]:
            raise RegistryError(f"Skill {name} changed since discovery; rediscover explicitly")
        return current

    def instructions(self, name: str) -> str:
        record = self.verify_entry(name)
        return read_instructions(Path(record["package_dir"]), record["manifest"])

    def validate_input(self, name: str, payload: dict) -> None:
        record = self.verify_entry(name)
        validate_payload(payload, record["input_schema"], label=f"{name} input")

    def validate_output(self, name: str, payload: dict) -> None:
        record = self.verify_entry(name)
        validate_payload(payload, record["output_schema"], label=f"{name} output")
