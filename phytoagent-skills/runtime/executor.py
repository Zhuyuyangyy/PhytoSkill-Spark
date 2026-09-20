"""Minimal local executor. Scheduling and model decisions belong to the Harness."""

import hashlib
from pathlib import Path
from time import perf_counter

from registry import SkillRegistry
from sdk import BaseSkill
from sdk.exceptions import ManifestError, SkillError, UnsupportedModeError
from sdk.schema import package_file, validate_payload


class SkillExecutor:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute(self, name: str, payload: dict, *, mode: str = "fixture") -> dict:
        record = self.registry.verify_entry(name)
        manifest = record["manifest"]
        if mode not in manifest["supported_modes"]:
            raise UnsupportedModeError(f"{name} does not support mode {mode!r}")
        validate_payload(payload, record["input_schema"], label=f"{name} input")
        package = Path(record["package_dir"])
        filename, class_name = manifest["entrypoint"].split(":")
        source_path = package_file(package, filename)
        source = source_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != manifest["files"][filename]:
            raise ManifestError("Entrypoint changed during loading")
        namespace = {"__name__": f"phyto_skill_{name}", "__file__": str(source_path)}
        # Compile the verified source, not an unverified .pyc. This is a trusted
        # local Python process, not a security sandbox.
        exec(compile(source, str(source_path), "exec"), namespace)
        skill_type = namespace.get(class_name)
        if not isinstance(skill_type, type) or not issubclass(skill_type, BaseSkill):
            raise ManifestError("Entrypoint must be a BaseSkill subclass")
        return skill_type(package).execute(payload, mode=mode)

    def call(self, name: str, payload: dict, *, mode: str, tool_call_id: str) -> dict:
        start = perf_counter()
        try:
            data = self.execute(name, payload, mode=mode)
            response = {"status": "success", "data": data, "error": None}
        except SkillError as exc:
            response = {"status": "failed", "data": None,
                        "error": {"code": type(exc).__name__, "message": str(exc)}}
        except Exception:
            # Arbitrary adapter errors might contain credentials or source data.
            response = {"status": "failed", "data": None,
                        "error": {"code": "SkillExecutionError", "message": "Skill adapter execution failed"}}
        return {"tool_call_id": tool_call_id, "name": name, "mode": mode,
                "duration_ms": round((perf_counter() - start) * 1000, 3), **response}
