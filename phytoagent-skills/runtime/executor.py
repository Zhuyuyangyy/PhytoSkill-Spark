"""Minimal local executor. Scheduling and model decisions belong to the Harness.

Security boundary, stated explicitly:

* This module executes the Skill's verified source **in this process**. It is a
  trusted local Python process, not a sandbox.
* It has **no permission layer**. Authorisation lives in
  :class:`runtime.shield.ShieldRuntime`, which wraps this executor. A caller that
  reaches ``SkillExecutor`` directly, or a Skill that imports ``subprocess`` or
  ``socket`` itself, bypasses that authorisation.
* The compile-time gate in ``shield/gate.py`` scans package text for those
  imports; it is a policy check, not an isolation mechanism.

Callers that need enforcement must go through ``ShieldRuntime.call``.
"""

import hashlib
import inspect
from pathlib import Path
from time import perf_counter

from registry import SkillRegistry
from sdk import BaseSkill
from sdk.exceptions import ManifestError, SkillError, UnsupportedModeError
from sdk.schema import package_file, validate_payload


def instantiate(skill_type: type, package: Path) -> BaseSkill:
    """Construct a Skill, supplying optional dependencies it declares.

    A Skill may accept keyword-only collaborators (for example a corpus) without
    the executor knowing about them. Anything it does not declare is not passed,
    so the executor stays independent of individual Skill implementations.
    """
    parameters = inspect.signature(skill_type.__init__).parameters
    # Only ``self`` and the package directory may be required; the executor
    # supplies the latter positionally. Anything else is a dependency the
    # package declares but the executor cannot satisfy.
    required = [name for name, parameter in parameters.items()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                       inspect.Parameter.KEYWORD_ONLY)]
    if required not in (["self"], ["self", "package_dir"]):
        raise ManifestError("A Skill entrypoint may not require constructor arguments the executor cannot supply")
    accepted = {name for name, parameter in parameters.items()
                if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.KEYWORD_ONLY)}
    extras: dict = {}
    if "corpus" in accepted:
        from corpus.retriever import LocalCorpus
        extras["corpus"] = LocalCorpus()
    if "model" in accepted:
        # Left as None: the Skill applies its own default. Passing a model name
        # here would make the executor decide something that belongs to the Skill.
        extras["model"] = None
    if "env_file" in accepted:
        extras["env_file"] = None
    return skill_type(package, **extras)


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
        return instantiate(skill_type, package).execute(payload, mode=mode)

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
