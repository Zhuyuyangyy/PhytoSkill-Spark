"""Minimal local executor. Scheduling and model decisions belong to the Harness.

Execution boundary, stated explicitly:

* ``fixture`` mode runs **in this process**: it returns sealed synthetic data,
  so there is nothing to bound, and the A/B harness needs the speed.
* Every other mode (``live``, ``herb``, ``corpus``) runs in a **separate
  process** under the OS-level limits in :mod:`runtime.sandbox` — memory, child
  process count and wall-clock time are enforced by the operating system, so a
  runaway adapter exhausts its own process instead of the host.
* That boundary is a **resource** boundary, not a security sandbox: it does not
  confine filesystem or network access. See :mod:`runtime.sandbox` for the full
  statement of what it does and does not cover.
* This module has **no permission layer**. Authorisation lives in
  :class:`runtime.shield.ShieldRuntime`, which wraps this executor. A caller that
  reaches ``SkillExecutor`` directly bypasses that authorisation.
* The compile-time gate in ``shield/gate.py`` scans package text for dangerous
  imports; it is a policy check, not an isolation mechanism.

Callers that need enforcement must go through ``ShieldRuntime.call``.
"""

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from registry import SkillRegistry
from runtime import sandbox
from runtime.sandbox import ExecutionBoundaryError
from runtime.worker import RESULT_SENTINEL
from sdk import BaseSkill, exceptions
from sdk.exceptions import (ManifestError, SkillError, SkillExecutionError,
                            UnsupportedModeError)
from sdk.schema import package_file, validate_payload

# Modes that run behind the OS-level execution boundary. Fixture mode is the
# deliberate exception: sealed synthetic data, nothing to bound.
BOUNDARY_MODES = ("live", "herb", "corpus")


def instantiate(skill_type: type, package: Path, *, cache=None) -> BaseSkill:
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
    # A shared cache is injected so repeated calls in one run reuse observations
    # instead of re-running a 12-167 s inference per call.
    if "cache" in accepted and cache is not None:
        extras["cache"] = cache
    return skill_type(package, **extras)


class SkillExecutor:
    def __init__(self, registry: SkillRegistry, *, cache=None,
                 boundary_modes=BOUNDARY_MODES,
                 limits: sandbox.ResourceLimits | None = None):
        self.registry = registry
        self._cache = cache
        # Which modes run behind the OS-level boundary, and with what caps.
        # An empty tuple forces fully in-process execution (tests use it to
        # compare the two paths); the default bounds every real mode.
        self.boundary_modes = tuple(boundary_modes)
        self.limits = limits or sandbox.ResourceLimits()

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
        if mode in self.boundary_modes:
            return self._execute_bounded(name, package, filename, class_name,
                                         source, payload, mode)
        namespace = {"__name__": f"phyto_skill_{name}", "__file__": str(source_path)}
        # Compile the verified source, not an unverified .pyc. Fixture mode is
        # trusted synthetic data; every other mode runs behind the boundary.
        exec(compile(source, str(source_path), "exec"), namespace)
        skill_type = namespace.get(class_name)
        if not isinstance(skill_type, type) or not issubclass(skill_type, BaseSkill):
            raise ManifestError("Entrypoint must be a BaseSkill subclass")
        return instantiate(skill_type, package, cache=self._cache).execute(
            payload, mode=mode)

    def _execute_bounded(self, name: str, package: Path, filename: str, class_name: str,
                         source: bytes, payload: dict, mode: str) -> dict:
        """Run one call in a separate process under OS-level limits.

        The child re-verifies the entrypoint hash inside the boundary, so a
        package that changed between this check and its import is refused. The
        outcome crosses back as a sentinel-prefixed stdout line, which nothing
        the Skill prints can corrupt. A timeout or a child that died without a
        result raises :class:`ExecutionBoundaryError`; a Skill error is
        re-raised with its original type so callers see the same classification
        the in-process path would have produced.
        """
        request = {
            "package_dir": str(package), "entrypoint": f"{filename}:{class_name}",
            "payload": payload, "mode": mode,
            "entrypoint_sha256": hashlib.sha256(source).hexdigest(),
            # The cache is file-backed, so the child shares the parent's
            # entries; only the directory travels across the boundary.
            "cache_dir": str(self._cache.cache_dir) if self._cache is not None else None,
        }
        repo_root = Path(__file__).resolve().parents[1]
        result = sandbox.run_bounded(
            [sys.executable, "-m", "runtime.worker"], limits=self.limits,
            cwd=repo_root, env=sandbox.child_environment(repo_root),
            input_bytes=json.dumps(request).encode("utf-8"))
        if result.timed_out:
            raise ExecutionBoundaryError(
                f"{name} exceeded the {self.limits.timeout_seconds:g}s execution boundary")
        lines = [line for line in result.stdout.splitlines()
                 if line.startswith(RESULT_SENTINEL)]
        if not lines:
            raise ExecutionBoundaryError(
                f"{name} produced no result inside the execution boundary "
                f"(exit {result.returncode}): {result.stderr.strip()[-200:]}")
        outcome = json.loads(lines[-1][len(RESULT_SENTINEL):])
        if outcome["ok"]:
            return outcome["data"]
        error_type = getattr(exceptions, outcome.get("code", ""), None)
        if isinstance(error_type, type) and issubclass(error_type, SkillError):
            raise error_type(outcome["message"])
        raise SkillExecutionError(outcome["message"])

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
