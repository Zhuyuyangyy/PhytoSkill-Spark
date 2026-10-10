"""Child-side entry point for bounded Skill execution.

The parent (see :mod:`runtime.executor`) writes a JSON request describing one
call to this process's stdin and runs it under the limits in
:mod:`runtime.sandbox`. This module imports the verified Skill source, runs
one call, and reports the outcome on stdout as a single sentinel-prefixed
line, so anything the Skill itself prints cannot corrupt the result.

The request is re-verified here rather than trusted from the parent: the
entrypoint hash and the constructor contract are checked again inside the
boundary, so a package that changed between the parent's check and this
process's import is refused, and a Skill demanding collaborators the executor
cannot supply is refused with the same error the in-process path raises.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

RESULT_SENTINEL = "__PHYTO_BOUNDED_RESULT__"


def _read_request() -> dict:
    request = json.loads(sys.stdin.read())
    for key in ("package_dir", "entrypoint", "payload", "mode", "entrypoint_sha256"):
        if key not in request:
            raise ValueError(f"request is missing {key!r}")
    return request


def _build_skill(skill_type: type, package: Path, request: dict):
    """Instantiate the Skill, mirroring the executor's constructor contract.

    Only ``self`` and the package directory may be required; anything else is
    a dependency the package declares but the boundary cannot supply. Optional
    collaborators are built from what the request carries: the cache directory
    (the cache is file-backed, so the child shares the parent's entries) and
    the corpus retriever.
    """
    from sdk.exceptions import ManifestError

    parameters = inspect.signature(skill_type.__init__).parameters
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
        # Left as None: the Skill applies its own default.
        extras["model"] = None
    if "env_file" in accepted:
        extras["env_file"] = None
    if "cache" in accepted and request.get("cache_dir"):
        from backends.observation_cache import ObservationCache
        extras["cache"] = ObservationCache(request["cache_dir"])
    return skill_type(package, **extras)


def main() -> int:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from sdk import BaseSkill
    from sdk.exceptions import SkillError

    try:
        request = _read_request()
        package = Path(request["package_dir"])
        filename, class_name = request["entrypoint"].split(":")
        source_path = package / filename
        source = source_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != request["entrypoint_sha256"]:
            raise ValueError("Entrypoint changed between verification and execution")
        namespace = {"__name__": "phyto_skill_bounded", "__file__": str(source_path)}
        exec(compile(source, str(source_path), "exec"), namespace)
        skill_type = namespace.get(class_name)
        if not isinstance(skill_type, type) or not issubclass(skill_type, BaseSkill):
            raise ValueError("Entrypoint must be a BaseSkill subclass")
        skill = _build_skill(skill_type, package, request)
        data = skill.execute(request["payload"], mode=request["mode"])
        print(RESULT_SENTINEL + json.dumps({"ok": True, "data": data}, ensure_ascii=False))
        return 0
    except SkillError as exc:
        print(RESULT_SENTINEL + json.dumps(
            {"ok": False, "code": type(exc).__name__, "message": str(exc)},
            ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - the parent maps this to a failed call
        print(RESULT_SENTINEL + json.dumps(
            {"ok": False, "code": "SkillExecutionError",
             "message": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
