"""The execution boundary: what the operating system enforces, and what it does not.

The registry and the hostile packages are built under ``tmp_path`` (pytest does
not delete those during a run), so these tests exercise the real boundary —
real Job Objects, real kills — rather than a mock of one.

One test pins a *non*-property on purpose: the boundary does not confine
filesystem access. That test exists so nobody mistakes this module for a
sandbox; the honest statement of coverage lives in :mod:`runtime.sandbox`.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.executor import BOUNDARY_MODES, SkillExecutor
from runtime.sandbox import (DEFAULT_MAX_PROCESSES, ExecutionBoundaryError,
                             ResourceLimits, child_environment, run_bounded)
from sdk.exceptions import ContractError
from sdk.manifest import seal_manifest
from sdk.schema import read_json

CORPUS_INPUT = {"case_id": "demo-huangqi-001", "species": "黄芪",
                "query": "黄芪 心脾两虚 归脾汤"}

TAMPERED_TEMPLATE = '''"""A hostile Skill: it exists only to test the execution boundary."""

from pathlib import Path

from sdk import BaseSkill


class {class_name}(BaseSkill):
    def __init__(self, package_dir, corpus=None):
        super().__init__(package_dir)
        self.corpus = corpus

    def execute(self, payload, mode="fixture"):
        # No validation, no contract: the boundary is what stands between this
        # and the host.
{body}

    def run(self, payload, *, mode="fixture"):
        raise NotImplementedError
'''

SLEEPER_BODY = ("        import time\n"
                "        time.sleep(120)\n"
                "        return {'status': 'success'}")
HOG_BODY = ("        chunks = []\n"
            "        for _ in range(64):\n"
            "            chunks.append(bytearray(256 * 1024 * 1024))\n"
            "        return {'status': 'success'}")
SPAWNER_BODY = ("        import subprocess, sys\n"
                "        for _ in range(16):\n"
                "            subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "        return {'status': 'success'}")
# Reads a file that has nothing to do with its package. This *succeeds*: the
# boundary bounds resources, not the filesystem. The test pins that fact.
READER_BODY = ("        outside = Path(payload['case_id'])\n"
               "        return {'status': 'success', 'bytes_read': len(outside.read_bytes())}")


def add_tampered_package(skills: Path, private, name: str, class_name: str, body: str):
    """A real package shape with a hostile skill.py, sealed and signed."""
    destination = skills / name
    shutil.copytree(PROJECT_ROOT / "skills" / "herbal_knowledge", destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
    metadata = read_json(destination / "metadata.json")
    metadata["name"] = name
    metadata["entrypoint"] = f"skill.py:{class_name}"
    metadata["supported_modes"] = ["corpus"]
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # SKILL.md carries the package name and discovery checks it.
    skill_md = (destination / "SKILL.md").read_text(encoding="utf-8")
    skill_md = skill_md.replace("name: herbal-knowledge", f"name: {name}")
    skill_md = skill_md.replace("# herbal-knowledge", f"# {name}")
    (destination / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (destination / "skill.py").write_text(
        TAMPERED_TEMPLATE.format(class_name=class_name, body=body), encoding="utf-8")
    seal_manifest(destination, metadata)
    sign_package(destination, private)


@pytest.fixture
def bounded_registry(tmp_path):
    """A signed registry holding the audited packages plus three hostile ones."""
    private, public = tmp_path / "publisher.private.pem", tmp_path / "publisher.public.pem"
    generate_keypair(private, public)
    skills = tmp_path / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)
    add_tampered_package(skills, private, "sleeper", "SleeperSkill", SLEEPER_BODY)
    add_tampered_package(skills, private, "hog", "HogSkill", HOG_BODY)
    add_tampered_package(skills, private, "spawner", "SpawnerSkill", SPAWNER_BODY)
    add_tampered_package(skills, private, "reader", "ReaderSkill", READER_BODY)
    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    return registry


# ── which modes are bounded ─────────────────────────────────────────────────


def test_fixture_mode_stays_in_process_and_real_modes_are_bounded():
    """Fixture data is sealed and synthetic; real adapters get the boundary."""
    assert BOUNDARY_MODES == ("live", "herb", "corpus")
    executor = SkillExecutor(registry=None)
    assert executor.boundary_modes == BOUNDARY_MODES
    assert executor.limits.timeout_seconds > 0
    # An empty tuple is the explicit opt-out, used to compare the two paths.
    assert SkillExecutor(registry=None, boundary_modes=()).boundary_modes == ()


def test_the_default_process_cap_accounts_for_the_interpreters_own_startup():
    """One slot is the interpreter's: a Windows venv redirector spawns the real
    interpreter, so a cap of one would stop the child from starting at all."""
    assert DEFAULT_MAX_PROCESSES == 2
    assert ResourceLimits().max_processes == DEFAULT_MAX_PROCESSES


def test_the_child_environment_drops_instrumentation_and_pins_the_import_surface():
    """The boundary is not the agent: the caller's sandbox instrumentation must
    not travel into it, and the child's import surface is this repository."""
    repo = Path("/repo/root")
    env = child_environment(repo, {"CODEBUDDY_TOOL_CALL_ID": "abc", "WORKBUDDY_X": "1",
                                   "PATH": "C:\\bin", "OLLAMA_HOST": "http://host:11434"})
    assert "CODEBUDDY_TOOL_CALL_ID" not in env
    assert "WORKBUDDY_X" not in env
    assert env["PATH"] == "C:\\bin"
    assert env["OLLAMA_HOST"] == "http://host:11434"
    assert env["PYTHONPATH"] == str(repo)


# ── what the boundary enforces ──────────────────────────────────────────────


def test_a_bounded_call_returns_what_the_in_process_path_returns(bounded_registry):
    """The boundary changes where code runs, not what it computes."""
    bounded = SkillExecutor(bounded_registry)
    in_process = SkillExecutor(bounded_registry, boundary_modes=())
    bounded_result = bounded.call("herbal_knowledge", CORPUS_INPUT, mode="corpus",
                                  tool_call_id="bounded-1")
    in_process_result = in_process.call("herbal_knowledge", CORPUS_INPUT, mode="corpus",
                                        tool_call_id="in-process-1")
    assert bounded_result["status"] == "success"
    assert bounded_result["data"] == in_process_result["data"]


def test_a_skill_error_crosses_the_boundary_with_its_classification(bounded_registry):
    """A refusal inside the boundary arrives with its original type."""
    executor = SkillExecutor(bounded_registry)
    response = executor.call("herbal_knowledge", {"case_id": "demo-huangqi-001",
                                                  "species": "黄芪", "query": "   "},
                             mode="corpus", tool_call_id="bounded-2")
    assert response["status"] == "failed"
    assert response["error"]["code"] == "ContractError"
    with pytest.raises(ContractError):
        executor.execute("herbal_knowledge", {"case_id": "demo-huangqi-001",
                                              "species": "黄芪", "query": "   "},
                         mode="corpus")


def test_a_skill_that_sleeps_forever_is_killed_at_the_wall_clock_limit(bounded_registry):
    """A hung adapter must not hang the run — and the host must not notice."""
    executor = SkillExecutor(bounded_registry,
                             limits=ResourceLimits(timeout_seconds=4))
    with pytest.raises(ExecutionBoundaryError, match="execution boundary"):
        executor.execute("sleeper", CORPUS_INPUT, mode="corpus")
    response = executor.call("sleeper", CORPUS_INPUT, mode="corpus",
                             tool_call_id="bounded-3")
    assert response["status"] == "failed"
    assert response["error"]["code"] == "ExecutionBoundaryError"


def test_a_skill_that_allocates_beyond_the_cap_dies_at_the_os_limit(bounded_registry):
    """The memory cap is enforced by the operating system, not by the Skill."""
    executor = SkillExecutor(bounded_registry,
                             limits=ResourceLimits(memory_bytes=512 * 1024 ** 2,
                                                   timeout_seconds=60))
    response = executor.call("hog", CORPUS_INPUT, mode="corpus",
                             tool_call_id="bounded-4")
    assert response["status"] == "failed"


def test_a_skill_that_spawns_helpers_is_stopped_by_the_process_limit(bounded_registry):
    """With the default cap the interpreter's startup uses one slot, so the
    Skill's own second spawn is what the operating system refuses."""
    executor = SkillExecutor(bounded_registry,
                             limits=ResourceLimits(max_processes=2, timeout_seconds=60))
    response = executor.call("spawner", CORPUS_INPUT, mode="corpus",
                             tool_call_id="bounded-5")
    assert response["status"] == "failed"


# ── what the boundary does NOT enforce (pinned on purpose) ──────────────────


def test_the_boundary_does_not_confine_the_filesystem(bounded_registry, tmp_path):
    """This test documents a limit, not a feature: a bounded Skill can still
    read a file that has nothing to do with its package. Filesystem and network
    governance stays with the permission model; anyone needing OS-level
    confinement must run the whole runtime in a container."""
    outside = tmp_path / "outside.txt"
    outside.write_text("x" * 32, encoding="utf-8")
    executor = SkillExecutor(bounded_registry)
    response = executor.call("reader", {"case_id": str(outside), "species": "黄芪",
                                        "query": "read something"},
                             mode="corpus", tool_call_id="bounded-6")
    # If this ever starts failing, the boundary grew filesystem confinement —
    # which would be welcome, but the module docstring must be rewritten first.
    assert response["status"] == "success"
    assert response["data"]["bytes_read"] == 32



@pytest.mark.skipif(sys.platform == "win32",
                    reason="process groups are POSIX; the Windows path is the Job Object, "
                           "covered by the tests above")
def test_a_posix_timeout_terminates_the_whole_process_tree(tmp_path):
    """The wall-clock timeout must kill the tree, not just the direct child.

    The timed-out process spawns a helper that writes a marker once the
    timeout has fired; if the process-group kill works, that marker never
    appears. Runs on Linux — which is what the DGX Spark node is.
    """
    marker = tmp_path / "grandchild-survived.txt"
    tree = tmp_path / "tree.py"
    tree.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c',\n"
        f"               'import time; time.sleep(10); open({str(marker)!r}, \"w\").write(\"x\")'])\n"
        "time.sleep(120)\n", encoding="utf-8")
    result = run_bounded([sys.executable, str(tree)],
                         limits=ResourceLimits(timeout_seconds=3))
    assert result.timed_out
    time.sleep(13)  # long enough for the helper's own sleep to have elapsed
    assert not marker.exists(), "the timed-out process tree outlived the boundary"
