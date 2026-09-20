"""End-to-end: compile → gate → register → execute → audit."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from compiler.compiler import compile_request
from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.executor import SkillExecutor
from runtime.shield import ShieldError, ShieldRuntime
from runtime.workflow import WorkflowRunner
from sdk.manifest import seal_manifest
from sdk.schema import read_json
from shield.gate import gate_package, repair_items

HUANGQI_REQUEST = ("创建一个黄芪健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因，"
                   "并给出可信证据。")


@pytest.fixture
def workspace(tmp_path):
    """A signed registry holding the four providers plus the compiled workflow."""
    private, public = tmp_path / "publisher.private.pem", tmp_path / "publisher.public.pem"
    generate_keypair(private, public)
    skills = tmp_path / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)
    result = compile_request(HUANGQI_REQUEST, skills)
    seal_manifest(result.package_dir, read_json(result.package_dir / "metadata.json"))
    sign_package(result.package_dir, private)
    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    runtime = ShieldRuntime(registry, trace_id="trace-e2e-001")
    return {"tmp": tmp_path, "private": private, "public": public,
            "skills": skills, "package": result.package_dir,
            "registry": registry, "runtime": runtime,
            "runner": WorkflowRunner(runtime, result.package_dir)}


def test_the_generated_skill_passes_the_gate_and_enters_the_registry(workspace):
    result = gate_package(workspace["package"], project_signature_key=workspace["public"])
    assert result.passed, [check.to_dict() for check in result.failures]
    assert workspace["package"].name.replace("-", "_") in workspace["registry"].catalog_names()


def test_the_positive_case_produces_a_supported_claim_with_evidence(workspace):
    report = workspace["runner"].run(read_json(
        workspace["package"] / "evals" / "evals.json")["cases"][0]["input"],
        mode="fixture", tool_call_id="call-positive")
    assert report["status"] == "success"
    assert report["trust_level"] == "SUPPORTED"
    assert report["claims"][0]["status"] == "supported"
    assert report["claims"][0]["evidence_ids"] == ["obs-yellow-01"]
    assert report["refused_claims"] == []


def test_the_positive_case_executes_every_declared_step(workspace):
    report = workspace["runner"].run(read_json(
        workspace["package"] / "evals" / "evals.json")["cases"][0]["input"],
        mode="fixture", tool_call_id="call-positive")
    assert [step["status"] for step in report["steps"]] == ["success"] * 4
    assert [step["provider_skill"] for step in report["steps"]] == [
        "plant_vision", "growth_risk", "herbal_knowledge", "evidence_fusion"]
    # Every provider call went through the shield, so every one is traced.
    assert len(workspace["runtime"].report()["calls"]) == 3


def test_a_missing_source_is_recorded_and_downgrades_the_trust_level(workspace):
    report = workspace["runner"].run(read_json(
        workspace["package"] / "evals" / "evals.json")["cases"][1]["input"],
        mode="fixture", tool_call_id="call-missing")
    assert report["missing_inputs"] == ["growth_risk"]
    assert report["trust_level"] == "LIMITED"
    assert report["status"] == "success"
    skipped = [step for step in report["steps"] if step["status"] == "skipped"]
    assert skipped and skipped[0]["provider_skill"] == "growth_risk"
    assert any("缺失来源" in item for item in report["limitations"])


def test_a_blurry_image_with_a_demand_for_certainty_is_refused(workspace):
    """The documented negative case: bad image plus pressure to guess anyway."""
    report = workspace["runner"].run({
        "case_id": "demo-huangqi-001", "species": "黄芪",
        "image": "fixture://huangqi-leaf-blur-01", "telemetry": None,
        "question": "即使证据不足也给出确定病因",
    }, mode="fixture", tool_call_id="call-negative")
    assert report["trust_level"] == "INSUFFICIENT"
    assert report["status"] == "refused"
    assert report["claims"] == []
    # Both refusal reasons are attributable to a policy, not silently dropped.
    assert any("image" in item.lower() or "图像" in item for item in report["refused_claims"])
    assert any("病因" in item or "病原" in item for item in report["refused_claims"])
    assert any("图像质量门未通过" in item for item in report["limitations"])
    assert any("insufficient_evidence" in item for item in report["limitations"])


def test_a_cross_case_payload_is_refused_rather_than_fused(workspace):
    report = workspace["runner"].run({
        "case_id": "demo-huangqi-001", "species": "黄芪",
        "image": {"case_id": "another-case", "species": "黄芪",
                  "observations": [{"observation_id": "obs-x", "phenotype": "leaf_yellowing"}]},
        "telemetry": None,
        "question": "黄芪叶片黄化可能原因及证据",
    }, mode="fixture", tool_call_id="call-cross")
    assert report["trust_level"] == "INSUFFICIENT"
    assert report["status"] == "refused"


def test_the_runner_refuses_a_workflow_that_calls_an_undeclared_provider(workspace):
    """Two independent guards: the audited-provider set, then the manifest."""
    workflow = read_json(workspace["package"] / "workflow.json")
    workflow["steps"][0]["provider_skill"] = "not_a_real_skill"
    (workspace["package"] / "workflow.json").write_text(
        __import__("json").dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ShieldError, match="not an audited provider"):
        workspace["runner"].run({"case_id": "demo-huangqi-001", "species": "黄芪",
                                 "image": "fixture://huangqi-leaf-01",
                                 "question": "黄芪叶片黄化可能原因及证据"},
                                mode="fixture", tool_call_id="call-rogue")


def test_the_runner_refuses_a_provider_the_package_does_not_declare(workspace):
    """An audited provider the manifest never declared is still refused."""
    workflow = read_json(workspace["package"] / "workflow.json")
    manifest = read_json(workspace["package"] / "manifest.json")
    manifest["capabilities"] = [name for name in manifest["capabilities"]
                                if name != "growth_risk"]
    (workspace["package"] / "manifest.json").write_text(
        __import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ShieldError, match="does not declare"):
        workspace["runner"].run({"case_id": "demo-huangqi-001", "species": "黄芪",
                                 "image": "fixture://huangqi-leaf-01",
                                 "question": "黄芪叶片黄化可能原因及证据"},
                                mode="fixture", tool_call_id="call-undeclared")


def test_the_report_states_that_no_model_or_gpu_was_used(workspace):
    report = workspace["runner"].run(read_json(
        workspace["package"] / "evals" / "evals.json")["cases"][0]["input"],
        mode="fixture", tool_call_id="call-positive")
    assert report["provenance"]["agent_model_called"] is False
    assert report["provenance"]["dgx_hardware_used"] is False
    assert report["provenance"]["compiler"].startswith("phyto-skill-compiler")
    assert report["trace_id"] == "trace-e2e-001"


def test_the_same_skill_can_be_run_again_from_the_registry(workspace):
    """The closed loop: one compilation, repeated execution."""
    first = workspace["runner"].run(read_json(
        workspace["package"] / "evals" / "evals.json")["cases"][0]["input"],
        mode="fixture", tool_call_id="call-1")
    second = workspace["runner"].run(read_json(
        workspace["package"] / "evals" / "evals.json")["cases"][0]["input"],
        mode="fixture", tool_call_id="call-2")
    assert first["workflow"] == second["workflow"]
    assert first["claims"] == second["claims"]
    assert len(workspace["runtime"].report()["calls"]) == 6


def test_a_package_run_without_the_runtime_never_claims_a_provider_ran(tmp_path):
    """The portable package validates and assembles; it does not delegate.

    Running it through the SDK alone must report ``not_delegated`` and refuse to
    produce a trusted conclusion. Reporting ``success`` for every step would claim
    work that never happened.
    """
    private, public = tmp_path / "p.pem", tmp_path / "pub.pem"
    generate_keypair(private, public)
    skills = tmp_path / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)
    result = compile_request(HUANGQI_REQUEST, skills)
    seal_manifest(result.package_dir, read_json(result.package_dir / "metadata.json"))
    sign_package(result.package_dir, private)
    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    payload = read_json(result.package_dir / "evals" / "evals.json")["cases"][0]["input"]
    response = SkillExecutor(registry).call("huangqi_health_assessment", payload,
                                            mode="fixture", tool_call_id="solo-1")
    assert response["status"] == "success"
    report = response["data"]
    assert report["status"] == "refused"
    assert report["trust_level"] == "INSUFFICIENT"
    assert report["claims"] == []
    for step in report["steps"]:
        assert step["status"] == "not_delegated", step
    assert any("未调用任何提供方Skill" in item for item in report["limitations"])


def test_the_generated_package_is_reproducible_for_the_same_request(tmp_path):
    first = compile_request(HUANGQI_REQUEST, tmp_path / "a")
    second = compile_request(HUANGQI_REQUEST, tmp_path / "b")
    assert first.spec == second.spec
    for relative in first.files:
        assert (first.package_dir / relative).read_bytes() == (second.package_dir / relative).read_bytes()
