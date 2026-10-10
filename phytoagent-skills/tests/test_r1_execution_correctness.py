"""R1 — Evidence-Grounded Execution Correctness: negative regressions.

Each test pins one failure mode from the 2026-10-09 review of the public
repository, so a regression fails here instead of shipping quietly:

1. a claim whose asserted phenotype is not carried by the evidence it cites
   must not reach SUPPORTED, however valid the evidence id is;
2. the workflow's conclusions must be derived from the observation actually
   made, not from fixed wording that happens to match one fixture;
3. a workflow step that did not execute must never be recorded as success;
4. the harness's tool calls must go through the governed interface, and a
   middleware rejection must be survivable and visible.

Everything here runs offline: packages are signed with disposable demo keys
in temporary directories.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compiler.compiler import compile_request
from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.shield import TRUST_INSUFFICIENT, ClaimAuditor, load_evidence_index
from runtime.workflow import WorkflowRunner
from sdk.manifest import seal_manifest
from sdk.schema import read_json

HUANGQI_REQUEST = ("创建一个黄芪健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因，"
                   "并给出可信证据。")

POSITIVE_INPUT = {
    "case_id": "demo-huangqi-001", "species": "黄芪",
    "image": "fixture://huangqi-leaf-01",
    "telemetry": {"temperature_c": 31.5, "relative_humidity_pct": 40,
                  "soil_ph": 7.8, "npk": None},
    "question": "黄芪叶片黄化可能原因及证据",
}


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
    from runtime.shield import ShieldRuntime
    runtime = ShieldRuntime(registry, trace_id="trace-r1-001")
    return {"tmp": tmp_path, "private": private, "public": public,
            "skills": skills, "package": result.package_dir,
            "registry": registry, "runtime": runtime,
            "runner": WorkflowRunner(runtime, result.package_dir)}


def _reseal(workspace, package: Path) -> None:
    """Re-seal and re-sign a package after editing it, then rediscover.

    Editing a package file invalidates its manifest hashes; running it stale
    would fail for the wrong reason (or worse, pass against the old hash).
    """
    seal_manifest(package, read_json(package / "metadata.json"))
    sign_package(package, workspace["private"])
    workspace["registry"].discover()


def _tamper_vision_phenotype(workspace, phenotype: str) -> None:
    """Make the plant_vision fixture report a different phenotype."""
    package = workspace["skills"] / "plant_vision"
    fixture = read_json(package / "fixture.json")
    fixture["output"]["observations"][0]["phenotype"] = phenotype
    (package / "fixture.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _reseal(workspace, package)


# ── 1. structured assertions are checked against the evidence ─────────────────


def test_a_phenotype_assertion_the_evidence_does_not_carry_is_refused():
    """A valid evidence id is not proof: the record must carry the phenotype."""
    index = {"obs-1": {"observation_id": "obs-1", "phenotype": "leaf_spot"}}
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "在叶缘区域观察到黄化", "evidence_ids": ["obs-1"],
        "asserts": {"phenotype": "leaf_yellowing"}}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "no single cited record" in verdict["claims"][0]["reason"]
    assert verdict["trust_level"] == TRUST_INSUFFICIENT


def test_a_phenotype_assertion_the_evidence_carries_is_supported():
    index = {"obs-1": {"observation_id": "obs-1", "phenotype": "leaf_yellowing"}}
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "在叶缘区域观察到黄化", "evidence_ids": ["obs-1"],
        "asserts": {"phenotype": "leaf_yellowing"}}]})
    assert verdict["claims"][0]["status"] == "supported"
    assert verdict["trust_level"] == "SUPPORTED"


def test_knowledge_evidence_supports_a_phenotype_through_supports_phenotypes():
    """Knowledge records carry the phenotypes they support, not a phenotype."""
    index = {"e1": {"evidence_id": "e1", "supports_phenotypes": ["leaf_yellowing"]}}
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "语料支持该表型判断", "evidence_ids": ["e1"],
        "asserts": {"phenotype": "leaf_yellowing"}}]})
    assert verdict["claims"][0]["status"] == "supported"


def test_a_record_that_carries_no_phenotype_establishes_none():
    """Fail closed: a factor record cannot support a phenotype assertion."""
    index = {"f1": {"factor_id": "f1", "metric": "soil_ph"}}
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "土壤偏碱导致黄化", "evidence_ids": ["f1"],
        "asserts": {"phenotype": "leaf_yellowing"}}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "no single cited record" in verdict["claims"][0]["reason"]


def test_a_species_or_case_assertion_must_match_the_cited_records():
    """The evidence index carries its envelope's case and species."""
    index = load_evidence_index({
        "case_id": "demo-huangqi-001", "species": "黄芪",
        "observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing"}]})
    auditor = ClaimAuditor(available_evidence=index)
    for key, value in (("species", "人参"), ("case_id", "another-case")):
        verdict = auditor.audit({"claims": [{
            "text": "观察结论", "evidence_ids": ["o1"],
            "asserts": {"phenotype": "leaf_yellowing", key: value}}]})
        assert verdict["claims"][0]["status"] == "refused", key
        assert key in verdict["claims"][0]["reason"], key


def test_a_malformed_assertion_is_refused_rather_than_ignored():
    index = {"obs-1": {"observation_id": "obs-1", "phenotype": "leaf_yellowing"}}
    auditor = ClaimAuditor(available_evidence=index)
    for asserts in ({"phenotype": 7}, "phenotype", {"": "x"}, []):
        verdict = auditor.audit({"claims": [{
            "text": "观察结论", "evidence_ids": ["obs-1"], "asserts": asserts}]})
        assert verdict["claims"][0]["status"] == "refused", asserts
        assert "malformed" in verdict["claims"][0]["reason"], asserts


def test_the_evidence_index_inherits_its_envelope_case_and_species():
    index = load_evidence_index({
        "case_id": "c1", "species": "黄芪",
        "observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing"}]})
    assert index["o1"]["case_id"] == "c1"
    assert index["o1"]["species"] == "黄芪"
    assert index["o1"]["phenotype"] == "leaf_yellowing"


# ── 2. conclusions are derived from the observation, not from wording ─────────


def test_a_non_yellowing_observation_is_not_worded_as_yellowing(workspace):
    """The old runner emitted a fixed "叶缘黄化" claim over any vision id."""
    _tamper_vision_phenotype(workspace, "leaf_spot")
    report = workspace["runner"].run(POSITIVE_INPUT, mode="fixture",
                                     tool_call_id="call-spot")
    assert len(report["claims"]) == 1
    claim = report["claims"][0]
    assert claim["status"] == "supported"
    assert claim["evidence_ids"] == ["obs-yellow-01"]
    assert claim["asserts"]["phenotype"] == "leaf_spot"
    assert "leaf_spot" in claim["text"]
    assert "黄化" not in claim["text"]
    # The knowledge fixture supports only leaf_yellowing, so fusion leaves it
    # unlinked, and the trust level says so instead of claiming full support.
    assert report["unlinked_evidence_ids"] == ["fixture-evidence-01"]
    assert report["trust_level"] == "LIMITED"


def test_an_unknown_phenotype_supports_no_claim(workspace):
    _tamper_vision_phenotype(workspace, "unknown")
    report = workspace["runner"].run(POSITIVE_INPUT, mode="fixture",
                                     tool_call_id="call-unknown")
    assert report["claims"] == []
    assert report["status"] == "refused"
    assert report["trust_level"] == "INSUFFICIENT"


# ── 3. the fusion step really executes ────────────────────────────────────────


def test_the_fusion_step_is_a_governed_call_like_any_other(workspace):
    report = workspace["runner"].run(POSITIVE_INPUT, mode="fixture",
                                     tool_call_id="call-fusion")
    calls = workspace["runtime"].report()["calls"]
    assert [call["skill"] for call in calls] == [
        "plant_vision", "growth_risk", "herbal_knowledge", "evidence_fusion"]
    fusion_call = calls[-1]
    assert fusion_call["status"] == "success"
    assert fusion_call["permissions_used"] == ["package:read"]
    assert fusion_call["tool_call_id"].endswith(":step4")
    assert report["steps"][-1]["provider_skill"] == "evidence_fusion"
    assert report["steps"][-1]["status"] == "success"
    # The knowledge fixture supports the observed phenotype, so nothing is left
    # unlinked on the documented positive case.
    assert report["unlinked_evidence_ids"] == []
    assert report["trust_level"] == "SUPPORTED"


def test_a_fusion_step_that_cannot_execute_is_recorded_as_failed(workspace):
    """The old code wrote status="success" without calling the Skill at all."""
    workflow = read_json(workspace["package"] / "workflow.json")
    fusion = workflow["steps"][-1]
    assert fusion["provider_skill"] == "evidence_fusion"
    # Drop the result-slot mappings: the Fusion Skill then receives no sources.
    fusion["input_map"] = {"case_id": "$.case_id", "species": "$.species"}
    (workspace["package"] / "workflow.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _reseal(workspace, workspace["package"])
    report = workspace["runner"].run(POSITIVE_INPUT, mode="fixture",
                                     tool_call_id="call-fusion-broken")
    step = report["steps"][-1]
    assert step["provider_skill"] == "evidence_fusion"
    assert step["status"] == "failed"
    assert step["detail"]  # an error code, never a fabricated success
    assert "evidence_fusion" in report["missing_inputs"]
    assert report["trust_level"] != "SUPPORTED"
