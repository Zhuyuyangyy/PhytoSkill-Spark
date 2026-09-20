from copy import deepcopy
import json
from pathlib import Path

import pytest

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES, fixture_registry
from demo.run_demo import run_demo
from evals.run_contracts import evaluate
from runtime.executor import SkillExecutor
from sdk.exceptions import ContractError, FixtureMismatchError, UnsupportedModeError
from sdk.schema import read_json


@pytest.fixture(scope="module")
def domain_registry():
    with fixture_registry() as registry:
        yield registry


def fusion_request():
    return read_json(PROJECT_ROOT / "skills/evidence_fusion/examples/request.json")


def test_independent_discovery_and_progressive_loading(domain_registry):
    catalog = domain_registry.catalog
    assert {item["name"] for item in catalog} == set(SKILL_NAMES)
    assert all(set(item) == {"name", "description"} for item in catalog)
    selected = domain_registry.load_skill("plant_vision")
    assert selected["tool"]["function"]["parameters"]["required"] == ["case_id", "species", "image_path"]
    assert selected["verification"]["signature"]["status"] == "passed"
    assert selected["manifest_sha256"] == domain_registry.get("plant_vision")["manifest"]["manifest_sha256"]
    assert len(selected["instructions"]) > 0


@pytest.mark.parametrize("name", SKILL_NAMES[:3])
def test_fixture_cannot_answer_real_requests_or_live_calls(domain_registry, name):
    executor = SkillExecutor(domain_registry)
    payload = read_json(PROJECT_ROOT / "skills" / name / "examples/request.json")
    payload["case_id"] = "user-real-case"
    with pytest.raises(FixtureMismatchError):
        executor.execute(name, payload)
    with pytest.raises(UnsupportedModeError):
        executor.execute(name, payload, mode="live")


def test_composition_contract_snapshots_match_upstream_outputs():
    schema = read_json(PROJECT_ROOT / "skills/evidence_fusion/schema.json")
    for field, name in {"vision": "plant_vision", "environment": "growth_risk", "knowledge": "herbal_knowledge"}.items():
        upstream = read_json(PROJECT_ROOT / "skills" / name / "schema.json")
        assert schema["input"]["$defs"][field] == upstream["output"]


def test_fusion_links_only_supported_evidence_and_preserves_unlinked_items(domain_registry):
    request = fusion_request()
    distractor = deepcopy(request["knowledge"]["evidence"][0])
    distractor["evidence_id"] = "irrelevant-evidence"
    distractor["supports_phenotypes"] = ["leaf_spot"]
    request["knowledge"]["evidence"].append(distractor)
    original = deepcopy(request)
    result = SkillExecutor(domain_registry).execute("evidence_fusion", request)
    assert request == original
    assert result["completeness"] == "complete"
    assert result["conclusion_strength"] == "synthetic_only"
    assert result["evidence_chain"][0]["observation_id"] == "obs-yellow-01"
    assert result["evidence_chain"][0]["knowledge_evidence_ids"] == ["fixture-evidence-01"]
    assert result["evidence_chain"][0]["relationship"] == "co_occurrence_only"
    assert result["unlinked_evidence_ids"] == ["irrelevant-evidence"]
    assert result["source_ids"]["environment"] == ["factor-temperature-01", "factor-ph-01"]


@pytest.mark.parametrize("field", ["case_id", "species"])
def test_fusion_rejects_cross_case_or_species(domain_registry, field):
    request = fusion_request()
    request["knowledge"][field] = "different"
    with pytest.raises(ContractError, match="mismatch"):
        SkillExecutor(domain_registry).execute("evidence_fusion", request)


@pytest.mark.parametrize("source,items", [("vision", "observations"), ("environment", "factors"), ("knowledge", "evidence")])
def test_fusion_rejects_ambiguous_duplicate_ids(domain_registry, source, items):
    request = fusion_request()
    request[source][items].append(deepcopy(request[source][items][0]))
    with pytest.raises(ContractError, match="Duplicate"):
        SkillExecutor(domain_registry).execute("evidence_fusion", request)


def test_fusion_rejects_inverted_boxes(domain_registry):
    request = fusion_request()
    request["vision"]["observations"][0]["region"]["bbox_normalized"] = [0.8, 0.2, 0.1, 0.7]
    with pytest.raises(ContractError, match="Bounding box"):
        SkillExecutor(domain_registry).execute("evidence_fusion", request)


def test_missing_sources_never_become_synthetic_replacements(domain_registry):
    request = fusion_request()
    request.update(vision=None, environment=None, knowledge=None)
    result = SkillExecutor(domain_registry).execute("evidence_fusion", request)
    assert result["completeness"] == "insufficient"
    assert result["missing_inputs"] == ["vision", "environment", "knowledge"]
    assert result["evidence_chain"] == []
    assert result["source_ids"] == {"vision": [], "environment": [], "knowledge": []}


def test_demo_failure_degrades_and_keeps_call_identity():
    report = run_demo(simulate_failure="herbal_knowledge")
    assert report["agent_model_called"] is False
    assert report["scheduler"] == "scripted_fixture"
    calls = [item for item in report["trace"] if item["event"] == "tool_result"]
    assert [item["tool_call_id"] for item in calls] == [f"fixture-call-{i}" for i in range(1, 5)]
    assert calls[2]["status"] == "failed" and calls[2]["data"] is None
    assert report["result"]["missing_inputs"] == ["knowledge"]
    assert report["result"]["completeness"] == "partial"
    assert report["result"]["evidence_chain"][0]["knowledge_evidence_ids"] == []


def test_embedded_contract_evaluations_pass_without_claiming_agent_benchmarks():
    report = evaluate()
    # 19 contract cases across the four professional Skills, 6 audit cases and
    # 3 corpus-mode cases.
    assert report["total"] == report["passed"] == 28
    assert report["failed"] == 0
    assert report["agent_ab"]["status"] == "not_run"
    assert report["dgx_hardware_used"] is False
    for skill in report["skills"]:
        assert len(skill["manifest_sha256"]) == 64
        assert skill["agent_behavior_cases_supplied"] >= 3


def test_skill_scripts_are_usable_from_copied_package(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    source = PROJECT_ROOT / "skills/plant_vision"
    copy = tmp_path / "plant-vision"
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Model an installed SDK via its import path; the script may not rely on
    # its position inside our checkout. Wheel validation runs separately.
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONUTF8": "1"}
    process = subprocess.run([sys.executable, "scripts/run.py", "--input", "examples/request.json",
                              "--mode", "fixture", "--allow-unsigned"], cwd=copy, env=environment,
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["provenance"]["skill"] == "plant_vision"
    assert result["provenance"]["model_called"] is False
