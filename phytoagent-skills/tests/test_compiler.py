"""Compiler tests: constrained composition, never arbitrary code generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compiler.catalog import CAPABILITIES, SPECIES_SLUGS, capabilities_in_order, union_permissions
from compiler.compiler import (COMPILER_VERSION, EXECUTOR_SOURCE, build_name, build_spec,
                               compile_request, compile_spec, validate_spec)
from compiler.intent import derive_inputs, parse_intent
from compiler.templates.workflow_executor import ALLOWED_PROVIDERS
from sdk.exceptions import CompileError
from sdk.schema import read_json

HUANGQI_REQUEST = ("创建一个黄芪健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因，"
                   "并给出可信证据。")


def test_the_documented_request_compiles_to_the_documented_skill(tmp_path):
    result = compile_request(HUANGQI_REQUEST, tmp_path)
    assert result.spec["name"] == "huangqi-health-assessment"
    assert result.spec["capabilities"] == ["plant_vision", "growth_risk",
                                           "herbal_knowledge", "evidence_fusion"]
    assert result.package_dir == tmp_path / "huangqi-health-assessment"
    assert result.package_dir.is_dir()


def test_the_emitted_executor_is_a_fixed_template_not_generated_code(tmp_path):
    first = compile_request(HUANGQI_REQUEST, tmp_path / "a")
    second = compile_request("创建一个人参健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因。",
                             tmp_path / "b")
    assert (first.package_dir / "skill.py").read_text(encoding="utf-8") == EXECUTOR_SOURCE
    assert (second.package_dir / "skill.py").read_text(encoding="utf-8") == EXECUTOR_SOURCE
    assert first.package_dir.name != second.package_dir.name


def test_the_emitted_executor_declares_no_dangerous_capability():
    for marker in ("subprocess", "os.system", "eval(", "exec(", "socket", "requests",
                   "__import__", "urllib"):
        assert marker not in EXECUTOR_SOURCE, marker


def test_the_compiled_package_reuses_providers_instead_of_copying_them(tmp_path):
    result = compile_request(HUANGQI_REQUEST, tmp_path)
    workflow = read_json(result.package_dir / "workflow.json")
    providers = [step["provider_skill"] for step in workflow["steps"]]
    assert providers == ["plant_vision", "growth_risk", "herbal_knowledge", "evidence_fusion"]
    assert set(providers) <= ALLOWED_PROVIDERS
    # The package declares dependencies; it does not reimplement them.
    assert not (result.package_dir / "plant_vision.py").exists()
    assert (result.package_dir / "skill.py").stat().st_size < 12_000


def test_the_spec_declares_the_fixed_policies_and_denies_network(tmp_path):
    spec = compile_request(HUANGQI_REQUEST, tmp_path).spec
    assert spec["evidence_policy"] == "every_claim_requires_evidence"
    assert spec["refusal_policy"] == "insufficient_evidence"
    assert spec["permissions"]["network"] == "deny"
    assert spec["provenance"]["model_called"] is False
    assert spec["provenance"]["compiler"].endswith(COMPILER_VERSION)


def test_permissions_are_the_least_privilege_union_of_the_selected_capabilities():
    assert union_permissions(["plant_vision", "evidence_fusion"]) == [
        "case_workspace:read", "package:read"]
    assert union_permissions(["herbal_knowledge", "evidence_fusion"]) == [
        "corpus:read", "package:read"]


def test_an_unknown_species_is_refused_rather_than_guessed(tmp_path):
    # "雪莲" is not in the approved catalogue, so no species resolves at all.
    with pytest.raises(CompileError, match="without species"):
        compile_request("创建一个天山雪莲健康研判Skill，根据叶片图片分析异常原因。", tmp_path)


def test_an_empty_request_is_refused(tmp_path):
    with pytest.raises(CompileError, match="non-empty"):
        compile_request("   ", tmp_path)


def test_an_overlong_request_is_refused(tmp_path):
    with pytest.raises(CompileError, match="4000"):
        compile_request("黄芪" * 3000, tmp_path)


def test_a_request_without_any_capability_keyword_is_refused(tmp_path):
    with pytest.raises(CompileError, match="capability"):
        compile_request("请帮我整理一下会议纪要。", tmp_path)


def test_the_compiler_refuses_to_emit_a_name_claiming_official_status(tmp_path):
    # A reserved word must never reach a generated package name, because the name
    # itself would assert an official verification status.
    with pytest.raises(CompileError, match="refusing to guess"):
        build_name("nvidia", "health-assessment")
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["name"] = "nvidia-verified-skill"
    with pytest.raises(CompileError, match="official verification"):
        compile_spec(spec, tmp_path)


def test_a_spec_that_requests_network_access_is_rejected(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["permissions"]["network"] = "allow"
    with pytest.raises(CompileError, match="network"):
        compile_spec(spec, tmp_path)


def test_a_spec_that_drops_the_fusion_step_is_rejected(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["workflow"] = [step for step in spec["workflow"]
                        if step["capability"] != "evidence_fusion"]
    with pytest.raises(CompileError, match="evidence_fusion"):
        compile_spec(spec, tmp_path)


def test_a_spec_with_an_out_of_order_workflow_is_rejected(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["workflow"] = list(reversed(spec["workflow"]))
    with pytest.raises(CompileError, match="capability order"):
        compile_spec(spec, tmp_path)


def test_a_spec_without_negative_evals_is_rejected(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["negative_evals"] = []
    with pytest.raises(CompileError, match="negative eval"):
        compile_spec(spec, tmp_path)


def test_a_spec_with_a_duplicate_negative_eval_id_is_rejected(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["negative_evals"][1]["id"] = spec["negative_evals"][0]["id"]
    with pytest.raises(CompileError, match="unique"):
        compile_spec(spec, tmp_path)


def test_a_spec_whose_input_map_escapes_the_request_is_rejected(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    spec["workflow"][0]["input_map"]["image_path"] = "telemetry.temperature_c"
    with pytest.raises(CompileError, match="request document"):
        compile_spec(spec, tmp_path)


def test_validate_spec_accepts_the_documented_spec_without_writing_files(tmp_path):
    spec = build_spec(parse_intent(HUANGQI_REQUEST))
    before = sorted(path.name for path in tmp_path.iterdir())
    assert validate_spec(spec)["name"] == "huangqi-health-assessment"
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_the_compiled_package_contains_the_documented_layout(tmp_path):
    result = compile_request(HUANGQI_REQUEST, tmp_path)
    expected = {"SKILL.md", "skill-card.md", "skill_spec.json", "workflow.json", "schema.json",
                "evals/evals.json", "examples/request.json", "skill.py", "metadata.json",
                "references/contract.md", "BENCHMARK.md"}
    assert expected <= set(result.files)
    assert json.loads((result.package_dir / "skill_spec.json").read_text(encoding="utf-8"))["name"] \
        == "huangqi-health-assessment"


def test_a_package_for_a_species_without_a_fixture_does_not_claim_a_pass(tmp_path):
    """The providers publish one fixture, for huangqi. Anything else must say so.

    Emitting a "positive" case that the providers are guaranteed to refuse, and
    presenting it as a passing case, would be a fabricated benchmark.
    """
    result = compile_request("创建一个人参健康研判Skill，根据叶片图片、环境参数和知识库"
                             "分析异常原因。", tmp_path)
    cases = read_json(result.package_dir / "evals" / "evals.json")["cases"]
    positive = next(case for case in cases if case["kind"] == "positive")
    assert positive["status"] == "declared_only"
    assert "fixture" in positive["reason"]
    document = read_json(result.package_dir / "evals" / "evals.json")
    assert "NOT RUN" in document["benchmark_status"]


def test_the_huangqi_package_still_declares_a_runnable_positive_case(tmp_path):
    result = compile_request(HUANGQI_REQUEST, tmp_path)
    cases = read_json(result.package_dir / "evals" / "evals.json")["cases"]
    positive = next(case for case in cases if case["kind"] == "positive")
    assert positive["status"] == "declared"
    assert "reason" not in positive


def test_the_compiled_evals_cover_positive_missing_and_negative_cases(tmp_path):
    result = compile_request(HUANGQI_REQUEST, tmp_path)
    cases = read_json(result.package_dir / "evals" / "evals.json")["cases"]
    kinds = {case["kind"] for case in cases}
    assert {"positive", "missing_parameter", "negative_trigger"} <= kinds
    negative_kinds = {case["eval_kind"] for case in cases if case["kind"] == "negative_trigger"}
    assert {"missing_input", "unsupported_claim", "permission_violation"} <= negative_kinds


def test_intent_derivation_matches_the_declared_input_slots():
    intent = parse_intent(HUANGQI_REQUEST)
    assert intent.species == "黄芪"
    assert intent.task == "health-assessment"
    assert derive_inputs(intent.capabilities) == intent.inputs


def test_capability_order_is_the_audited_execution_order():
    assert capabilities_in_order(["evidence_fusion", "herbal_knowledge", "plant_vision"]) == \
        ["plant_vision", "herbal_knowledge", "evidence_fusion"]
    assert capabilities_in_order(["plant_vision", "plant_vision"]) == ["plant_vision"]


def test_every_catalog_capability_has_a_provider_and_a_failure_mode():
    for name, entry in CAPABILITIES.items():
        assert entry["provider_skill"]
        assert entry["failure_mode"]
        assert entry["order"] > 0
        assert entry["description"]


def test_every_species_alias_resolves_to_an_approved_slug():
    for alias, species in SPECIES_SLUGS.items():
        assert species


def test_the_compiled_schema_matches_the_provider_contracts(tmp_path):
    """The generated request schema must accept exactly what providers accept."""
    result = compile_request(HUANGQI_REQUEST, tmp_path)
    schema = read_json(result.package_dir / "schema.json")
    assert schema["input"]["properties"]["telemetry"]["required"] == [
        "temperature_c", "relative_humidity_pct", "soil_ph", "npk"]
    # Provider-specific keys are mapped from the request's own fields, so the
    # generated schema uses the request names, not the provider names.
    request_fields = set(schema["input"]["properties"])
    assert {"case_id", "species", "image", "telemetry", "question"} <= request_fields
    assert "image_path" not in request_fields
    assert "measurements" not in request_fields
    # The generated request schema must stay inside what each provider accepts.
    workflow = read_json(result.package_dir / "workflow.json")
    for step in workflow["steps"]:
        for provider_field, source in step["input_map"].items():
            assert source.startswith("$."), (step["id"], provider_field)
