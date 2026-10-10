"""AgentShield Runtime: the interception an Agent cannot route around."""

from __future__ import annotations

from pathlib import Path

import pytest

from demo.fixture_workspace import fixture_registry
from registry.signer import generate_keypair, sign_package
from runtime.executor import SkillExecutor
from runtime.shield import (TRUST_INSUFFICIENT, TRUST_LIMITED, TRUST_SUPPORTED, BudgetExceeded,
                            ClaimAuditor, PermissionViolation, ShieldError, ShieldRuntime,
                            TraceError, call_or_failure, collect_evidence_ids,
                            load_evidence_index)
from sdk.exceptions import RegistryError
from sdk.schema import read_json

FIXTURE_INPUT = {
    "case_id": "demo-huangqi-001", "species": "黄芪",
    "image_path": "fixture://huangqi-leaf-01",
}


@pytest.fixture(scope="module")
def domain_registry():
    with fixture_registry() as registry:
        yield registry


@pytest.fixture
def runtime(domain_registry):
    return ShieldRuntime(domain_registry, trace_id="trace-test-001")


# ── permission interception ───────────────────────────────────────────────────


def test_an_undeclared_permission_is_blocked_before_execution(runtime):
    with pytest.raises(PermissionViolation, match="undeclared filesystem permission"):
        runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-1",
                     requested_permissions=["corpus:read"])
    assert runtime.report()["blocked"][0]["status"] == "blocked"
    assert runtime.report()["blocked"][0]["permissions_used"] == ["corpus:read"]


def test_a_network_request_is_blocked(runtime):
    with pytest.raises(PermissionViolation, match="network"):
        runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-1",
                     requested_permissions=["network"])


def test_a_declared_permission_is_allowed(runtime):
    result = runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-1",
                          requested_permissions=["case_workspace:read"])
    assert result["status"] == "success"
    assert result["data"]["observations"][0]["observation_id"] == "obs-yellow-01"


def test_a_call_without_a_tool_call_id_is_refused(runtime):
    with pytest.raises(TraceError, match="tool_call_id"):
        runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="   ")


def test_the_call_budget_is_enforced(runtime):
    runtime.max_calls = 2
    for index in range(2):
        runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id=f"call-{index}")
    with pytest.raises(BudgetExceeded, match="budget"):
        runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-3")


def test_an_unknown_skill_is_refused(runtime):
    """An unknown package is a governance refusal: recorded and structured.

    RegistryError is not a ShieldError, so letting it propagate would both skip
    the audit log and abort a Harness loop that only handles ShieldError.
    """
    with pytest.raises(ShieldError, match="registry verification") as error:
        runtime.call("not_a_skill", {}, tool_call_id="call-1")
    assert isinstance(error.value.__cause__, RegistryError)
    blocked = runtime.report()["blocked"]
    assert blocked
    assert blocked[-1]["skill"] == "not_a_skill"
    assert blocked[-1]["detail"] == "RegistryError"


def test_a_registry_verification_failure_is_recorded_and_structured():
    """A changed package or invalid signature must not bypass the audit trail."""

    class FailingRegistry:
        def verify_entry(self, name):
            raise RegistryError("Unknown Skill: plant_vision")

    runtime = ShieldRuntime(FailingRegistry())
    with pytest.raises(ShieldError) as error:
        runtime.call("plant_vision", {}, tool_call_id="call-1")
    assert isinstance(error.value.__cause__, RegistryError)
    blocked = runtime.report()["blocked"]
    assert len(blocked) == 1
    assert blocked[0]["skill"] == "plant_vision"
    assert blocked[0]["detail"] == "RegistryError"
    assert "RegistryError" in str(error.value)


def test_skipping_the_audit_skill_does_not_remove_interception(runtime):
    """The governance point is the middleware, not a callable audit Skill."""
    # An Agent that never calls agentshield_audit is still intercepted here.
    with pytest.raises(PermissionViolation):
        runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-1",
                     requested_permissions=["filesystem:write"])
    assert runtime.report()["blocked"]


# ── tracing ───────────────────────────────────────────────────────────────────


def test_every_call_is_traced_with_ids_and_the_manifest_hash(runtime):
    runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-1")
    record = runtime.report()["calls"][0]
    assert record["trace_id"] == "trace-test-001"
    assert record["tool_call_id"] == "call-1"
    assert len(record["manifest_sha256"]) == 64
    assert record["status"] == "success"
    assert record["duration_ms"] >= 0


def test_a_failed_call_is_traced_with_its_error_code(runtime):
    result = runtime.call("plant_vision", {**FIXTURE_INPUT, "case_id": "other-case"},
                          tool_call_id="call-2")
    assert result["status"] == "failed"
    record = runtime.report()["calls"][-1]
    assert record["status"] == "failed"
    assert record["detail"] == "FixtureMismatchError"


def test_the_trace_collects_evidence_ids(runtime):
    runtime.call("plant_vision", FIXTURE_INPUT, tool_call_id="call-1")
    assert runtime.report()["calls"][-1]["evidence_ids"] == ["obs-yellow-01"]


def test_the_runtime_report_states_what_it_did_not_do(runtime):
    report = runtime.report()
    assert report["agent_model_called"] is False
    assert report["dgx_hardware_used"] is False
    assert report["nvidia_verified"] is False


# ── claim-evidence audit ──────────────────────────────────────────────────────


def _evidence(*ids: str) -> dict:
    """A minimal evidence index: id -> the record that produced it."""
    return {evidence_id: {"evidence_id": evidence_id} for evidence_id in ids}


def test_a_claim_naming_a_specific_cause_is_refused():
    """Naming a pathogen is a diagnosis; no evidence can carry one."""
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    for text in ("病原是黄萎病菌", "病因是缺铁", "叶片感染了锈病",
                 "该病害由真菌引起"):
        verdict = auditor.audit({"claims": [{"text": text, "evidence_ids": ["e1"]}]})
        assert verdict["trust_level"] == TRUST_INSUFFICIENT, text
        assert verdict["claims"][0]["status"] == "refused", text
        assert "names a specific cause" in verdict["claims"][0]["reason"], text


def test_an_audit_with_a_check_that_never_ran_is_incomplete_not_pass():
    """A check that did not run cannot support a PASS verdict."""
    from demo.fixture_workspace import fixture_registry
    from runtime.executor import SkillExecutor
    with fixture_registry() as registry:
        executor = SkillExecutor(registry)
        response = executor.call("agentshield_audit", {
            "audit_id": "a", "stage": "pre_registration",
            "subject": {
                "kind": "skill_package",
                "manifest": {"name": "evidence_fusion", "version": "0.2.0",
                             "permissions": {"filesystem": ["package:read"], "network": "deny"},
                             "files": {"SKILL.md": "a", "skill.py": "b",
                                       "evals/evals.json": "c"}},
                "evals": {"contract_cases": [
                    {"id": "valid_fixture", "kind": "positive"},
                    {"id": "reject_unmatched_fixture", "kind": "negative"}]},
            },
        }, mode="fixture", tool_call_id="audit-incomplete")
        assert response["status"] == "success"
        data = response["data"]
        assert data["verdict"] == "INCOMPLETE"
        # The trust level is gated on completeness too, not just the verdict.
        assert data["trust_level"] == "LIMITED"
        assert any("未执行" in item for item in data["limitations"])
        trace_check = next(c for c in data["checks"] if c["name"] == "trace_complete")
        assert trace_check["status"] == "not_run"


def test_an_audit_refuses_a_report_citing_an_unresolvable_evidence_id():
    from demo.fixture_workspace import fixture_registry
    from runtime.executor import SkillExecutor
    with fixture_registry() as registry:
        executor = SkillExecutor(registry)
        response = executor.call("agentshield_audit", {
            "audit_id": "a", "stage": "pre_output",
            "subject": {"kind": "report",
                        "report": {"case_id": "demo-huangqi-001", "species": "黄芪",
                                   "claims": [{"text": "叶缘存在黄化区域",
                                               "evidence_ids": ["obs-yellow-01"]}]},
                        "trace": [{"tool_call_id": "c1", "skill": "plant_vision"}]},
        }, mode="fixture", tool_call_id="audit-unknown-id")
        assert response["status"] == "success"
        data = response["data"]
        assert data["trust_level"] == "INSUFFICIENT"
        check = next(c for c in data["checks"] if c["name"] == "evidence_verifiable")
        assert check["status"] == "failed"
        assert "obs-yellow-01" in check["detail"]


def test_a_fabricated_evidence_id_never_reaches_supported():
    """Fail closed: an id the run cannot account for is unverifiable.

    With no index supplied the auditor must still refuse, otherwise a claim
    citing an invented id would come back SUPPORTED.
    """
    auditor = ClaimAuditor()
    verdict = auditor.audit({"claims": [{"text": "叶缘黄化", "evidence_ids": ["made-up-id"]}]})
    assert verdict["trust_level"] == TRUST_INSUFFICIENT
    assert verdict["claims"][0]["status"] == "refused"
    assert "unverifiable evidence id" in verdict["claims"][0]["reason"]


def test_a_regex_diagnosis_marker_actually_matches():
    """A pattern marker must match text, not its own source."""
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    verdict = auditor.audit({"claims": [{"text": "黄化由土壤缺铁引起",
                                         "evidence_ids": ["e1"]}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "names a specific cause" in verdict["claims"][0]["reason"]


def test_an_observational_claim_is_not_treated_as_a_diagnosis():
    """The marker must not swallow ordinary observation wording."""
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    verdict = auditor.audit({"claims": [{"text": "叶缘存在黄化区域，原因待复核",
                                         "evidence_ids": ["e1"]}]})
    assert verdict["claims"][0]["status"] == "supported"


def test_a_claim_without_an_evidence_id_is_refused():
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    verdict = auditor.audit({"claims": [{"text": "叶缘黄化", "evidence_ids": []}]})
    assert verdict["trust_level"] == TRUST_INSUFFICIENT
    assert verdict["claims"][0]["status"] == "refused"
    assert "every_claim_requires_evidence" in verdict["claims"][0]["reason"]


def test_an_over_certain_claim_is_refused():
    auditor = ClaimAuditor()
    verdict = auditor.audit({"claims": [{"text": "病原已确定，是黄萎病菌", "evidence_ids": ["e1"]}]})
    assert verdict["trust_level"] == TRUST_INSUFFICIENT
    assert "over-certain" in verdict["claims"][0]["reason"]


def test_an_unknown_evidence_id_is_refused():
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    verdict = auditor.audit({"claims": [{"text": "叶缘黄化", "evidence_ids": ["e-unknown"]}]})
    assert verdict["trust_level"] == TRUST_INSUFFICIENT
    assert "unverifiable evidence id" in verdict["claims"][0]["reason"]


def test_a_supported_claim_with_a_verified_id_is_accepted():
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    verdict = auditor.audit({"claims": [{"text": "叶缘黄化", "evidence_ids": ["e1"]}]})
    assert verdict["trust_level"] == TRUST_SUPPORTED
    assert verdict["claims"][0]["status"] == "supported"
    assert verdict["evidence_ids"] == ["e1"]


def test_a_missing_source_downgrades_to_limited():
    auditor = ClaimAuditor(available_evidence=_evidence("e1"))
    verdict = auditor.audit({"claims": [{"text": "叶缘黄化", "evidence_ids": ["e1"]}],
                             "missing_inputs": ["environment"]})
    assert verdict["trust_level"] == TRUST_LIMITED


def test_a_report_with_no_supported_claim_is_insufficient():
    auditor = ClaimAuditor()
    verdict = auditor.audit({"claims": []})
    assert verdict["trust_level"] == TRUST_INSUFFICIENT


def test_a_malformed_claim_is_recorded_as_a_violation():
    auditor = ClaimAuditor()
    verdict = auditor.audit({"claims": ["not a claim object"]})
    assert verdict["trust_level"] == TRUST_INSUFFICIENT
    assert verdict["violations"]


def test_a_report_must_declare_a_claims_array():
    auditor = ClaimAuditor()
    with pytest.raises(ShieldError, match="claims array"):
        auditor.audit({"case_id": "x"})


def test_evidence_ids_are_collected_from_every_provider_shape():
    index = load_evidence_index(
        {"observations": [{"observation_id": "o1"}]},
        {"factors": [{"factor_id": "f1"}]},
        {"evidence": [{"evidence_id": "e1"}]},
    )
    assert set(index) == {"o1", "f1", "e1"}


def test_collect_evidence_ids_deduplicates_and_ignores_non_dicts():
    assert collect_evidence_ids({"evidence": [{"evidence_id": "e1"}, {"evidence_id": "e1"},
                                              "junk", {}]}) == ["e1"]
    assert collect_evidence_ids(None) == []
    assert collect_evidence_ids("nonsense") == []


def test_the_plain_executor_is_still_available(domain_registry):
    """ShieldRuntime wraps SkillExecutor; the plain path keeps working."""
    executor = SkillExecutor(domain_registry)
    assert executor.call("plant_vision", FIXTURE_INPUT, mode="fixture",
                         tool_call_id="plain-1")["status"] == "success"


def test_a_middleware_rejection_becomes_a_failed_call_not_a_crash(runtime):
    """Batch drivers survive one blocked call; the rejection is still traced."""
    result = call_or_failure(runtime, "plant_vision", FIXTURE_INPUT, mode="fixture",
                             tool_call_id="blocked-1", requested_permissions=["network"])
    assert result["status"] == "failed"
    assert result["error"]["code"] == "PermissionViolation"
    assert runtime.report()["blocked"]
    assert runtime.report()["blocked"][0]["tool_call_id"] == "blocked-1"


# ── audit completeness: door rejections are recorded too ─────────────────────


def test_a_budget_rejection_is_recorded_in_the_audit_log_before_raising():
    """A call refused at the door is still an attempt and must be traceable.

    Without the record the Shield's audit log and the caller's trace cannot be
    reconciled: the harness knows a call was blocked, the governance report
    does not.
    """
    runtime = ShieldRuntime(registry=None, max_calls=0)
    with pytest.raises(BudgetExceeded, match="budget"):
        runtime.call("plant_vision", {}, tool_call_id="call-blocked")
    blocked = runtime.report()["blocked"]
    assert len(blocked) == 1
    assert blocked[0]["tool_call_id"] == "call-blocked"
    assert blocked[0]["skill"] == "plant_vision"
    assert blocked[0]["status"] == "blocked"
    assert blocked[0]["detail"] == "BudgetExceeded"


def test_a_missing_tool_call_id_is_recorded_in_the_audit_log_before_raising():
    runtime = ShieldRuntime(registry=None)
    with pytest.raises(TraceError, match="tool_call_id"):
        runtime.call("plant_vision", {}, tool_call_id="   ")
    blocked = runtime.report()["blocked"]
    assert len(blocked) == 1
    assert blocked[0]["detail"] == "TraceError"
    assert blocked[0]["skill"] == "plant_vision"


# ── evidence index conflicts fail closed ─────────────────────────────────────


def test_conflicting_records_for_one_evidence_id_are_refused():
    """Two different records claiming one id: neither is trustworthy.

    Resolving by arrival order would let whichever record was indexed last
    decide what a citation means.
    """
    index = load_evidence_index(
        {"observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing"}]},
        {"observations": [{"observation_id": "o1", "phenotype": "leaf_spot"}]})
    assert index["o1"] == {"conflicted": True}
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{"text": "叶缘黄化", "evidence_ids": ["o1"]}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "unverifiable" in verdict["claims"][0]["reason"]


def test_identical_duplicate_records_do_not_conflict():
    index = load_evidence_index(
        {"case_id": "c1", "observations": [{"observation_id": "o1",
                                            "phenotype": "leaf_yellowing"}]},
        {"case_id": "c1", "observations": [{"observation_id": "o1",
                                            "phenotype": "leaf_yellowing"}]})
    assert index["o1"]["phenotype"] == "leaf_yellowing"
    assert index["o1"]["case_id"] == "c1"


def test_a_child_record_contradicting_its_envelope_poisons_the_id():
    """A record whose own case contradicts its envelope has no provenance."""
    index = load_evidence_index({
        "case_id": "c1", "species": "黄芪",
        "observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing",
                          "case_id": "other-case"}]})
    assert index["o1"] == {"conflicted": True}
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{"text": "观察结论", "evidence_ids": ["o1"],
                                         "asserts": {"phenotype": "leaf_yellowing"}}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "unverifiable" in verdict["claims"][0]["reason"]


def test_a_child_record_agreeing_with_its_envelope_is_kept():
    index = load_evidence_index({
        "case_id": "c1", "species": "黄芪",
        "observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing",
                          "case_id": "c1", "species": "黄芪"}]})
    assert index["o1"]["phenotype"] == "leaf_yellowing"
    assert index["o1"]["case_id"] == "c1"


def test_an_assertion_split_across_records_is_refused():
    """A phenotype from one record and a case id from another is not a context.

    The per-key check would let a claim assemble its assertion from records
    that never shared a case; one record must establish every asserted key.
    """
    index = load_evidence_index(
        {"observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing"}]},
        {"case_id": "case-b", "factors": [{"factor_id": "f1", "metric": "soil_ph"}]})
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "观察结论", "evidence_ids": ["o1", "f1"],
        "asserts": {"phenotype": "leaf_yellowing", "case_id": "case-b"}}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "no single cited record" in verdict["claims"][0]["reason"]


def test_one_record_establishing_every_asserted_key_is_supported():
    index = load_evidence_index({
        "case_id": "c1", "species": "黄芪",
        "observations": [{"observation_id": "o1", "phenotype": "leaf_yellowing"}]})
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "观察结论", "evidence_ids": ["o1"],
        "asserts": {"phenotype": "leaf_yellowing", "case_id": "c1", "species": "黄芪"}}]})
    assert verdict["claims"][0]["status"] == "supported"
    assert verdict["trust_level"] == TRUST_SUPPORTED


def test_a_citation_from_another_case_blocks_an_otherwise_supported_claim():
    """One record's support does not license citations from other contexts.

    The first record establishes every asserted key on its own; the second
    belongs to a different case and must not ride along on that support.
    """
    index = load_evidence_index(
        {"case_id": "case-a", "observations": [
            {"observation_id": "o1", "phenotype": "leaf_yellowing"}]},
        {"case_id": "case-b", "observations": [
            {"observation_id": "o2", "phenotype": "leaf_yellowing"}]})
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "观察结论", "evidence_ids": ["o1", "o2"],
        "asserts": {"phenotype": "leaf_yellowing", "case_id": "case-a"}}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "another case_id" in verdict["claims"][0]["reason"]
    assert verdict["trust_level"] == TRUST_INSUFFICIENT


def test_citing_only_the_matching_record_is_supported():
    index = load_evidence_index(
        {"case_id": "case-a", "observations": [
            {"observation_id": "o1", "phenotype": "leaf_yellowing"}]},
        {"case_id": "case-b", "observations": [
            {"observation_id": "o2", "phenotype": "leaf_yellowing"}]})
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "观察结论", "evidence_ids": ["o1"],
        "asserts": {"phenotype": "leaf_yellowing", "case_id": "case-a"}}]})
    assert verdict["claims"][0]["status"] == "supported"


def test_a_citation_from_another_species_blocks_the_claim():
    index = load_evidence_index(
        {"case_id": "c1", "species": "黄芪", "observations": [
            {"observation_id": "o1", "phenotype": "leaf_yellowing"}]},
        {"case_id": "c1", "species": "人参", "observations": [
            {"observation_id": "o2", "phenotype": "leaf_yellowing"}]})
    auditor = ClaimAuditor(available_evidence=index)
    verdict = auditor.audit({"claims": [{
        "text": "观察结论", "evidence_ids": ["o1", "o2"],
        "asserts": {"phenotype": "leaf_yellowing", "species": "黄芪"}}]})
    assert verdict["claims"][0]["status"] == "refused"
    assert "another species" in verdict["claims"][0]["reason"]
