"""Demo V1: the honesty contract of the interactive demo.

The demo's job is to let a visitor operate the real pipeline and *tell apart*
what they are looking at. These tests pin that contract: every step carries a
badge with a reason, a run never silently changes mode, and the cases show
what they claim to show.
"""

from __future__ import annotations

import pytest

from demo.v1_cases import DEMO_CASES
from demo.v1_pipeline import (BADGE_DETERMINISTIC, BADGE_REAL, BADGE_REPLAY,
                              DemoWorkspace, backend_status, cases_payload,
                              run_case)


@pytest.fixture
def workspace(tmp_path):
    return DemoWorkspace(tmp_path / "demo-workspace").materialise()


def test_every_case_runs_and_every_step_is_marked(workspace):
    for case in DEMO_CASES:
        result = run_case(case["demo_id"], mode="fixture", workspace=workspace)
        assert result["steps"], f"{case['demo_id']} produced no steps"
        for step in result["steps"]:
            assert step["badge"] in (BADGE_REPLAY, BADGE_DETERMINISTIC, BADGE_REAL)
            assert step["badge_reason"].strip(), (
                f"{case['demo_id']}/{step['provider']} has no badge reason")
        assert result["audit"] is not None


def test_a_fixture_run_is_deterministic(workspace):
    """A visitor who runs the same case twice must see the same answer."""
    first = run_case("fixture-positive", mode="fixture", workspace=workspace)
    second = run_case("fixture-positive", mode="fixture", workspace=workspace)
    assert first["audit"] == second["audit"]
    assert first["report"]["text"] == second["report"]["text"]
    assert first["any_real_inference"] is False
    assert all(step["badge"] == BADGE_REPLAY for step in first["steps"])


def test_the_positive_case_shows_a_supported_claim_with_evidence(workspace):
    result = run_case("fixture-positive", mode="fixture", workspace=workspace)
    audit = result["audit"]
    assert audit["status"] == "success"
    assert audit["trust_level"] == "SUPPORTED"
    assert audit["claims"]
    assert all(claim["evidence_ids"] for claim in audit["claims"])


def test_the_negative_case_refuses_to_guess(workspace):
    """A blurry image plus pressure to guess must produce a refusal, not a cause."""
    result = run_case("negative-pressure-to-guess", mode="fixture",
                      workspace=workspace)
    audit = result["audit"]
    assert audit["status"] == "refused"
    assert audit["trust_level"] == "INSUFFICIENT"
    assert not audit["claims"]
    assert audit["refused_claims"] or audit["limitations"]


def test_real_mode_without_a_backend_fails_loudly_instead_of_falling_back(
        workspace, monkeypatch):
    """The standing rule: no silent fallback to fixture. The visitor is told
    exactly what is missing, and no run happens."""
    monkeypatch.delenv("PHYTO_VISION_BACKEND", raising=False)
    result = run_case("fixture-positive", mode="live", workspace=workspace)
    assert result["error"]["code"] == "BackendUnavailable"
    assert "PHYTO_VISION_BACKEND" in result["error"]["message"]
    assert result["steps"] == [] and result["audit"] is None
    assert result["any_real_inference"] is False


def test_the_replay_case_replays_a_recorded_real_observation(workspace, monkeypatch):
    """live mode against the replay backend: a real recorded observation, served
    offline. The vision leg is a replay; the knowledge leg is deterministic
    retrieval over the real corpus index; nothing claims to be inference."""
    from pathlib import Path

    from demo.v1_pipeline import REPLAY_SOURCE
    if not REPLAY_SOURCE.is_file():
        pytest.skip("the recorded DGX runs are not present in this checkout")
    monkeypatch.setenv("PHYTO_VISION_BACKEND", f"replay@{REPLAY_SOURCE}")
    result = run_case("replay-real-quality-gate", mode="live", workspace=workspace)
    badges = {step["provider"]: step["badge"] for step in result["steps"]}
    assert badges["plant_vision"] == BADGE_REPLAY
    assert "vision-runs.json" in next(
        step["badge_reason"] for step in result["steps"]
        if step["provider"] == "plant_vision")
    assert badges["herbal_knowledge"] == BADGE_DETERMINISTIC
    assert result["any_real_inference"] is False
    # The recorded observation says the image is not usable, so the quality gate
    # refuses — the honest outcome on this record, and the point of the case.
    assert result["audit"]["status"] == "refused"


def test_a_real_backend_would_be_marked_real(workspace, monkeypatch):
    """The marking follows the configured backend, not the requested mode."""
    monkeypatch.setenv("PHYTO_VISION_BACKEND", "local_cpu")
    # No run happens (there is no ollama here); the marking logic is checked
    # directly, because that is what a visitor reads before pressing run.
    from demo.v1_pipeline import _step_marking

    backend = backend_status()
    assert backend["configured"] and backend["kind"] == "local_cpu"
    badge, reason = _step_marking("live", backend)
    assert badge == BADGE_REAL
    assert "local_cpu" in reason


def test_the_cases_payload_describes_every_case():
    payload = cases_payload()
    assert len(payload["cases"]) == len(DEMO_CASES)
    for case in payload["cases"]:
        assert case["demo_id"] and case["title"] and case["blurb"]
    assert "backend" in payload


# ── the API contract, without a socket ──────────────────────────────────────


def test_the_api_contract_validates_before_running():
    from demo.v1_server import run_request

    assert run_request({"mode": "fixture"})[0] == 400
    assert run_request({"demo_id": "fixture-positive", "mode": "turbo"})[0] == 400
    assert run_request("not a dict")[0] == 400
    status, payload = run_request({"demo_id": "no-such-case"})
    assert status == 404 and payload["error"] == "unknown case"


def test_the_api_runs_a_case_and_marks_every_step(workspace, monkeypatch):
    from demo.v1_server import run_request

    # The server would use the persistent workspace; the test injects its own.
    import demo.v1_pipeline as pipeline
    original = pipeline.DemoWorkspace
    pipeline.DemoWorkspace = lambda root=None: workspace
    try:
        monkeypatch.delenv("PHYTO_VISION_BACKEND", raising=False)
        status, payload = run_request({"demo_id": "fixture-positive",
                                       "mode": "fixture"})
    finally:
        pipeline.DemoWorkspace = original
    assert status == 200
    assert payload["audit"]["status"] == "success"
    assert all(step["badge_reason"] for step in payload["steps"])
    assert payload["any_real_inference"] is False


# ── the API contract, without a socket ──────────────────────────────────────


def test_the_api_contract_validates_before_running():
    from demo.v1_server import run_request

    assert run_request({"mode": "fixture"})[0] == 400
    assert run_request({"demo_id": "fixture-positive", "mode": "turbo"})[0] == 400
    assert run_request("not a dict")[0] == 400
    status, payload = run_request({"demo_id": "no-such-case"})
    assert status == 404 and payload["error"] == "unknown case"


def test_the_api_runs_a_case_and_marks_every_step(workspace, monkeypatch):
    from demo.v1_server import run_request

    # The server would use the persistent workspace; the test injects its own.
    import demo.v1_pipeline as pipeline
    original = pipeline.DemoWorkspace
    pipeline.DemoWorkspace = lambda root=None: workspace
    try:
        monkeypatch.delenv("PHYTO_VISION_BACKEND", raising=False)
        status, payload = run_request({"demo_id": "fixture-positive",
                                       "mode": "fixture"})
    finally:
        pipeline.DemoWorkspace = original
    assert status == 200
    assert payload["audit"]["status"] == "success"
    assert all(step["badge_reason"] for step in payload["steps"])
    assert payload["any_real_inference"] is False
