"""Four-Skill fixture composition; explicitly scripted, with no Agent model."""

import argparse
import json
from pathlib import Path

from demo.fixture_workspace import fixture_registry
from runtime.executor import SkillExecutor
from sdk.schema import package_file, read_json


def run_demo(*, simulate_failure: str | None = None) -> dict:
    first_skills = ("plant_vision", "growth_risk", "herbal_knowledge")
    if simulate_failure is not None and simulate_failure not in first_skills:
        raise ValueError("Failure simulation must target one of the first three Skills")
    with fixture_registry() as registry:
        executor = SkillExecutor(registry)
        trace = [{"event": "catalog_discovered", "catalog": registry.catalog}]
        outputs = {}
        for index, name in enumerate(first_skills, start=1):
            loaded = registry.load_skill(name)
            trace.append({"event": "skill_loaded", "name": name,
                          "manifest_sha256": loaded["manifest_sha256"]})
            package = Path(registry.get(name)["package_dir"])
            payload = read_json(package_file(package, "fixture.json"))["input"]
            if simulate_failure == name:
                payload["case_id"] = "intentionally-unmatched-fixture"
            call = executor.call(name, payload, mode="fixture", tool_call_id=f"fixture-call-{index}")
            trace.append({"event": "tool_result", **call})
            outputs[name] = call["data"]
        loaded = registry.load_skill("evidence_fusion")
        trace.append({"event": "skill_loaded", "name": "evidence_fusion",
                      "manifest_sha256": loaded["manifest_sha256"]})
        call = executor.call("evidence_fusion", {
            "case_id": "demo-huangqi-001", "species": "黄芪",
            "vision": outputs["plant_vision"], "environment": outputs["growth_risk"],
            "knowledge": outputs["herbal_knowledge"],
        }, mode="fixture", tool_call_id="fixture-call-4")
        trace.append({"event": "tool_result", **call})
        if call["status"] != "success":
            raise RuntimeError("Fusion failed; inspect the Skill contracts")
        return {
            "scope": "four_skill_fixture_composition", "mode": "fixture",
            "scheduler": "scripted_fixture", "agent_model_called": False,
            "dgx_hardware_used": False, "simulated_failure": simulate_failure,
            "trust": {"signature_format": "project_ed25519", "demo_key_generated_this_run": True,
                      "nvidia_verified": False},
            "trace": trace, "result": call["data"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate-failure", choices=["plant_vision", "growth_risk", "herbal_knowledge"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/four-skills-demo.json"))
    args = parser.parse_args()
    report = run_demo(simulate_failure=args.simulate_failure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "scheduler": report["scheduler"],
                      "agent_model_called": False, "result": report["result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
