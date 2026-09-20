"""Execute package-authored fixture contract cases; never claim Agent A/B results."""

import argparse
import json
from pathlib import Path

from demo.fixture_workspace import fixture_registry
from runtime.executor import SkillExecutor
from sdk.schema import package_file, read_json


def value_at(document, pointer):
    value = document
    for component in pointer.lstrip("/").split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def evaluate() -> dict:
    with fixture_registry() as registry:
        executor = SkillExecutor(registry)
        reports = []
        for record in registry.records:
            name = record["manifest"]["name"]
            spec = read_json(package_file(Path(record["package_dir"]), "evals/evals.json"))
            cases = []
            for case in spec["contract_cases"]:
                response = executor.call(name, case["input"], mode=case["mode"],
                                         tool_call_id=f"eval:{name}:{case['id']}")
                expected = case["expect"]
                passed = response["status"] == expected["status"]
                if expected["status"] == "failed":
                    passed = passed and response["error"]["code"] == expected["error_code"]
                else:
                    try:
                        passed = passed and all(value_at(response["data"], key) == value
                                                for key, value in expected["values"].items())
                    except (KeyError, IndexError, TypeError):
                        passed = False
                cases.append({"id": case["id"], "passed": bool(passed), "response": response})
            reports.append({"skill": name, "manifest_sha256": record["manifest"]["manifest_sha256"],
                            "cases": cases, "agent_behavior_cases_supplied": len(spec["agent_cases"])})
        total = sum(len(item["cases"]) for item in reports)
        passed = sum(case["passed"] for item in reports for case in item["cases"])
        return {"scope": "fixture_contracts_only", "total": total, "passed": passed,
                "failed": total - passed, "skills": reports,
                "agent_ab": {"status": "not_run", "reason": "Requires the real StepFun Harness and matched trials"},
                "security_scan": "not_run", "nvidia_verified": False, "dgx_hardware_used": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/skill-contract-evals.json"))
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("scope", "total", "passed", "failed", "agent_ab")}, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
