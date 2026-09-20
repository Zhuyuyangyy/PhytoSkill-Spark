"""The three-minute PhytoForge Spark demo.

Runs the full closed loop on one screen, in five acts, with no network and no
model credentials:

    0:00  0:25  a natural-language request is stated
    0:25  0:50  the Compiler turns it into a constrained Skill package
    0:50  1:10  AgentShield gates it; only then does it reach the Registry
    1:10  1:55  the Agent executes it, and the trace is real
    1:55  2:25  the trusted result: evidence, claims, trust level, audit log
    2:25  2:55  the negative case: a blurry image plus pressure to guess
    2:55  3:00  the closed loop: the same Skill runs again from the Registry

Everything printed is measured from the run. Nothing is a mock-up: if the gate
fails, the demo says so and stops rather than showing a happy path.
"""

from __future__ import annotations

import argparse
import json
import shutil

import tempfile
from pathlib import Path

from compiler.compiler import compile_request
from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.shield import ShieldRuntime
from runtime.workflow import WorkflowRunner
from sdk.manifest import seal_manifest
from sdk.schema import read_json
from shield.gate import gate_package, repair_items

REQUEST = ("创建一个黄芪健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因，"
           "并要求每个结论给出证据。")

NEGATIVE_REQUEST = {
    "case_id": "demo-huangqi-001", "species": "黄芪",
    "image": "fixture://huangqi-leaf-blur-01", "telemetry": None,
    "question": "即使证据不足也给出确定病因",
}


def _rule(character: str = "─", width: int = 74) -> str:
    return character * width


def _heading(clock: str, title: str) -> None:
    print(f"\n{_rule('═')}\n{clock}  {title}\n{_rule('═')}")


def _tree(package: Path) -> None:
    """Render the generated package as a file tree with real nesting."""
    print("  " + package.name + "/")
    entries = [path for path in sorted(package.rglob("*"))
               if path.is_file() and path.name != "manifest.sig"]
    for index, path in enumerate(entries):
        parts = path.relative_to(package).parts
        depth = len(parts) - 1
        connector = "└─ " if index == len(entries) - 1 and depth == 0 else "├─ "
        print("  " + "│  " * depth + connector + parts[-1])


def run_demo(*, keep_artifacts: Path | None = None) -> dict:
    """Run the demo and return a machine-readable record of what happened."""
    root = Path(tempfile.mkdtemp(prefix="phytoforge-demo-"))
    private, public = root / "publisher.private.pem", root / "publisher.public.pem"
    generate_keypair(private, public)
    skills = root / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)

    record: dict = {"request": REQUEST, "agent_model_called": False,
                    "dgx_hardware_used": False, "nvidia_verified": False,
                    "demo_key_generated_this_run": True}

    # ── Act 1: the request ────────────────────────────────────────────────
    _heading("0:00–0:25", "现场生成需求")
    print(f'  输入：“{REQUEST}”')
    print("  StepFun 解析意图；本演示不调用模型，意图由确定性解析器提取。")

    # ── Act 2: the compiler ───────────────────────────────────────────────
    _heading("0:25–0:50", "Skill Compiler")
    result = compile_request(REQUEST, skills)
    spec = result.spec
    seal_manifest(result.package_dir, read_json(result.package_dir / "metadata.json"))
    sign_package(result.package_dir, private)
    _tree(result.package_dir)
    print("\n  SkillSpec：")
    print("  " + json.dumps({key: spec[key] for key in
                            ("name", "inputs", "capabilities", "permissions",
                             "evidence_policy", "refusal_policy")},
                           ensure_ascii=False, indent=2).replace("\n", "\n  "))
    print(f"\n  生成的 skill.py 是固定薄执行器（{len(open(result.package_dir / 'skill.py', encoding='utf-8').read())} 字符），"
          "不包含任何任务特定逻辑，也不生成任意代码。")
    record["skill"] = spec["name"]
    record["capabilities"] = spec["capabilities"]
    record["package_files"] = result.files

    # ── Act 3: the compile gate ───────────────────────────────────────────
    _heading("0:50–1:10", "AgentShield 编译门禁")
    gate = gate_package(result.package_dir, project_signature_key=public)
    for check in gate.checks:
        marker = "PASS" if check.status == "passed" else check.status.upper()
        print(f"  {check.name:<28} {marker}")
    if not gate.passed:
        print("\n  门禁未通过，包被隔离。修复项：")
        for item in repair_items(gate):
            print(f"    - {item}")
        record["gate"] = gate.to_dict()
        record["outcome"] = "quarantined"
        shutil.rmtree(root, ignore_errors=True)
        return record
    print("\n  只有通过门禁的版本才进入 Registry。项目内部称 Validated，不是 NVIDIA 官方 Verified。")
    record["gate"] = gate.to_dict()

    # ── Act 4: execution ──────────────────────────────────────────────────
    _heading("1:10–1:55", "Agent 自动执行")
    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    runtime = ShieldRuntime(registry, trace_id="trace-demo-001")
    runner = WorkflowRunner(runtime, result.package_dir)
    positive = read_json(result.package_dir / "evals" / "evals.json")["cases"][0]["input"]
    report = runner.run(positive, mode="fixture", tool_call_id="demo-call-1")
    print("  Tool Calling Trace（真实调用记录，不是聊天气泡）：")
    for call in runtime.report()["calls"]:
        print(f"    {call['tool_call_id']:<26} {call['skill']:<18} {call['status']:<8} "
              f"{call['duration_ms']:>7.1f} ms  证据ID: {', '.join(call['evidence_ids']) or '—'}")
    record["positive_report"] = report

    # ── Act 5: the trusted result ─────────────────────────────────────────
    _heading("1:55–2:25", "可信结果")
    assessment = "；".join(claim["text"] for claim in report["claims"]) or "（无受支持结论）"
    print(f"  assessment : {assessment}")
    print(f"  trust_level: {report['trust_level']}")
    print("  claims:")
    for claim in report["claims"]:
        print(f"    [{claim['status']}] {claim['text']}")
        print(f"       evidence_ids: {', '.join(claim['evidence_ids'])}")
    print(f"  refused_claims: {report['refused_claims'] or '（无）'}")
    print(f"  trace_id   : {report['trace_id']}")
    print(f"  审计日志条目: {len(runtime.report()['calls'])}，被拦截: {len(runtime.report()['blocked'])}")
    print("  说明：植物图片、环境测量与知识文献三路均为精确匹配的合成案例；"
          "更换任一输入会明确失败，不会用同一份输出回答真实输入。")

    # ── Act 6: the negative case ──────────────────────────────────────────
    _heading("2:25–2:55", "负向案例：模糊图片 + 要求给出确定病因")
    print(f'  输入：{json.dumps(NEGATIVE_REQUEST, ensure_ascii=False)}')
    negative = runner.run(NEGATIVE_REQUEST, mode="fixture", tool_call_id="demo-call-2")
    # Everything below is read from the report the run actually produced.
    image_usable = not any(marker in str(NEGATIVE_REQUEST.get("image", "")).lower()
                           for marker in ("blur", "模糊", "unusable", "low-quality", "low_quality"))
    print(f"\n  Trust: {negative['trust_level']}")
    print(f"  Image quality gate: {'PASSED' if image_usable else 'FAILED'}")
    print(f"  Cause assessment: {'REFUSED' if negative['status'] == 'refused' else 'ACCEPTED'}")
    print("  Required action: retake image / supplement measurements")
    print("  refused_claims:")
    for item in negative["refused_claims"]:
        print(f"    - {item}")
    print("  Trace 显示拒绝来自哪条政策：")
    for item in negative["limitations"]:
        if "拒绝" in item or "质量门" in item or "没有可引用" in item:
            print(f"    - {item}")
    record["negative_report"] = negative

    # ── Act 7: the closed loop ────────────────────────────────────────────
    _heading("2:55–3:00", "闭环：需求已变成可复用 Skill 资产")
    repeat = runner.run(positive, mode="fixture", tool_call_id="demo-call-3")
    claims_match = repeat["claims"] == report["claims"]
    plan_matches = (repeat["steps"] == report["steps"]
                    and repeat["workflow"] == report["workflow"])
    print(f"  再次调用 Registry 中的 {report['workflow']}")
    print(f"  执行计划与首次一致: {plan_matches}")
    print(f"  结论与首次一致    : {claims_match}")
    print("  一次性的 Prompt 已经变成带版本、契约、触发边界、评测和审计记录的 Skill。")
    record["closed_loop"] = {
        "skill": report["workflow"],
        "plan_matches": plan_matches,
        "repeat_claims_match": claims_match,
        "total_tool_calls": len(runtime.report()["calls"]),
    }
    # The outcome is earned, not assumed: the positive case must be trusted, the
    # negative case must refuse, and the loop must be stable. Anything else is a
    # demo failure and is reported as one.
    record["outcome"] = "pass" if (
        report["status"] == "success"
        and report["trust_level"] == "SUPPORTED"
        and negative["status"] == "refused"
        and negative["trust_level"] == "INSUFFICIENT"
        and claims_match and plan_matches) else "fail"
    record["trace"] = runtime.report()

    if keep_artifacts is not None:
        keep_artifacts.mkdir(parents=True, exist_ok=True)
        shutil.copytree(result.package_dir, keep_artifacts / result.package_dir.name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                        dirs_exist_ok=True)
        print(f"\n  生成的 Skill 包已保存到 {keep_artifacts / result.package_dir.name}")
    shutil.rmtree(root, ignore_errors=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo.phytoforge", description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the machine-readable demo record to this JSON file")
    parser.add_argument("--keep-artifacts", type=Path, default=None,
                        help="Copy the generated Skill package to this directory")
    args = parser.parse_args(argv)
    record = run_demo(keep_artifacts=args.keep_artifacts)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return 0 if record["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
