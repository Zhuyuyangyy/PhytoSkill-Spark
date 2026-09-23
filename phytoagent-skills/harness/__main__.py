"""Command line for the live Harness.

    python -m harness dry-run        resolve config, print the plan, call nothing
    python -m harness preflight      verify endpoint, credential, tool calling, model identity
    python -m harness smoke          one task, both arms, verbose
    python -m harness ab             the full interleaved A/B

Run ``preflight`` against a real endpoint before trusting any A/B number: several
StepFun models do not support tool calling at all, and an endpoint that load
balances across model snapshots cannot hold the model constant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.ab import AbRunner, probe_environment, write_report
from harness.agent import AgentHarness
from harness.config import HarnessConfig
from harness.errors import AuthError, ConfigError, HarnessError, PreflightError
from harness.tasks import build_tasks
from harness.transport import Transport

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PREFLIGHT = 3
EXIT_HARNESS = 4

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "phyto_ping",
        "description": "连通性检查工具：把 text 参数原样返回。仅在用户明确要求调用时使用。",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": ["text"],
        },
    },
}

PING_PROMPT = "请调用 phyto_ping 工具，参数 text 设为 ping。"


def _config(args) -> HarnessConfig:
    return HarnessConfig.from_env(use_dotenv=not args.no_dotenv)


def _preflight(config: HarnessConfig, *, samples: int, allow_model_drift: bool) -> dict:
    config.require_key()
    transport = Transport(config)
    checks: dict = {}

    first = transport.chat({"model": config.model, "max_tokens": 16,
                            "messages": [{"role": "user", "content": "请只回复：ok"}]})
    checks["reachable"] = True
    checks["auth_accepted"] = True
    checks["first_response"] = {
        "http_status": first.http_status,
        "model_returned": first.model_returned,
        "finish_reason": first.finish_reason,
        "usage": first.usage_flat,
        "latency_ms": first.latency_ms,
    }

    probe = transport.chat({"model": config.model, "max_tokens": 128, "tools": [PING_TOOL],
                            "tool_choice": "auto",
                            "messages": [{"role": "user", "content": PING_PROMPT}]})
    names = [(call.get("function") or {}).get("name") for call in probe.tool_calls]
    checks["tool_calling"] = {
        "tool_calls_returned": len(names),
        "names": names,
        "works": names == ["phyto_ping"],
        "finish_reason": probe.finish_reason,
    }

    identity = []
    for _ in range(samples):
        identity.append(transport.chat({"model": config.model, "max_tokens": 8,
                                        "messages": [{"role": "user", "content": "ok"}]}))
    seen = transport.distinct_models
    reasoning = [item.usage_flat.get("reasoning_tokens") for item in identity]
    checks["model_identity"] = {
        "requests": samples + 2,
        "distinct_models": seen,
        "consistent": len(seen) <= 1,
        "reasoning_tokens_observed": sorted({value for value in reasoning if value is not None}),
        "thinking_mode_note": (
            "reasoning_effort 未固定，两臂使用同一默认值；已在报告的 sampling_parameters_sent 中记录。"
            if config.reasoning_effort is None else
            f"reasoning_effort 固定为 {config.reasoning_effort}。"
        ),
    }

    failures = []
    if not checks["tool_calling"]["works"]:
        failures.append("tool calling did not return the expected phyto_ping call")
    if not checks["model_identity"]["consistent"] and not allow_model_drift:
        failures.append(
            f"endpoint returned {len(seen)} distinct models {seen}; the model cannot be held constant"
        )
    checks["failures"] = failures
    checks["passed"] = not failures
    checks["config"] = config.redacted()
    checks["environment"] = probe_environment()

    if failures and not allow_model_drift:
        raise PreflightError("; ".join(failures))
    return checks


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-dotenv", action="store_true",
                        help="ignore .env and use the process environment only")
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="print the resolved plan without calling anything")
    dry.add_argument("--kind", action="append", help="restrict to a task kind")
    dry.add_argument("--repeat", type=int, default=1)

    pre = sub.add_parser("preflight", help="verify endpoint, credential, tool calling, model identity")
    pre.add_argument("--samples", type=int, default=None)
    pre.add_argument("--allow-model-drift", action="store_true")
    pre.add_argument("--output", type=Path, default=Path("artifacts/preflight.json"))

    smoke = sub.add_parser("smoke", help="run one task in both arms")
    smoke.add_argument("--task", default=None, help="task_id; defaults to the first positive trigger")

    ab = sub.add_parser("ab", help="run the full interleaved A/B")
    ab.add_argument("--kind", action="append", help="restrict to a task kind")
    ab.add_argument("--repeat", type=int, default=1)
    ab.add_argument("--output", type=Path, default=Path("artifacts/agent-ab.json"))
    ab.add_argument("--baseline-prompt-tokens", type=int, default=6000,
                    help="per-task prompt tokens used for the cost estimate")
    ab.add_argument("--baseline-completion-tokens", type=int, default=600,
                    help="per-task completion tokens used for the cost estimate")
    ab.add_argument("--real-images", action="store_true",
                    help="run only the real-photograph tasks. Each carries its own "
                         "mode (live/herb) and needs the images on disk plus a "
                         "reachable DGX Spark node. Not mixable with the fixture "
                         "set, whose inputs every real mode refuses by design.")
    ab.add_argument("--tool-mode", default="fixture", choices=["fixture", "live", "herb"],
                    help="default mode for tasks that do not specify one. Real-image "
                         "tasks always carry their own.")

    args = parser.parse_args(argv)
    try:
        config = _config(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    try:
        if args.command == "dry-run":
            tasks = build_tasks()
            if getattr(args, "kind", None):
                tasks = [task for task in tasks if task.kind in args.kind]
            from demo.fixture_workspace import fixture_registry
            from runtime.executor import SkillExecutor
            with fixture_registry() as registry:
                harness = AgentHarness(registry, SkillExecutor(registry),
                                       Transport(config, poster=_never_called), config)
                _print({
                    "scope": "dry_run_no_network",
                    "config": config.redacted(),
                    "api_key_present": bool(config.api_key),
                    "tool_count": len(harness.tool_definitions()),
                    "tool_names": [tool["function"]["name"] for tool in harness.tool_definitions()],
                    "arms": {arm: harness.context_report(arm) for arm in ("without_skill", "with_skill")},
                    "task_kinds": _count(tasks),
                    "task_count": len(tasks),
                    "task_ids": [task.task_id for task in tasks],
                    "estimated_model_calls_upper_bound":
                        len(tasks) * 2 * int(getattr(args, "repeat", 1)) * config.max_turns,
                    "note": "dry-run 不联网；密钥仅以存在性报告。",
                })
            return EXIT_OK

        if args.command == "preflight":
            samples = args.samples if args.samples is not None else config.preflight_samples
            checks = _preflight(config, samples=samples, allow_model_drift=args.allow_model_drift)
            write_report(checks, args.output)
            _print(checks)
            print(f"\npreflight report: {args.output}", file=sys.stderr)
            return EXIT_OK if checks["passed"] else EXIT_PREFLIGHT

        if args.command == "smoke":
            tasks = build_tasks()
            chosen = ([task for task in tasks if task.task_id == args.task]
                      or [task for task in tasks if task.kind == "positive_trigger"])
            if not chosen:
                print("no task matched", file=sys.stderr)
                return EXIT_CONFIG
            runner = AbRunner(config, tasks=[chosen[0]], tool_mode=args.tool_mode,
                              progress=lambda message: print(message, file=sys.stderr))
            report = runner.run(repeat=1)
            for row in report["scored"]:
                _print({"arm": row["arm"], "task_id": row["task_id"], "verdict": row["verdict"],
                        "selected_skills": row["selected_skills"],
                        "stop_reason": row["stop_reason"], "turns": row["turns"],
                        "final_text": row["final_text"]})
            _print(report["comparison"])
            return EXIT_OK

        tasks = build_tasks()
        if args.kind:
            tasks = [task for task in tasks if task.kind in args.kind]
        if not tasks:
            print("no task matched", file=sys.stderr)
            return EXIT_CONFIG
        from harness.tasks import build_tasks as _build
        if getattr(args, "real_images", False):
            tasks = _build(only_real_images=True)
        runner = AbRunner(config, tasks=tasks, tool_mode=args.tool_mode,
                          progress=lambda message: print(message, file=sys.stderr))
        runner._baseline_prompt_tokens = args.baseline_prompt_tokens
        runner._baseline_completion_tokens = args.baseline_completion_tokens
        # Print the estimate before spending anything: an exhausted account returns
        # HTTP 402 for every model, and a partial run looks like a result.
        estimate = runner._cost_estimate()
        if estimate.get("known_price"):
            print(json.dumps({"cost_estimate": estimate}, ensure_ascii=False, indent=2),
                  file=sys.stderr)
        report = runner.run(repeat=args.repeat)
        write_report(report, args.output)
        _print({"arms_summary": report["arms_summary"], "comparison": report["comparison"],
                "models_returned_all": report["models_returned_all"],
                "model_identity_consistent": report["model_identity_consistent"],
                "environment": report["environment"]})
        print(f"\nreport: {args.output}", file=sys.stderr)
        return EXIT_OK

    except AuthError as exc:
        print(f"authentication failed: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except PreflightError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except HarnessError as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return EXIT_HARNESS


def _never_called(*_args, **_kwargs):
    raise AssertionError("dry-run must not reach the network")


def _count(tasks) -> dict:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.kind] = counts.get(task.kind, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
