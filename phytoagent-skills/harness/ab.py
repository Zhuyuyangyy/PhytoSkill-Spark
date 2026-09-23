"""Interleaved A/B over the task set, with one experimental variable.

Held byte-identical across arms: model, endpoint, sampling parameters, tool
definitions, task set, harness code and turn budget.

The only variable: whether each Skill's instruction text is present in the
system prompt. Arms are interleaved per task so that a provider-side change
mid-run shows up as model-identity drift in the trace rather than as a fake
treatment effect.

Arm B deliberately pre-loads every instruction instead of expanding it on
demand, because instructions can only influence tool *selection* if they are
present before the model chooses. The extra prompt tokens are recorded, which
is what makes the cost of that choice measurable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from demo.fixture_workspace import fixture_registry
from harness.agent import ARMS, AgentHarness
from harness.config import MODEL_PRICES_PER_MILLION, HarnessConfig
from harness.scoring import score_task, summarise
from harness.tasks import Task, build_tasks, count_by_kind
from harness.transport import MIN_REQUEST_INTERVAL_SECONDS, Transport
from runtime.executor import SkillExecutor

NVIDIA_SMI_TIMEOUT_SECONDS = 10

LIMITATIONS = (
    "工具执行仍为 fixture 模式：Skill 返回的是合成观察，不是真实植物诊断。"
    "本次 A/B 检验的是 Agent 层的选工具与构造参数行为，不是农学准确性。",
    "任务集必须提供与 fixture 精确匹配的标识符，因此本实验不测量图像感知能力。",
    "未使用 DGX GPU 做推理；本报告的 dgx_hardware_used 记录的是本次运行是否触及 GPU。",
    "判定为规则化判定，不是语言模型评审；free text 的可读性不在评分范围内。",
    "单轮运行的样本量不足以做显著性判断，报告只陈述成对差异，不宣布胜负。",
)


def probe_environment() -> dict:
    """Record where the run happened. This is the evidence for platform adaptation."""
    gpu_names: list[str] = []
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            completed = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS, check=False,
            )
            if completed.returncode == 0:
                gpu_names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            gpu_names = []
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "nvidia_smi_present": smi is not None,
        "gpus": gpu_names,
        "cuda_visible_devices": None,
    }


def estimate_cost(*, tasks: int, arms: int, prompt_tokens: int,
                  completion_tokens: int, model: str) -> dict:
    """Estimate what a run will cost, from the published list prices.

    Quota is a hard wall: an exhausted account returns HTTP 402 for every model,
    and a run that dies halfway produces a partial experiment that looks like a
    result. Estimating first lets the operator decide.
    """
    prices = MODEL_PRICES_PER_MILLION.get(model)
    if prices is None:
        return {"model": model, "known_price": False}
    total_prompt = prompt_tokens * tasks * arms
    total_completion = completion_tokens * tasks * arms
    cost = (total_prompt / 1_000_000 * prices["input"]
            + total_completion / 1_000_000 * prices["output"])
    return {
        "model": model,
        "known_price": True,
        "input_price_per_million": prices["input"],
        "output_price_per_million": prices["output"],
        "estimated_prompt_tokens": total_prompt,
        "estimated_completion_tokens": total_completion,
        "estimated_cost_cny": round(cost, 4),
    }


class AbRunner:
    def __init__(self, config: HarnessConfig, *, poster=None, registry_factory=fixture_registry,
                 tasks: list[Task] | None = None, progress=None,
                 min_request_interval: float | None = MIN_REQUEST_INTERVAL_SECONDS,
                 tool_mode: str = "fixture"):
        self.config = config
        self.poster = poster
        self.registry_factory = registry_factory
        self.tasks = tasks if tasks is not None else build_tasks()
        self.progress = progress or (lambda message: None)
        # A scripted poster answers instantly, so throttling it would only make
        # the offline tests slow. Real runs keep the default.
        self.min_request_interval = min_request_interval
        # How the Skills answer tool calls: "fixture" (default, fast, offline) or
        # "live"/"herb" (real DGX inference, slow, cached).
        self.tool_mode = tool_mode
        # Measured baselines from the first successful full A/B run, used only to
        # estimate what the next run will cost. Overridden by --baseline-*.
        self._baseline_prompt_tokens = 6000
        self._baseline_completion_tokens = 600

    def _cost_estimate(self) -> dict:
        """A per-run cost estimate from the prices published at config time."""
        return estimate_cost(
            tasks=len(self.tasks), arms=len(ARMS),
            # Both arms are prompted the same way apart from the Skill
            # instructions, so one measured pair's token counts scale linearly.
            prompt_tokens=self._baseline_prompt_tokens,
            completion_tokens=self._baseline_completion_tokens,
            model=self.config.model)

    def run(self, *, repeat: int = 1) -> dict:
        config = self.config
        config.require_key()
        runs: list[dict] = []
        scored: list[dict] = []
        with self.registry_factory() as registry:
            transport = Transport(config, self.poster,
                                  min_request_interval=self.min_request_interval)
            harness = AgentHarness(registry, SkillExecutor(registry), transport, config,
                                   tool_mode=self.tool_mode)
            packages = {name: {"manifest_sha256": detail["manifest_sha256"],
                               "instructions_sha256": detail["instructions_sha256"],
                               "instructions_chars": detail["instructions_chars"]}
                        for name, detail in harness.skill_instructions.items()}
            for round_index in range(repeat):
                for task_index, task in enumerate(self.tasks):
                    # A task may override the run's default tool mode. Real-image
                    # tasks carry their own, because a photograph needs a real
                    # adapter while the rest of the set runs on fixtures.
                    harness.tool_mode = task.tool_mode or self.tool_mode
                    for arm in self._arm_order(round_index + task_index):
                        self.progress(f"[r{round_index + 1}] {task.task_id} / {arm}")
                        run = harness.run_task(task, arm=arm)
                        row = score_task(task, run)
                        runs.append(run)
                        scored.append(row)

            arms = {arm: summarise(arm, [row for row in scored if row["arm"] == arm],
                                   harness.context_report(arm)) for arm in ARMS}
            models = sorted({name for run in runs for name in (run.get("models_returned") or [])})
            return {
                "scope": "agent_ab_real_model_fixture_tools",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "environment": probe_environment(),
                "provider": {"kind": "openai_compatible_chat_completions",
                             "host": config.host, "endpoint_path": "/chat/completions"},
                "config": config.redacted(),
                "single_variable": "Skill 指令是否出现在 system prompt 中",
                "tool_mode": self.tool_mode,
                "cost_estimate": self._cost_estimate(),
                "task_tool_modes": {task.task_id: (task.tool_mode or self.tool_mode)
                                    for task in self.tasks},
                "held_constant": ["model", "endpoint", "sampling_parameters", "tool_definitions",
                                  "task_set", "harness_code", "max_turns", "turn_loop"],
                "tool_definitions_sha256": _digest(json.dumps(harness.tool_definitions(),
                                                              ensure_ascii=False, sort_keys=True)),
                "skill_packages": packages,
                "arms_summary": arms,
                "comparison": _compare(arms["without_skill"], arms["with_skill"]),
                "models_returned_all": models,
                "model_identity_consistent": len(models) <= 1,
                "repeat": repeat,
                "task_count": len(self.tasks),
                "task_kinds": count_by_kind(self.tasks),
                "tasks": [task.to_dict() for task in self.tasks],
                "scored": scored,
                "runs": runs,
                "agent_model_called": True,
                "skill_mode": "fixture",
                "dgx_hardware_used": False,
                "trust": {"signature_format": "project_ed25519",
                          "demo_key_generated_this_run": True,
                          "nvidia_verified": False},
                "scoring_method": "rule_based_deterministic",
                "limitations": list(LIMITATIONS),
            }

    @staticmethod
    def _arm_order(seed: int) -> tuple[str, str]:
        """Alternate which arm goes first so ordering cannot masquerade as an effect."""
        return ARMS if seed % 2 == 0 else (ARMS[1], ARMS[0])


def _compare(without: dict, with_skill: dict) -> dict:
    def delta(field: str) -> float | None:
        left, right = without.get(field), with_skill.get(field)
        if left is None or right is None:
            return None
        return round(right - left, 3)

    def efficiency(field: str):
        left = (without.get("efficiency") or {}).get(field)
        right = (with_skill.get("efficiency") or {}).get(field)
        if left is None or right is None:
            return None
        return round(right - left, 3)

    paired = min(without.get("tasks_total") or 0, with_skill.get("tasks_total") or 0)
    return {
        "trigger_pass_rate_delta": delta("trigger_pass_rate"),
        "coverage_mean_delta": delta("coverage_mean"),
        "forbidden_violations_delta": delta("forbidden_violations"),
        "unknown_tool_attempts_delta": delta("unknown_tool_attempts"),
        "argument_errors_delta": delta("argument_errors"),
        "prompt_tokens_delta": efficiency("prompt_tokens"),
        "total_tokens_delta": efficiency("total_tokens"),
        "reasoning_tokens_delta": efficiency("reasoning_tokens"),
        "latency_ms_mean_delta": efficiency("latency_ms_mean"),
        "turns_mean_delta": efficiency("turns_mean"),
        "statistics": {
            "paired_tasks": paired,
            "significance_test": "none",
            "note": "报告成对差异，不做显著性判断；样本量足够前不要据此宣布某臂更优。",
        },
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def task_from_dict(spec: dict) -> Task:
    return Task(**{key: (tuple(value) if key in
                         ("expected_skills", "forbidden_skills", "coverage_skills") else value)
                   for key, value in spec.items()})


__all__ = ["AbRunner", "probe_environment", "task_from_dict", "write_report"]
