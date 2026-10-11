"""Demo V1's run engine: the governed pipeline, with honest replay/real marking.

The demo runs the *same* pipeline the project runs — Compiler, Registry,
Shield, WorkflowRunner, ClaimAuditor — and adds one thing the pipeline itself
does not need: a per-step statement of whether what the visitor is looking at
is a replay or a real inference, and why.

The marking rules, in one place:

* ``fixture`` mode is a replay: it returns the sealed synthetic case, and no
  model is called.
* ``live``/``herb`` mode against the ``replay`` backend is a replay of a
  *recorded real* observation — the source file is named in the reason.
* ``live``/``herb`` mode against any other backend is real inference, and the
  endpoint is named.

A run never silently changes mode: if a real backend is requested and none is
configured, the run fails with the reason. That is the project's standing rule
and the demo's most important honesty property.
"""

from __future__ import annotations

import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from compiler.compiler import compile_request
from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from demo.v1_cases import DEMO_CASES, REPLAY_REAL, case_by_id
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.shield import ShieldRuntime
from runtime.workflow import WorkflowRunner
from sdk.manifest import seal_manifest
from sdk.schema import read_json

HUANGQI_REQUEST = ("创建一个黄芪健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因，"
                   "并给出可信证据。")

# The offline replay source: the recorded DGX Spark runs. Absolute, because a
# relative path would depend on the server's working directory.
REPLAY_SOURCE = PROJECT_ROOT / "artifacts" / "dgx" / "vision-runs.json"


class DemoWorkspace:
    """A persistent, signed workspace the demo reuses across runs.

    Materialised once into a stable directory and then reused: building it
    per run would spend seconds re-signing packages the visitor did not ask
    about, and a temporary directory would be deleted at teardown — which this
    project's machines do slowly enough to matter.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else PROJECT_ROOT / "artifacts" / "demo-workspace"
        self.private = self.root / "demo.publisher.private.pem"
        self.public = self.root / "demo.publisher.public.pem"
        self.skills = self.root / "skills"
        self.package = self.root / "skills" / "huangqi-health-assessment"
        self._registry: SkillRegistry | None = None

    def materialise(self) -> "DemoWorkspace":
        if self.package.is_dir() and (self.package / "manifest.json").is_file():
            return self
        self.root.mkdir(parents=True, exist_ok=True)
        generate_keypair(self.private, self.public)
        self.skills.mkdir(exist_ok=True)
        for name in SKILL_NAMES:
            destination = self.skills / name
            shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
            sign_package(destination, self.private)
        compiled = compile_request(HUANGQI_REQUEST, self.skills)
        seal_manifest(compiled.package_dir, read_json(compiled.package_dir / "metadata.json"))
        sign_package(compiled.package_dir, self.private)
        return self

    def registry(self) -> SkillRegistry:
        if self._registry is None:
            registry = SkillRegistry(self.skills, trusted_public_key=self.public)
            registry.discover()
            self._registry = registry
        return self._registry

    def runner(self) -> WorkflowRunner:
        return WorkflowRunner(ShieldRuntime(self.registry(), trace_id=None),
                              self.package)


def backend_status() -> dict:
    """What the demo would run against, without calling anything.

    The status is a statement about configuration, never a probe: whether a
    configured endpoint answers is discovered by the run itself, which fails
    loudly rather than falling back.
    """
    from backends.base import KIND_REPLAY
    from backends.registry import CHOICES, ENV_BACKEND, resolve

    configured = os.environ.get(ENV_BACKEND, "")
    if not configured.strip():
        return {"configured": False, "spec": "", "kind": None,
                "reason": (f"{ENV_BACKEND} is not set; fixture mode needs no backend, "
                           f"real inference needs one of: {', '.join(CHOICES)}"),
                "replay_available": REPLAY_SOURCE.is_file()}
    try:
        backend = resolve(configured)
        identity = backend.identity  # a property: where this backend executes
        return {"configured": True, "spec": configured, "kind": identity.kind,
                "endpoint": identity.endpoint, "reason": "",
                "replay_available": REPLAY_SOURCE.is_file()}
    except Exception as exc:  # noqa: BLE001 - reported to the visitor verbatim
        return {"configured": False, "spec": configured, "kind": None,
                "reason": f"{type(exc).__name__}: {exc}",
                "replay_available": REPLAY_SOURCE.is_file()}


# The badges a visitor must be able to tell apart. Each one is derived from
# what the step's output actually says about its own origin — never from what
# was requested. A live run whose vision leg refused a fixture:// path never
# ran a model, and marking it "real" because live was asked for would be a lie
# the visitor has no way to detect.
BADGE_REPLAY = "replay"                  # sealed synthetic, or a recorded observation
BADGE_DETERMINISTIC = "deterministic"    # real computation, no model call
BADGE_REAL = "real"                      # a model actually ran for this call
BADGE_CACHE = "cache"                    # a cached observation: real, but not this run's
BADGE_NOT_EXECUTED = "not_executed"      # provably never started
BADGE_EXECUTION_FAILED = "execution_failed"  # started, and failed; the model state is unknown

BADGE_LABELS = {
    BADGE_REPLAY: "回放",
    BADGE_DETERMINISTIC: "确定性",
    BADGE_REAL: "真实推理",
    BADGE_CACHE: "缓存命中",
    BADGE_NOT_EXECUTED: "未执行",
    BADGE_EXECUTION_FAILED: "执行失败",
}

# Error codes that provably precede any model call: the backend could not be
# resolved, the mode was refused, or the door refused the call. Everything
# else — a contract violation, an execution error, a boundary kill — may have
# happened on either side of the model call, and the record does not say
# which. Claiming "no model was called" for those would be a guess dressed as
# a measurement.
PRE_EXECUTION_CODES = frozenset({
    "BackendUnavailable",     # nothing to call: the backend could not be resolved
    "UnsupportedModeError",   # the mode was refused before the Skill ran
    "PermissionViolation",    # the door refused the call
    "BudgetExceeded",         # the door refused the call
    "TraceError",             # the door refused the call
    "ShieldError",            # registry verification refused the call
})

# Providers whose output is deterministic computation by construction: no
# model is involved in any mode, and the provenance block says so.
_DETERMINISTIC_PROVIDERS = ("growth_risk", "evidence_fusion", "herbal_knowledge")


def _step_marking(step: dict) -> tuple[str, str]:
    """(badge, reason) for one step, from the execution provenance it returned.

    The provenance travels with the provider's own output (see
    ``WorkflowRunner._step_provenance``), so this is a statement about what
    happened, not about what was asked for. Every branch names its evidence.
    """
    if step.get("status") == "skipped":
        # A skipped step never started: its required input was missing, and
        # the pipeline recorded that instead of fabricating one.
        return BADGE_NOT_EXECUTED, (
            f"本步未执行（{step.get('detail') or '所需输入缺失'}）；"
            "没有模型被调用，也没有产生任何数据")
    if step.get("status") != "success":
        code = step.get("detail") or "failed"
        if code in PRE_EXECUTION_CODES:
            return BADGE_NOT_EXECUTED, (
                f"本步在调用开始前被拒绝（{code}）；没有模型被调用，"
                "也没有产生任何数据")
        # The step started and failed. Input validation runs before the model
        # call and output validation after it, so the record alone cannot say
        # which side this was — asserting either would be a guess.
        return BADGE_EXECUTION_FAILED, (
            f"本步执行失败（{code}）；记录无法确认模型是否已被调用——"
            "输入校验在调用前、输出校验在调用后，此处不断言")

    provenance = step.get("provenance") or {}
    provider = step.get("provider_skill", "")
    cache_state = provenance.get("observation_cache")

    if cache_state == "replay":
        source = provenance.get("backend_endpoint") or "已记录的观测"
        return BADGE_REPLAY, (f"回放已记录的真实观测（{source}）；本次运行没有发起推理，"
                              "内容为当时真实推理所得")
    if cache_state == "hit":
        return BADGE_CACHE, ("观察缓存命中：内容与首次真实推理一致，但本次没有调用模型；"
                             "耗时同样不代表推理耗时")
    if provenance.get("data_origin") == "synthetic_fixture":
        return BADGE_REPLAY, "fixture 模式返回封存的合成案例；未调用任何模型"
    if provider in _DETERMINISTIC_PROVIDERS:
        return BADGE_DETERMINISTIC, ("确定性计算：真实语料索引检索或规则融合，"
                                     "无模型调用")
    if cache_state == "miss" or provenance.get("model_called"):
        endpoint = provenance.get("backend_endpoint") or provenance.get("backend_kind") or "已配置后端"
        model = provenance.get("model")
        suffix = f"（模型 {model}）" if model else ""
        return BADGE_REAL, f"真实推理：本次调用由后端 {endpoint} 执行{suffix}"
    return BADGE_EXECUTION_FAILED, "输出未声明执行来源；无法确认是否发生推理，不猜测"


# One run at a time. The case-declared backend is applied by setting the
# environment variable the Skill resolves at call time, and an environment
# variable is process-wide state: with a ThreadingHTTPServer, two concurrent
# runs would otherwise read each other's backend — visitor B's "real
# inference" could silently execute against the replay source visitor A's
# case declared. Serialising runs makes the declaration atomic. The proper
# fix is to make the backend a per-call parameter instead of ambient
# configuration; until the Skill takes one, this lock is what keeps the
# marking honest under concurrency.
_RUN_LOCK = threading.Lock()


@contextmanager
def _declared_backend(spec: str | None):
    """Run with a case-declared backend, restoring the environment afterwards.

    The declaration is the case's own statement of where its observations come
    from; the environment still wins for every other case, and is restored the
    moment the run ends. Held under ``_RUN_LOCK`` by ``run_case`` so the
    declaration cannot leak into a concurrent run.
    """
    if not spec:
        yield
        return
    from backends.registry import ENV_BACKEND

    previous = os.environ.get(ENV_BACKEND)
    os.environ[ENV_BACKEND] = spec
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_BACKEND, None)
        else:
            os.environ[ENV_BACKEND] = previous


def run_case(demo_id: str, *, mode: str = "fixture",
             workspace: DemoWorkspace | None = None) -> dict:
    """Run one demo case through the governed pipeline and mark every step."""
    case = case_by_id(demo_id)
    workspace = (workspace or DemoWorkspace()).materialise()
    # A case may carry its own backend declaration — the replay case points at
    # the recorded-observations file, so a real-record replay needs no
    # environment setup at all. Without one, the environment decides.
    declared = case.get("backend")
    with _RUN_LOCK, _declared_backend(declared):
        return _run_case(case, demo_id, mode, workspace)


def _run_case(case: dict, demo_id: str, mode: str, workspace: "DemoWorkspace") -> dict:
    backend = backend_status()

    if mode != "fixture" and not backend["configured"]:
        # The standing rule: no silent fallback to fixture. The visitor is told
        # exactly what is missing, and the run does not happen.
        return {"case": case, "mode": mode, "backend": backend,
                "any_real_inference": False,
                "error": {"code": "BackendUnavailable", "message": backend["reason"]},
                "steps": [], "audit": None, "report": None}

    # The vision leg runs in the requested mode; the deterministic legs run in
    # their own — the shape of the real chains, and the reason the workflow
    # runner takes a per-provider mode override at all.
    provider_modes = ({"plant_vision": mode, "growth_risk": "fixture",
                       "herbal_knowledge": "corpus"}
                      if mode != "fixture" else None)
    runner = workspace.runner()
    started = perf_counter()
    try:
        report = runner.run(case["payload"], mode=mode,
                            tool_call_id=f"demo-{demo_id}-{mode}",
                            provider_modes=provider_modes)
    except Exception as exc:  # noqa: BLE001 - a failed demo run is a result too
        return {"case": case, "mode": mode, "backend": backend,
                "any_real_inference": False,
                "error": {"code": type(exc).__name__, "message": str(exc)},
                "steps": [], "audit": None, "report": None,
                "duration_ms": round((perf_counter() - started) * 1000, 3)}
    duration_ms = round((perf_counter() - started) * 1000, 3)

    steps = []
    for step in report["steps"]:
        badge, reason = _step_marking(step)
        steps.append({
            "id": step.get("id"), "provider": step["provider_skill"],
            "status": step["status"], "mode": step.get("mode", mode),
            "badge": badge, "badge_label": BADGE_LABELS[badge],
            "badge_reason": reason,
            "provenance": step.get("provenance") or {},
            "detail": step.get("detail", ""),
            "evidence_ids": step.get("evidence_ids", []),
        })
    return {
        "case": case, "mode": mode, "backend": backend,
        "any_real_inference": any(step["badge"] == BADGE_REAL for step in steps),
        "steps": steps,
        "audit": {
            "status": report["status"], "trust_level": report["trust_level"],
            "claims": report["claims"], "refused_claims": report["refused_claims"],
            "violations": report["violations"],
            "unlinked_evidence_ids": report["unlinked_evidence_ids"],
            "missing_inputs": report["missing_inputs"],
            "limitations": report["limitations"],
            # How the image-usability verdict was reached — the visitor sees
            # whether a refusal came from the model's own flag or a file name.
            "image_usability_basis": report.get("image_usability_basis", ""),
        },
        "report": {"text": _report_text(report),
                   "provenance": report.get("provenance", {})},
        "trace_id": report.get("trace_id"),
        "duration_ms": duration_ms,
    }


def _report_text(report: dict) -> str:
    """The conclusion the visitor reads, composed from the measured report.

    Deliberately assembled here rather than inside the workflow: the workflow
    returns structured facts, and a demo that renders them as prose must not
    invent a single sentence the pipeline did not produce.
    """
    lines = [f"结论状态：{report['status']}；信任级别：{report['trust_level']}。"]
    for claim in report["claims"]:
        lines.append(f"· 有证据支持：{claim['text']}（证据 {', '.join(claim['evidence_ids'])}）")
    for refusal in report["refused_claims"]:
        # Refusals arrive as the auditor's own sentences, not as claim dicts.
        lines.append(f"· 已拒绝：{refusal}")
    for limitation in report["limitations"]:
        lines.append(f"· 限制：{limitation}")
    return "\n".join(lines)


def cases_payload() -> dict:
    return {"cases": [{"demo_id": case["demo_id"], "title": case["title"],
                       "blurb": case["blurb"],
                       "suggested_mode": case.get("suggested_mode", "fixture"),
                       "declares_backend": bool(case.get("backend"))}
                      for case in DEMO_CASES],
            "backend": backend_status()}
