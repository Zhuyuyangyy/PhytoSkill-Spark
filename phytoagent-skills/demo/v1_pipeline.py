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


# The three badges a visitor must be able to tell apart.
BADGE_REPLAY = "replay"              # sealed synthetic, or a recorded observation
BADGE_DETERMINISTIC = "deterministic"  # real computation, no model call
BADGE_REAL = "real"                  # a model actually ran


def _step_marking(step_mode: str, backend: dict) -> tuple[str, str]:
    """(badge, reason) for one step, from the mode it ran in and the backend."""
    if step_mode == "fixture":
        return BADGE_REPLAY, "fixture 模式返回封存的合成案例；未调用任何模型"
    if step_mode == "corpus":
        return BADGE_DETERMINISTIC, "corpus 模式：真实语料索引上的确定性检索；未调用任何模型"
    if backend.get("kind") == "replay":
        return BADGE_REPLAY, (f"replay 后端返回已记录的真实观测"
                              f"（{backend.get('endpoint')}）；本次运行没有发起推理")
    if backend.get("configured"):
        return BADGE_REAL, (f"真实推理：{backend.get('kind')} 后端"
                            f"（{backend.get('endpoint') or backend.get('spec')}）")
    return BADGE_REAL, "请求了真实推理，但没有可用的后端配置"


def run_case(demo_id: str, *, mode: str = "fixture",
             workspace: DemoWorkspace | None = None) -> dict:
    """Run one demo case through the governed pipeline and mark every step."""
    case = case_by_id(demo_id)
    workspace = (workspace or DemoWorkspace()).materialise()
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
        badge, reason = _step_marking(step.get("mode", mode), backend)
        steps.append({
            "id": step.get("id"), "provider": step["provider_skill"],
            "status": step["status"], "mode": step.get("mode", mode),
            "badge": badge, "badge_reason": reason,
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
                       "blurb": case["blurb"]} for case in DEMO_CASES],
            "backend": backend_status()}
