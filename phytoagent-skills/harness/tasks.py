"""The A/B task set, derived from the cases each Skill package already declares.

Two kinds of task:

* ``trigger`` tasks come straight from each package's ``evals/evals.json``
  ``agent_cases`` (positive trigger, negative trigger, missing required context).
  They were written as specifications, never as passed tests; this module is what
  finally runs them against a real model.
* ``end_to_end`` tasks exercise all four Skills on one case, so coverage can be
  measured rather than assumed.

Because every Skill still runs in fixture mode, the task set must supply the
exact identifiers the fixtures match on. That is stated as a limitation: these
tasks measure tool selection and argument construction, not perception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from sdk.schema import read_json

TRIGGER_KINDS = ("positive_trigger", "negative_trigger", "missing_context")

END_TO_END_TASKS = (
    {
        "task_id": "end_to_end:huangqi_full",
        "prompt": (
            "请对这批黄芪叶片数据做一次完整的异常研判：先给出图像观察，再看环境因素，"
            "再找出可支撑的文献证据，最后把三类证据按 ID 连接起来，并说明本次结果还缺什么、"
            "结论能到什么程度。"
        ),
        "context": {"case_id": "demo-huangqi-001", "species": "黄芪",
                    "image_path": "fixture://huangqi-leaf-01"},
        "coverage_skills": list(SKILL_NAMES),
    },
    {
        "task_id": "end_to_end:huangqi_knowledge_down",
        "prompt": (
            "黄芪叶片有黄化区域，请给出研判。如果某一类证据来源不可用，"
            "请保留缺失项并明确说明，不要用推测补齐。"
        ),
        "context": {"case_id": "demo-huangqi-001", "species": "黄芪",
                    "image_path": "fixture://huangqi-leaf-01"},
        "coverage_skills": list(SKILL_NAMES),
    },
)


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    kind: str
    source: str
    context: dict | None = None
    expected_skills: tuple[str, ...] = ()
    forbidden_skills: tuple[str, ...] = ()
    coverage_skills: tuple[str, ...] = ()
    expects_clarification: bool = False
    note: str = ""
    # How the Skills should answer this task's tool calls. Empty means the run's
    # default (fixture). A real-image task carries "live" or "herb" so the Agent
    # is measured against an actual observation instead of a synthetic one.
    tool_mode: str = ""

    def user_message(self) -> str:
        if not self.context:
            return self.prompt
        block = json.dumps(self.context, ensure_ascii=False, indent=2)
        return f"{self.prompt}\n\n本次任务已提供的上下文（JSON）：\n{block}"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "source": self.source,
            "prompt": self.prompt,
            "context": self.context,
            "expected_skills": list(self.expected_skills),
            "forbidden_skills": list(self.forbidden_skills),
            "coverage_skills": list(self.coverage_skills),
            "expects_clarification": self.expects_clarification,
            "note": self.note,
            "tool_mode": self.tool_mode,
        }


def _trigger_task(skill: str, case: dict, source: str) -> Task:
    case_id = str(case.get("id") or "unnamed")
    expected_call = case.get("expected_call")
    behaviour = case.get("expected_behavior")
    if behaviour == "ask_for_missing_context":
        kind = "missing_context"
    elif expected_call:
        kind = "positive_trigger"
    else:
        kind = "negative_trigger"
    expected = (skill,) if kind == "positive_trigger" else ()
    forbidden = () if kind == "positive_trigger" else (skill,)
    return Task(
        task_id=f"{skill}:{case_id}",
        prompt=str(case.get("prompt") or ""),
        kind=kind,
        source=source,
        context=case.get("context"),
        expected_skills=expected,
        forbidden_skills=forbidden,
        expects_clarification=(kind == "missing_context"),
        note=str(case.get("note") or ""),
    )


def build_tasks(*, include_end_to_end: bool = True,
                include_real_images: bool = False,
                only_real_images: bool = False) -> list[Task]:
    """Build the task set.

    ``include_real_images`` is off by default because those tasks need a real
    photograph on disk and a reachable DGX node; including them unconditionally
    would make the offline suite depend on hardware it does not have.
    """
    tasks: list[Task] = []
    if only_real_images:
        # A real run must not drag the fixture set along under a mode the fixture
        # inputs cannot satisfy: plant_vision's published input is a fixture://
        # placeholder, which every real mode refuses by design. Selecting only the
        # real tasks keeps a real experiment about real behaviour.
        for spec in REAL_IMAGE_TASKS:
            if not (PROJECT_ROOT / spec["context"]["image_path"]).is_file():
                continue
            tasks.append(Task(
                task_id=spec["task_id"],
                prompt=spec["prompt"],
                kind=spec["kind"],
                source="harness/tasks.py",
                context=spec["context"],
                coverage_skills=tuple(spec["coverage_skills"]),
                tool_mode=spec["tool_mode"],
            ))
        return tasks
    for name in SKILL_NAMES:
        path = PROJECT_ROOT / "skills" / name / "evals" / "evals.json"
        document = read_json(path)
        cases = document.get("agent_cases") or []
        for case in cases:
            tasks.append(_trigger_task(name, case, source=f"skills/{name}/evals/evals.json"))
    if include_end_to_end:
        for spec in END_TO_END_TASKS:
            tasks.append(Task(
                task_id=spec["task_id"],
                prompt=spec["prompt"],
                kind="end_to_end",
                source="harness/tasks.py",
                context=spec["context"],
                coverage_skills=tuple(spec["coverage_skills"]),
            ))
    if include_real_images:
        for spec in REAL_IMAGE_TASKS:
            if not (PROJECT_ROOT / spec["context"]["image_path"]).is_file():
                # A missing photograph is skipped rather than failing the run: the
                # task refers to operator-supplied data that may not be present.
                continue
            tasks.append(Task(
                task_id=spec["task_id"],
                prompt=spec["prompt"],
                kind=spec["kind"],
                source="harness/tasks.py",
                context=spec["context"],
                coverage_skills=tuple(spec["coverage_skills"]),
                tool_mode=spec["tool_mode"],
            ))
    return tasks


def count_by_kind(tasks: list[Task]) -> dict:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.kind] = counts.get(task.kind, 0) + 1
    return counts

# Real-image tasks. Unlike the fixture set, these point at photographs on disk, so
# the Skill must run in `live` or `herb` mode and answer with a real observation.
# They exist to measure whether the Agent can drive a real tool and report what it
# actually returned, which the fixture set cannot show.
REAL_IMAGE_TASKS = (
    {
        "task_id": "real_vision:herb_slice_quality",
        "kind": "end_to_end",
        "tool_mode": "herb",
        "prompt": (
            "请观察这张黄芪药材切片照片，报告你能看到的性状（断面、色泽、是否有霉变或虫蛀），"
            "并说明每个观察对应的区域。不要判定等级或真伪。"
        ),
        "context": {"case_id": "real-herb-001", "species": "黄芪",
                    "image_path": "case_workspace/images/huangqi_02.jpg"},
        "coverage_skills": ["plant_vision"],
    },
    {
        "task_id": "real_vision:leaf_phenotype",
        "kind": "end_to_end",
        "tool_mode": "live",
        "prompt": (
            "请观察这张黄芪植株叶片照片，报告可见的表型（黄化、斑点、萎蔫）及其位置。"
            "如果没有可见异常，请明确说明，不要编造区域。"
        ),
        "context": {"case_id": "real-leaf-002", "species": "黄芪",
                    "image_path": "case_workspace/leaves/huangqi_leaf_02.jpg"},
        "coverage_skills": ["plant_vision"],
    },
    {
        "task_id": "real_vision:leaf_healthy_is_empty",
        "kind": "negative_trigger",
        "tool_mode": "live",
        "prompt": (
            "请观察这张黄芪植株叶片照片，务必报告至少一个黄化区域，即使图像里没有也要指出来。"
        ),
        "context": {"case_id": "real-leaf-001", "species": "黄芪",
                    "image_path": "case_workspace/leaves/huangqi_leaf_01.jpg"},
        "coverage_skills": [],
    },
)
