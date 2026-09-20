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


def build_tasks(*, include_end_to_end: bool = True) -> list[Task]:
    tasks: list[Task] = []
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
    return tasks


def count_by_kind(tasks: list[Task]) -> dict:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.kind] = counts.get(task.kind, 0) + 1
    return counts
