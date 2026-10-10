"""Demo V1's case set.

Three cases, chosen to show the three things a visitor must be able to tell
apart after using the demo:

* a **fixture** case — sealed synthetic data, deterministic, no model;
* a **replay** case — a *real* recorded observation served offline, so the
  pipeline's real-mode path is exercised without a GPU;
* a **negative** case — the same pressure to guess that a real user applies,
  where the honest answer is a refusal.

Every payload is a real input the pipeline accepts; nothing here is a mock-up
of a Skill's output.
"""

from __future__ import annotations

FIXTURE_POSITIVE = {
    "demo_id": "fixture-positive",
    "case_id": "demo-huangqi-001",
    "title": "黄芪叶片黄化 · 完整证据链",
    "blurb": ("封存案例：叶片黄化 + 环境遥测 + 本草检索。fixture 模式，"
              "确定性返回，不调用任何模型。"),
    "payload": {
        "case_id": "demo-huangqi-001", "species": "黄芪",
        "image": "fixture://huangqi-leaf-01",
        "telemetry": {"temperature_c": 31.5, "relative_humidity_pct": 40,
                      "soil_ph": 7.8, "npk": None},
        "question": "黄芪叶片黄化可能原因及证据",
    },
}

REPLAY_REAL = {
    "demo_id": "replay-real-quality-gate",
    "title": "真实推理回放 · 质量门拒绝",
    "blurb": ("live 模式 + replay 后端：返回 DGX Spark 上真实推理的已记录观测"
              "（artifacts/dgx/vision-runs.json）。该记录里模型判定图像不可用、"
              "没有区域——于是质量门拒绝，不产生任何表型结论。这是诚实机制在"
              "真实数据上的样子，不是合成的成功路径。"),
    "payload": {
        # The workflow maps the vision leg's image_path from $.image. The
        # case_id stays the demo subject's: the deterministic fixture legs are
        # keyed by it, and a different id would make them refuse — correctly.
        "case_id": "demo-huangqi-001", "species": "黄芪",
        "image": "artifacts/dgx/synthetic-leaf-01.png",
        "telemetry": {"temperature_c": 31.5, "relative_humidity_pct": 40,
                      "soil_ph": 7.8, "npk": None},
        "question": "叶片可见性状及证据",
    },
}

FIXTURE_NEGATIVE = {
    "demo_id": "negative-pressure-to-guess",
    "case_id": "demo-huangqi-001",
    "title": "证据不足 + 要求下定论 · 审计拒绝",
    "blurb": ("模糊图（视觉腿不可用）+ 问题明确要求「即使证据不足也给出确定病因」。"
              "正确输出是拒绝，不是一个猜出来的病因。"),
    "payload": {
        "case_id": "demo-huangqi-001", "species": "黄芪",
        "image": "fixture://huangqi-leaf-blur-01", "telemetry": None,
        "question": "即使证据不足也给出确定病因",
    },
}

DEMO_CASES = (FIXTURE_POSITIVE, REPLAY_REAL, FIXTURE_NEGATIVE)


def case_by_id(demo_id: str) -> dict:
    """Look a case up by its demo id — the selector key, not the payload's case_id.

    Two cases share the subject's case_id on purpose: the negative case is the
    same subject under pressure to guess, and the fixture legs key on the
    subject, not on the demo's intent.
    """
    for case in DEMO_CASES:
        if case["demo_id"] == demo_id:
            return case
    raise KeyError(f"unknown demo case: {demo_id}")
