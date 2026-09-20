"""Rule-based, reproducible scoring of recorded runs. No language-model judge.

An LLM judge would introduce a second uncontrolled model into an experiment whose
entire purpose is to isolate one variable. Every verdict below is a deterministic
function of the recorded trace, and every heuristic is labelled as such so a
reader can audit it instead of trusting it.
"""

from __future__ import annotations

STRONG_CLARIFICATION_MARKERS = (
    "请提供", "需要您", "需要提供", "请上传", "请先提供", "缺少图片",
    "需要更多信息", "请补充", "未提供",
)
WEAK_CLARIFICATION_MARKERS = ("缺少", "无法确定", "无法完成", "？", "?")

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NA = "not_applicable"
VERDICT_ERROR = "error"


def _clarification(final_text: str | None) -> tuple[bool | None, str | None, str | None]:
    """Heuristic: does the reply explicitly ask for the missing context?

    Only an explicit request counts. A bare question mark does not, because the
    general-knowledge cases are themselves questions. Reported as an *assisted*
    check that never decides a verdict — free text has no deterministic score.
    """
    if not final_text:
        return None, None, None
    for marker in STRONG_CLARIFICATION_MARKERS:
        if marker in final_text:
            return True, marker, None
    for marker in WEAK_CLARIFICATION_MARKERS:
        if marker in final_text:
            return False, None, marker
    return False, None, None


def score_task(task, run: dict) -> dict:
    selected = list(run.get("selected_skills") or [])
    selected_set = set(selected)
    trace = run.get("trace") or []
    rejected = [item for item in trace if item.get("event") == "tool_call"
                and item.get("status") == "rejected"]
    unknown_tool = [item["name"] for item in rejected
                    if item.get("error_code") == "UnknownTool"]
    argument_errors = [{"name": item["name"], "code": item.get("error_code")}
                       for item in rejected
                       if item.get("error_code") in ("InvalidArgumentsJSON", "ArgumentsNotAnObject")]
    failed_calls = [{"name": item["name"], "code": item.get("error_code")}
                    for item in trace if item.get("event") == "tool_call"
                    and item.get("status") == "failed"]

    trigger_hit = None
    if task.expected_skills:
        trigger_hit = all(skill in selected_set for skill in task.expected_skills)
    forbidden_violated = sorted(skill for skill in task.forbidden_skills
                                if skill in selected_set)
    clarification_asked, marker, weak_marker = _clarification(run.get("final_text"))

    if run.get("stop_reason") == "transport_error":
        verdict, reason = VERDICT_ERROR, "the endpoint failed during this task"
    elif task.kind == "positive_trigger":
        verdict = VERDICT_PASS if trigger_hit else VERDICT_FAIL
        reason = ("expected Skill was selected" if trigger_hit
                  else f"expected {list(task.expected_skills)}, selected {selected}")
    elif task.kind == "negative_trigger":
        verdict = VERDICT_PASS if not forbidden_violated else VERDICT_FAIL
        reason = ("no Skill was selected" if not forbidden_violated
                  else f"selected a Skill that must not trigger: {forbidden_violated}")
    elif task.kind == "missing_context":
        verdict = VERDICT_PASS if not forbidden_violated else VERDICT_FAIL
        reason = ("did not call the Skill without required context"
                  if not forbidden_violated
                  else f"called {forbidden_violated} despite missing required context")
    else:
        verdict, reason = VERDICT_NA, "end-to-end coverage is reported, not scored pass/fail"

    coverage = None
    if task.coverage_skills:
        hit = len(selected_set & set(task.coverage_skills))
        coverage = round(hit / len(task.coverage_skills), 3)

    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "arm": run.get("arm"),
        "selected_skills": selected,
        "executed_skills": list(run.get("executed_skills") or []),
        "expected_skills": list(task.expected_skills),
        "forbidden_skills": list(task.forbidden_skills),
        "trigger_hit": trigger_hit,
        "forbidden_violated": forbidden_violated,
        "unknown_tool_attempts": unknown_tool,
        "argument_errors": argument_errors,
        "failed_tool_calls": failed_calls,
        "coverage": coverage,
        "verdict": verdict,
        "verdict_reason": reason,
        "assisted_checks": {
            "clarification_asked": clarification_asked,
            "clarification_marker": marker,
            "weak_marker_only": weak_marker,
            "method": "explicit-request substring match on the final reply; "
                      "heuristic, never a verdict input",
        },
        "stop_reason": run.get("stop_reason"),
        "turns": run.get("turns"),
        "latency_ms": run.get("latency_ms"),
        "usage_totals": run.get("usage_totals"),
        "models_returned": run.get("models_returned"),
        "final_text": run.get("final_text"),
        "final_text_sha256": run.get("final_text_sha256"),
    }


def summarise(arm: str, scored: list[dict], config_report: dict) -> dict:
    by_kind: dict[str, dict] = {}
    for row in scored:
        bucket = by_kind.setdefault(row["kind"], {"total": 0, "pass": 0, "fail": 0, "error": 0})
        bucket["total"] += 1
        if row["verdict"] == VERDICT_PASS:
            bucket["pass"] += 1
        elif row["verdict"] == VERDICT_FAIL:
            bucket["fail"] += 1
        elif row["verdict"] == VERDICT_ERROR:
            bucket["error"] += 1

    decidable = [row for row in scored if row["verdict"] in (VERDICT_PASS, VERDICT_FAIL)]
    coverage_values = [row["coverage"] for row in scored if row["coverage"] is not None]

    def total(field: str) -> int:
        return sum((row["usage_totals"] or {}).get(field) or 0 for row in scored)

    def latency_total() -> float:
        return round(sum(row["latency_ms"] or 0 for row in scored), 3)

    models = sorted({name for row in scored for name in (row["models_returned"] or [])})
    return {
        "arm": arm,
        "tasks_total": len(scored),
        "by_kind": by_kind,
        "trigger_pass_rate": (
            round(sum(1 for row in decidable if row["verdict"] == VERDICT_PASS) / len(decidable), 3)
            if decidable else None
        ),
        "forbidden_violations": sum(len(row["forbidden_violated"]) for row in scored),
        "unknown_tool_attempts": sum(len(row["unknown_tool_attempts"]) for row in scored),
        "argument_errors": sum(len(row["argument_errors"]) for row in scored),
        "failed_tool_calls": sum(len(row["failed_tool_calls"]) for row in scored),
        "coverage_mean": round(sum(coverage_values) / len(coverage_values), 3) if coverage_values else None,
        "clarification_asked": sum(1 for row in scored
                                   if row["assisted_checks"]["clarification_asked"]),
        "efficiency": {
            "latency_ms_total": latency_total(),
            "latency_ms_mean": round(latency_total() / len(scored), 3) if scored else None,
            "turns_total": sum(row["turns"] or 0 for row in scored),
            "turns_mean": round(sum(row["turns"] or 0 for row in scored) / len(scored), 3) if scored else None,
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "total_tokens": total("total_tokens"),
            "reasoning_tokens": total("reasoning_tokens"),
        },
        "models_returned": models,
        "model_identity_consistent": len(models) <= 1,
        "context": config_report,
    }
