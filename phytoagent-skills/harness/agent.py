"""The Agent Harness: a real model plans and calls locally verified Skills.

The single experimental variable between arms is whether the Skill *instructions*
are present in the system prompt. Tool definitions, model, endpoint, sampling
parameters, task set and loop logic are byte-identical across arms.

What is real here and what is not, stated plainly:

* Real: the planning model, the tool-call protocol, the selection decisions, the
  argument construction, the round trips, the latency and the token counts.
* Not real: the content returned by the tools. Every Skill still runs in
  ``fixture`` mode over synthetic cases, so a tool result is a synthetic
  observation, not a plant diagnosis. ``dgx_hardware_used`` stays false.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from harness.config import HarnessConfig
from harness.errors import HarnessError
from harness.transport import Transport
from sdk.exceptions import SkillError

BASE_SYSTEM_PROMPT = (
    "你是药用植物异常研判任务的调度者。"
    "根据用户任务决定是否调用可用工具。"
    "工具返回的内容是本次唯一可用的事实来源：不要编造观测、文献、测量值或区域坐标。"
    "如果任务缺少完成所必需的信息，直接说明缺少什么，不要用工具猜测替代。"
)

MAX_TRACE_TEXT_CHARS = 4000
MAX_TOOL_TEXT_CHARS = 20000

ARMS = ("without_skill", "with_skill")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clip(text: str | None, limit: int = MAX_TRACE_TEXT_CHARS) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class AgentHarness:
    def __init__(self, registry, executor, transport: Transport, config: HarnessConfig):
        self.registry = registry
        self.executor = executor
        self.transport = transport
        self.config = config
        self._instructions_cache: dict[str, dict] | None = None

    @property
    def skill_instructions(self) -> dict[str, dict]:
        """Load every Skill's instruction text once, with its manifest hash.

        Loaded eagerly so that the instruction block is fixed before the first
        model call and cannot drift between arms.
        """
        if self._instructions_cache is None:
            loaded = {}
            for entry in self.registry.catalog:
                name = entry["name"]
                detail = self.registry.load_skill(name)
                loaded[name] = {
                    "manifest_sha256": detail["manifest_sha256"],
                    "instructions": detail["instructions"],
                    "instructions_sha256": digest(detail["instructions"]),
                    "instructions_chars": len(detail["instructions"]),
                }
            self._instructions_cache = loaded
        return self._instructions_cache

    def tool_definitions(self) -> list[dict]:
        return self.registry.available_tools

    def system_prompt(self, arm: str) -> str:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
        if arm == "without_skill":
            return BASE_SYSTEM_PROMPT
        blocks = [BASE_SYSTEM_PROMPT, "", "# 可用 Skill 的完整指令", ""]
        for name, detail in sorted(self.skill_instructions.items()):
            blocks += [f"## {name}", detail["instructions"].strip(), ""]
        return "\n".join(blocks)

    def context_report(self, arm: str) -> dict:
        prompt = self.system_prompt(arm)
        return {
            "arm": arm,
            "system_prompt_chars": len(prompt),
            "system_prompt_sha256": digest(prompt),
            "tool_count": len(self.tool_definitions()),
            "instructions_included": sorted(self.skill_instructions) if arm == "with_skill" else [],
            "instruction_chars_included": (
                sum(d["instructions_chars"] for d in self.skill_instructions.values())
                if arm == "with_skill" else 0
            ),
        }

    def _execute_tool_call(self, call: dict, *, turn: int) -> tuple[dict, dict]:
        """Run one tool call and return the ``role: tool`` message plus its trace record."""
        function = call.get("function") or {}
        name = function.get("name")
        raw_arguments = function.get("arguments")
        call_id = call.get("id") or f"missing-id-{turn}"
        record: dict[str, Any] = {
            "event": "tool_call",
            "turn": turn,
            "tool_call_id": call_id,
            "name": name,
            "arguments_sha256": digest(raw_arguments if isinstance(raw_arguments, str) else ""),
        }
        known = {entry["name"] for entry in self.registry.catalog}

        if name not in known:
            record.update(status="rejected", error_code="UnknownTool")
            return self._tool_message(call_id, {
                "status": "failed",
                "error": {"code": "UnknownTool", "message": f"No such tool: {name!r}"},
            }), record

        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                record.update(status="rejected", error_code="InvalidArgumentsJSON")
                return self._tool_message(call_id, {
                    "status": "failed",
                    "error": {"code": "InvalidArgumentsJSON",
                              "message": f"arguments is not valid JSON: {exc.msg}"},
                }), record
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            record.update(status="rejected", error_code="ArgumentsNotAnObject")
            return self._tool_message(call_id, {
                "status": "failed",
                "error": {"code": "ArgumentsNotAnObject",
                          "message": f"arguments must be a JSON object, got {type(arguments).__name__}"},
            }), record

        # Second disclosure level: expand the selected package before executing it.
        try:
            loaded = self.registry.load_skill(name)
            record["manifest_sha256"] = loaded["manifest_sha256"]
        except SkillError as exc:
            record.update(status="rejected", error_code=type(exc).__name__)
            return self._tool_message(call_id, {
                "status": "failed",
                "error": {"code": type(exc).__name__, "message": str(exc)},
            }), record

        result = self.executor.call(name, arguments, mode="fixture", tool_call_id=call_id)
        record.update(
            status=result["status"],
            duration_ms=result["duration_ms"],
            output_sha256=digest(json.dumps(result.get("data"), ensure_ascii=False, sort_keys=True)),
        )
        if result["status"] != "success":
            record["error_code"] = (result.get("error") or {}).get("code")
        else:
            data = result["data"] or {}
            if isinstance(data, dict) and "completeness" in data:
                record["completeness"] = data["completeness"]
        return self._tool_message(call_id, result), record

    def _tool_message(self, call_id: str, payload: dict) -> dict:
        text = json.dumps(payload, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": call_id,
                "content": text[:MAX_TOOL_TEXT_CHARS]}

    def run_task(self, task, *, arm: str) -> dict:
        """Run one task in one arm. Never raises for a model-side failure."""
        config = self.config
        messages = [
            {"role": "system", "content": self.system_prompt(arm)},
            {"role": "user", "content": task.user_message()},
        ]
        tools = self.tool_definitions()
        sampling = config.sampling_parameters()
        trace: list[dict] = []
        tool_records: list[dict] = []
        final_text: str | None = None
        stop_reason = "max_turns_exceeded"
        turns = 0

        for turn in range(1, config.max_turns + 1):
            turns = turn
            payload = {"model": config.model, "messages": messages,
                       "tools": tools, "tool_choice": "auto", **sampling}
            try:
                completion = self.transport.chat(payload)
            except HarnessError as exc:
                # One unreachable turn must not abort the whole experiment: the
                # task is recorded as failed so the run and its report survive.
                trace.append({"event": "transport_error", "turn": turn,
                              "error": f"{type(exc).__name__}: {exc}"})
                stop_reason = "transport_error"
                break
            calls = completion.tool_calls
            trace.append({
                "event": "model_response",
                "turn": turn,
                "model_returned": completion.model_returned,
                "finish_reason": completion.finish_reason,
                "tool_call_count": len(calls),
                "tool_call_names": [(c.get("function") or {}).get("name") for c in calls],
                "content_chars": len(completion.message.get("content") or ""),
                "usage": completion.usage_flat,
                "latency_ms": completion.latency_ms,
                "attempts": completion.attempts,
                "response_id": completion.response_id,
            })
            if not calls:
                final_text = completion.message.get("content") or ""
                messages.append({"role": "assistant", "content": final_text})
                stop_reason = "final_answer"
                break
            messages.append({"role": "assistant",
                             "content": completion.message.get("content"),
                             "tool_calls": calls})
            for call in calls:
                message, record = self._execute_tool_call(call, turn=turn)
                tool_records.append(record)
                trace.append(record)
                messages.append(message)

        requests = [item for item in trace if item["event"] == "model_response"]
        usage_totals = {key: sum((item["usage"].get(key) or 0) for item in requests)
                        for key in ("prompt_tokens", "completion_tokens",
                                    "total_tokens", "reasoning_tokens")}
        return {
            "task_id": task.task_id,
            "arm": arm,
            "stop_reason": stop_reason,
            "turns": turns,
            "selected_skills": [record["name"] for record in tool_records
                                if record.get("status") != "rejected"],
            "executed_skills": [record["name"] for record in tool_records
                                if record.get("status") == "success"],
            "rejected_calls": [{"name": record["name"], "code": record.get("error_code")}
                               for record in tool_records if record.get("status") == "rejected"],
            "final_text": final_text,
            "final_text_sha256": digest(final_text or ""),
            "usage_totals": usage_totals,
            "latency_ms": round(sum(item["latency_ms"] for item in requests), 3),
            "models_returned": sorted({item["model_returned"] for item in requests
                                       if item["model_returned"]}),
            "trace": trace,
        }
