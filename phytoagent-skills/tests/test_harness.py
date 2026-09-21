"""Offline tests for the live Harness.

Every test here runs without a network. HTTP is replaced by a scripted poster, so
the retry logic, the credential hygiene, the tool loop and the scoring are all
exercised deterministically. Nothing in this file may ever reach a real endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.fixture_workspace import SKILL_NAMES, fixture_registry
from harness.ab import AbRunner
from harness.agent import BASE_SYSTEM_PROMPT, AgentHarness
from harness.config import (HarnessConfig, parse_dotenv)
from harness.errors import AuthError, ConfigError, TransportError
from harness.scoring import score_task
from harness.tasks import Task, build_tasks
from harness.transport import Transport
from runtime.executor import SkillExecutor
from sdk.schema import package_file, read_json

# A deliberately non-secret value. Shaped unlike a real credential (hyphenated,
# no long alphanumeric run) so repository-wide secret scans stay quiet.
SECRET = "sk-TEST-ONLY-NOT-A-CREDENTIAL"


class ScriptedPoster:
    """Returns queued ``(status, payload)`` pairs and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def __call__(self, url, body, headers, timeout):
        self.requests.append({"url": url, "body": json.loads(body), "headers": dict(headers)})
        if not self.script:
            raise AssertionError("the scripted poster ran out of responses")
        status, payload = self.script.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return status, text.encode("utf-8")


def completion(model="step-3.7-flash", *, content=None, tool_calls=None, finish="stop",
               usage=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"id": "resp_test", "model": model, "choices": [
        {"index": 0, "finish_reason": finish, "message": message}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def tool_call(name, arguments, call_id="call_1"):
    return [{"id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]


def make_transport(poster, **overrides):
    config = HarnessConfig(api_key=SECRET, **overrides)
    return Transport(config, poster, sleep=lambda _seconds: None,
                   min_request_interval=None), config


def fixture_input(registry, name):
    package = Path(registry.get(name)["package_dir"])
    return read_json(package_file(package, "fixture.json"))["input"]


def build_harness(registry, poster, **overrides):
    transport, config = make_transport(poster, **overrides)
    return AgentHarness(registry, SkillExecutor(registry), transport, config), transport


class TestConfig:
    def test_missing_key_is_a_hard_error_not_a_fixture_fallback(self):
        with pytest.raises(ConfigError) as error:
            HarnessConfig().require_key()
        assert "PHYTO_STEPFUN_API_KEY" in str(error.value)

    def test_repr_and_report_never_contain_the_credential(self):
        config = HarnessConfig(api_key=SECRET)
        assert SECRET not in repr(config)
        assert SECRET not in json.dumps(config.redacted(), ensure_ascii=False)
        assert config.redacted()["api_key_present"] is True

    def test_real_environment_beats_dotenv(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"PHYTO_STEPFUN_API_KEY={SECRET}\nPHYTO_STEPFUN_MODEL=from-dotenv\n",
                          encoding="utf-8")
        config = HarnessConfig.from_env(environ={"PHYTO_STEPFUN_MODEL": "from-env"},
                                        dotenv_path=dotenv)
        assert config.model == "from-env"
        assert config.api_key == SECRET
        assert config.api_key_source == "PHYTO_STEPFUN_API_KEY"

    def test_key_can_come_from_a_file_instead_of_the_environment(self, tmp_path):
        secret_file = tmp_path / "step.key"
        secret_file.write_text(SECRET + "\n", encoding="utf-8")
        config = HarnessConfig.from_env(
            environ={"PHYTO_STEPFUN_API_KEY_FILE": str(secret_file)},
            dotenv_path=tmp_path / "absent.env")
        assert config.api_key == SECRET
        assert config.api_key_source == "PHYTO_STEPFUN_API_KEY_FILE"

    def test_dotenv_parsing_handles_quotes_exports_and_comments(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            "# comment\n\nexport A=1\nB=\"two words\"\nC=plain # trailing\nD='quoted'\n",
            encoding="utf-8")
        assert parse_dotenv(path) == {"A": "1", "B": "two words", "C": "plain", "D": "quoted"}

    def test_malformed_dotenv_line_is_rejected(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("this is not an assignment\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            parse_dotenv(path)

    def test_sampling_parameters_are_only_what_was_actually_set(self):
        assert HarnessConfig().sampling_parameters() == {}
        config = HarnessConfig(temperature=0.0, max_tokens=64, reasoning_effort="low")
        assert config.sampling_parameters() == {"temperature": 0.0, "max_tokens": 64,
                                                "reasoning_effort": "low"}

    def test_invalid_reasoning_effort_is_rejected(self):
        with pytest.raises(ConfigError):
            HarnessConfig(reasoning_effort="ultra")


class TestTransport:
    def test_retries_a_429_then_succeeds(self):
        poster = ScriptedPoster([(429, {"error": "slow down"}), (200, completion(content="ok"))])
        transport, _ = make_transport(poster)
        result = transport.chat({"model": "m", "messages": []})
        assert result.http_status == 200
        assert result.attempts == 2
        assert len(poster.requests) == 2

    def test_does_not_retry_a_400(self):
        poster = ScriptedPoster([(400, {"error": "bad request"})])
        transport, _ = make_transport(poster)
        with pytest.raises(TransportError):
            transport.chat({"model": "m", "messages": []})
        assert len(poster.requests) == 1

    def test_a_rejected_credential_raises_auth_error(self):
        poster = ScriptedPoster([(401, {"error": "invalid key"})])
        transport, _ = make_transport(poster)
        with pytest.raises(AuthError):
            transport.chat({"model": "m", "messages": []})

    def test_error_messages_are_scrubbed_of_the_credential(self):
        poster = ScriptedPoster([(400, {"error": f"bad key {SECRET}"})])
        transport, _ = make_transport(poster)
        with pytest.raises(TransportError) as error:
            transport.chat({"model": "m", "messages": []})
        assert SECRET not in str(error.value)
        assert "REDACTED" in str(error.value)

    def test_non_json_body_is_an_error_not_a_crash(self):
        poster = ScriptedPoster([(502, "<html>gateway</html>")])
        transport, _ = make_transport(poster, max_retries=0)
        with pytest.raises(TransportError):
            transport.chat({"model": "m", "messages": []})

    def test_model_identity_is_captured_and_drift_is_visible(self):
        poster = ScriptedPoster([(200, completion(model="step-3.7-flash-2603")),
                                 (200, completion(model="step-3.7-flash-2501")),
                                 (200, completion(model="step-3.7-flash-2603"))])
        transport, _ = make_transport(poster)
        for _ in range(3):
            transport.chat({"model": "m", "messages": []})
        assert transport.distinct_models == ["step-3.7-flash-2603", "step-3.7-flash-2501"]

    def test_reasoning_tokens_are_read_from_the_usage_block(self):
        payload = completion(content="ok", usage={
            "prompt_tokens": 10, "completion_tokens": 40, "total_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 32}})
        poster = ScriptedPoster([(200, payload)])
        transport, _ = make_transport(poster)
        assert transport.chat({"model": "m", "messages": []}).usage_flat["reasoning_tokens"] == 32

    def test_missing_credential_never_reaches_the_network(self):
        poster = ScriptedPoster([])
        transport = Transport(HarnessConfig(), poster, sleep=lambda _s: None,
                            min_request_interval=None)
        with pytest.raises(ConfigError):
            transport.chat({"model": "m", "messages": []})
        assert poster.requests == []


class TestRequestThrottle:
    """A per-minute limit is met by spacing requests, not by retrying harder."""

    def test_requests_are_spaced_by_the_minimum_interval(self):
        poster = ScriptedPoster([(200, completion()) for _ in range(3)])
        waits: list[float] = []
        clock = {"now": 0.0}

        def sleep(seconds):
            waits.append(seconds)
            clock["now"] += seconds

        config = HarnessConfig(api_key=SECRET)
        transport = Transport(config, poster, sleep=sleep, monotonic=lambda: clock["now"],
                              min_request_interval=6.5)
        for _ in range(3):
            transport.chat({"model": "m", "messages": []})
        # The first request is not delayed; the next two each wait out the window.
        assert waits == [6.5, 6.5]

    def test_an_elapsed_window_is_not_waited_out_twice(self):
        poster = ScriptedPoster([(200, completion()) for _ in range(2)])
        waits: list[float] = []
        clock = {"now": 0.0}

        def sleep(seconds):
            waits.append(seconds)
            clock["now"] += seconds

        config = HarnessConfig(api_key=SECRET)
        transport = Transport(config, poster, sleep=sleep, monotonic=lambda: clock["now"],
                              min_request_interval=6.5)
        transport.chat({"model": "m", "messages": []})
        clock["now"] += 20.0  # the caller was slow on its own
        transport.chat({"model": "m", "messages": []})
        # The first request is never delayed, and the 20 s already spent covers
        # the second gap, so nothing is owed.
        assert waits == []

    def test_throttling_can_be_disabled_for_scripted_posters(self):
        poster = ScriptedPoster([(200, completion()) for _ in range(3)])
        config = HarnessConfig(api_key=SECRET)
        transport = Transport(config, poster, sleep=lambda _s: None, min_request_interval=None)
        for _ in range(3):
            transport.chat({"model": "m", "messages": []})
        assert len(poster.requests) == 3


class TestAgentLoop:
    def test_executes_a_real_skill_and_returns_the_tool_result_to_the_model(self):
        with fixture_registry() as registry:
            arguments = fixture_input(registry, "plant_vision")
            poster = ScriptedPoster([
                (200, completion(tool_calls=tool_call("plant_vision", arguments))),
                (200, completion(content="已根据工具返回结果完成研判。")),
            ])
            harness, _ = build_harness(registry, poster)
            task = Task(task_id="t", prompt="看看这张黄芪叶片", kind="positive_trigger",
                        source="test")
            run = harness.run_task(task, arm="without_skill")

        assert run["selected_skills"] == ["plant_vision"]
        assert run["executed_skills"] == ["plant_vision"]
        assert run["stop_reason"] == "final_answer"
        assert run["turns"] == 2
        assert run["final_text"] == "已根据工具返回结果完成研判。"
        second_call_messages = poster.requests[1]["body"]["messages"]
        tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_1"
        assert json.loads(tool_messages[0]["content"])["status"] == "success"
        assert run["usage_totals"]["total_tokens"] == 30

    def test_an_invented_tool_name_is_recorded_not_executed(self):
        with fixture_registry() as registry:
            poster = ScriptedPoster([
                (200, completion(tool_calls=tool_call("delete_everything", {"path": "/"}))),
                (200, completion(content="无法使用该工具。")),
            ])
            harness, _ = build_harness(registry, poster)
            run = harness.run_task(Task(task_id="t", prompt="x", kind="negative_trigger",
                                        source="test"), arm="without_skill")
        assert run["selected_skills"] == []
        assert run["rejected_calls"] == [{"name": "delete_everything", "code": "UnknownTool"}]

    def test_arguments_that_are_not_json_are_reported_back_to_the_model(self):
        with fixture_registry() as registry:
            broken = [{"id": "call_1", "type": "function",
                       "function": {"name": "plant_vision", "arguments": "{not json"}}]
            poster = ScriptedPoster([
                (200, completion(tool_calls=broken)),
                (200, completion(content="参数有误。")),
            ])
            harness, _ = build_harness(registry, poster)
            run = harness.run_task(Task(task_id="t", prompt="x", kind="positive_trigger",
                                        source="test"), arm="without_skill")
        assert run["rejected_calls"] == [{"name": "plant_vision", "code": "InvalidArgumentsJSON"}]
        assert run["final_text"] == "参数有误。"

    def test_a_contract_rejection_is_surfaced_as_a_failed_tool_call(self):
        with fixture_registry() as registry:
            mismatched = dict(fixture_input(registry, "plant_vision"))
            mismatched["case_id"] = "not-the-fixture-case"
            poster = ScriptedPoster([
                (200, completion(tool_calls=tool_call("plant_vision", mismatched))),
                (200, completion(content="该案例不在 fixture 范围内。")),
            ])
            harness, _ = build_harness(registry, poster)
            run = harness.run_task(Task(task_id="t", prompt="x", kind="positive_trigger",
                                        source="test"), arm="without_skill")
        assert run["selected_skills"] == ["plant_vision"]
        assert run["executed_skills"] == []
        failed = [item for item in run["trace"]
                  if item.get("event") == "tool_call" and item.get("status") == "failed"]
        assert failed and failed[0]["error_code"]

    def test_the_turn_budget_is_enforced(self):
        with fixture_registry() as registry:
            arguments = fixture_input(registry, "plant_vision")
            endless = [(200, completion(tool_calls=tool_call("plant_vision", arguments,
                                                             f"call_{n}"))) for n in range(10)]
            poster = ScriptedPoster(endless)
            harness, _ = build_harness(registry, poster, max_turns=3)
            run = harness.run_task(Task(task_id="t", prompt="x", kind="positive_trigger",
                                        source="test"), arm="without_skill")
        assert run["turns"] == 3
        assert run["stop_reason"] == "max_turns_exceeded"

    def test_a_transport_failure_ends_one_task_without_raising(self):
        with fixture_registry() as registry:
            poster = ScriptedPoster([(400, {"error": "boom"})])
            harness, _ = build_harness(registry, poster, max_retries=0)
            run = harness.run_task(Task(task_id="t", prompt="x", kind="positive_trigger",
                                        source="test"), arm="without_skill")
        assert run["stop_reason"] == "transport_error"
        assert run["final_text"] is None

    def test_the_only_difference_between_arms_is_the_instruction_block(self):
        with fixture_registry() as registry:
            harness, _ = build_harness(registry, ScriptedPoster([]))
            without = harness.system_prompt("without_skill")
            with_skill = harness.system_prompt("with_skill")
            tools_without = harness.tool_definitions()
            tools_with = harness.tool_definitions()
        assert without == BASE_SYSTEM_PROMPT
        assert with_skill.startswith(BASE_SYSTEM_PROMPT)
        assert "plant_vision" in with_skill
        assert with_skill != without
        assert tools_without == tools_with
        assert harness.context_report("with_skill")["instruction_chars_included"] > 0
        assert harness.context_report("without_skill")["instruction_chars_included"] == 0

    def test_an_unknown_arm_is_rejected(self):
        with fixture_registry() as registry:
            harness, _ = build_harness(registry, ScriptedPoster([]))
            with pytest.raises(ValueError):
                harness.system_prompt("with_everything")


class TestScoring:
    def make_run(self, *, selected, text="", stop_reason="final_answer", trace=None):
        return {"arm": "without_skill", "selected_skills": selected, "executed_skills": selected,
                "final_text": text, "final_text_sha256": "", "stop_reason": stop_reason,
                "turns": 1, "latency_ms": 1.0, "models_returned": ["m"], "trace": trace or [],
                "usage_totals": {"prompt_tokens": 1, "completion_tokens": 1,
                                 "total_tokens": 2, "reasoning_tokens": 0}}

    def test_positive_trigger_passes_only_when_the_skill_was_selected(self):
        task = Task(task_id="t", prompt="x", kind="positive_trigger", source="test",
                    expected_skills=("plant_vision",))
        assert score_task(task, self.make_run(selected=["plant_vision"]))["verdict"] == "pass"
        assert score_task(task, self.make_run(selected=[]))["verdict"] == "fail"

    def test_negative_trigger_fails_when_the_forbidden_skill_was_selected(self):
        task = Task(task_id="t", prompt="x", kind="negative_trigger", source="test",
                    forbidden_skills=("plant_vision",))
        assert score_task(task, self.make_run(selected=[]))["verdict"] == "pass"
        assert score_task(task, self.make_run(selected=["plant_vision"]))["verdict"] == "fail"

    def test_missing_context_passes_even_without_an_explicit_request(self):
        task = Task(task_id="t", prompt="x", kind="missing_context", source="test",
                    forbidden_skills=("plant_vision",), expects_clarification=True)
        row = score_task(task, self.make_run(selected=[], text="请提供叶片图片。"))
        assert row["verdict"] == "pass"
        assert row["assisted_checks"]["clarification_asked"] is True
        row = score_task(task, self.make_run(selected=[], text="黄芪可以补气。"))
        assert row["verdict"] == "pass"
        assert row["assisted_checks"]["clarification_asked"] is False

    def test_an_empty_reply_is_not_counted_as_a_clarification(self):
        task = Task(task_id="t", prompt="x", kind="missing_context", source="test",
                    forbidden_skills=("plant_vision",), expects_clarification=True)
        row = score_task(task, self.make_run(selected=[]))
        assert row["verdict"] == "pass"
        assert row["assisted_checks"]["clarification_asked"] is None

    def test_a_bare_question_mark_does_not_count_as_asking_for_context(self):
        task = Task(task_id="t", prompt="x", kind="missing_context", source="test",
                    forbidden_skills=("plant_vision",), expects_clarification=True)
        row = score_task(task, self.make_run(selected=[], text="黄芪是什么？"))
        assert row["assisted_checks"]["clarification_asked"] is False

    def test_end_to_end_is_reported_as_coverage_not_a_pass_fail(self):
        task = Task(task_id="t", prompt="x", kind="end_to_end", source="test",
                    coverage_skills=("a", "b", "c", "d"))
        row = score_task(task, self.make_run(selected=["a", "b"]))
        assert row["verdict"] == "not_applicable"
        assert row["coverage"] == 0.5

    def test_a_transport_failure_scores_as_an_error(self):
        task = Task(task_id="t", prompt="x", kind="positive_trigger", source="test",
                    expected_skills=("plant_vision",))
        row = score_task(task, self.make_run(selected=[], stop_reason="transport_error"))
        assert row["verdict"] == "error"


class TestTaskSet:
    def test_the_agent_cases_each_become_a_task(self):
        tasks = build_tasks()
        kinds = {task.kind for task in tasks}
        assert {"positive_trigger", "negative_trigger", "missing_context", "end_to_end"} <= kinds
        # One positive trigger per professional Skill plus the audit Skill.
        assert len([t for t in tasks if t.kind == "positive_trigger"]) == 5
        # One negative trigger per professional Skill, plus the audit Skill's
        # two: skipping the audit must not bypass middleware interception.
        assert len([t for t in tasks if t.kind == "negative_trigger"]) == 5
        assert len([t for t in tasks if t.kind == "missing_context"]) == 5
        assert all(task.prompt for task in tasks)

    def test_every_professional_skill_ships_a_trigger_task(self):
        tasks = build_tasks()
        for name in SKILL_NAMES:
            assert any(task.task_id.startswith(f"{name}:") for task in tasks), name

    def test_negative_and_missing_context_tasks_forbid_their_own_skill(self):
        for task in build_tasks():
            if task.kind in ("negative_trigger", "missing_context"):
                assert task.forbidden_skills and not task.expected_skills

    def test_context_is_rendered_into_the_user_message(self):
        task = Task(task_id="t", prompt="研判", kind="end_to_end", source="test",
                    context={"case_id": "demo-huangqi-001", "species": "黄芪"})
        message = task.user_message()
        assert "研判" in message and "demo-huangqi-001" in message and "黄芪" in message


class TestAbRunner:
    def test_a_full_run_produces_both_arms_without_a_network(self):
        poster = ScriptedPoster([(200, completion(content="本次无法完成，缺少必要的图片上下文。"))
                                 for _ in range(200)])
        config = HarnessConfig(api_key=SECRET, max_turns=2)
        runner = AbRunner(config, poster=poster, tasks=build_tasks(),
                          min_request_interval=None)
        report = runner.run(repeat=1)

        assert report["scope"] == "agent_ab_real_model_fixture_tools"
        assert set(report["arms_summary"]) == {"without_skill", "with_skill"}
        assert report["task_count"] == 17
        assert len(report["scored"]) == 34
        assert report["agent_model_called"] is True
        assert report["skill_mode"] == "fixture"
        assert report["dgx_hardware_used"] is False
        assert report["model_identity_consistent"] is True
        assert report["scoring_method"] == "rule_based_deterministic"
        assert report["limitations"]
        # no credential material anywhere in the report
        assert SECRET not in json.dumps(report, ensure_ascii=False)
        # negative triggers pass because nothing was called
        negative = [row for row in report["scored"]
                    if row["kind"] == "negative_trigger"]
        assert all(row["verdict"] == "pass" for row in negative)
        # positive triggers fail because nothing was called
        positive = [row for row in report["scored"] if row["kind"] == "positive_trigger"]
        assert all(row["verdict"] == "fail" for row in positive)
        assert report["comparison"]["trigger_pass_rate_delta"] == 0.0

    def test_arms_alternate_order_so_ordering_cannot_fake_an_effect(self):
        assert AbRunner._arm_order(0) == ("without_skill", "with_skill")
        assert AbRunner._arm_order(1) == ("with_skill", "without_skill")

    def test_the_runner_refuses_to_start_without_a_credential(self):
        runner = AbRunner(HarnessConfig(), poster=ScriptedPoster([]), tasks=build_tasks())
        with pytest.raises(ConfigError):
            runner.run(repeat=1)

    def test_every_task_is_still_run_once_per_arm_with_repeats(self):
        poster = ScriptedPoster([(200, completion(content="缺少图片，无法进行。"))
                                 for _ in range(400)])
        runner = AbRunner(HarnessConfig(api_key=SECRET, max_turns=2), poster=poster,
                          tasks=build_tasks()[:3], min_request_interval=None)
        report = runner.run(repeat=2)
        assert report["task_count"] == 3
        assert len(report["scored"]) == 12
        assert report["repeat"] == 2
