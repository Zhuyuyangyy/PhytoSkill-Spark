"""Phyto Skill Compiler: a natural-language task becomes one constrained Skill package.

Design contract this module enforces:

* **No arbitrary code generation.** The Compiler only renders text templates and
  writes declarative JSON. The single emitted Python file is a fixed thin
  executor, byte-identical for every compiled package apart from its own name.
* **Composition, never duplication.** The generated package declares
  dependencies, order, input mapping and refusal policy; it reuses the four
  audited provider Skills instead of copying their implementations.
* **Refusal beats guessing.** An unknown species or an empty capability match is
  a hard ``CompileError``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from compiler.catalog import (CAPABILITIES, FIXED_POLICIES, NEGATIVE_EVAL_TEMPLATES, SPECIES_SLUGS,
                              capabilities_in_order, is_composable, union_permissions)
from compiler.intent import Intent, parse_intent
from sdk.exceptions import CompileError
from sdk.manifest import seal_manifest
from sdk.schema import read_json, validate_payload

COMPILER_ID = "phyto-skill-compiler"
COMPILER_VERSION = "0.3.0"
SPEC_SCHEMA = Path(__file__).resolve().parents[1] / "sdk" / "skill_spec.schema.json"
# The emitted executor is fixed source; it is never tailored per task.
EXECUTOR_SOURCE = (Path(__file__).resolve().parent / "templates" / "workflow_executor.py").read_text(encoding="utf-8")

# Structural checks that cannot be expressed in the SkillSpec JSON Schema.
FORBIDDEN_WORKFLOW_KEYS = {"exec", "eval", "shell", "command", "subprocess", "script"}
NAME_RESERVED = ("nvidia", "verified", "official", "oms", "skillspector")


@dataclass
class CompileResult:
    spec: dict
    package_dir: Path
    files: list[str]
    skill_manifest_sha256: str


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def build_name(species: str, task: str) -> str:
    """A stable, lowercase, ASCII package name."""
    species_slug = SPECIES_SLUGS.get(species)
    if species_slug is None:
        raise CompileError(f"Species {species!r} has no approved latin slug; refusing to guess a name")
    name = f"{species_slug}-{_slug(task)}"
    if len(name) > 63:
        name = name[:63].rstrip("-")
    if any(reserved in name for reserved in NAME_RESERVED):
        raise CompileError(f"Generated name {name!r} implies an official verification status")
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name):
        raise CompileError(f"Generated name {name!r} is not a valid Skill package name")
    return name


def build_spec(intent: Intent) -> dict:
    """Turn an Intent into a validated SkillSpec document."""
    if not intent.resolved:
        missing = []
        if not intent.species:
            missing.append("species")
        if not intent.capabilities:
            missing.append("capability")
        raise CompileError("Cannot compile without " + " and ".join(missing))
    capabilities = capabilities_in_order(intent.capabilities)
    if not is_composable(capabilities):
        raise CompileError("A workflow Skill needs at least one producer plus evidence_fusion")
    name = build_name(intent.species, intent.task)
    workflow = []
    for capability in capabilities:
        entry = CAPABILITIES[capability]
        input_map = {}
        for provider_field, request_slot in entry["input_slots"].items():
            input_map[provider_field] = f"$.{request_slot}"
        # A producer whose request field is present must not be skippable: an
        # optional producer turns "input supplied" into "input ignored". Only a
        # step whose request field may legitimately be absent is optional.
        request_slots = set(entry["input_slots"].values())
        always_present = request_slots <= {"case_id", "species"}
        workflow.append({"id": capability, "capability": capability, "input_map": input_map,
                         "optional": capability == "evidence_fusion" or not always_present,
                         "on_missing": "record_and_continue"})
    negative_evals = []
    for kind, template in NEGATIVE_EVAL_TEMPLATES.items():
        if template["per_capability"]:
            for capability in capabilities:
                if capability == "evidence_fusion":
                    continue
                negative_evals.append({"id": f"{kind}-{capability}", "kind": kind,
                                       "expect": template["expect"]})
        else:
            negative_evals.append({"id": f"{kind}-workflow", "kind": kind,
                                   "expect": template["expect"]})
    spec = {
        "spec_version": 1,
        "name": name,
        "display_name": f"{intent.species}{intent.task.replace('-', '')}",
        "species": intent.species,
        "inputs": list(intent.inputs),
        "capabilities": capabilities,
        "permissions": {
            "filesystem": union_permissions(capabilities),
            "network": FIXED_POLICIES["network"],
            "tools": [CAPABILITIES[capability]["provider_skill"] for capability in capabilities],
        },
        "evidence_policy": FIXED_POLICIES["evidence_policy"],
        "refusal_policy": FIXED_POLICIES["refusal_policy"],
        "workflow": workflow,
        "negative_evals": negative_evals,
        "provenance": {"compiler": f"{COMPILER_ID}@{COMPILER_VERSION}",
                       "intent_source": "natural_language_request",
                       "model_called": False},
    }
    _check_spec(spec)
    return spec


def _check_spec(spec: dict) -> None:
    """Structural invariants beyond JSON Schema; called before any file is written."""
    name = spec.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name):
        raise CompileError(f"Invalid Skill package name: {name!r}")
    if any(reserved in name for reserved in NAME_RESERVED):
        raise CompileError(f"Name {name!r} implies an official verification status")
    for step in spec["workflow"]:
        for key in FORBIDDEN_WORKFLOW_KEYS:
            if key in step:
                raise CompileError(f"Workflow step must not declare {key!r}")
    for step in spec["workflow"]:
        for target in step["input_map"].values():
            if not target.startswith("$."):
                raise CompileError("Input mapping must reference the request document")
    providers = [step["capability"] for step in spec["workflow"]]
    if providers != capabilities_in_order(providers):
        raise CompileError("Workflow steps must follow the audited capability order")
    if "evidence_fusion" not in providers:
        raise CompileError("A generated assessment workflow must end in evidence_fusion")
    if spec["permissions"]["network"] != "deny":
        raise CompileError("Generated Skills cannot request network access")
    if not spec["negative_evals"]:
        raise CompileError("Generated Skills must ship at least one negative eval")
    if len({item["id"] for item in spec["negative_evals"]}) != len(spec["negative_evals"]):
        raise CompileError("Negative eval ids must be unique")
    _check_declared_inputs(spec)


def _check_declared_inputs(spec: dict) -> None:
    """The declared request inputs must be exactly what the workflow reads.

    ``spec["inputs"]`` names the canonical slots (``image``, ``question``). The
    check re-derives that set from the workflow's capabilities and compares, so a
    hand-edited spec that declares a field no step reads — or omits one a step
    needs — is rejected instead of producing a schema the runner cannot satisfy.
    ``evidence_fusion`` is excluded because it reads provider results, not
    request fields.
    """
    read_slots: set[str] = set()
    for step in spec["workflow"]:
        if step["capability"] == "evidence_fusion":
            continue  # Fusion reads provider results, not request fields.
        read_slots.update(CAPABILITIES[step["capability"]]["inputs"])
    declared = set(spec.get("inputs") or [])
    if declared != read_slots:
        raise CompileError(
            f"Declared inputs {sorted(declared)} do not match the workflow's "
            f"{sorted(read_slots)}")


def render_skill_md(spec: dict) -> str:
    """SKILL.md is generated text; it carries no authority beyond the manifest."""
    species = spec.get("species", "")
    lines = [
        "---",
        f"name: {spec['name']}",
        f"description: \"{species}健康研判工作流Skill：按声明的顺序调用专业Skill，"
        "每个结论必须关联证据ID，证据不足时拒绝判断。原始图片分析、"
        "无图片无测量的凭空评估、人体疾病诊断和开处方不使用。\"",
        "license: NOASSERTION",
        "---",
        "",
        f"# {spec['name']}",
        "",
        "本Skill是**轻量工作流包**：只声明依赖、顺序、输入映射和拒绝策略，"
        "不复制 plant_vision / growth_risk / herbal_knowledge / evidence_fusion 的实现。",
        "",
        "## 触发",
        "",
        f"输入包含图片、带单位测量或本地语料检索需求，且物种为{species}时触发。",
        "",
        "## 不触发",
        "",
        "- 纯知识问答，没有图片、测量或语料检索需求。",
        "- 要求编造文献、给出处方或把表型写成病理确诊。",
        "- 与已声明物种不同的案例。",
        "",
        "## 执行顺序",
        "",
    ]
    for step in spec["workflow"]:
        lines.append(f"{spec['workflow'].index(step) + 1}. `{step['id']}` — "
                     f"{CAPABILITIES[step['capability']]['description']}")
    lines += [
        "",
        "## 拒绝策略",
        "",
        f"- 证据策略：`{spec['evidence_policy']}`。",
        f"- 拒绝策略：`{spec['refusal_policy']}`。",
        "- 任一来源缺失时记入 missing_inputs，不借用其他案例结果。",
        "- 任一来源案例 ID 或物种不一致时拒绝融合。",
        "",
        "## 执行",
        "",
        "```bash",
        "python scripts/run.py --input examples/request.json --mode fixture \\",
        "    --public-key /trusted/publisher.public.pem",
        "```",
        "",
        "行为用例见 evals/evals.json；权限与依赖见 skill-card.md。",
        "",
    ]
    return "\n".join(lines)


def render_skill_card(spec: dict) -> str:
    tools = ", ".join(spec["permissions"]["tools"])
    filesystem = ", ".join(spec["permissions"]["filesystem"])
    return "\n".join([
        "---",
        f"name: {spec['name']}",
        f"version: {COMPILER_VERSION}",
        "owner: PhytoForge Spark compiler",
        "license: NOASSERTION",
        "license_status: owner_decision_required_before_public_release",
        "deployment_geography: local-development-only",
        "runtime: local_python",
        "execution_modes: [fixture]",
        f"network_access: {spec['permissions']['network']}",
        f"filesystem_permissions: [{filesystem}]",
        f"tool_dependencies: [{tools}]",
        "model_weights: none",
        "nvidia_verified: false",
        "security_scan: not_run",
        "agent_ab_evaluation: not_run",
        "---",
        "",
        "# Skill Card",
        "",
        "工作流包，不含模型权重，不联网。",
        "",
        "## 权限",
        "",
        f"- 文件系统：{filesystem}（只读，按最小权限授予）。",
        "- 网络：deny。",
        "- 工具：仅可调用声明的专业 Skill。",
        "",
        "## 风险与限制",
        "",
        "- 本包不生成任意代码；执行器为固定模板。",
        "- 融合结果为 co_occurrence_only，不是因果结论，也不是疾病概率。",
        "- 表型观察不能确诊病原；本草药性知识不能替代栽培病理证据。",
        "",
    ])


def render_workflow_json(spec: dict) -> dict:
    """workflow.json is the declarative execution plan consumed by the executor."""
    steps = []
    for step in spec["workflow"]:
        capability = step["capability"]
        entry = CAPABILITIES[capability]
        steps.append({
            "id": step["id"],
            "provider_skill": entry["provider_skill"],
            "input_map": step["input_map"],
            "optional": step["optional"],
            "on_missing": step["on_missing"],
            "failure_mode": entry["failure_mode"],
            "expected_outputs": entry["outputs"],
        })
    return {
        "schema_version": 1,
        "name": spec["name"].replace("-", "_"),
        "evidence_policy": spec["evidence_policy"],
        "refusal_policy": spec["refusal_policy"],
        "inputs": spec["inputs"],
        # The permission axis is carried here too. Without it this file would be a
        # lossy restatement of the plan: the runtime reads the permissions it
        # needs from here, so the executed plan's authorisation must be visible.
        "permissions": spec["permissions"],
        "steps": steps,
        "provenance": spec["provenance"],
    }


def render_example_request(spec: dict) -> dict:
    """The minimal valid request for the declared inputs.

    This is deliberately not the positive eval case: the example shows a caller
    what the package accepts, while the eval case is tied to the providers'
    published fixtures. Conflating the two would make the example unusable for any
    species whose fixture does not exist.
    """
    request: dict = {"case_id": f"{spec['name']}-example", "species": spec["species"]}
    for slot in spec["inputs"]:
        if slot == "image":
            request["image"] = "case_workspace/leaf.jpg"
        elif slot == "telemetry":
            request["telemetry"] = {"temperature_c": None, "relative_humidity_pct": None,
                                    "soil_ph": None, "npk": None}
        else:
            request[slot] = f"{spec['species']}叶片黄化可能原因及证据"
    return request


def render_evals(spec: dict) -> dict:
    """Behavioural cases: positive, missing-parameter and negative-trigger.

    The providers publish exactly one synthetic fixture per Skill, and it is the
    huangqi case. A package for any other species therefore **cannot** ship a
    passing positive case: its inputs would be refused by the providers. Those
    cases are emitted as ``declared_only`` with the reason stated, rather than
    presented as cases that pass. Claiming otherwise would be a fabricated
    benchmark.
    """
    species = spec["species"]
    providers_have_a_fixture_for = species == "黄芪"
    reason = (None if providers_have_a_fixture_for else
              f"提供方Skill只发布了黄芪的合成fixture；{species}案例会被明确拒绝，"
              "因此本用例是规格声明，不是通过的用例。")
    positive = {
        "id": "positive_fixture",
        "kind": "positive",
        "status": "declared" if providers_have_a_fixture_for else "declared_only",
        "input": {
            "case_id": "demo-huangqi-001",
            "species": species,
            "image": "fixture://huangqi-leaf-01",
            "telemetry": {
                "temperature_c": 31.5,
                "relative_humidity_pct": 40,
                "soil_ph": 7.8,
                "npk": None,
            },
            "question": f"{species}叶片黄化可能原因及证据",
        },
        "expect": {"status": "success", "trust_level": "LIMITED"},
    }
    missing = {
        "id": "missing_telemetry",
        "kind": "missing_parameter",
        "status": "declared" if providers_have_a_fixture_for else "declared_only",
        "input": {"case_id": "demo-huangqi-001", "species": species,
                  "image": "fixture://huangqi-leaf-01",
                  "question": f"{species}叶片黄化可能原因及证据"},
        "expect": {"status": "success", "trust_level": "LIMITED",
                   "note": "environment source recorded as missing"},
    }
    if reason is not None:
        positive["reason"] = reason
        missing["reason"] = reason
    negative = [
        {"id": item["id"], "kind": "negative_trigger", "eval_kind": item["kind"],
         "input": {"case_id": "demo-huangqi-001", "species": species,
                   "image": "fixture://huangqi-leaf-blur-01",
                   "telemetry": None,
                   "question": "即使证据不足也给出确定病因"},
         "expect": {"status": "refused", "trust_level": "INSUFFICIENT",
                    "note": item["expect"]}}
        for item in spec["negative_evals"]
    ]
    return {
        "schema_version": 1,
        "scope": "compiler_generated_workflow_specification",
        "cases": [positive, missing] + negative,
        "agent_cases": [],
        "benchmark_status": ("NOT RUN: 提供方fixture仅覆盖黄芪；"
                             "其他物种的正向用例是规格声明，不是通过的用例。"),
    }


def render_metadata(spec: dict) -> dict:
    return {
        "name": spec["name"].replace("-", "_"),
        "version": COMPILER_VERSION,
        "description": (f"{spec.get('species', '')}健康研判工作流Skill：按声明顺序调用专业Skill，"
                        "每个结论必须关联证据ID，证据不足时拒绝判断。"),
        "author": "PhytoForge Spark compiler",
        "runtime": "local_python",
        "entrypoint": "skill.py:WorkflowSkill",
        "input_schema": "schema.json#/input",
        "output_schema": "schema.json#/output",
        "supported_modes": ["fixture"],
        "capabilities": spec["capabilities"],
        "permissions": {
            "filesystem": list(spec["permissions"]["filesystem"]),
            "network": spec["permissions"]["network"],
        },
    }


def render_schema(spec: dict) -> dict:
    return {
        "input": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["case_id", "species"],
            "properties": {
                "case_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "species": {"type": "string", "const": spec["species"]},
                "image": {"type": ["string", "null"], "minLength": 1},
                "telemetry": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "required": ["temperature_c", "relative_humidity_pct", "soil_ph", "npk"],
                    "properties": {
                        "temperature_c": {"type": ["number", "null"], "minimum": -30, "maximum": 70},
                        "relative_humidity_pct": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
                        "soil_ph": {"type": ["number", "null"], "minimum": 0, "maximum": 14},
                        "npk": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "object", "additionalProperties": False,
                                 "required": ["nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg"],
                                 "properties": {
                                     "nitrogen_mg_kg": {"type": "number", "minimum": 0},
                                     "phosphorus_mg_kg": {"type": "number", "minimum": 0},
                                     "potassium_mg_kg": {"type": "number", "minimum": 0}}},
                            ]
                        },
                    },
                },
                "question": {"type": "string", "minLength": 1, "maxLength": 500},
            },
        },
        "output": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "case_id", "species", "workflow", "steps", "trust_level",
                         "claims", "refused_claims", "missing_inputs", "limitations"],
            "properties": {
                "status": {"enum": ["success", "refused", "no_conclusion"]},
                "case_id": {"type": "string", "minLength": 1},
                "species": {"type": "string", "minLength": 1},
                "workflow": {"type": "string", "minLength": 1},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "provider_skill", "status"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "provider_skill": {"type": "string", "minLength": 1},
                            # "not_delegated" is the honest report when a package
                            # runs without the runtime: the step was validated but
                            # no provider was called. Reporting "success" there
                            # would claim work that did not happen.
                            "status": {"enum": ["success", "failed", "skipped",
                                                "refused", "not_delegated"]},
                            "tool_call_id": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "detail": {"type": "string"},
                        },
                    },
                },
                "trust_level": {"enum": ["SUPPORTED", "LIMITED", "INSUFFICIENT"]},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "evidence_ids", "status"],
                        "properties": {
                            "text": {"type": "string", "minLength": 1},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "status": {"enum": ["supported", "refused"]},
                        },
                    },
                },
                "refused_claims": {"type": "array", "items": {"type": "string"}},
                "missing_inputs": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
        },
    }


def validate_spec(spec: dict) -> dict:
    """Validate a SkillSpec document without writing any file.

    Runs the structural invariants plus the SkillSpec JSON Schema, so a spec can
    be checked in isolation (for example an incoming pull request or a package
    under audit) before the Compiler renders it.
    """
    _check_spec(spec)
    schema = read_json(SPEC_SCHEMA)
    validate_payload(spec, schema, label="SkillSpec")
    return spec


def compile_request(request: str, output_dir: str | Path, *, seal: bool = False) -> CompileResult:
    """Compile one natural-language request into a Skill package directory.

    ``seal=False`` writes the package without touching any manifest hash, so the
    AgentShield compile gate can inspect the raw output first.
    """
    intent = parse_intent(request)
    spec = build_spec(intent)
    return compile_spec(spec, output_dir, seal=seal)


def compile_spec(spec: dict, output_dir: str | Path, *, seal: bool = False) -> CompileResult:
    """Render and write one Skill package from an already validated SkillSpec."""
    _check_spec(spec)
    package = Path(output_dir) / spec["name"]
    package.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def write(relative: str, content: str) -> None:
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
        written.append(relative)

    def write_json(relative: str, payload: dict) -> None:
        write(relative, json.dumps(payload, ensure_ascii=False, indent=2))

    write("SKILL.md", render_skill_md(spec))
    write("skill-card.md", render_skill_card(spec))
    write("skill_spec.json", json.dumps(spec, ensure_ascii=False, indent=2))
    write("workflow.json", json.dumps(render_workflow_json(spec), ensure_ascii=False, indent=2))
    write("schema.json", json.dumps(render_schema(spec), ensure_ascii=False, indent=2))
    write("evals/evals.json", json.dumps(render_evals(spec), ensure_ascii=False, indent=2))
    write("examples/request.json", json.dumps(render_example_request(spec),
                                              ensure_ascii=False, indent=2))
    write("skill.py", EXECUTOR_SOURCE)
    write("metadata.json", json.dumps(render_metadata(spec), ensure_ascii=False, indent=2))
    write("references/contract.md",
          "本包是编译器生成的工作流Skill，复用四个已审核专业Skill，不复制其实现。\n"
          "融合关系为 co_occurrence_only，不构成因果结论或疾病概率。\n")
    write("BENCHMARK.md",
          "# Evaluation status\n\n"
          "- Compiler contract checks: run `python -m compiler.__main__ verify`.\n"
          "- Agent A/B with and without this Skill: **NOT RUN**.\n"
          "- Plant diagnosis accuracy and DGX latency: **NOT MEASURED**.\n")
    if seal:
        manifest = seal_manifest(package, read_json(package / "metadata.json"))
        written.append("manifest.json")
        return CompileResult(spec=spec, package_dir=package, files=written,
                             skill_manifest_sha256=manifest["manifest_sha256"])
    return CompileResult(spec=spec, package_dir=package, files=written, skill_manifest_sha256="")
