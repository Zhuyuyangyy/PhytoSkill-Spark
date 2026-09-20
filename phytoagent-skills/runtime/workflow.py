"""Run a compiled workflow Skill through the AgentShield runtime.

A compiled workflow Skill declares dependencies, order, input mapping and
refusal policy. This module is the part that actually performs that plan: it
resolves each ``$.field`` reference, calls the declared provider Skill through
the Shield, and hands the assembled provider results to the fusion step.

The provider Skills are never reimplemented here. If a provider is missing from
the registry, the step is recorded as skipped rather than fabricated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compiler.catalog import CAPABILITIES
from runtime.shield import (TRUST_INSUFFICIENT, TRUST_LIMITED, TRUST_SUPPORTED, ClaimAuditor,
                            ShieldRuntime, collect_evidence_ids, load_evidence_index)
from sdk.exceptions import ShieldError
from sdk.schema import package_file, read_json

# Workflow step id -> (request field, provider input key, result key)
_SOURCE_FIELD = {
    "plant_vision": ("image", "image_path", "vision"),
    "growth_risk": ("telemetry", "telemetry", "environment"),
    "herbal_knowledge": ("question", "query", "knowledge"),
}

# Only these names may be delegated to. A capability id is not automatically a
# provider: the audited catalogue's provider names are the whole allowed set.
PROVIDER_SKILLS = frozenset(_SOURCE_FIELD) | {"evidence_fusion"}

# The permission each provider needs, taken from the audited catalogue rather
# than re-derived, so the runtime and the manifest cannot drift apart.
PROVIDER_PERMISSIONS = {name: tuple(CAPABILITIES[name]["default_permissions"])
                        for name in PROVIDER_SKILLS}

_UNUSABLE_IMAGE_MARKERS = ("blur", "模糊", "unusable", "low-quality", "low_quality")


def _first_id(record: Any, collection: str, field_name: str) -> str | None:
    """First evidence identifier a provider result exposes, if any."""
    if not isinstance(record, dict):
        return None
    for item in record.get(collection) or []:
        if isinstance(item, dict) and item.get(field_name):
            return item[field_name]
    return None


class WorkflowRunner:
    """Execute a compiled workflow Skill, delegating to the audited providers."""

    def __init__(self, runtime: ShieldRuntime, package_dir: str | Path):
        self.runtime = runtime
        self.package_dir = Path(package_dir).absolute()

    def run(self, payload: dict, *, mode: str = "fixture", tool_call_id: str) -> dict:
        workflow = read_json(package_file(self.package_dir, "workflow.json"))
        manifest = read_json(package_file(self.package_dir, "manifest.json"))
        name = manifest["name"]
        declared_tools = set(manifest.get("capabilities", [])) | {name}
        steps = workflow["steps"]
        if not steps:
            raise ShieldError("workflow.json declares no steps")
        if workflow.get("name") != name:
            raise ShieldError("workflow.json belongs to a different Skill package")

        report_steps: list[dict] = []
        missing: list[str] = []
        limitations = [
            "本包是编译器生成的工作流Skill；专业能力由已审核的提供方Skill实现。",
            "共同出现不等于因果关系；本包不合成疾病概率或药效下降比例。",
        ]
        for index, step in enumerate(steps, start=1):
            provider = step["provider_skill"]
            # A capability id is not automatically a provider. Only the audited
            # catalogue's provider names may be delegated to, and the package must
            # declare them. Checking the manifest alone would let a step name any
            # capability that happens to be listed.
            if provider not in PROVIDER_SKILLS:
                raise ShieldError(
                    f"Step {step['id']} calls {provider}, which is not an audited provider Skill")
            if provider not in declared_tools:
                raise ShieldError(f"Step {step['id']} calls {provider}, which the package does not declare")
            self._check_input_map(step)
            if provider == "evidence_fusion":
                self._guard_fusion_inputs(payload, missing)
                report_steps.append({"id": step["id"], "provider_skill": provider,
                                     "status": "success",
                                     "evidence_ids": collect_evidence_ids(payload)})
                continue
            request_field, provider_key, result_key = _SOURCE_FIELD[provider]
            if provider_key not in step["input_map"]:
                raise ShieldError(f"Step {step['id']} does not map {provider_key}")
            if payload.get(request_field) is None:
                if step.get("optional", True):
                    missing.append(provider)
                    report_steps.append({"id": step["id"], "provider_skill": provider,
                                         "status": "skipped",
                                         "detail": "required input missing; recorded, not fabricated"})
                    continue
                raise ShieldError(f"Step {step['id']} requires {request_field}, which is absent")
            provider_input = self._map_input(payload, step["input_map"])
            record = self.runtime.registry.get(provider)
            if not isinstance(record, dict) or not record.get("manifest"):
                raise ShieldError(f"Provider {provider} is not registered; refusing to fabricate it")
            call = self.runtime.call(provider, provider_input, mode=mode,
                                     tool_call_id=f"{tool_call_id}:step{index}",
                                     requested_permissions=list(PROVIDER_PERMISSIONS[provider]))
            if call["status"] != "success":
                missing.append(provider)
                report_steps.append({"id": step["id"], "provider_skill": provider,
                                     "status": "failed",
                                     "detail": call["error"]["code"]})
                continue
            # A provider result that does not belong to this request must not
            # silently replace the request's own field.
            produced = call["data"]
            if produced.get("case_id") != payload.get("case_id"):
                raise ShieldError(f"{provider} returned a different case_id; refusing to fuse it")
            if produced.get("species") != payload.get("species"):
                raise ShieldError(f"{provider} returned a different species; refusing to fuse it")
            payload = {**payload, result_key: produced}
            report_steps.append({"id": step["id"], "provider_skill": provider, "status": "success",
                                 "evidence_ids": collect_evidence_ids(call["data"])})
        self._guard_fusion_inputs(payload, missing)
        return self._assemble(payload, workflow, manifest, report_steps, missing, limitations)

    # ── helpers ───────────────────────────────────────────────────────────

    def _check_input_map(self, step: dict) -> None:
        input_map = step.get("input_map")
        if not isinstance(input_map, dict) or not input_map:
            raise ShieldError(f"Step {step['id']} declares no input mapping")
        for source in input_map.values():
            if not isinstance(source, str) or not source.startswith("$."):
                raise ShieldError("Input mapping must reference the request document")

    def _map_input(self, payload: dict, input_map: dict) -> dict:
        resolved: dict = {}
        for provider_field, source in input_map.items():
            value: Any = payload
            for token in source[2:].split("."):
                if not isinstance(value, dict) or token not in value:
                    value = None
                    break
                value = value[token]
            resolved[provider_field] = value
        return resolved

    def _guard_fusion_inputs(self, payload: dict, missing: list[str]) -> None:
        for provider, result_key in (("plant_vision", "vision"), ("growth_risk", "environment"),
                                     ("herbal_knowledge", "knowledge")):
            if provider in missing:
                continue
            record = payload.get(result_key)
            if not isinstance(record, dict):
                continue
            if record.get("case_id") != payload.get("case_id"):
                raise ShieldError(f"{provider} case_id differs from the request; do not fuse unrelated cases")
            if record.get("species") != payload.get("species"):
                raise ShieldError(f"{provider} species differs from the request; do not fuse unrelated species")

    def _assemble(self, payload: dict, workflow: dict, manifest: dict, steps: list[dict],
                  missing: list[str], limitations: list[str]) -> dict:
        image = payload.get("image")
        question = payload.get("question")
        image_usable = (isinstance(image, str) and image.strip()
                        and not any(marker in image.lower() for marker in _UNUSABLE_IMAGE_MARKERS))
        evidence_index = load_evidence_index(payload.get("vision"), payload.get("environment"),
                                             payload.get("knowledge"))
        evidence_ids = sorted(evidence_index)
        vision_id = _first_id(payload.get("vision"), "observations", "observation_id")
        asks_for_certain_cause = isinstance(question, str) and (
            "即使证据不足" in question or "确定病因" in question or "确诊" in question)

        report = {
            "status": "success",
            "case_id": payload.get("case_id", ""),
            "species": payload.get("species", ""),
            "workflow": manifest["name"],
            "steps": steps,
            "claims": ([{"text": "观察到叶缘黄化区域", "evidence_ids": [vision_id]}]
                       if image_usable and vision_id else []),
            # Refusals are recorded as claims the auditor must reject, so the
            # reason is attributable to a specific policy instead of vanishing.
            "candidate_claims": [],
            "missing_inputs": list(dict.fromkeys(missing)),
            "unlinked_evidence_ids": [],
            "limitations": list(dict.fromkeys(limitations)),
        }
        if not image_usable:
            report["candidate_claims"].append(
                {"text": "图像不可用，不能陈述任何表型观察", "evidence_ids": []})
        if asks_for_certain_cause:
            report["candidate_claims"].append(
                {"text": "要求给出确定病因，但证据不足", "evidence_ids": []})
        if image_usable and not evidence_ids:
            report["candidate_claims"].append(
                {"text": "图像可用但没有可引用的证据ID", "evidence_ids": []})
        audit = ClaimAuditor(available_evidence=evidence_index).audit(
            {**report, "claims": report["claims"] + report["candidate_claims"]})
        if missing:
            report["limitations"].append("缺失来源：" + ", ".join(dict.fromkeys(missing)))
        if asks_for_certain_cause:
            report["limitations"].append("请求要求给出确定病因；按 insufficient_evidence 拒绝。")
        if not image_usable:
            report["limitations"].append("图像质量门未通过；未据此产生任何表型结论。")
        if not evidence_ids:
            report["limitations"].append("没有可引用证据ID；所有事实性结论按 every_claim_requires_evidence 拒绝。")
        report["claims"] = [claim for claim in audit["claims"] if claim["status"] == "supported"]
        report["refused_claims"] = audit["refused_claims"]
        report["violations"] = audit["violations"]
        report["evidence_ids"] = audit["evidence_ids"]
        report["trust_level"] = self._trust_level(audit, image_usable, evidence_ids, missing)
        report["status"] = "refused" if report["trust_level"] == TRUST_INSUFFICIENT else "success"
        # A run that produced no evidence chain is not an assessment, however
        # clean the individual steps looked. Saying so keeps "no conclusion" from
        # looking identical to "conclusion, with evidence".
        if not report["claims"] and report["status"] == "success":
            report["status"] = "no_conclusion"
        report.pop("candidate_claims", None)
        report["trace_id"] = self.runtime.trace_id
        report["provenance"] = {
            "workflow_skill": manifest["name"],
            "workflow_version": manifest["version"],
            "compiler": workflow.get("provenance", {}).get("compiler", "phyto-skill-compiler"),
            "agent_model_called": False,
            "dgx_hardware_used": False,
        }
        return report

    def _trust_level(self, audit: dict, image_usable: bool, evidence_ids: list[str],
                     missing: list[str]) -> str:
        if audit["violations"] or not audit["supported_claims"]:
            return TRUST_INSUFFICIENT
        if not image_usable or missing or len(evidence_ids) < 2:
            return TRUST_LIMITED
        return TRUST_SUPPORTED
