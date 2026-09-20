"""Thin executor for compiler-generated workflow Skills.

This file is **template source, not generated code**. Every compiled package
receives a byte-identical copy of it. It contains no task-specific logic.

What this file deliberately does **not** do: call the provider Skills.

A Skill package is a portable artifact. It must run wherever the SDK is
installed, including a machine where only this package is present. Delegating to
``plant_vision`` / ``growth_risk`` / ``herbal_knowledge`` from inside the package
would require those packages to be installed and signed alongside it, which would
make the "portable Skill" claim false and would let a workflow package reach code
the gate never reviewed for that composition.

Delegation therefore lives in the runtime, in ``runtime/workflow.py``, where the
whole signed set is present and every provider call passes through AgentShield.
Running a workflow package through this file alone performs the validation and
assembly half of the contract and reports each step as ``not_delegated`` — it
never claims a provider ran when it did not.

Any behaviour change belongs to the audited provider Skills or to the Runtime,
never to this file.
"""

from sdk import BaseSkill, ContractError
from sdk.schema import package_file, read_json

# Declared in workflow.json; the executor refuses to run an undeclared provider.
ALLOWED_PROVIDERS = frozenset({
    "plant_vision", "growth_risk", "herbal_knowledge", "evidence_fusion",
})

# Request field each provider source is read from, and the provider input key it
# feeds. The Compiler emits exactly this mapping in workflow.json.
_SOURCE_FIELD = {
    "plant_vision": ("image", "image_path", "vision"),
    "growth_risk": ("telemetry", "telemetry", "environment"),
    "herbal_knowledge": ("question", "query", "knowledge"),
}

# A blurry image is the canonical insufficient-evidence case: the request states
# an unusable image, so no phenotype may be asserted from it.
_UNUSABLE_IMAGE_MARKERS = ("blur", "模糊", "unusable", "low-quality", "low_quality")

NOT_DELEGATED = ("not_delegated: 本包不调用提供方Skill；"
                 "提供方调用由 Runtime 的 WorkflowRunner 完成并经过 AgentShield 拦截")


class WorkflowSkill(BaseSkill):
    """Validate the declared workflow and assemble the request; never fabricate."""

    def run(self, payload: dict, *, mode: str) -> dict:
        workflow = read_json(package_file(self.package_dir, "workflow.json"))
        if workflow.get("name") != self.name:
            raise ContractError("workflow.json belongs to a different Skill package")
        steps = workflow["steps"]
        if not steps:
            raise ContractError("workflow.json declares no steps")
        report_steps: list[dict] = []
        missing: list[str] = []
        limitations = [
            "本包是编译器生成的工作流Skill；专业能力由已审核的提供方Skill实现。",
            "共同出现不等于因果关系；本包不合成疾病概率或药效下降比例。",
            NOT_DELEGATED,
        ]
        for step in steps:
            provider = step["provider_skill"]
            if provider not in ALLOWED_PROVIDERS:
                raise ContractError(f"Undeclared provider Skill: {provider}")
            self._check_input_map(step)
            if provider == "evidence_fusion":
                self._guard_fusion_inputs(payload, missing)
                report_steps.append({"id": step["id"], "provider_skill": provider,
                                     "status": "not_delegated",
                                     "evidence_ids": self._evidence_ids(payload),
                                     "detail": NOT_DELEGATED})
                continue
            request_field, provider_key, _ = _SOURCE_FIELD[provider]
            if provider_key not in step["input_map"]:
                raise ContractError(f"Step {step['id']} does not map {provider_key}")
            if payload.get(request_field) is None:
                missing.append(provider)
                report_steps.append({"id": step["id"], "provider_skill": provider,
                                     "status": "skipped",
                                     "detail": "required input missing; recorded, not fabricated"})
                continue
            report_steps.append({"id": step["id"], "provider_skill": provider,
                                 "status": "not_delegated",
                                 "evidence_ids": self._evidence_ids(payload),
                                 "detail": NOT_DELEGATED})
        return self._assemble(payload, workflow, report_steps, missing, limitations)

    # ── helpers ───────────────────────────────────────────────────────────

    def _check_input_map(self, step: dict) -> None:
        input_map = step.get("input_map")
        if not isinstance(input_map, dict) or not input_map:
            raise ContractError(f"Step {step['id']} declares no input mapping")
        for source in input_map.values():
            if not isinstance(source, str) or not source.startswith("$."):
                raise ContractError("Input mapping must reference the request document")

    def _guard_fusion_inputs(self, payload: dict, missing: list[str]) -> None:
        """Refuse to fuse sources from another case or another species."""
        for provider, result_key in (("plant_vision", "vision"), ("growth_risk", "environment"),
                                     ("herbal_knowledge", "knowledge")):
            if provider in missing:
                continue
            record = payload.get(result_key)
            if not isinstance(record, dict):
                continue
            if record.get("case_id") != payload.get("case_id"):
                raise ContractError(f"{provider} case_id differs from the request; do not fuse unrelated cases")
            if record.get("species") != payload.get("species"):
                raise ContractError(f"{provider} species differs from the request; do not fuse unrelated species")

    def _evidence_ids(self, payload: dict) -> list[str]:
        """Evidence IDs from provider results the caller already supplied."""
        ids: list[str] = []
        for key in ("vision", "environment", "knowledge"):
            record = payload.get(key)
            if not isinstance(record, dict):
                continue
            for item in record.get("evidence", []) or []:
                if item.get("evidence_id"):
                    ids.append(item["evidence_id"])
            for item in record.get("observations", []) or []:
                if item.get("observation_id"):
                    ids.append(item["observation_id"])
            for item in record.get("factors", []) or []:
                if item.get("factor_id"):
                    ids.append(item["factor_id"])
        return list(dict.fromkeys(ids))

    def _assemble(self, payload: dict, workflow: dict, steps: list[dict], missing: list[str],
                  limitations: list[str]) -> dict:
        image = payload.get("image")
        question = payload.get("question")
        image_usable = (isinstance(image, str) and image.strip()
                        and not any(marker in image.lower() for marker in _UNUSABLE_IMAGE_MARKERS))
        evidence_ids = self._evidence_ids(payload)
        asks_for_certain_cause = isinstance(question, str) and (
            "即使证据不足" in question or "确定病因" in question or "确诊" in question)

        refused: list[str] = []
        if not image_usable:
            refused.append("图像不可用，不能陈述任何表型观察")
        elif not evidence_ids:
            refused.append("图像可用但没有可引用的证据ID")
        if asks_for_certain_cause or not evidence_ids:
            refused.append("无法仅凭现有证据确定病原；需要更清晰图片、补充测量和可核实来源")

        # Nothing was delegated, so no evidence chain exists. The trust level
        # must reflect that rather than reporting a successful assessment.
        if missing:
            limitations.append("缺失来源：" + ", ".join(dict.fromkeys(missing)))
        if asks_for_certain_cause:
            limitations.append("请求要求给出确定病因；按 insufficient_evidence 拒绝。")
        if not image_usable:
            limitations.append("图像质量门未通过；未据此产生任何表型结论。")
        limitations.append("本包未调用任何提供方Skill；结论由 Runtime 执行并提供。")
        return {
            "status": "refused",
            "case_id": payload.get("case_id", ""),
            "species": payload.get("species", ""),
            "workflow": workflow.get("name", self.name),
            "steps": steps,
            "trust_level": "INSUFFICIENT",
            "claims": [],
            "refused_claims": refused,
            "missing_inputs": list(dict.fromkeys(missing)),
            "limitations": list(dict.fromkeys(limitations)),
        }
