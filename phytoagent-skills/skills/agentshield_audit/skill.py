"""Deterministic, offline audit entrypoint.

This Skill is a **callable audit surface**, not the enforcement point. The
enforcement that an Agent cannot route around lives in the Runtime middleware.
Calling this Skill produces an auditable verdict; skipping it does not remove the
middleware's interception.
"""

from sdk import BaseSkill, ContractError


class AgentShieldAuditSkill(BaseSkill):
    """Audit a Skill package or a report, offline and deterministically."""

    def run(self, payload: dict, *, mode: str) -> dict:
        subject = payload["subject"]
        kind = subject["kind"]
        limitations = [
            "本Skill是可调用的审计入口；权限拦截位于Runtime Middleware，不依赖Agent调用本Skill。",
            "审计结论基于本地确定性规则，不调用语言模型，也不构成NVIDIA官方认证。",
        ]
        if kind == "skill_package":
            checks, violations, trust = self._audit_package(subject)
        elif kind == "report":
            checks, violations, trust = self._audit_report(subject)
        else:
            raise ContractError(f"Unsupported audit subject kind: {kind}")
        verdict = "QUARANTINE" if violations and kind == "skill_package" else (
            "INSUFFICIENT" if violations else "PASS")
        status = "refused" if violations else "success"
        # A check that never ran cannot support a PASS, and it cannot support a
        # trust level either. Downgrading only the verdict would leave
        # trust_level: SUPPORTED next to verdict: INCOMPLETE, which reads as an
        # endorsement. Both are gated on completeness.
        incomplete = [check for check in checks if check["status"] == "not_run"]
        if incomplete:
            trust = "LIMITED"
            if not violations:
                verdict = "INCOMPLETE"
                limitations.append(
                    "以下检查未执行：" + ", ".join(check["name"] for check in incomplete)
                    + "；结论不构成完整审计。")
        return {
            "status": status,
            "audit_id": payload["audit_id"],
            "stage": payload["stage"],
            "verdict": verdict,
            "checks": checks,
            "trust_level": trust,
            "violations": list(dict.fromkeys(violations)),
            "provenance": {"data_origin": "local_audit", "skill": self.name,
                           "version": self.version, "model_called": False},
            "limitations": limitations,
        }

    # ── package audit ─────────────────────────────────────────────────────

    def _audit_package(self, subject: dict) -> tuple[list[dict], list[str], str]:
        """Audit a Skill package from its manifest, which the caller supplies.

        The manifest is data the caller already verified through the Registry; the
        audit Skill deliberately does not read arbitrary paths, because a package
        path supplied by an Agent is an attack surface.
        """
        manifest = subject.get("manifest")
        if not isinstance(manifest, dict):
            raise ContractError("A skill_package audit requires subject.manifest as an object")
        declared = (subject.get("declared_permissions") or manifest.get("permissions") or {})
        checks: list[dict] = []
        violations: list[str] = []

        filesystem = declared.get("filesystem")
        if not isinstance(filesystem, list) or not filesystem:
            checks.append({"name": "permission_least_privilege", "status": "failed",
                           "detail": "no filesystem permission declared"})
            violations.append("permission_least_privilege: no filesystem permission declared")
        elif any(p in ("network", "filesystem:write", "*") or "*" in p for p in filesystem):
            checks.append({"name": "permission_least_privilege", "status": "failed",
                           "detail": f"forbidden permission in {filesystem}"})
            violations.append(f"permission_least_privilege: forbidden permission in {filesystem}")
        else:
            checks.append({"name": "permission_least_privilege", "status": "passed",
                           "detail": ", ".join(filesystem)})
        if declared.get("network") != "deny":
            checks.append({"name": "network_denied", "status": "failed",
                           "detail": f"network={declared.get('network')!r}"})
            violations.append(f"network_denied: network={declared.get('network')!r}")
        else:
            checks.append({"name": "network_denied", "status": "passed", "detail": "network deny"})

        # The manifest only proves the file exists; it does not prove a negative
        # eval is in it. When the caller supplies the eval document, its contents
        # are checked; otherwise the check is not_run rather than passed.
        evals_document = subject.get("evals")
        if isinstance(evals_document, dict):
            cases = evals_document.get("cases") or evals_document.get("contract_cases") or []
            agent_cases = evals_document.get("agent_cases") or []
            kinds = {case.get("kind") for case in cases if isinstance(case, dict)}
            agent_kinds = {case.get("kind") for case in agent_cases if isinstance(case, dict)}
            ids = {case.get("id") for case in cases if isinstance(case, dict)}
            has_negative = ("negative_trigger" in kinds or "negative_trigger" in agent_kinds
                            or "reject_unmatched_fixture" in ids
                            or "live_has_no_fixture_fallback" in ids)
            status = "passed" if has_negative else "failed"
            detail = ("negative case present in evals/evals.json" if has_negative
                      else "evals/evals.json declares no negative case")
            if not has_negative:
                violations.append("negative_eval_present: no negative case in evals/evals.json")
        else:
            status = "not_run"
            detail = "evals/evals.json in manifest inventory, but its contents were not supplied"
        checks.append({"name": "negative_eval_present", "status": status, "detail": detail})

        inventory = manifest.get("files")
        checks.append({"name": "manifest_inventory", "status": "passed" if isinstance(inventory, dict)
                       and len(inventory) >= 3 else "failed",
                       "detail": f"{len(inventory) if isinstance(inventory, dict) else 0} hashed files"})
        if not isinstance(inventory, dict) or len(inventory) < 3:
            violations.append("manifest_inventory: fewer than three hashed files")

        trace = subject.get("trace") or []
        traced = all(isinstance(item, dict) and item.get("tool_call_id") for item in trace)
        checks.append({"name": "trace_complete", "status": "passed" if traced and trace else "not_run",
                       "detail": f"{len(trace)} traced calls" if trace else "no trace supplied"})
        if trace and not traced:
            violations.append("trace_complete: a traced call lacks tool_call_id")
        return checks, violations, ("INSUFFICIENT" if violations else "SUPPORTED")

    # ── report audit ──────────────────────────────────────────────────────

    def _audit_report(self, subject: dict) -> tuple[list[dict], list[str], str]:
        report = subject.get("report")
        if not isinstance(report, dict):
            raise ContractError("A report audit requires subject.report as an object")
        trace = subject.get("trace") or []
        available = {item.get("evidence_id") for item in
                     (report.get("evidence") or []) if isinstance(item, dict)}
        available |= {item.get("observation_id") for item in
                      (report.get("observations") or []) if isinstance(item, dict)}
        checks: list[dict] = []
        violations: list[str] = []
        claims = report.get("claims")
        if not isinstance(claims, list):
            raise ContractError("A report must declare a claims array")
        unsupported = [claim for claim in claims
                       if not isinstance(claim, dict) or not claim.get("evidence_ids")]
        checks.append({"name": "claim_evidence_linkage",
                       "status": "passed" if not unsupported else "failed",
                       "detail": f"{len(claims) - len(unsupported)}/{len(claims)} claims cite evidence"})
        for claim in unsupported:
            text = claim.get("text") if isinstance(claim, dict) else "<malformed claim>"
            violations.append(f"claim_evidence_linkage: {text!r} has no evidence id")
        # Fail closed: a cited id that the report cannot account for is
        # unverifiable, including when the report exposes no evidence records at
        # all. Skipping the check on an empty set would let a fabricated citation
        # pass.
        cited = {item for claim in claims if isinstance(claim, dict)
                 for item in (claim.get("evidence_ids") or []) if isinstance(item, str)}
        unknown_ids = sorted(cited - available)
        checks.append({"name": "evidence_verifiable",
                       "status": "passed" if not unknown_ids else "failed",
                       "detail": (f"unknown evidence id: {', '.join(unknown_ids)}"
                                  if unknown_ids else
                                  f"{len(cited)} cited ids all resolvable")})
        if unknown_ids:
            violations.append(
                f"evidence_verifiable: unknown evidence id {', '.join(unknown_ids)}")
        if not trace:
            checks.append({"name": "trace_present", "status": "not_run",
                           "detail": "no trace supplied to the audit"})
        else:
            traced = all(isinstance(item, dict) and item.get("tool_call_id") for item in trace)
            checks.append({"name": "trace_present", "status": "passed" if traced else "failed",
                           "detail": f"{len(trace)} traced calls"})
            if not traced:
                violations.append("trace_present: a traced call lacks tool_call_id")
        trust = "INSUFFICIENT" if violations else ("LIMITED" if len(claims) < 2 else "SUPPORTED")
        return checks, violations, trust
