"""AgentShield Runtime: the middleware an Agent cannot route around.

Two distinct concerns live here, deliberately separated:

* :class:`ShieldRuntime` enforces permissions and tracing around **every** tool
  call. An Agent that simply declines to call the audit Skill is still
  intercepted, because the interception is not a Skill.
* :class:`ClaimAuditor` turns a finished report into auditable claims. It runs
  after execution and refuses unsupported claims.

Nothing here calls a language model. Trust levels are derived from evidence
coverage and hard violations, never from a fabricated 0-100 score.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from registry.loader import SkillRegistry
from runtime.executor import SkillExecutor
from sdk.exceptions import (BudgetExceeded, PermissionViolation, ShieldError, TraceError)

TRUST_SUPPORTED = "SUPPORTED"
TRUST_LIMITED = "LIMITED"
TRUST_INSUFFICIENT = "INSUFFICIENT"


@dataclass
class CallRecord:
    trace_id: str
    tool_call_id: str
    skill: str
    manifest_sha256: str
    status: str
    duration_ms: float
    permissions_used: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id, "tool_call_id": self.tool_call_id, "skill": self.skill,
            "manifest_sha256": self.manifest_sha256, "status": self.status,
            "duration_ms": self.duration_ms, "permissions_used": self.permissions_used,
            "evidence_ids": self.evidence_ids, "detail": self.detail,
        }


class ShieldRuntime:
    """Permission-checking, tracing wrapper around the local Skill executor."""

    def __init__(self, registry: SkillRegistry, *, max_calls: int = 32,
                 trace_id: str | None = None):
        self.registry = registry
        self.executor = SkillExecutor(registry)
        self.max_calls = max_calls
        self.trace_id = trace_id or f"trace-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        self._calls = 0
        self.audit_log: list[CallRecord] = []

    # ── call interception ─────────────────────────────────────────────────

    def call(self, name: str, payload: dict, *, mode: str = "fixture",
             tool_call_id: str, requested_permissions: list[str] | None = None,
             case_workspace: Path | None = None) -> dict:
        """Authorise, trace, execute, and record one tool call."""
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise TraceError("Every tool call must carry a non-empty tool_call_id")
        if self._calls >= self.max_calls:
            raise BudgetExceeded(f"Session exceeded the {self.max_calls}-call budget")
        self._calls += 1
        record = self.registry.verify_entry(name)
        manifest = record["manifest"]
        start = perf_counter()
        try:
            self._authorise(name, manifest, requested_permissions or [], case_workspace)
        except ShieldError as exc:
            self.audit_log.append(CallRecord(
                trace_id=self.trace_id, tool_call_id=tool_call_id, skill=name,
                manifest_sha256=manifest["manifest_sha256"], status="blocked",
                duration_ms=round((perf_counter() - start) * 1000, 3),
                permissions_used=sorted(set(requested_permissions or [])), detail=str(exc)))
            raise
        result = self.executor.call(name, payload, mode=mode, tool_call_id=tool_call_id)
        self.audit_log.append(CallRecord(
            trace_id=self.trace_id, tool_call_id=tool_call_id, skill=name,
            manifest_sha256=manifest["manifest_sha256"], status=result["status"],
            duration_ms=result["duration_ms"],
            permissions_used=sorted(set(requested_permissions or [])),
            evidence_ids=collect_evidence_ids(result.get("data")),
            detail="" if result["status"] == "success" else result["error"]["code"]))
        return result

    def _authorise(self, name: str, manifest: dict, requested: list[str],
                   case_workspace: Path | None) -> None:
        """Least-privilege check against the manifest the package was sealed with."""
        declared = manifest.get("permissions", {})
        allowed_filesystem = set(declared.get("filesystem", []))
        for permission in requested:
            if permission == "network":
                raise PermissionViolation(f"{name} requested network access; the package declares none")
            # "tool"/"tools" is the capability axis, checked separately below.
            if permission not in allowed_filesystem and permission not in ("tool", "tools"):
                raise PermissionViolation(
                    f"{name} requested undeclared filesystem permission {permission!r}")
        if case_workspace is not None:
            if case_workspace.is_symlink() or not case_workspace.is_dir():
                raise PermissionViolation("case_workspace must be an existing directory, not a link")

    # ── reporting ─────────────────────────────────────────────────────────

    @property
    def trace(self) -> list[dict]:
        return [record.to_dict() for record in self.audit_log]

    def report(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "scope": "agentshield_runtime_local",
            "agent_model_called": False,
            "dgx_hardware_used": False,
            "nvidia_verified": False,
            "calls": self.trace,
            "blocked": [r.to_dict() for r in self.audit_log if r.status == "blocked"],
        }


def perf_counter() -> float:
    return time.perf_counter()


def collect_evidence_ids(data: Any) -> list[str]:
    """Collect every evidence identifier a Skill result exposes."""
    ids: list[str] = []
    if not isinstance(data, dict):
        return ids
    for key in ("evidence", "observations", "factors"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                for field_name in ("evidence_id", "observation_id", "factor_id"):
                    if item.get(field_name):
                        ids.append(item[field_name])
    return list(dict.fromkeys(ids))


# ── Claim-Evidence audit ─────────────────────────────────────────────────────

# Statements that assert certainty the evidence cannot carry. This is a
# deterministic denylist, not a language model judgement. "已确定" and "确定病因"
# are listed separately from "确诊" because a generated report uses all three.
OVER_CERTAIN_MARKERS = ("确诊", "已确定", "一定是", "必然是", "肯定是", "确定病因",
                        "100%", "绝对", "无疑", "确诊为")
# Assertions of a specific cause. These are refused outright: naming a pathogen
# or a cause is a diagnosis, and no lexical evidence can carry one.
UNSUPPORTED_DIAGNOSIS_MARKERS = ("病原是", "病因是", "感染了", "病害为", "病原为",
                                 "病因为", "导致该病害", "由.*引起")


def _matches_any(text: str, markers: tuple[str, ...]) -> str | None:
    """Return the first marker matched in ``text``, or ``None``.

    A marker is treated as a regular expression when it compiles as one and
    differs from a plain literal (``由.*引起``); otherwise it is a substring.
    Trying the literal form first would leave every regex marker dead, because a
    pattern's own source text is not what appears in the claim.
    """
    for marker in markers:
        if _is_pattern(marker):
            try:
                if re.search(marker, text):
                    return marker
            except re.error:
                continue
        elif marker in text:
            return marker
    return None


def _is_pattern(marker: str) -> bool:
    return any(character in marker for character in ".*+?[](){}|\\^$")


@dataclass
class ClaimVerdict:
    text: str
    evidence_ids: list[str]
    status: str
    reason: str = ""

    def to_dict(self) -> dict:
        payload = {"text": self.text, "evidence_ids": self.evidence_ids, "status": self.status}
        if self.reason:
            payload["reason"] = self.reason
        return payload


class ClaimAuditor:
    """Split a report into claims and require evidence for each factual one."""

    def __init__(self, *, available_evidence: dict[str, dict] | None = None):
        # Maps evidence_id -> the record that produced it, so a citation can be
        # checked against its source rather than trusted from the text.
        self.available_evidence = available_evidence or {}

    def audit(self, report: dict) -> dict:
        if not isinstance(report, dict):
            raise ShieldError("A report must be a JSON object")
        raw_claims = report.get("claims")
        if not isinstance(raw_claims, list):
            raise ShieldError("A report must declare a claims array")
        verdicts: list[ClaimVerdict] = []
        violations: list[str] = []
        for claim in raw_claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                violations.append("claim entry is not a text claim")
                continue
            text = claim["text"]
            evidence_ids = claim.get("evidence_ids")
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            evidence_ids = [item for item in evidence_ids if isinstance(item, str) and item]
            diagnosis = _matches_any(text, UNSUPPORTED_DIAGNOSIS_MARKERS)
            if diagnosis is not None:
                verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                             f"names a specific cause ({diagnosis}); no evidence can support a diagnosis"))
                violations.append(f"diagnosis claim refused: {text}")
                continue
            over_certain = _matches_any(text, OVER_CERTAIN_MARKERS)
            if over_certain is not None:
                verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                             f"over-certain wording ({over_certain}) without a qualifying source"))
                violations.append(f"over-certain claim refused: {text}")
                continue
            if not evidence_ids:
                verdicts.append(ClaimVerdict(text, [], "refused",
                                             "no evidence id; every_claim_requires_evidence"))
                violations.append(f"unsupported claim refused: {text}")
                continue
            # Fail closed: an id the run cannot account for is unverifiable, even
            # when no index was supplied. Silently accepting it would let a
            # fabricated citation reach SUPPORTED.
            unknown = [item for item in evidence_ids if item not in self.available_evidence]
            if unknown:
                verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                             f"unverifiable evidence id: {', '.join(sorted(unknown))}"))
                violations.append(f"unverifiable evidence id: {', '.join(sorted(unknown))}")
                continue
            verdicts.append(ClaimVerdict(text, evidence_ids, "supported"))
        trust = self._trust_level(verdicts, violations, report)
        return {
            "trust_level": trust,
            "claims": [verdict.to_dict() for verdict in verdicts],
            "supported_claims": [v.to_dict() for v in verdicts if v.status == "supported"],
            "refused_claims": [v.text for v in verdicts if v.status == "refused"],
            "violations": list(dict.fromkeys(violations)),
            "evidence_ids": sorted({item for v in verdicts for item in v.evidence_ids}),
        }

    def _trust_level(self, verdicts: list[ClaimVerdict], violations: list[str], report: dict) -> str:
        if violations:
            return TRUST_INSUFFICIENT
        supported = [v for v in verdicts if v.status == "supported"]
        if not supported:
            return TRUST_INSUFFICIENT
        missing = report.get("missing_inputs") or []
        unlinked = report.get("unlinked_evidence_ids") or []
        if missing or unlinked:
            return TRUST_LIMITED
        return TRUST_SUPPORTED


def load_evidence_index(*payloads: Any) -> dict[str, dict]:
    """Build the evidence_id -> record index a ClaimAuditor checks against."""
    index: dict[str, dict] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ("evidence", "observations", "factors"):
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    for field_name in ("evidence_id", "observation_id", "factor_id"):
                        if item.get(field_name):
                            index[item[field_name]] = item
    return index
