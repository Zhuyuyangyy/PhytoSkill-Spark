"""AgentShield Runtime: the middleware an Agent cannot route around.

Two distinct concerns live here, deliberately separated:

* :class:`ShieldRuntime` enforces permissions and tracing around **every** tool
  call. An Agent that simply declines to call the audit Skill is still
  intercepted, because the interception is not a Skill.
* :class:`ClaimAuditor` turns a finished report into auditable claims. It runs
  after execution and refuses unsupported claims. A claim may carry structured
  ``asserts``; the auditor then checks that the cited evidence records actually
  establish them, so a valid evidence id alone never proves a conclusion.

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

# The permission token a broker call requests. It is deliberately *not* the
# string "network": the Skill never holds network access, the broker does. A
# call carrying this token is allowed only when the manifest declared the
# matching ``permissions.backend_broker.inference`` block.
BROKER_INFERENCE = "backend_broker:inference"


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
                 trace_id: str | None = None, cache=None):
        self.registry = registry
        # The cache is accepted so a long live run can reuse observations
        # instead of paying a 12-167 s inference per repeated image. It changes
        # nothing about authorisation or tracing.
        self.executor = SkillExecutor(registry, cache=cache)
        self.max_calls = max_calls
        self.trace_id = trace_id or f"trace-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        self._calls = 0
        self.audit_log: list[CallRecord] = []

    # ── call interception ─────────────────────────────────────────────────

    def call(self, name: str, payload: dict, *, mode: str = "fixture",
             tool_call_id: str, requested_permissions: list[str] | None = None,
             case_workspace: Path | None = None) -> dict:
        """Authorise, trace, execute, and record one tool call.

        Every attempt is recorded, including the ones refused at the door: a
        call rejected for a missing ``tool_call_id`` or an exhausted budget is
        still an attempt, and an audit log that silently drops it cannot be
        reconciled against the caller's own trace.
        """
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            self.audit_log.append(CallRecord(
                trace_id=self.trace_id, tool_call_id=str(tool_call_id), skill=name,
                manifest_sha256="", status="blocked", duration_ms=0.0,
                permissions_used=sorted(set(requested_permissions or [])),
                detail="TraceError"))
            raise TraceError("Every tool call must carry a non-empty tool_call_id")
        if self._calls >= self.max_calls:
            self.audit_log.append(CallRecord(
                trace_id=self.trace_id, tool_call_id=tool_call_id, skill=name,
                manifest_sha256="", status="blocked", duration_ms=0.0,
                permissions_used=sorted(set(requested_permissions or [])),
                detail="BudgetExceeded"))
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
        broker = declared.get("backend_broker") or {}
        for permission in requested:
            if permission == "network":
                # The Skill itself never gets network access, whatever it asks
                # for. Inference reaches an endpoint only through the broker,
                # which is a separate, declared, endpoint-scoped permission.
                raise PermissionViolation(f"{name} requested network access; the package declares none")
            if permission == BROKER_INFERENCE:
                if not broker.get("inference"):
                    raise PermissionViolation(
                        f"{name} requested broker inference; the package declares no "
                        "permissions.backend_broker")
                continue
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


def call_or_failure(runtime: ShieldRuntime, name: str, payload: dict, *, mode: str,
                    tool_call_id: str, requested_permissions: list[str] | None = None) -> dict:
    """Run one governed call, reporting a middleware rejection as a failed call.

    A batch driver must survive a single blocked call: the record for that item
    says the middleware stopped it, and the run continues. Raising here would
    turn one rejection into the collapse of the whole batch — and a crashed
    batch is easily mistaken for a broken backend.
    """
    try:
        return runtime.call(name, payload, mode=mode, tool_call_id=tool_call_id,
                            requested_permissions=requested_permissions)
    except ShieldError as exc:
        return {"tool_call_id": tool_call_id, "name": name, "mode": mode,
                "status": "failed", "data": None, "duration_ms": 0.0,
                "error": {"code": type(exc).__name__, "message": str(exc)}}


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


def _record_supports(record: dict, key: str, value: str) -> bool:
    """Does this evidence record establish ``key == value``?

    ``phenotype`` is special: an observation record carries it directly, and a
    knowledge record carries the phenotypes it supports. Any other key must
    equal the record's field of the same name. A record that carries neither
    establishes nothing, which fails closed.
    """
    if not isinstance(record, dict):
        return False
    if key == "phenotype":
        return (record.get("phenotype") == value
                or value in (record.get("supports_phenotypes") or []))
    return record.get(key) == value


@dataclass
class ClaimVerdict:
    text: str
    evidence_ids: list[str]
    status: str
    reason: str = ""
    # The structured assertion the claim carried, echoed into the verdict so a
    # reader can see what was checked, not only that something passed.
    asserts: dict | None = None

    def to_dict(self) -> dict:
        payload = {"text": self.text, "evidence_ids": self.evidence_ids, "status": self.status}
        if self.reason:
            payload["reason"] = self.reason
        if self.asserts is not None:
            payload["asserts"] = self.asserts
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
            # Echoed into the verdict so the report shows what was asserted and
            # checked, not only the outcome. A non-dict assertion is malformed
            # and is refused below rather than echoed as if it were data.
            asserts = claim.get("asserts")
            echoed = asserts if isinstance(asserts, dict) else None
            diagnosis = _matches_any(text, UNSUPPORTED_DIAGNOSIS_MARKERS)
            if diagnosis is not None:
                verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                             f"names a specific cause ({diagnosis}); no evidence can support a diagnosis",
                                             asserts=echoed))
                violations.append(f"diagnosis claim refused: {text}")
                continue
            over_certain = _matches_any(text, OVER_CERTAIN_MARKERS)
            if over_certain is not None:
                verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                             f"over-certain wording ({over_certain}) without a qualifying source",
                                             asserts=echoed))
                violations.append(f"over-certain claim refused: {text}")
                continue
            if not evidence_ids:
                verdicts.append(ClaimVerdict(text, [], "refused",
                                             "no evidence id; every_claim_requires_evidence",
                                             asserts=echoed))
                violations.append(f"unsupported claim refused: {text}")
                continue
            # Fail closed: an id the run cannot account for is unverifiable, even
            # when no index was supplied. Silently accepting it would let a
            # fabricated citation reach SUPPORTED. The same holds for an id
            # whose records conflict: neither version can be trusted, so the
            # citation is refused rather than resolved by arrival order.
            unknown = [item for item in evidence_ids if item not in self.available_evidence]
            conflicted = [item for item in evidence_ids
                          if isinstance(self.available_evidence.get(item), dict)
                          and self.available_evidence[item].get("conflicted")]
            if unknown or conflicted:
                unverifiable = sorted(set(unknown) | set(conflicted))
                verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                             f"unverifiable evidence id: {', '.join(unverifiable)}",
                                             asserts=echoed))
                violations.append(f"unverifiable evidence id: {', '.join(unverifiable)}")
                continue
            # A structured assertion is checked against the evidence it cites.
            # The ids being real is necessary but not sufficient: a claim that
            # asserts a phenotype must cite a record that actually carries it,
            # otherwise any observation could be used to support any wording.
            if asserts is not None:
                if not isinstance(asserts, dict) or not all(
                        isinstance(key, str) and bool(key) and isinstance(value, str) and bool(value)
                        for key, value in asserts.items()):
                    verdicts.append(ClaimVerdict(text, evidence_ids, "refused",
                                                 "malformed structured assertion"))
                    violations.append(f"malformed assertion refused: {text}")
                    continue
                records = [self.available_evidence[item] for item in evidence_ids]
                # One record must establish every asserted key. Checking each
                # key against any record would let a claim assemble its
                # assertion from several records that never shared a context —
                # a phenotype from one case and a case id from another.
                if not any(all(_record_supports(record, key, value)
                               for key, value in asserts.items())
                           for record in records):
                    verdicts.append(ClaimVerdict(
                        text, evidence_ids, "refused",
                        f"no single cited record establishes {', '.join(sorted(asserts))}",
                        asserts=echoed))
                    violations.append(f"unsupported assertion refused: {text}")
                    continue
            verdicts.append(ClaimVerdict(text, evidence_ids, "supported", asserts=echoed))
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
    """Build the evidence_id -> record index a ClaimAuditor checks against.

    Each indexed record inherits the case and species of the envelope it
    arrived in, so a claim that asserts a case or species can be checked
    against the evidence instead of being trusted from the text.

    Conflicts fail closed rather than resolving by arrival order: two different
    records claiming one evidence id, or a child record whose own case/species
    contradicts its envelope, poison that id — a claim citing it is refused
    instead of being checked against whichever record happened to be indexed
    last. A poisoned id maps to ``{"conflicted": True}``, which establishes
    nothing.
    """
    index: dict[str, dict] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        context = {key: payload[key] for key in ("case_id", "species") if key in payload}
        for key in ("evidence", "observations", "factors"):
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    for field_name in ("evidence_id", "observation_id", "factor_id"):
                        identifier = item.get(field_name)
                        if not identifier:
                            continue
                        # A child record that carries its own case or species
                        # must agree with the envelope it arrived in; a
                        # contradiction means its provenance cannot be
                        # established, so the id is poisoned.
                        if any(field in item and item[field] != context[field]
                               for field in context):
                            index[identifier] = {"conflicted": True}
                            continue
                        record = {**context, **item}
                        existing = index.get(identifier)
                        if existing is None or existing == record:
                            index[identifier] = record
                        else:
                            index[identifier] = {"conflicted": True}
    return index
