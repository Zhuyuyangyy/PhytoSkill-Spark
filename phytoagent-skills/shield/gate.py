"""AgentShield compile gate: the checks a Skill package must pass before Registry.

Nothing reaches the Registry without passing every check. A rejected package is
quarantined together with the risks and the repair items, so the failure is
actionable rather than a bare error code.

The gate is deliberately independent of the Compiler: it re-reads the written
package from disk and re-validates it, so a buggy or malicious compiler cannot
mark its own output as clean.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compiler.catalog import CAPABILITIES
from sdk.exceptions import ShieldError
from sdk.manifest import verify_manifest
from sdk.schema import package_file, read_json

# Text patterns that indicate a generated package is trying to execute code,
# reach the network, or exfiltrate data. Matching is a hard rejection.
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("subprocess", re.compile(r"\bsubprocess\b")),
    ("os_system", re.compile(r"\bos\.(system|popen|exec[lv][ep]?)\b")),
    ("eval_exec", re.compile(r"\b(eval|exec)\s*\(")),
    ("network_call", re.compile(r"\b(socket|urllib|requests|httpx|http\.client)\b")),
    ("dynamic_import", re.compile(r"\b__import__\b|importlib\.import_module")),
    ("shell_payload", re.compile(r"(?i)\b(bash\s+-c|powershell|cmd\.exe|/bin/sh)\b")),
    ("hidden_instruction", re.compile(r"(?i)(ignore (all |the )?(previous|above) instructions|"
                                      r"do not tell the user|disregard the (system|audit))")),
    ("exfiltration", re.compile(r"(?i)(upload|POST|send).{0,40}(to|https?://)\s*\S+")),
    ("credential_read", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=")),
)

# Permission strings the gate is allowed to grant, and their scope.
ALLOWED_FILESYSTEM_PERMISSIONS = {
    "case_workspace:read", "corpus:read", "package:read",
}
# Skill names the registry may hold. The four audited providers plus the audit
# Skill; anything else is a capability the gate has never reviewed.
REGISTERED_SKILLS = {
    "plant_vision", "growth_risk", "herbal_knowledge", "evidence_fusion",
    "agentshield_audit",
}
FORBIDDEN_PERMISSIONS = {
    "network", "network:allow", "filesystem:write", "filesystem:read-write",
    "gpu", "shell", "tool:*", "*",
}
WILDCARD = re.compile(r"[*?]")


@dataclass
class GateCheck:
    name: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class GateResult:
    skill: str
    checks: list[GateCheck] = field(default_factory=list)
    quarantined: bool = False

    @property
    def passed(self) -> bool:
        return all(check.status == "passed" for check in self.checks) and not self.quarantined

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if check.status != "passed"]

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "verdict": "PASS" if self.passed else "QUARANTINE",
            "checks": [check.to_dict() for check in self.checks],
            "quarantined": self.quarantined,
        }


def _scan_text(relative: str, text: str) -> list[str]:
    findings: list[str] = []
    for label, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            findings.append(f"{relative}: {label}")
    return findings


def _iter_text_files(package_dir: Path) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name in ("manifest.json", "manifest.sig"):
            continue
        if path.suffix.lower() in (".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".zip", ".whl"):
            continue
        relative = path.relative_to(package_dir).as_posix()
        try:
            files.append((relative, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError):
            continue
    return files


def gate_package(package_dir: str | Path, *, trusted_public_key: str | Path | None = None,
                 project_signature_key: str | Path | None = None) -> GateResult:
    """Run every compile-gate check on one written Skill package directory."""
    package_dir = Path(package_dir).absolute()
    result = GateResult(skill=package_dir.name)
    try:
        _run_checks(package_dir, result, trusted_public_key, project_signature_key)
    except (ShieldError, OSError) as exc:
        result.checks.append(GateCheck("package_readable", "failed", str(exc)))
        result.quarantined = True
    return result


def _run_checks(package_dir: Path, result: GateResult, trusted_public_key, project_signature_key) -> None:
    # 1. Manifest integrity: file inventory and metadata hash.
    try:
        manifest = verify_manifest(package_dir)
        result.checks.append(GateCheck("manifest_integrity", "passed",
                                       f"manifest_sha256={manifest['manifest_sha256']}"))
    except Exception as exc:
        result.checks.append(GateCheck("manifest_integrity", "failed", str(exc)))
        result.quarantined = True
        return

    # 2. Schema conformance of every declared schema. A compiler-generated
    #    package also carries skill_spec.json; a hand-authored one does not, so
    #    its absence is only an error when the package claims to be generated.
    schema_errors: list[str] = []
    spec: dict | None = None
    try:
        for reference in (manifest["input_schema"], manifest["output_schema"]):
            package_file(package_dir, reference.split("#")[0])
        spec_path = package_dir / "skill_spec.json"
        if spec_path.is_file():
            spec = read_json(spec_path)
            if spec.get("name") != manifest["name"].replace("_", "-"):
                schema_errors.append("skill_spec.json name does not match the package")
    except Exception as exc:
        schema_errors.append(str(exc))
    result.checks.append(GateCheck("schema_conformance", "passed" if not schema_errors else "failed",
                                   "; ".join(schema_errors) or "input and output schemas parse"))

    # 3. Least privilege: only whitelisted, non-wildcard permissions; network denied.
    permission_errors: list[str] = []
    declared = manifest.get("permissions", {})
    if not isinstance(declared, dict):
        permission_errors.append("permissions must be an object")
    else:
        if declared.get("network") != "deny":
            permission_errors.append(f"network must be 'deny', got {declared.get('network')!r}")
        filesystem = declared.get("filesystem")
        if not isinstance(filesystem, list) or not filesystem:
            permission_errors.append("at least one filesystem permission must be declared")
        else:
            for permission in filesystem:
                # Order matters: the wildcard test must come first, otherwise a
                # wildcard string is reported as merely "undeclared" and the
                # reader never learns it was a wildcard.
                if WILDCARD.search(permission):
                    permission_errors.append(f"wildcard permission: {permission}")
                elif permission in FORBIDDEN_PERMISSIONS:
                    permission_errors.append(f"forbidden permission: {permission}")
                elif permission not in ALLOWED_FILESYSTEM_PERMISSIONS:
                    permission_errors.append(f"undeclared permission: {permission}")
        # A package that declares no capability at all is not reviewable: the
        # manifest schema allows the field to be absent, so an omitted list must
        # fail rather than silently skip the capability review.
        tools = manifest.get("capabilities")
        if not isinstance(tools, list) or not tools:
            permission_errors.append("no capability declared; the package cannot be reviewed")
        else:
            for capability in tools:
                # The audited capability catalogue, plus any Skill the registry
                # has already validated, is the allowed set. The Skill's own name
                # is always allowed, otherwise no package could declare itself.
                if (capability not in CAPABILITIES
                        and capability not in REGISTERED_SKILLS
                        and capability != manifest["name"].replace("-", "_")):
                    permission_errors.append(f"undeclared capability: {capability}")
    result.checks.append(GateCheck("permission_least_privilege",
                                   "passed" if not permission_errors else "failed",
                                   "; ".join(permission_errors) or "network denied; no wildcard or write scope"))

    # 4. Negative eval coverage. Two authoring styles are accepted:
    #    * compiler output: ``cases`` with an explicit ``kind``, plus a
    #      SkillSpec ``negative_evals`` list;
    #    * hand-authored packages: ``contract_cases`` whose ids encode the
    #      behaviour, plus ``agent_cases`` with positive/negative/missing kinds.
    #    Both must show a positive case, a missing-input case and a negative case.
    eval_errors: list[str] = []
    try:
        evals = read_json(package_file(package_dir, "evals/evals.json"))
        cases = evals.get("cases") or evals.get("contract_cases") or []
        if not cases:
            eval_errors.append("evals.json declares no cases")
        kinds = {case.get("kind") for case in cases if isinstance(case, dict)}
        ids = {case.get("id") for case in cases if isinstance(case, dict)}
        agent_cases = evals.get("agent_cases") or []
        agent_kinds = {case.get("kind") for case in agent_cases if isinstance(case, dict)}

        has_positive = ("positive" in kinds or "positive_fixture" in ids
                        or "valid_fixture" in ids or "positive_trigger" in agent_kinds)
        if not has_positive:
            eval_errors.append("no positive case")
        has_missing = ("missing_parameter" in kinds or "missing_species" in ids
                       or "missing_context" in agent_kinds)
        if not has_missing:
            eval_errors.append("no missing-parameter case")
        has_negative = ("negative_trigger" in kinds or "negative_trigger" in agent_kinds
                        or "reject_unmatched_fixture" in ids
                        or "live_has_no_fixture_fallback" in ids
                        or "reject_cross_case_fusion" in ids)
        if not has_negative:
            eval_errors.append("no negative-trigger case")

        if isinstance(spec, dict):
            spec_evals = spec.get("negative_evals")
            if not spec_evals:
                eval_errors.append("SkillSpec declares no negative_evals")
            negative_kinds = {item.get("kind") for item in spec_evals or []}
            for required in ("missing_input", "unsupported_claim", "permission_violation"):
                if required not in negative_kinds:
                    eval_errors.append(f"SkillSpec lacks negative_eval kind {required}")
    except Exception as exc:
        eval_errors.append(str(exc))
    result.checks.append(GateCheck("negative_eval_coverage",
                                   "passed" if not eval_errors else "failed",
                                   "; ".join(eval_errors) or "positive, missing-parameter and negative cases present"))

    # 5. Hidden instructions, dangerous patterns and data-exfiltration intent.
    findings: list[str] = []
    for relative, text in _iter_text_files(package_dir):
        findings.extend(_scan_text(relative, text))
    result.checks.append(GateCheck("hidden_instruction_scan",
                                   "passed" if not findings else "failed",
                                   "; ".join(findings) or "no hidden instructions or dangerous patterns"))

    # 6. Project signature: the package must carry one, verified against a key
    #    held outside the package. Without a key the check is reported as
    #    not_run; it is never reported as passed.
    if project_signature_key is None:
        result.checks.append(GateCheck("project_signature", "not_run",
                                       "no signing key supplied; not counted as passed"))
    else:
        try:
            from registry.signer import verify_signature
            verify_signature(package_dir, project_signature_key)
            result.checks.append(GateCheck("project_signature", "passed", "Ed25519 signature verified"))
        except Exception as exc:
            result.checks.append(GateCheck("project_signature", "failed", str(exc)))

    if result.failures:
        result.quarantined = True


def repair_items(result: GateResult) -> list[str]:
    """Human-actionable repair instructions for a quarantined package."""
    items: list[str] = []
    for check in result.failures:
        if check.name == "permission_least_privilege":
            items.append("Remove undeclared, wildcard or write permissions; keep network: deny.")
        elif check.name == "negative_eval_coverage":
            items.append("Add positive, missing-parameter and negative-trigger cases to evals/evals.json.")
        elif check.name == "hidden_instruction_scan":
            items.append("Remove subprocess/eval/exec/network code and any hidden instructions.")
        elif check.name == "manifest_integrity":
            items.append("Re-seal the package: python -m scripts.seal_skills.")
        elif check.name == "project_signature":
            items.append("Sign the sealed package with the project key held outside the package.")
        else:
            items.append(f"Resolve {check.name}: {check.detail}")
    return items


def gate_many(package_dirs: list[str | Path], **kwargs: Any) -> dict:
    """Gate several packages and return a registry-ready summary."""
    results = [gate_package(directory, **kwargs) for directory in package_dirs]
    return {
        "scope": "agentshield_compile_gate_local",
        "agent_model_called": False,
        "dgx_hardware_used": False,
        "nvidia_verified": False,
        "packages": [result.to_dict() for result in results],
        "registered": [result.skill for result in results if result.passed],
        "quarantined": [{"skill": result.skill, "repair": repair_items(result)}
                        for result in results if not result.passed],
    }


def package_digest(package_dir: str | Path) -> str:
    """Stable digest of a package's file inventory, for registry bookkeeping."""
    package_dir = Path(package_dir).absolute()
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.name not in ("manifest.sig",):
            digest.update(path.relative_to(package_dir).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
