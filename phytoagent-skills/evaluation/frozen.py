"""The committed freeze manifest, and the drift check that enforces it.

A freeze is only meaningful if it is recorded **outside the run**: a
fingerprint computed at run time and printed to a log can be edited
afterwards, and then nothing distinguishes a re-tuned experiment from the
original one. The manifest is a committed file that lists every input the
holdout blind test depends on, each with its SHA-256 and the reason it decides
an answer.

``verify`` recomputes both the per-file hashes and the aggregate fingerprints
and fails loudly on any drift. That is what makes the freeze practical rather
than ceremonial: after the blind test, editing a prompt, a threshold, a schema
or the protocol itself cannot be passed off as the same experiment — the next
``verify`` names the file that moved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evaluation.split import (Freeze, FreezeError, auditor_fingerprint, freeze,
                              ontology_fingerprint, parser_fingerprint,
                              prompt_fingerprint, protocol_fingerprint,
                              retriever_fingerprint, schema_fingerprint,
                              scorer_fingerprint)

DEFAULT_MANIFEST = Path("evaluation/frozen.json")

# Every input a holdout run's numbers depend on, with the reason it decides an
# answer. This list is the contract: a file that is not here is not frozen, and
# a file that is here cannot change silently.
FREEZE_INPUTS: tuple[tuple[str, str], ...] = (
    ("vision/prompts.py", "the exact text the model is shown, per mode"),
    ("vision/ontology.py", "the phenotype vocabulary and guard markers, in order"),
    ("vision/parser.py", "how a reply is read: extraction rules and thresholds"),
    ("skills/plant_vision/schema.json", "the observation contract: phenotype enum, area basis, model-score semantics"),
    ("skills/growth_risk/schema.json", "the environment contract and risk levels"),
    ("skills/herbal_knowledge/schema.json", "the evidence contract: retrieval_score semantics and provenance fields"),
    ("skills/evidence_fusion/schema.json", "the fusion contract: how legs are combined"),
    ("skills/agentshield_audit/schema.json", "the audit contract: claim-evidence linkage rules"),
    ("runtime/shield.py", "the auditor's trust-level rules (SUPPORTED / LIMITED / INSUFFICIENT)"),
    ("runtime/workflow.py", "how the legs are combined into a verdict: the quality gate and the evidence-linkage rules an end-to-end trust level is computed from"),
    ("corpus/retriever.py", "the retrieval gate: what counts as a literal match"),
    ("corpus/ranking.py", "the ranking constants and field weights"),
    ("evaluation/metrics.py", "the metric definitions the report is computed from"),
    ("docs/evaluation.md", "the protocol the run is held to"),
)

MANIFEST_NOTE = ("The frozen evaluation configuration for the holdout blind test. "
                 "Every listed input is pinned by SHA-256; verify-freeze recomputes "
                 "them and the aggregate fingerprints and fails on any drift. "
                 "Re-freezing is an explicit, reviewed act — never a side effect "
                 "of a tuning edit.")


@dataclass(frozen=True)
class DriftReport:
    """What moved since the manifest was written."""

    ok: bool
    freeze_matches: bool
    drifted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"ok": self.ok, "freeze_matches": self.freeze_matches,
                "drifted": list(self.drifted), "missing": list(self.missing),
                "unexpected": list(self.unexpected)}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project_root() -> Path:
    from demo.fixture_workspace import PROJECT_ROOT
    return PROJECT_ROOT


def input_hashes(root: Path | None = None) -> dict[str, str]:
    """The SHA-256 of every frozen input, keyed by repo-relative path.

    Hashes are computed over line-ending-normalised bytes: a fresh clone on
    any platform must verify against the manifest, and a file nobody edited
    must never appear as drift.
    """
    root = root or _project_root()
    hashes: dict[str, str] = {}
    for relative, _ in FREEZE_INPUTS:
        path = root / relative
        if not path.is_file():
            raise FreezeError(f"frozen input is missing from the tree: {relative}")
        hashes[relative] = _sha256(path.read_bytes().replace(b"\r\n", b"\n"))
    return hashes


def build_manifest(frozen: Freeze, *, at: str | None = None,
                   weight_source: str = "") -> dict:
    """Assemble the manifest: the freeze record plus the per-file inventory."""
    return {
        "schema_version": 1,
        "note": MANIFEST_NOTE,
        "frozen_at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freeze": frozen.to_dict(),
        "freeze_id": frozen.freeze_id,
        # Where the pinned weights' identity was observed. A digest typed by
        # hand is a claim; one read out of a recorded run is a measurement,
        # and the source path says which.
        "weight_identity_source": weight_source,
        "inputs": [{"path": relative, "sha256": input_hashes()[relative],
                    "role": role} for relative, role in FREEZE_INPUTS],
    }


def write_manifest(path: str | Path = DEFAULT_MANIFEST, *, model: str,
                   quantization: str, at: str | None = None,
                   model_digest: str = "") -> dict:
    """Freeze the current tree and write the manifest. An explicit publisher act.

    The weight identity is pinned, not just the model name: without an
    observable digest a ``:latest`` tag can be re-pulled as different bytes
    and the freeze would still pass. A caller may pass ``model_digest``
    explicitly; otherwise the identity recorded by an earlier real run is
    used, and if neither exists the freeze refuses — an unpinned freeze is
    not a freeze.
    """
    from evaluation.split import recorded_model_identity

    recorded = recorded_model_identity(model)
    if not model_digest:
        model_digest = recorded["digest"]
    if not model_digest:
        raise FreezeError(
            "no observable weight identity: pass --model-digest, or record a run "
            "whose artifact carries the model digest. A model name alone is not "
            "an identity — ':latest' can be re-pulled as different weights.")
    # The size and the source are only meaningful for the identity this freeze
    # actually pins; a digest that matches no recorded run is pinned but
    # unattributed, which the manifest says plainly.
    same = recorded["digest"] == model_digest
    source = recorded["source"] if same else ""
    size = recorded["size_bytes"] if same else None
    frozen = freeze(model=model, quantization=quantization,
                    model_digest=model_digest, model_size_bytes=size)
    manifest = build_manifest(frozen, at=at, weight_source=source)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return manifest


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FreezeError(f"no freeze manifest at {path}; run freeze first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "freeze" not in manifest:
        raise FreezeError(f"malformed freeze manifest: {path}")
    return manifest


def verify_manifest(path: str | Path = DEFAULT_MANIFEST,
                    *, model: str | None = None,
                    quantization: str | None = None) -> DriftReport:
    """Compare the current tree against the committed manifest.

    Two independent checks, because they fail differently: the per-file hashes
    name *which* input moved, and the aggregate fingerprints catch a change to
    anything the aggregates cover even if the file list were incomplete. The
    model and quantisation are compared when the caller supplies them — a
    manifest recorded for one model must not silently vouch for another.
    """
    manifest = load_manifest(path)
    recorded = {item["path"]: item["sha256"] for item in manifest.get("inputs", [])}
    current = input_hashes()
    drifted = tuple(sorted(path_ for path_, digest in current.items()
                           if recorded.get(path_) != digest))
    missing = tuple(sorted(set(recorded) - set(current)))
    unexpected = tuple(sorted(set(current) - set(recorded)))

    recorded_freeze = manifest["freeze"]
    frozen = freeze(model=model or recorded_freeze["model"],
                    quantization=quantization or recorded_freeze["quantization"],
                    model_digest=recorded_freeze.get("model_digest", ""),
                    model_size_bytes=recorded_freeze.get("model_size_bytes"))
    freeze_matches = frozen.to_dict() == recorded_freeze
    ok = freeze_matches and not (drifted or missing or unexpected)
    return DriftReport(ok=ok, freeze_matches=freeze_matches, drifted=drifted,
                       missing=missing, unexpected=unexpected)


def assert_manifest(path: str | Path = DEFAULT_MANIFEST, *, model: str | None = None,
                    quantization: str | None = None) -> DriftReport:
    # (loads the manifest so the failure message can name what is missing)
    """verify_manifest, but a drift raises instead of being reported.

    The holdout command calls this before it runs: a blind test executed on a
    configuration that no longer matches the frozen one is not that test. A
    missing manifest is a refusal, not a skip: the gate must not be optional.
    """
    manifest = load_manifest(path)
    report = verify_manifest(path, model=model, quantization=quantization)
    if report.ok:
        return report
    problems = []
    if not report.freeze_matches:
        detail = "the aggregate fingerprints no longer match the manifest"
        if not manifest["freeze"].get("model_digest"):
            detail += " (and the manifest pins no weight identity)"
        elif (model or manifest["freeze"].get("model")) and \
                model != manifest["freeze"].get("model"):
            detail += f" (model requested: {model!r})"
        problems.append(detail)
    if report.drifted:
        problems.append(f"inputs changed since the freeze: {list(report.drifted)}")
    if report.missing:
        problems.append(f"inputs missing from the tree: {list(report.missing)}")
    if report.unexpected:
        problems.append(f"inputs not in the manifest: {list(report.unexpected)}")
    raise FreezeError("; ".join(problems))


# The aggregate fingerprints and the per-file inventory must describe the same
# bytes. These equalities are the structural guarantee; a test pins them.
AGGREGATE_TO_INPUT = {
    "prompt_hash": None,          # rendered text, not a file — see the note below
    "parser_hash": "vision/parser.py",
    "ontology_hash": None,        # rendered values, not a file
    "scorer_hash": "evaluation/metrics.py",
    "auditor_hash": "runtime/shield.py",
    "retriever_hash": None,       # two modules hashed together
    "protocol_hash": "docs/evaluation.md",
}


def aggregates_match_inputs() -> dict[str, bool]:
    """Which aggregate fingerprints are directly pinned by a single input file.

    The prompt and ontology fingerprints hash *rendered* values, not source
    bytes, so a whitespace-only refactor that leaves the text identical does
    not invalidate a freeze. That is deliberate, and it is why they have no
    single-file counterpart here.
    """
    hashes = input_hashes()
    return {key: (hashes.get(relative) == _aggregate(key)) for key, relative
            in AGGREGATE_TO_INPUT.items() if relative is not None}


def _aggregate(name: str) -> str:
    return {"parser_hash": parser_fingerprint,
            "scorer_hash": scorer_fingerprint,
            "auditor_hash": auditor_fingerprint,
            "protocol_hash": protocol_fingerprint,
            "ontology_hash": ontology_fingerprint,
            "prompt_hash": prompt_fingerprint,
            "schema_hash": schema_fingerprint,
            "retriever_hash": retriever_fingerprint}[name]()


__all__ = ["AGGREGATE_TO_INPUT", "DEFAULT_MANIFEST", "DriftReport", "FREEZE_INPUTS",
           "aggregates_match_inputs", "assert_manifest", "build_manifest",
           "input_hashes", "load_manifest", "verify_manifest", "write_manifest"]
