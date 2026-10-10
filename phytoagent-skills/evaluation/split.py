"""Dataset roles, the freeze fingerprint, and the one-shot holdout rule.

Three roles, and the distinction is the whole point:

* **calibration** — used to tune the prompt. Its numbers may be quoted only as
  "calibration on <set>", never as validation.
* **development** — used to compare design choices.
* **holdout** — run *once*, after everything is frozen. Looking at the result
  and then changing anything invalidates it, and no amount of re-running repairs
  that.

The freeze is a fingerprint of everything that could change an answer: the
prompt text, the parser, the vocabulary, the scorer, the model and its
quantisation. It is computed from the *actual objects* for the prompt and the
vocabulary, and from the *source* for the parser and the scorer.

Why source hashes for the last two: their behaviour is the code, and there is no
cheaper faithful proxy. The cost is that a comment edit also changes the
fingerprint. That is the safe direction — over-invalidating a freeze wastes a
run, under-invalidating it silently produces a number nobody can defend.

The one-shot rule is enforced by a ledger on disk, not by convention. A holdout
evaluation whose fingerprint already has a recorded run is refused unless the
caller explicitly supersedes it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROLE_CALIBRATION = "calibration"
ROLE_DEVELOPMENT = "development"
ROLE_HOLDOUT = "holdout"
ROLES = (ROLE_CALIBRATION, ROLE_DEVELOPMENT, ROLE_HOLDOUT)

DEFAULT_LEDGER = Path("artifacts/evaluations/holdout-ledger.json")


class FreezeError(RuntimeError):
    """A freeze or a holdout rule was violated."""


class HoldoutAlreadyRun(FreezeError):
    """This exact freeze has already been evaluated on the holdout set."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _module_source_hash(module) -> str:
    """Hash a module's source file. Over-invalidates on a comment edit, on purpose."""
    path = Path(getattr(module, "__file__", ""))
    if not path.is_file():
        raise FreezeError(f"cannot fingerprint module source: {module!r}")
    return _sha256(path.read_bytes())


def prompt_fingerprint() -> str:
    """The exact text the model is shown, per mode.

    The prompt is the first thing that changes an answer, and it is the thing a
    tuning loop edits most. Hashing the rendered templates rather than the source
    means a whitespace-only refactor that leaves the text identical does *not*
    invalidate a freeze — the model cannot tell the difference.
    """
    from vision.prompts import build_prompt

    rendered = {mode: build_prompt(species="黄芪", mode=mode)
                for mode in ("live", "herb")}
    return _sha256(_canonical(rendered))


def ontology_fingerprint() -> str:
    """The vocabularies and guard markers, in order.

    Order is part of the value: the herb prompt lists the phenotypes in tuple
    order, so a reordering is a real change to what the model is asked for.
    """
    from vision.ontology import (DIAGNOSIS_MARKERS, HERB_PHENOTYPES,
                                 JUDGEMENT_MARKERS, LEAF_PHENOTYPES)

    return _sha256(_canonical({
        "leaf_phenotypes": list(LEAF_PHENOTYPES),
        "herb_phenotypes": list(HERB_PHENOTYPES),
        "diagnosis_markers": list(DIAGNOSIS_MARKERS),
        "judgement_markers": list(JUDGEMENT_MARKERS),
    }))


def parser_fingerprint() -> str:
    import vision.parser
    return _module_source_hash(vision.parser)


def scorer_fingerprint() -> str:
    import evaluation.metrics
    return _module_source_hash(evaluation.metrics)


@dataclass(frozen=True)
class Freeze:
    """Everything that must be held constant for a holdout run to mean anything."""

    model: str
    quantization: str
    prompt_hash: str
    parser_hash: str
    ontology_hash: str
    scorer_hash: str

    @property
    def freeze_id(self) -> str:
        return _sha256(_canonical(self.to_dict()))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "quantization": self.quantization,
            "prompt_hash": self.prompt_hash,
            "parser_hash": self.parser_hash,
            "ontology_hash": self.ontology_hash,
            "scorer_hash": self.scorer_hash,
        }

    def summary(self) -> dict:
        """The shape a report shows: full hashes are unreadable, prefixes are not."""
        return {**{key: (value[:12] if key.endswith("_hash") else value)
                   for key, value in self.to_dict().items()},
                "freeze_id": self.freeze_id[:12]}


def freeze(*, model: str, quantization: str) -> Freeze:
    """Fingerprint the current code and the named model.

    ``model`` and ``quantization`` are caller-supplied because they are runtime
    configuration, not source: the same code against a different quantisation is
    a different experiment.
    """
    if not model or not quantization:
        raise FreezeError("a freeze must name both the model and its quantisation")
    return Freeze(model=model, quantization=quantization,
                  prompt_hash=prompt_fingerprint(),
                  parser_hash=parser_fingerprint(),
                  ontology_hash=ontology_fingerprint(),
                  scorer_hash=scorer_fingerprint())


# ── dataset roles ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetSplit:
    """Which image belongs to which role. Roles must not overlap."""

    calibration: tuple[str, ...] = ()
    development: tuple[str, ...] = ()
    holdout: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for role in ROLES:
            for image in getattr(self, role):
                if image in seen:
                    raise FreezeError(
                        f"image {image!r} is in both {seen[image]!r} and {role!r}; "
                        "a holdout image must never have been inspected while tuning")
                seen[image] = role

    @property
    def size(self) -> int:
        return len(self.calibration) + len(self.development) + len(self.holdout)

    def role_of(self, image: str) -> str | None:
        for role in ROLES:
            if image in getattr(self, role):
                return role
        return None

    def to_dict(self) -> dict:
        return {"calibration": list(self.calibration),
                "development": list(self.development),
                "holdout": list(self.holdout),
                "sizes": {role: len(getattr(self, role)) for role in ROLES},
                "total": self.size}


def split_entries(entries: list[dict], *, holdout: tuple[str, ...] = (),
                  development: tuple[str, ...] = (),
                  image_key: str = "image") -> DatasetSplit:
    """Assign roles from explicit id lists; everything else is calibration.

    Explicit is the point. Assigning the holdout by a random seed that nobody
    recorded is how a holdout stops being one.
    """
    names = [str(entry.get(image_key)) for entry in entries
             if isinstance(entry, dict) and entry.get(image_key) is not None]
    known = set(names)
    for role, ids in ((ROLE_HOLDOUT, holdout), (ROLE_DEVELOPMENT, development)):
        unknown = sorted(set(ids) - known)
        if unknown:
            raise FreezeError(f"{role} names images not in the dataset: {unknown}")
    assigned = set(holdout) | set(development)
    return DatasetSplit(
        calibration=tuple(sorted(known - assigned)),
        development=tuple(sorted(development)),
        holdout=tuple(sorted(holdout)),
    )


# ── the one-shot ledger ──────────────────────────────────────────────────────


@dataclass
class HoldoutLedger:
    """A durable record of every holdout evaluation, keyed by freeze id.

    Kept on disk because the rule it enforces is about *history*: "this
    fingerprint has been run before" is not a fact a single process can know.
    """

    path: Path = DEFAULT_LEDGER
    runs: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LEDGER) -> "HoldoutLedger":
        path = Path(path)
        if not path.is_file():
            return cls(path=path, runs=[])
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload.get("runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            raise FreezeError(f"malformed holdout ledger: {path}")
        return cls(path=path, runs=runs)

    def runs_for(self, freeze_id: str) -> list[dict]:
        return [run for run in self.runs if run.get("freeze_id") == freeze_id]

    def has_run(self, freeze_id: str) -> bool:
        return bool(self.runs_for(freeze_id))

    def guard(self, frozen: Freeze, *, supersede: bool = False) -> None:
        """Refuse a second holdout evaluation of the same freeze.

        ``supersede`` is the escape hatch, and it is deliberately a separate
        argument rather than a default: re-running a holdout after seeing it is
        the one thing the protocol forbids, so it has to be asked for by name.
        """
        if not supersede and self.has_run(frozen.freeze_id):
            raise HoldoutAlreadyRun(
                f"holdout already evaluated for freeze {frozen.freeze_id[:12]}; "
                "changing anything now invalidates that run — pass supersede=True "
                "and record why")

    def record(self, frozen: Freeze, metrics: dict, *, role: str = ROLE_HOLDOUT,
               note: str = "", supersede: bool = False,
               at: str | None = None) -> dict:
        if role not in ROLES:
            raise FreezeError(f"unknown role {role!r}")
        if role == ROLE_HOLDOUT:
            self.guard(frozen, supersede=supersede)
        entry = {
            "freeze_id": frozen.freeze_id,
            "role": role,
            "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
            "supersedes": [run["at"] for run in self.runs_for(frozen.freeze_id)]
                          if supersede else [],
            "freeze": frozen.to_dict(),
            "metrics": metrics,
        }
        self.runs.append(entry)
        self.save()
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1,
                   "note": ("Every holdout evaluation, keyed by freeze id. A freeze "
                            "that already appears here must not be re-run without "
                            "an explicit supersede."),
                   "runs": self.runs}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")


__all__ = ["DEFAULT_LEDGER", "DatasetSplit", "Freeze", "FreezeError",
           "HoldoutAlreadyRun", "HoldoutLedger", "ROLES", "ROLE_CALIBRATION",
           "ROLE_DEVELOPMENT", "ROLE_HOLDOUT", "freeze", "ontology_fingerprint",
           "parser_fingerprint", "prompt_fingerprint", "scorer_fingerprint",
           "split_entries"]
