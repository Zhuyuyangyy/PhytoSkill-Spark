"""Freeze-then-join scoring: predictions are sealed before labels exist.

Hashing a file that already contains both predictions and labels proves
nothing about order — whoever produced it could have written either side last.
The only structural guarantee is temporal: **the predictions are frozen (their
bytes hashed and recorded) before the independent annotations arrive**, and
scoring joins the two files rather than trusting a pre-merged one.

This module implements that order:

* :func:`freeze_predictions` seals a predictions file — its hash, entry count,
  image list and the model identity it was produced with — at prediction time.
* :func:`join_for_scoring` refuses to merge unless the predictions file still
  hashes to its frozen record, and refuses any label whose image has no frozen
  prediction (a label for an image the model never saw under the freeze) or any
  frozen prediction whose image has no label (unscoreable).

The join output is the same entry shape the metrics consume, so the scoring
path is the one the holdout already uses.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.split import FreezeError
from evaluation.identity import _sha256

DEFAULT_PREDICTION_STORE = Path("artifacts/evaluations/predictions")

# Fields a predictions file contributes, per image. A prediction without a
# model verdict for an image is a hole, not a negative.
PREDICTION_FIELDS = ("model_predicted", "model_image_usable",
                     "model_observation_count")
# Fields a labels file contributes, per image. These come from the annotators,
# never from the model run.
LABEL_FIELDS = ("annotator_labels", "candidate_labels", "annotator_agrees",
                "annotator_note")


def _entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("entries", data.get("results", []))
    if not isinstance(data, list) or not data:
        raise FreezeError(f"no entries in {path}")
    return data


def _key(entry: dict) -> tuple[str, str]:
    image = entry.get("image")
    if not image:
        raise FreezeError("an entry has no image id; cannot join predictions to labels")
    return (str(image), str(entry.get("mode", "")))


def freeze_predictions(path: str | Path, *, model: str, model_digest: str,
                       freeze_id: str,
                       store: str | Path = DEFAULT_PREDICTION_STORE,
                       attest: str = "") -> dict:
    """Seal a predictions file against the published experiment identity.

    Three things are bound at seal time, because each one is otherwise
    swappable afterwards:

    * the **model identity** — name *and* weight digest. A name alone is a
      tag; the digest is the bytes. ``model_digest`` is required: sealing
      against a tag lets the server silently serve different weights.
    * the **freeze_id** of the published manifest — the predictions belong to
      one frozen configuration, and scoring under a different one is refused.
    * the **pre-registered image set** — the entries in this file are the
      evaluation set; scoring a subset of them is a different experiment.

    The record is content-addressed (written under the file's own hash) and
    **first-seal-wins**: re-sealing the same bytes cannot move the timestamp,
    so the earlier seal is the one that counts. That is auditable evidence of
    order — not tamper-proof proof of it; proof needs an independent
    custodian's delivery record, which ``attest`` names when there is one.
    """
    path = Path(path)
    if not path.is_file():
        raise FreezeError(f"predictions file not found: {path}")
    if not model:
        raise FreezeError("a predictions freeze must name the model")
    if not model_digest:
        raise FreezeError(
            "a predictions freeze must pin the weight digest; a model name is a "
            "tag, not an identity")
    if not freeze_id:
        raise FreezeError(
            "a predictions freeze must name the published freeze_id it belongs to")
    digest = _sha256(path)
    entries = _entries(path)

    # A predictions file that carries annotator fields is the shape of "the
    # labels were seen first": the model's output and a human's answer do not
    # belong in the same file at seal time.
    contaminated = sorted({field for entry in entries for field in LABEL_FIELDS
                           if field in entry})
    if contaminated:
        raise FreezeError(
            f"predictions file carries annotator fields {contaminated}; predictions "
            "must be sealed before any label exists, in a file of their own")

    images = sorted({str(entry.get("image")) for entry in entries if entry.get("image")})
    if len(images) != len(entries):
        raise FreezeError("predictions must hold exactly one entry per image")
    target = Path(store) / f"{digest}.json"
    if target.is_file():
        # First seal wins: the timestamp of the earliest record is the one that
        # counts, and a later re-seal cannot quietly replace it.
        return json.loads(target.read_text(encoding="utf-8"))
    record = {
        "schema_version": 1,
        "predictions_sha256": digest,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "model_digest": model_digest,
        "freeze_id": freeze_id,
        "entries": len(entries),
        "images": images,
        "source_file": str(path),
        "sealed_before_labels": True,
        "custodian_attestation": attest,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return record


def load_frozen_predictions(digest: str,
                            store: str | Path = DEFAULT_PREDICTION_STORE) -> dict:
    target = Path(store) / f"{digest}.json"
    if not target.is_file():
        raise FreezeError(
            f"no frozen prediction record for {digest[:12]}; freeze the predictions "
            "before the labels exist (python -m evaluation freeze-predictions ...)")
    return json.loads(target.read_text(encoding="utf-8"))


def verify_frozen_predictions(path: str | Path, record: dict) -> str:
    """The file must still be the bytes that were frozen."""
    path = Path(path)
    digest = _sha256(path)
    if digest != record["predictions_sha256"]:
        raise FreezeError(
            "predictions no longer match their frozen record: "
            f"{digest[:12]} != {record['predictions_sha256'][:12]}")
    return digest


def join_for_scoring(predictions_path: str | Path, labels_path: str | Path, *,
                     store: str | Path = DEFAULT_PREDICTION_STORE,
                     expected: dict | None = None) -> dict:
    """Merge frozen predictions with independent labels, or refuse.

    The merge is the scoring input. It exists only when the predictions were
    frozen first: the frozen record's hash must match the file's current bytes,
    the sealed model identity must match the scoring freeze (``expected``), and
    the two sides must cover exactly the same (image, mode) keys — a label for
    an image the model never saw under the freeze, or a frozen prediction with
    no label, is a refusal rather than a silent drop.

    What this proves: the bytes scored are the bytes sealed. What it does not
    prove: that the seal happened before the labels were written — that needs
    an independent custodian's delivery record, named in the seal.
    """
    predictions_path, labels_path = Path(predictions_path), Path(labels_path)
    digest = _sha256(predictions_path)
    record = load_frozen_predictions(digest, store)
    verify_frozen_predictions(predictions_path, record)

    # The sealed predictions belong to one published experiment. Scoring them
    # under a different freeze — or with the model identity the seal never
    # named — records one experiment's numbers against another's name.
    if expected is not None:
        for field in ("freeze_id", "model", "model_digest"):
            wanted = expected.get(field, "")
            sealed = record.get(field, "")
            if wanted and wanted != sealed:
                raise FreezeError(
                    f"sealed predictions do not match the scoring freeze: "
                    f"{field} is {sealed or 'unset'!r}, the freeze says {wanted!r}")

    prediction_entries = _entries(predictions_path)
    label_entries = _entries(labels_path)
    predictions = _index(prediction_entries, "predictions")
    labels = _index(label_entries, "labels")

    unlabelled = sorted(key for key in predictions if key not in labels)
    unpredicted = sorted(key for key in labels if key not in predictions)
    if unlabelled:
        raise FreezeError(f"frozen predictions without labels: {unlabelled}")
    if unpredicted:
        raise FreezeError(
            f"labels without frozen predictions (the model never saw them under "
            f"the freeze): {unpredicted}")

    merged = []
    for key, prediction in sorted(predictions.items()):
        label = labels[key]
        entry = {"image": key[0], "mode": key[1]}
        for field in PREDICTION_FIELDS:
            if field in prediction:
                entry[field] = prediction[field]
        for field in LABEL_FIELDS:
            if field in label:
                entry[field] = label[field]
        merged.append(entry)
    return {
        "entries": merged,
        "predictions_sha256": digest,
        "labels_sha256": _sha256(labels_path),
        "frozen_at": record["frozen_at"],
        "model": record["model"],
        "model_digest": record["model_digest"],
        "freeze_id": record.get("freeze_id", ""),
        "pre_registered": list(record.get("images", [])),
    }


def _index(entries: list[dict], side: str) -> dict[tuple[str, str], dict]:
    """Index entries by the full (image, mode) key, refusing duplicates.

    The key is the pair, not the image alone: the same photograph scored in two
    modes is two records, and a repeated key is a data error — not something a
    later entry should silently win.
    """
    indexed: dict[tuple[str, str], dict] = {}
    for entry in entries:
        key = _key(entry)
        if key in indexed:
            raise FreezeError(f"{side} file repeats the (image, mode) key {key}")
        indexed[key] = entry
    return indexed


__all__ = ["DEFAULT_PREDICTION_STORE", "PREDICTION_FIELDS", "LABEL_FIELDS",
           "freeze_predictions", "join_for_scoring", "load_frozen_predictions",
           "verify_frozen_predictions"]
