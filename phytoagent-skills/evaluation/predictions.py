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


def freeze_predictions(path: str | Path, *, model: str, model_digest: str = "",
                       store: str | Path = DEFAULT_PREDICTION_STORE) -> dict:
    """Seal a predictions file. An explicit, timestamped publisher act.

    The record is written under the file's own hash, so it can be found again
    without a registry: the predictions are identified by their bytes.
    """
    path = Path(path)
    if not path.is_file():
        raise FreezeError(f"predictions file not found: {path}")
    digest = _sha256(path)
    entries = _entries(path)
    images = sorted({str(entry.get("image")) for entry in entries if entry.get("image")})
    if len(images) != len(entries):
        raise FreezeError("predictions must hold exactly one entry per image")
    record = {
        "schema_version": 1,
        "predictions_sha256": digest,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "model_digest": model_digest,
        "entries": len(entries),
        "images": images,
        "source_file": str(path),
    }
    target = Path(store) / f"{digest}.json"
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
                     store: str | Path = DEFAULT_PREDICTION_STORE) -> dict:
    """Merge frozen predictions with independent labels, or refuse.

    The merge is the scoring input. It exists only when the predictions were
    frozen first: the frozen record's hash must match the file's current bytes,
    every label must have a frozen prediction behind it, and every frozen
    prediction must have a label — an unscoreable prediction is reported, not
    silently dropped.
    """
    predictions_path, labels_path = Path(predictions_path), Path(labels_path)
    digest = _sha256(predictions_path)
    record = load_frozen_predictions(digest, store)
    verify_frozen_predictions(predictions_path, record)

    predictions = {_key(entry): entry for entry in _entries(predictions_path)}
    labels = {_key(entry): entry for entry in _entries(labels_path)}

    unlabelled = sorted(image for image, _ in predictions if image not in
                        {image for image, _ in labels})
    unpredicted = sorted(image for image, _ in labels if image not in
                         {image for image, _ in predictions})
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
    }


__all__ = ["DEFAULT_PREDICTION_STORE", "PREDICTION_FIELDS", "LABEL_FIELDS",
           "freeze_predictions", "join_for_scoring", "load_frozen_predictions",
           "verify_frozen_predictions"]
