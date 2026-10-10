"""Parsing a model reply into structured observations.

Two defences apply to every region, in both modes:

1. A region that fails contract validation is dropped, not coerced. A bbox with
   out-of-range or inverted coordinates is not evidence.
2. A region whose text reads as a diagnosis (live) or a judgement (herb) is
   dropped. The model is asked to describe; a reply that names a cause or a
   grade is refused rather than recorded.

An empty region list is a valid outcome, not a failure to be repaired.

Hardware-independent: this module never imports a transport. Moving it out of
``dgx.vision`` / ``dgx.herb_vision`` is what lets a new backend depend on
*parsing* rather than on the DGX package name.
"""

from __future__ import annotations

import json

from vision.ontology import (
    DIAGNOSIS_MARKERS,
    JUDGEMENT_MARKERS,
    MODE_HERB,
    MODE_LIVE,
    phenotypes_for,
)

# ── low-level extraction ─────────────────────────────────────────────────────


def image_size(image_bytes: bytes) -> tuple[int, int] | None:
    """Width and height of the image, read from its header.

    Returns None when the bytes are not a decodable image: the caller then
    refuses pixel-space boxes rather than guessing a canvas.
    """
    import io

    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - Pillow is a test extra
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.size
    except Exception:  # noqa: BLE001 - any decode failure means "unknown size"
        return None


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a free-form model reply."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def salvage_regions(text: str) -> list[dict]:
    """Recover region objects from JSON the model did not finish.

    A model asked to enumerate regions sometimes runs out of budget mid-array,
    so the document never closes and ``json.loads`` rejects the whole thing.
    This collects every ``{...}`` span that parses on its own and keeps the ones
    that name a phenotype, so the entries it did finish are not lost with the
    document.

    Implementation note: a stack of open-brace offsets is kept, and every
    closing brace tries to parse the span it closes. That finds an inner region
    inside an outer wrapper as its own candidate, which a single depth counter
    cannot do — the wrapper is still open when the region closes.
    """
    if not text:
        return []
    recovered: list[dict] = []
    open_spans: list[int] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            open_spans.append(index)
        elif character == "}" and open_spans:
            span_start = open_spans.pop()
            try:
                candidate = json.loads(text[span_start:index + 1])
            except json.JSONDecodeError:
                continue
            # Only the region objects are wanted. The outer wrapper
            # ({"image_usable": ..., "regions": [...]}) has no phenotype and is
            # skipped; the regions inside it are their own candidates.
            if isinstance(candidate, dict) and "phenotype" in candidate:
                recovered.append(candidate)
    return recovered


# ── region validation ────────────────────────────────────────────────────────


def normalise_region(raw: object, *, image_size: tuple[int, int] | None = None,
                     mode: str = MODE_LIVE) -> dict | None:
    """Validate one model-reported region against the contract.

    Anything that fails validation is dropped rather than coerced: a bbox with
    out-of-range or inverted coordinates is not evidence.
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("phenotype") not in phenotypes_for(mode):
        return None
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        return None
    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    # A model that ignores the normalisation instruction returns pixel values.
    # Rescale rather than discard a real observation, but only against the real
    # image size, and only when the box is unambiguously in pixel space.
    #
    # The test is "every coordinate exceeds 1", not "any coordinate does". A
    # normalised box has at least one coordinate at or below 1 by definition, and
    # a mixed box such as [0, 0, 1.5, 0.5] is malformed rather than pixel-scaled:
    # dividing it by the canvas size would turn a nonsense box into a tiny
    # plausible-looking one, which is worse than refusing it.
    if min(left, top, right, bottom) > 1.0:
        if image_size is None:
            return None
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        left, right = left / width, right / width
        top, bottom = top / height, bottom / height
    if not all(0.0 <= value <= 1.0 for value in (left, top, right, bottom)):
        return None
    if left >= right or top >= bottom:
        return None
    # A box that covers the whole frame is not a region observation. It is the
    # model declining to localise, and reporting it would turn "I see the image"
    # into an evidence-backed claim about a place in it.
    if (right - left) >= 0.98 and (bottom - top) >= 0.98:
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"phenotype": raw["phenotype"], "label": label.strip()[:80],
            "bbox_normalized": [left, top, right, bottom],
            "model_score": max(0.0, min(1.0, confidence))}


def is_diagnosis(text: str) -> bool:
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in DIAGNOSIS_MARKERS)


def is_judgement(text: str) -> bool:
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in JUDGEMENT_MARKERS)


# ── the composed parser ──────────────────────────────────────────────────────


def parse_regions(text: str, *, image_bytes: bytes, mode: str) -> tuple[bool, list[dict]]:
    """Parse one model reply into structured regions.

    Returns ``(image_usable, regions)``. Both defences apply: a region that
    fails normalisation is dropped, and a region whose text reads as a diagnosis
    or a judgement is dropped. The herb parser additionally recovers complete
    regions from a reply that was truncated before it closed.

    An empty region list is a valid outcome, not a failure to be repaired.
    """
    if mode not in (MODE_LIVE, MODE_HERB):
        raise ValueError(f"unknown vision mode {mode!r}")
    size = image_size(image_bytes)
    parsed = extract_json(text)

    regions: list[dict] = []
    image_usable = False

    if isinstance(parsed, dict):
        image_usable = parsed.get("image_usable") is True
        raw_regions = parsed.get("regions")
        if not isinstance(raw_regions, list) and mode == MODE_HERB:
            raw_regions = salvage_regions(text)
    else:
        raw_regions = salvage_regions(text) if mode == MODE_HERB else None
        image_usable = bool(raw_regions)

    if isinstance(raw_regions, list):
        for raw in raw_regions:
            region = normalise_region(raw, image_size=size, mode=mode)
            if region is None:
                continue
            blob = json.dumps(raw, ensure_ascii=False)
            if mode == MODE_HERB:
                if is_judgement(blob):
                    continue
            elif is_diagnosis(blob):
                continue
            regions.append(region)

    return image_usable, regions


__all__ = [
    "extract_json", "image_size", "is_diagnosis", "is_judgement",
    "normalise_region", "parse_regions", "salvage_regions",
]
