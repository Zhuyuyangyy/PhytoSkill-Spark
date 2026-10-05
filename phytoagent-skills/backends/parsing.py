"""Hardware-independent vision prompts and parsers.

The prompt text and the reply parser are the project's real asset: they encode
what may be observed and what counts as a verdict. They have nothing to do with
where inference runs.

They currently live in ``dgx.vision`` and ``dgx.herb_vision`` for a historical
reason only — DGX Spark was the first and only backend, so that is where they
were written. This module is the public door to them, so that new backends
depend on *parsing* and not on the DGX package name.

Moving the implementations themselves here is a follow-up. Until then this
module re-exports them under stable names; the symbols are identical, so
existing imports keep working and no behaviour changes.

Requires no SSH library. Importing this module must never require paramiko.
"""

from __future__ import annotations

import json

from dgx.herb_vision import (JUDGEMENT_MARKERS, NUM_PREDICT, PHENOTYPES as HERB_PHENOTYPES,
                             PROMPT_TEMPLATE as HERB_PROMPT_TEMPLATE,
                             _extract_json as _herb_extract_json,
                             _is_judgement,
                             _normalise_region as _herb_normalise_region,
                             _salvage_regions)
from dgx.herb_vision import build_prompt as build_herb_prompt
from dgx.vision import (DIAGNOSIS_MARKERS, PHENOTYPES as LEAF_PHENOTYPES,
                        PROMPT_TEMPLATE as LEAF_PROMPT_TEMPLATE,
                        _extract_json as _leaf_extract_json, _image_size,
                        _is_diagnosis,
                        _normalise_region as _leaf_normalise_region)
from dgx.vision import build_prompt as build_leaf_prompt

MODE_LIVE = "live"
MODE_HERB = "herb"

__all__ = [
    "DIAGNOSIS_MARKERS", "HERB_PHENOTYPES", "HERB_PROMPT_TEMPLATE",
    "JUDGEMENT_MARKERS", "LEAF_PHENOTYPES", "LEAF_PROMPT_TEMPLATE",
    "MODE_HERB", "MODE_LIVE", "NUM_PREDICT", "build_prompt",
    "extract_json", "image_size", "normalise_region", "parse_regions",
]


def build_prompt(*, species: str, mode: str) -> str:
    """Render the prompt for one mode.

    The cache keys on this exact text: editing a prompt must invalidate every
    observation taken with the old one.
    """
    if mode == MODE_LIVE:
        return build_leaf_prompt(species=species)
    if mode == MODE_HERB:
        return build_herb_prompt(species=species)
    raise ValueError(f"unknown vision mode {mode!r}")


def extract_json(text: str, *, mode: str):
    if mode == MODE_HERB:
        return _herb_extract_json(text)
    return _leaf_extract_json(text)


def normalise_region(raw, *, image_size: tuple[int, int] | None, mode: str):
    if mode == MODE_HERB:
        return _herb_normalise_region(raw, image_size=image_size)
    return _leaf_normalise_region(raw, image_size=image_size)


def image_size(image_bytes: bytes) -> tuple[int, int] | None:
    return _image_size(image_bytes)


def parse_regions(text: str, *, image_bytes: bytes, mode: str) -> tuple[bool, list[dict]]:
    """Parse one model reply into structured regions.

    Returns ``(image_usable, regions)``. Both defences apply: a region that
    fails normalisation is dropped, and a region whose text reads as a diagnosis
    or a quality verdict is dropped. The herb parser additionally recovers
    complete regions from a reply that was truncated before it closed.

    An empty region list is a valid outcome, not a failure to be repaired.
    """
    if mode not in (MODE_LIVE, MODE_HERB):
        raise ValueError(f"unknown vision mode {mode!r}")
    size = image_size(image_bytes)
    parsed = extract_json(text, mode=mode)

    regions: list[dict] = []
    image_usable = False

    if isinstance(parsed, dict):
        image_usable = parsed.get("image_usable") is True
        raw_regions = parsed.get("regions")
        if not isinstance(raw_regions, list) and mode == MODE_HERB:
            raw_regions = _salvage_regions(text)
    else:
        raw_regions = _salvage_regions(text) if mode == MODE_HERB else None
        image_usable = bool(raw_regions)

    if isinstance(raw_regions, list):
        for raw in raw_regions:
            region = normalise_region(raw, image_size=size, mode=mode)
            if region is None:
                continue
            blob = json.dumps(raw, ensure_ascii=False)
            if mode == MODE_HERB:
                if _is_judgement(blob):
                    continue
            elif _is_diagnosis(blob):
                continue
            regions.append(region)

    return image_usable, regions
