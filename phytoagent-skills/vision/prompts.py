"""Prompt templates for the vision modes.

The prompt is the first line of defence: it asks for observation, not
diagnosis. The exact text is a contract — the observation cache keys on it, so
editing a prompt must invalidate every observation taken with the old one.

Hardware-independent. The text never mentions where inference runs.
"""

from __future__ import annotations

from vision.ontology import (
    HERB_PHENOTYPES,
    MODE_HERB,
    MODE_LIVE,
    NUM_PREDICT_BY_MODE,
)

# ── live (leaf / field) ──────────────────────────────────────────────────────

LEAF_PROMPT_TEMPLATE = """Report coloured regions in this image as JSON only.

Format: {"image_usable": true, "regions": [{"phenotype": "leaf_yellowing"|"leaf_spot"|"wilting"|"unknown", "label": "short text", "bbox": [l, t, r, b], "confidence": 0-1}]}

Rules: bbox normalised 0-1 (divide pixels by image width and height). Yellow areas -> leaf_yellowing, small dark spots -> leaf_spot, drooping -> wilting, else -> unknown. Never name a disease or cause. No visible region -> {"image_usable": false, "regions": []}. JSON only.

Species: @@SPECIES@@
"""

# ── herb (dried sliced material) ─────────────────────────────────────────────

HERB_PROMPT_TEMPLATE = """Observe this photograph of dried sliced medicinal herb material. Report what is visible as JSON only.

Format: {"image_usable": true, "regions": [{"phenotype": "<one of PHENOTYPES>", "label": "short text", "bbox": [l, t, r, b], "confidence": 0-1}]}

PHENOTYPES = @@PHENOTYPES@@

Definitions that matter (read before choosing):
- cut_surface_powder: the cut face looks starchy or mealy, sheds fine powder,
  matte and slightly granular. THIS IS THE NORMAL STATE for sliced Astragalus.
- cut_surface_dense: the cut face is glassy, translucent or waxy, NOT powdery,
  and cleanly compact such that you could not rub powder off it. Only choose this
  when powdery is clearly absent. A firm but powdery face is powder, NOT dense.
- cut_surface_fissure: a visible crack, split or shatter running into the slice.
- cut_surface_hollow: a real cavity or central void, not just a shallow dish.
- colour_amber: the cut face itself reads amber/honey-brown overall.
- colour_dark_brown: genuinely dark brown or scorched areas, not bark.

Rules:
- bbox normalised 0-1 (divide pixel coordinates by image width and height).
- Bark/skin colour is not cut-surface colour; describe it only as part of the label.
- Never state a quality grade, price, authenticity verdict, or cause. You are describing, not judging.
- If the only visible property is colour, report colour; do not add texture.
- No visible feature of interest -> {"image_usable": false, "regions": []}.
- JSON only, no explanation, no markdown fence.

Species: @@SPECIES@@
"""

PROMPT_TEMPLATE_BY_MODE = {MODE_LIVE: LEAF_PROMPT_TEMPLATE,
                           MODE_HERB: HERB_PROMPT_TEMPLATE}


def build_leaf_prompt(*, species: str) -> str:
    """Render the live-mode prompt for one species."""
    return LEAF_PROMPT_TEMPLATE.replace("@@SPECIES@@", species)


def build_herb_prompt(*, species: str) -> str:
    """Render the herb-mode prompt for one species."""
    return (HERB_PROMPT_TEMPLATE
            .replace("@@PHENOTYPES@@", "|".join(HERB_PHENOTYPES))
            .replace("@@SPECIES@@", species))


def build_prompt(*, species: str, mode: str) -> str:
    """Render the prompt for one mode.

    Exposed so a caller can feed the exact text to the observation cache;
    without it a prompt edit would reuse stale observations.
    """
    if mode == MODE_LIVE:
        return build_leaf_prompt(species=species)
    if mode == MODE_HERB:
        return build_herb_prompt(species=species)
    raise ValueError(f"unknown vision mode {mode!r}")


def num_predict_for(mode: str) -> int:
    try:
        return NUM_PREDICT_BY_MODE[mode]
    except KeyError:
        raise ValueError(f"unknown vision mode {mode!r}") from None


__all__ = [
    "HERB_PROMPT_TEMPLATE", "LEAF_PROMPT_TEMPLATE", "PROMPT_TEMPLATE_BY_MODE",
    "build_herb_prompt", "build_leaf_prompt", "build_prompt", "num_predict_for",
]
