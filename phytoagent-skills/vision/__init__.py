"""Vision semantics: what may be observed, and what counts as a verdict.

This package is the project's real domain asset, and it is hardware-independent
by construction. The prompt text and the reply parser encode the observation
contract; they have nothing to do with where inference runs.

It exists because that semantics used to live in ``dgx.vision`` and
``dgx.herb_vision`` — a historical accident, since DGX Spark was the first and
only backend. A new backend should depend on *this* package, never on the DGX
package name. ``dgx.vision`` and ``dgx.herb_vision`` remain as compatibility
shims that re-export from here.

Importing this package must never require an SSH library.
"""

from vision.ontology import (
    DIAGNOSIS_MARKERS,
    HERB_PHENOTYPES,
    JUDGEMENT_MARKERS,
    LEAF_PHENOTYPES,
    MODE_HERB,
    MODE_LIVE,
    MODES,
    NUM_PREDICT,
    NUM_PREDICT_BY_MODE,
    PHENOTYPES_BY_MODE,
    phenotypes_for,
)
from vision.parser import (
    extract_json,
    image_size,
    is_diagnosis,
    is_judgement,
    normalise_region,
    parse_regions,
    salvage_regions,
)
from vision.prompts import (
    HERB_PROMPT_TEMPLATE,
    LEAF_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_BY_MODE,
    build_herb_prompt,
    build_leaf_prompt,
    build_prompt,
    num_predict_for,
)

__all__ = [
    "DIAGNOSIS_MARKERS", "HERB_PHENOTYPES", "HERB_PROMPT_TEMPLATE",
    "JUDGEMENT_MARKERS", "LEAF_PHENOTYPES", "LEAF_PROMPT_TEMPLATE", "MODE_HERB",
    "MODE_LIVE", "MODES", "NUM_PREDICT", "NUM_PREDICT_BY_MODE",
    "PHENOTYPES_BY_MODE", "PROMPT_TEMPLATE_BY_MODE", "build_herb_prompt",
    "build_leaf_prompt", "build_prompt", "extract_json", "image_size",
    "is_diagnosis", "is_judgement", "normalise_region", "num_predict_for",
    "parse_regions", "phenotypes_for", "salvage_regions",
]
