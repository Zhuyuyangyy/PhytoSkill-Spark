"""The observation vocabulary for vision modes.

This module is hardware-independent by construction: it says *what may be
observed* and *what is not an observation*, and nothing here knows where
inference runs. It lived in ``dgx.vision`` / ``dgx.herb_vision`` only because
DGX Spark was the first and only backend; the semantics were never DGX's.

Two modes, two vocabularies:

* ``live`` — field/leaf phenotypes (yellowing, spots, wilting).
* ``herb`` — dried sliced medicinal material (cut surface, colour, defects).

The guard markers are the second line of defence. The prompt asks the model to
describe rather than diagnose; these markers drop a reply that names a cause or
a verdict anyway.
"""

from __future__ import annotations

# ── modes ────────────────────────────────────────────────────────────────────

MODE_LIVE = "live"
MODE_HERB = "herb"
MODES = (MODE_LIVE, MODE_HERB)

# ── live (leaf / field) vocabulary ───────────────────────────────────────────

LEAF_PHENOTYPES = ("leaf_yellowing", "leaf_spot", "wilting", "unknown")

# A reply containing any of these is a diagnosis, and the region is dropped.
DIAGNOSIS_MARKERS = ("病原", "病因", "病害是", "感染了", "确诊", "disease is",
                     "caused by", "pathogen")

# ── herb (dried sliced material) vocabulary ──────────────────────────────────

# These are what the model is asked to name; anything else it says is still
# recorded verbatim under ``raw_response`` but is not promoted to a structured
# observation.
HERB_PHENOTYPES = (
    "cut_surface_fissure",   # 裂隙 / 炸裂
    "cut_surface_powder",    # 粉性足，断面呈粉末状
    "cut_surface_dense",     # 角质 / 致密
    "cut_surface_hollow",    # 空心
    "colour_pale_yellow",    # 色泽淡黄（硫熏后常见）
    "colour_amber",          # 黄棕 / 琥珀色
    "colour_dark_brown",     # 深褐 / 焦褐
    "mould_visible",         # 可见霉斑
    "insect_damage",         # 虫蛀孔道
    "slice_irregular",       # 片型不整 / 厚薄不均
    "unknown",
)

# A reply containing any of these is a judgement, not an observation, and the
# region is dropped. "Grade A", "authentic", "because of mould" are all verdicts.
JUDGEMENT_MARKERS = ("等级", "为正品", "是假", "伪品", "优质", "劣质", "合格", "不合格",
                     "grade ", "authentic", "fake", "counterfeit", "because of",
                     "caused by", "due to", "should be rejected")

PHENOTYPES_BY_MODE = {MODE_LIVE: LEAF_PHENOTYPES, MODE_HERB: HERB_PHENOTYPES}

# ── generation budget ────────────────────────────────────────────────────────

# num_predict is the real constraint on the prompt, not its character count. A
# herb photo with many slices makes the model enumerate them one by one: the
# worst observed case ran to 1500 tokens and still had not finished, because
# salvage_regions recovers the completed entries. 1500 was reached on 4 of 15
# images, so the budget was raised rather than the prompt trimmed.
NUM_PREDICT_BY_MODE = {MODE_LIVE: 400, MODE_HERB: 3000}

# Kept for callers that predate the per-mode budget (``backends.parsing`` and
# ``dgx.herb_vision`` both export it). It is the herb budget.
NUM_PREDICT = NUM_PREDICT_BY_MODE[MODE_HERB]


def phenotypes_for(mode: str) -> tuple[str, ...]:
    try:
        return PHENOTYPES_BY_MODE[mode]
    except KeyError:
        raise ValueError(f"unknown vision mode {mode!r}") from None


__all__ = [
    "DIAGNOSIS_MARKERS", "HERB_PHENOTYPES", "JUDGEMENT_MARKERS", "LEAF_PHENOTYPES",
    "MODE_HERB", "MODE_LIVE", "MODES", "NUM_PREDICT", "NUM_PREDICT_BY_MODE",
    "PHENOTYPES_BY_MODE", "phenotypes_for",
]
