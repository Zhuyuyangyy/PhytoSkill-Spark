"""Tests for the herb-material vision prompt and parser.

Offline: these exercise the prompt, the JSON extraction and the region
normalisation without touching a live GPU. The real runs are recorded in
artifacts/dgx/herb-batch.json.
"""

from __future__ import annotations

import json

import pytest

from dgx.herb_vision import (JUDGEMENT_MARKERS, PHENOTYPES, build_prompt,
                             build_remote_script, _extract_json, _is_judgement,
                             _normalise_region, _salvage_regions)

MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


# ── prompt ───────────────────────────────────────────────────────────────────


def test_the_prompt_lists_the_herb_vocabulary():
    rendered = build_prompt(species="黄芪")
    for phenotype in PHENOTYPES:
        assert phenotype in rendered


def test_the_prompt_substitutes_the_species():
    rendered = build_prompt(species="人参")
    assert "人参" in rendered
    assert "@@SPECIES@@" not in rendered
    assert "@@PHENOTYPES@@" not in rendered


def test_the_prompt_forbids_judgement():
    lowered = build_prompt(species="黄芪").lower()
    assert "never state a quality grade" in lowered
    assert "authenticity verdict" in lowered


def test_the_prompt_allows_an_empty_result():
    """An honest empty answer must be an allowed outcome, not a failure."""
    assert '"regions": []' in build_prompt(species="黄芪")


def test_the_prompt_demands_normalised_coordinates():
    assert "normalised 0-1" in build_prompt(species="黄芪")


def test_the_prompt_is_short_relative_to_the_generation_budget():
    """The prompt must leave room for the answer.

    The constraint is not a character count: it is that prompt plus reply must fit
    the model's budget. A long prompt made one version exhaust ``num_predict``
    before answering at all, so the ratio is what the test guards.

    The prompt later grew to ~1670 characters when the phenotype definitions were
    written out. That was deliberate and correct — the definitions exist to stop
    exactly the ``cut_surface_dense`` / ``cut_surface_powder`` confusion the
    annotations found. The budget was raised instead of trimming them.
    """
    from dgx.herb_vision import NUM_PREDICT
    prompt = build_prompt(species="黄芪")
    # Roughly 1 token per 3 characters for mixed English/CJK; the reply needs at
    # least as much room as the prompt consumes.
    prompt_tokens_estimate = len(prompt) / 3
    assert prompt_tokens_estimate < NUM_PREDICT / 2, (
        f"prompt ~{prompt_tokens_estimate:.0f} tokens leaves too little of the "
        f"{NUM_PREDICT}-token generation budget")


def test_the_generation_budget_is_headroom_for_many_slices():
    """A photo with many slices makes the model enumerate each one.

    The worst observed case reached the previous 1500-token budget, so the budget
    must stay above that or the tail of the report is lost.
    """
    from dgx.herb_vision import NUM_PREDICT
    assert NUM_PREDICT > 1500


def test_the_herb_vocabulary_is_not_the_leaf_vocabulary():
    """The instrument must match the subject.

    The supplied images are sliced root material, so a leaf-chlorosis vocabulary
    would force the model to answer a question the photograph cannot answer.
    """
    assert "leaf_yellowing" not in PHENOTYPES
    assert "cut_surface_fissure" in PHENOTYPES
    assert "mould_visible" in PHENOTYPES


# ── JSON extraction ──────────────────────────────────────────────────────────


def test_a_fenced_json_block_is_extracted():
    assert _extract_json('```json\n{"image_usable": true, "regions": []}\n```') == \
        {"image_usable": True, "regions": []}


def test_prose_around_json_is_ignored():
    text = 'I see: {"image_usable": false, "regions": []} That is all.'
    assert _extract_json(text) == {"image_usable": False, "regions": []}


@pytest.mark.parametrize("text", ["", "no json", "{broken", "{}"])
def test_unusable_text_returns_none_or_empty(text):
    assert _extract_json(text) in (None, {})


# ── truncated-JSON salvage ───────────────────────────────────────────────────


TRUNCATED = (
    '{"image_usable": true, "regions": ['
    '{"phenotype": "colour_pale_yellow", "label": "a", "bbox": [1,2,3,4], "confidence": 0.9}, '
    '{"phenotype": "colour_amber", "label": "b", "bbox": [5,6,7,8], "confidence": 0.8}, '
    '{"phenotype": "mou'
)

COMPLETE = (
    '{"image_usable": true, "regions": ['
    '{"phenotype": "colour_amber", "label": "bark", "bbox": [0.1,0.2,0.3,0.4], "confidence": 0.8}'
    ']}'
)


def test_a_document_the_model_did_not_finish_still_yields_its_regions():
    """Running out of budget mid-array must not lose the finished entries."""
    regions = _salvage_regions(TRUNCATED)
    assert [r["phenotype"] for r in regions] == ["colour_pale_yellow", "colour_amber"]


def test_the_unfinished_final_region_is_dropped_not_guessed():
    regions = _salvage_regions(TRUNCATED)
    assert all(r["phenotype"] != "mou" for r in regions)
    assert len(regions) == 2


def test_a_complete_document_yields_only_its_regions():
    regions = _salvage_regions(COMPLETE)
    assert len(regions) == 1
    assert regions[0]["label"] == "bark"


def test_the_outer_wrapper_is_not_mistaken_for_a_region():
    regions = _salvage_regions(COMPLETE)
    assert all("phenotype" in r for r in regions)


@pytest.mark.parametrize("text", ["", "nothing here", "{", "}"])
def test_salvage_on_text_without_regions(text):
    assert _salvage_regions(text) == []


def test_nested_braves_inside_a_label_do_not_confuse_the_scanner():
    text = ('{"regions": [{"phenotype": "slice_irregular", "label": "a {b} c", '
            '"bbox": [0.1,0.1,0.2,0.2], "confidence": 0.5}]}')
    regions = _salvage_regions(text)
    assert len(regions) == 1
    assert regions[0]["label"] == "a {b} c"


def test_a_quoted_brace_does_not_open_a_span():
    text = '{"regions": [{"phenotype": "unknown", "label": "}{", "bbox": [0,0,0,0]}]}'
    # Malformed bbox is fine here; the point is the scanner survives odd quoting.
    assert isinstance(_salvage_regions(text), list)


# ── region normalisation ─────────────────────────────────────────────────────


def test_a_normalised_box_is_passed_through():
    raw = {"phenotype": "cut_surface_fissure", "label": "radial cracks",
           "bbox": [0.2, 0.3, 0.6, 0.7], "confidence": 0.85}
    assert _normalise_region(raw, image_size=(800, 800))["bbox_normalized"] == \
        [0.2, 0.3, 0.6, 0.7]


def test_a_pixel_box_is_rescaled_against_the_real_image_size():
    """The model really does answer in pixels; the runs show it repeatedly."""
    raw = {"phenotype": "colour_pale_yellow", "label": "pale area",
           "bbox": [100, 200, 300, 400], "confidence": 0.9}
    region = _normalise_region(raw, image_size=(800, 800))
    assert region["bbox_normalized"] == [0.125, 0.25, 0.375, 0.5]


def test_a_pixel_box_beyond_the_canvas_is_refused():
    raw = {"phenotype": "mould_visible", "label": "spot",
           "bbox": [100, 900, 300, 1100], "confidence": 0.9}
    assert _normalise_region(raw, image_size=(800, 800)) is None


def test_a_pixel_box_without_a_known_size_is_refused():
    raw = {"phenotype": "mould_visible", "label": "spot",
           "bbox": [100, 200, 300, 400], "confidence": 0.9}
    assert _normalise_region(raw, image_size=None) is None


def test_a_mixed_box_is_refused_rather_than_rescaled():
    """[0, 0, 1.5, 0.5] is malformed, not pixel-scaled.

    Dividing it by the canvas would produce a tiny plausible-looking box, which
    is worse than refusing it.
    """
    raw = {"phenotype": "slice_irregular", "label": "x",
           "bbox": [0.0, 0.0, 1.5, 0.5], "confidence": 0.5}
    assert _normalise_region(raw, image_size=(800, 800)) is None


@pytest.mark.parametrize("bbox", [
    [0.8, 0.2, 0.1, 0.7],
    [0.2, 0.9, 0.7, 0.3],
    [-0.1, 0.0, 0.5, 0.5],
    [0.1, 0.2, 0.3],
])
def test_a_malformed_box_is_refused(bbox):
    raw = {"phenotype": "slice_irregular", "label": "x", "bbox": bbox, "confidence": 0.5}
    assert _normalise_region(raw, image_size=(800, 800)) is None


def test_a_phenotype_outside_the_vocabulary_is_refused():
    raw = {"phenotype": "leaf_yellowing", "label": "x",
           "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.5}
    assert _normalise_region(raw, image_size=(800, 800)) is None


def test_a_box_covering_the_whole_frame_is_not_a_region():
    """A full-frame box is the model declining to localise.

    The herb prompt is especially prone to this: asked about a plate of slices,
    the model sometimes answers with one region spanning the whole photograph.
    """
    raw = {"phenotype": "unknown", "label": "everything",
           "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 0.5}
    assert _normalise_region(raw, image_size=(800, 800)) is None


def test_a_near_full_frame_box_is_refused_too():
    raw = {"phenotype": "colour_amber", "label": "most of it",
           "bbox": [0.01, 0.01, 0.99, 0.99], "confidence": 0.5}
    assert _normalise_region(raw, image_size=(800, 800)) is None


# ── judgement filter ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["此为正品", "优质药材", "等级一等", "grade A",
                                  "authentic astragalus", "fake product",
                                  "because of mould", "caused by fungi"])
def test_a_judgement_is_detected(text):
    assert _is_judgement(text) is True


@pytest.mark.parametrize("text", ["断面有裂隙", "色泽淡黄", "可见霉斑",
                                  "radial cracks on the cut surface"])
def test_an_observation_is_not_a_judgement(text):
    assert _is_judgement(text) is False


def test_the_marker_list_covers_both_languages():
    assert any("\u4e2d" <= c <= "\u9fff" for m in JUDGEMENT_MARKERS for c in m)
    assert any(m.isascii() for m in JUDGEMENT_MARKERS)


# ── remote script ────────────────────────────────────────────────────────────


def test_the_rendered_remote_script_is_self_contained():
    """The remote script runs in a separate interpreter on the node.

    A local constant referenced by name is NOT in scope there, so the value has to
    be interpolated at render time. This bit for real: a ``num_predict:
    NUM_PREDICT`` reached the node as a bare name and every call died with
    NameError.
    """
    script = build_remote_script(model=MODEL, image_b64="AAAA",
                                 species="黄芪", prompt="look")
    assert "num_predict" in script
    assert "NUM_PREDICT" not in script  # no unresolved local name
    compile(script, "<remote>", "exec")   # must parse standalone


def test_the_remote_script_reports_the_real_generation_budget():
    from dgx.herb_vision import NUM_PREDICT
    script = build_remote_script(model=MODEL, image_b64="AAAA",
                                 species="黄芪", prompt="look")
    assert str(NUM_PREDICT) in script


def test_the_remote_script_carries_the_prompt_and_model():
    script = build_remote_script(model=MODEL, image_b64="AAAA", species="黄芪",
                                 prompt=build_prompt(species="黄芪"))
    assert MODEL in script
    assert "黄芪" in script
    assert "cut_surface_fissure" in script
    assert "11434" in script
