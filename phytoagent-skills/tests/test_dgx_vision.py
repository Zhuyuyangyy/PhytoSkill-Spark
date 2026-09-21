"""Tests for the DGX vision parser.

These run offline: they exercise the prompt, the JSON extraction and the region
normalisation without touching a live GPU. The remote call itself is covered by
the recorded runs in artifacts/dgx/.
"""

from __future__ import annotations

import json

import pytest

from dgx.vision import (PROMPT_TEMPLATE, _extract_json, _is_diagnosis,
                        _normalise_region, build_remote_script)

MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


# ── prompt ───────────────────────────────────────────────────────────────────


def test_the_prompt_demands_normalised_coordinates():
    assert "0-1" in PROMPT_TEMPLATE
    assert "normalised" in PROMPT_TEMPLATE.lower()


def test_the_prompt_forbids_diagnosis():
    # "Never name a disease or cause" is the operative clause; the exact wording
    # changed when the prompt was shortened to fit the model's token budget.
    lowered = PROMPT_TEMPLATE.lower()
    assert "never name a disease or cause" in lowered


def test_the_prompt_asks_for_an_empty_result_when_nothing_is_visible():
    """An honest empty answer must be an allowed outcome, not a failure."""
    assert '"regions": []' in PROMPT_TEMPLATE
    assert "image_usable" in PROMPT_TEMPLATE


def test_the_prompt_stays_short_enough_for_the_token_budget():
    """A long prompt made the model exhaust num_predict before answering.

    The first version ran to eval_count=400 (the cap) and returned nothing; the
    shortened one answers in ~310 tokens. Keep it compact.
    """
    assert len(PROMPT_TEMPLATE) < 700


def test_the_prompt_substitutes_the_species():
    rendered = PROMPT_TEMPLATE.replace("@@SPECIES@@", "人参")
    assert "人参" in rendered
    assert "@@SPECIES@@" not in rendered


# ── JSON extraction ──────────────────────────────────────────────────────────


def test_a_fenced_json_block_is_extracted():
    text = '```json\n{"image_usable": true, "regions": []}\n```'
    assert _extract_json(text) == {"image_usable": True, "regions": []}


def test_prose_around_json_is_ignored():
    text = 'Here is what I see: {"image_usable": false, "regions": []} Hope that helps.'
    assert _extract_json(text) == {"image_usable": False, "regions": []}


def test_unparseable_text_returns_none():
    assert _extract_json("no json here") is None
    assert _extract_json("{broken") is None
    assert _extract_json("") is None


# ── region normalisation ─────────────────────────────────────────────────────


def test_a_normalised_box_is_passed_through():
    raw = {"phenotype": "leaf_yellowing", "label": "yellow margin",
           "bbox": [0.14, 0.5, 0.74, 0.86], "confidence": 0.9}
    region = _normalise_region(raw, image_size=(640, 640))
    assert region == {"phenotype": "leaf_yellowing", "label": "yellow margin",
                      "bbox_normalized": [0.14, 0.5, 0.74, 0.86], "model_score": 0.9}


def test_a_pixel_box_is_rescaled_against_the_real_image_size():
    """A model that ignores the instruction answers in pixels.

    Rescaling must use the true canvas. Guessing it from the largest coordinate
    collapses the box onto the far edge, which is what the first attempt did.
    """
    raw = {"phenotype": "leaf_spot", "label": "spot",
           "bbox": [380, 440, 435, 495], "confidence": 0.8}
    region = _normalise_region(raw, image_size=(640, 640))
    assert region["bbox_normalized"] == [pytest.approx(0.59375), pytest.approx(0.6875),
                                         pytest.approx(0.6796875), pytest.approx(0.7734375)]


def test_a_pixel_box_beyond_the_canvas_is_refused():
    """Coordinates the image cannot contain are not evidence."""
    raw = {"phenotype": "leaf_spot", "label": "spot",
           "bbox": [420, 625, 490, 695], "confidence": 0.8}
    assert _normalise_region(raw, image_size=(640, 640)) is None


def test_a_pixel_box_without_a_known_size_is_refused():
    """Without the real dimensions there is nothing honest to rescale against."""
    raw = {"phenotype": "leaf_spot", "label": "spot",
           "bbox": [380, 440, 435, 495], "confidence": 0.8}
    assert _normalise_region(raw, image_size=None) is None


@pytest.mark.parametrize("bbox", [
    [0.8, 0.2, 0.1, 0.7],   # inverted horizontally
    [0.2, 0.9, 0.7, 0.3],   # inverted vertically
    [0.0, 0.0, 1.5, 0.5],   # out of range
    [-0.1, 0.0, 0.5, 0.5],  # negative
    [0.1, 0.2, 0.3],        # wrong length
])
def test_a_malformed_box_is_refused(bbox):
    raw = {"phenotype": "leaf_spot", "label": "spot", "bbox": bbox, "confidence": 0.5}
    assert _normalise_region(raw, image_size=(640, 640)) is None


@pytest.mark.parametrize("phenotype", ["blight", "rust", "", None, 3])
def test_an_unknown_phenotype_is_refused(phenotype):
    raw = {"phenotype": phenotype, "label": "x", "bbox": [0.1, 0.1, 0.5, 0.5],
           "confidence": 0.5}
    assert _normalise_region(raw, image_size=(640, 640)) is None


def test_a_missing_label_is_refused():
    raw = {"phenotype": "leaf_spot", "label": "   ", "bbox": [0.1, 0.1, 0.5, 0.5],
           "confidence": 0.5}
    assert _normalise_region(raw, image_size=(640, 640)) is None


def test_confidence_is_clamped_and_defaulted():
    high = _normalise_region({"phenotype": "leaf_spot", "label": "x",
                              "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 9},
                             image_size=(10, 10))
    assert high["model_score"] == 1.0
    missing = _normalise_region({"phenotype": "leaf_spot", "label": "x",
                                 "bbox": [0.1, 0.1, 0.5, 0.5]}, image_size=(10, 10))
    assert missing["model_score"] == 0.0
    broken = _normalise_region({"phenotype": "leaf_spot", "label": "x",
                                "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": "high"},
                               image_size=(10, 10))
    assert broken["model_score"] == 0.0


def test_a_non_object_region_is_refused():
    assert _normalise_region("not a region", image_size=(10, 10)) is None
    assert _normalise_region(None, image_size=(10, 10)) is None


# ── diagnosis filter ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["病原是黄萎病菌", "病因是缺铁", "感染了锈病",
                                  "确诊为白粉病", "caused by fungus", "the pathogen is"])
def test_a_diagnosis_is_detected(text):
    assert _is_diagnosis(text) is True


@pytest.mark.parametrize("text", ["叶缘黄化区域", "small dark spots", "yellow margin"])
def test_an_observation_is_not_a_diagnosis(text):
    assert _is_diagnosis(text) is False


# ── remote script rendering ──────────────────────────────────────────────────


def test_the_remote_script_carries_the_prompt_and_model():
    script = build_remote_script(model=MODEL, image_b64="AAAA", species="黄芪",
                                 prompt="look at this")
    assert MODEL in script
    assert "look at this" in script
    assert "黄芪" in script
    assert "11434" in script  # the node's ollama port


def test_the_remote_script_reports_latency_and_gpu_state():
    script = build_remote_script(model=MODEL, image_b64="AAAA", species="黄芪",
                                 prompt="p")
    assert "latency_ms" in script
    assert "/api/ps" in script
    assert "eval_count" in script
