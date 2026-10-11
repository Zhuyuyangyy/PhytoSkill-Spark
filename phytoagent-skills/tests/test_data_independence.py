"""R3-2 data independence: identity, near-duplicates, freeze-then-join.

The blind test's credibility rests on facts established *before* scoring: the
holdout images were strangers to the development process, and the predictions
existed before the labels did. These tests pin both gates with real bytes.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from evaluation.identity import (NEAR_DUPLICATE_MAX_DISTANCE, assert_independent,
                                 cross_check, image_identity, load_identity_manifest,
                                 scan_directory, write_identity_manifest)
from evaluation.predictions import freeze_predictions, join_for_scoring

PROJECT = None  # resolved in the fixture; keeps the imports above module-level


@pytest.fixture(scope="module")
def project_root():
    from demo.fixture_workspace import PROJECT_ROOT
    return PROJECT_ROOT


@pytest.fixture(scope="module")
def development_manifest(project_root, tmp_path_factory):
    """The real development set, identified — the baseline everything checks against."""
    directory = tmp_path_factory.mktemp("dev")
    manifest = write_identity_manifest(
        scan_directory(project_root / "case_workspace" / "annotate"),
        directory / "development-identity.json",
        note="the real Dev-30 images")
    return load_identity_manifest(directory / "development-identity.json")


def _solid_image(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), colour).save(path)


# ── identity ────────────────────────────────────────────────────────────────


def test_the_development_set_is_identified(development_manifest):
    assert development_manifest["count"] == 30
    for image in development_manifest["images"]:
        assert len(image["sha256"]) == 64
        assert len(image["dhash"]) == 16
        assert image["width"] and image["height"]


def test_an_exact_duplicate_is_refused(development_manifest, project_root, tmp_path):
    """A byte-identical copy of a development image cannot enter a blind set."""
    duplicate = tmp_path / "duplicate.jpg"
    duplicate.write_bytes((project_root / "case_workspace" / "annotate"
                           / "huangqi_01.jpg").read_bytes())
    report = cross_check(scan_directory(tmp_path), development_manifest)
    assert report["exact_duplicates"]
    assert not report["clean"]
    with pytest.raises(Exception, match="overlap the development set"):
        assert_independent(scan_directory(tmp_path), development_manifest)


def test_a_re_encoded_copy_is_reported_as_a_near_duplicate(development_manifest,
                                                           project_root, tmp_path):
    """The same photograph under new bytes: the perceptual hash catches what
    SHA-256 cannot, because a re-encode is the silent way a blind set leaks."""
    reencoded = tmp_path / "reencoded.jpg"
    Image.open(project_root / "case_workspace" / "annotate" / "huangqi_01.jpg").save(
        reencoded, quality=72)
    assert reencoded.read_bytes() != (project_root / "case_workspace" / "annotate"
                                      / "huangqi_01.jpg").read_bytes()
    report = cross_check(scan_directory(tmp_path), development_manifest)
    assert report["near_duplicates"]
    assert report["near_duplicates"][0]["distance"] <= NEAR_DUPLICATE_MAX_DISTANCE
    assert not report["clean"]


def test_genuinely_new_images_register(development_manifest, tmp_path):
    for index in range(3):
        _solid_image(tmp_path / f"new_{index}.jpg", (index * 40, 90, 200 - index * 30))
    report = assert_independent(scan_directory(tmp_path), development_manifest)
    assert report["clean"] and report["checked"] == 3


def test_the_workspace_already_contains_the_leak_this_gates(development_manifest,
                                                            project_root, tmp_path):
    """The gate on real data: case_workspace/leaves holds byte-identical copies
    of development images, so registering it must refuse — the protection is
    not theoretical."""
    with pytest.raises(Exception, match="overlap the development set"):
        assert_independent(scan_directory(project_root / "case_workspace" / "leaves"),
                           development_manifest)


# ── freeze-then-join ────────────────────────────────────────────────────────


def _predictions(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "leaf", "model_predicted": ["leaf_yellowing"],
         "model_image_usable": True},
        {"image": "new_1.jpg", "mode": "leaf", "model_predicted": [],
         "model_image_usable": False},
    ]}, ensure_ascii=False), encoding="utf-8")
    return path


def _labels(tmp_path, name="labels.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "leaf", "annotator_labels": ["leaf_yellowing"],
         "candidate_labels": ["leaf_yellowing", "leaf_spot"]},
        {"image": "new_1.jpg", "mode": "leaf", "annotator_labels": [],
         "candidate_labels": ["leaf_yellowing", "leaf_spot"]},
    ]}, ensure_ascii=False), encoding="utf-8")
    return path


def test_predictions_are_sealed_with_their_bytes_and_the_model_identity(tmp_path):
    record = freeze_predictions(_predictions(tmp_path), model="m",
                                model_digest="a" * 64,
                                store=tmp_path / "store")
    assert len(record["predictions_sha256"]) == 64
    assert record["entries"] == 2 and record["model_digest"] == "a" * 64


def test_the_join_carries_both_sides_of_the_scoring_input(tmp_path):
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", store=tmp_path / "store")
    joined = join_for_scoring(predictions, _labels(tmp_path), store=tmp_path / "store")
    assert len(joined["entries"]) == 2
    first = joined["entries"][0]
    assert first["model_predicted"] == ["leaf_yellowing"]
    assert first["annotator_labels"] == ["leaf_yellowing"]
    assert len(joined["predictions_sha256"]) == 64
    assert len(joined["labels_sha256"]) == 64


def test_a_label_for_an_image_without_a_frozen_prediction_is_refused(tmp_path):
    """Scoring cannot reach an image the model never saw under the freeze —
    that is the shape of \"predictions produced after the labels\"."""
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", store=tmp_path / "store")
    late = tmp_path / "late.json"
    late.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "leaf", "annotator_labels": ["leaf_yellowing"]},
        {"image": "new_1.jpg", "mode": "leaf", "annotator_labels": []},
        {"image": "unseen.jpg", "mode": "leaf", "annotator_labels": ["leaf_spot"]},
    ]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception, match="without frozen predictions"):
        join_for_scoring(predictions, late, store=tmp_path / "store")


def test_predictions_that_no_longer_match_their_frozen_record_are_refused(tmp_path):
    """The frozen record is checked against the file's current bytes: editing a
    prediction after the freeze is caught by the record, not by trust."""
    from evaluation.predictions import load_frozen_predictions, verify_frozen_predictions

    predictions = _predictions(tmp_path)
    record = freeze_predictions(predictions, model="m", store=tmp_path / "store")
    tampered = tmp_path / "tampered.json"
    tampered.write_text(predictions.read_text(encoding="utf-8").replace(
        "leaf_yellowing", "wilting"), encoding="utf-8")
    with pytest.raises(Exception, match="no longer match"):
        verify_frozen_predictions(tampered, record)
    # And a file whose bytes nobody ever sealed has no record to match.
    with pytest.raises(Exception, match="no frozen prediction record"):
        load_frozen_predictions("0" * 64, store=tmp_path / "store")


def test_unfrozen_predictions_are_refused(tmp_path):
    """The order is the guarantee: freeze first, labels second. Scoring a
    predictions file that was never sealed cannot show which came first."""
    with pytest.raises(Exception, match="freeze the predictions"):
        join_for_scoring(_predictions(tmp_path), _labels(tmp_path),
                         store=tmp_path / "empty-store")
