"""R3-2 data independence: identity, near-duplicates, freeze-then-join.

The blind test's credibility rests on facts established *before* scoring: the
holdout images were strangers to the development process, and the predictions
existed before the labels did. These tests pin both gates with real bytes.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from evaluation.frozen import load_manifest, write_manifest
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


def _solid_image(path, colour, *, textured=True):
    """A test image with real structure.

    The structure must survive the 9x8 reduction the perceptual hash uses, and
    it must be *local*: a smooth gradient reduces to a monotonic ramp, and every
    monotonic ramp hashes identically no matter its steepness — which would
    (correctly) read as a batch-internal duplicate. Random blocks of varying
    brightness give the reduced image edges to describe, the way a photograph
    has.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not textured:
        Image.new("RGB", (160, 120), colour).save(path)
        return
    import random
    from PIL import ImageDraw

    random.seed(str(colour))
    image = Image.new("RGB", (160, 120), colour)
    draw = ImageDraw.Draw(image)
    for _ in range(24):
        x0 = random.randint(0, 140)
        y0 = random.randint(0, 100)
        shade = random.randint(-80, 80)
        box = [x0, y0, x0 + random.randint(8, 40), y0 + random.randint(8, 40)]
        draw.rectangle(box, fill=tuple(max(0, min(255, channel + shade))
                                       for channel in colour))
    image.save(path)


def _predictions(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "leaf", "model_predicted": ["leaf_yellowing"],
         "model_image_usable": True},
        {"image": "new_1.jpg", "mode": "leaf", "model_predicted": [],
         "model_image_usable": False},
    ]}, ensure_ascii=False), encoding="utf-8")
    return path


def _registered(tmp_path, images, *, source="unit-test", batch="batch-1"):
    """A registered-data manifest for the given image names (the formal path
    requires every scored image to have been admitted)."""
    from evaluation.identity import ImageIdentity

    identities = [ImageIdentity(path=image, sha256=f"{index:064x}", bytes=1,
                                width=8, height=8, dhash="0000000000000000",
                                texture=50.0)
                  for index, image in enumerate(images)]
    path = tmp_path / "registered.json"
    write_identity_manifest(identities, path, note="registered test data")
    manifest = load_identity_manifest(path)
    manifest["source"] = source
    manifest["batch"] = batch
    manifest["roles"] = {image: "holdout" for image in images}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
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


def test_predictions_are_sealed_with_their_bytes_and_the_model_identity(tmp_path):
    record = freeze_predictions(_predictions(tmp_path), model="m",
                                model_digest="a" * 64, freeze_id="f" * 64,
                                store=tmp_path / "store")
    assert len(record["predictions_sha256"]) == 64
    assert record["entries"] == 2 and record["model_digest"] == "a" * 64
    assert record["freeze_id"] == "f" * 64
    assert record["sealed_before_labels"] is True


def test_the_join_carries_both_sides_of_the_scoring_input(tmp_path):
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id="f" * 64, store=tmp_path / "store")
    joined = join_for_scoring(predictions, _labels(tmp_path), store=tmp_path / "store")
    assert len(joined["entries"]) == 2
    assert joined["pre_registered"] == ["new_0.jpg", "new_1.jpg"]
    assert joined["freeze_id"] == "f" * 64
    first = joined["entries"][0]
    assert first["model_predicted"] == ["leaf_yellowing"]
    assert first["annotator_labels"] == ["leaf_yellowing"]
    assert len(joined["predictions_sha256"]) == 64
    assert len(joined["labels_sha256"]) == 64


def test_a_label_for_an_image_without_a_frozen_prediction_is_refused(tmp_path):
    """Scoring cannot reach an image the model never saw under the freeze —
    that is the shape of \"predictions produced after the labels\"."""
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id="f" * 64, store=tmp_path / "store")
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
    record = freeze_predictions(predictions, model="m", model_digest="a" * 64,
                                freeze_id="f" * 64, store=tmp_path / "store")
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


# ── R3-2a: the seal is bound to the published identity, and the scored set
#    is the pre-registered set ──────────────────────────────────────────────


def test_a_seal_without_the_weight_digest_is_refused(tmp_path):
    """A model name is a tag; sealing against one lets the server silently
    serve different weights under the same name."""
    with pytest.raises(Exception, match="weight digest"):
        freeze_predictions(_predictions(tmp_path), model="m", model_digest="",
                           freeze_id="f" * 64, store=tmp_path / "store")


def test_a_seal_without_the_published_freeze_id_is_refused(tmp_path):
    with pytest.raises(Exception, match="freeze_id"):
        freeze_predictions(_predictions(tmp_path), model="m",
                           model_digest="a" * 64, freeze_id="",
                           store=tmp_path / "store")


def test_a_predictions_file_carrying_labels_is_refused(tmp_path):
    """The shape of \"the labels were seen first\": the model's output and a
    human's answer do not belong in one file at seal time."""
    contaminated = tmp_path / "contaminated.json"
    contaminated.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "leaf", "model_predicted": ["leaf_yellowing"],
         "annotator_labels": ["leaf_yellowing"]},
    ]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception, match="annotator fields"):
        freeze_predictions(contaminated, model="m", model_digest="a" * 64,
                           freeze_id="f" * 64, store=tmp_path / "store")


def test_the_first_seal_wins(tmp_path):
    """Re-sealing the same bytes cannot move the timestamp: the earlier seal is
    the one that counts, so a later act cannot backdate or refresh an order."""
    predictions = _predictions(tmp_path)
    first = freeze_predictions(predictions, model="m", model_digest="a" * 64,
                               freeze_id="f" * 64, store=tmp_path / "store")
    second = freeze_predictions(predictions, model="m", model_digest="a" * 64,
                                freeze_id="f" * 64, store=tmp_path / "store")
    assert second["frozen_at"] == first["frozen_at"]
    assert second["predictions_sha256"] == first["predictions_sha256"]


def test_scoring_under_a_different_model_digest_is_refused(tmp_path):
    """The reviewer's path: the freeze publishes model A, the seal names
    model B — the join refuses instead of recording A's experiment over B's
    predictions."""
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", model_digest="b" * 64,
                       freeze_id="f" * 64, store=tmp_path / "store")
    with pytest.raises(Exception, match="do not match the scoring freeze"):
        join_for_scoring(predictions, _labels(tmp_path), store=tmp_path / "store",
                         expected={"freeze_id": "f" * 64, "model": "m",
                                   "model_digest": "a" * 64})


def test_scoring_under_a_different_freeze_id_is_refused(tmp_path):
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id="f" * 64, store=tmp_path / "store")
    with pytest.raises(Exception, match="do not match the scoring freeze"):
        join_for_scoring(predictions, _labels(tmp_path), store=tmp_path / "store",
                         expected={"freeze_id": "e" * 64, "model": "m",
                                   "model_digest": "a" * 64})


def test_a_mode_mismatch_is_a_data_contract_refusal_not_a_keyerror(tmp_path):
    """The same image in a different mode is a different record; a label whose
    mode disagrees with the frozen prediction is refused with a clear message,
    not a bare KeyError."""
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id="f" * 64, store=tmp_path / "store")
    wrong_mode = tmp_path / "wrong-mode.json"
    wrong_mode.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "herb", "annotator_labels": ["leaf_yellowing"]},
        {"image": "new_1.jpg", "mode": "leaf", "annotator_labels": []},
    ]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception) as refusal:
        join_for_scoring(predictions, wrong_mode, store=tmp_path / "store")
    # A mode disagreement leaves a frozen key unlabelled (or a label
    # unpredicted); either way the refusal names the exact key.
    assert "new_0.jpg" in str(refusal.value)
    assert "frozen predictions without labels" in str(refusal.value) or \
           "labels without frozen predictions" in str(refusal.value)


def test_a_repeated_key_is_refused_on_either_side(tmp_path):
    """A repeated (image, mode) key is a data error, not something the later
    entry should silently win."""
    predictions = _predictions(tmp_path)
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id="f" * 64, store=tmp_path / "store")
    duplicated = tmp_path / "duplicated.json"
    duplicated.write_text(json.dumps({"entries": [
        {"image": "new_0.jpg", "mode": "leaf", "annotator_labels": ["leaf_yellowing"]},
        {"image": "new_0.jpg", "mode": "leaf", "annotator_labels": ["wilting"]},
        {"image": "new_1.jpg", "mode": "leaf", "annotator_labels": []},
    ]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception, match="repeats the .* key"):
        join_for_scoring(predictions, duplicated, store=tmp_path / "store")


def test_scoring_a_subset_of_the_frozen_set_is_refused(tmp_path):
    """Freezing 100 images and scoring the 20 that looked best is a different
    experiment with a smaller denominator nobody declared."""
    from evaluation.__main__ import main

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model="m", quantization="q4",
                   model_digest="a" * 64, at="2026-10-11T00:00:00+00:00")
    predictions = tmp_path / "predictions.json"
    images = [f"img_{index:03}.jpg" for index in range(5)]
    predictions.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "model_predicted": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id=load_manifest(manifest)["freeze_id"],
                       store=tmp_path / "store")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "annotator_labels": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")

    subset = images[:2]
    argv = ["holdout", str(predictions), "--model", "m", "--quantization", "q4",
            "--ledger", str(tmp_path / "ledger.json"),
            "--output", str(tmp_path / "out.json"),
            "--manifest", str(manifest),
            "--predictions", str(predictions), "--labels", str(labels),
            "--prediction-store", str(tmp_path / "store"),
            "--identity", str(_registered(tmp_path, images))]
    argv += [token for image in subset for token in ("--holdout", image)]
    assert main(argv) == 2  # the scored set must equal the pre-registered set

    # The full frozen set scores, and the payload records the denominator.
    argv_full = argv[:argv.index("--holdout")] + [
        token for image in images for token in ("--holdout", image)]
    argv_full[argv_full.index("--ledger") + 1] = str(tmp_path / "ledger2.json")
    argv_full[argv_full.index("--output") + 1] = str(tmp_path / "out2.json")
    assert main(argv_full) == 0
    payload = json.loads((tmp_path / "out2.json").read_text(encoding="utf-8"))
    assert payload["scored_entries"] == 5
    assert payload["pre_registered"] == images


def test_an_exclusion_is_recorded_with_its_reason_and_the_denominator(tmp_path):
    """A damaged image may shrink what is scored — with the reason and the full
    denominator travelling with the result."""
    from evaluation.__main__ import main

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model="m", quantization="q4",
                   model_digest="a" * 64, at="2026-10-11T00:00:00+00:00")
    images = [f"img_{index:03}.jpg" for index in range(3)]
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "model_predicted": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id=load_manifest(manifest)["freeze_id"],
                       store=tmp_path / "store")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "annotator_labels": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    argv = ["holdout", str(predictions), "--model", "m", "--quantization", "q4",
            "--ledger", str(tmp_path / "ledger.json"),
            "--output", str(tmp_path / "out.json"),
            "--manifest", str(manifest),
            "--predictions", str(predictions), "--labels", str(labels),
            "--prediction-store", str(tmp_path / "store"),
            "--identity", str(_registered(tmp_path, images)),
            "--holdout", images[0], "--holdout", images[1], "--holdout", images[2],
            "--exclude", f"{images[2]}=采集时镜头污损，标注缺失"]
    assert main(argv) == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["scored_entries"] == 2
    assert payload["excluded"] == [{"image": images[2],
                                    "reason": "采集时镜头污损，标注缺失"}]
    assert payload["pre_registered"] == images


def test_flat_frames_go_to_manual_review_not_an_automatic_refusal(development_manifest,
                                                                 tmp_path):
    """On a blank frame the difference hash is noise; refusing real data on
    noise is its own kind of error, so those hits are a human's call."""
    for index in range(2):
        _solid_image(tmp_path / f"flat_{index}.jpg", (index * 30, 120, 90),
                     textured=False)
    report = cross_check(scan_directory(tmp_path), development_manifest)
    assert report["manual_review"], "flat frames should surface as manual review"
    assert not report["near_duplicates"]
    with pytest.raises(Exception, match="needs a human look"):
        assert_independent(scan_directory(tmp_path), development_manifest)
    # An operator who has looked accepts them explicitly.
    accepted = assert_independent(scan_directory(tmp_path), development_manifest,
                                  allow_manual_review=True)
    assert accepted["manual_review"]


def test_a_re_encoded_copy_inside_the_batch_is_refused(development_manifest, tmp_path):
    """A hundred "new" images that are re-encodes of two photographs would pass
    a reference-only check while carrying almost no independent information."""
    _solid_image(tmp_path / "a.jpg", (10, 20, 30))
    Image.open(tmp_path / "a.jpg").save(tmp_path / "a_copy.jpg", quality=60)
    with pytest.raises(Exception, match="within the batch"):
        assert_independent(scan_directory(tmp_path), development_manifest)


def test_the_same_bytes_twice_in_one_batch_is_refused(development_manifest, tmp_path):
    _solid_image(tmp_path / "a.jpg", (10, 20, 30))
    (tmp_path / "b.jpg").write_bytes((tmp_path / "a.jpg").read_bytes())
    with pytest.raises(Exception, match="within the batch"):
        assert_independent(scan_directory(tmp_path), development_manifest)


def test_registration_requires_the_collectors_record(tmp_path, development_manifest):
    """The perceptual gate catches the same photograph re-encoded; only the
    collector's record can say whether two images share a plant."""
    from evaluation.__main__ import main

    _solid_image(tmp_path / "fresh" / "a.jpg", (10, 20, 30))
    argv = ["identity", "--new", str(tmp_path / "fresh")]
    assert main(argv) == 2
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        main(argv + ["--source", "s"])
    assert "--batch" in buffer.getvalue()


def test_roles_are_validated_against_the_defined_roles(tmp_path, development_manifest):
    from evaluation.__main__ import main

    _solid_image(tmp_path / "fresh" / "a.jpg", (10, 20, 30))
    argv = ["identity", "--new", str(tmp_path / "fresh"),
            "--source", "s", "--batch", "b", "--role", "training"]
    assert main(argv) == 2


def test_the_formal_path_requires_the_registered_manifest(tmp_path):
    """Admission and scoring are separate paths; the formal blind test refuses
    to score data nobody registered."""
    from evaluation.__main__ import main

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model="m", quantization="q4",
                   model_digest="a" * 64, at="2026-10-11T00:00:00+00:00")
    images = ["img_000.jpg", "img_001.jpg"]
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "model_predicted": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id=load_manifest(manifest)["freeze_id"],
                       store=tmp_path / "store")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "annotator_labels": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    argv = ["holdout", str(predictions), "--model", "m", "--quantization", "q4",
            "--ledger", str(tmp_path / "ledger.json"),
            "--output", str(tmp_path / "out.json"),
            "--manifest", str(manifest),
            "--predictions", str(predictions), "--labels", str(labels),
            "--prediction-store", str(tmp_path / "store")]
    argv += [token for image in images for token in ("--holdout", image)]
    assert main(argv) == 2  # no --identity: refuse


def test_an_unregistered_scored_image_is_refused(tmp_path):
    from evaluation.__main__ import main

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model="m", quantization="q4",
                   model_digest="a" * 64, at="2026-10-11T00:00:00+00:00")
    images = ["img_000.jpg", "img_001.jpg"]
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "model_predicted": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    freeze_predictions(predictions, model="m", model_digest="a" * 64,
                       freeze_id=load_manifest(manifest)["freeze_id"],
                       store=tmp_path / "store")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": [
        {"image": image, "mode": "leaf", "annotator_labels": ["leaf_yellowing"]}
        for image in images]}, ensure_ascii=False), encoding="utf-8")
    # The manifest registers only the first image.
    registered = _registered(tmp_path, images[:1])
    argv = ["holdout", str(predictions), "--model", "m", "--quantization", "q4",
            "--ledger", str(tmp_path / "ledger.json"),
            "--output", str(tmp_path / "out.json"),
            "--manifest", str(manifest),
            "--predictions", str(predictions), "--labels", str(labels),
            "--prediction-store", str(tmp_path / "store"),
            "--identity", str(registered)]
    argv += [token for image in images for token in ("--holdout", image)]
    assert main(argv) == 2
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        main(argv)
    assert "img_001.jpg" in buffer.getvalue()
