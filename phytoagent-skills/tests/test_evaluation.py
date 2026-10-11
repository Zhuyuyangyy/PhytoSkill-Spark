"""Evaluation layer: selective metrics, the freeze, and the one-shot holdout rule.

Offline: no model, no image, no network. The arithmetic is checked against
hand-computed values, and one test pins the frozen Dev-30 baseline so a
regression in the metric code cannot pass unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.agreement import (ACCEPT_BOTH, ACCEPT_REFERENCE, adjudicate,
                                  cohen_kappa, multilabel_agreement, resolution_queue)
from evaluation.metrics import (Evaluation, LabelCounts, evaluate, normalise_labels,
                                risk_coverage_curve, stratify)
from evaluation.split import (DatasetSplit, FreezeError, HoldoutAlreadyRun,
                              HoldoutLedger, freeze, split_entries)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = PROJECT_ROOT / "case_workspace" / "annotate" / "huangqi_annotated_results.json"
BASELINE = PROJECT_ROOT / "artifacts" / "dgx" / "baseline-v1.json"

# One image per interesting case: a hit + a miss, an abstention, a spurious
# answer, and an image where both sides are empty.
SAMPLE = [
    {"mode": "herb", "image": "i1",
     "model_predicted": ["a", "b"], "annotator_labels": ["a", "c"]},
    {"mode": "herb", "image": "i2",
     "model_predicted": [], "annotator_labels": ["d"]},
    {"mode": "herb", "image": "i3",
     "model_predicted": ["e"], "annotator_labels": []},
    {"mode": "live", "image": "i4",
     "model_predicted": [], "annotator_labels": []},
]


# ── counts and the headline ratios ───────────────────────────────────────────


def test_micro_counts_match_hand_computation():
    evaluation = evaluate(SAMPLE)
    assert evaluation.images == 4
    assert evaluation.annotated == 4
    assert (evaluation.micro.tp, evaluation.micro.fp, evaluation.micro.fn) == (1, 2, 2)
    assert evaluation.precision == 0.3333
    assert evaluation.recall == 0.3333
    assert evaluation.f1 == 0.3333


def test_answered_and_abstained_are_different_from_merely_empty():
    """An image where both sides are empty is correctly silent — not an abstention.

    i3 predicted something (answered), i2 withheld where the reference had a
    label (abstained), i4 was empty on both sides (neither).
    """
    evaluation = evaluate(SAMPLE)
    assert evaluation.answered == 2
    assert evaluation.abstained == 1
    assert evaluation.coverage == 0.5
    assert evaluation.abstention_rate == 0.25


def test_abstention_costs_recall_but_can_never_cost_precision():
    """The structural fact this whole module exists to expose."""
    evaluation = evaluate(SAMPLE)
    # An empty prediction cannot produce a false positive, so the two coincide.
    assert evaluation.selective_precision == evaluation.precision
    # Selective recall excludes the labels lost to abstention, so it is the
    # optimistic view: 1/2 on the answered images versus 1/3 overall.
    assert evaluation.selective_recall == 0.5
    assert evaluation.recall == 0.3333
    assert evaluation.selective_recall > evaluation.recall
    assert evaluation.risk_at_coverage == 0.6667


def test_f1_comes_from_raw_counts_not_from_rounded_ratios():
    """The Dev-30 counts: 22/79 = 0.2785, not the 0.2784 a rounded P/R gives."""
    counts = LabelCounts(tp=11, fp=9, fn=48)
    assert counts.precision == 0.55
    assert counts.recall == 0.1864
    assert counts.f1 == 0.2785


def test_a_zero_denominator_is_none_not_zero():
    """"No opportunity to be right" is a different claim from "wrong every time"."""
    counts = LabelCounts(tp=0, fp=0, fn=3)
    assert counts.precision is None      # nothing was ever predicted
    assert counts.recall == 0.0          # three chances, all missed
    assert evaluate([{"annotator_labels": []}]).precision is None


def test_entries_without_a_reference_are_skipped():
    """An unlabelled image is not evidence, and must not be counted as a miss."""
    entries = [*SAMPLE, {"mode": "herb", "image": "i5", "model_predicted": ["x"]}]
    evaluation = evaluate(entries)
    assert evaluation.images == 5
    assert evaluation.annotated == 4


def test_normalise_labels_ignores_casing_and_repeats():
    assert normalise_labels([" A ", "a", "B", "", None, 7]) == ["a", "b"]


# ── macro / per-label ────────────────────────────────────────────────────────


def test_a_label_with_support_but_no_prediction_scores_zero_in_macro():
    """Skipping it would let a model raise its macro score by ignoring a label."""
    # 'z' occurs on two images and is never predicted.
    entries = [{"model_predicted": ["a"], "annotator_labels": ["a", "z"]},
               {"model_predicted": ["a"], "annotator_labels": ["a", "z"]}]
    evaluation = evaluate(entries)
    assert evaluation.per_label["z"].support == 2
    assert evaluation.per_label["z"].precision is None
    # macro averages 'a' (F1 1.0) and 'z' (counted as 0.0), not just 'a'.
    assert evaluation.macro_f1 == 0.5


def test_a_label_the_reference_never_uses_is_excluded_and_listed():
    evaluation = evaluate(SAMPLE)
    assert evaluation.labels_without_support == ["b", "e"]
    assert "b" not in {name for name, counts in evaluation.per_label.items()
                       if counts.support > 0}


def test_stratify_reports_each_mode_and_the_overall():
    by_mode = stratify(SAMPLE, key="mode")
    assert set(by_mode) == {"herb", "live", "overall"}
    assert by_mode["herb"].annotated == 3
    assert by_mode["live"].annotated == 1
    assert by_mode["overall"].annotated == 4


# ── risk / coverage curve ────────────────────────────────────────────────────


def test_risk_coverage_curve_needs_a_confidence_and_does_not_invent_one():
    assert risk_coverage_curve(SAMPLE) is None
    ranked = [dict(entry, model_confidence=confidence)
              for entry, confidence in zip(SAMPLE, (0.9, 0.8, 0.5, 0.1))]
    curve = risk_coverage_curve(ranked)
    assert [point["coverage"] for point in curve] == [0.25, 0.5, 0.75, 1.0]
    # The top-confidence image carries one hit and one spurious label.
    assert curve[0]["risk"] == 0.5


# ── the frozen baseline must not drift ───────────────────────────────────────


def test_the_metrics_reproduce_the_frozen_dev30_baseline():
    """A regression in the metric code would silently disagree with a frozen number."""
    if not ANNOTATION.is_file() or not BASELINE.is_file():
        pytest.skip("annotation or baseline artifact not present")
    entries = json.loads(ANNOTATION.read_text(encoding="utf-8"))["entries"]
    frozen = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    evaluation = evaluate(entries)
    assert evaluation.images == frozen["images"]
    assert evaluation.annotated == frozen["annotated"]
    assert (evaluation.micro.tp, evaluation.micro.fp, evaluation.micro.fn) == (
        frozen["labels"]["true_positives"], frozen["labels"]["false_positives"],
        frozen["labels"]["false_negatives"])
    assert evaluation.precision == frozen["precision"]
    assert evaluation.recall == frozen["recall"]
    assert evaluation.f1 == frozen["f1"]


# ── the freeze ───────────────────────────────────────────────────────────────


def test_the_freeze_is_deterministic_for_the_same_code_and_model():
    first = freeze(model="m", quantization="q4")
    second = freeze(model="m", quantization="q4")
    assert first.freeze_id == second.freeze_id
    assert len(first.freeze_id) == 64


def test_a_different_model_or_quantisation_is_a_different_experiment():
    base = freeze(model="m", quantization="q4")
    assert freeze(model="other", quantization="q4").freeze_id != base.freeze_id
    assert freeze(model="m", quantization="q8").freeze_id != base.freeze_id


def test_a_freeze_must_name_a_model_and_a_quantisation():
    with pytest.raises(FreezeError):
        freeze(model="", quantization="q4")


def test_the_freeze_covers_the_semantics_the_scorer_and_the_prompt():
    frozen = freeze(model="m", quantization="q4")
    # Four independent fingerprints: editing any one of them changes the freeze.
    assert len({frozen.prompt_hash, frozen.parser_hash, frozen.ontology_hash,
                frozen.scorer_hash}) == 4


# ── dataset roles ────────────────────────────────────────────────────────────


def test_an_image_cannot_be_in_two_roles():
    with pytest.raises(FreezeError, match="both"):
        DatasetSplit(holdout=("a",), calibration=("a",))


def test_split_entries_rejects_a_holdout_image_that_is_not_in_the_dataset():
    with pytest.raises(FreezeError, match="not in the dataset"):
        split_entries(SAMPLE, holdout=("nope",))


def test_split_entries_defaults_everything_else_to_calibration():
    split = split_entries(SAMPLE, holdout=("i4",), development=("i3",))
    assert split.holdout == ("i4",)
    assert split.development == ("i3",)
    assert split.calibration == ("i1", "i2")
    assert split.role_of("i4") == "holdout"


# ── the one-shot holdout ledger ──────────────────────────────────────────────


def test_a_holdout_freeze_cannot_be_run_twice(tmp_path):
    """The rule the whole protocol rests on."""
    ledger = HoldoutLedger.load(tmp_path / "ledger.json")
    frozen = freeze(model="m", quantization="q4")
    ledger.record(frozen, metrics={"f1": 0.1})
    with pytest.raises(HoldoutAlreadyRun, match="already evaluated"):
        ledger.guard(frozen)


def test_superseding_a_holdout_records_what_it_replaced(tmp_path):
    ledger = HoldoutLedger.load(tmp_path / "ledger.json")
    frozen = freeze(model="m", quantization="q4")
    first = ledger.record(frozen, metrics={"f1": 0.1})
    second = ledger.record(frozen, metrics={"f1": 0.2}, supersede=True,
                           note="prompt was edited; the first run is void")
    assert second["supersedes"] == [first["at"]]
    assert len(ledger.runs_for(frozen.freeze_id)) == 2


def test_the_ledger_survives_a_round_trip(tmp_path):
    path = tmp_path / "ledger.json"
    frozen = freeze(model="m", quantization="q4")
    HoldoutLedger.load(path).record(frozen, metrics={"f1": 0.1})
    reloaded = HoldoutLedger.load(path)
    assert reloaded.has_run(frozen.freeze_id)
    with pytest.raises(HoldoutAlreadyRun):
        reloaded.guard(frozen)


def test_a_development_run_is_not_subject_to_the_one_shot_rule(tmp_path):
    ledger = HoldoutLedger.load(tmp_path / "ledger.json")
    frozen = freeze(model="m", quantization="q4")
    ledger.record(frozen, metrics={}, role="development")
    ledger.record(frozen, metrics={}, role="development")


# ── inter-annotator agreement ────────────────────────────────────────────────


def test_kappa_is_one_for_perfect_agreement_with_variance():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_kappa_is_none_when_there_is_no_variance_to_explain():
    """Both annotators always saying "absent" agree on nothing worth measuring."""
    assert cohen_kappa([False, False, False], [False, False, False]) is None
    assert cohen_kappa([True, True], [True, True]) is None


def test_kappa_is_negative_when_the_raters_are_systematically_opposed():
    assert cohen_kappa([True, True, False, False], [False, False, True, True]) == -1.0


def test_kappa_matches_a_hand_computed_example():
    first = [True, True, True, True, False, False, False, False, False, False]
    second = [True, True, True, False, False, False, False, False, False, False]
    # po=0.9, pe=0.54, kappa=0.36/0.46
    assert cohen_kappa(first, second) == 0.7826


def test_multilabel_agreement_compares_only_shared_images():
    first = {"i1": ["a"], "i2": ["b"], "i3": ["a"]}
    second = {"i1": ["a"], "i2": ["c"], "i9": ["z"]}
    report = multilabel_agreement(first, second)
    assert report["common_images"] == 2
    assert report["only_in_first"] == ["i3"]
    assert report["only_in_second"] == ["i9"]
    assert report["per_label"]["a"]["kappa"] == 1.0


def test_multilabel_agreement_says_so_when_there_is_no_overlap():
    report = multilabel_agreement({"i1": ["a"]}, {"i2": ["a"]})
    assert report["status"] == "no_overlap"


def test_the_resolution_queue_lists_only_the_disagreements():
    first = {"i1": ["a"], "i2": ["a", "b"]}
    second = {"i1": ["a"], "i2": ["a", "c"]}
    queue = resolution_queue(first, second)
    assert [item.image for item in queue] == ["i2"]
    assert queue[0].only_in_first == ["b"]
    assert queue[0].only_in_second == ["c"]


def test_adjudication_refuses_to_guess_an_undecided_disagreement():
    first = {"i1": ["a", "b"]}
    second = {"i1": ["a", "c"]}
    with pytest.raises(ValueError, match="without a recorded decision"):
        adjudicate(first, second, {})


def test_adjudication_applies_the_recorded_decisions():
    first = {"i1": ["a", "b"], "i2": ["x"]}
    second = {"i1": ["a", "c"], "i2": ["x"]}
    assert adjudicate(first, second, {"i1": ACCEPT_BOTH})["i1"] == ["a", "b", "c"]
    assert adjudicate(first, second, {"i1": ACCEPT_REFERENCE})["i1"] == ["a", "b"]


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_cli_report_writes_a_report_with_the_selective_metrics(tmp_path):
    from evaluation.__main__ import main

    output = tmp_path / "report.json"
    assert main(["report", str(ANNOTATION), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["modes"]["overall"]["coverage"] is not None
    assert report["modes"]["overall"]["abstention_rate"] is not None
    assert report["risk_coverage"]["available"] is False


def test_cli_holdout_refuses_a_second_run_of_the_same_freeze(tmp_path):
    from evaluation.__main__ import main
    from evaluation.frozen import write_manifest

    ledger, output = tmp_path / "ledger.json", tmp_path / "holdout.json"
    # The holdout command asserts the tree against the manifest it is given, so
    # this test freezes its own ("m", "q4") configuration rather than borrowing
    # the committed one — its subject is the ledger rule, not the freeze.
    manifest = tmp_path / "frozen.json"
    # A freeze pins the weights' identity; a test model gets a test digest
    # explicitly — the command would otherwise refuse to freeze at all.
    write_manifest(manifest, model="m", quantization="q4",
                   model_digest="a" * 64)
    argv = ["holdout", str(ANNOTATION), "--model", "m", "--quantization", "q4",
            "--ledger", str(ledger), "--output", str(output),
            "--manifest", str(manifest),
            "--holdout", "huangqi_01.jpg", "--holdout", "huangqi_02.jpg"]
    assert main(argv) == 0
    # Same freeze, same holdout: the protocol forbids looking twice.
    assert main(argv) == 2


# ── the frozen manifest: recorded outside the run, enforced by drift checks ──


MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"
QUANTIZATION = "Q4_K_M"


def test_the_committed_manifest_verifies_against_the_current_tree():
    from evaluation.frozen import DEFAULT_MANIFEST, verify_manifest

    report = verify_manifest(DEFAULT_MANIFEST)
    assert report.ok, report.to_dict()
    assert report.freeze_matches
    assert report.drifted == () and report.missing == () and report.unexpected == ()


def test_the_manifest_round_trips_through_the_file(tmp_path):
    from evaluation.frozen import build_manifest, load_manifest, write_manifest
    from evaluation.split import freeze as make_freeze

    frozen = make_freeze(model=MODEL, quantization=QUANTIZATION,
                         model_digest="b" * 64)
    manifest = build_manifest(frozen, at="2026-10-10T00:00:00+00:00")
    path = tmp_path / "frozen.json"
    write_manifest(path, model=MODEL, quantization=QUANTIZATION,
                   at="2026-10-10T00:00:00+00:00",
                   model_digest="b" * 64)
    loaded = load_manifest(path)
    assert loaded["freeze_id"] == manifest["freeze_id"] == frozen.freeze_id
    assert [item["path"] for item in loaded["inputs"]] == [
        item["path"] for item in manifest["inputs"]]


def test_a_drifted_input_is_named_by_the_verification(tmp_path):
    """The whole point: after the blind test, a prompt or threshold edit cannot
    be passed off as the same experiment — the check names the file that moved."""
    from evaluation.frozen import DEFAULT_MANIFEST, load_manifest, verify_manifest

    manifest = load_manifest(DEFAULT_MANIFEST)
    victim = manifest["inputs"][0]["path"]
    manifest["inputs"][0]["sha256"] = "0" * 64
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = verify_manifest(path)
    assert not report.ok
    assert report.drifted == (victim,)
    assert report.freeze_matches  # the aggregates still match; only the file moved


def test_a_changed_aggregate_fingerprint_fails_the_freeze(tmp_path):
    from evaluation.frozen import DEFAULT_MANIFEST, load_manifest, verify_manifest

    manifest = load_manifest(DEFAULT_MANIFEST)
    manifest["freeze"]["parser_hash"] = "1" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = verify_manifest(path)
    assert not report.ok
    assert not report.freeze_matches


def test_an_input_missing_from_the_manifest_is_reported(tmp_path):
    """A frozen input that the manifest forgot to pin is not frozen at all."""
    from evaluation.frozen import DEFAULT_MANIFEST, load_manifest, verify_manifest

    manifest = load_manifest(DEFAULT_MANIFEST)
    forgotten = manifest["inputs"].pop()["path"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = verify_manifest(path)
    assert not report.ok
    assert forgotten in report.unexpected


def test_the_aggregates_and_the_file_inventory_describe_the_same_bytes():
    """The per-file pins and the aggregate fingerprints must not drift apart."""
    from evaluation.frozen import aggregates_match_inputs

    matches = aggregates_match_inputs()
    assert matches, "no file-backed aggregate found"
    assert all(matches.values()), matches


def test_the_manifest_covers_the_protocol_and_every_contract():
    """The reviewer's R3-1 list: model, prompt, schema, thresholds, protocol."""
    from evaluation.frozen import FREEZE_INPUTS

    paths = {relative for relative, _ in FREEZE_INPUTS}
    assert "docs/evaluation.md" in paths, "the protocol itself must be frozen"
    for skill in ("plant_vision", "growth_risk", "herbal_knowledge",
                  "evidence_fusion", "agentshield_audit"):
        assert f"skills/{skill}/schema.json" in paths, f"{skill} contract not frozen"
    # The decision thresholds: the parser, the auditor's trust rules, the
    # retrieval gate and the ranking constants.
    for threshold_source in ("vision/parser.py", "runtime/shield.py",
                             "corpus/retriever.py", "corpus/ranking.py"):
        assert threshold_source in paths, f"{threshold_source} not frozen"


def test_assert_manifest_raises_on_drift(tmp_path):
    from evaluation.frozen import DEFAULT_MANIFEST, assert_manifest, load_manifest
    from evaluation.split import FreezeError

    manifest = load_manifest(DEFAULT_MANIFEST)
    manifest["freeze"]["scorer_hash"] = "2" * 64
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(FreezeError, match="fingerprints"):
        assert_manifest(path)


# ── the holdout gate is mandatory, and the experiment has an identity ───────


def test_a_holdout_run_without_a_manifest_is_refused(tmp_path):
    """The gate must not be optional: a missing manifest used to skip the
    freeze check entirely, so a drifted tree could still be scored."""
    from evaluation.__main__ import main

    argv = ["holdout", str(ANNOTATION), "--model", MODEL, "--quantization", QUANTIZATION,
            "--ledger", str(tmp_path / "ledger.json"),
            "--output", str(tmp_path / "out.json"),
            "--manifest", str(tmp_path / "does-not-exist.json"),
            "--holdout", "huangqi_01.jpg"]
    # The CLI reports a refusal as a non-zero exit with the reason on stderr;
    # main() never tracebacks at a caller.
    assert main(argv) == 2


def test_a_holdout_run_without_a_declared_holdout_is_refused(tmp_path):
    """An undeclared holdout scores whatever it is given — including data the
    prompts were developed against. That is the bypass this closes."""
    from evaluation.__main__ import main
    from evaluation.frozen import write_manifest

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model=MODEL, quantization=QUANTIZATION,
                   model_digest="c" * 64)
    argv = ["holdout", str(ANNOTATION), "--model", MODEL, "--quantization", QUANTIZATION,
            "--ledger", str(tmp_path / "ledger.json"),
            "--output", str(tmp_path / "out.json"),
            "--manifest", str(manifest)]
    assert main(argv) == 2
    # The refusal is attributable: the message names the rule that refused.
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        main(argv)
    assert "must name its holdout" in buffer.getvalue()


def test_the_committed_freeze_pins_the_weights_identity():
    """A model name is not an identity: ':latest' can be re-pulled as different
    bytes. The committed manifest pins the digest a recorded run observed, and
    says where it was observed."""
    from evaluation.frozen import DEFAULT_MANIFEST, load_manifest

    manifest = load_manifest(DEFAULT_MANIFEST)
    digest = manifest["freeze"]["model_digest"]
    assert len(digest) == 64 and digest.strip("0") != ""
    assert manifest["freeze"]["model_size_bytes"] > 0
    assert manifest["weight_identity_source"]


def test_the_recorded_identity_matches_the_committed_freeze():
    """The digest in the manifest is the one the recorded run reported — the
    manifest and the artifact cannot disagree about which weights ran."""
    from evaluation.frozen import DEFAULT_MANIFEST, load_manifest
    from evaluation.split import recorded_model_identity

    manifest = load_manifest(DEFAULT_MANIFEST)
    recorded = recorded_model_identity(manifest["freeze"]["model"])
    assert recorded["digest"] == manifest["freeze"]["model_digest"]


def test_the_holdout_record_identifies_the_predictions_it_scored(tmp_path):
    """The scored predictions and the written report are both hashed: a later
    reader can tell which bytes produced the numbers."""
    from evaluation.__main__ import main
    from evaluation.frozen import write_manifest
    from evaluation.split import HoldoutLedger

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model=MODEL, quantization=QUANTIZATION,
                   model_digest="d" * 64)
    ledger_path = tmp_path / "ledger.json"
    output = tmp_path / "holdout.json"
    argv = ["holdout", str(ANNOTATION), "--model", MODEL, "--quantization", QUANTIZATION,
            "--ledger", str(ledger_path), "--output", str(output),
            "--manifest", str(manifest),
            "--holdout", "huangqi_01.jpg", "--holdout", "huangqi_02.jpg"]
    assert main(argv) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["inputs_sha256"]) == 64
    assert payload["scored_entries"] == 2
    entry = HoldoutLedger.load(ledger_path).runs[-1]
    assert entry["inputs_sha256"] == payload["inputs_sha256"]
    assert len(entry["output_sha256"]) == 64
    assert entry["output_sha256"] != payload["inputs_sha256"]


def test_the_freeze_covers_the_workflow_that_produces_the_verdict():
    """An end-to-end trust level is computed by the workflow runner as much as
    by the auditor: its quality gate and linkage rules are frozen too."""
    from evaluation.frozen import FREEZE_INPUTS

    paths = {relative for relative, _ in FREEZE_INPUTS}
    assert "runtime/workflow.py" in paths


def test_the_freeze_verifies_from_a_fresh_checkout(tmp_path):
    """The manifest must verify against the repository's own bytes, not just
    the machine that froze it.

    Git checks out with the platform's line-ending convention, so a raw byte
    hash would report drift for every untouched file on a fresh clone. The
    hashes normalise line endings first; this pins that by re-hashing a copy
    of a frozen input with the other convention and comparing.
    """
    from evaluation.frozen import FREEZE_INPUTS, _project_root, input_hashes

    root = tmp_path / "tree"
    project = _project_root()
    for relative, _ in FREEZE_INPUTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((project / relative).read_bytes())
    victim = next(relative for relative, _ in FREEZE_INPUTS
                  if b"\n" in (project / relative).read_bytes())
    target = root / victim
    original = target.read_bytes()
    # Flip to the *other* convention: a CRLF file becomes LF, an LF file
    # becomes CRLF. Either way the text is identical and the hash must be.
    if b"\r\n" in original:
        flipped = original.replace(b"\r\n", b"\n")
    else:
        flipped = original.replace(b"\n", b"\r\n")
    assert flipped != original, "the flip must actually change the bytes"
    target.write_bytes(flipped)
    assert input_hashes(root)[victim] == input_hashes()[victim], (
        "line endings changed the hash; a fresh checkout would report drift")


def test_the_holdout_report_the_manifest_and_the_ledger_name_one_experiment(tmp_path):
    """The P0 the reviewer found, pinned: the holdout must score the freeze the
    manifest *published*.

    A second ``freeze()`` call inside the holdout command rebuilt the Freeze
    without the weight identity, so the report and ledger recorded an
    experiment id nobody committed — and the one-shot ledger guarded the wrong
    fingerprint. All three names on a run must now be the same id.
    """
    from evaluation.__main__ import main
    from evaluation.frozen import load_manifest, write_manifest
    from evaluation.split import HoldoutLedger

    manifest = tmp_path / "frozen.json"
    written = write_manifest(manifest, model=MODEL, quantization=QUANTIZATION,
                             model_digest="e" * 64)
    ledger_path, output = tmp_path / "ledger.json", tmp_path / "holdout.json"
    argv = ["holdout", str(ANNOTATION), "--model", MODEL, "--quantization", QUANTIZATION,
            "--ledger", str(ledger_path), "--output", str(output),
            "--manifest", str(manifest),
            "--holdout", "huangqi_01.jpg", "--holdout", "huangqi_02.jpg"]
    assert main(argv) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    published = load_manifest(manifest)["freeze_id"]
    entry = HoldoutLedger.load(ledger_path).runs[-1]
    assert entry["freeze_id"] == published
    assert payload["freeze"]["freeze_id"] == published[:12]
    assert entry["freeze"]["model_digest"] == written["freeze"]["model_digest"]
    assert entry["freeze"]["model_digest"], "the scored freeze must carry the weight identity"


def test_a_second_scoring_freeze_without_the_identity_is_refused(tmp_path):
    """The failure mode, isolated: rebuilding the Freeze from the current tree
    alone yields a different id, and frozen_from_manifest refuses rather than
    letting the run record an unpublished experiment."""
    from evaluation.frozen import frozen_from_manifest, load_manifest, write_manifest

    manifest = tmp_path / "frozen.json"
    write_manifest(manifest, model=MODEL, quantization=QUANTIZATION,
                   model_digest="f" * 64)
    scoring = frozen_from_manifest(manifest)
    assert scoring.model_digest == "f" * 64
    assert scoring.freeze_id == load_manifest(manifest)["freeze_id"]
    # A freeze rebuilt the old way — no identity — is a different experiment.
    from evaluation.split import freeze as make_freeze
    naive = make_freeze(model=MODEL, quantization=QUANTIZATION)
    assert naive.freeze_id != scoring.freeze_id
