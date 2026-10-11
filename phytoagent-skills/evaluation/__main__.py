"""CLI: python -m evaluation <command>.

    report   <annotations.json>   full metric set, overall and per mode
    split    <annotations.json>   validate and show the dataset roles
    freeze   --model --quantization [--manifest PATH]
                                  write the committed freeze manifest
    verify-freeze [--manifest PATH]
                                  recompute and compare; non-zero on any drift
    holdout  <annotations.json> --model --quantization   guarded one-shot run
            [--predictions P --labels L]   formal path: frozen predictions joined
                                           with independent labels at scoring time
    freeze-predictions <file> --model [--model-digest]  seal predictions first
    identity --build-dev <dir> | --new <dir> [--source --batch --role]
    agree    <first.json> <second.json>                  inter-annotator agreement

``report`` and ``holdout`` write JSON under ``artifacts/evaluations/``. Nothing
here reaches a network or a model. ``holdout`` refuses to run when the tree no
longer matches the committed freeze manifest — a blind test on a drifted
configuration is not that test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.agreement import multilabel_agreement, resolution_queue
from evaluation.frozen import (DEFAULT_MANIFEST, assert_manifest,
                               frozen_from_manifest, load_manifest,
                               verify_manifest, write_manifest)
from evaluation.metrics import risk_coverage_curve, stratify
from evaluation.identity import (DEFAULT_IDENTITY, load_identity_manifest,
                                 registered_images, verify_registration)
from evaluation.predictions import (DEFAULT_PREDICTION_STORE,
                                    freeze_predictions, join_for_scoring)
from evaluation.split import (DEFAULT_LEDGER, FreezeError, HoldoutLedger, ROLES,
                          split_entries)

DEFAULT_ANNOTATION = Path("case_workspace/annotate/huangqi_annotated_results.json")
REPORT_PATH = Path("artifacts/evaluations/dev30-report.json")
HOLDOUT_PATH = Path("artifacts/evaluations/holdout-run.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_exclusions(pairs: list[str] | None) -> dict[str, str]:
    """--exclude image=reason pairs: an exclusion is a recorded act, not a silence.

    A damaged image or a missing annotation may legitimately shrink what is
    scored; the protocol requires the reason and the full denominator to travel
    with the result, so the reader sees N of M and why.
    """
    exclusions: dict[str, str] = {}
    for pair in pairs or []:
        image, _, reason = pair.partition("=")
        image, reason = image.strip(), reason.strip()
        if not image or not reason:
            raise FreezeError(
                f"--exclude wants image=reason with a non-empty reason: {pair!r}")
        if image in exclusions:
            raise FreezeError(f"--exclude repeats {image}")
        exclusions[image] = reason
    return exclusions


def _entries(path: Path) -> list[dict]:
    payload = _read(path)
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise SystemExit(f"no entries list in {path}")
    return entries


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _report(args: argparse.Namespace) -> int:
    entries = _entries(Path(args.annotations))
    by_mode = stratify(entries, key="mode")
    report = {
        "scope": "evaluation_report",
        "source": str(args.annotations),
        "caveats": [
            "这是 calibration/development 数据上的度量，不是验证集结果。",
            "标注者只有一人时，任何准确率都无法与「一个人的判断」分开；"
            "先跑 python -m evaluation agree 得到标注者间一致性。",
            "coverage 与 abstention_rate 必须与 precision/recall 一起读："
            "precision 高但 coverage 低，等于只回答了少数样本。",
        ],
        "modes": {name: evaluation.to_dict() for name, evaluation in by_mode.items()},
        "risk_coverage": {
            "available": risk_coverage_curve(entries) is not None,
            "note": ("需要每条样本带 model_confidence；当前标注文件没有该字段，"
                     "因此只报告自然工作点 risk_at_coverage。"),
        },
    }
    _write(Path(args.output), report)
    overall = by_mode["overall"]
    print(json.dumps({
        "source": str(args.annotations),
        "annotated": overall.annotated,
        "coverage": overall.coverage,
        "abstention_rate": overall.abstention_rate,
        "precision": overall.precision,
        "recall": overall.recall,
        "f1": overall.f1,
        "selective_recall": overall.selective_recall,
        "risk_at_coverage": overall.risk_at_coverage,
        "macro_f1": overall.macro_f1,
        "per_mode": {name: {"coverage": evaluation.coverage,
                            "precision": evaluation.precision,
                            "recall": evaluation.recall,
                            "macro_f1": evaluation.macro_f1}
                     for name, evaluation in by_mode.items() if name != "overall"},
        "written": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


def _split(args: argparse.Namespace) -> int:
    entries = _entries(Path(args.annotations))
    split = split_entries(entries,
                          holdout=tuple(args.holdout or ()),
                          development=tuple(args.development or ()))
    print(json.dumps(split.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    """Write the committed freeze manifest: the record the holdout is held to."""
    manifest = write_manifest(Path(args.manifest), model=args.model,
                              quantization=args.quantization,
                              model_digest=args.model_digest)
    print(json.dumps({"written": str(args.manifest),
                      "freeze_id": manifest["freeze_id"],
                      "model_digest": manifest["freeze"]["model_digest"],
                      "weight_identity_source": manifest["weight_identity_source"],
                      "inputs": len(manifest["inputs"])}, ensure_ascii=False, indent=2))
    return 0


def _freeze_predictions(args: argparse.Namespace) -> int:
    """Seal a predictions file before the independent labels exist.

    The seal binds the published experiment: freeze_id, model and weight
    digest. Defaults come from the committed manifest, so the command cannot
    seal predictions against an identity nobody published.
    """
    # The identity comes from the verified manifest — freeze_id, model and
    # digest together. Reading freeze_id out of the wrong place (it lives at the
    # manifest's top level, not inside the freeze object) is exactly the bug the
    # default path had; going through frozen_from_manifest makes that impossible.
    manifest_path = Path(args.manifest)
    freeze_id, digest = args.freeze_id, args.model_digest
    if not (freeze_id and digest):
        if not manifest_path.is_file():
            raise FreezeError(
                "no freeze manifest to read the identity from; pass --freeze-id "
                "and --model-digest explicitly")
        published = frozen_from_manifest(manifest_path)
        freeze_id = freeze_id or published.freeze_id
        digest = digest or published.model_digest
    if not digest:
        raise FreezeError(
            "no weight digest to seal against; the published manifest pins none — "
            "re-freeze with an observable model identity first")
    if manifest_path.is_file() and args.model != frozen_from_manifest(manifest_path).model:
        raise FreezeError(
            f"--model {args.model!r} is not the model the published freeze names "
            f"({frozen_from_manifest(manifest_path).model!r}); predictions for a "
            "different model cannot be sealed against this freeze")
    record = freeze_predictions(Path(args.predictions), model=args.model,
                                model_digest=digest, freeze_id=freeze_id,
                                store=Path(args.prediction_store),
                                attest=args.attest)
    print(json.dumps({"frozen": record["predictions_sha256"],
                      "entries": record["entries"],
                      "model": record["model"],
                      "model_digest": record["model_digest"][:16],
                      "frozen_at": record["frozen_at"]}, ensure_ascii=False, indent=2))
    return 0


def _identity(args: argparse.Namespace) -> int:
    """Build the development identity manifest, or register new data.

    Registration is the R3-2 data-independence gate: every new image is
    identified by SHA-256 and perceptual hash and cross-checked against the
    development set — an exact duplicate or a near duplicate refuses, because
    a blind test cannot contain an image the prompts were tuned on.
    """
    from evaluation.identity import (assert_independent, load_identity_manifest,
                                     scan_directory, write_identity_manifest)

    if args.build_dev:
        identities = scan_directory(args.build_dev)
        manifest = write_identity_manifest(
            identities, args.manifest,
            note="The development and calibration images, identified. New holdout "
                 "data is cross-checked against this list before registration.")
        print(json.dumps({"written": str(args.manifest), "count": manifest["count"]},
                         ensure_ascii=False, indent=2))
        return 0

    if not args.new:
        raise FreezeError("identity needs --build-dev <dir> or --new <dir>")
    if not args.source or not args.batch:
        raise FreezeError(
            "registering new data requires --source and --batch: the perceptual "
            "gate catches the same photograph re-encoded, and only the collector's "
            "record can say whether two images share a plant")
    unknown_roles = sorted(set(args.role) - set(ROLES))
    if unknown_roles:
        raise FreezeError(f"unknown roles {unknown_roles}; the roles are {sorted(ROLES)}")
    reference = load_identity_manifest(args.manifest)
    identities = scan_directory(args.new)
    report = assert_independent(identities, reference,
                                allow_manual_review=args.allow_manual_review)
    roles = args.role or ["holdout"] * len(identities)
    if len(roles) != len(identities):
        raise FreezeError("one role per image (--role, repeatable), or none at all")
    manifest = write_identity_manifest(
        identities, args.out,
        note=f"Independent holdout data. source={args.source or 'unrecorded'} "
             f"batch={args.batch or 'unrecorded'}; roles assigned before any model "
             f"output was seen; cross-checked against {args.manifest} "
             f"({report['checked']} checked, {report['reference_count']} reference).")
    manifest["source"] = args.source or ""
    manifest["batch"] = args.batch or ""
    manifest["roles"] = {identity.path: role for identity, role in zip(identities, roles)}
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps({"written": str(args.out), "count": manifest["count"],
                      "cross_check": report}, ensure_ascii=False, indent=2))
    return 0


def _verify_freeze(args: argparse.Namespace) -> int:
    report = verify_manifest(Path(args.manifest))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.ok:
        print(json.dumps({"error": "FreezeDrift",
                          "message": "the tree no longer matches the frozen manifest"},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


def _holdout(args: argparse.Namespace) -> int:
    # A holdout run on a drifted configuration is not the frozen experiment.
    # The ledger guards re-runs of the *same* freeze; this guards the freeze
    # itself against silent edits between freezing and running. The manifest
    # is mandatory: a missing one is a refusal, not a skip.
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FreezeError(
            f"no freeze manifest at {manifest_path}; the holdout gate is not "
            "optional — freeze first (python -m evaluation freeze ...)")
    assert_manifest(manifest_path, model=args.model, quantization=args.quantization)
    # The scoring Freeze is the one the manifest published — same aggregates,
    # same weight identity, same id. Re-fingerprinting here would produce a
    # second experiment whose id nobody committed, and the ledger would then
    # guard the wrong thing.
    frozen = frozen_from_manifest(manifest_path, model=args.model,
                                  quantization=args.quantization)

    # Exclusions are parsed up front: they are recorded acts, and the formal
    # path resolves them before the pairing requirement (see below).
    exclusions = _parse_exclusions(args.exclude)

    # The holdout set is mandatory. Without an explicit list every entry would
    # default to the calibration role and then be scored as holdout anyway —
    # a "blind test" over data the prompts were developed against.
    if not args.holdout:
        raise FreezeError(
            "a holdout run must name its holdout images (--holdout, repeatable); "
            "an undeclared holdout scores whatever it is given")

    # The formal path: predictions frozen before the labels existed, scored by
    # joining the two files. Hashing one file that already contains both sides
    # proves nothing about order; this does.
    joined = None
    if args.predictions or args.labels:
        if not (args.predictions and args.labels):
            raise FreezeError(
                "--predictions and --labels go together: the formal blind test "
                "scores frozen predictions joined with independent labels")
        # The sealed predictions must belong to the freeze being scored: the
        # same freeze_id, the same model, the same weight digest.
        # Admission and scoring are separate paths; this is the bridge. Every
        # scored image must be registered — same bytes, role holdout — or the
        # holdout is scoring data nobody vetted, or a same-name substitution.
        if not args.identity:
            raise FreezeError(
                "the formal blind test requires --identity: the registered-data "
                "manifest every scored image must appear in")
        identity_manifest = load_identity_manifest(args.identity)
        registered = registered_images(identity_manifest)
        # Exclusions are resolved before the join: a damaged or unlabelled image
        # is recorded as excluded, not left to fail the pairing check.
        joined = join_for_scoring(
            Path(args.predictions), Path(args.labels), store=Path(args.prediction_store),
            expected={"freeze_id": frozen.freeze_id, "model": frozen.model,
                      "model_digest": frozen.model_digest},
            excluded=exclusions)
        entries = joined["entries"]
        # The full declared set (scored + excluded) drives the role split, so an
        # excluded image stays "in the dataset" and the declared roles remain
        # checkable.
        all_entries = joined["all_entries"]
        annotations_path = Path(args.predictions)
        verified = verify_registration(entries, registered)
        admission = {
            "manifest": str(args.identity),
            "manifest_sha256": _sha256_file(Path(args.identity)),
            "source": identity_manifest.get("source", ""),
            "batch": identity_manifest.get("batch", ""),
            "roles": {identifier: verified[identifier].get("role", "")
                      for identifier in sorted(verified)},
        }

        # The scored set must equal the pre-registered set. Freezing 100 images
        # and scoring the 20 that looked best is a different experiment, and a
        # smaller denominator nobody declared.
        pre_registered = set(joined["pre_registered"])
        scored = {entry["image"] for entry in entries}
        declared = set(args.holdout)
        if declared != pre_registered:
            raise FreezeError(
                "the formal holdout set must equal the pre-registered set: "
                f"{len(pre_registered)} images were frozen, --holdout names "
                f"{len(declared)}; missing={sorted(pre_registered - declared)[:5]}, "
                f"extra={sorted(declared - pre_registered)[:5]}")
        unknown = sorted(set(exclusions) - pre_registered)
        if unknown:
            raise FreezeError(f"--exclude names images outside the frozen set: {unknown}")
        # The join already applied the exclusions: ``entries`` is the scored set,
        # ``all_entries`` the full declared set — so the split below sees every
        # declared image and the reader can tell N of M and why.
        if not entries:
            raise FreezeError("every pre-registered image was excluded; nothing to score")
    else:
        entries = _entries(Path(args.annotations))
        annotations_path = Path(args.annotations)
        all_entries = list(entries)
        admission = None
    holdout = tuple(args.holdout)
    split = split_entries(all_entries, holdout=holdout)
    if joined is None:
        entries = [entry for entry in entries if entry.get("image") in set(holdout)]
        if not entries:
            raise FreezeError("no annotation entries matched the declared holdout set")
    # Development and calibration samples must never be scored as holdout: the
    # split already assigned them, and the metrics below cover the holdout only.
    if any(entry.get("image") in set(split.development) | set(split.calibration)
           for entry in entries):
        raise FreezeError("holdout entries overlap the development or calibration set")

    ledger = HoldoutLedger.load(Path(args.ledger))
    metrics = stratify(entries, key="mode")
    # The denominator must not shrink silently. The metrics skip entries with no
    # reference, so a joined entry whose annotator_labels went missing would
    # quietly reduce what was scored; the join now refuses that, and this check
    # is the belt-and-braces assertion that what the metrics saw is what the
    # report says was scored.
    # The overall slice, not a sum over the per-mode slices: stratify reports
    # both, and adding them counts every entry twice.
    annotated = metrics["overall"].annotated
    if annotated != len(entries):
        raise FreezeError(
            f"the metrics annotated {annotated} of {len(entries)} scored entries; "
            "a silently shrunken denominator is a refusal, not a rounding detail")
    # The predictions are the artifact the numbers were computed from; their
    # hash is recorded so a later reader can tell which bytes were scored. On
    # the formal path the frozen record already carries it; the labels' hash
    # travels alongside, so both sides of the join are identifiable.
    inputs_sha256 = (joined["predictions_sha256"] if joined
                     else _sha256_file(annotations_path))
    payload = {
        # The formal path is a holdout evaluation; the legacy single-file path
        # is a development evaluation, and says so — it does not require
        # predictions, labels or a registered-data manifest, so it cannot claim
        # the same guarantees.
        "scope": "holdout_evaluation" if joined else "development_evaluation",
        "freeze": frozen.summary(),
        "split": split.to_dict(),
        "inputs_sha256": inputs_sha256,
        "labels_sha256": joined["labels_sha256"] if joined else None,
        "predictions_frozen_at": joined["frozen_at"] if joined else None,
        "pre_registered": sorted(joined["pre_registered"]) if joined else [],
        "admission": admission if joined else None,
        "excluded": [{"image": image, "reason": reason}
                     for image, reason in sorted(exclusions.items())],
        "scored_entries": len(entries),
        "annotated_entries": annotated,
        "metrics": {name: evaluation.to_dict() for name, evaluation in metrics.items()},
    }
    # The ledger write is what enforces the one-shot rule; it happens after the
    # numbers are computed so a refusal leaves no half-recorded run behind.
    ledger.record(frozen, metrics=payload["metrics"], note=args.note,
                  supersede=args.supersede)
    output = Path(args.output)
    _write(output, payload)
    # The report's own identity goes into the ledger, not into itself: a file
    # cannot contain its own hash, and the ledger is the audit trail.
    entry = ledger.runs[-1]
    entry["output_sha256"] = _sha256_file(output)
    entry["inputs_sha256"] = inputs_sha256
    ledger.save()
    print(json.dumps({"freeze_id": frozen.freeze_id[:12],
                      "holdout_images": len(entries),
                      "inputs_sha256": inputs_sha256[:12],
                      "output_sha256": entry["output_sha256"][:12],
                      "written": str(output),
                      "ledger": str(ledger.path)}, ensure_ascii=False, indent=2))
    return 0


def _agree(args: argparse.Namespace) -> int:
    report = multilabel_agreement(_read(Path(args.first)), _read(Path(args.second)))
    queue = resolution_queue(_read(Path(args.first)), _read(Path(args.second)))
    report["disagreements"] = [item.to_dict() for item in queue]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhytoForge evaluation tools")
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="metrics, overall and per mode")
    report.add_argument("annotations", nargs="?", default=str(DEFAULT_ANNOTATION))
    report.add_argument("--output", default=str(REPORT_PATH))
    report.set_defaults(handler=_report)

    split = commands.add_parser("split", help="show and validate dataset roles")
    split.add_argument("annotations", nargs="?", default=str(DEFAULT_ANNOTATION))
    split.add_argument("--holdout", nargs="*", default=[])
    split.add_argument("--development", nargs="*", default=[])
    split.set_defaults(handler=_split)

    frozen = commands.add_parser("freeze", help="write the committed freeze manifest")
    frozen.add_argument("--model", required=True)
    frozen.add_argument("--quantization", required=True)
    frozen.add_argument("--model-digest", default="",
                        help="the weights' digest; defaults to the identity "
                             "recorded by an earlier real run")
    frozen.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    frozen.set_defaults(handler=_freeze)

    predictions = commands.add_parser(
        "freeze-predictions", help="seal predictions before the labels exist")
    predictions.add_argument("predictions")
    predictions.add_argument("--model", required=True)
    predictions.add_argument("--model-digest", default="",
                             help="defaults to the digest the manifest pins")
    predictions.add_argument("--freeze-id", default="",
                             help="defaults to the manifest's freeze_id")
    predictions.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    predictions.add_argument("--attest", default="",
                             help="independent custodian's delivery record, if one exists")
    predictions.add_argument("--prediction-store", default=str(DEFAULT_PREDICTION_STORE),
                             help="where the frozen prediction records live")
    predictions.set_defaults(handler=_freeze_predictions)

    identity = commands.add_parser(
        "identity", help="build the development identity manifest, or register new data")
    identity.add_argument("--build-dev", default="",
                          help="directory of development/calibration images")
    identity.add_argument("--new", default="",
                          help="directory of new candidate images")
    identity.add_argument("--source", default="",
                          help="where the new images were collected")
    identity.add_argument("--batch", default="",
                          help="plant or batch identifier, for near-duplicate provenance")
    identity.add_argument("--role", action="append", default=[],
                          help="role per new image, assigned before any model output")
    identity.add_argument("--allow-manual-review", action="store_true",
                          help="accept flat-frame dHash hits a human has inspected")
    identity.add_argument("--manifest", default=str(DEFAULT_IDENTITY))
    identity.add_argument("--out", default="artifacts/evaluations/new-data-identity.json")
    identity.set_defaults(handler=_identity)

    verify = commands.add_parser("verify-freeze", help="recompute and compare the freeze")
    verify.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    verify.set_defaults(handler=_verify_freeze)

    holdout = commands.add_parser("holdout", help="one-shot guarded holdout evaluation")
    holdout.add_argument("annotations", nargs="?", default=str(DEFAULT_ANNOTATION))
    holdout.add_argument("--model", required=True)
    holdout.add_argument("--quantization", required=True)
    # action=append, not nargs="*": repeating the flag must accumulate the
    # holdout set. With nargs="*" a second --holdout silently replaces the
    # first, and half the declared blind set would vanish without a word.
    holdout.add_argument("--holdout", action="append", default=[])
    holdout.add_argument("--note", default="")
    holdout.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    holdout.add_argument("--output", default=str(HOLDOUT_PATH))
    holdout.add_argument("--supersede", action="store_true")
    holdout.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    holdout.add_argument("--predictions", default="",
                         help="frozen predictions file (the formal blind-test path)")
    holdout.add_argument("--labels", default="",
                         help="independent labels file, joined at scoring time")
    holdout.add_argument("--exclude", action="append", default=[],
                         help="image=reason: a recorded exclusion from the frozen set")
    holdout.add_argument("--prediction-store", default=str(DEFAULT_PREDICTION_STORE),
                         help="where the frozen prediction records live")
    holdout.add_argument("--identity", default="",
                         help="registered-data manifest; required on the formal path")
    holdout.set_defaults(handler=_holdout)

    agree = commands.add_parser("agree", help="inter-annotator agreement")
    agree.add_argument("first")
    agree.add_argument("second")
    agree.set_defaults(handler=_agree)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
