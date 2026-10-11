"""CLI: python -m evaluation <command>.

    report   <annotations.json>   full metric set, overall and per mode
    split    <annotations.json>   validate and show the dataset roles
    freeze   --model --quantization [--manifest PATH]
                                  write the committed freeze manifest
    verify-freeze [--manifest PATH]
                                  recompute and compare; non-zero on any drift
    holdout  <annotations.json> --model --quantization   guarded one-shot run
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
                               frozen_from_manifest, verify_manifest,
                               write_manifest)
from evaluation.metrics import risk_coverage_curve, stratify
from evaluation.split import DEFAULT_LEDGER, FreezeError, HoldoutLedger, split_entries

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

    # The holdout set is mandatory. Without an explicit list every entry would
    # default to the calibration role and then be scored as holdout anyway —
    # a "blind test" over data the prompts were developed against.
    if not args.holdout:
        raise FreezeError(
            "a holdout run must name its holdout images (--holdout, repeatable); "
            "an undeclared holdout scores whatever it is given")
    entries = _entries(Path(args.annotations))
    holdout = tuple(args.holdout)
    split = split_entries(entries, holdout=holdout)
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
    # The predictions are the artifact the numbers were computed from; their
    # hash is recorded so a later reader can tell which bytes were scored.
    inputs_sha256 = _sha256_file(Path(args.annotations))
    payload = {
        "scope": "holdout_evaluation",
        "freeze": frozen.summary(),
        "split": split.to_dict(),
        "inputs_sha256": inputs_sha256,
        "scored_entries": len(entries),
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
