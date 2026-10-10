"""CLI: python -m evaluation <command>.

    report   <annotations.json>   full metric set, overall and per mode
    split    <annotations.json>   validate and show the dataset roles
    freeze   --model --quantization
    holdout  <annotations.json> --model --quantization   guarded one-shot run
    agree    <first.json> <second.json>                  inter-annotator agreement

``report`` and ``holdout`` write JSON under ``artifacts/evaluations/``. Nothing
here reaches a network or a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.agreement import multilabel_agreement, resolution_queue
from evaluation.metrics import risk_coverage_curve, stratify
from evaluation.split import (DEFAULT_LEDGER, HoldoutLedger, freeze as make_freeze,
                              split_entries)

DEFAULT_ANNOTATION = Path("case_workspace/annotate/huangqi_annotated_results.json")
REPORT_PATH = Path("artifacts/evaluations/dev30-report.json")
HOLDOUT_PATH = Path("artifacts/evaluations/holdout-run.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    frozen = make_freeze(model=args.model, quantization=args.quantization)
    print(json.dumps(frozen.summary(), ensure_ascii=False, indent=2))
    return 0


def _holdout(args: argparse.Namespace) -> int:
    entries = _entries(Path(args.annotations))
    split = split_entries(entries, holdout=tuple(args.holdout or ()))
    if args.holdout:
        entries = [entry for entry in entries if entry.get("image") in set(args.holdout)]
    frozen = make_freeze(model=args.model, quantization=args.quantization)
    ledger = HoldoutLedger.load(Path(args.ledger))
    metrics = stratify(entries, key="mode")
    payload = {
        "scope": "holdout_evaluation",
        "freeze": frozen.summary(),
        "split": split.to_dict(),
        "metrics": {name: evaluation.to_dict() for name, evaluation in metrics.items()},
    }
    # The ledger write is what enforces the one-shot rule; it happens after the
    # numbers are computed so a refusal leaves no half-recorded run behind.
    ledger.record(frozen, metrics=payload["metrics"], note=args.note,
                  supersede=args.supersede)
    _write(Path(args.output), payload)
    print(json.dumps({"freeze_id": frozen.freeze_id[:12],
                      "holdout_images": len(entries),
                      "written": str(args.output),
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

    frozen = commands.add_parser("freeze", help="fingerprint the current code and model")
    frozen.add_argument("--model", required=True)
    frozen.add_argument("--quantization", required=True)
    frozen.set_defaults(handler=_freeze)

    holdout = commands.add_parser("holdout", help="one-shot guarded holdout evaluation")
    holdout.add_argument("annotations", nargs="?", default=str(DEFAULT_ANNOTATION))
    holdout.add_argument("--model", required=True)
    holdout.add_argument("--quantization", required=True)
    holdout.add_argument("--holdout", nargs="*", default=[])
    holdout.add_argument("--note", default="")
    holdout.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    holdout.add_argument("--output", default=str(HOLDOUT_PATH))
    holdout.add_argument("--supersede", action="store_true")
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
