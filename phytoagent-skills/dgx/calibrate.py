"""Calibrate the vision prompt against the frozen Dev-30 set, and label it honestly.

This is **calibration**, not validation. The 30 images were annotated, their
errors were inspected, and the prompt was then changed to address those specific
errors. Reporting a metric from the same images as an improvement would be
evaluation leakage — the same mistake the AgentShield harness guards against on
the Agent side.

So this script refuses to call the result a generalisation number. It prints
"calibration on Dev-30" and points at the blind holdout as the only place a
generalisation claim may come from.

    python -m dgx.calibrate --mode herb
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ANNOTATION = Path("case_workspace/annotate/huangqi_annotated_results.json")
BASELINE = Path("artifacts/dgx/baseline-v1.json")
SCORE = Path("artifacts/dgx/annotation-score.json")
OUTPUT = Path("artifacts/dgx/calibration-v1.json")


def load_entries(mode: str) -> list[dict]:
    annotation = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    return [entry for entry in annotation.get("entries", [])
            if entry.get("mode") == mode]


def _norm(labels) -> set[str]:
    seen: set[str] = set()
    for label in labels or []:
        if isinstance(label, str):
            key = label.strip().lower()
            if key:
                seen.add(key)
    return seen


def score_against(annotation_path: Path, mode: str) -> dict:
    """Reuse the project's own scorer so calibration and baseline cannot drift."""
    from dgx.score_annotations import score
    return score(json.loads(annotation_path.read_text(encoding="utf-8")))


def _baseline_for_mode(mode: str) -> dict:
    """Per-mode baseline, recomputed rather than read whole.

    The frozen baseline stores overall metrics across both modes; comparing a
    single mode's calibrated number against the overall one would be meaningless.
    """
    entries = load_entries(mode)
    tp = fp = fn = 0
    for entry in entries:
        predicted = _norm(entry.get("model_predicted"))
        actual = _norm(entry.get("annotator_labels"))
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "images": len(entries),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(prog="dgx.calibrate", description=__doc__)
    parser.add_argument("--mode", default="herb", choices=["herb", "live"])
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSON mapping image name -> list of phenotypes the "
                             "calibrated prompt produced")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not BASELINE.is_file():
        print(f"baseline not frozen: run `python -m dgx.freeze_baseline` first",
              file=sys.stderr)
        return 2
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if not baseline.get("frozen"):
        print("baseline exists but is not frozen", file=sys.stderr)
        return 2

    entries = load_entries(args.mode)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    tp = fp = fn = 0
    changes: list[dict] = []
    for entry in entries:
        predicted = _norm(predictions.get(entry["image"]))
        actual = _norm(entry.get("annotator_labels"))
        before = _norm(entry.get("model_predicted"))
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        if predicted != before:
            changes.append({"image": entry["image"], "before": sorted(before),
                            "after": sorted(predicted)})
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)

    record = {
        "scope": "calibration_on_dev30",
        # Stated in the artifact itself, so a reader cannot mistake it.
        "is_generalisation_evidence": False,
        "why_not": ("Dev-30 was inspected before this prompt was written. A metric "
                    "from it measures error-driven calibration, not generalisation. "
                    "Generalisation requires a blind holdout that was never looked "
                    "at while tuning."),
        "mode": args.mode,
        "baseline": _baseline_for_mode(args.mode),
        "calibrated": {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        },
        "predictions_changed": len(changes),
        "changes": changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
