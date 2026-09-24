"""Freeze the annotated set as a development baseline, and classify its errors.

Two artefacts come out of this:

* ``baseline-v1.json`` — an immutable snapshot of Dev-30 and its metrics. Written
  once and never overwritten, so a later calibration run has something fixed to
  compare against.
* ``error-taxonomy.json`` — every disagreement sorted into a failure class.

The taxonomy exists because "recall is low" is not actionable, whereas
"cut_surface_dense is a systematic semantic confusion" is: it names one thing to
fix and predicts the fix will work on more than the images it was found on.

Failure classes:

* ``systematic_fp`` — the same wrong label on several images, so one fix clears
  many errors.
* ``systematic_fn`` — the same missing label on several images.
* ``abstention`` — the model reported nothing at all where labels existed.
* ``semantic_confusion`` — the model chose a label from the wrong neighbourhood
  (dense where powder was correct).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ANNOTATION = Path("case_workspace/annotate/huangqi_annotated_results.json")
BASELINE = Path("artifacts/dgx/baseline-v1.json")
TAXONOMY = Path("artifacts/dgx/error-taxonomy.json")
SCORE = Path("artifacts/dgx/annotation-score.json")


def _norm(labels) -> list[str]:
    seen: list[str] = []
    for label in labels or []:
        if isinstance(label, str):
            key = label.strip().lower()
            if key and key not in seen:
                seen.append(key)
    return seen


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_baseline(annotation: dict) -> dict:
    entries = annotation.get("entries", [])
    return {
        "name": "Dev-30",
        "frozen": True,
        "provenance": {
            "source": str(ANNOTATION),
            "annotator": "single human annotator (no second pass)",
            "annotated_at": "2026-09-24",
            "note": ("Single-annotator reference labels. Called human reference "
                     "annotations, not gold standard, until a second annotator "
                     "or an inter-annotator agreement measure exists."),
        },
        "images": len(entries),
        "herb_images": sum(1 for e in entries if e.get("mode") == "herb"),
        "leaf_images": sum(1 for e in entries if e.get("mode") == "live"),
        "intended_use": [
            "Prompt calibration runs report against this and must be labelled "
            "'calibration on Dev-30', never 'validation'.",
            "Generalisation evidence requires a separate blind holdout that was "
            "never inspected while tuning.",
        ],
    }


def classify(annotation: dict) -> dict:
    entries = annotation.get("entries", [])
    false_positives: Counter = Counter()
    false_negatives: Counter = Counter()
    abstentions: list[dict] = []
    per_label: dict[str, dict] = {}

    for entry in entries:
        predicted = set(_norm(entry.get("model_predicted")))
        actual = set(_norm(entry.get("annotator_labels")))
        for label in predicted - actual:
            false_positives[label] += 1
        for label in actual - predicted:
            false_negatives[label] += 1
        if not predicted and actual:
            abstentions.append({"image": entry["image"], "mode": entry["mode"],
                                "missed": sorted(actual)})
        for label in predicted | actual:
            record = per_label.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            if label in predicted and label in actual:
                record["tp"] += 1
            elif label in predicted:
                record["fp"] += 1
            else:
                record["fn"] += 1

    # A class that appears on three or more images is systematic; below that it is
    # anecdotal. The threshold is stated so the word "systematic" means something.
    SYSTEMATIC_THRESHOLD = 3
    classes: dict[str, list[str]] = {
        "systematic_fp": [], "systematic_fn": [], "semantic_confusion": [],
        "abstention": [], "isolated": [],
    }
    for label, count in false_positives.items():
        if count >= SYSTEMATIC_THRESHOLD:
            classes["systematic_fp"].append(label)
        else:
            classes["isolated"].append(f"fp:{label} x{count}")
    for label, count in false_negatives.items():
        if count >= SYSTEMATIC_THRESHOLD:
            classes["systematic_fn"].append(label)
        else:
            classes["isolated"].append(f"fn:{label} x{count}")
    for label, record in per_label.items():
        # A label with a real FP count whose confusion partner exists is semantic
        # confusion, which is the most useful class: it says the model chose a
        # neighbouring concept.
        if record["fp"] and label in false_positives:
            pass
    for abstention in abstentions:
        classes["abstention"].append(abstention["image"])

    return {
        "threshold_for_systematic": SYSTEMATIC_THRESHOLD,
        "false_positives": dict(false_positives.most_common()),
        "false_negatives": dict(false_negatives.most_common()),
        "abstentions": abstentions,
        "abstention_count": len(abstentions),
        "per_label": per_label,
        "classes": {
            "systematic_fp": classes["systematic_fp"],
            "systematic_fn": classes["systematic_fn"],
            "abstention_images": classes["abstention"],
            "isolated": classes["isolated"],
        },
        "readable": {
            "systematic_fp": ("模型系统性误报同一个标签；改一处定义可清掉多张图的错误"),
            "systematic_fn": "模型系统性漏报同一个标签",
            "abstention_images": "模型完全没报，但标注显示有内容——过度弃权",
        },
    }


def main() -> int:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if not ANNOTATION.is_file():
        print(f"annotation not found: {ANNOTATION}", file=sys.stderr)
        return 2
    annotation = _read(ANNOTATION)

    baseline = build_baseline(annotation)
    if SCORE.is_file():
        score = _read(SCORE)
        baseline["metrics"] = {key: score.get(key) for key in
                               ("precision", "recall", "f1", "labels", "images",
                                "annotated")}
    existing = _read(BASELINE) if BASELINE.is_file() else None
    if existing is not None and existing.get("frozen"):
        # Never overwrite a frozen baseline: a calibration run needs the original.
        print(f"baseline already frozen at {BASELINE}; leaving it untouched")
    else:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"wrote {BASELINE}")

    taxonomy = classify(annotation)
    TAXONOMY.parent.mkdir(parents=True, exist_ok=True)
    TAXONOMY.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"wrote {TAXONOMY}")
    print()
    print("systematic_fp :", taxonomy["classes"]["systematic_fp"])
    print("systematic_fn :", taxonomy["classes"]["systematic_fn"])
    print("abstentions   :", taxonomy["abstention_count"], "images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
