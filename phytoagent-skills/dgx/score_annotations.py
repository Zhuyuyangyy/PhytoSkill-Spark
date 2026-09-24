"""Score the model's observations against the annotator's labels.

The metric is deliberately two-sided. Reporting precision alone would let the
model score well by saying little; recall alone by saying everything. Both are
computed, together with the image-usability agreement, so a claim about accuracy
says what it measured.

Reads artifacts/dgx/annotation-worksheet.json (annotator_labels and
annotator_agrees filled in) and writes artifacts/dgx/annotation-score.json.
"""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path("artifacts/dgx/annotation-worksheet.json")
OUTPUT = Path("artifacts/dgx/annotation-score.json")


def _normalise(labels) -> list[str]:
    """Lower-case and de-duplicate, so casing is not a disagreement."""
    seen: list[str] = []
    for label in labels or []:
        if not isinstance(label, str):
            continue
        key = label.strip().lower()
        if key and key not in seen:
            seen.append(key)
    return seen


def score(worksheet: dict) -> dict:
    entries = worksheet.get("entries", [])
    annotated = [entry for entry in entries
                 if entry.get("annotator_agrees") is not None
                 or entry.get("annotator_labels")]
    if not annotated:
        return {"status": "not_annotated",
                "images": len(entries),
                "annotated": 0,
                "note": "Fill annotator_labels and/or annotator_agrees, then re-run."}

    true_positives = false_positives = false_negatives = 0
    disagreements: list[dict] = []
    usability_both_usable = usability_both_unusable = 0
    usability_disagree = 0

    for entry in annotated:
        predicted = set(_normalise(entry.get("model_predicted")))
        actual = set(_normalise(entry.get("annotator_labels")))
        hit = predicted & actual
        true_positives += len(hit)
        false_positives += len(predicted - actual)
        false_negatives += len(actual - predicted)
        if predicted != actual:
            disagreements.append({
                "image": entry["image"],
                "mode": entry["mode"],
                "model_predicted": sorted(predicted),
                "annotator_labels": sorted(actual),
                "missed_by_model": sorted(actual - predicted),
                "extra_in_model": sorted(predicted - actual),
            })
        model_usable = entry.get("model_image_usable")
        annotator_usable = entry.get("annotator_image_usable")
        if isinstance(annotator_usable, bool) and isinstance(model_usable, bool):
            if model_usable == annotator_usable:
                if model_usable:
                    usability_both_usable += 1
                else:
                    usability_both_unusable += 1
            else:
                usability_disagree += 1

    precision = (true_positives / (true_positives + false_positives)
                 if true_positives + false_positives else None)
    recall = (true_positives / (true_positives + false_negatives)
              if true_positives + false_negatives else None)
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "status": "scored",
        "images": len(entries),
        "annotated": len(annotated),
        "labels": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "image_usability": {
            "both_usable": usability_both_usable,
            "both_unusable": usability_both_unusable,
            "disagree": usability_disagree,
            "annotated": (usability_both_usable + usability_both_unusable
                          + usability_disagree),
        },
        "disagreements": disagreements,
        "caveats": [
            "样本量小（当前标注 %d 张），这些数字只能当方向性观察。" % len(annotated),
            "标注者只有一人，没有第二人交叉验证，标注者间一致性未测。",
            "precision 与 recall 同时报：只报一个会高估。",
        ],
    }


def main() -> int:
    if not WORKSPACE.is_file():
        print(f"worksheet not found: {WORKSPACE}", file=sys.stderr)
        return 2
    workspace = json.loads(WORKSPACE.read_text(encoding="utf-8"))
    result = score(workspace)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    if result["status"] == "not_annotated":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({key: result[key] for key in
                      ("status", "images", "annotated", "labels",
                       "precision", "recall", "f1", "image_usability")},
                     ensure_ascii=False, indent=2))
    print(f"\n{len(result['disagreements'])} images disagree; details in {OUTPUT}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
