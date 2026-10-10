"""Selective evaluation for label-set observations.

Why this module exists: precision alone let a model look good by saying little.
The Dev-30 leaf subset reported ``precision = 1.000`` — because the model
answered *three times out of fifteen*. A metric that cannot see abstention
cannot see that.

Two families are computed.

**Set metrics** — micro, macro and per-label precision / recall / F1 over the
whole annotated set. Micro is the overall figure. Macro averages the per-label
F1 over the labels that actually occur, so a frequent label cannot drown a rare
one.

**Selective metrics** — coverage, abstention rate, and the scores restricted to
the images the model actually answered, plus the risk at that coverage.

One structural fact is worth stating, because it is the whole reason this module
exists: **an abstention can only ever cost recall, never precision.** An empty
prediction cannot produce a false positive. So a model that abstains freely will
report a high precision at a low coverage, and precision alone will not reveal
it. The gap between :attr:`Evaluation.selective_recall` and
:attr:`Evaluation.recall` is the exact price of the abstentions.

Terminology, fixed here so reports cannot drift:

* *answered* — the model produced at least one label for the image.
* *abstained* — the model produced none **and** the reference found something.
  An image where both are empty is correctly silent, not an abstention.
* *coverage* — answered / annotated.
* *abstention rate* — abstained / annotated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Reference and prediction are both label *sets* per image. These are the keys
# the annotation files use; a caller may override them.
DEFAULT_REFERENCE_KEY = "annotator_labels"
DEFAULT_PREDICTION_KEY = "model_predicted"
DEFAULT_CONFIDENCE_KEY = "model_confidence"


def normalise_labels(labels) -> list[str]:
    """Lower-case, strip and de-duplicate, preserving order.

    Casing and repeats are formatting, not disagreement, so they must not show
    up as an error. Order is preserved only so a report reads the way the
    annotator wrote it.
    """
    seen: list[str] = []
    for label in labels or []:
        if not isinstance(label, str):
            continue
        key = label.strip().lower()
        if key and key not in seen:
            seen.append(key)
    return seen


def _ratio(numerator: int, denominator: int) -> float | None:
    """A ratio, or None when the denominator is zero.

    ``None`` is deliberately distinct from ``0.0``: "no opportunity to be right"
    is not the same claim as "wrong every time".
    """
    return round(numerator / denominator, 4) if denominator else None


@dataclass(frozen=True)
class LabelCounts:
    """Confusion counts for one label, or for the micro aggregate.

    Counted in **images**, not regions: a label is either present in an image's
    reference set or it is not. That is the multi-label convention, and it keeps
    per-label and micro numbers on the same footing.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        """How many images the reference actually carries this label on."""
        return self.tp + self.fn

    @property
    def predicted(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        """F1 computed from the raw counts, never from the rounded P and R.

        Deriving it from :attr:`precision` and :attr:`recall` propagates their
        rounding: on the Dev-30 counts (tp=11, fp=9, fn=48) that yields 0.2784
        where the exact value is 22/79 = 0.2785 — a drift in the fourth decimal
        that would silently disagree with the frozen baseline.
        """
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return None
        return round(2 * self.tp / denominator, 4)

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "support": self.support, "predicted": self.predicted,
                "precision": self.precision, "recall": self.recall,
                "f1": self.f1}


@dataclass(frozen=True)
class Evaluation:
    """The full metric set for one slice of the annotated data."""

    images: int
    annotated: int
    answered: int
    abstained: int
    micro: LabelCounts
    per_label: dict[str, LabelCounts] = field(default_factory=dict)
    labels_without_support: list[str] = field(default_factory=list)
    # The (predicted, actual) pairs for the answered images only. Internal: it
    # exists so the selective metrics can be recomputed without re-walking the
    # entries, and it is excluded from ``to_dict`` and from equality.
    answered_pairs: list[tuple[set[str], set[str]]] = field(
        default_factory=list, repr=False, compare=False)

    # ── the headline numbers ──────────────────────────────────────────────

    @property
    def coverage(self) -> float | None:
        """Fraction of annotated images the model answered."""
        return _ratio(self.answered, self.annotated)

    @property
    def abstention_rate(self) -> float | None:
        """Fraction of annotated images the model withheld on where it should not."""
        return _ratio(self.abstained, self.annotated)

    @property
    def precision(self) -> float | None:
        """Micro precision. Identical to :attr:`selective_precision` by construction."""
        return self.micro.precision

    @property
    def recall(self) -> float | None:
        """Micro recall over the whole set — every missed label counts, including
        the ones lost to abstention."""
        return self.micro.recall

    @property
    def f1(self) -> float | None:
        return self.micro.f1

    @property
    def selective_precision(self) -> float | None:
        """Micro precision. An abstention cannot create a false positive, so this
        necessarily equals :attr:`precision`; it is named separately so a report
        can state the selective view without implying a different number."""
        return self.micro.precision

    @property
    def selective_recall(self) -> float | None:
        """Micro recall computed only on the answered images.

        Optimistic by construction: the labels lost to abstention are excluded
        from the denominator. Read it next to :attr:`recall`, never instead.
        """
        answered = self._answered_counts()
        return _ratio(answered.tp, answered.tp + answered.fn)

    @property
    def risk_at_coverage(self) -> float | None:
        """1 − selective precision: the error rate a user actually experiences
        on the images the model chose to answer."""
        precision = self.selective_precision
        return None if precision is None else round(1 - precision, 4)

    @property
    def macro_f1(self) -> float | None:
        return self._macro("f1")

    @property
    def macro_precision(self) -> float | None:
        return self._macro("precision")

    @property
    def macro_recall(self) -> float | None:
        return self._macro("recall")

    # ── internals ─────────────────────────────────────────────────────────

    def _macro(self, attribute: str) -> float | None:
        """Mean over labels with support > 0.

        Two conventions are fixed here, because both change the number:

        * A label the reference never uses has no recall to average, so it is
          excluded. Including it as 0.0 would punish the model for a label
          nobody asked about. Excluded labels are listed in
          ``labels_without_support`` so the choice is visible.
        * A label that *does* have support but was never predicted is a total
          miss, not an undefined score. Its precision/recall/F1 are counted as
          0.0 rather than skipped — skipping it would let a model raise its
          macro score by silently ignoring the hardest label, which is the
          exact failure this module exists to expose.
        """
        values: list[float] = []
        for counts in self.per_label.values():
            if counts.support == 0:
                continue
            value = getattr(counts, attribute)
            values.append(0.0 if value is None else value)
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def _answered_counts(self) -> LabelCounts:
        """Micro counts recomputed over answered images only.

        Every term is derived from the answered pairs alone, so this cannot
        silently inherit an aggregate that was computed over all images.
        """
        tp = sum(len(predicted & actual) for predicted, actual in self.answered_pairs)
        fp = sum(len(predicted - actual) for predicted, actual in self.answered_pairs)
        fn = sum(len(actual - predicted) for predicted, actual in self.answered_pairs)
        return LabelCounts(tp=tp, fp=fp, fn=fn)

    def to_dict(self) -> dict:
        return {
            "images": self.images,
            "annotated": self.annotated,
            "answered": self.answered,
            "abstained": self.abstained,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "selective_precision": self.selective_precision,
            "selective_recall": self.selective_recall,
            "risk_at_coverage": self.risk_at_coverage,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "labels": self.micro.to_dict(),
            "per_label": {name: counts.to_dict()
                          for name, counts in sorted(self.per_label.items())},
            "labels_without_support": self.labels_without_support,
        }


def evaluate(entries: list[dict], *,
             reference_key: str = DEFAULT_REFERENCE_KEY,
             prediction_key: str = DEFAULT_PREDICTION_KEY) -> Evaluation:
    """Compute every metric for one list of annotation entries.

    Entries without a reference are skipped: an image nobody labelled is not
    evidence, and counting it as an abstention would manufacture one.
    """
    annotated = [entry for entry in entries
                 if isinstance(entry, dict) and entry.get(reference_key) is not None]

    micro_tp = micro_fp = micro_fn = 0
    per_label: dict[str, LabelCounts] = {}
    answered = abstained = 0
    answered_pairs: list[tuple[set[str], set[str]]] = []

    for entry in annotated:
        predicted = set(normalise_labels(entry.get(prediction_key)))
        actual = set(normalise_labels(entry.get(reference_key)))

        hit = predicted & actual
        micro_tp += len(hit)
        micro_fp += len(predicted - actual)
        micro_fn += len(actual - predicted)

        if predicted:
            answered += 1
            answered_pairs.append((predicted, actual))
        elif actual:
            # Withheld where the reference found something. An image where both
            # are empty is correctly silent and is not counted here.
            abstained += 1

        for label in predicted | actual:
            counts = per_label.get(label, LabelCounts())
            if label in predicted and label in actual:
                counts = LabelCounts(counts.tp + 1, counts.fp, counts.fn)
            elif label in predicted:
                counts = LabelCounts(counts.tp, counts.fp + 1, counts.fn)
            else:
                counts = LabelCounts(counts.tp, counts.fp, counts.fn + 1)
            per_label[label] = counts

    evaluation = Evaluation(
        images=len(entries),
        annotated=len(annotated),
        answered=answered,
        abstained=abstained,
        micro=LabelCounts(tp=micro_tp, fp=micro_fp, fn=micro_fn),
        per_label=per_label,
        labels_without_support=sorted(name for name, counts in per_label.items()
                                      if counts.support == 0),
        answered_pairs=answered_pairs,
    )
    return evaluation


def stratify(entries: list[dict], key: str = "mode", **kwargs) -> dict[str, Evaluation]:
    """Evaluate each value of ``key`` separately, plus the whole set under ``overall``.

    Reporting only the aggregate is how the Dev-30 leaf subset hid behind a
    strong herb number: the two modes observe different subjects and must be
    readable apart.
    """
    buckets: dict[str, list[dict]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            buckets.setdefault(str(entry.get(key)), []).append(entry)
    result = {name: evaluate(bucket, **kwargs) for name, bucket in sorted(buckets.items())}
    result["overall"] = evaluate(entries, **kwargs)
    return result


def risk_coverage_curve(entries: list[dict], *,
                        reference_key: str = DEFAULT_REFERENCE_KEY,
                        prediction_key: str = DEFAULT_PREDICTION_KEY,
                        confidence_key: str = DEFAULT_CONFIDENCE_KEY) -> list[dict] | None:
    """Error rate as a function of how much the model chooses to answer.

    Images are ranked by a per-image confidence, then answered in that order:
    at coverage *k/n* the risk is the error rate over the top *k* images. A model
    whose confidence is informative trades coverage for accuracy along this
    curve; one whose confidence is noise does not.

    Returns ``None`` when any annotated entry lacks a numeric confidence, rather
    than inventing a ranking. The current Dev-30 worksheet carries no per-image
    confidence, so this is not yet computable on it — that is a data gap, and it
    is reported as one.
    """
    annotated = [entry for entry in entries
                 if isinstance(entry, dict) and entry.get(reference_key) is not None]
    if not annotated:
        return None
    confidences = []
    for entry in annotated:
        value = entry.get(confidence_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        confidences.append(float(value))

    order = sorted(range(len(annotated)), key=lambda index: confidences[index], reverse=True)
    points: list[dict] = []
    tp = fp = fn = 0
    for rank, index in enumerate(order, start=1):
        entry = annotated[index]
        predicted = set(normalise_labels(entry.get(prediction_key)))
        actual = set(normalise_labels(entry.get(reference_key)))
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        precision = _ratio(tp, tp + fp)
        points.append({
            "coverage": round(rank / len(annotated), 4),
            "risk": None if precision is None else round(1 - precision, 4),
            "precision": precision,
        })
    return points


__all__ = ["DEFAULT_CONFIDENCE_KEY", "DEFAULT_PREDICTION_KEY", "DEFAULT_REFERENCE_KEY",
           "Evaluation", "LabelCounts", "evaluate", "normalise_labels",
           "risk_coverage_curve", "stratify"]
