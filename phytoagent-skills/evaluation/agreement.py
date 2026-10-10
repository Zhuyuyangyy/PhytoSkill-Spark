"""Inter-annotator agreement and disagreement resolution.

Single-annotator labels are human *reference* annotations, not a gold standard.
Until two people label the same images independently, no accuracy number can be
separated from one person's judgement — which is exactly the caveat the Dev-30
report has to carry today.

Why kappa and not raw agreement: raw agreement flatters a rare label. If a label
occurs on 10% of images, two annotators who both always say "absent" agree 90% of
the time while agreeing on nothing. Cohen's kappa subtracts the agreement
expected by chance, so a label nobody can agree on scores near zero however
common the default answer is.

``kappa`` is reported next to the raw agreement, never instead of it: kappa is
unstable on small samples and undefined when a label has no variance at all, and
the raw number is what a reader can sanity-check by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How a disagreement was settled. Naming the outcomes stops "resolved" from
# meaning "somebody decided something".
ACCEPT_REFERENCE = "accept_reference"
ACCEPT_SECOND = "accept_second"
ACCEPT_BOTH = "accept_both"
DROP = "drop"
DECISIONS = (ACCEPT_REFERENCE, ACCEPT_SECOND, ACCEPT_BOTH, DROP)


def cohen_kappa(first: list[bool], second: list[bool]) -> float | None:
    """Cohen's kappa for two raters over paired binary judgements.

    Returns ``None`` when the coefficient is undefined: no items, or no variance
    at all (both raters constant), where the chance-agreement term is 1 and the
    formula divides by zero. ``None`` is reported as "not measurable", never as
    0.0 — those are different findings.
    """
    if len(first) != len(second):
        raise ValueError("kappa needs equally long rating vectors")
    total = len(first)
    if total == 0:
        return None
    observed = sum(1 for a, b in zip(first, second) if a == b) / total
    first_positive = sum(first) / total
    second_positive = sum(second) / total
    expected = (first_positive * second_positive
                + (1 - first_positive) * (1 - second_positive))
    if abs(1.0 - expected) < 1e-12:
        return None
    return round((observed - expected) / (1 - expected), 4)


def _labels_by_image(payload: dict, key: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for image, record in (payload or {}).items():
        if isinstance(record, dict):
            values = record.get(key, record.get("labels", []))
        else:
            values = record
        result[str(image)] = {str(item).strip().lower() for item in (values or [])
                              if isinstance(item, str) and item.strip()}
    return result


def multilabel_agreement(first: dict, second: dict, *,
                         label_key: str = "labels") -> dict:
    """Per-label and macro agreement between two annotators.

    Each annotator is ``{image: [labels]}``. Only images both annotated are
    compared; images seen by one annotator are reported separately rather than
    silently counted as disagreement.
    """
    left = _labels_by_image(first, label_key)
    right = _labels_by_image(second, label_key)
    common = sorted(set(left) & set(right))
    if not common:
        return {"status": "no_overlap", "common_images": 0,
                "only_in_first": sorted(set(left) - set(right)),
                "only_in_second": sorted(set(right) - set(left)),
                "per_label": {}, "macro_kappa": None, "observed_agreement": None}

    vocabulary = sorted({label for image in common for label in (left[image] | right[image])})
    per_label: dict[str, dict] = {}
    kappas: list[float] = []
    agreement_pairs = agreements = 0
    for label in vocabulary:
        a = [label in left[image] for image in common]
        b = [label in right[image] for image in common]
        matches = sum(1 for x, y in zip(a, b) if x == y)
        agreement_pairs += len(common)
        agreements += matches
        kappa = cohen_kappa(a, b)
        if kappa is not None:
            kappas.append(kappa)
        per_label[label] = {
            "kappa": kappa,
            "observed_agreement": round(matches / len(common), 4),
            "support_first": sum(a),
            "support_second": sum(b),
        }

    return {
        "status": "measured",
        "common_images": len(common),
        "only_in_first": sorted(set(left) - set(right)),
        "only_in_second": sorted(set(right) - set(left)),
        "labels": len(vocabulary),
        "per_label": per_label,
        "macro_kappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
        "labels_with_undefined_kappa": sorted(
            name for name, record in per_label.items() if record["kappa"] is None),
        "observed_agreement": (round(agreements / agreement_pairs, 4)
                               if agreement_pairs else None),
        "reading": ("kappa 与原始一致率并列报告：原始一致率会被稀有标签抬高，"
                    "kappa 扣掉了随机一致的成分。"),
    }


@dataclass
class DisagreementItem:
    """One image's disagreement, and the decision that settles it."""

    image: str
    only_in_first: list[str] = field(default_factory=list)
    only_in_second: list[str] = field(default_factory=list)
    decision: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {"image": self.image, "only_in_first": self.only_in_first,
                "only_in_second": self.only_in_second,
                "decision": self.decision, "note": self.note}


def resolution_queue(first: dict, second: dict, *,
                     label_key: str = "labels") -> list[DisagreementItem]:
    """The images the two annotators do not agree on, ready to be adjudicated."""
    left = _labels_by_image(first, label_key)
    right = _labels_by_image(second, label_key)
    queue: list[DisagreementItem] = []
    for image in sorted(set(left) & set(right)):
        only_first = sorted(left[image] - right[image])
        only_second = sorted(right[image] - left[image])
        if only_first or only_second:
            queue.append(DisagreementItem(image=image, only_in_first=only_first,
                                          only_in_second=only_second))
    return queue


def adjudicate(first: dict, second: dict, resolutions: dict[str, str], *,
               label_key: str = "labels") -> dict:
    """Build one settled label set per image from an adjudication map.

    ``resolutions`` maps an image to one of :data:`DECISIONS`. Agreement needs no
    decision. An image with a disagreement and no decision is an error, not a
    default — quietly picking one annotator is how a disagreement disappears
    without being resolved.
    """
    left = _labels_by_image(first, label_key)
    right = _labels_by_image(second, label_key)
    settled: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for image in sorted(set(left) & set(right)):
        if left[image] == right[image]:
            settled[image] = sorted(left[image])
            continue
        decision = resolutions.get(image)
        if decision is None:
            unresolved.append(image)
            continue
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r} for {image!r}")
        if decision == ACCEPT_REFERENCE:
            settled[image] = sorted(left[image])
        elif decision == ACCEPT_SECOND:
            settled[image] = sorted(right[image])
        elif decision == ACCEPT_BOTH:
            settled[image] = sorted(left[image] | right[image])
        else:
            settled[image] = sorted(left[image] & right[image])
    if unresolved:
        raise ValueError(f"disagreements without a recorded decision: {unresolved}")
    return settled


__all__ = ["ACCEPT_BOTH", "ACCEPT_REFERENCE", "ACCEPT_SECOND", "DECISIONS", "DROP",
           "DisagreementItem", "adjudicate", "cohen_kappa", "multilabel_agreement",
           "resolution_queue"]
