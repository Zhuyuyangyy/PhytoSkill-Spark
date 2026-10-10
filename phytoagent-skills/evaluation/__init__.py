"""Evaluation: whether the model is actually good, and on what evidence.

This layer exists because the project could report precision without being able
to report *coverage*. The Dev-30 leaf subset showed ``precision = 1.000`` — on
three answers out of fifteen. A number like that is not wrong, it is
uninterpretable, and the fix is structural rather than a matter of care:

* :mod:`evaluation.metrics` — set metrics *and* selective metrics, so abstention
  is a reported quantity rather than a hidden one.
* :mod:`evaluation.split` — dataset roles, a freeze fingerprint over everything
  that can change an answer, and a ledger that enforces the one-shot holdout
  rule.
* :mod:`evaluation.agreement` — inter-annotator agreement, so a single
  annotator's judgement is not silently promoted to a gold standard.

Nothing here calls a model, reads an image, or decides anything about plants. It
only measures, and it is careful about what a measurement can carry.
"""

from evaluation.agreement import (
    ACCEPT_BOTH,
    ACCEPT_REFERENCE,
    ACCEPT_SECOND,
    DROP,
    DisagreementItem,
    adjudicate,
    cohen_kappa,
    multilabel_agreement,
    resolution_queue,
)
from evaluation.metrics import (
    DEFAULT_CONFIDENCE_KEY,
    DEFAULT_PREDICTION_KEY,
    DEFAULT_REFERENCE_KEY,
    Evaluation,
    LabelCounts,
    evaluate,
    normalise_labels,
    risk_coverage_curve,
    stratify,
)
from evaluation.split import (
    DEFAULT_LEDGER,
    ROLES,
    ROLE_CALIBRATION,
    ROLE_DEVELOPMENT,
    ROLE_HOLDOUT,
    DatasetSplit,
    Freeze,
    FreezeError,
    HoldoutAlreadyRun,
    HoldoutLedger,
    freeze,
    ontology_fingerprint,
    parser_fingerprint,
    prompt_fingerprint,
    scorer_fingerprint,
    split_entries,
)

__all__ = [
    "ACCEPT_BOTH", "ACCEPT_REFERENCE", "ACCEPT_SECOND", "DEFAULT_CONFIDENCE_KEY",
    "DEFAULT_LEDGER", "DEFAULT_PREDICTION_KEY", "DEFAULT_REFERENCE_KEY", "DROP",
    "DatasetSplit", "DisagreementItem", "Evaluation", "Freeze", "FreezeError",
    "HoldoutAlreadyRun", "HoldoutLedger", "LabelCounts", "ROLES",
    "ROLE_CALIBRATION", "ROLE_DEVELOPMENT", "ROLE_HOLDOUT", "adjudicate",
    "cohen_kappa", "evaluate", "freeze", "multilabel_agreement",
    "normalise_labels", "ontology_fingerprint", "parser_fingerprint",
    "prompt_fingerprint", "resolution_queue", "risk_coverage_curve",
    "scorer_fingerprint", "split_entries", "stratify",
]
