"""Golden-set evaluation for the corpus retriever and the grounding contract.

Why this exists: the research this layer was built from concluded that a small
team earns trust by measuring a few things honestly and refusing to over-claim
the rest. This module is the *measurement* half. It deliberately does three
things and no more:

1. **Ranking metrics on the retriever** — recall@k, MRR and nDCG@k against a
   small set of adjudicated queries.
2. **Refusal-policy conformance** — does the pipeline abstain exactly on the
   cases where it must?
3. **Deterministic grounding proxies** — properties checkable by code with **no
   new labels**: does every claim cite at least one ID, does every cited ID
   resolve, and does a claim silently mix evidence layers?

What it explicitly does NOT do is judge whether a cited record *entails* the
claim. That is entailment, it is not machine-checkable on pharmacopoeia text,
and no number here should be presented as if it were. See ``docs/evaluation.md``.

The honesty rule: a golden case with no adjudicated relevant IDs is **not**
scored. It is counted and reported as unlabelled, because silently treating an
unlabelled case as "not relevant" would manufacture a metric.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

# Evidence layers. An ID must declare which layer produced it, so that a
# photograph-derived observation can never silently inherit a monograph's
# authority (and vice versa).
LAYER_OBSERVATION = "observation"
LAYER_CORPUS = "corpus"
LAYER_UNKNOWN = "unknown"


def evidence_layer(evidence_id: str) -> str:
    """Infer the evidence layer from an ID's prefix.

    This is a convention check, not a lookup: the point is that a mixed-layer
    claim should be visible, and an ID that matches no known prefix is reported
    as unknown rather than assumed to be safe.
    """
    if not isinstance(evidence_id, str) or not evidence_id:
        return LAYER_UNKNOWN
    lowered = evidence_id.lower()
    if lowered.startswith("obs"):
        return LAYER_OBSERVATION
    if lowered.startswith("corpus"):
        return LAYER_CORPUS
    return LAYER_UNKNOWN


@dataclass(frozen=True)
class GoldenCase:
    """One adjudicated query.

    ``relevant_ids`` empty means **not yet adjudicated**, not "nothing is
    relevant". ``expect_refusal`` marks cases whose correct behaviour is to
    abstain (out-of-scope or diagnostic prompts).
    """

    case_id: str
    query: str
    species: str | None = None
    phenotypes: tuple[str, ...] = ()
    relevant_ids: tuple[str, ...] = ()
    expect_refusal: bool = False
    notes: str = ""

    @property
    def adjudicated(self) -> bool:
        return bool(self.relevant_ids) or self.expect_refusal

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "species": self.species,
            "phenotypes": list(self.phenotypes),
            "relevant_ids": list(self.relevant_ids),
            "expect_refusal": self.expect_refusal,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GoldenCase":
        if not isinstance(payload, dict):
            raise ValueError("a golden case must be an object")
        case_id = payload.get("case_id")
        query = payload.get("query")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("a golden case needs a case_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"golden case {case_id!r} needs a non-empty query")
        return cls(
            case_id=case_id,
            query=query,
            species=payload.get("species"),
            phenotypes=tuple(payload.get("phenotypes") or ()),
            relevant_ids=tuple(payload.get("relevant_ids") or ()),
            expect_refusal=bool(payload.get("expect_refusal", False)),
            notes=str(payload.get("notes", "")),
        )


@dataclass
class GoldenSet:
    name: str = "unnamed"
    cases: list[GoldenCase] = field(default_factory=list)
    adjudicator: str = "unrecorded"
    frozen_at: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "GoldenSet":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [GoldenCase.from_dict(entry)
                 for entry in document.get("cases", [])]
        seen: set[str] = set()
        for case in cases:
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id {case.case_id!r}")
            seen.add(case.case_id)
        return cls(name=document.get("name", "unnamed"), cases=cases,
                   adjudicator=document.get("adjudicator", "unrecorded"),
                   frozen_at=document.get("frozen_at", ""))

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "adjudicator": self.adjudicator,
            "frozen_at": self.frozen_at,
            "cases": [case.to_dict() for case in self.cases],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    @property
    def adjudicated(self) -> list[GoldenCase]:
        return [case for case in self.cases if case.adjudicated]

    @property
    def unlabelled(self) -> list[GoldenCase]:
        return [case for case in self.cases if not case.adjudicated]

    def stats(self) -> dict:
        refusal_cases = sum(1 for case in self.cases if case.expect_refusal)
        return {
            "name": self.name,
            "adjudicator": self.adjudicator,
            "cases": len(self.cases),
            "adjudicated": len(self.adjudicated),
            "unlabelled": len(self.unlabelled),
            "refusal_cases": refusal_cases,
        }


# ── retrieval metrics ────────────────────────────────────────────────────────


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def evaluate_retrieval(corpus, golden: GoldenSet, *, k: int = 5,
                       min_score: float = 0.05) -> dict:
    """Recall@k, MRR and nDCG@k over the adjudicated cases.

    Cases with no adjudicated relevant IDs are excluded and counted. Nothing is
    inferred about them.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scored_cases: list[dict] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for case in golden.adjudicated:
        if case.expect_refusal or not case.relevant_ids:
            # A refusal case has no retrieval target; it is scored by
            # evaluate_refusal, not here.
            continue
        results = corpus.search(case.query, species=case.species,
                                phenotypes=list(case.phenotypes), limit=k,
                                min_score=min_score)
        returned = [record["evidence_id"] for record in results]
        relevant = set(case.relevant_ids)
        hits = [1.0 if evidence_id in relevant else 0.0 for evidence_id in returned]
        found = sum(hits)
        recall = found / len(relevant) if relevant else 0.0
        first_rank = next((index + 1 for index, value in enumerate(hits) if value), None)
        # IDCG is the ceiling for this query: the best achievable ordering would
        # put min(k, |relevant|) relevant documents at the top.
        idcg = _dcg([1.0] * int(min(k, len(relevant))))
        ndcgs.append(_dcg(hits) / idcg if idcg > 0 else 0.0)
        recalls.append(recall)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        scored_cases.append({
            "case_id": case.case_id,
            "recall_at_k": round(recall, 4),
            "reciprocal_rank": round(reciprocal_ranks[-1], 4),
            "missed_ids": sorted(relevant - set(returned)),
            "returned": returned,
        })
    return {
        "k": k,
        "min_score": min_score,
        "cases_scored": len(scored_cases),
        "cases_unlabelled": len(golden.unlabelled),
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None,
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else None,
        "per_case": scored_cases,
        "caveat": ("Only adjudicated, non-refusal cases are scored. An empty or "
                   "unlabelled golden set yields None, not 0.0."),
    }


# ── deterministic grounding proxies ───────────────────────────────────────────


def check_grounding(claims: list[dict],
                    evidence_index: dict[str, dict] | None = None) -> dict:
    """Structural grounding checks that need no new labels.

    Checks the machine-checkable half: every claim carries at least one ID, and
    every ID resolves to a record. It also reports evidence-layer mixing, so a
    claim cannot quietly combine an observation with a monograph citation.

    It does **not** check entailment. A passing report means the citations are
    well-formed and resolvable — nothing more.
    """
    index = evidence_index or {}
    total = 0
    uncited: list[str] = []
    unresolved: list[str] = []
    mixed_layer: list[str] = []
    layer_counts: dict[str, int] = {}
    for entry in claims or []:
        if not isinstance(entry, dict):
            unresolved.append("<non-object claim>")
            continue
        total += 1
        text = entry.get("text")
        label = text if isinstance(text, str) else repr(entry)
        raw_ids = entry.get("evidence_ids")
        ids = [item for item in raw_ids
               if isinstance(item, str) and item] if isinstance(raw_ids, list) else []
        if not ids:
            uncited.append(label)
            continue
        for evidence_id in ids:
            if evidence_id not in index:
                unresolved.append(evidence_id)
        layers = {evidence_layer(evidence_id) for evidence_id in ids}
        for layer in layers:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(layers) > 1:
            mixed_layer.append(label)
    cited = total - len(uncited)
    return {
        "claims": total,
        "cited_claims": cited,
        "citation_coverage": round(cited / total, 4) if total else None,
        "uncited_claims": uncited,
        "unresolved_ids": sorted(set(unresolved)),
        "mixed_layer_claims": mixed_layer,
        "layers_used": layer_counts,
        "checks_performed": [
            "every claim cites at least one evidence id",
            "every cited id resolves to a record in the supplied index",
            "no claim mixes evidence layers without declaring it",
        ],
        "checks_not_performed": [
            "entailment: whether the cited record actually supports the claim",
            "calibration: whether the refusal gate fires at the right threshold",
        ],
    }


def evaluate_refusal(cases: list[GoldenCase],
                     run_pipeline) -> dict:
    """Does the pipeline abstain exactly on the cases that require abstention?

    ``run_pipeline(case) -> bool`` returns True when the pipeline abstained.
    A false positive (abstaining when it should answer) and a false negative
    (answering when it should abstain) are counted separately, because they have
    different costs.
    """
    if not callable(run_pipeline):
        raise ValueError("run_pipeline must be callable")
    correct = wrong_abstain = wrong_answer = 0
    details: list[dict] = []
    for case in cases:
        abstained = bool(run_pipeline(case))
        if case.expect_refusal and abstained:
            correct += 1
            outcome = "correct_abstention"
        elif not case.expect_refusal and not abstained:
            correct += 1
            outcome = "correct_answer"
        elif case.expect_refusal and not abstained:
            wrong_answer += 1
            outcome = "FAILED_TO_REFUSE"
        else:
            wrong_abstain += 1
            outcome = "ABSTAINED_UNNECESSARILY"
        details.append({"case_id": case.case_id, "outcome": outcome})
    evaluated = correct + wrong_abstain + wrong_answer
    return {
        "cases": evaluated,
        "correct": correct,
        "failed_to_refuse": wrong_answer,
        "abstained_unnecessarily": wrong_abstain,
        "conformance": round(correct / evaluated, 4) if evaluated else None,
        "per_case": details,
    }


__all__ = ["GoldenCase", "GoldenSet", "LAYER_CORPUS", "LAYER_OBSERVATION",
           "LAYER_UNKNOWN", "check_grounding", "evaluate_refusal",
           "evaluate_retrieval", "evidence_layer"]
