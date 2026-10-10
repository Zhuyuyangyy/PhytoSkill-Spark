"""BM25 ranking over the vendored corpus.

Why this module exists: the previous ranking was raw bigram overlap,
``|query ∩ document| / |query|``. That measure has two blind spots that show up
immediately on this corpus:

* **No inverse document frequency.** 甘草 occurs in a large share of chunks, so a
  query on it returns every one of them at the same score, with no way to prefer
  the chunk that also carries the *other* herb named in the query (e.g. 甘遂).
  BM25 weights a term by how much it discriminates.
* **No length normalisation.** A long chunk that happens to contain all the
  query bigrams scores 1.0, the same as a short chunk built around exactly those
  bigrams. BM25 discounts term frequency as document length grows.

The tokeniser is deliberately mixed-script: CJK runs become overlapping
character bigrams (Chinese has no word delimiters, and this sidesteps the
SQLite FTS5 ``unicode61`` / ``trigram`` failure modes entirely), while Latin and
digit runs become lowercased whole words so that identifiers such as
``herb_interactions`` or ``leaf_yellowing`` stay intact.

BM25 here is a *ranking* function, not a relevance oracle. It still carries no
claim about factual certainty, and it is still fully deterministic and offline.
"""

from __future__ import annotations

import math
import re

# Standard BM25 constants. These are also the values SQLite FTS5 hard-codes, so
# a SQLite-backed implementation would be directly comparable to this one.
K1 = 1.2
B = 0.75

# Fields and their relative weight. A match in the title is a stronger signal
# than the same match buried in the body; tags are explicit curated metadata and
# carry the most weight per token.
DEFAULT_WEIGHTS = {"title": 3.0, "content": 1.0, "tags": 2.0}

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LATIN = re.compile(r"[A-Za-z0-9_]+")


def tokenise(text: str) -> list[str]:
    """Split text into tokens: CJK runs as character bigrams, Latin runs as words.

    A single CJK character yields itself rather than nothing, so a one-character
    query is still searchable.
    """
    if not text:
        return []
    tokens: list[str] = []
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    tokens.extend(match.lower() for match in _LATIN.findall(text))
    return tokens


def _term_frequencies(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


class BM25Index:
    """A small, deterministic BM25 index over a list of fielded documents.

    ``documents`` is a list of ``{field_name: token list}``. Field weights are
    applied at score time, so the same index can be re-weighted cheaply.
    """

    def __init__(self, documents: list[dict[str, list[str]]], *,
                 k1: float = K1, b: float = B,
                 weights: dict[str, float] | None = None):
        if not documents:
            raise ValueError("BM25Index needs at least one document")
        self.k1 = k1
        self.b = b
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._fields = sorted({name for document in documents
                               for name in document
                               if name in self.weights})
        self._lengths: dict[str, list[int]] = {}
        self._average_length: dict[str, float] = {}
        self._document_frequency: dict[str, dict[str, int]] = {}
        self._frequencies: list[dict[str, dict[str, int]]] = []
        total = len(documents)
        for name in self._fields:
            lengths = [len(document.get(name) or []) for document in documents]
            self._lengths[name] = lengths
            self._average_length[name] = (sum(lengths) / total) if total else 0.0
        for document in documents:
            per_field = {name: _term_frequencies(document.get(name) or [])
                         for name in self._fields}
            self._frequencies.append(per_field)
            for name in self._fields:
                field_df = self._document_frequency.setdefault(name, {})
                for token in per_field[name]:
                    field_df[token] = field_df.get(token, 0) + 1
        self.document_count = total

    # ── scoring ───────────────────────────────────────────────────────────

    def _idf(self, field: str, token: str) -> float:
        """Probabilistic IDF, always positive, so a common term cannot dominate."""
        count = self.document_count
        df = self._document_frequency.get(field, {}).get(token, 0)
        return math.log(1.0 + (count - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], position: int) -> float:
        """BM25 score of document ``position`` for ``query_tokens``."""
        if not query_tokens:
            return 0.0
        total = 0.0
        for field in self._fields:
            weight = self.weights.get(field, 1.0)
            if weight == 0.0:
                continue
            length = self._lengths[field][position]
            average = self._average_length[field]
            # A field with no tokens has nothing to normalise against; skipping
            # it is correct rather than dividing by zero.
            if average <= 0.0 or length == 0:
                continue
            frequencies = self._frequencies[position][field]
            if not frequencies:
                continue
            denominator_base = self.k1 * (1.0 - self.b
                                         + self.b * length / average)
            field_total = 0.0
            for token in query_tokens:
                tf = frequencies.get(token)
                if not tf:
                    continue
                field_total += (self._idf(field, token) * tf * (self.k1 + 1.0)
                                / (tf + denominator_base))
            total += weight * field_total
        return total

    def search(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """Every document scored, highest first. Ties break on position."""
        scored = [(index, self.score(query_tokens, index))
                  for index in range(self.document_count)]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored


__all__ = ["B", "BM25Index", "DEFAULT_WEIGHTS", "K1", "tokenise"]
