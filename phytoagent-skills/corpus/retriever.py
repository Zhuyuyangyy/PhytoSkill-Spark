"""Offline retrieval over a local, versioned corpus copy of TCM-Mind-RAG data.

Design constraints, in order of importance:

* **The corpus is copied into this repository.** Nothing here imports, shells out
  to, or reads from ``D:\\ZYY Project\\TCM-Mind-RAG``. A copy with a recorded
  SHA-256 is the only thing consulted, so a result is reproducible and the
  upstream project stays an unmodified dependency.
* **No fabrication.** A query that matches nothing returns an empty result and
  the caller records the gap. The retriever never invents a citation, a page
  number, or a passage.
* **Retrieval is not evidence of truth.** ``retrieval_score`` is a lexical
  overlap measure. It carries no claim about factual certainty, and the corpus
  is TCM syndrome knowledge, not plant pathology.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from corpus.ranking import BM25Index, DEFAULT_WEIGHTS, tokenise


CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"

# Chinese text has no word delimiters; score on character bigrams instead.
_PUNCTUATION = re.compile(r"[\s，。、；：（）()\[\]【】,.;:<>\"'`~!?！？\-_/\\|]+")


@dataclass(frozen=True)
class CorpusChunk:
    evidence_id: str
    source_file: str
    source_kind: str
    title: str
    content: str
    line_start: int
    line_end: int
    species_tags: list[str]
    phenotype_tags: list[str]
    corpus_version: str

    def to_evidence(self, score: float, *, bm25_score: float = 0.0,
                    corpus_digest: str | None = None) -> dict:
        """Build the evidence record. ``score`` is overlap; ``bm25_score`` is rank.

        ``corpus_digest`` pins the corpus build a citation came from. Without it
        an evidence ID names a chunk but not which corpus snapshot produced it,
        which is not enough to reproduce a citation later.
        """
        return {
            "evidence_id": self.evidence_id,
            "source_kind": "local_corpus",
            "source_title": self.title,
            "source_file": self.source_file,
            "source_location": f"{self.source_file}:{self.line_start}-{self.line_end}",
            "content": self.content,
            "supports_phenotypes": list(self.phenotype_tags),
            "retrieval_score": round(score, 4),
            "bm25_score": round(bm25_score, 4),
            "corpus_version": self.corpus_version,
            "corpus_digest": corpus_digest,
        }


def _normalise(text: str) -> str:
    return _PUNCTUATION.sub("", text.lower())


def _bigrams(text: str) -> set[str]:
    normalised = _normalise(text)
    if len(normalised) < 2:
        return {normalised} if normalised else set()
    return {normalised[index:index + 2] for index in range(len(normalised) - 1)}


def _overlap(query: str, content: str) -> float:
    """Deterministic lexical overlap in [0, 1]; no embedding model is involved."""
    query_terms = _bigrams(query)
    if not query_terms:
        return 0.0
    content_terms = _bigrams(content)
    return len(query_terms & content_terms) / len(query_terms)


class LocalCorpus:
    """Read-only access to the vendored corpus index."""

    def __init__(self, corpus_dir: str | Path = CORPUS_DIR):
        self.corpus_dir = Path(corpus_dir).absolute()
        self.index_path = self.corpus_dir / "index.json"
        self._chunks: list[CorpusChunk] | None = None
        self._digest: str | None = None
        self._bm25: BM25Index | None = None

    # ── loading ───────────────────────────────────────────────────────────

    @property
    def chunks(self) -> list[CorpusChunk]:
        if self._chunks is None:
            self._chunks = self._load()
        return self._chunks

    def _load(self) -> list[CorpusChunk]:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Corpus index not found: {self.index_path}")
        document = json.loads(self.index_path.read_text(encoding="utf-8"))
        chunks: list[CorpusChunk] = []
        for entry in document.get("chunks", []):
            chunks.append(CorpusChunk(
                evidence_id=entry["evidence_id"],
                source_file=entry["source_file"],
                source_kind=entry["source_kind"],
                title=entry["title"],
                content=entry["content"],
                line_start=int(entry["line_start"]),
                line_end=int(entry["line_end"]),
                species_tags=list(entry.get("species_tags", [])),
                phenotype_tags=list(entry.get("phenotype_tags", [])),
                corpus_version=document.get("corpus_version", "unknown"),
            ))
        return chunks

    def digest(self) -> str:
        """SHA-256 of the corpus index, so a citation can name its corpus build."""
        if self._digest is None:
            self._digest = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
        return self._digest

    @property
    def version(self) -> str:
        if not self.chunks:
            return "unknown"
        return self.chunks[0].corpus_version

    # ── retrieval ─────────────────────────────────────────────────────────

    # ── BM25 index ────────────────────────────────────────────────────────

    @property
    def bm25(self) -> BM25Index:
        """BM25 over every chunk, built once and reused.

        The index is built over **all** chunks, not just the ones a given query
        matches, because inverse document frequency is a property of the whole
        corpus. Filtering happens afterwards, at scoring time.
        """
        if self._bm25 is None:
            documents = [{"title": tokenise(chunk.title),
                          "content": tokenise(chunk.content),
                          "tags": tokenise(" ".join(chunk.species_tags
                                                    + chunk.phenotype_tags))}
                         for chunk in self.chunks]
            self._bm25 = BM25Index(documents, weights=dict(DEFAULT_WEIGHTS))

        return self._bm25

    def search(self, query: str, *, species: str | None = None,
               phenotypes: list[str] | None = None, limit: int = 5,
               min_score: float = 0.05) -> list[dict]:
        """Return evidence records, best first. Empty is a valid, honest result.

        Two numbers are computed per candidate, and they do different jobs:

        * ``retrieval_score`` — lexical overlap in [0, 1]. This is the **gate**.
          ``min_score`` defaults to a non-zero threshold because a chunk with no
          overlap is not a match, and returning it would be a fabricated
          citation.
        * ``bm25_score`` — the BM25 score, used for **ranking**. It applies
          inverse document frequency and length normalisation, so a term that
          occurs everywhere and a term that occurs once are not treated alike.

        Keeping the gate and the ranker separate means the no-fabrication
        property is unchanged while ordering improves.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty query is required")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")
        wanted = set(phenotypes or [])
        digest = self.digest()
        candidates: list[tuple[float, int, CorpusChunk]] = []
        for position, chunk in enumerate(self.chunks):
            if species and chunk.species_tags and species not in chunk.species_tags:
                continue
            if wanted and not wanted.intersection(chunk.phenotype_tags):
                continue
            overlap = _overlap(query, f"{chunk.title} {chunk.content}")
            if overlap >= min_score:
                candidates.append((overlap, position, chunk))

        query_tokens = tokenise(query)
        ranked = [(overlap, self.bm25.score(query_tokens, position), chunk)
                  for overlap, position, chunk in candidates]
        ranked.sort(key=lambda item: (-item[1], item[0], item[2].evidence_id))
        return [chunk.to_evidence(overlap, bm25_score=bm25, corpus_digest=digest)
                for overlap, bm25, chunk in ranked[:limit]]

    def evidence_ids(self) -> list[str]:
        return sorted(chunk.evidence_id for chunk in self.chunks)

    def stats(self) -> dict:
        by_file: dict[str, int] = {}
        for chunk in self.chunks:
            by_file[chunk.source_file] = by_file.get(chunk.source_file, 0) + 1
        return {
            "corpus_version": self.version,
            "corpus_sha256": self.digest(),
            "chunks": len(self.chunks),
            "source_files": by_file,
            # Ranking is BM25 (inverse document frequency + length normalisation)
            # over character bigrams for CJK and whole words for Latin script.
            "ranker": "bm25_character_bigram",
            "ranker_constants": {"k1": self.bm25.k1, "b": self.bm25.b},
            "field_weights": dict(DEFAULT_WEIGHTS),
            "gate": "lexical_overlap",
            "embedding_model": "none",
            "vector_index": "none",
        }


def load_evidence_index_from_corpus(corpus: LocalCorpus) -> dict[str, dict]:
    """Build the evidence_id -> record index a ClaimAuditor checks against."""
    digest = corpus.digest()
    return {chunk.evidence_id: chunk.to_evidence(0.0, corpus_digest=digest)
            for chunk in corpus.chunks}
