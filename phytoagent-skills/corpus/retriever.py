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

    def to_evidence(self, score: float) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_kind": "local_corpus",
            "source_title": self.title,
            "source_file": self.source_file,
            "source_location": f"{self.source_file}:{self.line_start}-{self.line_end}",
            "content": self.content,
            "supports_phenotypes": list(self.phenotype_tags),
            "retrieval_score": round(score, 4),
            "corpus_version": self.corpus_version,
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

    def search(self, query: str, *, species: str | None = None,
               phenotypes: list[str] | None = None, limit: int = 5,
               min_score: float = 0.05) -> list[dict]:
        """Return evidence records, best first. Empty is a valid, honest result.

        ``min_score`` defaults to a non-zero threshold: a chunk with no lexical
        overlap is not a match, and returning it would be a fabricated citation.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty query is required")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")
        wanted = set(phenotypes or [])
        scored: list[tuple[float, CorpusChunk]] = []
        for chunk in self.chunks:
            if species and chunk.species_tags and species not in chunk.species_tags:
                continue
            if wanted and not wanted.intersection(chunk.phenotype_tags):
                continue
            score = _overlap(query, f"{chunk.title} {chunk.content}")
            if score >= min_score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].evidence_id))
        return [chunk.to_evidence(score) for score, chunk in scored[:limit]]

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
            "retriever": "deterministic_character_bigram_overlap",
            "embedding_model": "none",
            "vector_index": "none",
        }


def load_evidence_index_from_corpus(corpus: LocalCorpus) -> dict[str, dict]:
    """Build the evidence_id -> record index a ClaimAuditor checks against."""
    return {chunk.evidence_id: chunk.to_evidence(0.0) for chunk in corpus.chunks}
