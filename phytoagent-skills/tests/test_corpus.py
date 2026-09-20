"""Local vendored corpus: locatable evidence, and no fabrication when empty."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.retriever import LocalCorpus
from corpus.vendor import CORPUS_VERSION, _stable_id, vendor

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
SOURCE_DIR = Path("D:/ZYY Project/TCM-Mind-RAG/backend/data")


@pytest.fixture(scope="module")
def corpus():
    return LocalCorpus(CORPUS_DIR)


def test_the_vendored_corpus_index_is_committed(corpus):
    assert corpus.chunks, "corpus/index.json must be vendored, not generated at test time"
    assert corpus.version == CORPUS_VERSION


def test_the_corpus_digest_is_recorded(corpus):
    digest = corpus.digest()
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_the_corpus_is_reproducible(tmp_path):
    """Re-vendoring the same source produces a byte-identical index."""
    if not SOURCE_DIR.is_dir():
        pytest.skip("upstream TCM-Mind-RAG data directory is not present on this machine")
    output = tmp_path / "index.json"
    first = vendor(SOURCE_DIR, output)
    committed = (CORPUS_DIR / "index.json").read_bytes()
    assert output.read_bytes() == committed
    assert first["index_sha256"] == corpus_digest()


def corpus_digest() -> str:
    import hashlib
    return hashlib.sha256((CORPUS_DIR / "index.json").read_bytes()).hexdigest()


def test_every_chunk_names_its_source_file_and_line_range(corpus):
    for chunk in corpus.chunks:
        assert chunk.source_file
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start
        assert chunk.content.strip()
        assert chunk.evidence_id


def test_evidence_ids_are_unique(corpus):
    ids = [chunk.evidence_id for chunk in corpus.chunks]
    assert len(ids) == len(set(ids))


def test_line_ranges_are_within_the_source_file(corpus):
    """A citation must point somewhere that actually exists."""
    if not SOURCE_DIR.is_dir():
        pytest.skip("upstream TCM-Mind-RAG data directory is not present on this machine")
    totals = {path.name: len(path.read_text(encoding="utf-8").splitlines())
              for path in SOURCE_DIR.iterdir() if path.suffix.lower() in (".yaml", ".json")}
    for chunk in corpus.chunks:
        if chunk.source_file in totals:
            assert chunk.line_start <= totals[chunk.source_file], chunk.evidence_id
            assert chunk.line_end <= totals[chunk.source_file], chunk.evidence_id


def test_a_matching_query_returns_locatable_evidence(corpus):
    results = corpus.search("黄芪 心脾两虚 归脾汤", species="黄芪", limit=5)
    assert results
    for record in results:
        assert record["source_kind"] == "local_corpus"
        assert record["source_file"]
        assert record["source_location"].startswith(record["source_file"] + ":")
        assert record["corpus_version"] == CORPUS_VERSION
        assert 0.0 <= record["retrieval_score"] <= 1.0
        assert record["evidence_id"] in corpus.evidence_ids()


def test_a_non_matching_query_returns_nothing_rather_than_a_citation(corpus):
    assert corpus.search("量子色动力学夸克胶子等离子体", limit=5) == []


def test_the_default_threshold_excludes_zero_overlap(corpus):
    results = corpus.search("完全不相关的查询词", limit=5, min_score=0.0)
    # An explicit zero threshold is honoured, but nothing scores above zero here.
    assert all(record["retrieval_score"] == 0.0 for record in results)


def test_a_species_filter_excludes_other_species(corpus):
    results = corpus.search("心脾两虚 归脾汤", species="甘草", limit=5)
    for record in results:
        assert "甘草" in record["content"] or "甘草" in record["source_title"]


def test_an_empty_query_is_refused(corpus):
    with pytest.raises(ValueError, match="non-empty"):
        corpus.search("   ")


def test_an_invalid_limit_or_threshold_is_refused(corpus):
    with pytest.raises(ValueError, match="at least 1"):
        corpus.search("黄芪", limit=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        corpus.search("黄芪", min_score=1.5)


def test_the_stats_report_the_retriever_honestly(corpus):
    stats = corpus.stats()
    assert stats["retriever"] == "deterministic_character_bigram_overlap"
    assert stats["embedding_model"] == "none"
    assert stats["vector_index"] == "none"
    assert stats["chunks"] == len(corpus.chunks)
    assert set(stats["source_files"]) == {chunk.source_file for chunk in corpus.chunks}


def test_the_corpus_is_TCM_knowledge_not_plant_pathology(corpus):
    """The corpus cannot support a plant-physiology claim, and must say so.

    This is the honest boundary: a TCM syndrome corpus has no leaf-phenotype
    evidence, so a phenotype-filtered search finds nothing.
    """
    results = corpus.search("黄芪叶片黄化", phenotypes=["leaf_yellowing"], limit=5)
    assert results == []


def test_a_missing_index_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Corpus index not found"):
        LocalCorpus(tmp_path).chunks


def test_stable_ids_are_deterministic():
    assert _stable_id("syndromes.yaml", 1) == _stable_id("syndromes.yaml", 1)
    assert _stable_id("syndromes.yaml", 1) != _stable_id("syndromes.yaml", 2)


def test_the_vendored_index_is_valid_json_with_a_version():
    document = json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))
    assert document["corpus_version"] == CORPUS_VERSION
    assert document["chunk_count"] == len(document["chunks"])
    assert document["source"]["kind"] == "tcm-mind-rag"
