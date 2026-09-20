"""herbal_knowledge adapter: fixture mode by default, corpus mode when configured.

The two modes are deliberately separate:

* ``fixture`` is the published synthetic case. It answers exactly one input and
  refuses everything else, which is what makes the package's contract testable
  offline.
* ``corpus`` queries the vendored, hashed local corpus. It is selected
  explicitly by the caller, never silently, and it returns an empty evidence
  list when nothing matches instead of inventing a citation.
"""

from corpus.retriever import LocalCorpus
from sdk import BaseSkill, ContractError
from sdk.exceptions import UnsupportedModeError
from sdk.fixture_skill import FixtureSkill


class HerbalKnowledgeSkill(BaseSkill):
    """One narrowly scoped, contract-validated capability."""

    def __init__(self, package_dir, *, corpus: LocalCorpus | None = None):
        super().__init__(package_dir)
        self._corpus = corpus

    @property
    def corpus(self) -> LocalCorpus:
        if self._corpus is None:
            self._corpus = LocalCorpus()
        return self._corpus

    def run(self, payload: dict, *, mode: str) -> dict:
        if mode == "fixture":
            return self._run_fixture(payload)
        if mode == "corpus":
            return self._run_corpus(payload)
        raise UnsupportedModeError(f"herbal_knowledge does not support mode {mode!r}")

    # ── modes ─────────────────────────────────────────────────────────────

    def _run_fixture(self, payload: dict) -> dict:
        """Delegate to the sealed fixture; a mismatch is an explicit failure."""
        return FixtureSkill.run(self, payload, mode="fixture")

    def _run_corpus(self, payload: dict) -> dict:
        """Retrieve from the local vendored corpus, with locations and version."""
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ContractError("corpus mode requires a non-empty query")
        species = payload.get("species")
        evidence = self.corpus.search(query, species=species, limit=5)
        limitations = [
            "检索来自本地vendored语料，未调用Embedding模型或向量库；检索分数只反映字面重合度。",
            "本草药性与证型知识不能替代栽培病理证据，也不能据此确诊病害。",
            "语料为TCM-Mind-RAG知识数据的副本；未连接原项目的运行服务。",
        ]
        if not evidence:
            limitations.append("未检索到与问题有字面重合的条目；此处不给出来源，也不编造引用。")
        return {
            "status": "success",
            "case_id": payload.get("case_id", ""),
            "species": species,
            "provenance": {
                "data_origin": "local_corpus",
                "skill": "herbal_knowledge",
                "version": self.version,
                "model_called": False,
                "corpus_version": self.corpus.version,
                "corpus_sha256": self.corpus.digest(),
                "retriever": "deterministic_character_bigram_overlap",
            },
            "evidence": evidence,
            "limitations": limitations,
        }
