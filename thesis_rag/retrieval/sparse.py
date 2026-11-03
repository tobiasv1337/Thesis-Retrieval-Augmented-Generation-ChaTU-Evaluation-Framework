"""CandidateSource backed by verified language-specific local BM25 shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thesis_rag.indexing.catalog import (
    SUPPORTED_INDEX_LANGUAGES,
    ChunkDocumentCatalog,
)
from thesis_rag.indexing.manifests import RetrievalIndexManifest
from thesis_rag.indexing.sparse import SparseIndex, SparseIndexBuilder
from thesis_rag.interfaces import CorpusSnapshot, RagQuery
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateMatch
from thesis_rag.retrieval.config import (
    LanguagePolicy,
    RepresentationConfig,
    SparseSourceConfig,
)
from thesis_rag.retrieval.fusion import reciprocal_rank_contribution
from thesis_rag.retrieval.protocols import CandidateScope, ComponentRuntimeIdentity
from thesis_rag.schema import hash_file, hash_jsonable


SPARSE_RETRIEVAL_VERSION = "bm25_candidate_retrieval_v2"


class SparseRetrievalError(ValueError):
    """Raised when the Sparse component is not ready or receives invalid input."""


@dataclass(frozen=True)
class _ShardHit:
    row: int
    rank: int
    score: float
    language: str


class SparseCandidateSource:
    """BM25 retrieval with RRF across incomparable language-shard scores."""

    def __init__(
        self,
        *,
        config: SparseSourceConfig,
        representation: RepresentationConfig,
        language_policy: LanguagePolicy,
        catalog: ChunkDocumentCatalog,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.language_policy = language_policy
        self.catalog = catalog
        self.builder = SparseIndexBuilder(
            config=config,
            representation=representation,
            catalog=catalog,
            repo_root=repo_root,
        )
        self._index: SparseIndex | None = None
        self._identity: ComponentRuntimeIdentity | None = None

    @property
    def component_id(self) -> str:
        return self.config.component_id

    def build_index(self, *, force: bool = False) -> RetrievalIndexManifest:
        return self.builder.build(force=force)

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        self.catalog.load(expected_corpus_hash=corpus.corpus_hash)
        self.catalog.verify_unchanged()
        index = SparseIndex.load(builder=self.builder)
        manifest_path = self.builder.output_dir / "index_manifest.json"
        retrieval_semantics = {
            "implementation": SPARSE_RETRIEVAL_VERSION,
            "language_policy": self.language_policy,
            "zero_score_policy": "exclude",
            "cross_shard_fusion": (
                "reciprocal_rank_fusion"
                if self.language_policy == "cross_lingual"
                else "not_applicable"
            ),
            "cross_shard_rrf_k": self.config.cross_shard_rrf_k,
        }
        self._index = index
        self._identity = ComponentRuntimeIdentity(
            component_id=self.component_id,
            corpus_hash=corpus.corpus_hash,
            identity_hashes={
                "index_manifest": hash_file(manifest_path),
                "index_files": index.manifest.aggregate_file_hash,
                "sparse_semantics": index.manifest.semantic_config_hash,
                "sparse_retrieval": hash_jsonable(retrieval_semantics),
            },
            metadata={
                "index_id": index.manifest.index_id,
                "index_kind": "sparse",
                "index_path": str(manifest_path),
                "language_policy": self.language_policy,
                "language_counts": self.catalog.identity.language_counts,
                "query_language_view": (
                    self.catalog.identity.query_language_view.to_dict()
                    if self.catalog.identity.query_language_view is not None
                    else None
                ),
                "build_duration_ms": index.manifest.build_duration_ms,
                "index_byte_size": index.manifest.total_byte_size,
                "cross_shard_rrf_k": self.config.cross_shard_rrf_k,
                "retrieval_semantics": retrieval_semantics,
            },
        )

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self._identity is None:
            raise SparseRetrievalError("Sparse component has not been loaded")
        return self._identity

    def search(
        self,
        query: RagQuery,
        *,
        limit: int,
        scope: CandidateScope | None = None,
    ) -> CandidateBatch:
        if self._index is None:
            raise SparseRetrievalError("Sparse component has not been loaded")
        if limit <= 0:
            raise SparseRetrievalError("Sparse retrieval limit must be positive")
        if query.language not in SUPPORTED_INDEX_LANGUAGES:
            raise SparseRetrievalError(
                f"unsupported query language: {query.language}",
            )
        if self.language_policy in {
            "same_language_only",
            "query_language_complete",
        }:
            hits, eligible_count = self._rank_shard(
                query.text,
                language=query.language,
                limit=limit,
                scope=scope,
            )
            return CandidateBatch(
                matches=[
                    self._direct_match(hit, query_language=query.language)
                    for hit in hits
                ],
                exhausted=eligible_count <= limit,
            )

        shard_results = [
            self._rank_shard(
                query.text,
                language=language,
                limit=limit,
                scope=scope,
            )
            for language in SUPPORTED_INDEX_LANGUAGES
        ]
        hits = [hit for ranking, _ in shard_results for hit in ranking]
        scored = [
            (
                reciprocal_rank_contribution(
                    hit.rank,
                    rank_constant=self.config.cross_shard_rrf_k,
                ),
                hit,
            )
            for hit in hits
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                self.catalog.documents[item[1].row].target_id,
            ),
        )
        selected = scored[:limit]
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=self.catalog.documents[hit.row],
                    score=rrf_score,
                    score_name="cross_shard_rrf",
                    transition_reason="cross_language_sparse_rrf",
                    metadata={
                        "document_language": hit.language,
                        "query_language": query.language,
                        "language_policy": self.language_policy,
                        "native_bm25_score": hit.score,
                        "native_shard_rank": hit.rank,
                        "rank_constant": self.config.cross_shard_rrf_k,
                        "rrf_contribution": rrf_score,
                    },
                )
                for rrf_score, hit in selected
            ],
            exhausted=sum(count for _, count in shard_results) <= limit,
        )

    def close(self) -> None:
        self._index = None
        self._identity = None

    def _rank_shard(
        self,
        query_text: str,
        *,
        language: str,
        limit: int,
        scope: CandidateScope | None,
    ) -> tuple[list[_ShardHit], int]:
        assert self._index is not None
        shard = self._index.shards[language]
        tokens = self.builder.tokenizer.tokenize(query_text, language=language)
        scores = shard.scores(tokens)
        eligible = np.flatnonzero(scores > 0.0).astype(np.int64, copy=False)
        if scope is not None:
            allows = scope.matcher()
            eligible = np.asarray(
                [
                    local_row
                    for local_row in eligible
                    if allows(
                        self.catalog.documents[int(shard.rows[int(local_row)])]
                    )
                ],
                dtype=np.int64,
            )
        if not len(eligible):
            return [], 0
        order = np.argsort(-scores[eligible], kind="stable")[:limit]
        hits = [
            _ShardHit(
                row=int(shard.rows[int(eligible[int(local_index)])]),
                rank=rank,
                score=float(scores[int(eligible[int(local_index)])]),
                language=language,
            )
            for rank, local_index in enumerate(order, start=1)
        ]
        return hits, len(eligible)

    def _direct_match(
        self,
        hit: _ShardHit,
        *,
        query_language: str,
    ) -> CandidateMatch:
        document = self.catalog.documents[hit.row]
        return CandidateMatch(
            candidate=document,
            score=hit.score,
            score_name="bm25",
            transition_reason=(
                "query_language_complete_bm25"
                if self.language_policy == "query_language_complete"
                else "same_language_bm25"
            ),
            metadata={
                "document_language": document.language,
                "index_view_language": hit.language,
                "query_language": query_language,
                "language_policy": self.language_policy,
                "native_shard_rank": hit.rank,
            },
        )
