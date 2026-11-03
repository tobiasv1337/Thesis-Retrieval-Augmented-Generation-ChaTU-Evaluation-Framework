"""CandidateSource backed by one verified local Dense chunk index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from thesis_rag.indexing.catalog import (
    SUPPORTED_INDEX_LANGUAGES,
    ChunkDocumentCatalog,
)
from thesis_rag.indexing.dense import (
    DenseIndex,
    DenseIndexBuilder,
    EmbeddingProvider,
    normalize_query,
)
from thesis_rag.indexing.manifests import RetrievalIndexManifest
from thesis_rag.interfaces import CorpusSnapshot, RagQuery
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateMatch
from thesis_rag.retrieval.config import (
    DenseSourceConfig,
    LanguagePolicy,
    RepresentationConfig,
)
from thesis_rag.retrieval.protocols import CandidateScope, ComponentRuntimeIdentity
from thesis_rag.retrieval.query_encoding import encode_query
from thesis_rag.schema import hash_file, hash_jsonable


DENSE_RETRIEVAL_VERSION = "numpy_exact_cosine_retrieval_v2"


class DenseRetrievalError(ValueError):
    """Raised when the Dense component is not ready or receives invalid input."""


class DenseCandidateSource:
    """Exact cosine retrieval over normalized, memory-mapped embeddings."""

    def __init__(
        self,
        *,
        config: DenseSourceConfig,
        representation: RepresentationConfig,
        language_policy: LanguagePolicy,
        catalog: ChunkDocumentCatalog,
        provider: EmbeddingProvider,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.language_policy = language_policy
        self.catalog = catalog
        self.provider = provider
        self.builder = DenseIndexBuilder(
            config=config,
            representation=representation,
            catalog=catalog,
            provider=provider,
            repo_root=repo_root,
        )
        self._index: DenseIndex | None = None
        self._identity: ComponentRuntimeIdentity | None = None

    @property
    def component_id(self) -> str:
        return self.config.component_id

    def build_index(self, *, force: bool = False) -> RetrievalIndexManifest:
        return self.builder.build(force=force)

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        self.catalog.load(expected_corpus_hash=corpus.corpus_hash)
        self.catalog.verify_unchanged()
        index = DenseIndex.load(builder=self.builder)
        manifest_path = self.builder.output_dir / "index_manifest.json"
        retrieval_semantics = {
            "implementation": DENSE_RETRIEVAL_VERSION,
            "language_policy": self.language_policy,
            "score": "cosine_over_normalized_embeddings",
            "unscoped_layout": (
                "query_language_rows"
                if self.language_policy == "query_language_complete"
                else "contiguous_language_views"
            ),
        }
        self._index = index
        self._identity = ComponentRuntimeIdentity(
            component_id=self.component_id,
            corpus_hash=corpus.corpus_hash,
            identity_hashes={
                "index_manifest": hash_file(manifest_path),
                "index_files": index.manifest.aggregate_file_hash,
                "model_artifact": self.provider.identity().model_artifact_digest,
                "dense_semantics": index.manifest.semantic_config_hash,
                "dense_retrieval": hash_jsonable(retrieval_semantics),
            },
            metadata={
                "index_id": index.manifest.index_id,
                "index_kind": "dense",
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
                "embedding_model": self.provider.identity().to_dict(),
                "retrieval_semantics": retrieval_semantics,
            },
        )

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self._identity is None:
            raise DenseRetrievalError("Dense component has not been loaded")
        return self._identity

    def search(
        self,
        query: RagQuery,
        *,
        limit: int,
        scope: CandidateScope | None = None,
    ) -> CandidateBatch:
        if self._index is None:
            raise DenseRetrievalError("Dense component has not been loaded")
        if limit <= 0:
            raise DenseRetrievalError("Dense retrieval limit must be positive")
        if query.language not in SUPPORTED_INDEX_LANGUAGES:
            raise DenseRetrievalError(
                f"unsupported query language: {query.language}",
            )

        selected_rows: NDArray[np.int64] | None = None
        row_offset = 0
        if scope is not None or self.language_policy == "query_language_complete":
            rows = self._candidate_rows(query.language)
            allows = scope.matcher() if scope is not None else None
            selected_rows = np.asarray(
                [
                    int(row)
                    for row in rows
                    if allows is None or allows(self.catalog.documents[int(row)])
                ],
                dtype=np.int64,
            )
            if not len(selected_rows):
                return CandidateBatch(matches=[], exhausted=True)
            candidate_count = len(selected_rows)
        else:
            candidate_embeddings, row_offset = self._unscoped_embedding_view(
                query.language,
            )
            candidate_count = len(candidate_embeddings)
        if not candidate_count:
            return CandidateBatch(matches=[], exhausted=True)

        query_embedding = normalize_query(
            encode_query(self.provider, query.text),
            dimensions=self.provider.identity().dimensions,
        )
        if selected_rows is not None:
            scores = np.asarray(
                (self._index.embeddings[selected_rows] @ query_embedding)
                if scope is not None
                else (self._index.embeddings @ query_embedding)[selected_rows],
                dtype=np.float32,
            )
        else:
            scores = np.asarray(
                candidate_embeddings @ query_embedding,
                dtype=np.float32,
            )
        order = np.argsort(-scores, kind="stable")
        selected = order[:limit]
        matches = []
        for local_index in selected:
            row = (
                int(selected_rows[int(local_index)])
                if selected_rows is not None
                else row_offset + int(local_index)
            )
            document = self.catalog.documents[row]
            matches.append(
                CandidateMatch(
                    candidate=document,
                    score=float(scores[int(local_index)]),
                    score_name="dense_cosine",
                    transition_reason="global_dense_similarity",
                    metadata={
                        "document_language": document.language,
                        "query_language": query.language,
                        "language_policy": self.language_policy,
                    },
                ),
            )
        return CandidateBatch(
            matches=matches,
            exhausted=candidate_count <= limit,
        )

    def close(self) -> None:
        self._index = None
        self._identity = None
        self.provider.close()

    def _candidate_rows(self, query_language: str) -> NDArray[np.int64]:
        assert self._index is not None
        if self.language_policy == "same_language_only":
            return self._index.shard_rows[query_language]
        if self.language_policy == "query_language_complete":
            return self._index.shard_rows[query_language]
        return np.sort(
            np.concatenate(
                [
                    self._index.shard_rows[language]
                    for language in SUPPORTED_INDEX_LANGUAGES
                ],
            ),
        )

    def _unscoped_embedding_view(
        self,
        query_language: str,
    ) -> tuple[NDArray[np.float32], int]:
        assert self._index is not None
        if self.language_policy == "cross_lingual":
            return self._index.embeddings, 0
        if self.language_policy == "query_language_complete":
            raise DenseRetrievalError(
                "query-language rows do not form one contiguous embedding view",
            )
        shard_slice = self._index.shard_slices[query_language]
        return self._index.embeddings[shard_slice], int(shard_slice.start or 0)
