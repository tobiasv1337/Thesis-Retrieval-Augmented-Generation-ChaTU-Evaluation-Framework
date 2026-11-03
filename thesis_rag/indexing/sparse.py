"""Deterministic memory-mapped BM25 indexes with separate language statistics."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stopwordsiso import stopwords

from thesis_rag.indexing.catalog import (
    SUPPORTED_INDEX_LANGUAGES,
    ChunkDocumentCatalog,
)
from thesis_rag.indexing.manifests import (
    LanguageShardManifest,
    RetrievalIndexManifest,
    artifact_file,
    commit_staging_directory,
    create_manifest,
    create_staging_directory,
    load_and_verify_manifest,
    remove_staging_directory,
    resolve_index_root,
    write_manifest,
)
from thesis_rag.retrieval.config import RepresentationConfig, SparseSourceConfig
from thesis_rag.schema import canonical_json, hash_jsonable


SPARSE_BACKEND_ID = "mmap_bm25"
SPARSE_BACKEND_VERSION = "mmap_bm25_v1"
MIXED_LANGUAGE_DOCUMENT_TOKENIZATION_POLICY = "document_language_stopwords_v1"
UNICODE_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class SparseIndexError(ValueError):
    """Raised when a BM25 index cannot be built, verified, or queried."""


class SparseTokenizer:
    """Small explicit tokenizer whose behavior is part of index identity."""

    def __init__(self, *, remove_stopwords: bool) -> None:
        self.remove_stopwords = remove_stopwords
        self._stopwords = {
            language: {
                unicodedata.normalize("NFKC", item).casefold()
                for item in stopwords(language)
            }
            if remove_stopwords
            else set()
            for language in SUPPORTED_INDEX_LANGUAGES
        }

    def tokenize(self, text: str, *, language: str) -> list[str]:
        if language not in self._stopwords:
            raise SparseIndexError(f"unsupported tokenizer language: {language}")
        normalized = unicodedata.normalize("NFKC", text).casefold()
        blocked = self._stopwords[language]
        return [
            token
            for token in UNICODE_WORD_RE.findall(normalized)
            if token not in blocked
        ]

    def stopword_hashes(self) -> dict[str, str]:
        return {
            language: hash_jsonable(sorted(words))
            for language, words in self._stopwords.items()
        }


class SparseIndexBuilder:
    """Build one independent BM25 posting index for each language shard."""

    def __init__(
        self,
        *,
        config: SparseSourceConfig,
        representation: RepresentationConfig,
        catalog: ChunkDocumentCatalog,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.representation = representation
        self.catalog = catalog
        self.repo_root = repo_root.resolve()
        self.tokenizer = SparseTokenizer(
            remove_stopwords=config.remove_stopwords,
        )

    @property
    def output_dir(self) -> Path:
        root = resolve_index_root(self.config.index_root, repo_root=self.repo_root)
        return root / self.catalog.index_subdirectory

    @property
    def semantic_config_hash(self) -> str:
        view = self.catalog.identity.query_language_view
        return sparse_semantic_config_hash(
            self.config,
            self.representation,
            tokenizer=self.tokenizer,
            query_language_view=(view.to_dict() if view is not None else None),
        )

    def build(self, *, force: bool = False) -> RetrievalIndexManifest:
        self.catalog.load()
        existing_manifest = self.output_dir / "index_manifest.json"
        if self.output_dir.exists() and not force:
            return load_and_verify_manifest(
                existing_manifest,
                index_kind="sparse",
                component_id=self.config.component_id,
                source=self.catalog.identity,
                semantic_config_hash=self.semantic_config_hash,
                embedding_model=None,
            )

        started = time.perf_counter()
        staging = create_staging_directory(self.output_dir)
        try:
            shards: list[LanguageShardManifest] = []
            shard_statistics: dict[str, dict[str, int | float]] = {}
            for language in SUPPORTED_INDEX_LANGUAGES:
                rows = np.asarray(self.catalog.index_rows(language), dtype=np.int64)
                shard_dir = staging / language
                shard_dir.mkdir()
                statistics = self._build_shard(
                    rows=rows,
                    output_dir=shard_dir,
                )
                shard_statistics[language] = statistics
                shards.append(
                    LanguageShardManifest(
                        language=language,
                        document_count=len(rows),
                        row_ids_hash=hash_jsonable(rows.tolist()),
                        metadata={
                            **statistics,
                            "artifact_directory": language,
                            "row_semantics": (
                                "query_language_view"
                                if self.catalog.language_policy
                                == "query_language_complete"
                                else "document_language"
                            ),
                        },
                    ),
                )

            self.catalog.verify_unchanged()
            files = [
                artifact_file(path, relative_to=staging)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ]
            manifest = create_manifest(
                index_kind="sparse",
                component_id=self.config.component_id,
                source=self.catalog.identity,
                semantic_config_hash=self.semantic_config_hash,
                backend_id=SPARSE_BACKEND_ID,
                backend_version=SPARSE_BACKEND_VERSION,
                embedding_model=None,
                shards=shards,
                files=files,
                build_duration_ms=max(
                    0.0,
                    (time.perf_counter() - started) * 1000.0,
                ),
                metadata={
                    "tokenizer_id": self.config.tokenizer_id,
                    "k1": self.config.k1,
                    "b": self.config.b,
                    "remove_stopwords": self.config.remove_stopwords,
                    "stopword_hashes": self.tokenizer.stopword_hashes(),
                    "shard_statistics": shard_statistics,
                    "language_policy": self.catalog.language_policy,
                    "query_language_view": (
                        self.catalog.identity.query_language_view.to_dict()
                        if self.catalog.identity.query_language_view is not None
                        else None
                    ),
                    **(
                        {
                            "mixed_language_document_tokenization_policy": (
                                MIXED_LANGUAGE_DOCUMENT_TOKENIZATION_POLICY
                            ),
                        }
                        if self.catalog.identity.query_language_view is not None
                        else {}
                    ),
                },
            )
            write_manifest(staging / "index_manifest.json", manifest)
            load_and_verify_manifest(
                staging / "index_manifest.json",
                index_kind="sparse",
                component_id=self.config.component_id,
                source=self.catalog.identity,
                semantic_config_hash=self.semantic_config_hash,
                embedding_model=None,
            )
            commit_staging_directory(staging, self.output_dir, force=force)
            return manifest
        except Exception:
            remove_staging_directory(staging)
            raise

    def _build_shard(
        self,
        *,
        rows: NDArray[np.int64],
        output_dir: Path,
    ) -> dict[str, int | float]:
        documents = self.catalog.documents
        document_frequencies: Counter[str] = Counter()
        document_lengths = np.empty(len(rows), dtype=np.int32)
        representations = [
            self.catalog.representation_text(documents[int(row)])
            for row in rows
        ]
        for local_row, text in enumerate(representations):
            document = documents[int(rows[local_row])]
            tokens = self.tokenizer.tokenize(text, language=document.language)
            document_lengths[local_row] = len(tokens)
            document_frequencies.update(set(tokens))

        vocabulary = sorted(document_frequencies)
        vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
        offsets = np.zeros(len(vocabulary) + 1, dtype=np.int64)
        if vocabulary:
            offsets[1:] = np.cumsum(
                np.asarray(
                    [document_frequencies[token] for token in vocabulary],
                    dtype=np.int64,
                ),
            )
        posting_count = int(offsets[-1])
        document_indices = np.empty(posting_count, dtype=np.int32)
        weights = np.empty(posting_count, dtype=np.float32)
        cursors = offsets[:-1].copy()
        average_length = (
            float(document_lengths.mean()) if len(document_lengths) else 0.0
        )
        for local_row, text in enumerate(representations):
            document = documents[int(rows[local_row])]
            frequencies = Counter(
                self.tokenizer.tokenize(text, language=document.language),
            )
            for token, frequency in frequencies.items():
                token_index = vocabulary_index[token]
                position = int(cursors[token_index])
                document_indices[position] = local_row
                weights[position] = _bm25_term_weight(
                    frequency=frequency,
                    document_frequency=document_frequencies[token],
                    document_count=len(rows),
                    document_length=int(document_lengths[local_row]),
                    average_document_length=average_length,
                    k1=self.config.k1,
                    b=self.config.b,
                )
                cursors[token_index] += 1

        np.save(output_dir / "rows.npy", rows, allow_pickle=False)
        np.save(output_dir / "offsets.npy", offsets, allow_pickle=False)
        np.save(
            output_dir / "document_indices.npy",
            document_indices,
            allow_pickle=False,
        )
        np.save(output_dir / "weights.npy", weights, allow_pickle=False)
        (output_dir / "vocabulary.json").write_text(
            canonical_json(vocabulary) + "\n",
            encoding="utf-8",
        )
        return {
            "vocabulary_size": len(vocabulary),
            "posting_count": posting_count,
            "average_document_length": average_length,
        }


class SparseShard:
    """One verified BM25 score matrix represented as mmap posting arrays."""

    def __init__(
        self,
        *,
        language: str,
        rows: NDArray[np.int64],
        vocabulary: dict[str, int],
        offsets: NDArray[np.int64],
        document_indices: NDArray[np.int32],
        weights: NDArray[np.float32],
    ) -> None:
        self.language = language
        self.rows = rows
        self.vocabulary = vocabulary
        self.offsets = offsets
        self.document_indices = document_indices
        self.weights = weights

    def scores(self, tokens: list[str]) -> NDArray[np.float32]:
        scores = np.zeros(len(self.rows), dtype=np.float32)
        for token, query_frequency in Counter(tokens).items():
            token_index = self.vocabulary.get(token)
            if token_index is None:
                continue
            start = int(self.offsets[token_index])
            stop = int(self.offsets[token_index + 1])
            indices = self.document_indices[start:stop]
            scores[indices] += self.weights[start:stop] * query_frequency
        return scores


class SparseIndex:
    """Verified pair of language-specific BM25 shards."""

    def __init__(
        self,
        *,
        manifest: RetrievalIndexManifest,
        shards: dict[str, SparseShard],
    ) -> None:
        self.manifest = manifest
        self.shards = shards

    @classmethod
    def load(cls, *, builder: SparseIndexBuilder) -> "SparseIndex":
        manifest = load_and_verify_manifest(
            builder.output_dir / "index_manifest.json",
            index_kind="sparse",
            component_id=builder.config.component_id,
            source=builder.catalog.identity,
            semantic_config_hash=builder.semantic_config_hash,
            embedding_model=None,
        )
        shards: dict[str, SparseShard] = {}
        for shard_manifest in manifest.shards:
            language = shard_manifest.language
            root = builder.output_dir / language
            try:
                raw_vocabulary = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8"),
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise SparseIndexError(
                    f"invalid {language} BM25 vocabulary: {exc}",
                ) from exc
            if not isinstance(raw_vocabulary, list) or not all(
                isinstance(item, str) for item in raw_vocabulary
            ):
                raise SparseIndexError(f"invalid {language} BM25 vocabulary")
            if raw_vocabulary != sorted(set(raw_vocabulary)):
                raise SparseIndexError(
                    f"{language} BM25 vocabulary must be sorted and unique",
                )
            rows = np.load(root / "rows.npy", mmap_mode="r", allow_pickle=False)
            offsets = np.load(
                root / "offsets.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            document_indices = np.load(
                root / "document_indices.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            weights = np.load(
                root / "weights.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            _validate_shard_arrays(
                language=language,
                rows=rows,
                offsets=offsets,
                document_indices=document_indices,
                weights=weights,
                vocabulary_size=len(raw_vocabulary),
                expected_count=shard_manifest.document_count,
                expected_rows_hash=shard_manifest.row_ids_hash,
            )
            shards[language] = SparseShard(
                language=language,
                rows=rows,
                vocabulary={
                    token: index for index, token in enumerate(raw_vocabulary)
                },
                offsets=offsets,
                document_indices=document_indices,
                weights=weights,
            )
        return cls(manifest=manifest, shards=shards)


def sparse_semantic_config_hash(
    config: SparseSourceConfig,
    representation: RepresentationConfig,
    *,
    tokenizer: SparseTokenizer,
    query_language_view: dict[str, object] | None = None,
) -> str:
    semantics: dict[str, object] = {
        "implementation": SPARSE_BACKEND_VERSION,
        "representation": representation.to_semantic_dict(),
        "tokenizer_id": config.tokenizer_id,
        "remove_stopwords": config.remove_stopwords,
        "stopword_hashes": tokenizer.stopword_hashes(),
        "k1": config.k1,
        "b": config.b,
    }
    if query_language_view is not None:
        semantics["query_language_view"] = query_language_view
        semantics["mixed_language_document_tokenization_policy"] = (
            MIXED_LANGUAGE_DOCUMENT_TOKENIZATION_POLICY
        )
    return hash_jsonable(semantics)


def _bm25_term_weight(
    *,
    frequency: int,
    document_frequency: int,
    document_count: int,
    document_length: int,
    average_document_length: float,
    k1: float,
    b: float,
) -> float:
    if document_count <= 0 or average_document_length <= 0.0:
        return 0.0
    inverse_document_frequency = np.log(
        1.0
        + (document_count - document_frequency + 0.5)
        / (document_frequency + 0.5),
    )
    denominator = frequency + k1 * (
        1.0 - b + b * document_length / average_document_length
    )
    return float(
        inverse_document_frequency * frequency * (k1 + 1.0) / denominator,
    )


def _validate_shard_arrays(
    *,
    language: str,
    rows,
    offsets,
    document_indices,
    weights,
    vocabulary_size: int,
    expected_count: int,
    expected_rows_hash: str,
) -> None:
    if rows.dtype != np.int64 or rows.ndim != 1 or len(rows) != expected_count:
        raise SparseIndexError(f"invalid {language} BM25 row array")
    if hash_jsonable(rows.tolist()) != expected_rows_hash:
        raise SparseIndexError(f"{language} BM25 row hash mismatch")
    if offsets.dtype != np.int64 or offsets.shape != (vocabulary_size + 1,):
        raise SparseIndexError(f"invalid {language} BM25 offsets")
    if np.any(offsets[1:] < offsets[:-1]) or int(offsets[0]) != 0:
        raise SparseIndexError(f"non-monotonic {language} BM25 offsets")
    posting_count = int(offsets[-1])
    if (
        document_indices.dtype != np.int32
        or weights.dtype != np.float32
        or document_indices.shape != (posting_count,)
        or weights.shape != (posting_count,)
    ):
        raise SparseIndexError(f"invalid {language} BM25 posting arrays")
    if posting_count and (
        int(document_indices.min()) < 0
        or int(document_indices.max()) >= expected_count
    ):
        raise SparseIndexError(f"out-of-range {language} BM25 document index")
    if not np.isfinite(weights).all():
        raise SparseIndexError(f"non-finite {language} BM25 weights")
