from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from thesis_rag.corpus.chunking import ChunkRecord
from thesis_rag.corpus.context_groups import SectionContextGroupRecord
from thesis_rag.corpus.page_table import PageTableRecord
from thesis_rag.corpus.quality import PageQualityRecord
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.corpus.source_relationships import SourceRelationshipRecord
from thesis_rag.indexing.catalog import (
    STANDARD_INDEX_ELIGIBILITY_POLICY,
    ChunkCatalogError,
    ChunkDocumentCatalog,
)
from thesis_rag.indexing.dense import DenseIndexError, EmbeddingProvider
from thesis_rag.indexing.manifests import (
    EmbeddingModelIdentity,
    IndexManifestError,
    RetrievalIndexManifest,
)
from thesis_rag.indexing.sparse import (
    MIXED_LANGUAGE_DOCUMENT_TOKENIZATION_POLICY,
)
from thesis_rag.interfaces import RagQuery
from thesis_rag.registry import load_system_config, resolve_system_config
from thesis_rag.retrieval.candidates import CandidateRanking, RankedCandidate
from thesis_rag.retrieval.config import ModernSystemSettings, RepresentationConfig
from thesis_rag.retrieval.fusion import ReciprocalRankFusion
from thesis_rag.retrieval.system import (
    ModernRetrievalSystem,
    ModernRetrievalSystemError,
)
from thesis_rag.schema import canonical_json, hash_file, hash_text


CORPUS_HASH = hash_text("modern-fixture-corpus")
SOURCE_HASH = hash_text("modern-fixture-source")
CHUNKER_CONFIG_HASH = hash_text("modern-fixture-chunker")
MODEL_DIGEST = hash_text("modern-fixture-model")
CHUNKER_ID = "fixture_chunks"


@dataclass(frozen=True)
class _Snapshot:
    corpus_hash: str = CORPUS_HASH


class _FakeEmbeddingProvider:
    def __init__(self, *, metadata_seed: str = "fixture") -> None:
        self.metadata_seed = metadata_seed
        self.encoded_texts: list[str] = []
        self.closed = False

    def identity(self) -> EmbeddingModelIdentity:
        return EmbeddingModelIdentity(
            backend_id="ollama",
            model_id="fixture-embedding",
            model_artifact_digest=MODEL_DIGEST,
            model_metadata_hash=hash_text(self.metadata_seed),
            dimensions=3,
        )

    def encode(self, texts):
        self.encoded_texts.extend(texts)
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float(normalized.count("alpha")),
                    float(normalized.count("beta")),
                    float(normalized.count("gamma")),
                ],
            )
        return np.asarray(vectors, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_corpus(
    repo_root: Path,
    *,
    ineligible_page_ids: frozenset[str] = frozenset(),
    include_language_specific_stopwords: bool = False,
) -> Path:
    en_text = (
        "the gamma study" if include_language_specific_stopwords else "gamma study"
    )
    de_text = (
        "und beta berlin" if include_language_specific_stopwords else "beta berlin"
    )
    definitions = [
        ("a", "de", "alpha alpha berlin"),
        ("b", "en", en_text),
        ("c", "en", "alpha study"),
        ("d", "de", de_text),
    ]
    chunks = []
    context_groups = []
    pages = []
    page_quality = []
    for index, (name, language, text) in enumerate(definitions):
        page_id = f"page:{name}"
        section_id = f"section:{name}"
        text_hash = hash_text(text)
        chunks.append(
            ChunkRecord(
                chunk_id=f"chunk:{name}",
                chunker_id=CHUNKER_ID,
                chunker_mode="fixed_char",
                chunker_config_hash=CHUNKER_CONFIG_HASH,
                chunk_index=0,
                page_key=f"page_key:{name}",
                page_id=page_id,
                section_key=f"section_key:{name}",
                section_id=section_id,
                language=language,
                intent=f"intent-{name}",
                source_path=f"topics/{name}/content.html",
                url=f"https://example.test/{name}",
                title=f"Page {name}",
                heading_path=[f"Heading {name}"],
                text=text,
                text_hash=text_hash,
                char_start=0,
                char_end=len(text),
                section_char_start=0,
                section_char_end=len(text),
                boundary_start_kind="page",
                boundary_end_kind="page",
                corpus_hash=CORPUS_HASH,
                source_page_text_hash=text_hash,
                source_section_text_hash=text_hash,
                source_artifact_hashes={"pages": hash_text("pages")},
                overlapping_section_ids=[section_id],
                overlapping_context_group_ids=[],
                overlapping_content_block_ids=[f"content_block:{name}"],
                equivalence_hint=f"equivalence:{name}",
            ).to_dict(),
        )
        context_groups.append(
            SectionContextGroupRecord(
                group_id=f"section_context_group:{name}",
                group_index=index,
                page_key=f"page_key:{name}",
                page_id=page_id,
                language=language,
                intent=f"intent-{name}",
                source_path=f"topics/{name}/content.html",
                url=f"https://example.test/{name}",
                title=f"Page {name}",
                parent_section_id=None,
                ordered_section_ids=[section_id],
                context_section_ids=[section_id] if index == 0 else [],
                focus_section_ids=[] if index == 0 else [section_id],
                focus_effective_chars=0 if index == 0 else len(text),
                context_effective_chars=len(text) if index == 0 else 0,
                reason_codes=(
                    ["orphan_structural"] if index == 0 else ["substantive_section"]
                ),
                source_sections_hash=hash_text(f"sections:{name}"),
                group_config_hash=hash_text("context-group-config"),
                corpus_hash=CORPUS_HASH,
            ).to_dict(),
        )
        pages.append(
            PageTableRecord(
                page_key=f"page_key:{name}",
                page_id=page_id,
                language=language,
                intent=f"intent-{name}",
                url=f"https://example.test/{name}",
                title=f"Page {name}",
                mapping_title=f"Page {name}",
                breadcrumbs=[
                    {
                        "name": "TU Berlin",
                        "position": 1,
                        "url": "https://example.test",
                    },
                    {
                        "name": f"Page {name}",
                        "position": 2,
                        "url": f"https://example.test/{name}",
                    },
                ],
                heading_list=[f"Heading {name}"],
                parent_intent=None,
                depth=1,
                etag=None,
                content_flags=["prose"],
                table_count=0,
                image_count=0,
                section_count=1,
                text_hash=text_hash,
                source_paths=[f"topics/{name}/content.html"],
                corpus_hash=CORPUS_HASH,
                keywords=[name],
            ).to_dict(),
        )
        eligible = page_id not in ineligible_page_ids
        exclusion_reasons = [] if eligible else ["fixture_excluded"]
        page_quality.append(
            PageQualityRecord(
                page_key=f"page_key:{name}",
                page_id=page_id,
                language=language,
                text_hash=text_hash,
                empty=False,
                short_text=False,
                error_template=False,
                exact_duplicate=False,
                duplicate_cluster_id=None,
                canonical_page_id=page_id,
                eligible_question_generation=eligible,
                eligible_standard_index=eligible,
                exclusion_reasons=exclusion_reasons,
                flags=exclusion_reasons,
                quality_rule_version="fixture_quality_rules_v1",
                corpus_hash=CORPUS_HASH,
            ).to_dict(),
        )

    chunk_path = repo_root / "corpus/chunks/fixture.jsonl"
    context_group_path = repo_root / "corpus/section_context_groups.jsonl"
    page_table_path = repo_root / "corpus/page_table.jsonl"
    page_quality_path = repo_root / "corpus/page_quality.jsonl"
    source_relationships_path = repo_root / "corpus/source_relationships.jsonl"
    _write_jsonl(chunk_path, chunks)
    _write_jsonl(context_group_path, context_groups)
    _write_jsonl(page_table_path, pages)
    _write_jsonl(page_quality_path, page_quality)
    relationships = [
        SourceRelationshipRecord(
            source_relationship_id="source_relationship:fixture-counterpart",
            relationship_kind="language_counterpart",
            member_page_ids=["page:a", "page:c"],
            confidence="high",
            evidence={"language_pair": ["de", "en"]},
            source_relationship_rule_version="source_relationship_rules_v1",
            construction_config_hash=hash_text("source-relationship-config"),
            corpus_hash=CORPUS_HASH,
        ).to_dict(),
    ]
    _write_jsonl(source_relationships_path, relationships)
    manifest = SnapshotManifest(
        corpus_id="fixture-corpus",
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SOURCE_HASH,
        source_manifest_hash=hash_text("source-manifest"),
        archive_hash=hash_text("archive"),
        artifact_hashes={
            "chunks": {CHUNKER_ID: hash_file(chunk_path)},
            "page_table": hash_file(page_table_path),
            "page_quality": hash_file(page_quality_path),
            "section_context_groups": hash_file(context_group_path),
            "source_relationships": hash_file(source_relationships_path),
        },
        artifact_record_counts={
            "chunks": {CHUNKER_ID: len(chunks)},
            "page_table": len(pages),
            "page_quality": len(page_quality),
            "section_context_groups": len(context_groups),
            "source_relationships": len(relationships),
        },
        artifact_paths={
            "chunks": {CHUNKER_ID: "corpus/chunks/fixture.jsonl"},
            "page_table": "corpus/page_table.jsonl",
            "page_quality": "corpus/page_quality.jsonl",
            "section_context_groups": "corpus/section_context_groups.jsonl",
            "source_relationships": "corpus/source_relationships.jsonl",
        },
        parser_version="fixture-parser",
        parser_config_hash=hash_text("parser-config"),
        extraction_config_hash=hash_text("extraction-config"),
        artifact_config_hashes={},
        archive_compression="zstd",
        created_at_utc="2026-08-15T00:00:00+00:00",
        source_file_count=0,
        source_counts_by_role={},
        source_counts_by_language={"de": 2, "en": 2},
        page_count=4,
        section_count=4,
        content_block_count=4,
        table_count=0,
        image_count=0,
        chunking={
            "chunkers": [
                {
                    "chunker_id": CHUNKER_ID,
                    "config_hash": CHUNKER_CONFIG_HASH,
                    "record_count": len(chunks),
                },
            ],
        },
    )
    manifest_path = repo_root / "snapshot.json"
    manifest_path.write_text(
        canonical_json(manifest.to_dict()) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _config(
    *,
    dense: bool,
    sparse: bool,
    language_policy: str = "cross_lingual",
    fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "system_config_v2",
        "system_id": "modern_retrieval_fixture",
        "class_path": "thesis_rag.retrieval.system.ModernRetrievalSystem",
        "settings": {
            "snapshot_manifest_path": "snapshot.json",
            "pipeline": {
                "representation": {
                    "chunker_id": CHUNKER_ID,
                    "fields": fields or ["text"],
                },
                "language_policy": language_policy,
                "candidate_k": 20,
                "dense_global": {
                    "enabled": dense,
                    "component_id": "fixture_dense",
                    "index_root": "indexes/dense",
                    "normalize": True,
                    "model": {
                        "backend": "ollama",
                        "endpoint": "http://127.0.0.1:11434",
                        "model_id": "fixture-embedding",
                        "expected_model_artifact_digest": MODEL_DIGEST,
                        "dimensions": 3,
                        "batch_size": 2,
                        "timeout_seconds": 1.0,
                        "truncate": True,
                    },
                },
                "sparse_global": {
                    "enabled": sparse,
                    "component_id": "fixture_sparse",
                    "index_root": "indexes/sparse",
                    "tokenizer_id": "unicode_words_v1",
                    "k1": 1.5,
                    "b": 0.75,
                    "remove_stopwords": False,
                    "cross_shard_rrf_k": 60,
                },
                "content_rrf": {
                    "enabled": dense and sparse,
                    "component_id": "fixture_rrf",
                    "rank_constant": 60,
                },
                "page_routing": {
                    "enabled": False,
                    "component_id": "fixture_page",
                    "top_p": 5,
                },
                "intent_prior": {
                    "enabled": False,
                    "component_id": "fixture_prior",
                },
                "reranker": {
                    "enabled": False,
                    "component_id": "fixture_reranker",
                },
                "context": {
                    "enabled": False,
                    "component_id": "fixture_context",
                    "policy_id": "direct_children",
                    "source_top_k": 15,
                    "token_budget": 1000,
                },
                "config_version": "modern_pipeline_config_v3",
            },
        },
    }


def _query(text: str = "alpha", *, language: str = "de") -> RagQuery:
    return RagQuery(query_id="query:fixture", text=text, language=language)


def _system(
    config: dict[str, Any],
    *,
    repo_root: Path,
    provider: _FakeEmbeddingProvider,
) -> ModernRetrievalSystem:
    return ModernRetrievalSystem(
        config,
        repo_root=repo_root,
        embedding_provider_factory=lambda _config: provider,
    )


@pytest.mark.parametrize("policy,dedup", [
    ("query_language_complete", None), ("cross_lingual", None),
    ("cross_lingual", "highest_ranked_page"),
])
def test_language_signal_changes_only_retrieval_input_and_reuses_indexes(
    tmp_path: Path, policy: str, dedup: str | None,
) -> None:
    from thesis_rag.eval.schemas import QuestionRecord

    _write_corpus(tmp_path)
    config = _config(dense=True, sparse=True, language_policy=policy)
    if dedup is not None:
        config["settings"]["pipeline"]["counterpart_deduplication"] = dedup
    original = _system(config, repo_root=tmp_path, provider=_FakeEmbeddingProvider())
    original.build_index()
    original.build_or_load(_Snapshot())
    expected = {
        language: [item.target_id for item in original.retrieve(_query(language=language), top_k=4).items]
        for language in ("de", "en")
    }
    index_hashes = original.runtime_identity().identity_hashes
    original.close()
    for signal in ("correct", "always_de", "always_en", "swapped"):
        configured = {**config, "settings": {**config["settings"], "query_language_signal": signal}}
        provider = _FakeEmbeddingProvider()
        system = _system(configured, repo_root=tmp_path, provider=provider)
        system.build_or_load(_Snapshot())
        assert provider.encoded_texts == []  # No new document embeddings.
        assert system.runtime_identity().identity_hashes == index_hashes
        assert system.config_hash != original.config_hash
        for language in ("de", "en"):
            query = QuestionRecord.from_dict(_query(language=language).to_dict())
            before = query.to_dict()
            result = system.retrieve(query, top_k=4)
            supplied = {"correct": language, "always_de": "de", "always_en": "en",
                        "swapped": "en" if language == "de" else "de"}[signal]
            assert query.to_dict() == before
            assert result.query_id == query.query_id
            assert [item.target_id for item in result.items] == expected[supplied]
            assert result.debug["query_language_signal"] == {
                "condition": signal, "true_language": language, "retrieval_language": supplied,
            }
        system.close()
    if policy == "cross_lingual":
        assert expected["de"] == expected["en"]


def test_dense_index_uses_one_embedding_matrix_and_exact_language_shards(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    provider = _FakeEmbeddingProvider()
    same_language = _system(
        _config(dense=True, sparse=False, language_policy="same_language_only"),
        repo_root=tmp_path,
        provider=provider,
    )

    report = same_language.build_index()

    assert report["indexed_document_count"] == 4
    assert len(provider.encoded_texts) == 4
    index_dir = tmp_path / "indexes/dense" / CHUNKER_ID
    manifest = RetrievalIndexManifest.from_dict(
        json.loads((index_dir / "index_manifest.json").read_text()),
    )
    assert [item.path for item in manifest.files] == [
        "embeddings.npy",
        "rows.de.npy",
        "rows.en.npy",
    ]
    assert manifest.metadata["embedding_generation_count"] == 4
    assert np.load(index_dir / "rows.de.npy").tolist() == [0, 1]
    assert np.load(index_dir / "rows.en.npy").tolist() == [2, 3]
    same_language.close()
    assert provider.closed is True

    provider = _FakeEmbeddingProvider()
    same_language = _system(
        _config(dense=True, sparse=False, language_policy="same_language_only"),
        repo_root=tmp_path,
        provider=provider,
    )
    same_language.build_or_load(_Snapshot())
    dense_source = same_language.components.dense_global
    assert dense_source is not None
    assert dense_source._index is not None
    de_view, de_offset = dense_source._unscoped_embedding_view("de")
    assert np.shares_memory(de_view, dense_source._index.embeddings)
    assert de_offset == 0
    assert len(de_view) == 2
    result = same_language.retrieve(_query(), top_k=2, stage_evaluation_depth=2)
    assert [item.target_id for item in result.items] == ["chunk:a", "chunk:d"]
    assert len(result.items) == 2

    cross_provider = _FakeEmbeddingProvider()
    cross_language = _system(
        _config(dense=True, sparse=False, language_policy="cross_lingual"),
        repo_root=tmp_path,
        provider=cross_provider,
    )
    cross_language.build_or_load(_Snapshot())
    cross_source = cross_language.components.dense_global
    assert cross_source is not None
    assert cross_source._index is not None
    cross_view, cross_offset = cross_source._unscoped_embedding_view("de")
    assert np.shares_memory(cross_view, cross_source._index.embeddings)
    assert cross_view is cross_source._index.embeddings
    assert cross_offset == 0
    cross_result = cross_language.retrieve(
        _query(),
        top_k=4,
        stage_evaluation_depth=4,
    )
    assert [item.target_id for item in cross_result.items[:2]] == [
        "chunk:a",
        "chunk:c",
    ]
    assert cross_result.items[0].score == pytest.approx(1.0)
    assert cross_result.items[1].score == pytest.approx(1.0)


def test_query_language_complete_builds_verified_self_contained_views(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path, include_language_specific_stopwords=True)
    config = _config(
        dense=True,
        sparse=True,
        language_policy="query_language_complete",
    )
    config["settings"]["pipeline"]["sparse_global"]["remove_stopwords"] = True
    system = _system(
        config,
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )

    system.build_index()
    dense_root = (
        tmp_path
        / "indexes/dense"
        / "query_language_complete"
        / CHUNKER_ID
    )
    sparse_root = (
        tmp_path
        / "indexes/sparse"
        / "query_language_complete"
        / CHUNKER_ID
    )
    assert np.load(dense_root / "rows.de.npy").tolist() == [0, 1, 2]
    assert np.load(dense_root / "rows.en.npy").tolist() == [1, 2, 3]
    dense_manifest = RetrievalIndexManifest.from_dict(
        json.loads((dense_root / "index_manifest.json").read_text()),
    )
    view = dense_manifest.source.query_language_view
    assert view is not None
    assert view.source_relationships_path == "corpus/source_relationships.jsonl"
    assert view.document_counts == {"de": 3, "en": 3}
    assert view.fallback_page_counts == {"de": 1, "en": 1}
    assert view.suppressed_page_counts == {"de": 1, "en": 1}
    assert (sparse_root / "de/rows.npy").is_file()
    sparse_manifest = RetrievalIndexManifest.from_dict(
        json.loads((sparse_root / "index_manifest.json").read_text()),
    )
    shard_metadata = {
        shard.language: shard.metadata for shard in sparse_manifest.shards
    }
    assert shard_metadata["de"]["average_document_length"] == pytest.approx(7 / 3)
    assert shard_metadata["en"]["average_document_length"] == pytest.approx(2.0)
    assert (
        sparse_manifest.metadata["mixed_language_document_tokenization_policy"]
        == MIXED_LANGUAGE_DOCUMENT_TOKENIZATION_POLICY
    )

    system.build_or_load(_Snapshot())
    de_alpha = system.retrieve(
        _query("alpha", language="de"),
        top_k=3,
        stage_evaluation_depth=3,
    )
    assert de_alpha.items[0].target_id == "chunk:a"
    assert "chunk:c" not in {item.target_id for item in de_alpha.items}
    de_gamma = system.retrieve(
        _query("gamma", language="de"),
        top_k=3,
        stage_evaluation_depth=3,
    )
    assert de_gamma.items[0].target_id == "chunk:b"
    en_alpha = system.retrieve(
        _query("alpha", language="en"),
        top_k=3,
        stage_evaluation_depth=3,
    )
    assert en_alpha.items[0].target_id == "chunk:c"
    assert "chunk:a" not in {item.target_id for item in en_alpha.items}
    en_beta = system.retrieve(
        _query("beta", language="en"),
        top_k=3,
        stage_evaluation_depth=3,
    )
    assert en_beta.items[0].target_id == "chunk:d"


def test_query_language_complete_keeps_foreign_page_without_eligible_counterpart(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path, ineligible_page_ids=frozenset({"page:c"}))
    catalog = ChunkDocumentCatalog(
        repo_root=tmp_path,
        snapshot_manifest_path="snapshot.json",
        representation=RepresentationConfig(
            chunker_id=CHUNKER_ID,
            fields=["text"],
        ),
        language_policy="query_language_complete",
    )
    catalog.load()

    en_targets = {
        catalog.documents[row].target_id for row in catalog.index_rows("en")
    }
    assert "chunk:a" in en_targets
    assert "chunk:c" not in en_targets

    relationship_path = tmp_path / "corpus/source_relationships.jsonl"
    relationship_path.write_text(
        relationship_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ChunkCatalogError, match="source relationships"):
        catalog.verify_unchanged()


def test_cross_lingual_counterpart_selection_deduplicates_and_backfills(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    config = _config(
        dense=True,
        sparse=False,
        language_policy="cross_lingual",
    )
    config["settings"]["pipeline"]["counterpart_deduplication"] = (
        "highest_ranked_page"
    )
    system = _system(
        config,
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )

    system.build_index()
    assert (tmp_path / "indexes/dense" / CHUNKER_ID).is_dir()
    assert not (tmp_path / "indexes/dense/counterpart_deduplication").exists()
    system.build_or_load(_Snapshot())
    result = system.retrieve(
        _query("alpha", language="de"),
        top_k=3,
        stage_evaluation_depth=3,
    )

    assert len(result.items) == 3
    assert result.items[0].target_id == "chunk:a"
    assert "chunk:c" not in {item.target_id for item in result.items}
    final_trace = next(
        trace for trace in result.stage_traces if trace.stage == "final_selection"
    )
    assert final_trace.mode == "language_counterpart_final_selection_v1"
    assert final_trace.metadata["removed_reasons"]["chunk:c"] == (
        "language_counterpart_family_duplicate"
    )

    dense = system.components.dense_global
    selector = system.components.final_selector
    assert dense is not None
    assert selector is not None
    documents = {item.target_id: item for item in dense.catalog.documents}
    second_a = documents["chunk:a"].model_copy(
        update={"target_id": "chunk:a-second", "chunk_id": "chunk:a-second"},
    )
    ranking_documents = [
        documents["chunk:a"],
        second_a,
        documents["chunk:c"],
        documents["chunk:b"],
    ]
    ranking = CandidateRanking(
        stage_id="fixture_ranking",
        stage_label="Fixture ranking",
        items=[
            RankedCandidate(
                candidate=document,
                rank=rank,
                score=1.0 / rank,
                score_name="fixture_score",
                component_scores=[],
                transitions=[],
            )
            for rank, document in enumerate(ranking_documents, start=1)
        ],
        exhausted=True,
    )
    selected = selector.select(ranking, limit=3)
    assert [match.candidate.target_id for match in selected.matches] == [
        "chunk:a",
        "chunk:a-second",
        "chunk:b",
    ]


def test_counterpart_selection_requires_cross_lingual_policy() -> None:
    config = _config(
        dense=True,
        sparse=False,
        language_policy="same_language_only",
    )
    config["settings"]["pipeline"]["counterpart_deduplication"] = (
        "highest_ranked_page"
    )

    with pytest.raises(
        ValueError,
        match="counterpart deduplication requires cross_lingual retrieval",
    ):
        ModernSystemSettings.from_dict(config["settings"])


def test_all_modern_indexes_exclude_ineligible_page_chunks(
    tmp_path: Path,
) -> None:
    _write_corpus(
        tmp_path,
        ineligible_page_ids=frozenset({"page:b"}),
    )
    catalog = ChunkDocumentCatalog(
        repo_root=tmp_path,
        snapshot_manifest_path="snapshot.json",
        representation=RepresentationConfig(
            chunker_id=CHUNKER_ID,
            fields=["text"],
        ),
    )
    catalog.load()

    assert [document.target_id for document in catalog.documents] == [
        "chunk:a",
        "chunk:d",
        "chunk:c",
    ]
    assert catalog.identity.eligibility_policy == (
        STANDARD_INDEX_ELIGIBILITY_POLICY
    )
    assert catalog.identity.source_document_count == 4
    assert catalog.identity.excluded_document_count == 1
    assert catalog.identity.document_count == 3
    assert all(
        name not in catalog.identity.to_index_identity_dict()
        for name in (
            "chunk_artifact_set_manifest_path",
            "chunk_artifact_set_manifest_hash",
            "chunk_artifact_set_id",
        )
    )
    assert RepresentationConfig(
        chunker_id=CHUNKER_ID,
        chunk_artifact_set_manifest_path="derived/manifest.json",
        fields=["text"],
    ).to_semantic_dict() == RepresentationConfig(
        chunker_id=CHUNKER_ID,
        fields=["text"],
    ).to_semantic_dict()
    assert catalog.identity.page_quality_path == "corpus/page_quality.jsonl"
    assert catalog.identity.page_quality_hash == hash_file(
        tmp_path / "corpus/page_quality.jsonl",
    )
    assert all(
        document.metadata["overlapping_context_group_ids"]
        == [document.section_id.replace("section:", "section_context_group:")]
        for document in catalog.documents
    )

    provider = _FakeEmbeddingProvider()
    system = _system(
        _config(dense=True, sparse=True),
        repo_root=tmp_path,
        provider=provider,
    )
    report = system.build_index()

    assert report["indexed_document_count"] == 3
    assert len(provider.encoded_texts) == 3
    for root in ("dense", "sparse"):
        manifest = RetrievalIndexManifest.from_dict(
            json.loads(
                (
                    tmp_path
                    / f"indexes/{root}"
                    / CHUNKER_ID
                    / "index_manifest.json"
                ).read_text(),
            ),
        )
        assert manifest.source == catalog.identity


def test_catalog_detects_page_quality_changes_after_loading(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    catalog = ChunkDocumentCatalog(
        repo_root=tmp_path,
        snapshot_manifest_path="snapshot.json",
        representation=RepresentationConfig(
            chunker_id=CHUNKER_ID,
            fields=["text"],
        ),
    )
    catalog.load()
    quality_path = tmp_path / "corpus/page_quality.jsonl"
    quality_path.write_text(
        quality_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ChunkCatalogError, match="page quality"):
        catalog.verify_unchanged()


def test_dense_manifest_rejects_model_config_source_and_file_mismatch(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    provider = _FakeEmbeddingProvider()
    system = _system(
        _config(dense=True, sparse=False),
        repo_root=tmp_path,
        provider=provider,
    )
    system.build_index()

    changed_model = _system(
        _config(dense=True, sparse=False),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(metadata_seed="different"),
    )
    with pytest.raises(IndexManifestError, match="embedding_model"):
        changed_model.build_or_load(_Snapshot())

    changed_representation = _system(
        _config(dense=True, sparse=False, fields=["title", "text"]),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )
    with pytest.raises(IndexManifestError, match="semantic_config_hash"):
        changed_representation.build_or_load(_Snapshot())

    wrong_corpus = _system(
        _config(dense=True, sparse=False),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )
    with pytest.raises(ValueError, match="corpus hash mismatch"):
        wrong_corpus.build_or_load(_Snapshot(hash_text("other-corpus")))

    rows_path = tmp_path / "indexes/dense" / CHUNKER_ID / "rows.de.npy"
    rows_path.write_bytes(rows_path.read_bytes() + b"tampered")
    tampered = _system(
        _config(dense=True, sparse=False),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )
    with pytest.raises(IndexManifestError, match="size mismatch"):
        tampered.build_or_load(_Snapshot())


def test_bm25_scores_omit_zero_matches_and_cross_shard_rrf(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    same_language = _system(
        _config(dense=False, sparse=True, language_policy="same_language_only"),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )
    report = same_language.build_index()
    assert [item["index_kind"] for item in report["components"]] == ["sparse"]
    same_language.build_or_load(_Snapshot())
    result = same_language.retrieve(_query(), top_k=2, stage_evaluation_depth=2)

    inverse_document_frequency = np.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = inverse_document_frequency * 2 * 2.5 / (
        2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2.5)
    )
    assert [item.target_id for item in result.items] == ["chunk:a"]
    assert result.items[0].score == pytest.approx(expected, rel=1e-6)

    cross_language = _system(
        _config(dense=False, sparse=True, language_policy="cross_lingual"),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )
    cross_language.build_or_load(_Snapshot())
    cross = cross_language.retrieve(_query(), top_k=4, stage_evaluation_depth=4)
    assert [item.target_id for item in cross.items] == ["chunk:a", "chunk:c"]
    assert cross.items[0].score == pytest.approx(1.0 / 61.0)
    assert cross.items[1].score == pytest.approx(1.0 / 61.0)
    provenance = cross.items[0].metadata["candidate_provenance"]
    sparse_score = provenance["component_scores"][0]
    assert sparse_score["score_name"] == "cross_shard_rrf"
    assert sparse_score["metadata"]["native_bm25_score"] > 0.0

    german_only = cross_language.retrieve(
        _query("berlin"),
        top_k=4,
        stage_evaluation_depth=4,
    )
    assert {item.target_id for item in german_only.items} == {
        "chunk:a",
        "chunk:d",
    }
    assert all(
        item.metadata["candidate_provenance"]["component_scores"][0]["metadata"][
            "native_bm25_score"
        ]
        > 0.0
        for item in german_only.items
    )

    no_matches = cross_language.retrieve(
        _query("unseen-token"),
        top_k=4,
        stage_evaluation_depth=4,
    )
    assert no_matches.items == []


def test_closed_modern_system_is_terminal_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    system = _system(
        _config(dense=True, sparse=False),
        repo_root=tmp_path,
        provider=_FakeEmbeddingProvider(),
    )
    system.build_index()
    system.build_or_load(_Snapshot())

    system.close()
    system.close()

    with pytest.raises(ModernRetrievalSystemError, match="closed retrieval system"):
        system.build_or_load(_Snapshot())
    with pytest.raises(ModernRetrievalSystemError, match="closed retrieval system"):
        system.build_index()


def test_ordinary_rrf_scores_and_ranks_deterministic_fixture(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    catalog = ChunkDocumentCatalog(
        repo_root=tmp_path,
        snapshot_manifest_path="snapshot.json",
        representation=RepresentationConfig(
            chunker_id=CHUNKER_ID,
            fields=["text"],
        ),
    )
    catalog.load()
    by_id = {item.target_id: item for item in catalog.documents}

    def ranking(stage: str, target_ids: list[str]) -> CandidateRanking:
        return CandidateRanking(
            stage_id=stage,
            stage_label=stage,
            items=[
                RankedCandidate(
                    candidate=by_id[target_id],
                    rank=rank,
                    score=float(len(target_ids) - rank + 1),
                    score_name=stage,
                    component_scores=[],
                    transitions=[],
                )
                for rank, target_id in enumerate(target_ids, start=1)
            ],
            exhausted=True,
        )

    config = ModernSystemSettings.from_dict(
        _config(dense=True, sparse=True)["settings"],
    ).pipeline.content_rrf
    fusion = ReciprocalRankFusion(config)
    fusion.build_or_load(_Snapshot())
    batch = fusion.fuse(
        [
            ranking("dense", ["chunk:a", "chunk:b"]),
            ranking("sparse", ["chunk:b", "chunk:c"]),
        ],
        limit=3,
    )

    assert [match.candidate.target_id for match in batch.matches] == [
        "chunk:b",
        "chunk:a",
        "chunk:c",
    ]
    assert batch.matches[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert batch.matches[1].score == pytest.approx(1 / 61)
    assert batch.matches[2].score == pytest.approx(1 / 62)
    weighted = fusion.fuse(
        [ranking("global", ["chunk:a", "chunk:b"]),
         ranking("priority", ["chunk:b", "chunk:c"])],
        limit=3, weights=(2, 1),
    )
    assert [m.candidate.target_id for m in weighted.matches] == ["chunk:b", "chunk:a", "chunk:c"]
    assert weighted.matches[0].score == pytest.approx(2 / 62 + 1 / 61)
    assert weighted.matches[1].score == pytest.approx(2 / 61)
    assert [c["branch_weight"] for c in weighted.matches[0].metadata["contributions"]] == [2, 1]
    assert "branch_weight" not in batch.matches[0].metadata["contributions"][0]
    for invalid in ((1,), (0, 1), (-1, 1), (float("nan"), 1), (float("inf"), 1), (True, 1)):
        with pytest.raises(ValueError, match="weights"):
            fusion.fuse([ranking("a", []), ranking("b", [])], limit=3, weights=invalid)


def test_complete_hybrid_factory_wiring_preserves_all_global_checkpoints(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    provider = _FakeEmbeddingProvider()
    system = _system(
        _config(dense=True, sparse=True),
        repo_root=tmp_path,
        provider=provider,
    )
    report = system.build_index()
    assert [item["index_kind"] for item in report["components"]] == [
        "dense",
        "sparse",
    ]
    assert report["indexed_document_count"] == 4

    system.build_or_load(_Snapshot())
    result = system.retrieve(_query(), top_k=4, stage_evaluation_depth=4)
    assert [stage.stage_id for stage in result.stage_results] == [
        "dense_global",
        "sparse_global",
        "content_rrf",
    ]
    assert result.enabled_stages["dense_global"]["enabled"] is True
    assert result.enabled_stages["sparse_global"]["enabled"] is True
    assert result.enabled_stages["content_rrf"]["enabled"] is True
    identity = system.runtime_identity()
    assert set(identity.metadata["components"]) == {
        "content_rrf",
        "dense_global",
        "sparse_global",
    }
    dense_identity = identity.metadata["components"]["dense_global"]
    sparse_identity = identity.metadata["components"]["sparse_global"]
    assert "dense_retrieval" in dense_identity["identity_hashes"]
    assert "sparse_retrieval" in sparse_identity["identity_hashes"]
    assert (
        sparse_identity["metadata"]["retrieval_semantics"]["zero_score_policy"]
        == "exclude"
    )


def test_language_signal_suite_is_a_controlled_top5_factorial() -> None:
    from thesis_rag.eval.suite_config import load_experiment_suite_config, _apply_overrides
    from thesis_rag.eval.suite_report import SuiteReportConfig

    root = Path(__file__).parents[2]
    suite = load_experiment_suite_config(root / "configs/thesis/experiments/retrieval_language_signal.yaml")
    report = SuiteReportConfig.from_dict(yaml.safe_load(
        (root / "configs/thesis/retrieval_reports/language_signal.yaml").read_text(),
    ))
    assert suite.defaults.top_k == 5 and suite.defaults.query_limit is None
    assert suite.questions.manifest_path.endswith("base_question_set_manifest.json")
    assert len(suite.experiments) == 12
    hashes, conditions = set(), set()
    for experiment in suite.experiments:
        assert experiment.splits == ["development"]
        assert experiment.query_limit is None
        base = load_system_config(root / experiment.system_config)
        raw, _ = _apply_overrides(base.raw_config, experiment.overrides)
        resolved = resolve_system_config(raw)
        hashes.add(resolved.config_hash)
        settings = ModernRetrievalSystem.validate_config(raw)
        conditions.add((settings.pipeline.language_policy, settings.pipeline.counterpart_deduplication,
                        settings.query_language_signal))
        assert settings.pipeline.content_rrf.rank_constant == 10
        assert settings.pipeline.representation.chunker_id == "section_preferred_800_100"
        assert settings.pipeline.candidate_k == 100
    assert len(hashes) == 12
    assert conditions == {(policy, dedup, signal) for policy, dedup in (
                             ("cross_lingual", None), ("query_language_complete", None),
                             ("cross_lingual", "highest_ranked_page"))
                          for signal in ("correct", "always_de", "always_en", "swapped")}
    assert all(policy.cutoff == 5 for policy in report.selection_policies)
    assert {comparison.comparison_id for comparison in suite.comparisons} == set(report.comparison_ids)


def test_checked_system_and_experiment_configs_preserve_controlled_scope() -> None:
    repo_root = Path(__file__).parents[2]
    dense = load_system_config(
        repo_root / "configs/thesis/systems/modern_dense.yaml",
    )
    hybrid = load_system_config(
        repo_root / "configs/thesis/systems/modern_hybrid.yaml",
    )
    sparse = load_system_config(
        repo_root / "configs/thesis/systems/modern_sparse.yaml",
    )
    assert dense.raw_config["settings"]["pipeline"]["dense_global"]["enabled"]
    assert not dense.raw_config["settings"]["pipeline"]["sparse_global"]["enabled"]
    assert not sparse.raw_config["settings"]["pipeline"]["dense_global"]["enabled"]
    assert sparse.raw_config["settings"]["pipeline"]["sparse_global"]["enabled"]
    assert not sparse.raw_config["settings"]["pipeline"]["content_rrf"]["enabled"]
    assert hybrid.raw_config["settings"]["pipeline"]["content_rrf"]["enabled"]

    same_language = deepcopy(dense.raw_config)
    same_language["settings"]["pipeline"]["language_policy"] = (
        "same_language_only"
    )
    resolved_same = resolve_system_config(same_language)
    assert resolved_same.config_hash != dense.config_hash

    chunker_suite = yaml.safe_load(
        (
            repo_root
            / "configs/thesis/experiments/retrieval_chunking_hybrid.yaml"
        ).read_text(),
    )
    assert len(chunker_suite["experiments"]) == 6
    assert chunker_suite["defaults"]["final_evaluation"]["target_levels"] == [
        "page",
        "chunk",
    ]
    for experiment in chunker_suite["experiments"]:
        assert experiment["system_config"].endswith("modern_hybrid.yaml")
        assert experiment["splits"] == ["development"]
        assert "settings.pipeline.representation.chunker_id" in experiment[
            "overrides"
        ]
        assert set(experiment["overrides"]) <= {
            "settings.pipeline.representation.chunker_id",
            "settings.pipeline.representation.chunk_artifact_set_manifest_path",
        }

    assert "test" not in canonical_json(chunker_suite)

    language_suite = yaml.safe_load(
        (
            repo_root
            / "configs/thesis/experiments/retrieval_language_policy.yaml"
        ).read_text(),
    )
    assert language_suite["defaults"]["top_k"] == 5
    assert "query_limit" not in language_suite["defaults"]
    assert language_suite["defaults"]["final_evaluation"]["target_levels"] == [
        "page",
        "chunk",
    ]
    policies = [
        experiment["overrides"]["settings.pipeline.language_policy"]
        for experiment in language_suite["experiments"]
    ]
    assert policies == [
        "same_language_only",
        "cross_lingual",
        "query_language_complete",
        "cross_lingual",
    ]
    assert language_suite["experiments"][-1]["overrides"] == {
        "settings.pipeline.language_policy": "cross_lingual",
        "settings.pipeline.counterpart_deduplication": "highest_ranked_page",
    }
    assert language_suite["comparisons"] == [
        {
            "comparison_id": "language_policy",
            "experiment_ids": [
                "language_same_only_hybrid",
                "language_cross_lingual_hybrid",
                "language_query_complete_hybrid",
                "language_cross_lingual_counterpart_dedup_hybrid",
            ],
        },
    ]

    rrf_suite = yaml.safe_load(
        (
            repo_root
            / "configs/thesis/experiments/retrieval_rrf_constant.yaml"
        ).read_text(),
    )
    assert rrf_suite["defaults"]["top_k"] == 5
    assert "query_limit" not in rrf_suite["defaults"]
    assert rrf_suite["defaults"]["final_evaluation"]["target_levels"] == [
        "page",
        "chunk",
    ]
    assert [
        experiment["overrides"]["settings.pipeline.content_rrf.rank_constant"]
        for experiment in rrf_suite["experiments"]
    ] == [10, 30, 60, 100]
    assert [
        experiment["overrides"]["settings.pipeline.content_rrf.component_id"]
        for experiment in rrf_suite["experiments"]
    ] == [
        "ordinary_rrf_k10_v1",
        "ordinary_rrf_k30_v1",
        "ordinary_rrf_k60_v1",
        "ordinary_rrf_k100_v1",
    ]
    for experiment in rrf_suite["experiments"]:
        assert experiment["splits"] == ["development"]
        assert experiment["overrides"][
            "settings.pipeline.representation.chunker_id"
        ] == "section_preferred_800_100"
        assert experiment["overrides"]["settings.pipeline.language_policy"] == (
            "query_language_complete"
        )
    assert rrf_suite["comparisons"] == [
        {
            "comparison_id": "content_rrf_constant",
            "experiment_ids": [
                "content_rrf_k10",
                "content_rrf_k30",
                "content_rrf_k60",
                "content_rrf_k100",
            ],
        },
    ]
    assert "test" not in canonical_json(rrf_suite)


def test_ollama_embedding_config_rejects_public_production_endpoint() -> None:
    config = _config(dense=True, sparse=False)
    config["settings"]["pipeline"]["dense_global"]["model"]["endpoint"] = (
        "https://api.example.com"
    )
    with pytest.raises(DenseIndexError, match="local or on a private IP"):
        ModernRetrievalSystem(config)


def test_fake_provider_satisfies_embedding_protocol() -> None:
    assert isinstance(_FakeEmbeddingProvider(), EmbeddingProvider)
