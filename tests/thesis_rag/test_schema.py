import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from thesis_rag.eval.relevance_schemas import QrelRecord
from thesis_rag.interfaces import AnswerResult, RetrievalResult, StageTrace
from thesis_rag.schema import (
    CHUNK_SCHEMA_VERSION,
    CONTENT_BLOCK_SCHEMA_VERSION,
    IMAGE_SCHEMA_VERSION,
    JsonlValidationError,
    PAGE_SCHEMA_VERSION,
    PAGE_TABLE_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    CORPUS_CONFIG_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    SECTION_TABLE_SCHEMA_VERSION,
    SOURCE_FILE_SCHEMA_VERSION,
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    SYSTEM_CONFIG_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
    UnsupportedSchemaVersionError,
    config_hash,
    hash_text,
    prompt_hash,
    read_jsonl,
    stable_id,
    validate_jsonl,
    validate_record_shape,
    write_jsonl,
)


def test_schema_import_does_not_eagerly_import_domain_owners() -> None:
    script = """
import sys
import thesis_rag.schema
forbidden = {
    'thesis_rag.corpus.snapshot',
    'thesis_rag.eval.suite',
    'thesis_rag.eval.suite_judge',
    'thesis_rag.eval.suite_metrics',
    'thesis_rag.eval.suite_report',
}
loaded = forbidden.intersection(sys.modules)
if loaded:
    raise SystemExit(f'eager schema imports: {sorted(loaded)}')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _expanded_corpus_config_record() -> dict:
    return {
        "schema_version": CORPUS_CONFIG_SCHEMA_VERSION,
        "corpus_id": "az_fixture",
        "source_root": "data/az/scraper/topics",
        "mapping_path": "data/az/mapping.csv",
        "manual_utterances_path": "data/az/manual_utterances.csv",
        "output_dir": "data/thesis/corpus",
        "languages": ["de", "en"],
        "additional_files": [
            {
                "path": "data/az/blacklist.csv",
                "role": "az_blacklist",
                "required": True,
            },
        ],
        "snapshot": {
            "include_versions": True,
            "include_page_html": True,
            "archive_path": "data/thesis/corpus/raw_source_archive.tar.zst",
            "source_manifest_path": "data/thesis/corpus/source_files.jsonl",
            "manifest_path": "data/thesis/corpus/snapshot_manifest.json",
        },
        "parser": {
            "content_files": {"de": "content.html", "en": "content_english.html"},
            "page_files": {"de": "page.html", "en": "page_english.html"},
        },
        "section_context_groups": {
            "output_path": "data/thesis/corpus/section_context_groups.jsonl",
            "micro_max_effective_chars": 200,
            "target_group_effective_chars": 400,
            "max_group_effective_chars": 1600,
            "max_focus_sections": 4,
            "lead_in_max_chars": 120,
        },
        "titles": {
            "resolution_order": ["manual", "page_html", "extracted", "intent"],
            "generated_titles_enabled": False,
        },
        "keywords": {"limit": 12, "ngram_range": [1, 3], "max_features": 50_000},
        "chunking": {
            "output_dir": "data/thesis/corpus/chunks",
            "chunkers": [
                {
                    "chunker_id": "fixed_char_800_100",
                    "mode": "fixed_char",
                    "output_path": "data/thesis/corpus/chunks/fixed_char_800_100.jsonl",
                    "target_chars": 800,
                    "overlap_chars": 100,
                },
                {
                    "chunker_id": "section_preferred_800_100",
                    "mode": "section_preferred",
                    "output_path": "data/thesis/corpus/chunks/section_preferred_800_100.jsonl",
                    "target_chars": 800,
                    "overlap_chars": 100,
                    "preferred_min_chars": 400,
                    "hard_max_chars": 1000,
                },
            ],
        },
        "tables": {
            "extract": True,
            "output_path": "data/thesis/corpus/tables.jsonl",
        },
        "media": {
            "extract_images": True,
            "extract_videos": True,
            "images_output_path": "data/thesis/corpus/images.jsonl",
            "media_descriptions_output_path": (
                "data/thesis/corpus/media_descriptions.jsonl"
            ),
        },
        "summaries": {
            "mode": "none",
            "targets": ["page", "context_group"],
            "allowed_modes": ["none", "extractive", "local-llm"],
            "prompts": {
                "page": {
                    "prompt_id": "page_summary_v1",
                    "path": "configs/thesis/prompts/page_summary.md",
                },
                "context_group": {
                    "prompt_id": "context_group_summary_v1",
                    "path": "configs/thesis/prompts/context_group_summary.md",
                },
            },
            "output_dir": "data/thesis/summaries",
            "local_llm": {
                "required_for_thesis_results": True,
                "model_id": None,
                "model_revision": None,
            },
        },
        "page_table": {"output_path": "data/thesis/corpus/page_table.jsonl"},
        "section_table": {"output_path": "data/thesis/corpus/section_table.jsonl"},
        "quality": {
            "output_path": "data/thesis/corpus/page_quality.jsonl",
            "report_path": "data/thesis/corpus/corpus_quality_report.json",
            "short_text_threshold": 200,
            "error_template_phrases": [],
            "count_change_review_fraction": 0.1,
            "large_duplicate_cluster_threshold": 10,
        },
    }


def _snapshot_manifest_record() -> dict:
    digest = "sha256:" + ("a" * 64)
    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "corpus_id": "az_fixture",
        "corpus_hash": digest,
        "source_snapshot_hash": digest,
        "source_manifest_hash": digest,
        "archive_hash": digest,
        "artifact_hashes": {},
        "artifact_record_counts": {},
        "artifact_paths": {},
        "parser_version": "html_parser_fixture",
        "parser_config_hash": digest,
        "extraction_config_hash": digest,
        "artifact_config_hashes": {"chunking": digest},
        "archive_compression": "gzip",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "source_file_count": 0,
        "source_counts_by_role": {},
        "source_counts_by_language": {},
        "page_count": 0,
        "section_count": 0,
        "content_block_count": 0,
        "table_count": 0,
        "image_count": 0,
        "section_context_groups": None,
    }


def test_stable_ids_and_hashes_are_deterministic() -> None:
    assert stable_id("page", "a", {"b": 1}) == stable_id("page", "a", {"b": 1})
    assert stable_id("page", "a") != stable_id("page", "b")

    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert prompt_hash("one prompt") != prompt_hash("another prompt")
    assert hash_text("same") == hash_text("same")


def test_jsonl_validation_roundtrip(tmp_path) -> None:
    path = tmp_path / "qrels.jsonl"
    record = QrelRecord(
        query_id="q1",
        target_id="page:1",
        target_level="page",
        target_space_id="page",
        score=3,
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        view="hybrid",
    ).to_dict()

    write_jsonl(path, [record])

    assert validate_jsonl(path, expected_schema_version="qrel_v3") == 1
    assert read_jsonl(path, expected_schema_version="qrel_v3") == [
        record,
    ]


def test_jsonl_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(JsonlValidationError):
        read_jsonl(path)


def test_jsonl_rejects_invalid_record_shape(tmp_path) -> None:
    path = tmp_path / "invalid_qrel.jsonl"
    path.write_text(
        '{"schema_version":"qrel_v3"}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlValidationError, match="Invalid qrel_v3"):
        read_jsonl(path)


def test_unsupported_schema_version_fails_clearly(tmp_path) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_text('{"schema_version":"future_v9"}\n', encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersionError, match="Unsupported schema_version"):
        read_jsonl(path)


@pytest.mark.parametrize(
    "legacy_version",
    [
        "candidate_pool_v1",
        "candidate_pool_v2",
        "judgment_v1",
        "judgment_v2",
        "qrel_v1",
        "relevance_metrics_v2",
    ],
)
def test_legacy_relevance_contracts_are_not_silently_supported(
    tmp_path,
    legacy_version: str,
) -> None:
    path = tmp_path / f"{legacy_version}.jsonl"
    path.write_text(
        f'{{"schema_version":"{legacy_version}"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedSchemaVersionError):
        read_jsonl(path)


def test_manifest_and_config_schema_versions_validate_real_shapes() -> None:
    validate_record_shape(
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": "run1",
            "system_id": "fixture",
            "config_hash": "sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        },
    )
    validate_record_shape(_snapshot_manifest_record())
    validate_record_shape(
        _expanded_corpus_config_record(),
    )
    validate_record_shape(
        {
            "schema_version": SYSTEM_CONFIG_SCHEMA_VERSION,
            "system_id": "fixture",
            "class_path": "tests.thesis_rag.fixtures.registry_system.FixtureRagSystem",
            "settings": {},
        },
    )
    validate_record_shape(
        {
            "schema_version": SOURCE_RELATIONSHIP_SCHEMA_VERSION,
            "source_relationship_id": "source_relationship:fixture",
            "relationship_kind": "fragment_subrecord",
            "member_page_ids": ["page:base", "page:fragment"],
            "confidence": "deterministic",
            "evidence": {
                "parent_page_id": "page:base",
                "fragment_page_id": "page:fragment",
            },
            "source_relationship_rule_version": "source_relationship_rules_v1",
            "construction_config_hash": "sha256:62598ffd108a5404ad7240935eb8f2ef2b7cef702b2bf7962375b09e9afb176c",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        },
    )
    validate_record_shape(
        {
            "schema_version": "question_v1",
            "query_id": "q-table",
            "text": "What fee is listed in the table?",
            "language": "en",
            "source_targets": [
                {
                    "target_level": "span",
                    "target_id": None,
                    "page_id": "page:1",
                    "section_id": "section:1",
                    "char_start": 0,
                    "char_end": 10,
                    "text_hash": "sha256:37a0025aa1f2e29f9f88f59f961e44a320849dfc79a85ef3c02654d50d2e7480",
                    "metadata": {
                        "table_id": "table:1",
                        "table_cell_refs": ["r2c1"],
                        "lexical_overlap": {
                            "score": 0.4,
                            "bucket": "medium",
                        },
                    },
                },
            ],
            "acceptable_targets": [],
        },
    )

    with pytest.raises(JsonlValidationError, match="Run manifest"):
        validate_record_shape({"schema_version": RUN_MANIFEST_SCHEMA_VERSION})

    with pytest.raises(JsonlValidationError, match="Snapshot manifest"):
        validate_record_shape(
            {
                "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
                "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
            },
        )


def test_checked_in_corpus_config_matches_slice_one_contract() -> None:
    config_path = Path("configs/thesis/corpus.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    validate_record_shape(config)

    legacy = dict(config)
    legacy["schema_version"] = "corpus_config_v1"
    with pytest.raises(UnsupportedSchemaVersionError, match="Unsupported"):
        validate_record_shape(legacy)

    assert {item["role"] for item in config["additional_files"]} == {
        "az_blacklist",
        "az_overwrite_topics",
        "az_translations",
    }
    assert [
        chunker["chunker_id"]
        for chunker in config["chunking"]["chunkers"]
    ] == [
        "fixed_char_800_100",
        "recursive_page_800_100",
        "section_strict_800_100",
        "section_preferred_800_100",
    ]
    assert config["snapshot"]["archive_path"].endswith("raw_source_archive.tar.zst")
    assert config["summaries"]["mode"] == "none"
    assert config["tables"]["extract"] is True
    assert config["media"]["extract_images"] is True
    assert config["page_table"]["output_path"].endswith("page_table.jsonl")
    assert config["section_table"]["output_path"].endswith("section_table.jsonl")


def test_corpus_artifact_schema_versions_validate_real_shapes() -> None:
    artifact_records = [
        {
            "schema_version": SOURCE_FILE_SCHEMA_VERSION,
            "source_path": "data/az/scraper/topics/topic/content.html",
            "archive_path": "data/az/scraper/topics/topic/content.html",
            "role": "content_html",
            "byte_size": 123,
            "file_hash": "sha256:3b9c358f36f0a31b6ad3e14f309c7cf198ac9246e8316f9ce543d5b19ac02b80",
            "language": "de",
            "intent": "topic",
        },
        {
            "schema_version": PAGE_SCHEMA_VERSION,
            "page_key": "page_key:1",
            "page_id": "page:1",
            "language": "de",
            "intent": "topic",
            "source_path": "data/az/scraper/topics/topic/content.html",
            "page_source_path": "data/az/scraper/topics/topic/page.html",
            "url": "https://www.tu.berlin/topic",
            "canonical_url": "https://www.tu.berlin/topic",
            "source_url": "https://www.tu.berlin/topic#c1",
            "title": "Topic",
            "meta_description": None,
            "page_html_language": "de",
            "hreflang": {},
            "breadcrumbs": [],
            "text": "Page text",
            "text_hash": "sha256:3660315a9af3df255d8f19ab077e4797822b41488a0e2a04bc6af71213c23274",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
            "content_source_hash": "sha256:3660315a9af3df255d8f19ab077e4797822b41488a0e2a04bc6af71213c23274",
            "page_source_hash": "sha256:3b9c358f36f0a31b6ad3e14f309c7cf198ac9246e8316f9ce543d5b19ac02b80",
            "content_flags": ["prose"],
        },
        {
            "schema_version": SECTION_SCHEMA_VERSION,
            "section_key": "section_key:1",
            "section_id": "section:1",
            "page_key": "page_key:1",
            "page_id": "page:1",
            "language": "de",
            "intent": "topic",
            "source_path": "data/az/scraper/topics/topic/content.html",
            "section_index": 0,
            "url": "https://www.tu.berlin/topic",
            "title": "Topic",
                "heading_path": ["Topic", "Details"],
                "heading_text": None,
                "heading_level": None,
                "parent_section_id": None,
                "ancestor_section_ids": [],
                "frame_ids": [],
                "text": "Section text",
                "text_hash": "sha256:3fb45623cb3075a8af9a2f81ef3beaa4038541cd71f08e4bf6c79c088f7a4026",
                "body_text": "Section text",
                "body_text_hash": "sha256:3fb45623cb3075a8af9a2f81ef3beaa4038541cd71f08e4bf6c79c088f7a4026",
                "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
                "char_start": 0,
                "char_end": 12,
                "heading_char_start": None,
                "heading_char_end": None,
                "body_char_start": 0,
                "body_char_end": 12,
                "content_flags": ["prose"],
                "structure_quality_flags": ["body_only"],
            },
            {
                "schema_version": CONTENT_BLOCK_SCHEMA_VERSION,
                "block_id": "content_block:1",
                "block_index": 0,
                "section_block_index": 0,
                "page_id": "page:1",
                "section_id": "section:1",
                "language": "de",
                "intent": "topic",
                "source_path": "data/az/scraper/topics/topic/content.html",
                "kind": "text",
                "text": "Section text",
                "text_hash": "sha256:3fb45623cb3075a8af9a2f81ef3beaa4038541cd71f08e4bf6c79c088f7a4026",
                "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
                "char_start": 0,
                "char_end": 12,
                "section_char_start": 0,
                "section_char_end": 12,
                "container_path": [
                    {
                        "tag": "div",
                        "element_id": "c1",
                        "container_types": ["frame", "frame--type-text"],
                    },
                ],
                "content_flags": ["prose"],
            },
        {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "chunk_id": "chunk:1",
            "chunker_id": "fixed_char_800_100",
            "chunker_mode": "fixed_char",
            "chunker_config_hash": "sha256:5b3e3a38efa67b7d6a8377f9003a7893e407514da98116a59bc991959a8748c6",
            "chunk_index": 0,
            "page_key": "page_key:1",
            "page_id": "page:1",
            "section_key": "section_key:1",
            "section_id": "section:1",
            "language": "de",
            "intent": "topic",
            "source_path": "data/az/scraper/topics/topic/content.html",
            "url": "https://www.tu.berlin/topic",
            "title": "Topic",
            "heading_path": ["Topic", "Details"],
            "text": "Chunk text",
            "text_hash": "sha256:6c87f68371b28954707ebb92afee7ccffb74c6f71ec8fea8a98cf6104289585b",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
            "char_start": 0,
            "char_end": 10,
            "section_char_start": 0,
            "section_char_end": 10,
            "boundary_start_kind": "page",
            "boundary_end_kind": "page",
            "source_page_text_hash": "sha256:3660315a9af3df255d8f19ab077e4797822b41488a0e2a04bc6af71213c23274",
            "source_section_text_hash": "sha256:3fb45623cb3075a8af9a2f81ef3beaa4038541cd71f08e4bf6c79c088f7a4026",
            "source_artifact_hashes": {},
            "overlapping_section_ids": ["section:1"],
            "overlapping_context_group_ids": ["section_context_group:1"],
            "overlapping_content_block_ids": ["content_block:1"],
            "equivalence_hint": "chunk_equivalence:fixture",
        },
        {
            "schema_version": TABLE_SCHEMA_VERSION,
            "table_id": "table:1",
            "table_index": 0,
            "page_id": "page:1",
            "section_id": "section:1",
            "language": "de",
            "intent": "topic",
            "source_path": "data/az/scraper/topics/topic/content.html",
            "caption": None,
            "text": "Header: value",
            "text_hash": "sha256:0d4fc4a78d3706edccafb665a8b2fdd9309e82c78625bb0f2b8e7bb9e1c4d21c",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
            "char_start": 0,
            "char_end": 13,
        },
        {
            "schema_version": IMAGE_SCHEMA_VERSION,
            "image_id": "image:1",
            "image_index": 0,
            "page_id": "page:1",
            "section_id": "section:1",
            "language": "de",
            "intent": "topic",
            "source_path": "data/az/scraper/topics/topic/content.html",
            "media_type": "image",
            "media_url": None,
            "alt_text": None,
            "title": None,
            "caption": None,
            "copyright": None,
            "nearby_text": None,
            "media_source_hash": "sha256:721c9525ade2ea8903d343ef25cf68b9bf4ab0aad56bb7b01fbe48d09bc7fcf4",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
            "char_start": None,
            "char_end": None,
        },
        {
            "schema_version": PAGE_TABLE_SCHEMA_VERSION,
            "page_key": "page_key:1",
            "page_id": "page:1",
            "language": "de",
            "intent": "topic",
            "url": "https://www.tu.berlin/topic",
            "title": "Topic",
            "mapping_title": "Topic",
            "breadcrumbs": [],
            "heading_list": ["Topic", "Details"],
            "parent_intent": None,
            "depth": 0,
            "etag": None,
            "content_flags": ["prose", "table"],
            "table_count": 1,
            "image_count": 1,
            "section_count": 1,
            "source_paths": ["data/az/scraper/topics/topic/content.html"],
            "keywords": ["topic"],
            "text_hash": "sha256:2eb5a43a6d8092afccbf5f758f1e635a87555a2241fc60875539d6b6bde29b27",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        },
        {
            "schema_version": SECTION_TABLE_SCHEMA_VERSION,
            "section_key": "section_key:1",
            "section_id": "section:1",
            "page_key": "page_key:1",
            "page_id": "page:1",
            "language": "de",
            "intent": "topic",
            "url": "https://www.tu.berlin/topic",
            "title": "Topic",
            "heading_path": ["Topic", "Details"],
            "section_index": 0,
            "section_type": "prose",
            "content_flags": ["prose"],
            "keywords": ["topic"],
            "table_ids": ["table:1"],
            "image_ids": ["image:1"],
            "text_hash": "sha256:2c262362ff994901a276d295a0f33584ef065153872d7044e5195a320bd34b26",
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
            "char_start": 0,
            "char_end": 12,
        },
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "summary_id": "summary:1",
            "target_level": "page",
            "page_id": "page:1",
            "summary": "Short page summary.",
            "source_input_hash": "sha256:3660315a9af3df255d8f19ab077e4797822b41488a0e2a04bc6af71213c23274",
            "source_dependency_hash": "sha256:5b3e3a38efa67b7d6a8377f9003a7893e407514da98116a59bc991959a8748c6",
            "source_section_ids": [],
            "prompt_id": "page_summary_v1",
            "prompt_hash": "sha256:cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
            "model_id": "extractive_first_sentences_v1",
            "local_model_path": None,
            "model_revision": None,
            "temperature": None,
            "generated_at_utc": None,
            "generation_method": "extractive",
            "generation_config_hash": "sha256:e661f4c935e8a5a83349afb5e347695c2e972e967b50efcd618f93b0b7b4c24b",
            "representation_kind": "extractive_summary",
            "finish_reason": None,
            "usage": {},
            "generation_metadata": {"llm_called": False},
            "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        },
    ]

    for record in artifact_records:
        validate_record_shape(record)


def test_corpus_artifact_records_reject_unstable_or_incomplete_shapes() -> None:
    broken_source = {
        "schema_version": SOURCE_FILE_SCHEMA_VERSION,
        "source_path": "data/az/scraper/topics/topic/content.html",
        "archive_path": "data/az/scraper/topics/topic/content.html",
        "role": "content_html",
        "byte_size": 123,
        "file_hash": "not-a-hash",
    }
    with pytest.raises(JsonlValidationError, match="file_hash"):
        validate_record_shape(broken_source)

    broken_chunk = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_id": "chunk:1",
        "chunker_id": "",
        "chunk_index": 0,
        "page_key": "page_key:1",
        "page_id": "page:1",
        "section_key": "section_key:1",
        "section_id": "section:1",
        "language": "de",
        "intent": "topic",
        "source_path": "data/az/scraper/topics/topic/content.html",
        "text": "Chunk text",
        "text_hash": "sha256:6c87f68371b28954707ebb92afee7ccffb74c6f71ec8fea8a98cf6104289585b",
        "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        "char_start": 0,
        "char_end": 10,
    }
    with pytest.raises(JsonlValidationError, match="chunker_id"):
        validate_record_shape(broken_chunk)

    broken_offsets = dict(broken_chunk)
    broken_offsets["chunker_id"] = "fixed_char_800_100"
    broken_offsets["char_start"] = 5
    broken_offsets["char_end"] = 3
    with pytest.raises(JsonlValidationError, match="char_end"):
        validate_record_shape(broken_offsets)

    broken_summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_id": "summary:1",
        "target_level": "context_group",
        "page_id": "page:1",
        "summary": "Missing context group id.",
        "source_input_hash": "sha256:3660315a9af3df255d8f19ab077e4797822b41488a0e2a04bc6af71213c23274",
        "source_dependency_hash": "sha256:5b3e3a38efa67b7d6a8377f9003a7893e407514da98116a59bc991959a8748c6",
        "source_section_ids": ["section:1"],
        "prompt_id": "page_summary_v1",
        "prompt_hash": "sha256:cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
        "model_id": "extractive_first_sentences_v1",
        "local_model_path": None,
        "model_revision": None,
        "temperature": None,
        "generated_at_utc": None,
        "generation_method": "extractive",
        "generation_config_hash": "sha256:e661f4c935e8a5a83349afb5e347695c2e972e967b50efcd618f93b0b7b4c24b",
        "representation_kind": "extractive_summary",
        "finish_reason": None,
        "usage": {},
        "generation_metadata": {"llm_called": False},
        "corpus_hash": "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
    }
    with pytest.raises(JsonlValidationError, match="context_group_id"):
        validate_record_shape(broken_summary)


def test_qrels_default_to_standard_four_column_tsv() -> None:
    qrel = QrelRecord(
        query_id="q1",
        target_id="page:1",
        target_level="page",
        target_space_id="page",
        view="hybrid",
        score=3,
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
    )

    assert qrel.to_tsv_row() == "q1\t0\tpage:1\t3"


def test_store_hash_fields_require_sha256_prefix() -> None:
    with pytest.raises(ValueError, match="corpus_hash"):
        QrelRecord(
            query_id="q1",
            target_id="page:1",
            target_level="page",
            target_space_id="page",
            view="hybrid",
            score=3,
            corpus_hash="not-a-hash",
        )

    retrieval = RetrievalResult(
        query_id="q1",
        system_id="fixture",
        run_id="run1",
        items=[],
        context_items=[],
        direct_context_items=[],
        enabled_stages={},
        stage_traces=[],
        debug={},
        timings_ms={},
        config_hash="sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
    ).to_dict()
    retrieval["config_hash"] = "not-a-hash"
    with pytest.raises(JsonlValidationError, match="config_hash"):
        validate_record_shape(retrieval)


def test_result_records_can_be_written_as_jsonl(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    trace = StageTrace(
        stage="dense",
        enabled=True,
        mode="fixture",
        input_count=0,
        output_count=0,
    )
    retrieval = RetrievalResult(
        query_id="q1",
        system_id="fixture",
        run_id="run1",
        items=[],
        context_items=[],
        direct_context_items=[],
        enabled_stages={},
        stage_traces=[trace],
        debug={},
        timings_ms={},
        config_hash="sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
    )
    answer = AnswerResult(
        query_id="q1",
        system_id="fixture",
        answer="",
        context_item_ids=[],
        prompt_hash="sha256:cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
        model_id="fixture",
        model_revision=None,
        timings_ms={},
    )

    write_jsonl(path, [retrieval.to_dict(), answer.to_dict()], append=False)

    assert [record["schema_version"] for record in read_jsonl(path)] == [
        "retrieval_result_v5",
        "answer_result_v2",
    ]

    legacy_payload = retrieval.to_dict()
    legacy_payload["schema_version"] = "retrieval_result_v4"
    with pytest.raises(UnsupportedSchemaVersionError, match="Unsupported"):
        validate_record_shape(legacy_payload)
