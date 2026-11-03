from __future__ import annotations

import csv
import json
import shutil
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

import thesis_rag.corpus.snapshot as snapshot_module
from thesis_rag.cli import app
from thesis_rag.corpus.chunk_artifacts import (
    ChunkArtifactSetBuilder,
    ChunkArtifactSetError,
    ChunkArtifactSetManifest,
    resolve_chunk_artifact,
)
from thesis_rag.indexing.catalog import ChunkDocumentCatalog
from thesis_rag.retrieval.config import RepresentationConfig
from thesis_rag.corpus.inspector import (
    CorpusInspectionError,
    CorpusInspector,
    _file_link_text,
    render_page_detail,
)
from thesis_rag.corpus.snapshot import (
    SnapshotBuildError,
    SnapshotBuilder,
    SnapshotManifest,
    SnapshotVerificationError,
    SnapshotVerifier,
)
from thesis_rag.corpus.summary_job import (
    SummaryArtifactBuilder,
    SummaryArtifactVerifier,
    SummaryJobError,
    SummaryManifest,
    SummaryVerificationError,
    _summary_source_set_hash,
)
from thesis_rag.eval.relevance import _selected_chunkers
from thesis_rag.eval.relevance_types import RelevanceError
from thesis_rag.corpus.summaries import SummaryBuildError
from thesis_rag.schema import (
    SOURCE_FILE_SCHEMA_VERSION,
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    hash_file,
    hash_text,
    read_jsonl,
    validate_record_shape,
    write_jsonl,
)


MAPPING_COLUMNS = [
    "depth",
    "intent",
    "extracted_title_deutsch",
    "extracted_title_english",
    "generated_title_deutsch",
    "generated_title_english",
    "manual_title_deutsch",
    "manual_title_english",
    "link",
    "mapping",
    "parent",
    "etag",
]


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_snapshot_fixture_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n",
        encoding="utf-8",
    )
    (repo_root / "thesis_rag").mkdir()
    (repo_root / "configs/thesis").mkdir(parents=True)
    (repo_root / "configs/thesis/prompts").mkdir(parents=True)
    (repo_root / "configs/thesis/prompts/page_summary.md").write_text(
        "# page_summary_v1\n\nSummarize the page.\n",
        encoding="utf-8",
    )
    (repo_root / "configs/thesis/prompts/context_group_summary.md").write_text(
        "# context_group_summary_v1\n\nSummarize the context group.\n",
        encoding="utf-8",
    )

    _write_csv(
        repo_root / "data/az/mapping.csv",
        [
            {
                "depth": "0",
                "intent": "child-topic",
                "extracted_title_deutsch": "Kind",
                "extracted_title_english": "Child",
                "generated_title_deutsch": "",
                "generated_title_english": "",
                "manual_title_deutsch": "Kind manuell",
                "manual_title_english": "Child manual",
                "link": "https://www.tu.berlin/kind",
                "mapping": "https://www.tu.berlin/en/child",
                "parent": "",
                "etag": "child-etag",
            },
            {
                "depth": "0",
                "intent": "root-topic",
                "extracted_title_deutsch": "Wurzel",
                "extracted_title_english": "Root",
                "generated_title_deutsch": "",
                "generated_title_english": "",
                "manual_title_deutsch": "",
                "manual_title_english": "",
                "link": "https://www.tu.berlin/root",
                "mapping": "https://www.tu.berlin/en/root",
                "parent": "",
                "etag": "root-etag",
            },
        ],
        MAPPING_COLUMNS,
    )
    _write_csv(
        repo_root / "data/az/manual_utterances.csv",
        [
            {
                "utterance": "Wie frage ich?",
                "link": "https://www.tu.berlin/kind",
                "intent": "child-topic",
            },
        ],
        ["utterance", "link", "intent"],
    )
    _write_csv(
        repo_root / "data/az/blacklist.csv",
        [{"intent": "ignored"}],
        ["intent"],
    )
    (repo_root / "data/az/translations.json").write_text(
        '{"child-topic": "Child"}\n',
        encoding="utf-8",
    )
    child_dir = repo_root / "data/az/scraper/topics/child-topic"
    child_dir.mkdir(parents=True)
    (child_dir / "content.html").write_text(
        """
        <main>
          <h1>Kind manuell</h1>
          <p>Deutscher Absatz.</p>
          <table><caption>Gebuehren</caption><tr><th>Name</th><th>Wert</th></tr>
          <tr><td>Preis</td><td>10 Euro</td></tr></table>
          <img src="bild.jpg" alt="Bildbeschreibung" title="Bildtitel" />
        </main>
        """,
        encoding="utf-8",
    )
    (child_dir / "content_english.html").write_text(
        "<main><h1>Child manual</h1><p>English paragraph.</p></main>",
        encoding="utf-8",
    )
    (child_dir / "page.html").write_text(
        """
        <html lang="de"><head><title>Kind Seite</title>
        <link rel="canonical" href="https://www.tu.berlin/kind" />
        <meta name="description" content="Beschreibung" /></head></html>
        """,
        encoding="utf-8",
    )
    (child_dir / "page_english.html").write_text(
        """
        <html lang="en"><head><title>Child Page</title>
        <link rel="canonical" href="https://www.tu.berlin/en/child" />
        </head></html>
        """,
        encoding="utf-8",
    )
    _write_csv(child_dir / "versions.csv", [{"version": "1"}], ["version"])

    root_dir = repo_root / "data/az/scraper/topics/root-topic"
    root_dir.mkdir(parents=True)
    (root_dir / "content.html").write_text(
        "<main><h1>Wurzel</h1><p>Root text.</p></main>",
        encoding="utf-8",
    )
    (root_dir / "page.html").write_text(
        "<html lang='de'><head><title>Wurzel Seite</title></head></html>",
        encoding="utf-8",
    )
    _write_csv(root_dir / "versions.csv", [{"version": "1"}], ["version"])

    config = {
        "schema_version": "corpus_config_v5",
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
            {
                "path": "data/az/translations.json",
                "role": "az_translations",
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
            "fallback_section_heading": "__page__",
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
                    "chunker_id": "recursive_page_800_100",
                    "mode": "recursive_page",
                    "output_path": "data/thesis/corpus/chunks/recursive_page_800_100.jsonl",
                    "target_chars": 800,
                    "overlap_chars": 100,
                    "preferred_min_chars": 400,
                    "hard_max_chars": 1000,
                },
                {
                    "chunker_id": "section_strict_800_100",
                    "mode": "section_strict",
                    "output_path": "data/thesis/corpus/chunks/section_strict_800_100.jsonl",
                    "target_chars": 800,
                    "overlap_chars": 100,
                    "preferred_min_chars": 400,
                    "hard_max_chars": 1000,
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
        "tables": {"extract": True, "output_path": "data/thesis/corpus/tables.jsonl"},
        "media": {
            "extract_images": True,
            "extract_videos": True,
            "images_output_path": "data/thesis/corpus/images.jsonl",
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
        },
        "page_table": {"output_path": "data/thesis/corpus/page_table.jsonl"},
        "section_table": {"output_path": "data/thesis/corpus/section_table.jsonl"},
        "quality": {
            "output_path": "data/thesis/corpus/page_quality.jsonl",
            "report_path": "data/thesis/corpus/corpus_quality_report.json",
            "short_text_threshold": 30,
            "error_template_phrases": ["Page not found", "Seite nicht gefunden"],
            "count_change_review_fraction": 0.1,
            "large_duplicate_cluster_threshold": 10,
        },
    }
    (repo_root / "configs/thesis/corpus.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return repo_root


def _manifest(repo_root: Path) -> dict:
    return json.loads(
        (repo_root / "data/thesis/corpus/snapshot_manifest.json").read_text(
            encoding="utf-8",
        ),
    )


def test_snapshot_builder_writes_deterministic_manifest_and_artifacts(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    builder = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    )

    summary = builder.build()
    manifest = _manifest(repo_root)
    validate_record_shape(manifest)
    source_manifest_text = (
        repo_root / "data/thesis/corpus/source_files.jsonl"
    ).read_text(encoding="utf-8")

    assert summary.archive_compression == "gzip"
    assert summary.source_file_count == 12
    assert summary.page_count == 3
    assert summary.section_count == 3
    assert summary.content_block_count > summary.section_count
    assert summary.chunk_counts == {
        chunker_id: manifest["artifact_record_counts"]["chunks"][chunker_id]
        for chunker_id in (
            "fixed_char_800_100",
            "recursive_page_800_100",
            "section_strict_800_100",
            "section_preferred_800_100",
        )
    }
    assert summary.table_count == 1
    assert summary.image_count == 1
    assert manifest["source_counts_by_role"] == {
        "az_blacklist": 1,
        "az_translations": 1,
        "content_html": 3,
        "manual_utterances": 1,
        "mapping": 1,
        "page_html": 3,
        "versions_csv": 2,
    }
    assert manifest["source_counts_by_language"] == {"de": 4, "en": 2}
    assert set(manifest["artifact_hashes"]) == {
        "chunks",
        "content_blocks",
        "images",
        "page_table",
        "page_quality",
        "quality_report",
        "pages",
        "section_table",
        "section_context_groups",
        "sections",
        "source_relationships",
        "tables",
    }
    assert set(manifest["artifact_config_hashes"]) == {
        "chunking",
        "keywords",
        "media",
        "quality",
        "section_context_groups",
        "source_relationships",
        "tables",
    }
    assert manifest["chunking"]["ran"] is True
    assert set(manifest["artifact_hashes"]["chunks"]) == {
        "fixed_char_800_100",
        "recursive_page_800_100",
        "section_strict_800_100",
        "section_preferred_800_100",
    }
    assert set(manifest["artifact_paths"]["chunks"]) == {
        "fixed_char_800_100",
        "recursive_page_800_100",
        "section_strict_800_100",
        "section_preferred_800_100",
    }
    assert [
        chunker["chunker_id"] for chunker in manifest["chunking"]["chunkers"]
    ] == [
        "fixed_char_800_100",
        "recursive_page_800_100",
        "section_strict_800_100",
        "section_preferred_800_100",
    ]

    source_records = read_jsonl(
        repo_root / "data/thesis/corpus/source_files.jsonl",
        expected_schema_version=SOURCE_FILE_SCHEMA_VERSION,
    )
    assert all(not record["archive_path"].startswith("/") for record in source_records)
    assert [
        record["archive_path"]
        for record in source_records
    ] == sorted(record["archive_path"] for record in source_records)
    assert read_jsonl(repo_root / "data/thesis/corpus/pages.jsonl")
    assert read_jsonl(repo_root / "data/thesis/corpus/sections.jsonl")
    assert len(
        read_jsonl(
            repo_root / "data/thesis/corpus/section_context_groups.jsonl",
        ),
    ) == summary.section_context_group_count
    assert len(read_jsonl(repo_root / "data/thesis/corpus/content_blocks.jsonl")) == (
        summary.content_block_count
    )
    assert read_jsonl(repo_root / "data/thesis/corpus/tables.jsonl")
    assert read_jsonl(repo_root / "data/thesis/corpus/images.jsonl")
    assert len(read_jsonl(repo_root / "data/thesis/corpus/page_table.jsonl")) == (
        summary.page_count
    )
    assert len(read_jsonl(repo_root / "data/thesis/corpus/section_table.jsonl")) == (
        summary.section_count
    )
    assert read_jsonl(repo_root / "data/thesis/corpus/source_relationships.jsonl") == []
    assert len(read_jsonl(repo_root / "data/thesis/corpus/page_quality.jsonl")) == (
        summary.page_count
    )
    assert (
        repo_root
        / "data/thesis/corpus/chunks/fixed_char_800_100.jsonl"
    ).exists()
    assert read_jsonl(
        repo_root / "data/thesis/corpus/chunks/recursive_page_800_100.jsonl",
    )
    assert read_jsonl(
        repo_root / "data/thesis/corpus/chunks/section_strict_800_100.jsonl",
    )
    assert read_jsonl(
        repo_root / "data/thesis/corpus/chunks/section_preferred_800_100.jsonl",
    )

    verified = SnapshotVerifier(repo_root=repo_root).verify(
        repo_root / "data/thesis/corpus/snapshot_manifest.json",
    )
    assert verified.corpus_hash == summary.corpus_hash

    rebuilt = builder.build(force=True)
    rebuilt_manifest = _manifest(repo_root)
    assert rebuilt.corpus_hash == summary.corpus_hash
    assert rebuilt_manifest["source_snapshot_hash"] == manifest["source_snapshot_hash"]
    assert rebuilt_manifest["source_manifest_hash"] == manifest["source_manifest_hash"]
    assert rebuilt_manifest["corpus_hash"] == manifest["corpus_hash"]
    assert rebuilt_manifest["archive_hash"] == manifest["archive_hash"]
    assert rebuilt_manifest["artifact_hashes"] == manifest["artifact_hashes"]
    assert (
        rebuilt_manifest["artifact_record_counts"]
        == manifest["artifact_record_counts"]
    )
    assert (
        rebuilt_manifest["artifact_config_hashes"]
        == manifest["artifact_config_hashes"]
    )
    assert rebuilt_manifest["chunking"] == manifest["chunking"]
    assert (
        repo_root / "data/thesis/corpus/source_files.jsonl"
    ).read_text(encoding="utf-8") == source_manifest_text


def test_derived_chunk_set_preserves_frozen_snapshot(tmp_path: Path) -> None:
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    snapshot_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    snapshot_hash = hash_file(snapshot_path)
    config_path = repo_root / "configs/thesis/chunk-set.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "chunk_artifact_set_config_v1",
                "base_snapshot_manifest_path": (
                    "data/thesis/corpus/snapshot_manifest.json"
                ),
                "output_manifest_path": (
                    "data/thesis/corpus/chunk_sets/lengths.json"
                ),
                "chunkers": [
                    {
                        "chunker_id": "section_preferred_600_100",
                        "mode": "section_preferred",
                        "output_path": (
                            "data/thesis/corpus/chunks/"
                            "section_preferred_600_100.jsonl"
                        ),
                        "target_chars": 600,
                        "overlap_chars": 100,
                        "preferred_min_chars": 400,
                        "hard_max_chars": 1000,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    builder = ChunkArtifactSetBuilder(config_path, repo_root=repo_root)

    first = builder.build()
    second = builder.build()

    assert first.reused is False
    assert second.reused is True
    assert first.chunk_artifact_set_id == second.chunk_artifact_set_id
    assert first.chunk_counts["section_preferred_600_100"] > 0
    assert hash_file(snapshot_path) == snapshot_hash
    manifest_path = repo_root / first.manifest_path
    manifest_text = manifest_path.read_text(encoding="utf-8")
    changed_sources = json.loads(manifest_text)
    changed_sources["source_artifact_paths"]["pages"] = changed_sources[
        "source_artifact_paths"
    ]["sections"]
    manifest_path.write_text(json.dumps(changed_sources), encoding="utf-8")
    with pytest.raises(ChunkArtifactSetError, match="source paths changed"):
        builder.build()
    manifest_path.write_text(manifest_text, encoding="utf-8")
    validate_record_shape(
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    catalog = ChunkDocumentCatalog(
        repo_root=repo_root,
        snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json",
        representation=RepresentationConfig(
            chunker_id="section_preferred_600_100",
            chunk_artifact_set_manifest_path=first.manifest_path,
            fields=["text"],
        ),
    )
    catalog.load(expected_corpus_hash=first.corpus_hash)
    assert catalog.documents
    assert catalog.identity.chunk_artifact_set_id == first.chunk_artifact_set_id
    assert catalog.identity.to_index_identity_dict()[
        "chunk_artifact_set_id"
    ] == first.chunk_artifact_set_id
    snapshot = SnapshotManifest.from_dict(
        json.loads(snapshot_path.read_text(encoding="utf-8")),
    )
    selected = _selected_chunkers(
        SimpleNamespace(
            target_spaces=SimpleNamespace(chunker_ids="all_configured"),
        ),
        snapshot,
        snapshot_path=snapshot_path,
        repo_root=repo_root,
        suite_representations={
            "section_preferred_600_100": first.manifest_path,
        },
    )
    assert "section_preferred_600_100" in selected
    assert "fixed_char_800_100" in selected

    legacy_source = {
        "source_kind": "legacy_index_items_v1",
        "chunker_id": "legacy_recursive_character_800_100_min200",
        "config_hash": hash_text("legacy-config"),
        "artifact_hash": hash_text("legacy-items"),
        "output_path": "indexes/thesis/legacy/legacy_index_items.jsonl",
        "record_count": 1,
        "source_manifest_hash": hash_text("legacy-manifest"),
        "corpus_hash": snapshot.corpus_hash,
    }
    selected_with_legacy = _selected_chunkers(
        SimpleNamespace(target_spaces=SimpleNamespace(chunker_ids=[])),
        snapshot,
        snapshot_path=snapshot_path,
        repo_root=repo_root,
        suite_representations={
            "legacy_recursive_character_800_100_min200": legacy_source,
        },
    )
    assert selected_with_legacy[
        "legacy_recursive_character_800_100_min200"
    ]["source_kind"] == "legacy_index_items_v1"
    with pytest.raises(RelevanceError, match="belong to another corpus"):
        _selected_chunkers(
            SimpleNamespace(target_spaces=SimpleNamespace(chunker_ids=[])),
            snapshot,
            snapshot_path=snapshot_path,
            repo_root=repo_root,
            suite_representations={
                "legacy_recursive_character_800_100_min200": {
                    **legacy_source,
                    "corpus_hash": hash_text("other-corpus"),
                },
            },
        )

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["chunker_configs"]["section_preferred_600_100"][
        "target_chars"
    ] = 601
    with pytest.raises(ValueError, match="chunker config hash mismatch"):
        ChunkArtifactSetManifest.from_dict(manifest_payload)

    changed_hashes = dict(snapshot.artifact_hashes)
    changed_hashes["pages"] = hash_text("changed pages")
    changed_snapshot = snapshot.model_copy(
        update={"artifact_hashes": changed_hashes},
    )
    with pytest.raises(ChunkArtifactSetError, match="source hashes changed"):
        resolve_chunk_artifact(
            first.manifest_path,
            "section_preferred_600_100",
            base_snapshot=changed_snapshot,
            base_snapshot_path=snapshot_path,
            repo_root=repo_root,
        )


def test_snapshot_refuses_accidental_overwrite_unless_force_and_supports_dry_run(
    tmp_path,
):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    builder = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    )

    builder.build()

    with pytest.raises(SnapshotBuildError, match="already exist"):
        builder.build()

    dry_run = builder.build(dry_run=True)
    assert dry_run.dry_run is True
    assert dry_run.page_count == 3


def test_failed_force_build_keeps_previous_snapshot(tmp_path, monkeypatch):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    builder = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    )
    builder.build()
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    pages_path = repo_root / "data/thesis/corpus/pages.jsonl"
    previous_manifest = manifest_path.read_bytes()
    previous_pages = pages_path.read_bytes()

    def fail_archive(*args, **kwargs):
        raise SnapshotBuildError("fixture archive failure")

    monkeypatch.setattr(snapshot_module, "_write_archive", fail_archive)

    with pytest.raises(SnapshotBuildError, match="fixture archive failure"):
        builder.build(force=True)

    assert manifest_path.read_bytes() == previous_manifest
    assert pages_path.read_bytes() == previous_pages
    assert list((repo_root / "data/thesis").glob(".corpus-staging-*")) == []


def test_corpus_hash_excludes_derived_artifact_configuration(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    config_path = repo_root / "configs/thesis/corpus.yaml"

    initial = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    page_prompt = repo_root / "configs/thesis/prompts/page_summary.md"
    page_prompt.write_text(
        "# page_summary_v1\n\nUse a completely different summary prompt.\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output_dir"] = "different/output"
    config["snapshot"]["archive_path"] = "different/archive.tar.zst"
    config["snapshot"]["source_manifest_path"] = "different/source_files.jsonl"
    config["snapshot"]["manifest_path"] = "different/snapshot_manifest.json"
    config["chunking"]["output_dir"] = "different/chunks"
    for chunker in config["chunking"]["chunkers"]:
        chunker["output_path"] = f"different/{chunker['chunker_id']}.jsonl"
    config["tables"]["output_path"] = "different/tables.jsonl"
    config["tables"]["extract"] = not config["tables"]["extract"]
    config["media"]["images_output_path"] = "different/images.jsonl"
    config["media"]["extract_images"] = not config["media"]["extract_images"]
    config["media"]["extract_videos"] = not config["media"]["extract_videos"]
    config["summaries"]["output_dir"] = "different/summaries"
    config["summaries"]["local_llm"] = {
        "endpoint": "http://another-machine:11434/v1",
        "model_id": "another-model",
        "temperature": 0.7,
    }
    config["page_table"]["output_path"] = "different/page_table.jsonl"
    config["section_table"]["output_path"] = "different/section_table.jsonl"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    changed_derived_config = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    assert changed_derived_config.corpus_hash == initial.corpus_hash


def test_corpus_hash_changes_only_when_semantic_page_records_change(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    config_path = repo_root / "configs/thesis/corpus.yaml"
    content_path = repo_root / "data/az/scraper/topics/child-topic/content.html"

    initial = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    content_path.write_text(
        content_path.read_text(encoding="utf-8") + "\n<p>Changed source.</p>",
        encoding="utf-8",
    )
    changed_source = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)
    assert changed_source.corpus_hash != initial.corpus_hash

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["parser"]["fallback_section_heading"] = "__different_fallback__"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    changed_parser = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)
    assert changed_parser.corpus_hash == changed_source.corpus_hash


def test_source_snapshot_hash_tracks_archived_metadata_without_invalidating_corpus(
    tmp_path,
):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    manual_path = repo_root / "data/az/manual_utterances.csv"
    content_path = repo_root / "data/az/scraper/topics/child-topic/content.html"

    initial = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    manual_path.write_text(
        manual_path.read_text(encoding="utf-8")
        + "Neue Frage,https://www.tu.berlin/kind,child-topic\n",
        encoding="utf-8",
    )
    changed_metadata = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    assert changed_metadata.source_snapshot_hash != initial.source_snapshot_hash
    assert changed_metadata.corpus_hash == initial.corpus_hash

    content_path.write_text(
        content_path.read_text(encoding="utf-8") + "\n<!-- source-only change -->\n",
        encoding="utf-8",
    )
    changed_raw_html = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    assert (
        changed_raw_html.source_snapshot_hash
        != changed_metadata.source_snapshot_hash
    )
    assert changed_raw_html.corpus_hash == changed_metadata.corpus_hash


def test_source_filename_is_provenance_not_semantic_corpus_identity(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    config_path = repo_root / "configs/thesis/corpus.yaml"
    original_paths = sorted(
        (repo_root / "data/az/scraper/topics").glob("*/content.html"),
    )

    initial = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)
    for original_path in original_paths:
        original_path.rename(original_path.with_name("content-renamed.html"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["parser"]["content_files"]["de"] = "content-renamed.html"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    renamed = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(dry_run=True)

    assert renamed.source_snapshot_hash != initial.source_snapshot_hash
    assert renamed.corpus_hash == initial.corpus_hash


def test_source_file_modification_and_missing_file_make_verification_fail(tmp_path):
    modified_repo = _write_snapshot_fixture_repo(tmp_path / "modified")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=modified_repo,
        compression="gzip",
    ).build()
    content_path = (
        modified_repo / "data/az/scraper/topics/child-topic/content.html"
    )
    content_path.write_text(
        content_path.read_text(encoding="utf-8") + "\n<p>Changed.</p>",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotVerificationError, match="source hash mismatch"):
        SnapshotVerifier(repo_root=modified_repo).verify(
            modified_repo / "data/thesis/corpus/snapshot_manifest.json",
        )

    missing_repo = _write_snapshot_fixture_repo(tmp_path / "missing")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=missing_repo,
        compression="gzip",
    ).build()
    (missing_repo / "data/az/scraper/topics/child-topic/content.html").unlink()

    with pytest.raises(SnapshotVerificationError, match="source file is missing"):
        SnapshotVerifier(repo_root=missing_repo).verify(
            missing_repo / "data/thesis/corpus/snapshot_manifest.json",
        )


def test_archive_hash_mismatch_makes_verification_fail(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    archive_path = repo_root / "data/thesis/corpus/raw_source_archive.tar.zst"
    archive_path.write_bytes(archive_path.read_bytes() + b"corrupt")

    with pytest.raises(SnapshotVerificationError, match="archive hash mismatch"):
        SnapshotVerifier(repo_root=repo_root).verify(
            repo_root / "data/thesis/corpus/snapshot_manifest.json",
        )


def test_archive_only_verification_and_restore_reproduce_source_hashes(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    source_records = read_jsonl(repo_root / "data/thesis/corpus/source_files.jsonl")
    expected_by_path = {
        record["archive_path"]: record["file_hash"] for record in source_records
    }
    shutil.rmtree(repo_root / "data/az/scraper")

    restore_dir = tmp_path / "restore"
    summary = SnapshotVerifier(repo_root=repo_root).verify(
        repo_root / "data/thesis/corpus/snapshot_manifest.json",
        archive_only=True,
        restore_dir=restore_dir,
    )

    assert summary.restored_file_count == len(source_records)
    restored_content = restore_dir / "data/az/scraper/topics/child-topic/content.html"
    assert restored_content.exists()
    assert hash_file(restored_content) == expected_by_path[
        "data/az/scraper/topics/child-topic/content.html"
    ]


def test_verifier_rejects_semantic_corruption_with_updated_artifact_hash(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    pages_path = repo_root / "data/thesis/corpus/pages.jsonl"
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    pages = read_jsonl(pages_path)
    pages[0]["page_id"] = "page:forged"
    write_jsonl(pages_path, pages)
    manifest = _manifest(repo_root)
    manifest["artifact_hashes"]["pages"] = hash_file(pages_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotVerificationError, match="page_id mismatch"):
        SnapshotVerifier(repo_root=repo_root).verify(manifest_path)


def test_verifier_rejects_incomplete_section_routing_references(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    section_table_path = repo_root / "data/thesis/corpus/section_table.jsonl"
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    rows = read_jsonl(section_table_path)
    row = next(item for item in rows if item["table_ids"])
    row["table_ids"] = []
    write_jsonl(section_table_path, rows)
    manifest = _manifest(repo_root)
    manifest["artifact_hashes"]["section_table"] = hash_file(section_table_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotVerificationError, match="table_ids mismatch"):
        SnapshotVerifier(repo_root=repo_root).verify(manifest_path)


def test_verifier_rejects_cross_page_content_block_reference(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    blocks_path = repo_root / "data/thesis/corpus/content_blocks.jsonl"
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    pages = read_jsonl(repo_root / "data/thesis/corpus/pages.jsonl")
    blocks = read_jsonl(blocks_path)
    block = blocks[0]
    block["page_id"] = next(
        page["page_id"] for page in pages if page["page_id"] != block["page_id"]
    )
    write_jsonl(blocks_path, blocks)
    manifest = _manifest(repo_root)
    manifest["artifact_hashes"]["content_blocks"] = hash_file(blocks_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotVerificationError,
        match="belongs to a different page than its section",
    ):
        SnapshotVerifier(repo_root=repo_root).verify(manifest_path)


def test_verifier_rejects_table_span_outside_its_section(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    tables_path = repo_root / "data/thesis/corpus/tables.jsonl"
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    sections = {
        row["section_id"]: row
        for row in read_jsonl(repo_root / "data/thesis/corpus/sections.jsonl")
    }
    tables = read_jsonl(tables_path)
    table = tables[0]
    table["char_end"] = int(sections[table["section_id"]]["char_end"]) + 1
    write_jsonl(tables_path, tables)
    manifest = _manifest(repo_root)
    manifest["artifact_hashes"]["tables"] = hash_file(tables_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotVerificationError,
        match="character span is outside its section",
    ):
        SnapshotVerifier(repo_root=repo_root).verify(manifest_path)


def test_verifier_rejects_chunker_submanifest_count_mismatch(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    manifest = _manifest(repo_root)
    chunker = manifest["chunking"]["chunkers"][0]
    chunker["record_count"] = int(chunker["record_count"]) + 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotVerificationError,
        match="chunker manifest .* record_count mismatch",
    ):
        SnapshotVerifier(repo_root=repo_root).verify(manifest_path)


def test_verifier_rejects_source_relationship_corruption_with_updated_hash(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    relationships_path = repo_root / "data/thesis/corpus/source_relationships.jsonl"
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"
    manifest = _manifest(repo_root)
    write_jsonl(
        relationships_path,
        [
            {
                "schema_version": SOURCE_RELATIONSHIP_SCHEMA_VERSION,
                "source_relationship_id": "source_relationship:forged",
                "relationship_kind": "exact_duplicate",
                "member_page_ids": ["page:forged-a", "page:forged-b"],
                "confidence": "deterministic",
                "evidence": {"text_hash": "sha256:ccdd35168ab474fa5764a526cfb83621351e23682c5075b2e18d56bddf96aa30"},
                "source_relationship_rule_version": "source_relationship_rules_v1",
                "construction_config_hash": manifest["artifact_config_hashes"][
                    "source_relationships"
                ],
                "corpus_hash": manifest["corpus_hash"],
            },
        ],
    )
    manifest["artifact_hashes"]["source_relationships"] = hash_file(
        relationships_path,
    )
    manifest["artifact_record_counts"]["source_relationships"] = 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotVerificationError,
        match="source relationship records mismatch",
    ):
        SnapshotVerifier(repo_root=repo_root).verify(manifest_path)


def test_restore_refuses_nonempty_target(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    restore_dir = tmp_path / "restore"
    restore_dir.mkdir()
    marker = restore_dir / "keep.txt"
    marker.write_text("do not replace", encoding="utf-8")

    with pytest.raises(SnapshotVerificationError, match="not empty"):
        SnapshotVerifier(repo_root=repo_root).verify(
            repo_root / "data/thesis/corpus/snapshot_manifest.json",
            archive_only=True,
            restore_dir=restore_dir,
        )

    assert marker.read_text(encoding="utf-8") == "do not replace"


def test_summary_artifacts_resume_from_page_only_to_default_targets(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path

    page_builder = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
        targets=["page"],
    )
    page_run = page_builder.run()

    assert page_run.status == "partial"
    assert page_run.generated_counts == {"page": 3, "context_group": 0}
    assert page_run.record_counts == {"page": 3, "context_group": 0}

    all_builder = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )
    assert all_builder.output_dir == page_builder.output_dir

    completed = all_builder.run(resume=True)
    summary_manifest = SummaryManifest.from_dict(
        json.loads(all_builder.summary_manifest_path.read_text(encoding="utf-8")),
    )

    assert completed.status == "complete"
    assert completed.generated_counts == {"page": 0, "context_group": 3}
    assert completed.reused_counts == {"page": 3, "context_group": 0}
    assert completed.record_counts == {"page": 3, "context_group": 3}
    assert summary_manifest.status == "complete"
    assert summary_manifest.target_status["page"]["complete"] is True
    assert summary_manifest.target_status["context_group"]["complete"] is True

    fully_reused = all_builder.run(resume=True)
    assert fully_reused.generated_counts == {"page": 0, "context_group": 0}
    assert fully_reused.reused_counts == {"page": 3, "context_group": 3}

    forced = all_builder.run(force=True, limit=1)
    assert forced.status == "partial"
    assert forced.generated_counts == {"page": 1, "context_group": 1}
    assert forced.record_counts == {"page": 1, "context_group": 1}


def test_summary_artifacts_reuse_unchanged_content_after_snapshot_rebuild(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    first_snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    first_builder = SummaryArtifactBuilder(
        repo_root / first_snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
        targets=["page"],
    )
    first_run = first_builder.run()
    first_output_dir = first_builder.output_dir

    assert first_run.generated_counts == {"page": 3, "context_group": 0}
    root_content = repo_root / "data/az/scraper/topics/root-topic/content.html"
    root_content.write_text(
        "<main><h1>Wurzel</h1><p>Changed root text.</p></main>",
        encoding="utf-8",
    )
    second_snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build(force=True)

    assert first_output_dir.exists()
    second_builder = SummaryArtifactBuilder(
        repo_root / second_snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
        targets=["page"],
    )
    second_run = second_builder.run()

    assert second_builder.output_dir != first_output_dir
    assert second_run.generated_counts == {"page": 1, "context_group": 0}
    assert second_run.reused_counts == {"page": 2, "context_group": 0}

    forced_run = second_builder.run(force=True)

    assert forced_run.generated_counts == {"page": 3, "context_group": 0}
    assert forced_run.reused_counts == {"page": 0, "context_group": 0}


def test_summary_dry_run_limit_and_snapshot_hash_validation(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path
    builder = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )

    dry_run = builder.run(dry_run=True, limit=1)
    assert dry_run.dry_run is True
    assert dry_run.targets == ("page", "context_group")
    assert dry_run.source_counts == {"page": 3, "context_group": 3}
    assert not builder.output_dir.exists()

    limited = builder.run(limit=1)
    assert limited.status == "partial"
    assert limited.generated_counts == {"page": 1, "context_group": 1}

    pages_path = repo_root / _manifest(repo_root)["artifact_paths"]["pages"]
    pages_path.write_text(
        pages_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SummaryJobError, match="pages artifact hash mismatch"):
        builder.run(dry_run=True)


def test_summary_source_set_identity_tracks_structural_artifacts(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest = _manifest(repo_root)
    source_hashes = {
        name: manifest["artifact_hashes"][name]
        for name in ("pages", "sections", "section_context_groups")
    }
    first = _summary_source_set_hash(
        corpus_hash=snapshot.corpus_hash,
        source_artifact_hashes=source_hashes,
    )
    changed = _summary_source_set_hash(
        corpus_hash=snapshot.corpus_hash,
        source_artifact_hashes={
            **source_hashes,
            "section_context_groups": f"sha256:{'0' * 64}",
        },
    )

    assert first != changed


def test_summary_page_selection_preserves_full_manifest_scope(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path
    manifest = _manifest(repo_root)
    pages = read_jsonl(repo_root / manifest["artifact_paths"]["pages"])
    selected_page_id = str(pages[0]["page_id"])
    builder = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )

    selected = builder.run(page_ids=[selected_page_id])

    assert selected.status == "partial"
    assert selected.source_counts == {"page": 3, "context_group": 3}
    assert selected.generated_counts == {"page": 1, "context_group": 1}
    verified = SummaryArtifactVerifier(repo_root=repo_root).verify(
        builder.summary_manifest_path,
    )
    assert verified.status == "partial"
    assert verified.page_summary_count == 1
    assert verified.context_group_summary_count == 1

    completed = builder.run(resume=True)
    assert completed.status == "complete"
    assert completed.generated_counts == {"page": 2, "context_group": 2}


def test_summary_page_selection_rejects_unknown_page(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )

    with pytest.raises(SummaryJobError, match="Unknown summary page_id"):
        builder.run(page_ids=["page:missing"])

    assert not builder.output_dir.exists()


def test_summary_cli_builds_and_resumes_fixture_artifacts(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path
    common_args = [
        "corpus",
        "summarize",
        str(manifest_path),
        "--config",
        "configs/thesis/corpus.yaml",
        "--repo-root",
        str(repo_root),
        "--mode",
        "extractive",
    ]

    limited = CliRunner().invoke(app, [*common_args, "--limit", "1"])
    assert limited.exit_code == 0, limited.output
    assert "Summary set partial" in limited.output
    assert "page=1" in limited.output
    assert "context_group=1" in limited.output

    resumed = CliRunner().invoke(app, common_args)
    assert resumed.exit_code == 0, resumed.output
    assert "Summary set complete" in resumed.output
    assert "page=2" in resumed.output
    assert "context_group=2" in resumed.output


def test_snapshot_and_verify_cli_run_the_fixture_workflow(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "snapshot",
            "--config",
            "configs/thesis/corpus.yaml",
            "--repo-root",
            str(repo_root),
            "--compression",
            "gzip",
        ],
    )

    assert snapshot_result.exit_code == 0, snapshot_result.output
    assert "Snapshot created" in snapshot_result.output
    manifest_path = repo_root / "data/thesis/corpus/snapshot_manifest.json"

    verify_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "verify",
            str(manifest_path),
            "--repo-root",
            str(repo_root),
        ],
    )
    assert verify_result.exit_code == 0, verify_result.output
    assert "Snapshot verified" in verify_result.output


def test_corpus_inspector_searches_and_loads_verified_summaries(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path
    inspector = CorpusInspector(manifest_path, repo_root=repo_root)

    matches = inspector.search(query="English paragraph", language="en")

    assert len(matches) == 1
    selected_page_id = str(matches[0].record["page_id"])
    summary_result = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    ).run(page_ids=[selected_page_id])
    detail = inspector.inspect_page(selected_page_id)
    leaf_detail = inspector.inspect_page(
        selected_page_id, summary_root=(repo_root / summary_result.manifest_path).parent,
    )
    assert leaf_detail.summaries == detail.summaries
    assert leaf_detail.context_group_summaries == detail.context_group_summaries

    assert detail.page.record["language"] == "en"
    assert len(detail.sections) == 1
    assert len(detail.context_groups) == 1
    assert detail.summaries
    assert detail.context_group_summaries
    assert detail.context_group_summaries[0].context_group_id == (
        detail.context_groups[0]["group_id"]
    )
    assert detail.summary_warnings == ()
    output = StringIO()
    render_page_detail(Console(file=output, width=160), detail, max_text_chars=100)
    group_output = output.getvalue().split("SectionContextGroup G1 of 1", 1)[1]
    assert detail.context_group_summaries[0].text in group_output
    assert "Section 1 of 1 · Focus" in group_output


def test_corpus_inspector_renders_sections_inside_overlapping_groups(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    source = repo_root / "data/az/scraper/topics/child-topic/content.html"
    source.write_text(
        "<main><h1>Guide [red]</h1><h2>First</h2><p>"
        + "First section content. " * 20
        + "</p><h2>Second</h2><p>"
        + "Second section content. " * 20
        + "</p><h2>Contact</h2><p>Telephone</p>"
        + "<h2>Address</h2><p>Office location</p><h2>Empty</h2></main>",
        encoding="utf-8",
    )
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml", repo_root=repo_root, compression="gzip",
    ).build()
    inspector = CorpusInspector(
        repo_root / snapshot.manifest_path, repo_root=repo_root,
    )
    page = inspector.search(query="child-topic", language="de")[0]
    page_id = str(page.record["page_id"])
    detail = inspector.inspect_page(page_id, include_summaries=False)
    output = StringIO()
    render_page_detail(Console(file=output, width=140), detail, max_text_chars=100)
    rendered = output.getvalue()

    assert len(detail.context_groups) == 4
    assert len(detail.sections) == 6
    assert "groups may overlap" in rendered
    assert rendered.count("Section 1 of 6 · Context · shared") == 3
    assert rendered.count("Section 2 of 6 · Focus") == 1
    assert rendered.count("Section 3 of 6 · Focus") == 1
    assert "orphan_structural" in rendered
    assert "Guide [red]" in rendered  # Source text is not interpreted as markup.
    first = rendered.index("SectionContextGroup G1 of 4")
    second = rendered.index("SectionContextGroup G2 of 4")
    third = rendered.index("SectionContextGroup G3 of 4")
    fourth = rendered.index("SectionContextGroup G4 of 4")
    assert first < rendered.index("Section 2 of 6 · Focus") < second
    assert second < rendered.index("Section 3 of 6 · Focus") < third
    assert third < rendered.index("Section 4 of 6 · Focus") < fourth
    assert third < rendered.index("Section 5 of 6 · Focus") < fourth
    assert fourth < rendered.index("Section 6 of 6 · Context")


def test_corpus_inspector_verifies_context_group_artifact(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml", repo_root=repo_root, compression="gzip",
    ).build()
    inspector = CorpusInspector(
        repo_root / snapshot.manifest_path, repo_root=repo_root,
    )
    page_id = str(inspector.search(query="English paragraph")[0].record["page_id"])
    groups = repo_root / "data/thesis/corpus/section_context_groups.jsonl"
    groups.write_text(groups.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(
        CorpusInspectionError, match="section_context_groups artifact hash mismatch",
    ):
        inspector.inspect_page(page_id, include_summaries=False)


def test_corpus_inspect_cli_supports_numbered_selection(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "inspect",
            str(repo_root / snapshot.manifest_path),
            "--repo-root",
            str(repo_root),
            "--limit",
            "2",
            "--interactive",
            "--no-summaries",
        ],
        input="1\n",
    )

    assert result.exit_code == 0, result.output
    assert "3 matching pages · showing 2" in result.output
    assert "Page number to inspect" in result.output
    assert "Index?" in result.output
    assert "Page" in result.output
    assert "Size" in result.output
    assert "Source (content):" in result.output
    assert "Source (page):" in result.output
    assert "file://" in result.output
    assert "Section 1 of 1" in result.output
    assert "SectionContextGroup G1 of 1" in result.output
    assert "Focus" in result.output


def test_corpus_inspector_source_paths_render_as_terminal_hyperlinks(tmp_path):
    source_path = tmp_path / "source with spaces.html"
    output = StringIO()
    Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
    ).print(_file_link_text(source_path))

    rendered = output.getvalue()
    assert source_path.resolve().as_uri() in rendered
    assert "\x1b]8;" in rendered


def _write_test_inspector_repo(tmp_path: Path) -> Path:
    repo_root = _write_snapshot_fixture_repo(tmp_path)
    _write_csv(
        repo_root / "data/az/mapping.csv",
        [
            {
                "depth": "0",
                "intent": "child-topic",
                "extracted_title_deutsch": "Kind",
                "extracted_title_english": "Child",
                "generated_title_deutsch": "",
                "generated_title_english": "",
                "manual_title_deutsch": "Kind manuell",
                "manual_title_english": "Child manual",
                "link": "https://www.tu.ber.kind",
                "mapping": "https://www.tu.ber.en/child",
                "parent": "",
                "etag": "child-etag",
            },
            {
                "depth": "0",
                "intent": "root-topic",
                "extracted_title_deutsch": "Wurzel",
                "extracted_title_english": "Root",
                "generated_title_deutsch": "",
                "generated_title_english": "",
                "manual_title_deutsch": "",
                "manual_title_english": "",
                "link": "https://www.tu.ber.root",
                "mapping": "https://www.tu.ber.en/root",
                "parent": "",
                "etag": "root-etag",
            },
            {
                "depth": "0",
                "intent": "duplicate-topic",
                "extracted_title_deutsch": "Dublette",
                "extracted_title_english": "Duplicate",
                "generated_title_deutsch": "",
                "generated_title_english": "",
                "manual_title_deutsch": "",
                "manual_title_english": "",
                "link": "https://www.tu.ber.duplicate",
                "mapping": "https://www.tu.ber.en/duplicate",
                "parent": "",
                "etag": "duplicate-etag",
            },
            {
                "depth": "0",
                "intent": "empty-topic",
                "extracted_title_deutsch": "Leer",
                "extracted_title_english": "Empty",
                "generated_title_deutsch": "",
                "generated_title_english": "",
                "manual_title_deutsch": "",
                "manual_title_english": "",
                "link": "https://www.tu.ber.empty",
                "mapping": "https://www.tu.ber.en/empty",
                "parent": "",
                "etag": "empty-etag",
            },
        ],
        MAPPING_COLUMNS,
    )
    dup_dir = repo_root / "data/az/scraper/topics/duplicate-topic"
    dup_dir.mkdir(parents=True)
    (dup_dir / "content.html").write_text(
        "<main><h1>Duplicate</h1><p>Deutscher Absatz.</p></main>",
        encoding="utf-8",
    )
    (dup_dir / "content_english.html").write_text(
        "<main><h1>Child manual</h1><p>English paragraph.</p></main>",
        encoding="utf-8",
    )
    (dup_dir / "page.html").write_text(
        "<html lang='de'><head><title>Duplicate Page</title></head></html>",
        encoding="utf-8",
    )
    (dup_dir / "page_english.html").write_text(
        "<html lang='en'><head><title>Duplicate English</title></head></html>",
        encoding="utf-8",
    )
    _write_csv(dup_dir / "versions.csv", [{"version": "1"}], ["version"])

    empty_dir = repo_root / "data/az/scraper/topics/empty-topic"
    empty_dir.mkdir(parents=True)
    (empty_dir / "content.html").write_text(
        "<main></main>",
        encoding="utf-8",
    )
    (empty_dir / "content_english.html").write_text(
        "<main></main>",
        encoding="utf-8",
    )
    (empty_dir / "page.html").write_text(
        "<html lang='de'><head><title>Empty Page</title></head></html>",
        encoding="utf-8",
    )
    (empty_dir / "page_english.html").write_text(
        "<html lang='en'><head><title>Empty English</title></head></html>",
        encoding="utf-8",
    )
    _write_csv(empty_dir / "versions.csv", [{"version": "1"}], ["version"])

    return repo_root


def test_corpus_inspector_search_ignored_and_duplicates(tmp_path):
    repo_root = _write_test_inspector_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path
    inspector = CorpusInspector(manifest_path, repo_root=repo_root)

    all_matches = inspector.search()
    assert len(all_matches) > 0

    ignored_matches = inspector.search(ignored_only=True)
    assert len(ignored_matches) > 0
    assert all(
        not bool(match.quality["eligible_standard_index"])
        for match in ignored_matches
    )
    ignored_intents = {m.record["intent"] for m in ignored_matches}
    assert "empty-topic" in ignored_intents

    duplicate_matches = inspector.search(duplicates_only=True)
    assert len(duplicate_matches) > 0
    assert all(bool(match.quality["exact_duplicate"]) for match in duplicate_matches)
    duplicate_intents = {m.record["intent"] for m in duplicate_matches}
    assert "child-topic" in duplicate_intents or "duplicate-topic" in duplicate_intents

    clusters = inspector.get_duplicate_clusters()
    assert len(clusters) > 0
    assert clusters[0]["size"] == 2
    assert clusters[0]["members"][0]["canonical"] is True
    assert clusters[0]["canonical_page_id"] == clusters[0]["members"][0]["page_id"]

    relationships = inspector.get_source_relationships(
        relationship_kind="exact_duplicate",
    )
    assert len(relationships) > 0
    assert relationships[0]["relationship_kind"] == "exact_duplicate"
    assert relationships[0]["size"] == 2
    assert relationships[0]["members"][0]["page_id"].startswith("page:")
    assert "text_hash" in relationships[0]["evidence"]

    page_relationships = inspector.get_source_relationships(
        page_id=relationships[0]["members"][0]["page_id"],
    )
    assert relationships[0]["source_relationship_id"] in {
        relationship["source_relationship_id"]
        for relationship in page_relationships
    }


def test_corpus_inspect_cli_ignored_and_duplicates(tmp_path):
    repo_root = _write_test_inspector_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()

    result_ignored = CliRunner().invoke(
        app,
        [
            "corpus",
            "inspect",
            str(repo_root / snapshot.manifest_path),
            "--repo-root",
            str(repo_root),
            "--ignored-only",
            "--no-summaries",
        ],
    )
    assert result_ignored.exit_code == 0, result_ignored.output
    assert "Empty" in result_ignored.output
    assert "No filters supplied" not in result_ignored.output

    result_dup = CliRunner().invoke(
        app,
        [
            "corpus",
            "inspect",
            str(repo_root / snapshot.manifest_path),
            "--repo-root",
            str(repo_root),
            "--duplicates-only",
            "--no-summaries",
        ],
    )
    assert result_dup.exit_code == 0, result_dup.output
    assert "Child" in result_dup.output or "Duplicate" in result_dup.output
    assert "No filters supplied" not in result_dup.output


def test_corpus_inspect_cli_duplicate_clusters(tmp_path):
    repo_root = _write_test_inspector_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "duplicate-clusters",
            str(repo_root / snapshot.manifest_path),
            "--repo-root",
            str(repo_root),
            "--members-limit",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "duplicate cluster" in result.output
    assert "Cluster #1" in result.output
    assert "Canonical?" in result.output
    assert "Title / Intent" in result.output
    assert "child-topic" in result.output
    assert "duplicate-topic" in result.output


def test_corpus_inspect_cli_relationships(tmp_path):
    repo_root = _write_test_inspector_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "relationships",
            str(repo_root / snapshot.manifest_path),
            "--repo-root",
            str(repo_root),
            "--kind",
            "exact_duplicate",
            "--members-limit",
            "0",
            "--no-evidence",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "source relationship" in result.output
    assert "Relationship #1" in result.output
    assert "exact_duplicate" in result.output
    assert "Index?" in result.output
    assert "QGen?" in result.output
    assert "child-topic" in result.output
    assert "duplicate-topic" in result.output


def test_summary_failure_checkpoints_completed_records_for_resume(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    config_path = repo_root / "configs/thesis/corpus.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["summaries"]["local_llm"] = {
        "model_id": "fixture-model",
        "model_revision": "fixture-revision",
        "temperature": 0.0,
        "max_tokens": 64,
        "max_input_chars": 1000,
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    manifest_path = repo_root / snapshot.manifest_path
    call_count = 0

    def interrupted_provider(request):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise SummaryBuildError("fixture interruption")
        return f"Summary for {request.page_id}."

    interrupted = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="local-llm",
        targets=["page"],
        local_summary_provider=interrupted_provider,
    )
    with pytest.raises(SummaryBuildError, match="fixture interruption"):
        interrupted.run()

    partial_manifest = SummaryManifest.from_dict(
        json.loads(interrupted.summary_manifest_path.read_text(encoding="utf-8")),
    )
    assert partial_manifest.status == "partial"
    assert partial_manifest.target_status["page"]["record_count"] == 1

    resumed = SummaryArtifactBuilder(
        manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="local-llm",
        targets=["page"],
        local_summary_provider=lambda request: f"Summary for {request.page_id}.",
    ).run(resume=True)

    assert resumed.generated_counts == {"page": 2, "context_group": 0}
    assert resumed.reused_counts == {"page": 1, "context_group": 0}


def test_local_llm_summary_job_skips_empty_sources(tmp_path, monkeypatch):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    config_path = repo_root / "configs/thesis/corpus.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["summaries"]["local_llm"] = {
        "model_id": "fixture-model",
        "model_revision": "fixture-revision",
        "temperature": 0.0,
        "max_tokens": 64,
        "max_input_chars": 1000,
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="local-llm",
        local_summary_provider=lambda request: f"Summary for {request.page_id}.",
    )
    pages, sections, context_groups, source_hashes = builder._load_snapshot_sources()
    empty_page = {
        **pages[0],
        "page_id": "page:empty",
        "text": "",
        "text_hash": hash_text(""),
    }
    monkeypatch.setattr(
        builder,
        "_load_snapshot_sources",
        lambda: ([empty_page, *pages], sections, context_groups, source_hashes),
    )

    result = builder.run()
    summary_manifest = SummaryManifest.from_dict(
        json.loads(builder.summary_manifest_path.read_text(encoding="utf-8")),
    )

    assert result.status == "complete"
    assert result.source_counts == {"page": 4, "context_group": 3}
    assert result.skipped_counts == {"page": 1, "context_group": 0}
    assert result.record_counts == {"page": 3, "context_group": 3}
    assert summary_manifest.target_status["page"] == {
        "source_count": 4,
        "record_count": 3,
        "skipped_count": 1,
        "complete": True,
    }


def test_extractive_summary_job_skips_empty_sources(tmp_path, monkeypatch):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
        targets=["page"],
    )
    pages, sections, context_groups, source_hashes = builder._load_snapshot_sources()
    empty_page = {
        **pages[0],
        "page_id": "page:empty",
        "text": "",
        "text_hash": hash_text(""),
    }
    monkeypatch.setattr(
        builder,
        "_load_snapshot_sources",
        lambda: ([empty_page, *pages], sections, context_groups, source_hashes),
    )

    result = builder.run()

    assert result.skipped_counts == {"page": 1, "context_group": 0}
    assert result.record_counts == {"page": 3, "context_group": 0}


def test_summary_verifier_accepts_partial_and_complete_sets(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )
    verifier = SummaryArtifactVerifier(repo_root=repo_root)

    builder.run(limit=1)
    partial = verifier.verify(builder.summary_manifest_path)
    assert partial.status == "partial"
    assert partial.page_summary_count == 1
    assert partial.context_group_summary_count == 1

    builder.run()
    complete = verifier.verify(builder.summary_manifest_path)
    assert complete.status == "complete"
    assert complete.page_summary_count == 3
    assert complete.context_group_summary_count == 3


def test_summary_verifier_rejects_modified_artifact(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )
    builder.run(limit=1)
    builder.page_summaries_path.write_text(
        builder.page_summaries_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SummaryVerificationError, match="artifact hash mismatch"):
        SummaryArtifactVerifier(repo_root=repo_root).verify(
            builder.summary_manifest_path,
        )


def test_summary_verifier_rejects_incompatible_source_hash(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )
    builder.run(limit=1)
    manifest = json.loads(
        builder.summary_manifest_path.read_text(encoding="utf-8"),
    )
    manifest["source_artifact_hashes"]["pages"] = f"sha256:{'0' * 64}"
    builder.summary_manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(SummaryVerificationError, match="source hash"):
        SummaryArtifactVerifier(repo_root=repo_root).verify(
            builder.summary_manifest_path,
        )


def test_summary_verifier_rejects_incorrect_record_count(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )
    builder.run(limit=1)
    manifest = json.loads(
        builder.summary_manifest_path.read_text(encoding="utf-8"),
    )
    manifest["target_status"]["page"]["record_count"] = 0
    builder.summary_manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(SummaryVerificationError, match="summary count mismatch"):
        SummaryArtifactVerifier(repo_root=repo_root).verify(
            builder.summary_manifest_path,
        )


def test_summary_verify_cli_reports_success_and_corruption(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    snapshot = SnapshotBuilder(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    builder = SummaryArtifactBuilder(
        repo_root / snapshot.manifest_path,
        config="configs/thesis/corpus.yaml",
        repo_root=repo_root,
        mode="extractive",
    )
    builder.run(limit=1)
    args = [
        "corpus",
        "summary-verify",
        str(builder.summary_manifest_path),
        "--repo-root",
        str(repo_root),
    ]

    verified = CliRunner().invoke(app, args)
    assert verified.exit_code == 0, verified.output
    assert "Summary set verified" in verified.output
    assert "page=1" in verified.output
    assert "context_group=1" in verified.output

    builder.context_group_summaries_path.write_text(
        builder.context_group_summaries_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    corrupted = CliRunner().invoke(app, args)
    assert corrupted.exit_code == 1
    assert "Summary verification failed" in corrupted.output
