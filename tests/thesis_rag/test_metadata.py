from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from thesis_rag.corpus.metadata import (
    CorpusConfig,
    MappingSchemaError,
    MetadataFileError,
    MetadataStore,
    normalize_language,
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


def _base_mapping_rows() -> list[dict[str, str]]:
    return [
        {
            "depth": "1",
            "intent": "child-topic",
            "extracted_title_deutsch": "Kind extrahiert",
            "extracted_title_english": "Child extracted",
            "generated_title_deutsch": "Kind generiert",
            "generated_title_english": "Child generated",
            "manual_title_deutsch": "Kind manuell",
            "manual_title_english": "Child manual",
            "link": "https://www.tu.berlin/kind",
            "mapping": "https://www.tu.berlin/en/child",
            "parent": "root-topic",
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
        {
            "depth": "0",
            "intent": "english-fallback",
            "extracted_title_deutsch": "Fallback DE",
            "extracted_title_english": "Fallback EN",
            "generated_title_deutsch": "",
            "generated_title_english": "",
            "manual_title_deutsch": "",
            "manual_title_english": "",
            "link": "https://www.tu.berlin/fallback",
            "mapping": "",
            "parent": "",
            "etag": "fallback-etag",
        },
    ]


def _write_fixture_repo(
    tmp_path: Path,
    *,
    mapping_rows: list[dict[str, str]] | None = None,
    mapping_columns: list[str] | None = None,
    include_manual_utterances: bool = True,
    missing_optional_file: bool = False,
    missing_required_file: bool = False,
) -> Path:
    repo_root = tmp_path
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n",
        encoding="utf-8",
    )
    (repo_root / "thesis_rag").mkdir()
    (repo_root / "data/az/scraper/topics").mkdir(parents=True)
    (repo_root / "configs/thesis").mkdir(parents=True)

    mapping_rows = mapping_rows or _base_mapping_rows()
    mapping_columns = mapping_columns or MAPPING_COLUMNS
    _write_csv(repo_root / "data/az/mapping.csv", mapping_rows, mapping_columns)

    if include_manual_utterances:
        _write_csv(
            repo_root / "data/az/manual_utterances.csv",
            [
                {
                    "utterance": "Question one",
                    "link": "https://www.tu.berlin/kind",
                    "intent": "child-topic",
                },
                {
                    "utterance": "Question two",
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
    _write_csv(
        repo_root / "data/az/overwrite_topics.csv",
        [{"intent": "child-topic", "topic": "Override"}],
        ["intent", "topic"],
    )
    (repo_root / "data/az/translations.json").write_text(
        '{"child-topic": "Child"}\n',
        encoding="utf-8",
    )
    if not missing_required_file:
        _write_csv(
            repo_root / "data/az/required_metadata.csv",
            [
                {
                    "etag": "child-etag",
                    "intent": "child-topic",
                    "link": "https://www.tu.berlin/kind",
                    "mapping": "https://www.tu.berlin/en/child",
                },
            ],
            ["etag", "intent", "link", "mapping"],
        )

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
                "path": "data/az/overwrite_topics.csv",
                "role": "az_overwrite_topics",
                "required": True,
            },
            {
                "path": "data/az/translations.json",
                "role": "az_translations",
                "required": True,
            },
            {
                "path": "data/az/required_metadata.csv",
                "role": "required_metadata",
                "required": True,
            },
            {
                "path": "data/az/missing_optional.csv",
                "role": "missing_optional",
                "required": False,
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
            "short_text_threshold": 200,
            "error_template_phrases": [],
            "count_change_review_fraction": 0.1,
            "large_duplicate_cluster_threshold": 10,
        },
    }
    if missing_optional_file:
        config["additional_files"].append(
            {
                "path": "data/az/also_missing_optional.csv",
                "role": "also_missing_optional",
                "required": False,
            },
        )
    (repo_root / "configs/thesis/corpus.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return repo_root


def test_metadata_store_loads_mapping_manual_utterances_and_additional_files(
    tmp_path,
) -> None:
    repo_root = _write_fixture_repo(tmp_path)

    store = MetadataStore.from_config_file(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
    )

    child = store.get("child-topic")
    assert child.depth == 1
    assert child.parent == "root-topic"
    assert child.manual_utterance_count == 2
    assert store.manual_utterances_by_intent["child-topic"][0]["utterance"] == (
        "Question one"
    )
    assert set(store.additional_data_by_role) == {
        "az_blacklist",
        "az_overwrite_topics",
        "az_translations",
        "required_metadata",
    }
    assert store.intents == ["child-topic", "english-fallback", "root-topic"]
    assert [topic.intent for topic in store.topics_sorted()] == store.intents


def test_metadata_resolves_urls_titles_and_breadcrumbs(tmp_path) -> None:
    repo_root = _write_fixture_repo(tmp_path)
    store = MetadataStore.from_config_file(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
    )

    child = store.get("child-topic")
    assert child.url("de") == "https://www.tu.berlin/kind"
    assert child.url("en") == "https://www.tu.berlin/en/child"
    assert store.get("english-fallback").url("en") == "https://www.tu.berlin/fallback"
    assert child.title("de") == "Kind manuell"
    assert child.title("en") == "Child manual"

    breadcrumbs = store.breadcrumbs_for("child-topic", language="english")
    assert [entry["intent"] for entry in breadcrumbs] == ["root-topic", "child-topic"]
    assert [entry["title"] for entry in breadcrumbs] == ["Root", "Child manual"]
    assert store.page_metadata_for("child-topic", language="de") == {
        "intent": "child-topic",
        "language": "de",
        "depth": 1,
        "parent": "root-topic",
        "etag": "child-etag",
        "title": "Kind manuell",
        "url": "https://www.tu.berlin/kind",
        "breadcrumbs": [
            {
                "intent": "root-topic",
                "title": "Wurzel",
                "url": "https://www.tu.berlin/root",
                "depth": 0,
            },
            {
                "intent": "child-topic",
                "title": "Kind manuell",
                "url": "https://www.tu.berlin/kind",
                "depth": 1,
            },
        ],
        "manual_utterance_count": 2,
    }


def test_title_resolution_fallback_order_is_deterministic(tmp_path) -> None:
    rows = [
        {
            **{column: "" for column in MAPPING_COLUMNS},
            "depth": "0",
            "intent": "manual",
            "manual_title_deutsch": "Manual",
        },
        {
            **{column: "" for column in MAPPING_COLUMNS},
            "depth": "0",
            "intent": "generated",
            "generated_title_deutsch": "Generated",
        },
        {
            **{column: "" for column in MAPPING_COLUMNS},
            "depth": "0",
            "intent": "extracted",
            "extracted_title_deutsch": "Extracted",
        },
        {
            **{column: "" for column in MAPPING_COLUMNS},
            "depth": "0",
            "intent": "page-title",
        },
        {
            **{column: "" for column in MAPPING_COLUMNS},
            "depth": "0",
            "intent": "plain-slug",
        },
    ]
    repo_root = _write_fixture_repo(tmp_path, mapping_rows=rows)
    store = MetadataStore.from_config_file(
        "configs/thesis/corpus.yaml",
        repo_root=repo_root,
    )

    assert store.get("manual").title("de", page_title="Page") == "Manual"
    assert store.get("generated").title("de", page_title="Page") == "Page"
    assert store.get("extracted").title("de", page_title="Page") == "Page"
    assert store.get("extracted").title("de") == "Extracted"
    assert store.get("page-title").title("de", page_title="Page") == "Page"
    assert store.get("plain-slug").title("de") == "plain slug"


def test_mapping_missing_required_columns_fails_clearly(tmp_path) -> None:
    columns = [column for column in MAPPING_COLUMNS if column != "etag"]
    row = {column: "" for column in columns}
    row.update({"depth": "0", "intent": "missing-etag"})
    repo_root = _write_fixture_repo(
        tmp_path,
        mapping_rows=[row],
        mapping_columns=columns,
    )

    with pytest.raises(MappingSchemaError, match="etag"):
        MetadataStore.from_config_file(
            "configs/thesis/corpus.yaml",
            repo_root=repo_root,
        )


def test_configured_source_files_are_repo_relative_and_include_optionals(
    tmp_path,
) -> None:
    repo_root = _write_fixture_repo(
        tmp_path,
        include_manual_utterances=False,
        missing_optional_file=True,
    )
    config = CorpusConfig.from_file("configs/thesis/corpus.yaml", repo_root=repo_root)

    inputs = config.metadata_source_files()
    roles = [source_file.role for source_file in inputs]

    assert roles == [
        "mapping",
        "manual_utterances",
        "az_blacklist",
        "az_overwrite_topics",
        "az_translations",
        "required_metadata",
        "missing_optional",
        "also_missing_optional",
    ]
    assert all(not source_file.relative_path.startswith("/") for source_file in inputs)
    assert inputs[0].relative_path == "data/az/mapping.csv"
    assert inputs[1].role == "manual_utterances"
    assert inputs[1].required is False
    assert inputs[1].exists is False
    assert inputs[-1].role == "also_missing_optional"
    assert inputs[-1].required is False
    assert inputs[-1].exists is False


def test_missing_required_configured_file_fails_but_optional_missing_file_loads(
    tmp_path,
) -> None:
    optional_repo = _write_fixture_repo(tmp_path / "optional", missing_optional_file=True)
    store = MetadataStore.from_config_file(
        "configs/thesis/corpus.yaml",
        repo_root=optional_repo,
    )
    assert "missing_optional" not in store.additional_data_by_role
    assert "also_missing_optional" not in store.additional_data_by_role

    required_repo = _write_fixture_repo(
        tmp_path / "required",
        missing_required_file=True,
    )
    with pytest.raises(MetadataFileError, match="required_metadata"):
        MetadataStore.from_config_file(
            "configs/thesis/corpus.yaml",
            repo_root=required_repo,
        )


def test_languages_are_normalized() -> None:
    assert normalize_language("deutsch") == "de"
    assert normalize_language("English") == "en"
    with pytest.raises(ValueError, match="Unsupported corpus language"):
        normalize_language("fr")
