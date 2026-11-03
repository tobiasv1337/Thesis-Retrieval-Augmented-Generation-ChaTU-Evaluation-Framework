"""Metadata loading helpers for the thesis AZ corpus snapshot."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
)
from thesis_rag.corpus.chunking import ChunkingConfig
from thesis_rag.schema import (
    CORPUS_CONFIG_SCHEMA_VERSION,
)


LANGUAGE_ALIASES = {
    "de": "de",
    "deutsch": "de",
    "german": "de",
    "en": "en",
    "english": "en",
    "englisch": "en",
}

REQUIRED_MAPPING_COLUMNS = frozenset(
    {
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
    },
)

MANUAL_TITLE_COLUMN_BY_LANGUAGE = {
    "de": "manual_title_deutsch",
    "en": "manual_title_english",
}

EXTRACTED_TITLE_COLUMN_BY_LANGUAGE = {
    "de": "extracted_title_deutsch",
    "en": "extracted_title_english",
}


class MetadataError(ValueError):
    """Base class for metadata loading failures."""


class CorpusConfigError(MetadataError):
    """Raised when the corpus config cannot be loaded or validated."""


class MetadataFileError(MetadataError):
    """Raised when a configured metadata file cannot be loaded."""


class MappingSchemaError(MetadataError):
    """Raised when mapping.csv has an unsupported shape."""


class AdditionalFileConfig(ArtifactModel):
    path: NonEmptyStr
    role: NonEmptyStr
    required: bool


class SnapshotConfig(ArtifactModel):
    include_versions: bool
    include_page_html: bool
    archive_path: NonEmptyStr
    source_manifest_path: NonEmptyStr
    manifest_path: NonEmptyStr


class ParserConfig(ArtifactModel):
    content_files: dict[str, NonEmptyStr]
    page_files: dict[str, NonEmptyStr]
    fallback_section_heading: NonEmptyStr = "__page__"


class SectionContextGroupConfig(ArtifactModel):
    """Grouping thresholds for short, structurally adjacent sections."""

    output_path: NonEmptyStr
    micro_max_effective_chars: PositiveInt = 200
    target_group_effective_chars: PositiveInt = 400
    max_group_effective_chars: PositiveInt = 1600
    max_focus_sections: PositiveInt = 4
    lead_in_max_chars: PositiveInt = 120

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "SectionContextGroupConfig":
        if self.target_group_effective_chars < self.micro_max_effective_chars:
            raise ValueError(
                "target_group_effective_chars must be >= micro_max_effective_chars",
            )
        if self.max_group_effective_chars < self.target_group_effective_chars:
            raise ValueError(
                "max_group_effective_chars must be >= target_group_effective_chars",
            )
        if self.lead_in_max_chars > self.micro_max_effective_chars:
            raise ValueError("lead_in_max_chars must not exceed the micro threshold")
        return self


class ChunkingConfigContract(ArtifactModel):
    output_dir: NonEmptyStr
    chunkers: Annotated[list[ChunkingConfig], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_chunkers(self) -> "ChunkingConfigContract":
        ids = [chunker.chunker_id for chunker in self.chunkers]
        if len(ids) != len(set(ids)):
            raise ValueError("chunker_id values must be unique")
        return self


class TableConfig(ArtifactModel):
    extract: bool
    output_path: NonEmptyStr


class MediaConfig(ArtifactModel):
    extract_images: bool
    extract_videos: bool
    images_output_path: NonEmptyStr
    media_descriptions_output_path: NonEmptyStr | None = None


class SummaryPromptConfig(ArtifactModel):
    prompt_id: NonEmptyStr
    path: NonEmptyStr


class LocalLLMConfig(ArtifactModel):
    backend: Literal["openai", "ollama"] = "openai"
    context_window: PositiveInt = 32768
    num_batch: PositiveInt | None = None
    required_for_thesis_results: bool = False
    endpoint: str | None = None
    model_id: str | None = None
    local_model_path: str | None = None
    model_revision: str | None = None
    temperature: float = 0.0
    timeout_seconds: float | None = None
    max_tokens: PositiveInt = 4_000
    max_input_chars: PositiveInt = 48_000
    reasoning_effort: Literal["none", "low", "medium", "high"] = "low"
    generated_at_utc: str | None = None
    api_key: str | None = None


class SummariesConfig(ArtifactModel):
    eligible_only: bool = False
    mode: Literal["none", "extractive", "local-llm"]
    targets: Annotated[
        list[Literal["page", "context_group"]],
        Field(min_length=1),
    ]
    source_passthrough_max_chars: PositiveInt = 200
    page_passthrough_max_chars: NonNegativeInt = 0
    allowed_modes: Annotated[
        list[Literal["none", "extractive", "local-llm"]],
        Field(min_length=1),
    ]
    prompts: dict[Literal["page", "context_group"], SummaryPromptConfig]
    output_dir: NonEmptyStr
    local_llm: LocalLLMConfig | None = None

    @model_validator(mode="after")
    def _validate_modes_and_prompts(self) -> "SummariesConfig":
        if self.mode not in self.allowed_modes:
            raise ValueError("mode must be listed in allowed_modes")
        if set(self.prompts) != {"page", "context_group"}:
            raise ValueError("prompts must contain page and context_group")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("targets must be unique")
        return self


class OutputPathConfig(ArtifactModel):
    output_path: NonEmptyStr


class QualityConfig(ArtifactModel):
    output_path: NonEmptyStr
    report_path: NonEmptyStr
    short_text_threshold: NonNegativeInt
    error_template_phrases: list[str]
    count_change_review_fraction: Annotated[float, Field(ge=0.0)] = 0.10
    large_duplicate_cluster_threshold: PositiveInt = 10


class KeywordConfig(ArtifactModel):
    limit: NonNegativeInt
    ngram_range: tuple[PositiveInt, PositiveInt]
    max_features: NonNegativeInt

    @model_validator(mode="before")
    @classmethod
    def _accept_yaml_range(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("ngram_range"), list):
            return {**data, "ngram_range": tuple(data["ngram_range"])}
        return data

    @model_validator(mode="after")
    def _ordered_range(self) -> "KeywordConfig":
        if self.ngram_range[1] < self.ngram_range[0]:
            raise ValueError("ngram_range must be ascending")
        return self


class TitlesConfig(ArtifactModel):
    resolution_order: list[NonEmptyStr]
    generated_titles_enabled: bool


class CorpusConfigContract(ArtifactModel):
    """Strict persisted YAML contract before paths are resolved."""

    schema_version: Literal[CORPUS_CONFIG_SCHEMA_VERSION] = (
        CORPUS_CONFIG_SCHEMA_VERSION
    )
    corpus_id: NonEmptyStr
    source_root: NonEmptyStr
    mapping_path: NonEmptyStr
    manual_utterances_path: NonEmptyStr | None = None
    output_dir: NonEmptyStr
    languages: Annotated[list[NonEmptyStr], Field(min_length=1)]
    additional_files: list[AdditionalFileConfig] = Field(default_factory=list)
    snapshot: SnapshotConfig
    parser: ParserConfig
    section_context_groups: SectionContextGroupConfig
    titles: TitlesConfig
    keywords: KeywordConfig
    chunking: ChunkingConfigContract
    tables: TableConfig
    media: MediaConfig
    summaries: SummariesConfig
    page_table: OutputPathConfig
    section_table: OutputPathConfig
    quality: QualityConfig

    @model_validator(mode="after")
    def _validate_languages(self) -> "CorpusConfigContract":
        normalized = [normalize_language(language) for language in self.languages]
        if len(normalized) != len(set(normalized)):
            raise ValueError("languages must be unique")
        for field_name in ("content_files", "page_files"):
            configured = getattr(self.parser, field_name)
            missing = sorted(set(normalized) - set(configured))
            if missing:
                raise ValueError(f"parser.{field_name} is missing languages: {missing}")
        return self


def normalize_language(language: str) -> str:
    """Normalize user-facing language names to the corpus language IDs."""

    normalized = LANGUAGE_ALIASES.get((language or "").strip().lower())
    if normalized is None:
        raise CorpusConfigError(f"Unsupported corpus language: {language!r}")
    return normalized


def find_repo_root(start: str | Path) -> Path:
    """Find the implementation repository root from a file or directory path."""

    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "thesis_rag"
        ).is_dir():
            return candidate
    raise CorpusConfigError(f"Could not find repository root from {start}")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_string(value: Any) -> str | None:
    cleaned = _clean(value)
    return cleaned or None


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    except OSError as exc:
        raise MetadataFileError(f"Could not read CSV metadata file {path}") from exc
    except csv.Error as exc:
        raise MetadataFileError(f"Invalid CSV metadata file {path}") from exc


def _load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except OSError as exc:
            raise MetadataFileError(f"Could not read JSON metadata file {path}") from exc
        except json.JSONDecodeError as exc:
            raise MetadataFileError(f"Invalid JSON metadata file {path}") from exc
    if suffix == ".csv":
        return _read_csv_dicts(path)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MetadataFileError(f"Could not read metadata file {path}") from exc


@dataclass(frozen=True)
class ConfiguredSourceFile:
    """A configured metadata input that should be part of the source manifest."""

    path: Path
    relative_path: str
    role: str
    required: bool
    exists: bool
    language: str | None = None
    intent: str | None = None

    def to_manifest_seed(self) -> JsonDict:
        return {
            "source_path": self.relative_path,
            "role": self.role,
            "required": self.required,
            "exists": self.exists,
            "language": self.language,
            "intent": self.intent,
        }


@dataclass(frozen=True)
class CorpusConfig:
    """Resolved and validated corpus config with repo-relative path handling."""

    config_path: Path
    repo_root: Path
    raw_config: JsonDict
    corpus_id: str
    source_root: Path
    mapping_path: Path
    output_dir: Path
    languages: tuple[str, ...]
    manual_utterances_path: Path | None
    additional_files: tuple[ConfiguredSourceFile, ...]
    snapshot: JsonDict
    parser: JsonDict
    section_context_groups: JsonDict
    chunking: JsonDict
    tables: JsonDict
    media: JsonDict
    summaries: JsonDict
    page_table: JsonDict
    section_table: JsonDict
    quality: JsonDict
    keywords: JsonDict

    @classmethod
    def from_file(
        cls,
        config_path: str | Path = "configs/thesis/corpus.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> "CorpusConfig":
        resolved_config_path = Path(config_path)
        if repo_root is not None:
            resolved_repo_root = Path(repo_root).resolve()
            if not resolved_config_path.is_absolute():
                resolved_config_path = resolved_repo_root / resolved_config_path
        else:
            resolved_config_path = resolved_config_path.resolve()
            resolved_repo_root = find_repo_root(resolved_config_path)

        try:
            with resolved_config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except OSError as exc:
            raise CorpusConfigError(
                f"Could not read corpus config {resolved_config_path}",
            ) from exc
        except yaml.YAMLError as exc:
            raise CorpusConfigError(
                f"Invalid YAML in corpus config {resolved_config_path}",
            ) from exc

        if not isinstance(loaded, dict):
            raise CorpusConfigError(
                f"Corpus config {resolved_config_path} must be a YAML mapping",
            )
        try:
            contract = CorpusConfigContract.from_dict(loaded)
        except ValueError as exc:
            raise CorpusConfigError(f"Invalid corpus config: {exc}") from exc

        # Use the validated representation as the sole runtime configuration.
        # This materializes Pydantic defaults once and avoids validate-then-use-
        # unvalidated-data drift.
        raw_config = contract.to_dict()
        languages = tuple(normalize_language(language) for language in raw_config["languages"])
        manual_path = raw_config.get("manual_utterances_path")
        return cls(
            config_path=resolved_config_path,
            repo_root=resolved_repo_root,
            raw_config=raw_config,
            corpus_id=raw_config["corpus_id"],
            source_root=cls._resolve_path(
                raw_config["source_root"],
                repo_root=resolved_repo_root,
            ),
            mapping_path=cls._resolve_path(
                raw_config["mapping_path"],
                repo_root=resolved_repo_root,
            ),
            output_dir=cls._resolve_path(
                raw_config["output_dir"],
                repo_root=resolved_repo_root,
            ),
            languages=languages,
            manual_utterances_path=(
                cls._resolve_path(manual_path, repo_root=resolved_repo_root)
                if manual_path
                else None
            ),
            additional_files=tuple(
                cls._configured_source_file(
                    item,
                    repo_root=resolved_repo_root,
                )
                for item in raw_config.get("additional_files", [])
            ),
            snapshot=dict(raw_config["snapshot"]),
            parser=dict(raw_config["parser"]),
            section_context_groups=dict(raw_config["section_context_groups"]),
            chunking=dict(raw_config["chunking"]),
            tables=dict(raw_config["tables"]),
            media=dict(raw_config["media"]),
            summaries=dict(raw_config["summaries"]),
            page_table=dict(raw_config["page_table"]),
            section_table=dict(raw_config["section_table"]),
            quality=dict(raw_config["quality"]),
            keywords=dict(raw_config["keywords"]),
        )

    @staticmethod
    def _resolve_path(path_value: str | Path, *, repo_root: Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path.resolve()
        return (repo_root / path).resolve()

    @classmethod
    def _configured_source_file(
        cls,
        item: JsonDict,
        *,
        repo_root: Path,
    ) -> ConfiguredSourceFile:
        path = cls._resolve_path(str(item["path"]), repo_root=repo_root)
        return ConfiguredSourceFile(
            path=path,
            relative_path=repo_relative_path(path, repo_root),
            role=str(item["role"]),
            required=bool(item["required"]),
            exists=path.exists(),
        )

    def relative_path(self, path: str | Path) -> str:
        return repo_relative_path(Path(path), self.repo_root)

    def metadata_source_files(self) -> list[ConfiguredSourceFile]:
        """Return metadata inputs that should seed `source_files.jsonl`."""

        files = [
            ConfiguredSourceFile(
                path=self.mapping_path,
                relative_path=self.relative_path(self.mapping_path),
                role="mapping",
                required=True,
                exists=self.mapping_path.exists(),
            ),
        ]
        if self.manual_utterances_path is not None:
            files.append(
                ConfiguredSourceFile(
                    path=self.manual_utterances_path,
                    relative_path=self.relative_path(self.manual_utterances_path),
                    role="manual_utterances",
                    required=False,
                    exists=self.manual_utterances_path.exists(),
                ),
            )
        files.extend(self.additional_files)
        return files


@dataclass(frozen=True)
class TopicMetadata:
    """One normalized row from the AZ mapping."""

    intent: str
    depth: int
    parent: str | None
    etag: str | None
    link: str | None
    mapping: str | None
    row: JsonDict
    manual_utterance_count: int = 0

    def title(self, language: str, *, page_title: str | None = None) -> str:
        """Resolve manual, authoritative page, and extracted title fallbacks."""

        normalized_language = normalize_language(language)
        manual_title = _optional_string(
            self.row.get(
                MANUAL_TITLE_COLUMN_BY_LANGUAGE[normalized_language],
            ),
        )
        if manual_title:
            return manual_title
        authoritative_title = _optional_string(page_title)
        if authoritative_title:
            return authoritative_title
        extracted_title = _optional_string(
            self.row.get(
                EXTRACTED_TITLE_COLUMN_BY_LANGUAGE[normalized_language],
            ),
        )
        if extracted_title:
            return extracted_title
        return self.intent.replace("-", " ").strip() or self.intent

    def url(self, language: str) -> str | None:
        normalized_language = normalize_language(language)
        if normalized_language == "en":
            return self.mapping or self.link
        return self.link or self.mapping

    def to_page_metadata(
        self,
        *,
        language: str,
        breadcrumbs: list[JsonDict] | None = None,
        page_title: str | None = None,
    ) -> JsonDict:
        normalized_language = normalize_language(language)
        return {
            "intent": self.intent,
            "language": normalized_language,
            "depth": self.depth,
            "parent": self.parent,
            "etag": self.etag,
            "title": self.title(normalized_language, page_title=page_title),
            "url": self.url(normalized_language),
            "breadcrumbs": breadcrumbs or [],
            "manual_utterance_count": self.manual_utterance_count,
        }


@dataclass(frozen=True)
class MetadataStore:
    """Loaded corpus metadata keyed by deterministic AZ intent IDs."""

    config: CorpusConfig
    topics_by_intent: dict[str, TopicMetadata]
    manual_utterances_by_intent: dict[str, list[JsonDict]] = field(default_factory=dict)
    additional_data_by_role: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: CorpusConfig,
        *,
        require_existing_files: bool = True,
    ) -> "MetadataStore":
        _validate_configured_files(config, require_existing_files=require_existing_files)
        mapping_rows = _load_mapping_rows(config.mapping_path)
        manual_utterances = _load_manual_utterances(config.manual_utterances_path)
        additional_data = _load_additional_files(config.additional_files)
        topics = _build_topics(mapping_rows, manual_utterances)
        return cls(
            config=config,
            topics_by_intent=topics,
            manual_utterances_by_intent=manual_utterances,
            additional_data_by_role=additional_data,
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = "configs/thesis/corpus.yaml",
        *,
        repo_root: str | Path | None = None,
        require_existing_files: bool = True,
    ) -> "MetadataStore":
        return cls.from_config(
            CorpusConfig.from_file(config_path, repo_root=repo_root),
            require_existing_files=require_existing_files,
        )

    @property
    def intents(self) -> list[str]:
        return sorted(self.topics_by_intent)

    def topics_sorted(self) -> list[TopicMetadata]:
        return [self.topics_by_intent[intent] for intent in self.intents]

    def get(self, intent: str) -> TopicMetadata:
        try:
            return self.topics_by_intent[intent]
        except KeyError as exc:
            raise MetadataError(f"Unknown AZ intent: {intent}") from exc

    def breadcrumbs_for(self, intent: str, *, language: str = "de") -> list[JsonDict]:
        normalized_language = normalize_language(language)
        chain: list[TopicMetadata] = []
        seen: set[str] = set()
        current: TopicMetadata | None = self.get(intent)
        while current is not None:
            if current.intent in seen:
                raise MetadataError(f"Cycle detected in mapping parent chain: {intent}")
            seen.add(current.intent)
            chain.append(current)
            current = (
                self.topics_by_intent.get(current.parent)
                if current.parent
                else None
            )
        return [
            {
                "intent": topic.intent,
                "title": topic.title(normalized_language),
                "url": topic.url(normalized_language),
                "depth": topic.depth,
            }
            for topic in reversed(chain)
        ]

    def page_metadata_for(
        self,
        intent: str,
        *,
        language: str,
        page_title: str | None = None,
    ) -> JsonDict:
        topic = self.get(intent)
        return topic.to_page_metadata(
            language=language,
            breadcrumbs=self.breadcrumbs_for(intent, language=language),
            page_title=page_title,
        )

    def source_manifest_inputs(self) -> list[ConfiguredSourceFile]:
        return self.config.metadata_source_files()


def repo_relative_path(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = repo_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _validate_configured_files(
    config: CorpusConfig,
    *,
    require_existing_files: bool,
) -> None:
    if require_existing_files and not config.mapping_path.exists():
        raise MetadataFileError(f"Required mapping file is missing: {config.mapping_path}")
    for source_file in config.metadata_source_files():
        if source_file.required and not source_file.exists:
            raise MetadataFileError(
                f"Required metadata file is missing: {source_file.relative_path}",
            )


def _load_mapping_rows(path: Path) -> list[JsonDict]:
    rows = _read_csv_dicts(path)
    if not rows:
        raise MappingSchemaError(f"Mapping file {path} has no data rows")
    fieldnames = set(rows[0])
    missing = sorted(REQUIRED_MAPPING_COLUMNS - fieldnames)
    if missing:
        raise MappingSchemaError(
            f"Mapping file {path} is missing required columns: {missing}",
        )
    return rows


def _load_manual_utterances(path: Path | None) -> dict[str, list[JsonDict]]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv_dicts(path)
    by_intent: dict[str, list[JsonDict]] = {}
    for row in rows:
        intent = _clean(row.get("intent"))
        if not intent:
            continue
        by_intent.setdefault(intent, []).append(dict(row))
    return by_intent


def _load_additional_files(
    source_files: Iterable[ConfiguredSourceFile],
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for source_file in source_files:
        if not source_file.exists:
            if source_file.required:
                raise MetadataFileError(
                    f"Required metadata file is missing: {source_file.relative_path}",
                )
            continue
        loaded[source_file.role] = _load_structured_file(source_file.path)
    return loaded


def _build_topics(
    mapping_rows: Iterable[JsonDict],
    manual_utterances: dict[str, list[JsonDict]],
) -> dict[str, TopicMetadata]:
    topics: dict[str, TopicMetadata] = {}
    for row in mapping_rows:
        intent = _clean(row.get("intent"))
        if not intent:
            continue
        try:
            depth = int(_clean(row.get("depth")) or "0")
        except ValueError as exc:
            raise MappingSchemaError(
                f"Mapping row for intent {intent!r} has invalid depth",
            ) from exc
        topics[intent] = TopicMetadata(
            intent=intent,
            depth=depth,
            parent=_optional_string(row.get("parent")),
            etag=_optional_string(row.get("etag")),
            link=_optional_string(row.get("link")),
            mapping=_optional_string(row.get("mapping")),
            row=dict(row),
            manual_utterance_count=len(manual_utterances.get(intent, [])),
        )
    return dict(sorted(topics.items()))
