"""Build and verify frozen thesis corpus snapshots."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Literal

from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)
from thesis_rag.corpus.chunking import (
    ChunkingConfig,
    build_chunk_artifacts,
    chunking_identity_config,
)
from thesis_rag.corpus.context_groups import (
    build_section_context_groups,
    normalized_context_group_config,
)
from thesis_rag.corpus.html_parser import parse_html_page, resolve_citation_url
from thesis_rag.corpus.metadata import (
    ConfiguredSourceFile,
    CorpusConfig,
    MetadataStore,
    repo_relative_path,
)
from thesis_rag.corpus.keywords import CorpusKeywordExtractor
from thesis_rag.corpus.page_table import write_page_section_tables
from thesis_rag.corpus.quality import (
    build_page_quality,
    normalized_quality_config,
    quality_records_to_dicts,
)
from thesis_rag.corpus.source_relationships import (
    build_source_relationships,
    contained_fragment_page_ids,
    normalized_source_relationship_config,
    source_relationship_records_to_dicts,
)
from thesis_rag.schema import (
    SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    SOURCE_FILE_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    corpus_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    sha256_prefixed,
    stable_id,
    validate_record_shape,
    write_jsonl,
)


PARSER_VERSION = "html_parser_v8"
SNAPSHOT_BUILDER_VERSION = "snapshot_builder_v4"
# Parsing happens before the final corpus digest can be computed.  Use a valid,
# unmistakable all-zero digest during that in-memory phase so the same strict
# persisted-record models can be used without weakening the SHA-256 contract.
PENDING_CORPUS_HASH = "sha256:" + ("0" * 64)
CompressionMode = Literal["auto", "zstd", "gzip"]
ResolvedCompression = Literal["zstd", "gzip"]


class SnapshotError(ValueError):
    """Base class for snapshot build and verification failures."""


class SnapshotBuildError(SnapshotError):
    """Raised when a corpus snapshot cannot be built safely."""


class SnapshotVerificationError(SnapshotError):
    """Raised when a snapshot does not match its manifest."""


class SourceFileRecord(ArtifactModel):
    """One immutable source file captured in the raw corpus archive."""

    omit_none = True

    source_path: NonEmptyStr
    archive_path: NonEmptyStr
    role: NonEmptyStr
    byte_size: NonNegativeInt
    file_hash: Sha256
    language: str | None = None
    intent: str | None = None
    mtime: NonNegativeInt | None = None
    schema_version: Literal["source_file_v1"] = SOURCE_FILE_SCHEMA_VERSION

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        repo_root: Path,
        role: str,
        language: str | None = None,
        intent: str | None = None,
        include_mtime: bool = False,
    ) -> "SourceFileRecord":
        if not path.exists():
            raise SnapshotBuildError(f"Source file is missing: {path}")
        if not path.is_file():
            raise SnapshotBuildError(f"Source path is not a file: {path}")
        relative_path = repo_relative_path(path, repo_root)
        stat = path.stat()
        return cls(
            source_path=relative_path,
            archive_path=relative_path,
            role=role,
            byte_size=stat.st_size,
            file_hash=hash_file(path),
            language=language,
            intent=intent,
            mtime=stat.st_mtime_ns if include_mtime else None,
        )

    @classmethod
    def from_configured_file(
        cls,
        source_file: ConfiguredSourceFile,
        *,
        repo_root: Path,
        include_mtime: bool = False,
    ) -> "SourceFileRecord | None":
        if not source_file.exists:
            if source_file.required:
                raise SnapshotBuildError(
                    f"Required source file is missing: {source_file.relative_path}",
                )
            return None
        return cls.from_path(
            source_file.path,
            repo_root=repo_root,
            role=source_file.role,
            language=source_file.language,
            intent=source_file.intent,
            include_mtime=include_mtime,
        )

    def to_hash_seed(self) -> JsonDict:
        """Return the stable source identity used for corpus hash input."""

        return {
            "source_path": self.source_path,
            "archive_path": self.archive_path,
            "role": self.role,
            "byte_size": self.byte_size,
            "file_hash": self.file_hash,
            "language": self.language,
            "intent": self.intent,
        }


class SnapshotManifest(ArtifactModel):
    """Manifest describing one frozen corpus snapshot."""

    omit_none = True

    corpus_id: NonEmptyStr
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    source_manifest_hash: Sha256
    archive_hash: Sha256
    artifact_hashes: dict[NonEmptyStr, Sha256 | dict[NonEmptyStr, Sha256]]
    artifact_record_counts: JsonDict
    artifact_paths: JsonDict
    parser_version: NonEmptyStr
    parser_config_hash: Sha256
    extraction_config_hash: Sha256
    artifact_config_hashes: dict[NonEmptyStr, Sha256]
    archive_compression: Literal["zstd", "gzip"]
    created_at_utc: NonEmptyStr
    source_file_count: NonNegativeInt
    source_counts_by_role: JsonDict
    source_counts_by_language: JsonDict
    page_count: NonNegativeInt
    section_count: NonNegativeInt
    content_block_count: NonNegativeInt
    table_count: NonNegativeInt
    image_count: NonNegativeInt
    snapshot_builder_version: NonEmptyStr = SNAPSHOT_BUILDER_VERSION
    chunking: JsonDict | None = None
    section_context_groups: JsonDict | None = None
    schema_version: Literal["snapshot_manifest_v5"] = SNAPSHOT_MANIFEST_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_manifest_maps(self) -> "SnapshotManifest":
        for name, value in self.artifact_hashes.items():
            if name == "chunks" and not isinstance(value, dict):
                raise ValueError("artifact_hashes.chunks must be a hash mapping")
            if name != "chunks" and not isinstance(value, str):
                raise ValueError(
                    f"artifact_hashes.{name} must be a single SHA-256 digest",
                )
        return self


class SnapshotSummary(ArtifactModel):
    """Small result object returned by build and verify operations."""

    corpus_id: NonEmptyStr
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    source_file_count: NonNegativeInt
    page_count: NonNegativeInt
    section_count: NonNegativeInt
    content_block_count: NonNegativeInt
    section_context_group_count: NonNegativeInt
    chunk_counts: dict[NonEmptyStr, NonNegativeInt]
    table_count: NonNegativeInt
    image_count: NonNegativeInt
    manifest_path: NonEmptyStr
    archive_path: NonEmptyStr
    source_manifest_path: NonEmptyStr
    archive_compression: ResolvedCompression
    dry_run: bool = False
    restored_file_count: NonNegativeInt = 0


@dataclass(frozen=True)
class _ArtifactPaths:
    source_manifest: Path
    archive: Path
    manifest: Path
    pages: Path
    sections: Path
    content_blocks: Path
    section_context_groups: Path
    tables: Path | None
    images: Path | None
    page_table: Path
    section_table: Path
    source_relationships: Path
    page_quality: Path
    quality_report: Path
    chunks: dict[str, Path]

    def staged(self, *, output_dir: Path, staging_dir: Path) -> "_ArtifactPaths":
        def stage(path: Path | None) -> Path | None:
            if path is None:
                return None
            try:
                relative = path.relative_to(output_dir)
            except ValueError as exc:
                raise SnapshotBuildError(
                    f"Generated artifact must be inside output_dir: {path}",
                ) from exc
            return staging_dir / relative

        return _ArtifactPaths(
            source_manifest=stage(self.source_manifest),  # type: ignore[arg-type]
            archive=stage(self.archive),  # type: ignore[arg-type]
            manifest=stage(self.manifest),  # type: ignore[arg-type]
            pages=stage(self.pages),  # type: ignore[arg-type]
            sections=stage(self.sections),  # type: ignore[arg-type]
            content_blocks=stage(self.content_blocks),  # type: ignore[arg-type]
            section_context_groups=stage(
                self.section_context_groups,
            ),  # type: ignore[arg-type]
            tables=stage(self.tables),
            images=stage(self.images),
            page_table=stage(self.page_table),  # type: ignore[arg-type]
            section_table=stage(self.section_table),  # type: ignore[arg-type]
            source_relationships=stage(
                self.source_relationships,
            ),  # type: ignore[arg-type]
            page_quality=stage(self.page_quality),  # type: ignore[arg-type]
            quality_report=stage(self.quality_report),  # type: ignore[arg-type]
            chunks={
                chunker_id: stage(path)  # type: ignore[misc]
                for chunker_id, path in self.chunks.items()
            },
        )

    def as_manifest_paths(self, repo_root: Path) -> JsonDict:
        data = {
            "source_files": repo_relative_path(self.source_manifest, repo_root),
            "archive": repo_relative_path(self.archive, repo_root),
            "pages": repo_relative_path(self.pages, repo_root),
            "sections": repo_relative_path(self.sections, repo_root),
            "content_blocks": repo_relative_path(self.content_blocks, repo_root),
            "section_context_groups": repo_relative_path(
                self.section_context_groups,
                repo_root,
            ),
        }
        if self.tables is not None:
            data["tables"] = repo_relative_path(self.tables, repo_root)
        if self.images is not None:
            data["images"] = repo_relative_path(self.images, repo_root)
        data["page_table"] = repo_relative_path(self.page_table, repo_root)
        data["section_table"] = repo_relative_path(self.section_table, repo_root)
        data["source_relationships"] = repo_relative_path(
            self.source_relationships,
            repo_root,
        )
        data["page_quality"] = repo_relative_path(self.page_quality, repo_root)
        data["quality_report"] = repo_relative_path(self.quality_report, repo_root)
        data["chunks"] = {
            chunker_id: repo_relative_path(path, repo_root)
            for chunker_id, path in sorted(self.chunks.items())
        }
        return data

    def generated_paths(self) -> list[Path]:
        paths = [
            self.source_manifest,
            self.archive,
            self.manifest,
            self.pages,
            self.sections,
            self.content_blocks,
            self.section_context_groups,
        ]
        if self.tables is not None:
            paths.append(self.tables)
        if self.images is not None:
            paths.append(self.images)
        paths.append(self.page_table)
        paths.append(self.section_table)
        paths.append(self.source_relationships)
        paths.append(self.page_quality)
        paths.append(self.quality_report)
        paths.extend(path for _, path in sorted(self.chunks.items()))
        return paths


class _StagingDirectory:
    """Own a staging directory until it is promoted or explicitly cleaned."""

    def __init__(self, parent: Path) -> None:
        self.path = Path(
            tempfile.mkdtemp(
                prefix=".corpus-staging-",
                dir=parent,
            ),
        )
        self._active = True

    def promote(self, output_dir: Path) -> None:
        _promote_staged_output(
            staging_dir=self.path,
            output_dir=output_dir,
        )
        self._active = False

    def cleanup(self) -> None:
        if self._active and self.path.exists():
            shutil.rmtree(self.path)
        self._active = False

    def __del__(self) -> None:
        self.cleanup()


class SnapshotBuilder:
    """Create deterministic source archives and parsed corpus artifacts."""

    def __init__(
        self,
        config: CorpusConfig | str | Path = "configs/thesis/corpus.yaml",
        *,
        repo_root: str | Path | None = None,
        compression: CompressionMode = "auto",
        include_mtime: bool = False,
    ) -> None:
        self.config = (
            config
            if isinstance(config, CorpusConfig)
            else CorpusConfig.from_file(config, repo_root=repo_root)
        )
        self.repo_root = self.config.repo_root
        self.compression = compression
        self.include_mtime = include_mtime
        self.metadata_store = MetadataStore.from_config(self.config)

    def build(self, *, force: bool = False, dry_run: bool = False) -> SnapshotSummary:
        """Build the configured snapshot artifacts."""

        source_records = self.discover_source_files()
        artifact_paths = self._artifact_paths()
        compression = _resolve_compression(self.compression)
        source_snapshot_hash = _hash_jsonl_records(
            record.to_dict() for record in source_records
        )
        parser_config_hash = config_hash(
            _semantic_parser_config(self.config),
        )
        extraction_config_hash = hash_jsonable(
            _corpus_extraction_config(self.config),
        )
        artifact_config_hashes = _artifact_config_hashes(self.config)
        parsed_pages = self._parse_source_pages(
            source_records,
            PENDING_CORPUS_HASH,
        )
        page_records = [parsed.page.to_dict() for parsed in parsed_pages]
        section_records = [
            section.to_dict()
            for parsed in parsed_pages
            for section in parsed.sections
        ]
        content_block_records = [
            block.to_dict()
            for parsed in parsed_pages
            for block in parsed.content_blocks
        ]
        table_records = (
            [
                table.to_dict()
                for parsed in parsed_pages
                for table in parsed.tables
            ]
            if bool(self.config.tables.get("extract", False))
            else []
        )
        image_records = _selected_media_records(
            parsed_pages,
            extract_images=bool(self.config.media.get("extract_images", False)),
            extract_videos=bool(self.config.media.get("extract_videos", False)),
        )
        resolved_corpus_hash = _canonical_corpus_hash(
            corpus_id=self.config.corpus_id,
            parser_config_hash=parser_config_hash,
            extraction_config_hash=extraction_config_hash,
            pages=page_records,
            sections=section_records,
            tables=table_records,
            images=image_records,
        )
        for records in (
            page_records,
            section_records,
            content_block_records,
            table_records,
            image_records,
        ):
            for record in records:
                record["corpus_hash"] = resolved_corpus_hash
        section_context_group_records = [
            record.to_dict()
            for record in build_section_context_groups(
                sections=section_records,
                content_blocks=content_block_records,
                config=self.config.section_context_groups,
                corpus_hash=resolved_corpus_hash,
            )
        ]
        chunk_source_hashes = {
            "pages": _hash_jsonl_records(page_records),
            "sections": _hash_jsonl_records(section_records),
            "content_blocks": _hash_jsonl_records(content_block_records),
            "section_context_groups": _hash_jsonl_records(
                section_context_group_records,
            ),
        }
        chunk_artifacts = build_chunk_artifacts(
            pages=page_records,
            sections=section_records,
            context_groups=section_context_group_records,
            content_blocks=content_block_records,
            chunker_configs=list(self.config.chunking["chunkers"]),
            corpus_hash=resolved_corpus_hash,
            repo_root=self.repo_root,
            source_artifact_hashes=chunk_source_hashes,
            write_artifacts=False,
        )
        source_relationships = build_source_relationships(
            page_records,
            corpus_hash=resolved_corpus_hash,
        )
        source_relationship_records = source_relationship_records_to_dicts(
            source_relationships,
        )
        previous_quality_report = _read_optional_json(
            artifact_paths.quality_report,
        )
        base_page_quality, _ = build_page_quality(
            page_records,
            quality_config=self.config.quality,
            corpus_hash=resolved_corpus_hash,
        )
        eligible_parent_page_ids = {
            record.page_id
            for record in base_page_quality
            if record.eligible_standard_index
        }
        page_quality, quality_report = build_page_quality(
            page_records,
            quality_config=self.config.quality,
            corpus_hash=resolved_corpus_hash,
            previous_report=previous_quality_report,
            contained_fragment_page_ids=contained_fragment_page_ids(
                source_relationships,
                eligible_parent_page_ids=eligible_parent_page_ids,
            ),
        )
        page_quality_records = quality_records_to_dicts(page_quality)

        if dry_run:
            return SnapshotSummary(
                corpus_id=self.config.corpus_id,
                corpus_hash=resolved_corpus_hash,
                source_snapshot_hash=source_snapshot_hash,
                source_file_count=len(source_records),
                page_count=len(page_records),
                section_count=len(section_records),
                content_block_count=len(content_block_records),
                section_context_group_count=len(section_context_group_records),
                chunk_counts={
                    chunker_id: len(records)
                    for chunker_id, records in sorted(chunk_artifacts.items())
                },
                table_count=len(table_records),
                image_count=len(image_records),
                manifest_path=repo_relative_path(artifact_paths.manifest, self.repo_root),
                archive_path=repo_relative_path(artifact_paths.archive, self.repo_root),
                source_manifest_path=repo_relative_path(
                    artifact_paths.source_manifest,
                    self.repo_root,
                ),
                archive_compression=compression,
                dry_run=True,
            )

        final_artifact_paths = artifact_paths
        self._check_overwrite_policy(final_artifact_paths, force=force)
        final_output_dir = self.config.output_dir
        final_output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = _StagingDirectory(final_output_dir.parent)
        staging_dir = staging.path
        artifact_paths = final_artifact_paths.staged(
            output_dir=final_output_dir,
            staging_dir=staging_dir,
        )
        _copy_keep_files(final_output_dir, staging_dir)

        write_jsonl(
            artifact_paths.source_manifest,
            [record.to_dict() for record in source_records],
        )
        _write_archive(
            artifact_paths.archive,
            source_records,
            repo_root=self.repo_root,
            compression=compression,
        )
        write_jsonl(artifact_paths.pages, page_records)
        write_jsonl(artifact_paths.sections, section_records)
        write_jsonl(artifact_paths.content_blocks, content_block_records)
        write_jsonl(
            artifact_paths.section_context_groups,
            section_context_group_records,
        )
        if artifact_paths.tables is not None:
            write_jsonl(artifact_paths.tables, table_records)
        if artifact_paths.images is not None:
            write_jsonl(artifact_paths.images, image_records)
        write_jsonl(artifact_paths.source_relationships, source_relationship_records)
        write_page_section_tables(
            page_table_path=artifact_paths.page_table,
            section_table_path=artifact_paths.section_table,
            pages=page_records,
            sections=section_records,
            tables=table_records,
            images=image_records,
            metadata_store=self.metadata_store,
            keyword_extractor=CorpusKeywordExtractor.from_config(
                self.config.keywords,
            ),
        )
        write_jsonl(artifact_paths.page_quality, page_quality_records)
        _write_json(artifact_paths.quality_report, quality_report)
        for chunker_id, records in chunk_artifacts.items():
            write_jsonl(
                artifact_paths.chunks[chunker_id],
                [record.to_dict() for record in records],
            )

        artifact_hashes: JsonDict = {
            "pages": hash_file(artifact_paths.pages),
            "sections": hash_file(artifact_paths.sections),
            "content_blocks": hash_file(artifact_paths.content_blocks),
            "section_context_groups": hash_file(
                artifact_paths.section_context_groups,
            ),
            "page_table": hash_file(artifact_paths.page_table),
            "section_table": hash_file(artifact_paths.section_table),
            "source_relationships": hash_file(artifact_paths.source_relationships),
            "page_quality": hash_file(artifact_paths.page_quality),
            "quality_report": hash_file(artifact_paths.quality_report),
        }
        if artifact_paths.tables is not None:
            artifact_hashes["tables"] = hash_file(artifact_paths.tables)
        if artifact_paths.images is not None:
            artifact_hashes["images"] = hash_file(artifact_paths.images)
        artifact_hashes["chunks"] = {
            chunker_id: hash_file(artifact_paths.chunks[chunker_id])
            for chunker_id in sorted(chunk_artifacts)
        }
        artifact_record_counts: JsonDict = {
            "pages": len(page_records),
            "sections": len(section_records),
            "content_blocks": len(content_block_records),
            "section_context_groups": len(section_context_group_records),
            "page_table": len(page_records),
            "section_table": len(section_records),
            "source_relationships": len(source_relationship_records),
            "page_quality": len(page_quality_records),
            "chunks": {
                chunker_id: len(records)
                for chunker_id, records in sorted(chunk_artifacts.items())
            },
        }
        if artifact_paths.tables is not None:
            artifact_record_counts["tables"] = len(table_records)
        if artifact_paths.images is not None:
            artifact_record_counts["images"] = len(image_records)
        chunker_manifest = [
            {
                "chunker_id": str(raw_config["chunker_id"]),
                "mode": str(raw_config["mode"]),
                "config": ChunkingConfig.from_dict(raw_config).to_identity_dict(),
                "output_path": repo_relative_path(
                    final_artifact_paths.chunks[str(raw_config["chunker_id"])],
                    self.repo_root,
                ),
                "record_count": len(chunk_artifacts[str(raw_config["chunker_id"])]),
                "artifact_hash": artifact_hashes["chunks"][
                    str(raw_config["chunker_id"])
                ],
                "config_hash": ChunkingConfig.from_dict(
                    raw_config,
                ).identity_hash,
            }
            for raw_config in self.config.chunking["chunkers"]
        ]

        manifest = SnapshotManifest(
            corpus_id=self.config.corpus_id,
            corpus_hash=resolved_corpus_hash,
            source_snapshot_hash=source_snapshot_hash,
            source_manifest_hash=hash_file(artifact_paths.source_manifest),
            archive_hash=hash_file(artifact_paths.archive),
            artifact_hashes=artifact_hashes,
            artifact_record_counts=artifact_record_counts,
            artifact_paths=final_artifact_paths.as_manifest_paths(self.repo_root),
            parser_version=PARSER_VERSION,
            parser_config_hash=parser_config_hash,
            extraction_config_hash=extraction_config_hash,
            artifact_config_hashes=artifact_config_hashes,
            archive_compression=compression,
            created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            source_file_count=len(source_records),
            source_counts_by_role=_counts_by(
                record.role for record in source_records
            ),
            source_counts_by_language=_counts_by(
                record.language for record in source_records if record.language
            ),
            page_count=len(page_records),
            section_count=len(section_records),
            content_block_count=len(content_block_records),
            table_count=len(table_records),
            image_count=len(image_records),
            chunking={
                "ran": True,
                "config_hash": artifact_config_hashes["chunking"],
                "chunkers": chunker_manifest,
            },
            section_context_groups={
                "config": normalized_context_group_config(
                    self.config.section_context_groups,
                ),
                "config_hash": artifact_config_hashes[
                    "section_context_groups"
                ],
                "record_count": len(section_context_group_records),
            },
        )
        _write_json(artifact_paths.manifest, manifest.to_dict())
        _verify_staged_snapshot(
            manifest,
            artifact_paths=artifact_paths,
            source_records=source_records,
        )
        staging.promote(final_output_dir)
        return SnapshotSummary(
            corpus_id=self.config.corpus_id,
            corpus_hash=resolved_corpus_hash,
            source_snapshot_hash=source_snapshot_hash,
            source_file_count=len(source_records),
            page_count=len(page_records),
            section_count=len(section_records),
            content_block_count=len(content_block_records),
            section_context_group_count=len(section_context_group_records),
            chunk_counts={
                chunker_id: len(records)
                for chunker_id, records in sorted(chunk_artifacts.items())
            },
            table_count=len(table_records),
            image_count=len(image_records),
            manifest_path=repo_relative_path(
                final_artifact_paths.manifest,
                self.repo_root,
            ),
            archive_path=repo_relative_path(
                final_artifact_paths.archive,
                self.repo_root,
            ),
            source_manifest_path=repo_relative_path(
                final_artifact_paths.source_manifest,
                self.repo_root,
            ),
            archive_compression=compression,
        )

    def discover_source_files(self) -> list[SourceFileRecord]:
        """Return all configured and scraped source files in stable order."""

        records: dict[str, SourceFileRecord] = {}
        for configured in self.metadata_store.source_manifest_inputs():
            record = SourceFileRecord.from_configured_file(
                configured,
                repo_root=self.repo_root,
                include_mtime=self.include_mtime,
            )
            if record is not None:
                records[record.archive_path] = record

        source_root = self.config.source_root
        if not source_root.exists():
            raise SnapshotBuildError(f"Configured source_root is missing: {source_root}")

        content_names = {
            str(file_name): language
            for language, file_name in sorted(self.config.parser["content_files"].items())
        }
        page_names = {
            str(file_name): language
            for language, file_name in sorted(self.config.parser["page_files"].items())
        }
        include_page_html = bool(self.config.snapshot["include_page_html"])
        include_versions = bool(self.config.snapshot["include_versions"])

        for topic_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            intent = topic_dir.name
            for file_name, language in content_names.items():
                path = topic_dir / file_name
                if path.exists():
                    records[repo_relative_path(path, self.repo_root)] = (
                        SourceFileRecord.from_path(
                            path,
                            repo_root=self.repo_root,
                            role="content_html",
                            language=language,
                            intent=intent,
                            include_mtime=self.include_mtime,
                        )
                    )
            if include_page_html:
                for file_name, language in page_names.items():
                    path = topic_dir / file_name
                    if path.exists():
                        records[repo_relative_path(path, self.repo_root)] = (
                            SourceFileRecord.from_path(
                                path,
                                repo_root=self.repo_root,
                                role="page_html",
                                language=language,
                                intent=intent,
                                include_mtime=self.include_mtime,
                            )
                        )
            if include_versions:
                path = topic_dir / "versions.csv"
                if path.exists():
                    records[repo_relative_path(path, self.repo_root)] = (
                        SourceFileRecord.from_path(
                            path,
                            repo_root=self.repo_root,
                            role="versions_csv",
                            intent=intent,
                            include_mtime=self.include_mtime,
                        )
                    )

        return sorted(records.values(), key=lambda record: record.archive_path)

    def _parse_source_pages(
        self,
        source_records: list[SourceFileRecord],
        resolved_corpus_hash: str,
    ):
        page_lookup = {
            (record.intent, record.language): self.repo_root / record.source_path
            for record in source_records
            if record.role == "page_html" and record.intent and record.language
        }
        content_records = [
            record
            for record in source_records
            if record.role == "content_html" and record.intent and record.language
        ]
        parsed_pages = []
        fallback_heading = str(
            self.config.parser.get("fallback_section_heading", "__page__"),
        )
        for record in content_records:
            topic_metadata = self.metadata_store.topics_by_intent.get(
                str(record.intent),
            )
            page_path = page_lookup.get((record.intent, record.language))
            parsed_pages.append(
                parse_html_page(
                    content_path=self.repo_root / record.source_path,
                    page_path=page_path,
                    repo_root=self.repo_root,
                    corpus_id=self.config.corpus_id,
                    corpus_hash=resolved_corpus_hash,
                    intent=str(record.intent),
                    language=str(record.language),
                    topic_metadata=topic_metadata,
                    fallback_section_heading=fallback_heading,
                ),
            )
        return parsed_pages

    def _artifact_paths(self) -> _ArtifactPaths:
        output_dir = self.config.output_dir
        return _ArtifactPaths(
            source_manifest=_resolve_repo_path(
                self.config.snapshot["source_manifest_path"],
                self.repo_root,
            ),
            archive=_resolve_repo_path(
                self.config.snapshot["archive_path"],
                self.repo_root,
            ),
            manifest=_resolve_repo_path(
                self.config.snapshot["manifest_path"],
                self.repo_root,
            ),
            pages=output_dir / "pages.jsonl",
            sections=output_dir / "sections.jsonl",
            content_blocks=output_dir / "content_blocks.jsonl",
            section_context_groups=_resolve_repo_path(
                self.config.section_context_groups["output_path"],
                self.repo_root,
            ),
            tables=(
                _resolve_repo_path(self.config.tables["output_path"], self.repo_root)
                if self.config.tables.get("extract", False)
                else None
            ),
            images=(
                _resolve_repo_path(
                    self.config.media["images_output_path"],
                    self.repo_root,
                )
                if self.config.media.get("extract_images", False)
                or self.config.media.get("extract_videos", False)
                else None
            ),
            page_table=_resolve_repo_path(
                self.config.page_table["output_path"],
                self.repo_root,
            ),
            section_table=_resolve_repo_path(
                self.config.section_table["output_path"],
                self.repo_root,
            ),
            source_relationships=output_dir / "source_relationships.jsonl",
            page_quality=_resolve_repo_path(
                self.config.quality["output_path"],
                self.repo_root,
            ),
            quality_report=_resolve_repo_path(
                self.config.quality["report_path"],
                self.repo_root,
            ),
            chunks={
                str(chunker["chunker_id"]): _resolve_repo_path(
                    str(chunker["output_path"]),
                    self.repo_root,
                )
                for chunker in self.config.chunking["chunkers"]
            },
        )

    def _check_overwrite_policy(
        self,
        artifact_paths: _ArtifactPaths,
        *,
        force: bool,
    ) -> None:
        if force:
            return
        existing = [
            repo_relative_path(path, self.repo_root)
            for path in artifact_paths.generated_paths()
            if path.exists()
        ]
        if existing:
            raise SnapshotBuildError(
                "Snapshot artifacts already exist; pass force=True or --force "
                f"to overwrite: {existing}",
            )


class SnapshotVerifier:
    """Verify a frozen snapshot manifest and its recorded artifacts."""

    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None

    def verify(
        self,
        manifest_path: str | Path,
        *,
        archive_only: bool = False,
        restore_dir: str | Path | None = None,
    ) -> SnapshotSummary:
        manifest_file = Path(manifest_path)
        repo_root = self.repo_root or _find_repo_root_from_manifest(manifest_file)
        manifest_data = _read_json(manifest_file)
        manifest = SnapshotManifest.from_dict(manifest_data)

        errors: list[str] = []
        source_manifest_path = _manifest_artifact_path(
            manifest,
            "source_files",
            repo_root,
        )
        archive_path = _manifest_artifact_path(manifest, "archive", repo_root)

        _check_file_hash(
            source_manifest_path,
            manifest.source_manifest_hash,
            "source file manifest",
            errors,
        )
        _check_file_hash(archive_path, manifest.archive_hash, "archive", errors)
        for artifact_name, artifact_path, expected_hash in _iter_artifact_hashes(
            manifest,
            repo_root,
        ):
            _check_file_hash(
                artifact_path,
                expected_hash,
                f"{artifact_name} artifact",
                errors,
            )

        source_records = _read_source_records(source_manifest_path, errors)
        if len(source_records) != manifest.source_file_count:
            errors.append(
                "source file count mismatch: "
                f"manifest={manifest.source_file_count} actual={len(source_records)}",
            )
        _verify_manifest_semantics(
            manifest,
            source_records=source_records,
            artifact_paths=_artifact_paths_from_manifest(manifest, repo_root),
            errors=errors,
        )

        restored_file_count = 0
        if archive_only:
            errors.extend(
                _verify_archive_contents(
                    archive_path,
                    source_records,
                    compression=manifest.archive_compression,
                ),
            )
        else:
            errors.extend(_verify_source_tree(repo_root, source_records))

        if errors:
            raise SnapshotVerificationError("; ".join(errors))

        if restore_dir is not None:
            restored_file_count = _restore_archive(
                archive_path,
                restore_dir=Path(restore_dir),
                source_records=source_records,
                compression=manifest.archive_compression,
            )

        return SnapshotSummary(
            corpus_id=manifest.corpus_id,
            corpus_hash=manifest.corpus_hash,
            source_snapshot_hash=manifest.source_snapshot_hash,
            source_file_count=manifest.source_file_count,
            page_count=manifest.page_count,
            section_count=manifest.section_count,
            content_block_count=manifest.content_block_count,
            section_context_group_count=int(
                manifest.artifact_record_counts["section_context_groups"],
            ),
            chunk_counts={
                str(chunker_id): int(record_count)
                for chunker_id, record_count in dict(
                    manifest.artifact_record_counts["chunks"],
                ).items()
            },
            table_count=manifest.table_count,
            image_count=manifest.image_count,
            manifest_path=repo_relative_path(manifest_file, repo_root),
            archive_path=repo_relative_path(archive_path, repo_root),
            source_manifest_path=repo_relative_path(source_manifest_path, repo_root),
            archive_compression=manifest.archive_compression,
            restored_file_count=restored_file_count,
        )


def _resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _corpus_extraction_config(config: CorpusConfig) -> JsonDict:
    """Return source-selection settings that define corpus membership."""

    return {
        "languages": list(config.languages),
        "snapshot": {
            "include_page_html": config.snapshot["include_page_html"],
        },
        "titles": dict(config.raw_config.get("titles", {})),
    }


def _semantic_parser_config(config: CorpusConfig) -> JsonDict:
    """Return only parser settings that affect canonical page semantics.

    Source filenames select raw inputs and are therefore covered by the source
    snapshot.  Parser implementation behavior is covered by ``PARSER_VERSION``.
    """

    return {
        "fallback_section_heading": str(
            config.parser.get("fallback_section_heading", "__page__"),
        ),
    }


def _selected_media_records(
    parsed_pages: Iterable,
    *,
    extract_images: bool,
    extract_videos: bool,
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for parsed in parsed_pages:
        for image in parsed.images:
            is_video = image.media_type == "video"
            if (is_video and extract_videos) or (not is_video and extract_images):
                records.append(image.to_dict())
    return records


def _artifact_config_hashes(config: CorpusConfig) -> JsonDict:
    return {
        "chunking": config_hash(
            chunking_identity_config(list(config.chunking["chunkers"])),
        ),
        "media": config_hash(
            {
                "extract_images": bool(config.media.get("extract_images", False)),
                "extract_videos": bool(config.media.get("extract_videos", False)),
            },
        ),
        "section_context_groups": config_hash(
            normalized_context_group_config(config.section_context_groups),
        ),
        "tables": config_hash(
            {"extract": bool(config.tables.get("extract", False))},
        ),
        "quality": config_hash(
            normalized_quality_config(config.quality),
        ),
        "source_relationships": config_hash(
            normalized_source_relationship_config(),
        ),
        "keywords": CorpusKeywordExtractor.from_config(
            config.keywords,
        ).identity_hash,
    }


def _jsonl_bytes(records: Iterable[JsonDict]) -> bytes:
    return "".join(f"{canonical_json(record)}\n" for record in records).encode("utf-8")


def _hash_jsonl_records(records: Iterable[JsonDict]) -> str:
    return sha256_prefixed(_jsonl_bytes(records))


def _copy_keep_files(output_dir: Path, staging_dir: Path) -> None:
    if not output_dir.exists():
        return
    for keep_file in output_dir.rglob(".gitkeep"):
        relative = keep_file.relative_to(output_dir)
        target = staging_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(keep_file, target)


def _verify_staged_snapshot(
    manifest: SnapshotManifest,
    *,
    artifact_paths: _ArtifactPaths,
    source_records: list[SourceFileRecord],
) -> None:
    errors: list[str] = []
    _check_file_hash(
        artifact_paths.source_manifest,
        manifest.source_manifest_hash,
        "source file manifest",
        errors,
    )
    _check_file_hash(
        artifact_paths.archive,
        manifest.archive_hash,
        "archive",
        errors,
    )
    paths: JsonDict = {
        "pages": artifact_paths.pages,
        "sections": artifact_paths.sections,
        "content_blocks": artifact_paths.content_blocks,
        "section_context_groups": artifact_paths.section_context_groups,
        "page_table": artifact_paths.page_table,
        "section_table": artifact_paths.section_table,
        "source_relationships": artifact_paths.source_relationships,
        "page_quality": artifact_paths.page_quality,
        "quality_report": artifact_paths.quality_report,
        "chunks": artifact_paths.chunks,
    }
    if artifact_paths.tables is not None:
        paths["tables"] = artifact_paths.tables
    if artifact_paths.images is not None:
        paths["images"] = artifact_paths.images
    for artifact_name, expected_hash in manifest.artifact_hashes.items():
        if isinstance(expected_hash, dict):
            nested_paths = paths.get(artifact_name)
            if not isinstance(nested_paths, dict):
                errors.append(f"missing staged paths for {artifact_name}")
                continue
            for nested_name, nested_hash in expected_hash.items():
                _check_file_hash(
                    Path(nested_paths[nested_name]),
                    str(nested_hash),
                    f"{artifact_name}.{nested_name} artifact",
                    errors,
                )
        else:
            _check_file_hash(
                Path(paths[artifact_name]),
                str(expected_hash),
                f"{artifact_name} artifact",
                errors,
            )
    errors.extend(
        _verify_archive_contents(
            artifact_paths.archive,
            source_records,
            compression=manifest.archive_compression,
        ),
    )
    if _hash_jsonl_records(record.to_dict() for record in source_records) != (
        manifest.source_snapshot_hash
    ):
        errors.append("source snapshot hash mismatch")

    for artifact_name, expected_count in manifest.artifact_record_counts.items():
        if isinstance(expected_count, dict):
            nested_paths = paths.get(artifact_name)
            if not isinstance(nested_paths, dict):
                continue
            for nested_name, nested_count in expected_count.items():
                actual_count = len(read_jsonl(nested_paths[nested_name]))
                if actual_count != nested_count:
                    errors.append(
                        f"{artifact_name}.{nested_name} record count mismatch: "
                        f"expected {nested_count}, got {actual_count}",
                    )
        else:
            actual_count = len(read_jsonl(paths[artifact_name]))
            if actual_count != expected_count:
                errors.append(
                    f"{artifact_name} record count mismatch: "
                    f"expected {expected_count}, got {actual_count}",
                )
    _verify_manifest_semantics(
        manifest,
        source_records=source_records,
        artifact_paths=paths,
        errors=errors,
    )
    if errors:
        raise SnapshotBuildError("; ".join(errors))


def _artifact_paths_from_manifest(
    manifest: SnapshotManifest,
    repo_root: Path,
) -> JsonDict:
    paths: JsonDict = {}
    for artifact_name, raw_path in manifest.artifact_paths.items():
        if artifact_name in {"archive", "source_files"}:
            continue
        if isinstance(raw_path, dict):
            paths[artifact_name] = {
                nested_name: _resolve_repo_path(str(nested_path), repo_root)
                for nested_name, nested_path in raw_path.items()
            }
        else:
            paths[artifact_name] = _resolve_repo_path(str(raw_path), repo_root)
    return paths


def _verify_manifest_semantics(
    manifest: SnapshotManifest,
    *,
    source_records: list[SourceFileRecord],
    artifact_paths: JsonDict,
    errors: list[str],
) -> None:
    """Recompute record identities and relationships independently of file hashes."""

    if manifest.parser_version != PARSER_VERSION:
        errors.append(
            f"parser version mismatch: expected {PARSER_VERSION}, "
            f"got {manifest.parser_version}",
        )
    if manifest.snapshot_builder_version != SNAPSHOT_BUILDER_VERSION:
        errors.append(
            f"snapshot builder version mismatch: expected {SNAPSHOT_BUILDER_VERSION}, "
            f"got {manifest.snapshot_builder_version}",
        )

    actual_source_hash = _hash_jsonl_records(
        record.to_dict() for record in source_records
    )
    if actual_source_hash != manifest.source_snapshot_hash:
        errors.append(
            "source snapshot hash mismatch: "
            f"expected {manifest.source_snapshot_hash}, got {actual_source_hash}",
        )
    if _counts_by(record.role for record in source_records) != (
        manifest.source_counts_by_role
    ):
        errors.append("source counts by role mismatch")
    if _counts_by(
        record.language for record in source_records if record.language
    ) != manifest.source_counts_by_language:
        errors.append("source counts by language mismatch")

    records: dict[str, list[JsonDict]] = {}
    for artifact_name in (
        "pages",
        "sections",
        "content_blocks",
        "section_context_groups",
        "tables",
        "images",
        "page_table",
        "section_table",
        "source_relationships",
        "page_quality",
    ):
        path = artifact_paths.get(artifact_name)
        if path is None:
            records[artifact_name] = []
            continue
        try:
            records[artifact_name] = read_jsonl(Path(path))
        except (OSError, ValueError) as exc:
            errors.append(f"{artifact_name} artifact cannot be read: {exc}")
            records[artifact_name] = []

    quality_report: JsonDict | None = None
    quality_report_path = artifact_paths.get("quality_report")
    if quality_report_path is None:
        errors.append("quality report artifact is missing")
    else:
        try:
            quality_report = _read_json(Path(quality_report_path))
            validate_record_shape(quality_report)
        except (OSError, ValueError) as exc:
            errors.append(f"quality report artifact cannot be read: {exc}")

    chunk_records: dict[str, list[JsonDict]] = {}
    raw_chunk_paths = artifact_paths.get("chunks", {})
    if not isinstance(raw_chunk_paths, dict):
        errors.append("chunk artifact paths must be an object")
        raw_chunk_paths = {}
    for chunker_id, path in sorted(raw_chunk_paths.items()):
        try:
            chunk_records[str(chunker_id)] = read_jsonl(Path(path))
        except (OSError, ValueError) as exc:
            errors.append(f"chunks.{chunker_id} artifact cannot be read: {exc}")
            chunk_records[str(chunker_id)] = []

    _verify_record_counts(manifest, records, chunk_records, errors)
    _verify_core_records(manifest, records, chunk_records, errors)
    _verify_section_context_groups(
        manifest,
        sections=records["sections"],
        content_blocks=records["content_blocks"],
        groups=records["section_context_groups"],
        artifact_path=artifact_paths.get("section_context_groups"),
        errors=errors,
    )
    _verify_source_relationship_records(
        manifest,
        pages=records["pages"],
        source_relationship_records=records["source_relationships"],
        errors=errors,
    )
    _verify_quality_records(
        manifest,
        pages=records["pages"],
        source_relationship_records=records["source_relationships"],
        quality_records=records["page_quality"],
        quality_report=quality_report,
        errors=errors,
    )


def _verify_record_counts(
    manifest: SnapshotManifest,
    records: dict[str, list[JsonDict]],
    chunk_records: dict[str, list[JsonDict]],
    errors: list[str],
) -> None:
    actual_counts: JsonDict = {
        name: len(items)
        for name, items in records.items()
        if name in manifest.artifact_record_counts
    }
    actual_counts["chunks"] = {
        name: len(items) for name, items in sorted(chunk_records.items())
    }
    for name, expected in manifest.artifact_record_counts.items():
        actual = actual_counts.get(name)
        if actual != expected:
            errors.append(
                f"{name} record count mismatch: expected {expected}, got {actual}",
            )
    top_level_counts = {
        "pages": manifest.page_count,
        "sections": manifest.section_count,
        "content_blocks": manifest.content_block_count,
        "tables": manifest.table_count,
        "images": manifest.image_count,
    }
    for name, expected in top_level_counts.items():
        actual = len(records[name])
        if actual != expected:
            errors.append(
                f"{name} manifest count mismatch: expected {expected}, got {actual}",
            )


def _verify_core_records(
    manifest: SnapshotManifest,
    records: dict[str, list[JsonDict]],
    chunk_records: dict[str, list[JsonDict]],
    errors: list[str],
) -> None:
    pages = records["pages"]
    sections = records["sections"]
    content_blocks = records["content_blocks"]
    tables = records["tables"]
    images = records["images"]
    pages_by_id = _unique_records(pages, "page_id", errors)
    sections_by_id = _unique_records(sections, "section_id", errors)
    tables_by_id = _unique_records(tables, "table_id", errors)
    images_by_id = _unique_records(images, "image_id", errors)
    content_blocks_by_id = _unique_records(content_blocks, "block_id", errors)
    context_groups_by_id = _unique_records(
        records["section_context_groups"],
        "group_id",
        errors,
    )
    _unique_records(pages, "page_key", errors)
    _unique_records(sections, "section_key", errors)

    for page in pages:
        _check_record_corpus_hash(page, manifest.corpus_hash, "page", errors)
        expected_text_hash = hash_text(str(page["text"]))
        _check_value(page, "text_hash", expected_text_hash, "page", errors)
        expected_key = stable_id(
            "page_key",
            manifest.corpus_id,
            page["language"],
            page["intent"],
        )
        _check_value(page, "page_key", expected_key, "page", errors)
        expected_id = stable_id(
            "page",
            expected_key,
            page.get("url"),
            expected_text_hash,
        )
        _check_value(page, "page_id", expected_id, "page", errors)
        _check_url_provenance(page, "page", errors)

    for section in sections:
        label = f"section {section.get('section_id')}"
        _check_record_corpus_hash(section, manifest.corpus_hash, label, errors)
        page = pages_by_id.get(str(section["page_id"]))
        if page is None:
            errors.append(f"{label} references missing page {section['page_id']}")
            continue
        expected_hash = hash_text(str(section["text"]))
        _check_value(section, "text_hash", expected_hash, label, errors)
        expected_key = stable_id(
            "section_key",
            page["page_key"],
            section["heading_path"],
            section["section_index"],
        )
        _check_value(section, "section_key", expected_key, label, errors)
        _check_value(
            section,
            "section_id",
            stable_id("section", expected_key, expected_hash),
            label,
            errors,
        )
        _check_value(section, "page_key", page["page_key"], label, errors)
        _check_text_slice(section, page, label, errors)
        expected_body_hash = hash_text(str(section["body_text"]))
        _check_value(section, "body_text_hash", expected_body_hash, label, errors)
        _check_optional_text_slice(
            section,
            page,
            start_field="heading_char_start",
            end_field="heading_char_end",
            expected_text=section.get("heading_text"),
            label=f"{label} heading",
            errors=errors,
        )
        _check_optional_text_slice(
            section,
            page,
            start_field="body_char_start",
            end_field="body_char_end",
            expected_text=section.get("body_text") or None,
            label=f"{label} body",
            errors=errors,
        )
        parent_id = section.get("parent_section_id")
        parent = sections_by_id.get(str(parent_id)) if parent_id is not None else None
        if parent_id is not None and parent is None:
            errors.append(f"{label} references missing parent {parent_id}")
        elif parent is not None and parent["page_id"] != section["page_id"]:
            errors.append(f"{label} parent belongs to another page")
        for ancestor_id in section.get("ancestor_section_ids", []):
            ancestor = sections_by_id.get(str(ancestor_id))
            if ancestor is None:
                errors.append(f"{label} references missing ancestor {ancestor_id}")
            elif ancestor["page_id"] != section["page_id"]:
                errors.append(f"{label} ancestor belongs to another page")
        expected_ancestors: list[str] = []
        cursor = parent
        while cursor is not None:
            cursor_id = str(cursor["section_id"])
            if cursor_id in expected_ancestors:
                errors.append(f"{label} has a cyclic parent chain")
                break
            expected_ancestors.append(cursor_id)
            next_parent = cursor.get("parent_section_id")
            cursor = (
                sections_by_id.get(str(next_parent))
                if next_parent is not None
                else None
            )
        expected_ancestors.reverse()
        _check_value(
            section,
            "ancestor_section_ids",
            expected_ancestors,
            label,
            errors,
        )

    block_indexes_by_page: dict[str, list[int]] = defaultdict(list)
    section_block_indexes_by_section: dict[str, list[int]] = defaultdict(list)
    for block in content_blocks:
        label = f"content block {block.get('block_id')}"
        _check_record_corpus_hash(block, manifest.corpus_hash, label, errors)
        section = sections_by_id.get(str(block["section_id"]))
        page = pages_by_id.get(str(block["page_id"]))
        if section is None or page is None:
            errors.append(f"{label} references a missing page or section")
            continue
        _check_section_ownership(block, section, label, errors)
        _check_record_within_section(block, section, label, errors)
        expected_hash = hash_text(str(block["text"]))
        _check_value(block, "text_hash", expected_hash, label, errors)
        _check_value(
            block,
            "block_id",
            stable_id(
                "content_block",
                section["section_id"],
                block["section_block_index"],
                block["kind"],
                expected_hash,
            ),
            label,
            errors,
        )
        _check_text_slice(block, page, label, errors)
        section_start = int(section["char_start"])
        _check_value(
            block,
            "section_char_start",
            int(block["char_start"]) - section_start,
            label,
            errors,
        )
        _check_value(
            block,
            "section_char_end",
            int(block["char_end"]) - section_start,
            label,
            errors,
        )
        block_indexes_by_page[str(block["page_id"])].append(
            int(block["block_index"]),
        )
        section_block_indexes_by_section[str(block["section_id"])].append(
            int(block["section_block_index"]),
        )

    for page_id in pages_by_id:
        indexes = sorted(block_indexes_by_page.get(page_id, []))
        if indexes != list(range(len(indexes))):
            errors.append(f"page {page_id} content block indexes are not contiguous")
    for section_id in sections_by_id:
        indexes = sorted(section_block_indexes_by_section.get(section_id, []))
        if indexes != list(range(len(indexes))):
            errors.append(
                f"section {section_id} content block indexes are not contiguous",
            )

    for table in tables:
        label = f"table {table.get('table_id')}"
        _check_record_corpus_hash(table, manifest.corpus_hash, label, errors)
        section = sections_by_id.get(str(table["section_id"]))
        page = pages_by_id.get(str(table["page_id"]))
        if section is None or page is None:
            errors.append(f"{label} references a missing page or section")
            continue
        _check_section_ownership(table, section, label, errors)
        _check_record_within_section(table, section, label, errors)
        expected_hash = hash_text(str(table["text"]))
        _check_value(table, "text_hash", expected_hash, label, errors)
        _check_value(
            table,
            "table_id",
            stable_id(
                "table",
                section["section_id"],
                table["table_index"],
                expected_hash,
            ),
            label,
            errors,
        )
        _check_text_slice(table, page, label, errors)

    for image in images:
        label = f"image {image.get('image_id')}"
        _check_record_corpus_hash(image, manifest.corpus_hash, label, errors)
        page = pages_by_id.get(str(image["page_id"]))
        section = sections_by_id.get(str(image["section_id"]))
        if page is None or section is None:
            errors.append(f"{label} references a missing page or section")
            continue
        _check_section_ownership(image, section, label, errors)
        _check_value(
            image,
            "image_id",
            stable_id(
                "image",
                image["section_id"],
                image["image_index"],
                image["media_source_hash"],
            ),
            label,
            errors,
        )

    _verify_routing_tables(
        records["page_table"],
        records["section_table"],
        pages_by_id,
        sections_by_id,
        tables_by_id,
        images_by_id,
        manifest.corpus_hash,
        errors,
    )
    raw_chunking = manifest.chunking or {}
    raw_chunker_manifests = raw_chunking.get("chunkers", [])
    if not isinstance(raw_chunker_manifests, list):
        errors.append("chunker manifest entries must be a list")
        raw_chunker_manifests = []
    chunker_manifests = [
        item for item in raw_chunker_manifests if isinstance(item, dict)
    ]
    if len(chunker_manifests) != len(raw_chunker_manifests):
        errors.append("chunker manifest entries must be objects")
    chunker_manifest_ids = [
        str(item.get("chunker_id", "")) for item in chunker_manifests
    ]
    if len(chunker_manifest_ids) != len(set(chunker_manifest_ids)):
        errors.append("chunker manifest contains duplicate chunker IDs")
    if set(chunker_manifest_ids) != set(chunk_records):
        errors.append("chunker manifest coverage mismatch")
    if raw_chunking.get("config_hash") != manifest.artifact_config_hashes.get(
        "chunking",
    ):
        errors.append("chunking manifest configuration hash mismatch")

    for chunker_id, chunks in chunk_records.items():
        chunker_manifest = next(
            (
                item
                for item in chunker_manifests
                if item.get("chunker_id") == chunker_id
            ),
            None,
        )
        _verify_chunks(
            chunker_id,
            chunks,
            pages_by_id,
            sections_by_id,
            context_groups_by_id,
            content_blocks_by_id,
            chunker_manifest,
            manifest.artifact_hashes,
            manifest.corpus_hash,
            errors,
        )

    actual_corpus_hash = _canonical_corpus_hash(
        corpus_id=manifest.corpus_id,
        parser_config_hash=manifest.parser_config_hash,
        extraction_config_hash=manifest.extraction_config_hash,
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
    )
    if actual_corpus_hash != manifest.corpus_hash:
        errors.append(
            "corpus hash mismatch: "
            f"expected {manifest.corpus_hash}, got {actual_corpus_hash}",
        )


def _verify_section_context_groups(
    manifest: SnapshotManifest,
    *,
    sections: list[JsonDict],
    content_blocks: list[JsonDict],
    groups: list[JsonDict],
    artifact_path: object,
    errors: list[str],
) -> None:
    raw_manifest = manifest.section_context_groups
    if not isinstance(raw_manifest, dict):
        errors.append("section context group manifest metadata is missing")
        return
    raw_config = raw_manifest.get("config")
    if not isinstance(raw_config, dict):
        errors.append("section context group manifest config is missing")
        return
    expected_config_hash = config_hash(raw_config)
    if raw_manifest.get("config_hash") != expected_config_hash:
        errors.append("section context group manifest config hash mismatch")
    if manifest.artifact_config_hashes.get("section_context_groups") != (
        expected_config_hash
    ):
        errors.append("section context group artifact config hash mismatch")
    if raw_manifest.get("record_count") != len(groups):
        errors.append("section context group manifest record count mismatch")
    if artifact_path is None:
        errors.append("section context group artifact path is missing")
        return
    runtime_config = {
        key: value
        for key, value in raw_config.items()
        if key != "policy_version"
    }
    runtime_config["output_path"] = str(artifact_path)
    try:
        expected = [
            record.to_dict()
            for record in build_section_context_groups(
                sections=sections,
                content_blocks=content_blocks,
                config=runtime_config,
                corpus_hash=manifest.corpus_hash,
            )
        ]
    except ValueError as exc:
        errors.append(f"section context groups cannot be recomputed: {exc}")
        return
    if groups != expected:
        errors.append("section context group records mismatch")


def _verify_source_relationship_records(
    manifest: SnapshotManifest,
    *,
    pages: list[JsonDict],
    source_relationship_records: list[JsonDict],
    errors: list[str],
) -> None:
    expected_config_hash = config_hash(normalized_source_relationship_config())
    if manifest.artifact_config_hashes.get("source_relationships") != (
        expected_config_hash
    ):
        errors.append("source relationship configuration hash mismatch")
    expected_records = source_relationship_records_to_dicts(
        build_source_relationships(
            pages,
            corpus_hash=manifest.corpus_hash,
        ),
    )
    if source_relationship_records != expected_records:
        errors.append("source relationship records mismatch")


def _verify_quality_records(
    manifest: SnapshotManifest,
    *,
    pages: list[JsonDict],
    source_relationship_records: list[JsonDict],
    quality_records: list[JsonDict],
    quality_report: JsonDict | None,
    errors: list[str],
) -> None:
    if quality_report is None:
        return
    if quality_report.get("quality_config_hash") != (
        manifest.artifact_config_hashes.get("quality")
    ):
        errors.append("quality configuration hash mismatch")
    base_records, _ = build_page_quality(
        pages,
        quality_config=dict(quality_report.get("rules", {})),
        corpus_hash=manifest.corpus_hash,
    )
    eligible_parent_page_ids = {
        record.page_id for record in base_records if record.eligible_standard_index
    }
    expected_records, expected_report = build_page_quality(
        pages,
        quality_config=dict(quality_report.get("rules", {})),
        corpus_hash=manifest.corpus_hash,
        previous_report=quality_report,
        contained_fragment_page_ids=contained_fragment_page_ids(
            source_relationship_records,
            eligible_parent_page_ids=eligible_parent_page_ids,
        ),
    )
    actual_by_page = _unique_records(quality_records, "page_id", errors)
    expected_by_page = {
        record.page_id: record.to_dict() for record in expected_records
    }
    if set(actual_by_page) != set(expected_by_page):
        errors.append("page quality coverage mismatch")
    for page_id, expected in expected_by_page.items():
        if actual_by_page.get(page_id) != expected:
            errors.append(f"page quality record mismatch for {page_id}")
    for field in (
        "corpus_hash",
        "quality_config_hash",
        "quality_rule_version",
        "rules",
        "page_count",
        "counts",
        "counts_by_language",
        "largest_duplicate_clusters",
    ):
        if quality_report.get(field) != expected_report.get(field):
            errors.append(f"quality report {field} mismatch")


def _check_url_provenance(
    record: JsonDict,
    label: str,
    errors: list[str],
) -> None:
    expected_url = resolve_citation_url(
        canonical_url=(
            str(record["canonical_url"]) if record.get("canonical_url") else None
        ),
        source_url=str(record["source_url"]) if record.get("source_url") else None,
    )
    if record.get("url") != expected_url:
        errors.append(
            f"{label} URL provenance mismatch: expected citation URL "
            f"{expected_url!r}, got {record.get('url')!r}",
        )


def _verify_routing_tables(
    page_table: list[JsonDict],
    section_table: list[JsonDict],
    pages_by_id: dict[str, JsonDict],
    sections_by_id: dict[str, JsonDict],
    tables_by_id: dict[str, JsonDict],
    images_by_id: dict[str, JsonDict],
    expected_corpus_hash: str,
    errors: list[str],
) -> None:
    page_rows = _unique_records(page_table, "page_id", errors)
    section_rows = _unique_records(section_table, "section_id", errors)
    if set(page_rows) != set(pages_by_id):
        errors.append("page table coverage mismatch")
    if set(section_rows) != set(sections_by_id):
        errors.append("section table coverage mismatch")

    section_counts_by_page: dict[str, int] = defaultdict(int)
    table_counts_by_page: dict[str, int] = defaultdict(int)
    image_counts_by_page: dict[str, int] = defaultdict(int)
    table_ids_by_section: dict[str, list[tuple[int, str]]] = defaultdict(list)
    image_ids_by_section: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for section in sections_by_id.values():
        section_counts_by_page[str(section["page_id"])] += 1
    for table_id, table in tables_by_id.items():
        table_counts_by_page[str(table["page_id"])] += 1
        table_ids_by_section[str(table["section_id"])].append(
            (int(table.get("table_index", 0)), table_id),
        )
    for image_id, image in images_by_id.items():
        image_counts_by_page[str(image["page_id"])] += 1
        image_ids_by_section[str(image["section_id"])].append(
            (int(image.get("image_index", 0)), image_id),
        )

    for page_id, row in page_rows.items():
        page = pages_by_id.get(page_id)
        if page is None:
            continue
        label = f"page table row {page_id}"
        _check_record_corpus_hash(row, expected_corpus_hash, label, errors)
        for field in ("page_key", "language", "intent", "text_hash"):
            _check_value(row, field, page[field], label, errors)
        _check_value(
            row,
            "section_count",
            section_counts_by_page.get(page_id, 0),
            label,
            errors,
        )
        _check_value(
            row,
            "table_count",
            table_counts_by_page.get(page_id, 0),
            label,
            errors,
        )
        _check_value(
            row,
            "image_count",
            image_counts_by_page.get(page_id, 0),
            label,
            errors,
        )
    for section_id, row in section_rows.items():
        section = sections_by_id.get(section_id)
        if section is None:
            continue
        label = f"section table row {section_id}"
        _check_record_corpus_hash(row, expected_corpus_hash, label, errors)
        for field in (
            "section_key",
            "page_id",
            "page_key",
            "language",
            "intent",
            "text_hash",
            "char_start",
            "char_end",
        ):
            _check_value(row, field, section[field], label, errors)
        expected_table_ids = [
            table_id
            for _, table_id in sorted(table_ids_by_section.get(section_id, []))
        ]
        expected_image_ids = [
            image_id
            for _, image_id in sorted(image_ids_by_section.get(section_id, []))
        ]
        _check_value(row, "table_ids", expected_table_ids, label, errors)
        _check_value(row, "image_ids", expected_image_ids, label, errors)


def _verify_chunks(
    chunker_id: str,
    chunks: list[JsonDict],
    pages_by_id: dict[str, JsonDict],
    sections_by_id: dict[str, JsonDict],
    context_groups_by_id: dict[str, JsonDict],
    content_blocks_by_id: dict[str, JsonDict],
    chunker_manifest: JsonDict | None,
    artifact_hashes: JsonDict,
    expected_corpus_hash: str,
    errors: list[str],
) -> None:
    _unique_records(chunks, "chunk_id", errors)
    if chunker_manifest is None:
        errors.append(f"chunker {chunker_id} is missing manifest metadata")
        return
    chunk_hashes = artifact_hashes.get("chunks", {})
    expected_artifact_hash = (
        chunk_hashes.get(chunker_id) if isinstance(chunk_hashes, dict) else None
    )
    _check_value(
        chunker_manifest,
        "artifact_hash",
        expected_artifact_hash,
        f"chunker manifest {chunker_id}",
        errors,
    )
    _check_value(
        chunker_manifest,
        "record_count",
        len(chunks),
        f"chunker manifest {chunker_id}",
        errors,
    )
    raw_config = chunker_manifest.get("config")
    if not isinstance(raw_config, dict):
        errors.append(f"chunker {chunker_id} is missing its identity config")
        return
    expected_config_hash = config_hash(raw_config)
    if chunker_manifest.get("config_hash") != expected_config_hash:
        errors.append(f"chunker {chunker_id} config hash mismatch")
    mode = str(chunker_manifest.get("mode"))
    hard_max_chars = int(
        raw_config.get("hard_max_chars", raw_config["target_chars"]),
    )
    expected_source_hashes = {
        "pages": artifact_hashes["pages"],
        "sections": artifact_hashes["sections"],
    }
    if mode == "section_preferred":
        expected_source_hashes.update(
            {
                "content_blocks": artifact_hashes["content_blocks"],
                "section_context_groups": artifact_hashes[
                    "section_context_groups"
                ],
            },
        )
    sections_by_page: dict[str, list[JsonDict]] = defaultdict(list)
    for candidate in sections_by_id.values():
        sections_by_page[str(candidate["page_id"])].append(candidate)
    for candidates in sections_by_page.values():
        candidates.sort(key=lambda candidate: int(candidate["section_index"]))

    blocks_by_page: dict[str, list[tuple[str, JsonDict]]] = defaultdict(list)
    group_ids_by_focus_section: dict[str, set[str]] = defaultdict(set)
    if mode == "section_preferred":
        for block_id, block in content_blocks_by_id.items():
            blocks_by_page[str(block["page_id"])].append((block_id, block))
        for candidates in blocks_by_page.values():
            candidates.sort(key=lambda item: int(item[1]["block_index"]))
        for group_id, group in context_groups_by_id.items():
            for section_id in group.get("focus_section_ids", []):
                group_ids_by_focus_section[str(section_id)].add(group_id)

    chunks_by_page: dict[str, list[JsonDict]] = defaultdict(list)
    for chunk in chunks:
        label = f"chunk {chunk.get('chunk_id')}"
        _check_record_corpus_hash(chunk, expected_corpus_hash, label, errors)
        _check_value(chunk, "chunker_id", chunker_id, label, errors)
        _check_value(chunk, "chunker_mode", mode, label, errors)
        _check_value(
            chunk,
            "chunker_config_hash",
            expected_config_hash,
            label,
            errors,
        )
        if chunk.get("source_artifact_hashes") != expected_source_hashes:
            errors.append(f"{label} source artifact hashes mismatch")
        page = pages_by_id.get(str(chunk["page_id"]))
        section = sections_by_id.get(str(chunk["section_id"]))
        if page is None or section is None:
            errors.append(f"{label} references a missing page or section")
            continue
        expected_hash = hash_text(str(chunk["text"]))
        _check_value(chunk, "text_hash", expected_hash, label, errors)
        if len(str(chunk["text"])) > hard_max_chars:
            errors.append(f"{label} exceeds configured hard maximum")
        _check_value(chunk, "page_key", page["page_key"], label, errors)
        _check_value(chunk, "section_key", section["section_key"], label, errors)
        _check_value(
            chunk,
            "source_page_text_hash",
            page["text_hash"],
            label,
            errors,
        )
        _check_value(
            chunk,
            "source_section_text_hash",
            section["text_hash"],
            label,
            errors,
        )
        _check_value(chunk, "heading_path", section["heading_path"], label, errors)
        _check_value(
            chunk,
            "chunk_id",
            stable_id(
                "chunk",
                page["page_id"],
                chunker_id,
                chunk["chunk_index"],
                chunk["char_start"],
                chunk["char_end"],
                expected_hash,
            ),
            label,
            errors,
        )
        _check_value(
            chunk,
            "equivalence_hint",
            stable_id(
                "chunk_equivalence",
                page["page_id"],
                chunk["char_start"],
                chunk["char_end"],
                expected_hash,
            ),
            label,
            errors,
        )
        _check_text_slice(chunk, page, label, errors)
        chunks_by_page[str(page["page_id"])].append(chunk)
        section_start = chunk.get("section_char_start")
        section_end = chunk.get("section_char_end")
        if section_start is not None and section_end is not None:
            if section["text"][section_start:section_end] != chunk["text"]:
                errors.append(f"{label} section-local text slice mismatch")
        expected_section_ids = [
            str(candidate["section_id"])
            for candidate in sections_by_page.get(str(page["page_id"]), [])
            if max(int(chunk["char_start"]), int(candidate["char_start"]))
            < min(int(chunk["char_end"]), int(candidate["char_end"]))
        ]
        if chunk.get("overlapping_section_ids") != expected_section_ids:
            errors.append(f"{label} overlapping section IDs mismatch")
        if mode == "section_strict" and len(expected_section_ids) != 1:
            errors.append(f"{label} crosses a strict Section boundary")
        if expected_section_ids:
            expected_primary = max(
                (sections_by_id[section_id] for section_id in expected_section_ids),
                key=lambda candidate: (
                    min(int(chunk["char_end"]), int(candidate["char_end"]))
                    - max(int(chunk["char_start"]), int(candidate["char_start"])),
                    -int(candidate["section_index"]),
                ),
            )
            if chunk["section_id"] != expected_primary["section_id"]:
                errors.append(f"{label} primary section mismatch")
        expected_group_ids = (
            sorted(
                group_id
                for section_id in expected_section_ids
                for group_id in group_ids_by_focus_section.get(section_id, set())
            )
            if mode == "section_preferred"
            else []
        )
        expected_group_ids = sorted(set(expected_group_ids))
        if chunk.get("overlapping_context_group_ids") != expected_group_ids:
            errors.append(f"{label} overlapping context group IDs mismatch")
        expected_block_ids = (
            [
                block_id
                for block_id, block in blocks_by_page.get(str(page["page_id"]), [])
                if max(int(chunk["char_start"]), int(block["char_start"]))
                < min(int(chunk["char_end"]), int(block["char_end"]))
            ]
            if mode == "section_preferred"
            else []
        )
        if chunk.get("overlapping_content_block_ids") != expected_block_ids:
            errors.append(f"{label} overlapping content block IDs mismatch")
    for page_id, page_chunks in chunks_by_page.items():
        indexes = sorted(int(chunk["chunk_index"]) for chunk in page_chunks)
        if indexes != list(range(len(page_chunks))):
            errors.append(
                f"chunker {chunker_id} page {page_id} indexes are not contiguous",
            )
    for page_id, page in pages_by_id.items():
        page_chunks = chunks_by_page.get(page_id, [])
        intervals = sorted(
            (int(chunk["char_start"]), int(chunk["char_end"]))
            for chunk in page_chunks
        )
        cursor = 0
        page_text = str(page["text"])
        for start, end in intervals:
            if start > cursor and page_text[cursor:start].strip():
                errors.append(
                    f"chunker {chunker_id} page {page_id} has uncovered source text",
                )
                break
            cursor = max(cursor, end)
        if cursor < len(page_text) and page_text[cursor:].strip():
            errors.append(
                f"chunker {chunker_id} page {page_id} has uncovered source text",
            )


def _unique_records(
    records: list[JsonDict],
    id_field: str,
    errors: list[str],
) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for record in records:
        record_id = str(record.get(id_field, ""))
        if record_id in result:
            errors.append(f"duplicate {id_field}: {record_id}")
        result[record_id] = record
    return result


def _check_record_corpus_hash(
    record: JsonDict,
    expected: str,
    label: str,
    errors: list[str],
) -> None:
    _check_value(record, "corpus_hash", expected, label, errors)


def _check_section_ownership(
    record: JsonDict,
    section: JsonDict,
    label: str,
    errors: list[str],
) -> None:
    if str(record.get("page_id")) != str(section.get("page_id")):
        errors.append(f"{label} belongs to a different page than its section")


def _check_record_within_section(
    record: JsonDict,
    section: JsonDict,
    label: str,
    errors: list[str],
) -> None:
    start = record.get("char_start")
    end = record.get("char_end")
    if not isinstance(start, int) or not isinstance(end, int):
        return
    section_start = section.get("char_start")
    section_end = section.get("char_end")
    if not isinstance(section_start, int) or not isinstance(section_end, int):
        return
    if not (section_start <= start <= end <= section_end):
        errors.append(f"{label} character span is outside its section")


def _check_value(
    record: JsonDict,
    field: str,
    expected: object,
    label: str,
    errors: list[str],
) -> None:
    if record.get(field) != expected:
        errors.append(
            f"{label} {field} mismatch: expected {expected!r}, "
            f"got {record.get(field)!r}",
        )


def _check_text_slice(
    record: JsonDict,
    page: JsonDict,
    label: str,
    errors: list[str],
) -> None:
    start = record.get("char_start")
    end = record.get("char_end")
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"{label} has invalid offsets")
        return
    if start < 0 or end < start or end > len(str(page["text"])):
        errors.append(f"{label} offsets are outside the page text")
        return
    if str(page["text"])[start:end] != record.get("text"):
        errors.append(f"{label} page text slice mismatch")


def _check_optional_text_slice(
    record: JsonDict,
    page: JsonDict,
    *,
    start_field: str,
    end_field: str,
    expected_text: object,
    label: str,
    errors: list[str],
) -> None:
    start = record.get(start_field)
    end = record.get(end_field)
    if start is None and end is None and expected_text is None:
        return
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"{label} has incomplete offsets")
        return
    page_text = str(page["text"])
    if start < 0 or end < start or end > len(page_text):
        errors.append(f"{label} offsets are outside the page text")
        return
    if page_text[start:end] != expected_text:
        errors.append(f"{label} page text slice mismatch")


def _promote_staged_output(*, staging_dir: Path, output_dir: Path) -> None:
    previous_dir: Path | None = None
    if output_dir.exists():
        previous_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}-previous-",
                dir=output_dir.parent,
            ),
        )
        previous_dir.rmdir()
        output_dir.replace(previous_dir)
    try:
        staging_dir.replace(output_dir)
    except Exception:
        if previous_dir is not None and previous_dir.exists():
            previous_dir.replace(output_dir)
        raise
    if previous_dir is not None:
        shutil.rmtree(previous_dir)


def _canonical_corpus_hash(
    *,
    corpus_id: str,
    parser_config_hash: str,
    extraction_config_hash: str,
    pages: list[JsonDict],
    sections: list[JsonDict],
    tables: list[JsonDict],
    images: list[JsonDict],
) -> str:
    # Configuration, sections, tables, and media have separate provenance and
    # artifact hashes. Canonical page records alone define semantic corpus
    # identity; a behavior change only changes this hash when page semantics do.
    _ = parser_config_hash, extraction_config_hash, sections, tables, images
    return corpus_hash(
        {
            "corpus_id": corpus_id,
            "records": {
                "pages": _corpus_identity_records(pages, "page_id"),
            },
        },
    )


def _corpus_identity_records(
    records: list[JsonDict],
    id_field: str,
) -> list[JsonDict]:
    provenance_fields = {
        "corpus_hash",
        "content_source_hash",
        "page_source_hash",
        "source_path",
        "page_source_path",
    }
    return [
        {
            key: value
            for key, value in record.items()
            if key not in provenance_fields
        }
        for record in sorted(records, key=lambda item: str(item[id_field]))
    ]


def _write_json(path: Path, data: JsonDict) -> None:
    validate_record_shape(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{canonical_json(data)}\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotVerificationError(f"Could not read manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotVerificationError(f"Invalid JSON manifest: {path}") from exc
    if not isinstance(loaded, dict):
        raise SnapshotVerificationError(f"Manifest must be a JSON object: {path}")
    return loaded


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except SnapshotVerificationError:
        return None


def _counts_by(values: Iterable[str]) -> JsonDict:
    return dict(sorted(Counter(values).items()))


def _resolve_compression(mode: CompressionMode) -> ResolvedCompression:
    if mode == "auto":
        return "zstd" if shutil.which("zstd") else "gzip"
    if mode in {"zstd", "gzip"}:
        return mode
    raise SnapshotBuildError(f"Unsupported compression mode: {mode!r}")


def _write_archive(
    archive_path: Path,
    source_records: list[SourceFileRecord],
    *,
    repo_root: Path,
    compression: ResolvedCompression,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "raw_source_archive.tar"
        _write_stable_tar(tar_path, source_records, repo_root=repo_root)
        if compression == "zstd":
            _compress_with_zstd(tar_path, archive_path)
        else:
            _compress_with_gzip(tar_path, archive_path)


def _write_stable_tar(
    tar_path: Path,
    source_records: list[SourceFileRecord],
    *,
    repo_root: Path,
) -> None:
    with tar_path.open("wb") as raw_file:
        with tarfile.open(fileobj=raw_file, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for record in sorted(source_records, key=lambda item: item.archive_path):
                source_path = repo_root / record.source_path
                if not source_path.exists():
                    raise SnapshotBuildError(
                        f"Source file is missing during archive build: "
                        f"{record.source_path}",
                    )
                if not _safe_archive_path(record.archive_path):
                    raise SnapshotBuildError(
                        f"Unsafe archive path: {record.archive_path}",
                    )
                info = tarfile.TarInfo(record.archive_path)
                info.size = source_path.stat().st_size
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with source_path.open("rb") as source_file:
                    tar.addfile(info, source_file)


def _compress_with_zstd(tar_path: Path, archive_path: Path) -> None:
    if not shutil.which("zstd"):
        raise SnapshotBuildError(
            "zstd compression requested but the zstd executable is unavailable",
        )
    result = subprocess.run(
        [
            "zstd",
            "-q",
            "--no-progress",
            "-f",
            "-o",
            str(archive_path),
            str(tar_path),
        ],
        check=False,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise SnapshotBuildError(
            f"zstd compression failed: {result.stderr.strip()}",
        )


def _compress_with_gzip(tar_path: Path, archive_path: Path) -> None:
    with tar_path.open("rb") as source, archive_path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=9,
            mtime=0,
        ) as gzip_output:
            shutil.copyfileobj(source, gzip_output)


def _manifest_artifact_path(
    manifest: SnapshotManifest,
    artifact_name: str,
    repo_root: Path,
) -> Path:
    try:
        relative_path = manifest.artifact_paths[artifact_name]
    except KeyError as exc:
        raise SnapshotVerificationError(
            f"Snapshot manifest is missing artifact path for {artifact_name!r}",
        ) from exc
    return _resolve_repo_path(str(relative_path), repo_root)


def _iter_artifact_hashes(
    manifest: SnapshotManifest,
    repo_root: Path,
) -> Iterator[tuple[str, Path, str]]:
    for artifact_name, expected_hash in sorted(manifest.artifact_hashes.items()):
        if isinstance(expected_hash, dict):
            artifact_paths = manifest.artifact_paths.get(artifact_name)
            if not isinstance(artifact_paths, dict):
                raise SnapshotVerificationError(
                    "Snapshot manifest has nested artifact hashes without "
                    f"matching paths for {artifact_name!r}",
                )
            for nested_name, nested_hash in sorted(expected_hash.items()):
                try:
                    nested_path = artifact_paths[nested_name]
                except KeyError as exc:
                    raise SnapshotVerificationError(
                        "Snapshot manifest is missing artifact path for "
                        f"{artifact_name}.{nested_name}",
                    ) from exc
                yield (
                    f"{artifact_name}.{nested_name}",
                    _resolve_repo_path(str(nested_path), repo_root),
                    str(nested_hash),
                )
            continue
        yield (
            artifact_name,
            _manifest_artifact_path(manifest, artifact_name, repo_root),
            str(expected_hash),
        )


def _check_file_hash(
    path: Path,
    expected_hash: str,
    label: str,
    errors: list[str],
) -> None:
    if not path.exists():
        errors.append(f"{label} is missing: {path}")
        return
    actual_hash = hash_file(path)
    if actual_hash != expected_hash:
        errors.append(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}",
        )


def _read_source_records(path: Path, errors: list[str]) -> list[SourceFileRecord]:
    records: list[SourceFileRecord] = []
    if not path.exists():
        return records
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError("source file JSONL record must be an object")
            records.append(SourceFileRecord.from_dict(loaded))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"source file manifest cannot be read: {exc}")
    return records


def _verify_source_tree(
    repo_root: Path,
    source_records: list[SourceFileRecord],
) -> list[str]:
    errors: list[str] = []
    for record in source_records:
        source_path = repo_root / record.source_path
        if not source_path.exists():
            errors.append(f"source file is missing: {record.source_path}")
            continue
        if not source_path.is_file():
            errors.append(f"source path is not a file: {record.source_path}")
            continue
        actual_size = source_path.stat().st_size
        if actual_size != record.byte_size:
            errors.append(
                f"source byte size mismatch for {record.source_path}: "
                f"expected {record.byte_size}, got {actual_size}",
            )
        actual_hash = hash_file(source_path)
        if actual_hash != record.file_hash:
            errors.append(
                f"source hash mismatch for {record.source_path}: "
                f"expected {record.file_hash}, got {actual_hash}",
            )
    return errors


def _verify_archive_contents(
    archive_path: Path,
    source_records: list[SourceFileRecord],
    *,
    compression: str,
) -> list[str]:
    errors: list[str] = []
    expected = {record.archive_path: record for record in source_records}
    seen: set[str] = set()
    try:
        with _open_archive(archive_path, compression) as tar:
            for member in tar:
                if member.isdir():
                    continue
                archive_name = member.name
                if not _safe_archive_path(archive_name):
                    errors.append(f"archive contains unsafe path: {archive_name}")
                    continue
                seen.add(archive_name)
                record = expected.get(archive_name)
                if record is None:
                    errors.append(f"archive contains unexpected file: {archive_name}")
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    errors.append(f"archive member cannot be read: {archive_name}")
                    continue
                data = extracted.read()
                actual_hash = sha256_prefixed(data)
                if actual_hash != record.file_hash:
                    errors.append(
                        f"archive hash mismatch for {archive_name}: "
                        f"expected {record.file_hash}, got {actual_hash}",
                    )
                if len(data) != record.byte_size:
                    errors.append(
                        f"archive byte size mismatch for {archive_name}: "
                        f"expected {record.byte_size}, got {len(data)}",
                    )
    except (OSError, tarfile.TarError, SnapshotVerificationError) as exc:
        errors.append(f"archive cannot be read: {exc}")
        return errors

    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"archive is missing files: {missing}")
    return errors


def _restore_archive(
    archive_path: Path,
    *,
    restore_dir: Path,
    source_records: list[SourceFileRecord],
    compression: str,
) -> int:
    if restore_dir.exists() and any(restore_dir.iterdir()):
        raise SnapshotVerificationError(
            f"restore target is not empty: {restore_dir}",
        )
    expected = {record.archive_path: record for record in source_records}
    restored = 0
    with _open_archive(archive_path, compression) as tar:
        for member in tar:
            if member.isdir():
                continue
            if not _safe_archive_path(member.name):
                raise SnapshotVerificationError(
                    f"archive contains unsafe path: {member.name}",
                )
            if member.name not in expected:
                raise SnapshotVerificationError(
                    f"archive contains unexpected file: {member.name}",
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise SnapshotVerificationError(
                    f"archive member cannot be read: {member.name}",
                )
            target_path = _safe_restore_path(restore_dir, member.name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            data = extracted.read()
            target_path.write_bytes(data)
            record = expected[member.name]
            if sha256_prefixed(data) != record.file_hash:
                raise SnapshotVerificationError(
                    f"restored file hash mismatch: {member.name}",
                )
            restored += 1
    return restored


@contextmanager
def _open_archive(
    archive_path: Path,
    compression: str,
) -> Iterator[tarfile.TarFile]:
    if compression == "zstd":
        if not shutil.which("zstd"):
            raise SnapshotVerificationError(
                "zstd archive verification requires the zstd executable",
            )
        process = subprocess.Popen(
            ["zstd", "-q", "-d", "-c", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
                yield tar
        finally:
            process.stdout.close()
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise SnapshotVerificationError(
                    f"zstd decompression failed: {stderr.decode('utf-8', 'replace').strip()}",
                )
        return
    if compression == "gzip":
        with gzip.open(archive_path, "rb") as raw_archive:
            with tarfile.open(fileobj=raw_archive, mode="r|") as tar:
                yield tar
        return
    raise SnapshotVerificationError(f"Unsupported archive compression: {compression!r}")


def _safe_archive_path(path: str) -> bool:
    archive_path = PurePosixPath(path)
    return (
        bool(path)
        and not archive_path.is_absolute()
        and ".." not in archive_path.parts
        and not path.startswith("/")
    )


def _safe_restore_path(restore_dir: Path, archive_path: str) -> Path:
    if not _safe_archive_path(archive_path):
        raise SnapshotVerificationError(f"Unsafe archive path: {archive_path}")
    resolved_root = restore_dir.resolve()
    target_path = (resolved_root / archive_path).resolve()
    try:
        target_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SnapshotVerificationError(
            f"Archive path escapes restore directory: {archive_path}",
        ) from exc
    return target_path


def _find_repo_root_from_manifest(manifest_path: Path) -> Path:
    current = manifest_path.resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "thesis_rag"
        ).is_dir():
            return candidate
    raise SnapshotVerificationError(
        f"Could not infer repository root from manifest path {manifest_path}",
    )
