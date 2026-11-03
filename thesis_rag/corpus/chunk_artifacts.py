"""Derived chunk artifacts bound to an immutable corpus snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)
from thesis_rag.corpus.chunking import (
    ChunkRecord,
    ChunkingConfig,
    build_chunk_artifacts,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.artifact_bundle import (
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.artifact_io import (
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.schema import (
    CONTENT_BLOCK_SCHEMA_VERSION,
    CHUNK_ARTIFACT_SET_MANIFEST_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


CHUNK_ARTIFACT_SET_CONFIG_VERSION = "chunk_artifact_set_config_v1"
CHUNK_ARTIFACT_SET_BUILDER_VERSION = "chunk_artifact_set_builder_v1"
_SOURCE_ARTIFACT_VERSIONS = {
    "pages": PAGE_SCHEMA_VERSION,
    "sections": SECTION_SCHEMA_VERSION,
    "section_context_groups": SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
    "content_blocks": CONTENT_BLOCK_SCHEMA_VERSION,
}


class ChunkArtifactSetError(ValueError):
    """Raised when derived chunks cannot be built or verified."""


class ChunkArtifactSetConfig(ArtifactModel):
    base_snapshot_manifest_path: NonEmptyStr
    output_manifest_path: NonEmptyStr
    chunkers: list[ChunkingConfig] = Field(min_length=1)
    schema_version: Literal["chunk_artifact_set_config_v1"] = (
        CHUNK_ARTIFACT_SET_CONFIG_VERSION
    )

    @model_validator(mode="after")
    def chunkers_are_unique(self) -> Self:
        identifiers = [item.chunker_id for item in self.chunkers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("derived chunker IDs must be unique")
        return self


class ChunkArtifactSetManifest(ArtifactModel):
    chunk_artifact_set_id: NonEmptyStr
    base_snapshot_manifest_path: NonEmptyStr
    base_snapshot_manifest_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    source_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    source_artifact_hashes: dict[NonEmptyStr, Sha256]
    chunker_configs: dict[NonEmptyStr, JsonDict]
    chunker_config_hashes: dict[NonEmptyStr, Sha256]
    artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    artifact_hashes: dict[NonEmptyStr, Sha256]
    artifact_record_counts: dict[NonEmptyStr, NonNegativeInt]
    build_config_hash: Sha256
    builder_implementation_hash: Sha256
    created_at_utc: NonEmptyStr
    schema_version: Literal["chunk_artifact_set_manifest_v1"] = (
        CHUNK_ARTIFACT_SET_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if set(self.source_artifact_paths) != set(self.source_artifact_hashes):
            raise ValueError(
                "chunk artifact-set source maps must cover identical artifacts",
            )
        names = set(self.chunker_configs)
        if not names or any(
            set(values) != names
            for values in (
                self.chunker_config_hashes,
                self.artifact_paths,
                self.artifact_hashes,
                self.artifact_record_counts,
            )
        ):
            raise ValueError("chunk artifact-set maps must cover identical IDs")
        for chunker_id, raw_config in self.chunker_configs.items():
            if raw_config.get("chunker_id") != chunker_id:
                raise ValueError("chunker config key does not match chunker_id")
            if config_hash(raw_config) != self.chunker_config_hashes[chunker_id]:
                raise ValueError("chunker config hash mismatch")
        expected = stable_id(
            "chunk_artifact_set",
            self.base_snapshot_manifest_hash,
            self.corpus_hash,
            self.source_artifact_hashes,
            self.chunker_config_hashes,
            self.artifact_hashes,
            self.artifact_record_counts,
            self.build_config_hash,
            self.builder_implementation_hash,
        )
        if self.chunk_artifact_set_id != expected:
            raise ValueError("chunk artifact-set identity mismatch")
        return self


@dataclass(frozen=True)
class ChunkArtifactSetSummary:
    chunk_artifact_set_id: str | None
    manifest_path: str
    corpus_hash: str
    chunk_counts: dict[str, int]
    dry_run: bool
    reused: bool


@dataclass(frozen=True)
class ResolvedChunkArtifact:
    chunker_id: str
    chunk_path: Path
    chunk_hash: str
    chunker_config: JsonDict
    chunker_config_hash: str
    record_count: int
    manifest: ChunkArtifactSetManifest
    manifest_path: Path
    manifest_hash: str


def load_chunk_artifact_set_config(path: Path) -> ChunkArtifactSetConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ChunkArtifactSetError(f"could not load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChunkArtifactSetError("chunk artifact-set config must be a mapping")
    try:
        return ChunkArtifactSetConfig.from_dict(payload)
    except ValueError as exc:
        raise ChunkArtifactSetError(f"invalid chunk artifact-set config: {exc}") from exc


def load_chunk_artifact_set_manifest(path: Path) -> ChunkArtifactSetManifest:
    try:
        return ChunkArtifactSetManifest.from_dict(
            read_json_object(path, error_type=ChunkArtifactSetError),
        )
    except ValueError as exc:
        raise ChunkArtifactSetError(f"invalid chunk artifact-set manifest: {exc}") from exc


def _snapshot_source_references(
    snapshot: SnapshotManifest,
) -> tuple[dict[str, str], dict[str, str]]:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in _SOURCE_ARTIFACT_VERSIONS:
        raw_path = snapshot.artifact_paths.get(name)
        raw_hash = snapshot.artifact_hashes.get(name)
        if not isinstance(raw_path, str) or not isinstance(raw_hash, str):
            raise ChunkArtifactSetError(f"snapshot has no {name} artifact")
        paths[name] = raw_path
        hashes[name] = raw_hash
    return paths, hashes


def resolve_chunk_artifact(
    manifest_ref: str,
    chunker_id: str,
    *,
    base_snapshot: SnapshotManifest,
    base_snapshot_path: Path,
    repo_root: Path,
) -> ResolvedChunkArtifact:
    """Resolve one derived chunker against its exact frozen snapshot."""

    manifest_path = resolve_repo_path(manifest_ref, repo_root)
    manifest = load_chunk_artifact_set_manifest(manifest_path)
    if (
        manifest.base_snapshot_manifest_hash != hash_file(base_snapshot_path)
        or manifest.corpus_hash != base_snapshot.corpus_hash
        or manifest.source_snapshot_hash != base_snapshot.source_snapshot_hash
    ):
        raise ChunkArtifactSetError(
            "derived chunks belong to another corpus snapshot",
        )
    snapshot_source_paths, snapshot_source_hashes = _snapshot_source_references(
        base_snapshot,
    )
    if manifest.source_artifact_hashes != snapshot_source_hashes:
        raise ChunkArtifactSetError("derived chunk source hashes changed")
    for name, snapshot_ref in snapshot_source_paths.items():
        manifest_ref = manifest.source_artifact_paths.get(name)
        if manifest_ref is None or (
            resolve_repo_path(manifest_ref, repo_root).resolve()
            != resolve_repo_path(snapshot_ref, repo_root).resolve()
        ):
            raise ChunkArtifactSetError("derived chunk source paths changed")
        source_path = resolve_repo_path(snapshot_ref, repo_root)
        if (
            not source_path.is_file()
            or hash_file(source_path) != snapshot_source_hashes[name]
        ):
            raise ChunkArtifactSetError(f"snapshot {name} hash mismatch")
    try:
        chunk_path = resolve_repo_path(manifest.artifact_paths[chunker_id], repo_root)
        expected_chunk_hash = manifest.artifact_hashes[chunker_id]
        if not chunk_path.is_file() or hash_file(chunk_path) != expected_chunk_hash:
            raise ChunkArtifactSetError(
                f"derived chunk artifact {chunker_id!r} hash mismatch",
            )
        return ResolvedChunkArtifact(
            chunker_id=chunker_id,
            chunk_path=chunk_path,
            chunk_hash=expected_chunk_hash,
            chunker_config=manifest.chunker_configs[chunker_id],
            chunker_config_hash=manifest.chunker_config_hashes[chunker_id],
            record_count=manifest.artifact_record_counts[chunker_id],
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_hash=hash_file(manifest_path),
        )
    except KeyError as exc:
        raise ChunkArtifactSetError(
            f"chunk artifact set has no chunker {chunker_id!r}",
        ) from exc


class ChunkArtifactSetBuilder:
    """Build derived chunk sizes without changing the sealed snapshot."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_chunk_artifact_set_config(
            resolve_repo_path(self.config_path, self.repo_root),
        )

    def build(self, *, dry_run: bool = False) -> ChunkArtifactSetSummary:
        root = self.repo_root
        snapshot_path = resolve_repo_path(
            self.config.base_snapshot_manifest_path,
            root,
        )
        snapshot = SnapshotManifest.from_dict(
            read_json_object(snapshot_path, error_type=ChunkArtifactSetError),
        )
        snapshot_source_paths, snapshot_source_hashes = _snapshot_source_references(
            snapshot,
        )
        source_paths: dict[str, Path] = {}
        source_hashes: dict[str, str] = {}
        for name in _SOURCE_ARTIFACT_VERSIONS:
            raw_path = snapshot_source_paths[name]
            raw_hash = snapshot_source_hashes[name]
            path = resolve_repo_path(raw_path, root)
            if hash_file(path) != raw_hash:
                raise ChunkArtifactSetError(f"snapshot {name} hash mismatch")
            source_paths[name] = path
            source_hashes[name] = raw_hash

        configs = [item.to_dict() for item in self.config.chunkers]
        build_config_hash = config_hash(
            {
                "base_snapshot_manifest_hash": hash_file(snapshot_path),
                "chunkers": [item.to_identity_dict() for item in self.config.chunkers],
            },
        )
        implementation_hash = hash_jsonable(
            {
                "builder": hash_file(Path(__file__)),
                "chunking": hash_file(Path(build_chunk_artifacts.__code__.co_filename)),
                "version": CHUNK_ARTIFACT_SET_BUILDER_VERSION,
            },
        )
        manifest_path = resolve_repo_path(self.config.output_manifest_path, root)
        if manifest_path.is_file():
            manifest = load_chunk_artifact_set_manifest(manifest_path)
            require_compatible(
                (
                    manifest.base_snapshot_manifest_hash,
                    manifest.corpus_hash,
                    manifest.source_artifact_hashes,
                    manifest.build_config_hash,
                    manifest.builder_implementation_hash,
                ),
                (
                    hash_file(snapshot_path),
                    snapshot.corpus_hash,
                    source_hashes,
                    build_config_hash,
                    implementation_hash,
                ),
                label="chunk artifact-set active inputs",
                error_type=ChunkArtifactSetError,
            )
            if any(
                resolve_repo_path(manifest.source_artifact_paths[name], root).resolve()
                != source_paths[name].resolve()
                for name in source_paths
            ):
                raise ChunkArtifactSetError("chunk artifact-set source paths changed")
            artifact_references(
                manifest.artifact_paths,
                manifest.artifact_hashes,
                repo_root=root,
                label="chunk artifact set",
                error_type=ChunkArtifactSetError,
            )
            return ChunkArtifactSetSummary(
                chunk_artifact_set_id=manifest.chunk_artifact_set_id,
                manifest_path=repo_relative_path(manifest_path, root),
                corpus_hash=manifest.corpus_hash,
                chunk_counts=dict(manifest.artifact_record_counts),
                dry_run=dry_run,
                reused=True,
            )

        source_records = {
            name: read_jsonl(
                source_paths[name],
                expected_schema_version=schema_version,
            )
            for name, schema_version in _SOURCE_ARTIFACT_VERSIONS.items()
        }
        chunks = build_chunk_artifacts(
            pages=source_records["pages"],
            sections=source_records["sections"],
            context_groups=source_records["section_context_groups"],
            content_blocks=source_records["content_blocks"],
            chunker_configs=configs,
            corpus_hash=snapshot.corpus_hash,
            repo_root=root,
            source_artifact_hashes=source_hashes,
            write_artifacts=False,
        )
        counts = {name: len(records) for name, records in chunks.items()}
        if dry_run:
            return ChunkArtifactSetSummary(
                chunk_artifact_set_id=None,
                manifest_path=repo_relative_path(manifest_path, root),
                corpus_hash=snapshot.corpus_hash,
                chunk_counts=counts,
                dry_run=True,
                reused=False,
            )

        output_paths = {
            item.chunker_id: resolve_repo_path(item.output_path, root)
            for item in self.config.chunkers
        }
        bundle = ImmutableArtifactBundle(
            manifest_path=manifest_path,
            output_paths=output_paths,
            label="chunk artifact set",
            error_type=ChunkArtifactSetError,
        )
        for name, records in chunks.items():
            bundle.write_jsonl(name, records)
        artifact_hashes = bundle.output_hashes
        chunker_configs = {
            item.chunker_id: item.to_identity_dict() for item in self.config.chunkers
        }
        chunker_config_hashes = {
            item.chunker_id: item.identity_hash for item in self.config.chunkers
        }
        artifact_set_id = stable_id(
            "chunk_artifact_set",
            hash_file(snapshot_path),
            snapshot.corpus_hash,
            source_hashes,
            chunker_config_hashes,
            artifact_hashes,
            counts,
            build_config_hash,
            implementation_hash,
        )
        manifest = ChunkArtifactSetManifest(
            chunk_artifact_set_id=artifact_set_id,
            base_snapshot_manifest_path=repo_relative_path(snapshot_path, root),
            base_snapshot_manifest_hash=hash_file(snapshot_path),
            corpus_hash=snapshot.corpus_hash,
            source_snapshot_hash=snapshot.source_snapshot_hash,
            source_artifact_paths={
                name: repo_relative_path(path, root)
                for name, path in source_paths.items()
            },
            source_artifact_hashes=source_hashes,
            chunker_configs=chunker_configs,
            chunker_config_hashes=chunker_config_hashes,
            artifact_paths={
                name: repo_relative_path(path, root)
                for name, path in output_paths.items()
            },
            artifact_hashes=artifact_hashes,
            artifact_record_counts=counts,
            build_config_hash=build_config_hash,
            builder_implementation_hash=implementation_hash,
            created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        bundle.commit(manifest)
        return ChunkArtifactSetSummary(
            chunk_artifact_set_id=artifact_set_id,
            manifest_path=repo_relative_path(manifest_path, root),
            corpus_hash=snapshot.corpus_hash,
            chunk_counts=counts,
            dry_run=False,
            reused=False,
        )
