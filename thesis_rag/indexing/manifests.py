"""Content-pinned manifests and safe directory commits for retrieval indexes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.indexing.catalog import ChunkSourceIdentity, ParentSourceIdentity
from thesis_rag.schema import (
    RETRIEVAL_INDEX_MANIFEST_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    stable_id,
)


IndexKind = Literal["dense", "sparse"]


class IndexManifestError(ValueError):
    """Raised when an index is incomplete, incompatible, or corrupted."""


class EmbeddingModelIdentity(ArtifactModel):
    """Exact local model artifact and effective embedding shape."""

    backend_id: Literal["ollama"]
    model_id: NonEmptyStr
    model_artifact_digest: Sha256
    model_metadata_hash: Sha256
    dimensions: PositiveInt


class IndexArtifactFile(ArtifactModel):
    path: NonEmptyStr
    file_hash: Sha256
    byte_size: NonNegativeInt


class LanguageShardManifest(ArtifactModel):
    language: Literal["de", "en"]
    document_count: NonNegativeInt
    row_ids_hash: Sha256
    metadata: JsonDict = Field(default_factory=dict)


class RetrievalIndexManifest(ArtifactModel):
    """Commit-last manifest for one local Dense or Sparse chunk index."""

    index_id: NonEmptyStr
    index_kind: IndexKind
    component_id: NonEmptyStr
    source: ChunkSourceIdentity | ParentSourceIdentity
    semantic_config_hash: Sha256
    backend_id: NonEmptyStr
    backend_version: NonEmptyStr
    embedding_model: EmbeddingModelIdentity | None
    shards: list[LanguageShardManifest]
    files: list[IndexArtifactFile]
    aggregate_file_hash: Sha256
    build_duration_ms: Annotated[float, Field(ge=0.0)]
    total_byte_size: NonNegativeInt
    created_at_utc: NonEmptyStr
    status: Literal["complete"] = "complete"
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal["retrieval_index_manifest_v2"] = (
        RETRIEVAL_INDEX_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        languages = [shard.language for shard in self.shards]
        if languages != ["de", "en"]:
            raise ValueError("index manifest shards must be ordered as de, en")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("index artifact paths must be sorted and unique")
        if self.index_kind == "dense" and self.embedding_model is None:
            raise ValueError("dense index manifest requires an embedding model")
        if self.index_kind == "sparse" and self.embedding_model is not None:
            raise ValueError("sparse index manifest cannot define an embedding model")
        if self.total_byte_size != sum(item.byte_size for item in self.files):
            raise ValueError("index total byte size does not match artifact files")
        if self.aggregate_file_hash != aggregate_file_hash(self.files):
            raise ValueError("index aggregate file hash is inconsistent")
        expected_id = index_identity(
            index_kind=self.index_kind,
            component_id=self.component_id,
            source=self.source,
            semantic_config_hash=self.semantic_config_hash,
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            embedding_model=self.embedding_model,
            shards=self.shards,
            files=self.files,
        )
        if self.index_id != expected_id:
            raise ValueError("index ID is inconsistent with manifest content")
        return self


def artifact_file(path: Path, *, relative_to: Path) -> IndexArtifactFile:
    relative = _safe_relative(path, relative_to)
    return IndexArtifactFile(
        path=relative.as_posix(),
        file_hash=hash_file(path),
        byte_size=path.stat().st_size,
    )


def aggregate_file_hash(files: list[IndexArtifactFile]) -> str:
    return hash_jsonable([item.to_dict() for item in files])


def index_identity(
    *,
    index_kind: IndexKind,
    component_id: str,
    source: ChunkSourceIdentity,
    semantic_config_hash: str,
    backend_id: str,
    backend_version: str,
    embedding_model: EmbeddingModelIdentity | None,
    shards: list[LanguageShardManifest],
    files: list[IndexArtifactFile],
) -> str:
    return stable_id(
        "retrieval_index",
        {
            "index_kind": index_kind,
            "component_id": component_id,
            "source": source.to_index_identity_dict(),
            "semantic_config_hash": semantic_config_hash,
            "backend_id": backend_id,
            "backend_version": backend_version,
            "embedding_model": (
                embedding_model.to_dict() if embedding_model is not None else None
            ),
            "shards": [shard.to_dict() for shard in shards],
            "files": [item.to_dict() for item in files],
        },
    )


def create_manifest(
    *,
    index_kind: IndexKind,
    component_id: str,
    source: ChunkSourceIdentity,
    semantic_config_hash: str,
    backend_id: str,
    backend_version: str,
    embedding_model: EmbeddingModelIdentity | None,
    shards: list[LanguageShardManifest],
    files: list[IndexArtifactFile],
    build_duration_ms: float,
    metadata: JsonDict | None = None,
) -> RetrievalIndexManifest:
    ordered_files = sorted(files, key=lambda item: item.path)
    aggregate = aggregate_file_hash(ordered_files)
    return RetrievalIndexManifest(
        index_id=index_identity(
            index_kind=index_kind,
            component_id=component_id,
            source=source,
            semantic_config_hash=semantic_config_hash,
            backend_id=backend_id,
            backend_version=backend_version,
            embedding_model=embedding_model,
            shards=shards,
            files=ordered_files,
        ),
        index_kind=index_kind,
        component_id=component_id,
        source=source,
        semantic_config_hash=semantic_config_hash,
        backend_id=backend_id,
        backend_version=backend_version,
        embedding_model=embedding_model,
        shards=shards,
        files=ordered_files,
        aggregate_file_hash=aggregate,
        build_duration_ms=build_duration_ms,
        total_byte_size=sum(item.byte_size for item in ordered_files),
        created_at_utc=datetime.now(UTC).isoformat(),
        metadata=metadata or {},
    )


def write_manifest(path: Path, manifest: RetrievalIndexManifest) -> None:
    path.write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")


def load_and_verify_manifest(
    manifest_path: Path,
    *,
    index_kind: IndexKind,
    component_id: str,
    source: ChunkSourceIdentity,
    semantic_config_hash: str,
    embedding_model: EmbeddingModelIdentity | None,
) -> RetrievalIndexManifest:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest must contain an object")
        manifest = RetrievalIndexManifest.from_dict(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IndexManifestError(
            f"invalid index manifest {manifest_path}: {exc}",
        ) from exc
    expected = {
        "index_kind": (manifest.index_kind, index_kind),
        "component_id": (manifest.component_id, component_id),
        "source": (manifest.source, source),
        "semantic_config_hash": (
            manifest.semantic_config_hash,
            semantic_config_hash,
        ),
        "embedding_model": (manifest.embedding_model, embedding_model),
    }
    mismatches = [name for name, (actual, pinned) in expected.items() if actual != pinned]
    if mismatches:
        raise IndexManifestError(
            "index manifest is incompatible: " + ", ".join(mismatches),
        )
    for item in manifest.files:
        path = _artifact_path(manifest_path.parent, item.path)
        if not path.is_file():
            raise IndexManifestError(f"index artifact is missing: {item.path}")
        if path.stat().st_size != item.byte_size:
            raise IndexManifestError(f"index artifact size mismatch: {item.path}")
        if hash_file(path) != item.file_hash:
            raise IndexManifestError(f"index artifact hash mismatch: {item.path}")
    return manifest


def create_staging_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".staging",
            dir=output_dir.parent,
        ),
    )


def commit_staging_directory(
    staging_dir: Path,
    output_dir: Path,
    *,
    force: bool,
) -> None:
    if not (staging_dir / "index_manifest.json").is_file():
        raise IndexManifestError("staging index has no commit manifest")
    if not output_dir.exists():
        os.replace(staging_dir, output_dir)
        return
    if not force:
        raise IndexManifestError(
            f"index directory already exists: {output_dir}",
        )
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    os.replace(output_dir, backup)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def remove_staging_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def resolve_index_root(value: str, *, repo_root: Path) -> Path:
    """Resolve an index directory below the repository without allowing broad writes."""

    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    try:
        relative = path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise IndexManifestError(
            f"index root escapes repository root: {value}",
        ) from exc
    if not relative.parts:
        raise IndexManifestError("index root cannot be the repository root")
    return path


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise IndexManifestError(f"index artifact escapes its directory: {path}") from exc
    if not relative.parts:
        raise IndexManifestError("index artifact path must name a file")
    return relative


def _artifact_path(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise IndexManifestError(f"unsafe index artifact path: {value}")
    path = (root / raw).resolve()
    _safe_relative(path, root)
    return path
