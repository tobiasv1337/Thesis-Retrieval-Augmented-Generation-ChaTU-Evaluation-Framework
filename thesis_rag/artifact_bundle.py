"""Small shared layer for immutable, hash-addressed artifact bundles.

Mutable review logs, judge caches, and execution checkpoints deliberately do not
use this publisher.  A bundle consists of atomically written outputs followed by
one manifest written last as its commit marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from thesis_rag.artifacts import ArtifactModel, JsonDict
from thesis_rag.artifact_io import (
    capture_git_state,
    resolve_repo_path,
    write_json_object_atomic,
    write_jsonl_atomic,
    write_bytes_atomic,
    write_text_atomic,
)
from thesis_rag.schema import hash_file


VerificationKind = Literal["integrity", "compatibility"]
ErrorType = type[ValueError]


class ArtifactVerificationError(ValueError):
    """Base class for explicit artifact verification failures."""


class ArtifactIntegrityError(ArtifactVerificationError):
    """Persisted bytes are missing or differ from their committed digest."""


class ArtifactCompatibilityError(ArtifactVerificationError):
    """Intact bytes do not match the requested code/config/model contract."""


def _failure(
    kind: VerificationKind,
    message: str,
    *,
    error_type: ErrorType | None,
) -> ValueError:
    prefix = f"artifact {kind} failure"
    if error_type is not None:
        return error_type(f"{prefix}: {message}")
    if kind == "integrity":
        return ArtifactIntegrityError(f"{prefix}: {message}")
    return ArtifactCompatibilityError(f"{prefix}: {message}")


def require_compatible(
    actual: object,
    expected: object,
    *,
    label: str,
    error_type: ErrorType | None = None,
) -> None:
    """Fail explicitly when intact state is incompatible with an active contract."""

    if actual != expected:
        raise _failure(
            "compatibility",
            f"{label} mismatch: expected {expected!r}, got {actual!r}",
            error_type=error_type,
        )


@dataclass(frozen=True)
class ArtifactReference:
    """One resolved persisted dependency or output and its committed digest."""

    name: str
    path: Path
    digest: str

    def verify(self, *, error_type: ErrorType | None = None) -> None:
        if not self.path.is_file():
            raise _failure(
                "integrity",
                f"{self.name} is missing: {self.path}",
                error_type=error_type,
            )
        actual = hash_file(self.path)
        if actual != self.digest:
            raise _failure(
                "integrity",
                f"{self.name} hash mismatch: expected {self.digest}, got {actual}",
                error_type=error_type,
            )


def artifact_references(
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    *,
    repo_root: Path,
    label: str,
    error_type: ErrorType | None = None,
) -> tuple[ArtifactReference, ...]:
    """Resolve and recursively verify one exact manifest-owned artifact set."""

    require_compatible(
        set(paths),
        set(hashes),
        label=f"{label} path/hash names",
        error_type=error_type,
    )
    references = tuple(
        ArtifactReference(
            name=name,
            path=resolve_repo_path(paths[name], repo_root),
            digest=hashes[name],
        )
        for name in sorted(paths)
    )
    for reference in references:
        reference.verify(error_type=error_type)
    return references


def hash_dependencies(paths: dict[str, Path]) -> dict[str, str]:
    """Hash one exact, named dependency set after requiring every file."""

    references: list[ArtifactReference] = []
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise ArtifactIntegrityError(
                f"artifact integrity failure: {name} is missing: {path}",
            )
        references.append(ArtifactReference(name, path, hash_file(path)))
    return {reference.name: reference.digest for reference in references}


@dataclass(frozen=True)
class ArtifactProvenance:
    """Repository and implementation provenance captured for a new bundle."""

    git_commit: str | None
    git_dirty: bool
    git_state_hash: str
    code_hashes: dict[str, str]

    @classmethod
    def capture(
        cls,
        repo_root: Path,
        *,
        code_paths: dict[str, Path],
    ) -> ArtifactProvenance:
        git_commit, git_dirty, git_state_hash = capture_git_state(repo_root)
        return cls(
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_state_hash=git_state_hash,
            code_hashes=hash_dependencies(code_paths),
        )


class ImmutableArtifactBundle:
    """Publish an exact set of immutable outputs and commit them with a manifest."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        output_paths: dict[str, Path],
        label: str,
        error_type: ErrorType | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.output_paths = dict(sorted(output_paths.items()))
        self.label = label
        self.error_type = error_type
        self._hashes: dict[str, str] = {}
        if not self.output_paths:
            raise _failure(
                "compatibility",
                f"{label} declares no outputs",
                error_type=error_type,
            )
        if len(set(self.output_paths.values())) != len(self.output_paths):
            raise _failure(
                "compatibility",
                f"{label} maps multiple names to one output path",
                error_type=error_type,
            )
        if manifest_path.is_file():
            raise _failure(
                "compatibility",
                f"{label} is already committed: {manifest_path}",
                error_type=error_type,
            )
        orphaned = [str(path) for path in self.output_paths.values() if path.exists()]
        if orphaned:
            raise _failure(
                "integrity",
                f"{label} has uncommitted outputs: {orphaned}",
                error_type=error_type,
            )

    def _path(self, name: str) -> Path:
        if name not in self.output_paths:
            raise _failure(
                "compatibility",
                f"{self.label} received undeclared output {name!r}",
                error_type=self.error_type,
            )
        if name in self._hashes:
            raise _failure(
                "compatibility",
                f"{self.label} output {name!r} was written twice",
                error_type=self.error_type,
            )
        return self.output_paths[name]

    def write_jsonl(
        self,
        name: str,
        records: list[JsonDict | ArtifactModel] | tuple[JsonDict | ArtifactModel, ...],
    ) -> None:
        path = self._path(name)
        write_jsonl_atomic(path, records)
        self._hashes[name] = hash_file(path)

    def write_json(self, name: str, record: JsonDict | ArtifactModel) -> None:
        path = self._path(name)
        payload = record.to_dict() if isinstance(record, ArtifactModel) else record
        write_json_object_atomic(path, payload, validate=isinstance(record, ArtifactModel))
        self._hashes[name] = hash_file(path)

    def write_text(self, name: str, value: str) -> None:
        path = self._path(name)
        write_text_atomic(path, value)
        self._hashes[name] = hash_file(path)

    def write_bytes(self, name: str, value: bytes) -> None:
        path = self._path(name)
        write_bytes_atomic(path, value)
        self._hashes[name] = hash_file(path)

    @property
    def output_hashes(self) -> dict[str, str]:
        require_compatible(
            set(self._hashes),
            set(self.output_paths),
            label=f"{self.label} published output names",
            error_type=self.error_type,
        )
        return dict(self._hashes)

    def commit(self, manifest: JsonDict | ArtifactModel) -> dict[str, str]:
        """Verify every output and write the manifest last as the commit marker."""

        hashes = self.output_hashes
        artifact_references(
            self.output_paths,
            hashes,
            repo_root=Path("/"),
            label=self.label,
            error_type=self.error_type,
        )
        payload = manifest.to_dict() if isinstance(manifest, ArtifactModel) else manifest
        write_json_object_atomic(
            self.manifest_path,
            payload,
            validate=isinstance(manifest, ArtifactModel),
        )
        return hashes
