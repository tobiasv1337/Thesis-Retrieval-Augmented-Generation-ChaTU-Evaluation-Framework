"""Verified repository-local inputs shared by runtime and offline evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from thesis_rag.artifact_bundle import ArtifactReference
from thesis_rag.artifact_io import read_json_object, resolve_repo_path
from thesis_rag.artifacts import JsonDict
from thesis_rag.schema import hash_file


def jsonl_records(path: Path) -> Iterator[JsonDict]:
    """Stream records, retaining no file-sized intermediate list."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")  # noqa: TRY004 - malformed input artifact
            yield row


class VerifiedRepositoryFiles:
    """Resolve repository-local inputs and record every consumed file digest."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def path(self, value: str | Path, expected: str | None = None) -> Path:
        path = resolve_repo_path(value, self.root).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Evidence inputs must be inside the repository")
        relative = str(path.relative_to(self.root))
        digest = expected or hash_file(path)
        if relative in self.hashes and self.hashes[relative] != digest:
            raise ValueError(f"Conflicting evidence dependency: {relative}")
        if relative not in self.hashes:
            ArtifactReference(relative, path, digest).verify()
            self.hashes[relative] = digest
        return path

    def object(self, value: str | Path, expected: str | None = None) -> JsonDict:
        return read_json_object(self.path(value, expected), error_type=ValueError)

    def recheck(self) -> None:
        for path, digest in self.hashes.items():
            ArtifactReference(path, self.root / path, digest).verify()
