"""Shared I/O and integrity helpers for deterministic artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from thesis_rag.artifacts import ArtifactModel
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.schema import (
    JsonDict,
    canonical_json,
    hash_file,
    hash_jsonable,
    hash_text,
    validate_record_shape,
    write_jsonl,
)


ErrorType = type[ValueError]


@contextmanager
def file_lock(
    path: Path,
    *,
    mode: Literal["shared", "exclusive"],
    operation: str,
    conflict_message: str,
    error_type: ErrorType,
) -> Iterator[None]:
    """Acquire one non-blocking advisory artifact lock with owner metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_mode = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), lock_mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            detail = f" ({owner})" if owner else ""
            raise error_type(f"{conflict_message}{detail}") from exc
        if mode == "exclusive":
            handle.seek(0)
            handle.truncate()
            handle.write(
                canonical_json(
                    {
                        "operation": operation,
                        "pid": os.getpid(),
                        "started_at_utc": datetime.now(UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    },
                )
                + "\n",
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            yield
        finally:
            if mode == "exclusive":
                handle.seek(0)
                handle.truncate()
                handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def capture_git_state(repo_root: Path) -> tuple[str | None, bool, str]:
    """Capture the repository revision and exact tracked/untracked worktree state."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if commit.returncode != 0:
        return None, False, hash_text("git-unavailable")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    revision = commit.stdout.strip() if commit.returncode == 0 else None
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return (
        revision or None,
        dirty,
        hash_jsonable(
            {
                "commit": revision or None,
                "status": status.stdout if status.returncode == 0 else "unavailable",
                "diff": diff.stdout if diff.returncode == 0 else "unavailable",
            },
        ),
    )


def resolve_repo_root(
    config_path: Path,
    repo_root: str | Path | None,
) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    for start in (Path.cwd(), config_path.resolve().parent):
        for candidate in (start.resolve(), *start.resolve().parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return Path.cwd().resolve()


def resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def read_json_object(path: Path, *, error_type: ErrorType) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"JSON artifact must contain an object: {path}")
    return value


def write_json_object(
    path: Path,
    record: JsonDict,
    *,
    validate: bool = False,
) -> None:
    if validate:
        validate_record_shape(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(record) + "\n", encoding="utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def write_json_object_atomic(
    path: Path,
    record: JsonDict,
    *,
    validate: bool = False,
) -> None:
    """Validate and durably replace one JSON object artifact."""

    if validate:
        validate_record_shape(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    """Durably replace one UTF-8 text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_bytes_atomic(path: Path, value: bytes) -> None:
    """Durably replace one binary artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def hash_jsonl_records(records: Iterable[JsonDict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update((canonical_json(record) + "\n").encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def write_jsonl_atomic(path: Path, records: Iterable[JsonDict | ArtifactModel]) -> None:
    """Validate and atomically replace one derived JSONL artifact."""

    materialized = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        write_jsonl(temporary, materialized)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_snapshot_artifact_hashes(
    snapshot_manifest: JsonDict,
    paths: dict[str, Path],
    *,
    error_type: ErrorType,
) -> None:
    hashes = snapshot_manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise error_type("Snapshot manifest has invalid artifact_hashes")
    for name, path in sorted(paths.items()):
        expected = hashes.get(name)
        if not isinstance(expected, str):
            raise error_type(f"Snapshot manifest has no hash for {name}")
        actual = hash_file(path)
        if actual != expected:
            raise error_type(
                f"{name} hash mismatch: expected {expected}, got {actual}",
            )


def check_overwrite(
    paths: Iterable[Path],
    *,
    repo_root: Path,
    artifact_label: str,
    error_type: ErrorType,
) -> None:
    existing = [
        repo_relative_path(path, repo_root)
        for path in paths
        if path.exists()
    ]
    if existing:
        raise error_type(
            f"{artifact_label} already exist; pass --force to overwrite: {existing}",
        )


def unique_records_by_id(
    records: Iterable[JsonDict],
    id_field: str,
    label: str,
    *,
    error_type: ErrorType,
    allow_empty: bool = True,
) -> dict[str, JsonDict]:
    indexed: dict[str, JsonDict] = {}
    for record in records:
        record_id = str(record[id_field])
        if record_id in indexed:
            raise error_type(f"Duplicate {label} ID: {record_id}")
        indexed[record_id] = record
    if not indexed and not allow_empty:
        raise error_type(f"{label} must not be empty")
    return indexed


def require_corpus_hash(
    records: Iterable[JsonDict],
    corpus_hash: str,
    label: str,
    *,
    error_type: ErrorType,
) -> None:
    mismatched = [
        str(
            record.get("page_id")
            or record.get("source_relationship_id")
            or index
        )
        for index, record in enumerate(records)
        if record.get("corpus_hash") != corpus_hash
    ]
    if mismatched:
        raise error_type(
            f"{label} records have mismatched corpus_hash: {mismatched[:5]}",
        )


def normalize_url(value: object, *, keep_fragment: bool) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    default_port = (scheme == "http" and parsed.port == 80) or (
        scheme == "https" and parsed.port == 443
    )
    host = hostname if not parsed.port or default_port else f"{hostname}:{parsed.port}"
    if parsed.username or parsed.password:
        userinfo = parsed.username or ""
        if parsed.password:
            userinfo += f":{parsed.password}"
        host = f"{userinfo}@{host}"
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    fragment = parsed.fragment if keep_fragment else ""
    return urlunsplit((scheme, host, path, parsed.query, fragment))


def url_fragment(value: object) -> str | None:
    return urlsplit(str(value or "").strip()).fragment or None
