"""Question-specific I/O and compatibility imports for evaluation callers."""

from __future__ import annotations

from pathlib import Path

import yaml

from thesis_rag.artifact_io import (
    ErrorType as ErrorType,
    capture_git_state as capture_git_state,
    check_overwrite as check_overwrite,
    file_lock as file_lock,
    hash_jsonl_records as hash_jsonl_records,
    normalize_url as normalize_url,
    read_json_object as read_json_object,
    require_corpus_hash as require_corpus_hash,
    resolve_repo_path as resolve_repo_path,
    resolve_repo_root as resolve_repo_root,
    unique_records_by_id as unique_records_by_id,
    url_fragment as url_fragment,
    verify_snapshot_artifact_hashes as verify_snapshot_artifact_hashes,
    write_bytes_atomic as write_bytes_atomic,
    write_json_object as write_json_object,
    write_json_object_atomic as write_json_object_atomic,
    write_jsonl_atomic as write_jsonl_atomic,
    write_text_atomic as write_text_atomic,
)
from thesis_rag.eval.question_config import QuestionConfig


def load_question_config(
    config_path: Path,
    repo_root: Path,
    *,
    error_type: ErrorType,
) -> QuestionConfig:
    resolved = resolve_repo_path(config_path, repo_root)
    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("configuration must be a YAML mapping")
        return QuestionConfig.from_dict(loaded)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise error_type(f"Invalid question config {resolved}: {exc}") from exc


def load_snapshot_created_at_utc(
    config: QuestionConfig,
    repo_root: Path,
    *,
    error_type: ErrorType,
) -> str:
    """Return the frozen snapshot time used for temporal question semantics."""

    manifest = read_json_object(
        resolve_repo_path(config.source_artifacts.snapshot_manifest_path, repo_root),
        error_type=error_type,
    )
    value = manifest.get("created_at_utc")
    if not isinstance(value, str) or not value.strip():
        raise error_type("snapshot manifest requires a non-empty created_at_utc")
    return value
