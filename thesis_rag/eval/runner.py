"""Generic, retrieval-only execution of one registered RAG system."""

from __future__ import annotations

import csv
import inspect
import io
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.query_selection import (
    QUERY_SELECTION_VERSION,
    select_query_cohort,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.interfaces import (
    ContextItem,
    RAG_RUNTIME_IDENTITY_SCHEMA_VERSION,
    RETRIEVAL_RESULT_SCHEMA_VERSION,
    RagRuntimeIdentity,
    RetrievalResult,
    RetrievedItem,
    StageCandidate,
    StageResult,
)
from thesis_rag.registry import (
    ResolvedSystemConfig,
    instantiate_system,
    load_system_config,
)
from thesis_rag.schema import (
    QUESTION_SCHEMA_VERSION,
    RETRIEVAL_QUERY_RECORD_SCHEMA_VERSION,
    RETRIEVAL_RUN_MANIFEST_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    stable_id,
)


RETRIEVAL_RUNNER_VERSION = "retrieval_runner_v2"
RETRIEVAL_WARMUP_VERSION = "system_warmup_v1"

RunStatus = Literal["running", "complete", "incomplete", "interrupted", "failed"]
QueryStatus = Literal["success", "error"]
SystemRole = Literal["canonical_baseline", "ablation", "evaluated_system"]


class RetrievalRunError(ValueError):
    """Raised when a retrieval run cannot be executed or resumed safely."""


class RetrievalRunInterrupted(KeyboardInterrupt):
    """Raised after durable state has been materialized for an interrupted run."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"retrieval run interrupted: {run_id}")
        self.run_id = run_id


class ExecutionProvenance(ArtifactModel):
    """Exact repository and Python sources allowed to contribute to one run."""

    source_hashes: dict[NonEmptyStr, Sha256]
    source_set_hash: Sha256

    @model_validator(mode="after")
    def _validate_source_set(self) -> Self:
        if not self.source_hashes:
            raise ValueError("execution provenance requires source hashes")
        if self.source_set_hash != hash_jsonable(self.source_hashes):
            raise ValueError("execution source-set hash mismatch")
        return self


class RetrievalRunIdentity(ArtifactModel):
    """Immutable inputs that define one retrieval-only run."""

    system_id: NonEmptyStr
    system_config_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    snapshot_manifest_hash: Sha256
    questions_hash: Sha256
    selected_query_ids: list[NonEmptyStr]
    selected_query_ids_hash: Sha256
    selected_queries_hash: Sha256
    selected_splits: list[NonEmptyStr]
    query_selection_version: Literal["stratified_query_prefix_v1"]
    selection_seed: int
    query_limit: PositiveInt | None
    top_k: PositiveInt
    stage_evaluation_depth: PositiveInt | None
    runtime_identity: RagRuntimeIdentity
    execution_provenance: ExecutionProvenance
    runner_version: Literal["retrieval_runner_v2"] = RETRIEVAL_RUNNER_VERSION
    schema_versions: dict[NonEmptyStr, NonEmptyStr]


class RetrievalQueryError(ArtifactModel):
    """Structured, retryable failure for one query attempt."""

    error_type: NonEmptyStr
    message: NonEmptyStr
    retryable: bool = True


class RetrievalQueryRecord(ArtifactModel):
    """Atomic authoritative checkpoint for one selected query."""

    run_id: NonEmptyStr
    query_id: NonEmptyStr
    status: QueryStatus
    attempt: PositiveInt
    adapter_run_id: str | None
    retrieval_result: RetrievalResult | None
    error: RetrievalQueryError | None
    completed_at_utc: NonEmptyStr
    schema_version: Literal["retrieval_query_record_v2"] = (
        RETRIEVAL_QUERY_RECORD_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        if self.status == "success":
            if self.retrieval_result is None or self.error is not None:
                raise ValueError("successful query record requires only a result")
            if self.adapter_run_id is None:
                raise ValueError("successful query record requires adapter_run_id")
            if self.retrieval_result.query_id != self.query_id:
                raise ValueError("query record and retrieval result IDs differ")
            if self.retrieval_result.run_id != self.run_id:
                raise ValueError("query record and retrieval result run IDs differ")
        elif self.retrieval_result is not None or self.error is None:
            raise ValueError("error query record requires only error details")
        return self


class RetrievalQueryRecordCommit(ArtifactModel):
    """Manifest pointer to one immutable, durably published query attempt."""

    attempt: PositiveInt
    path: NonEmptyStr
    record_hash: Sha256


class RetrievalRunManifest(ArtifactModel):
    """Mutable lifecycle state around one immutable retrieval-run identity."""

    run_id: NonEmptyStr
    status: RunStatus
    system_role: SystemRole
    identity: RetrievalRunIdentity
    system_config_path: NonEmptyStr
    questions_path: NonEmptyStr
    snapshot_manifest_path: NonEmptyStr
    output_path: NonEmptyStr
    query_selection: JsonDict
    query_counts: dict[NonEmptyStr, NonNegativeInt]
    query_record_commits: dict[NonEmptyStr, RetrievalQueryRecordCommit]
    output_hashes: dict[NonEmptyStr, Sha256]
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    runtime_environment: JsonDict
    created_at_utc: NonEmptyStr
    updated_at_utc: NonEmptyStr
    last_error: str | None = None
    schema_version: Literal["retrieval_run_manifest_v2"] = (
        RETRIEVAL_RUN_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        expected = stable_id("retrieval_run", self.identity.to_dict())
        if self.run_id != expected:
            raise ValueError("run_id does not match immutable run identity")
        selected_ids = set(self.identity.selected_query_ids)
        if not set(self.query_record_commits).issubset(selected_ids):
            raise ValueError("manifest contains records outside the selected query cohort")
        expected_count_keys = {"selected", "successful", "failed", "pending"}
        if set(self.query_counts) != expected_count_keys:
            raise ValueError("manifest has invalid query count keys")
        if self.query_counts["selected"] != len(self.identity.selected_query_ids):
            raise ValueError("manifest selected count differs from run identity")
        if (
            self.query_counts["successful"]
            + self.query_counts["failed"]
            + self.query_counts["pending"]
            != self.query_counts["selected"]
        ):
            raise ValueError("manifest query counts do not add up")
        if len(self.query_record_commits) != (
            self.query_counts["successful"] + self.query_counts["failed"]
        ):
            raise ValueError("manifest query counts differ from committed records")
        if self.status == "complete" and (
            self.query_counts["successful"] != self.query_counts["selected"]
            or self.query_counts["failed"]
            or self.query_counts["pending"]
        ):
            raise ValueError("complete manifest contains unfinished queries")
        required_hashes = {"config_resolved", "query_record_set"}
        if not required_hashes.issubset(self.output_hashes):
            raise ValueError("manifest is missing required integrity hashes")
        if self.status != "running" and not {
            "retrieval_results",
            "latency",
        }.issubset(self.output_hashes):
            raise ValueError("finalized manifest is missing derived output hashes")
        return self


class RetrievalRunSummary(ArtifactModel):
    """Compact CLI-facing outcome of one runner invocation."""

    run_id: NonEmptyStr
    status: RunStatus
    selected_queries: PositiveInt
    successful_queries: NonNegativeInt
    failed_queries: NonNegativeInt
    reused_queries: NonNegativeInt
    manifest_path: NonEmptyStr
    retrieval_results_path: NonEmptyStr
    latency_path: NonEmptyStr


class RetrievalRunProgress(ArtifactModel):
    """Progress update emitted after one selected query is handled."""

    total: PositiveInt
    completed: NonNegativeInt
    new_successes: NonNegativeInt
    reused_successes: NonNegativeInt
    failures: NonNegativeInt


ProgressCallback = Callable[[RetrievalRunProgress], None]


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _system_role(
    system_id: str,
    runtime_identity: RagRuntimeIdentity,
) -> SystemRole:
    if system_id == "legacy_refreshed_nlu":
        expected: SystemRole = "canonical_baseline"
    elif system_id in {"legacy_gold_intent", "legacy_no_routing"}:
        expected = "ablation"
    else:
        return "evaluated_system"
    if runtime_identity.metadata.get("profile_kind") != expected:
        raise RetrievalRunError(
            f"legacy profile {system_id} did not report role {expected}",
        )
    return expected


def _display_path(path: Path, repo_root: Path) -> str:
    return repo_relative_path(path, repo_root)


def _execution_provenance(
    repo_root: Path,
    resolved: ResolvedSystemConfig,
) -> ExecutionProvenance:
    package_root = Path(__file__).resolve().parents[1]
    resolved_root = repo_root.resolve()
    source_paths: dict[str, Path] = {}

    def add_source(path: Path) -> None:
        resolved_path = path.resolve()
        try:
            label = resolved_path.relative_to(resolved_root).as_posix()
        except ValueError:
            label = f"external:{resolved_path.as_posix()}"
        source_paths[label] = resolved_path

    for path in sorted(package_root.rglob("*.py")):
        add_source(path)
    system_package_root = resolved_root / resolved.class_path.split(".", maxsplit=1)[0]
    if system_package_root.is_dir() and system_package_root.resolve() != package_root:
        for path in sorted(system_package_root.rglob("*.py")):
            add_source(path)
    for dependency_path in (resolved_root / "pyproject.toml", resolved_root / "uv.lock"):
        if dependency_path.is_file():
            add_source(dependency_path)
    system_source = inspect.getsourcefile(resolved.system_class)
    if system_source is None:
        raise RetrievalRunError(
            f"could not resolve source for system class {resolved.class_path}",
        )
    add_source(Path(system_source))
    missing = [label for label, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RetrievalRunError(f"execution source files are missing: {missing}")
    source_hashes = {
        label: hash_file(path)
        for label, path in sorted(source_paths.items())
    }
    return ExecutionProvenance(
        source_hashes=source_hashes,
        source_set_hash=hash_jsonable(source_hashes),
    )


def _atomic_write_text(path: Path, payload: str) -> None:
    """Durably write one file before atomically publishing its final name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_model(path: Path, model: ArtifactModel) -> None:
    _atomic_write_text(path, canonical_json(model.to_dict()) + "\n")


def _read_model(path: Path, model_type: type[ArtifactModel]) -> ArtifactModel:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("record must be a JSON object")
        return model_type.from_dict(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RetrievalRunError(f"invalid retrieval artifact {path}: {exc}") from exc


def _query_record_name(query_id: str, attempt: int) -> str:
    digest = stable_id("query_record", query_id).split(":", maxsplit=1)[1]
    return f"{digest}.attempt-{attempt:06d}.json"


def _validate_ranked_stream(items: Sequence[RetrievedItem], label: str) -> None:
    ranks = [item.rank for item in items]
    expected = list(range(1, len(items) + 1))
    if ranks != expected:
        raise RetrievalRunError(f"{label} ranks must be contiguous from one")
    target_ids = [item.target_id for item in items]
    if len(set(target_ids)) != len(target_ids):
        raise RetrievalRunError(f"{label} target IDs must be unique")
    if any(hash_text(item.text) != item.text_hash for item in items):
        raise RetrievalRunError(f"{label} contains an invalid text hash")


def _validate_context_stream(items: Sequence[ContextItem]) -> None:
    ranks = [item.rank for item in items]
    if ranks != list(range(1, len(items) + 1)):
        raise RetrievalRunError("context_items ranks must be contiguous from one")
    context_ids = [item.context_item_id for item in items]
    if len(context_ids) != len(set(context_ids)):
        raise RetrievalRunError("context_items context IDs must be unique")
    if any(hash_text(item.text) != item.text_hash for item in items):
        raise RetrievalRunError("context_items contains an invalid text hash")


def _validate_result(
    result: RetrievalResult,
    *,
    query: QuestionRecord,
    resolved: ResolvedSystemConfig,
    runtime_identity: RagRuntimeIdentity,
    top_k: int,
    expected_run_id: str | None = None,
) -> None:
    if result.query_id != query.query_id:
        raise RetrievalRunError("retrieval result query ID mismatch")
    if result.system_id != resolved.system_id:
        raise RetrievalRunError("retrieval result system ID mismatch")
    if result.config_hash != resolved.config_hash:
        raise RetrievalRunError("retrieval result config hash mismatch")
    if result.corpus_hash != runtime_identity.corpus_hash:
        raise RetrievalRunError("retrieval result corpus hash mismatch")
    if result.schema_version != RETRIEVAL_RESULT_SCHEMA_VERSION:
        raise RetrievalRunError("retrieval result schema version mismatch")
    if expected_run_id is not None and result.run_id != expected_run_id:
        raise RetrievalRunError("retrieval result runner run ID mismatch")
    if len(result.items) > top_k:
        raise RetrievalRunError("retrieval result exceeds the requested top_k")
    _validate_ranked_stream(result.items, "items")
    _validate_context_stream(result.context_items)
    _validate_ranked_stream(result.direct_context_items, "direct_context_items")


def _canonical_result(
    result: RetrievalResult,
    *,
    query: QuestionRecord,
    resolved: ResolvedSystemConfig,
    runtime_identity: RagRuntimeIdentity,
    run_id: str,
    top_k: int,
    stage_evaluation_depth: int | None,
) -> tuple[RetrievalResult, str]:
    if stage_evaluation_depth is None and result.stage_results:
        raise RetrievalRunError(
            "retrieval result contains stage rankings although stage evaluation is disabled",
        )
    if stage_evaluation_depth is not None:
        if any(stage.stage_id == "final" for stage in result.stage_results):
            raise RetrievalRunError("adapter uses reserved stage ID final")
        if any(
            stage.requested_depth != stage_evaluation_depth
            for stage in result.stage_results
        ):
            raise RetrievalRunError("retrieval result stage depth mismatch")
        final_items = [
            StageCandidate(
                rank=index,
                target_id=item.target_id,
                target_level=item.target_level,
                chunk_id=item.chunk_id,
                chunker_id=item.chunker_id,
                section_id=item.section_id,
                page_id=item.page_id,
                score=item.score,
                score_name=item.score_name,
                metadata={"source": "final_retrieval_items"},
            )
            for index, item in enumerate(
                result.items[:stage_evaluation_depth],
                start=1,
            )
        ]
        final_stage = StageResult(
            stage_id="final",
            stage_label="Final retrieval",
            stage_index=max(
                (stage.stage_index for stage in result.stage_results),
                default=-1,
            )
            + 1,
            variant_id=None,
            input_count=len(result.items),
            output_count=len(result.items),
            requested_depth=stage_evaluation_depth,
            evaluable_up_to_k=(
                stage_evaluation_depth
                if len(final_items) == len(result.items)
                else len(final_items) or None
            ),
            ranking_complete=len(final_items) == len(result.items),
            items=final_items,
            metadata={"origin": "canonical_final_output"},
        )
        result = RetrievalResult.from_dict(
            {
                **result.to_dict(),
                "stage_results": [
                    *[stage.to_dict() for stage in result.stage_results],
                    final_stage.to_dict(),
                ],
            },
        )
    _validate_result(
        result,
        query=query,
        resolved=resolved,
        runtime_identity=runtime_identity,
        top_k=top_k,
    )
    adapter_run_id = result.run_id
    if "evaluation_runner" in result.debug:
        raise RetrievalRunError(
            "retrieval result uses reserved debug key evaluation_runner",
        )
    canonical = RetrievalResult.from_dict(
        {
            **result.to_dict(),
            "run_id": run_id,
            "debug": {
                **result.debug,
                "evaluation_runner": {
                    "adapter_run_id": adapter_run_id,
                    "runner_run_id": run_id,
                    "runner_version": RETRIEVAL_RUNNER_VERSION,
                    "stage_evaluation_depth": stage_evaluation_depth,
                },
            },
        },
    )
    return canonical, adapter_run_id


def _query_record_set_hash(
    selected: Sequence[QuestionRecord],
    records: dict[str, RetrievalQueryRecord],
) -> str:
    return hash_jsonable(
        [
            records[query.query_id].to_dict()
            for query in selected
            if query.query_id in records
        ],
    )


def _query_record_commit_set_hash(
    selected: Sequence[QuestionRecord],
    commits: dict[str, RetrievalQueryRecordCommit],
) -> str:
    return hash_jsonable(
        [
            {
                "query_id": query.query_id,
                "commit": commits[query.query_id].to_dict(),
            }
            for query in selected
            if query.query_id in commits
        ],
    )


class RetrievalRunJob:
    """Execute and checkpoint one system over one deterministic query cohort."""

    def __init__(
        self,
        *,
        system_config_path: str | Path,
        questions_path: str | Path,
        snapshot_manifest_path: str | Path = (
            "data/thesis/corpus/snapshot_manifest.json"
        ),
        splits: Sequence[str] = (),
        selection_seed: int = 14731,
        query_limit: int | None = None,
        top_k: int = 20,
        stage_evaluation_depth: int | None = None,
        output_root: str | Path = "outputs/thesis/runs",
        repo_root: str | Path | None = None,
    ) -> None:
        if top_k <= 0:
            raise RetrievalRunError("top_k must be greater than zero")
        if stage_evaluation_depth is not None and stage_evaluation_depth <= 0:
            raise RetrievalRunError("stage_evaluation_depth must be greater than zero")
        self.system_config_path = Path(system_config_path)
        self.questions_path = Path(questions_path)
        self.snapshot_manifest_path = Path(snapshot_manifest_path)
        self.splits = tuple(splits)
        self.selection_seed = selection_seed
        self.query_limit = query_limit
        self.top_k = top_k
        self.stage_evaluation_depth = stage_evaluation_depth
        self.output_root = Path(output_root)
        self.repo_root = repo_root

    def _prepare_inputs(
        self,
    ) -> tuple[
        Path,
        ResolvedSystemConfig,
        Path,
        SnapshotManifest,
        str,
        Path,
        str,
        tuple[QuestionRecord, ...],
        JsonDict,
    ]:
        root = resolve_repo_root(self.system_config_path, self.repo_root)
        config_path = resolve_repo_path(self.system_config_path, root)
        resolved = load_system_config(config_path)
        snapshot_path = resolve_repo_path(self.snapshot_manifest_path, root)
        snapshot_raw = read_json_object(
            snapshot_path,
            error_type=RetrievalRunError,
        )
        try:
            snapshot = SnapshotManifest.from_dict(snapshot_raw)
        except ValueError as exc:
            raise RetrievalRunError(f"invalid snapshot manifest: {exc}") from exc
        snapshot_hash = hash_file(snapshot_path)
        questions_path = resolve_repo_path(self.questions_path, root)
        try:
            all_questions = tuple(
                QuestionRecord.from_dict(record)
                for record in read_jsonl(
                    questions_path,
                    expected_schema_version=QUESTION_SCHEMA_VERSION,
                )
            )
        except (OSError, ValueError) as exc:
            raise RetrievalRunError(
                f"invalid question artifact {questions_path}: {exc}",
            ) from exc
        if not all_questions:
            raise RetrievalRunError("question artifact must not be empty")
        query_ids = [query.query_id for query in all_questions]
        if len(set(query_ids)) != len(query_ids):
            raise RetrievalRunError("question artifact contains duplicate query IDs")
        selected, diagnostics = select_query_cohort(
            all_questions,
            splits=self.splits,
            limit=self.query_limit,
            seed=self.selection_seed,
        )
        return (
            root,
            resolved,
            snapshot_path,
            snapshot,
            snapshot_hash,
            questions_path,
            hash_file(questions_path),
            selected,
            diagnostics,
        )

    @staticmethod
    def _validate_runtime_identity(
        runtime_identity: RagRuntimeIdentity,
        resolved: ResolvedSystemConfig,
        snapshot: SnapshotManifest,
    ) -> None:
        if runtime_identity.system_id != resolved.system_id:
            raise RetrievalRunError("runtime identity system ID mismatch")
        if runtime_identity.config_hash != resolved.config_hash:
            raise RetrievalRunError("runtime identity config hash mismatch")
        if runtime_identity.corpus_hash != snapshot.corpus_hash:
            raise RetrievalRunError("runtime identity corpus hash mismatch")
        if not runtime_identity.identity_hashes:
            raise RetrievalRunError("runtime identity contains no artifact hashes")

    def _identity(
        self,
        *,
        resolved: ResolvedSystemConfig,
        snapshot: SnapshotManifest,
        snapshot_hash: str,
        questions_hash: str,
        selected: tuple[QuestionRecord, ...],
        diagnostics: JsonDict,
        runtime_identity: RagRuntimeIdentity,
        execution_provenance: ExecutionProvenance,
    ) -> RetrievalRunIdentity:
        selected_ids = [query.query_id for query in selected]
        return RetrievalRunIdentity(
            system_id=resolved.system_id,
            system_config_hash=resolved.config_hash,
            corpus_hash=snapshot.corpus_hash,
            source_snapshot_hash=snapshot.source_snapshot_hash,
            snapshot_manifest_hash=snapshot_hash,
            questions_hash=questions_hash,
            selected_query_ids=selected_ids,
            selected_query_ids_hash=hash_jsonable(selected_ids),
            selected_queries_hash=hash_jsonable(
                [query.to_dict() for query in selected],
            ),
            selected_splits=list(diagnostics["active_splits"]),
            query_selection_version=QUERY_SELECTION_VERSION,
            selection_seed=self.selection_seed,
            query_limit=self.query_limit,
            top_k=self.top_k,
            stage_evaluation_depth=self.stage_evaluation_depth,
            runtime_identity=runtime_identity,
            execution_provenance=execution_provenance,
            schema_versions={
                "question": QUESTION_SCHEMA_VERSION,
                "retrieval_result": RETRIEVAL_RESULT_SCHEMA_VERSION,
                "runtime_identity": RAG_RUNTIME_IDENTITY_SCHEMA_VERSION,
                "run_manifest": RETRIEVAL_RUN_MANIFEST_SCHEMA_VERSION,
                "query_record": RETRIEVAL_QUERY_RECORD_SCHEMA_VERSION,
            },
        )

    def _load_records(
        self,
        records_dir: Path,
        selected: tuple[QuestionRecord, ...],
        run_id: str,
        resolved: ResolvedSystemConfig,
        runtime_identity: RagRuntimeIdentity,
        query_record_commits: dict[str, RetrievalQueryRecordCommit],
        expected_record_set_hash: str,
        *,
        verify_record_set: bool,
    ) -> dict[str, RetrievalQueryRecord]:
        selected_by_id = {query.query_id: query for query in selected}
        unknown_commit_ids = sorted(set(query_record_commits) - set(selected_by_id))
        if unknown_commit_ids:
            raise RetrievalRunError(
                f"manifest contains unexpected query record commits: {unknown_commit_ids}",
            )
        if records_dir.exists():
            allowed_prefixes = {
                stable_id("query_record", query_id).split(":", maxsplit=1)[1]
                for query_id in selected_by_id
            }
            unknown = sorted(
                path.name
                for path in records_dir.glob("*.json")
                if not any(
                    path.name.startswith(f"{prefix}.attempt-")
                    and path.name.removeprefix(f"{prefix}.attempt-")
                    .removesuffix(".json")
                    .isdigit()
                    for prefix in allowed_prefixes
                )
            )
            if unknown:
                raise RetrievalRunError(
                    f"query record directory contains unexpected records: {unknown}",
                )
        records: dict[str, RetrievalQueryRecord] = {}
        for query in selected:
            commit = query_record_commits.get(query.query_id)
            if commit is None:
                continue
            expected_name = _query_record_name(query.query_id, commit.attempt)
            if commit.path != expected_name or Path(commit.path).name != commit.path:
                raise RetrievalRunError(
                    f"query record commit path mismatch for {query.query_id}",
                )
            path = records_dir / commit.path
            if not path.is_file():
                raise RetrievalRunError(f"committed query record is missing: {path}")
            if hash_file(path) != commit.record_hash:
                raise RetrievalRunError(f"query record hash mismatch: {path}")
            record = _read_model(path, RetrievalQueryRecord)
            assert isinstance(record, RetrievalQueryRecord)
            if record.query_id in records:
                raise RetrievalRunError(
                    f"duplicate query record for {record.query_id}",
                )
            if (
                record.query_id != query.query_id
                or record.run_id != run_id
                or record.attempt != commit.attempt
            ):
                raise RetrievalRunError(f"query record identity mismatch: {path}")
            if record.status == "success":
                assert record.retrieval_result is not None
                _validate_result(
                    record.retrieval_result,
                    query=query,
                    resolved=resolved,
                    runtime_identity=runtime_identity,
                    top_k=self.top_k,
                    expected_run_id=run_id,
                )
                runner_debug = record.retrieval_result.debug.get("evaluation_runner")
                if not isinstance(runner_debug, dict):
                    raise RetrievalRunError(
                        f"query record has no runner provenance: {path}",
                    )
                adapter_run_id = str(runner_debug.get("adapter_run_id") or "")
                if adapter_run_id != record.adapter_run_id:
                    raise RetrievalRunError(
                        f"query record adapter identity mismatch: {path}",
                    )
            records[record.query_id] = record
        if (
            verify_record_set
            and _query_record_set_hash(selected, records) != expected_record_set_hash
        ):
            raise RetrievalRunError("query record set hash mismatch")
        return records

    @staticmethod
    def _materialize_results(
        path: Path,
        selected: Sequence[QuestionRecord],
        records: dict[str, RetrievalQueryRecord],
    ) -> None:
        lines: list[str] = []
        for query in selected:
            record = records.get(query.query_id)
            if record is None or record.status != "success":
                continue
            assert record.retrieval_result is not None
            lines.append(canonical_json(record.retrieval_result.to_dict()))
        _atomic_write_text(path, "".join(line + "\n" for line in lines))

    @staticmethod
    def _materialize_latency(
        path: Path,
        selected: Sequence[QuestionRecord],
        records: dict[str, RetrievalQueryRecord],
    ) -> None:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "run_id",
                "query_id",
                "system_id",
                "attempt",
                "stage_index",
                "stage",
                "timing_name",
                "duration_ms",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for query in selected:
            record = records.get(query.query_id)
            if record is None or record.status != "success":
                continue
            assert record.retrieval_result is not None
            result = record.retrieval_result
            common = {
                "run_id": result.run_id,
                "query_id": result.query_id,
                "system_id": result.system_id,
                "attempt": record.attempt,
            }
            for timing_name, duration_ms in sorted(result.timings_ms.items()):
                writer.writerow(
                    {
                        **common,
                        "stage_index": "",
                        "stage": "pipeline",
                        "timing_name": timing_name,
                        "duration_ms": format(duration_ms, ".17g"),
                    },
                )
            for stage_index, trace in enumerate(result.stage_traces):
                for timing_name, duration_ms in sorted(trace.timings_ms.items()):
                    writer.writerow(
                        {
                            **common,
                            "stage_index": stage_index,
                            "stage": trace.stage,
                            "timing_name": timing_name,
                            "duration_ms": format(duration_ms, ".17g"),
                        },
                    )
        _atomic_write_text(path, output.getvalue())

    def _commit_query_record(
        self,
        *,
        records_dir: Path,
        record: RetrievalQueryRecord,
        manifest_path: Path,
        manifest: RetrievalRunManifest,
        selected: Sequence[QuestionRecord],
        records: dict[str, RetrievalQueryRecord],
    ) -> RetrievalRunManifest:
        path = records_dir / _query_record_name(record.query_id, record.attempt)
        _write_model(path, record)
        committed_records = {**records, record.query_id: record}
        record_commits = {
            **manifest.query_record_commits,
            record.query_id: RetrievalQueryRecordCommit(
                attempt=record.attempt,
                path=path.name,
                record_hash=hash_file(path),
            ),
        }
        updated = RetrievalRunManifest.from_dict(
            {
                **manifest.to_dict(),
                "query_counts": self._query_counts(selected, committed_records),
                "query_record_commits": record_commits,
                "output_hashes": {
                    **manifest.output_hashes,
                    "query_record_commit_set": _query_record_commit_set_hash(
                        selected,
                        record_commits,
                    ),
                },
                "updated_at_utc": _now_utc(),
            },
        )
        _write_model(manifest_path, updated)
        records[record.query_id] = record
        return updated

    @staticmethod
    def _query_counts(
        selected: Sequence[QuestionRecord],
        records: dict[str, RetrievalQueryRecord],
    ) -> dict[str, int]:
        successful = sum(record.status == "success" for record in records.values())
        failed = sum(record.status == "error" for record in records.values())
        return {
            "selected": len(selected),
            "successful": successful,
            "failed": failed,
            "pending": len(selected) - successful - failed,
        }

    def run(
        self,
        *,
        resume: bool | Literal["auto"] = False,
        expected_run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RetrievalRunSummary:
        """Execute the run, optionally resuming or reusing the exact run identity.

        ``resume="auto"`` is reserved for higher-level orchestration. It resumes
        an identical partial run, reuses an identical verified completed run,
        and creates a new run when the current immutable identity has not run
        before. The boolean modes retain the stricter CLI behavior.
        """

        (
            root,
            resolved,
            snapshot_path,
            snapshot,
            snapshot_hash,
            questions_path,
            questions_hash,
            selected,
            diagnostics,
        ) = self._prepare_inputs()
        output_root = resolve_repo_path(self.output_root, root)
        system = instantiate_system(resolved, repo_root=root)
        run_dir: Path | None = None
        manifest: RetrievalRunManifest | None = None
        records: dict[str, RetrievalQueryRecord] = {}
        reused_queries = 0
        new_successes = 0
        invocation_failures = 0
        handled_queries = 0
        interrupted = False
        fatal_error: BaseException | None = None
        runtime_identity: RagRuntimeIdentity | None = None
        run_id: str | None = None
        results_path: Path | None = None
        latency_path: Path | None = None
        manifest_path: Path | None = None
        config_path: Path | None = None
        completed_reuse = False

        try:
            system.build_or_load(snapshot)
            runtime_identity = system.runtime_identity()
            self._validate_runtime_identity(runtime_identity, resolved, snapshot)
            info = system.info()
            if info.system_id != resolved.system_id:
                raise RetrievalRunError("system info ID differs from configuration")
            if info.config_hash is not None and info.config_hash != resolved.config_hash:
                raise RetrievalRunError("system info config hash mismatch")
            execution_provenance = _execution_provenance(root, resolved)
            identity = self._identity(
                resolved=resolved,
                snapshot=snapshot,
                snapshot_hash=snapshot_hash,
                questions_hash=questions_hash,
                selected=selected,
                diagnostics=diagnostics,
                runtime_identity=runtime_identity,
                execution_provenance=execution_provenance,
            )
            run_id = stable_id("retrieval_run", identity.to_dict())
            if expected_run_id is not None and run_id != expected_run_id:
                raise RetrievalRunError(
                    "current retrieval identity does not match the suite-pinned "
                    f"run {expected_run_id}; create a new suite identity before "
                    "running changed code or runtime artifacts",
                )
            run_dir = output_root / run_id
            manifest_path = run_dir / "manifest.json"
            results_path = run_dir / "retrieval_results.jsonl"
            latency_path = run_dir / "latency.csv"
            config_path = run_dir / "config_resolved.yaml"
            records_dir = run_dir / "query_records"
            created_at = _now_utc()
            config_payload = yaml.safe_dump(
                resolved.raw_config,
                allow_unicode=True,
                sort_keys=True,
            )

            if run_dir.exists():
                if not resume:
                    raise RetrievalRunError(
                        f"retrieval run already exists; pass --resume: {run_dir}",
                    )
                if not manifest_path.is_file():
                    raise RetrievalRunError(
                        f"existing run has no valid manifest: {run_dir}",
                    )
                loaded = _read_model(manifest_path, RetrievalRunManifest)
                assert isinstance(loaded, RetrievalRunManifest)
                if loaded.identity != identity or loaded.run_id != run_id:
                    raise RetrievalRunError("retrieval run identity changed")
                if not config_path.is_file() or hash_file(config_path) != loaded.output_hashes.get(
                    "config_resolved",
                ):
                    raise RetrievalRunError("resolved system config was modified")
                expected_commit_set_hash = loaded.output_hashes.get(
                    "query_record_commit_set",
                )
                if loaded.status == "running" and expected_commit_set_hash is None:
                    raise RetrievalRunError(
                        "running retrieval manifest has no query record commit set hash",
                    )
                if (
                    expected_commit_set_hash is not None
                    and _query_record_commit_set_hash(
                        selected,
                        loaded.query_record_commits,
                    )
                    != expected_commit_set_hash
                ):
                    raise RetrievalRunError("query record commit set hash mismatch")
                records = self._load_records(
                    records_dir,
                    selected,
                    run_id,
                    resolved,
                    runtime_identity,
                    loaded.query_record_commits,
                    loaded.output_hashes["query_record_set"],
                    verify_record_set=loaded.status != "running",
                )
                reused_queries = sum(
                    record.status == "success" for record in records.values()
                )
                if loaded.status == "complete":
                    if resume != "auto":
                        raise RetrievalRunError(
                            "retrieval run is already complete and immutable: "
                            f"{run_dir}",
                        )
                    for label, path in (
                        ("retrieval_results", results_path),
                        ("latency", latency_path),
                    ):
                        if (
                            not path.is_file()
                            or hash_file(path) != loaded.output_hashes.get(label)
                        ):
                            raise RetrievalRunError(
                                f"completed retrieval run has invalid {label} artifact",
                            )
                    manifest = loaded
                    completed_reuse = True
                else:
                    manifest = RetrievalRunManifest.from_dict(
                        {
                            **loaded.to_dict(),
                            "status": "running",
                            "output_hashes": {
                                **loaded.output_hashes,
                                "query_record_commit_set": (
                                    _query_record_commit_set_hash(
                                        selected,
                                        loaded.query_record_commits,
                                    )
                                ),
                            },
                            "updated_at_utc": _now_utc(),
                            "last_error": None,
                        },
                    )
            else:
                if resume is True:
                    raise RetrievalRunError(
                        "no compatible retrieval run exists to resume; "
                        "start a new run without --resume",
                    )
                run_dir.mkdir(parents=True)
                _atomic_write_text(config_path, config_payload)
                git_commit, git_dirty, git_state_hash = capture_git_state(root)
                manifest = RetrievalRunManifest(
                    run_id=run_id,
                    status="running",
                    system_role=_system_role(resolved.system_id, runtime_identity),
                    identity=identity,
                    system_config_path=_display_path(resolved.path.resolve(), root),
                    questions_path=_display_path(questions_path, root),
                    snapshot_manifest_path=_display_path(snapshot_path, root),
                    output_path=_display_path(run_dir, root),
                    query_selection=diagnostics,
                    query_counts={
                        "selected": len(selected),
                        "successful": 0,
                        "failed": 0,
                        "pending": len(selected),
                    },
                    query_record_commits={},
                    output_hashes={
                        "config_resolved": hash_file(config_path),
                        "query_record_commit_set": _query_record_commit_set_hash(
                            selected,
                            {},
                        ),
                        "query_record_set": _query_record_set_hash(selected, {}),
                    },
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                    git_state_hash=git_state_hash,
                    runtime_environment={
                        "python_version": platform.python_version(),
                        "python_implementation": platform.python_implementation(),
                        "platform": platform.platform(),
                        "executable": sys.executable,
                        **(
                            {"retrieval_warmup": RETRIEVAL_WARMUP_VERSION}
                            if callable(getattr(system, "warm_up", None))
                            else {}
                        ),
                    },
                    created_at_utc=created_at,
                    updated_at_utc=created_at,
                )
            if not completed_reuse:
                _write_model(manifest_path, manifest)

            has_pending_queries = any(
                records.get(query.query_id) is None
                or records[query.query_id].status != "success"
                for query in selected
            )
            warm_up = getattr(system, "warm_up", None)
            if has_pending_queries and callable(warm_up):
                warm_up()

            for query in (() if completed_reuse else selected):
                existing = records.get(query.query_id)
                if existing is not None and existing.status == "success":
                    handled_queries += 1
                    if progress_callback is not None:
                        progress_callback(
                            RetrievalRunProgress(
                                total=len(selected),
                                completed=handled_queries,
                                new_successes=new_successes,
                                reused_successes=reused_queries,
                                failures=invocation_failures,
                            ),
                        )
                    continue
                attempt = existing.attempt + 1 if existing is not None else 1
                try:
                    evaluation_started = time.perf_counter()
                    raw_result = system.retrieve(
                        query,
                        top_k=self.top_k,
                        stage_evaluation_depth=self.stage_evaluation_depth,
                    )
                    evaluation_wall_ms = max(
                        0.0,
                        (time.perf_counter() - evaluation_started) * 1000.0,
                    )
                    raw_result = RetrievalResult.from_dict(
                        {
                            **raw_result.to_dict(),
                            "timings_ms": {
                                **raw_result.timings_ms,
                                "evaluation_wall_total": evaluation_wall_ms,
                            },
                        },
                    )
                    canonical_result, adapter_run_id = _canonical_result(
                        raw_result,
                        query=query,
                        resolved=resolved,
                        runtime_identity=runtime_identity,
                        run_id=run_id,
                        top_k=self.top_k,
                        stage_evaluation_depth=self.stage_evaluation_depth,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except Exception as exc:
                    record = RetrievalQueryRecord(
                        run_id=run_id,
                        query_id=query.query_id,
                        status="error",
                        attempt=attempt,
                        adapter_run_id=None,
                        retrieval_result=None,
                        error=RetrievalQueryError(
                            error_type=type(exc).__name__,
                            message=str(exc)[:4000] or type(exc).__name__,
                        ),
                        completed_at_utc=_now_utc(),
                    )
                    manifest = self._commit_query_record(
                        records_dir=records_dir,
                        record=record,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        selected=selected,
                        records=records,
                    )
                    invocation_failures += 1
                else:
                    record = RetrievalQueryRecord(
                        run_id=run_id,
                        query_id=query.query_id,
                        status="success",
                        attempt=attempt,
                        adapter_run_id=adapter_run_id,
                        retrieval_result=canonical_result,
                        error=None,
                        completed_at_utc=_now_utc(),
                    )
                    manifest = self._commit_query_record(
                        records_dir=records_dir,
                        record=record,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        selected=selected,
                        records=records,
                    )
                    new_successes += 1
                handled_queries += 1
                if progress_callback is not None:
                    progress_callback(
                        RetrievalRunProgress(
                            total=len(selected),
                            completed=handled_queries,
                            new_successes=new_successes,
                            reused_successes=reused_queries,
                            failures=invocation_failures,
                        ),
                    )
        except KeyboardInterrupt:
            interrupted = True
        except BaseException as exc:
            fatal_error = exc
        finally:
            try:
                system.close()
            except BaseException as exc:
                if fatal_error is None:
                    fatal_error = RetrievalRunError(
                        f"could not close retrieval system: {exc}",
                    )

            if (
                manifest is not None
                and manifest_path is not None
                and results_path is not None
                and latency_path is not None
                and run_id is not None
                and run_dir is not None
                and not completed_reuse
            ):
                try:
                    self._materialize_results(results_path, selected, records)
                    self._materialize_latency(latency_path, selected, records)
                    counts = self._query_counts(selected, records)
                    if fatal_error is not None:
                        final_status: RunStatus = "failed"
                        last_error = str(fatal_error)[:4000] or type(fatal_error).__name__
                    elif interrupted:
                        final_status = "interrupted"
                        last_error = "interrupted by user"
                    elif counts["successful"] == len(selected):
                        final_status = "complete"
                        last_error = None
                    else:
                        final_status = "incomplete"
                        last_error = None
                    output_hashes = {
                        **manifest.output_hashes,
                        "retrieval_results": hash_file(results_path),
                        "latency": hash_file(latency_path),
                        "query_record_commit_set": _query_record_commit_set_hash(
                            selected,
                            manifest.query_record_commits,
                        ),
                        "query_record_set": _query_record_set_hash(selected, records),
                    }
                    manifest = RetrievalRunManifest.from_dict(
                        {
                            **manifest.to_dict(),
                            "status": final_status,
                            "query_counts": counts,
                            "output_hashes": output_hashes,
                            "updated_at_utc": _now_utc(),
                            "last_error": last_error,
                        },
                    )
                    _write_model(manifest_path, manifest)
                except BaseException as exc:
                    if fatal_error is None:
                        fatal_error = RetrievalRunError(
                            f"could not finalize retrieval run: {exc}",
                        )

        if fatal_error is not None:
            if isinstance(fatal_error, RetrievalRunError):
                raise fatal_error
            raise RetrievalRunError(str(fatal_error) or type(fatal_error).__name__) from fatal_error
        assert manifest is not None
        assert manifest_path is not None
        assert results_path is not None
        assert latency_path is not None
        assert run_id is not None
        if interrupted:
            raise RetrievalRunInterrupted(run_id)
        return RetrievalRunSummary(
            run_id=run_id,
            status=manifest.status,
            selected_queries=len(selected),
            successful_queries=manifest.query_counts["successful"],
            failed_queries=manifest.query_counts["failed"],
            reused_queries=reused_queries,
            manifest_path=_display_path(manifest_path, root),
            retrieval_results_path=_display_path(results_path, root),
            latency_path=_display_path(latency_path, root),
        )
