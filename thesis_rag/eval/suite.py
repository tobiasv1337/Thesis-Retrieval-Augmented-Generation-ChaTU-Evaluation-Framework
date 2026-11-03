"""Plan and sequentially execute immutable retrieval experiment suites."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import ArtifactReference
from thesis_rag.eval.artifact_io import file_lock, resolve_repo_path
from thesis_rag.eval.runner import (
    RetrievalRunInterrupted,
    RetrievalRunJob,
    RetrievalRunManifest,
    RetrievalRunProgress,
    RetrievalRunSummary,
)
from thesis_rag.eval.suite_config import (
    ComparisonDefinition,
    ExperimentSuiteError,
    ResolvedCorpusInput,
    ResolvedExperiment,
    ResolvedQuestionInput,
    SuiteResolution,
    resolution_payload_from_manifest,
    resolve_experiment_suite,
)
from thesis_rag.registry import instantiate_system, load_system_config
from thesis_rag.schema import (
    COHORT_SUITE_MANIFEST_SCHEMA_VERSION,
    LEGACY_SUITE_MANIFEST_SCHEMA_VERSION,
    SUITE_MANIFEST_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    stable_id,
)


SuiteStatus = Literal["planned", "running", "partial", "complete"]
ExperimentRunStatus = Literal[
    "pending",
    "running",
    "complete",
    "incomplete",
    "interrupted",
    "failed",
]


class ExperimentSuiteInterrupted(KeyboardInterrupt):
    """Raised after both runner and suite state have been safely checkpointed."""

    def __init__(self, suite_run_id: str, run_id: str) -> None:
        super().__init__(f"retrieval suite interrupted: {suite_run_id} / {run_id}")
        self.suite_run_id = suite_run_id
        self.run_id = run_id


class ExperimentRunState(ArtifactModel):
    experiment_id: NonEmptyStr
    status: ExperimentRunStatus
    run_id: NonEmptyStr | None = None
    run_manifest_path: NonEmptyStr | None = None
    run_manifest_hash: Sha256 | None = None
    selected_queries: NonNegativeInt = 0
    successful_queries: NonNegativeInt = 0
    failed_queries: NonNegativeInt = 0
    reused_queries_last_invocation: NonNegativeInt = 0
    invocation_count: NonNegativeInt = 0
    last_error: str | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        referenced = (
            self.run_id,
            self.run_manifest_path,
            self.run_manifest_hash,
        )
        if any(value is not None for value in referenced) and not all(
            value is not None for value in referenced
        ):
            raise ValueError("run identity, manifest path, and hash must be atomic")
        if self.status == "pending" and any(value is not None for value in referenced):
            raise ValueError("pending experiment cannot reference a retrieval run")
        if self.status in {"complete", "incomplete", "interrupted"} and not all(
            value is not None for value in referenced
        ):
            raise ValueError(f"{self.status} experiment requires a retrieval run")
        if self.successful_queries + self.failed_queries > self.selected_queries:
            raise ValueError("experiment query counts exceed selected count")
        if self.status == "complete" and (
            self.successful_queries != self.selected_queries
            or self.failed_queries != 0
        ):
            raise ValueError("complete experiment has unfinished queries")
        return self


class ExperimentSuiteManifest(ArtifactModel):
    suite_run_id: NonEmptyStr
    suite_id: NonEmptyStr
    status: SuiteStatus
    resolution_hash: Sha256
    suite_config_path: NonEmptyStr
    suite_config_hash: Sha256
    question_set: ResolvedQuestionInput
    corpus: ResolvedCorpusInput
    experiments: list[ResolvedExperiment]
    comparisons: list[ComparisonDefinition]
    experiment_runs: list[ExperimentRunState]
    created_at_utc: NonEmptyStr
    updated_at_utc: NonEmptyStr
    last_error: str | None = None
    schema_version: Literal[
        LEGACY_SUITE_MANIFEST_SCHEMA_VERSION,
        SUITE_MANIFEST_SCHEMA_VERSION,
        COHORT_SUITE_MANIFEST_SCHEMA_VERSION,
    ] = SUITE_MANIFEST_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if (
            any(item.evaluation_query_limit is not None for item in self.experiments)
            and self.schema_version != COHORT_SUITE_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("evaluation subcohorts require a v4 suite manifest")
        if (
            self.schema_version == LEGACY_SUITE_MANIFEST_SCHEMA_VERSION
            and self.question_set.seal_kind == "calibration"
        ):
            raise ValueError(
                "legacy suite manifests cannot contain calibration cohorts",
            )
        expected_resolution_hash = hash_jsonable(resolution_payload_from_manifest(self))
        if self.resolution_hash != expected_resolution_hash:
            raise ValueError("suite resolution hash mismatch")
        if self.suite_run_id != stable_id("retrieval_suite", self.resolution_hash):
            raise ValueError("suite_run_id does not match the resolution hash")
        experiment_ids = [experiment.experiment_id for experiment in self.experiments]
        run_ids = [state.experiment_id for state in self.experiment_runs]
        if run_ids != experiment_ids:
            raise ValueError("suite run states must match experiment order exactly")
        statuses = [state.status for state in self.experiment_runs]
        if self.status == "planned" and any(status != "pending" for status in statuses):
            raise ValueError("planned suite may contain only pending experiments")
        if self.status == "complete" and any(status != "complete" for status in statuses):
            raise ValueError("complete suite contains unfinished experiments")
        if self.status == "running" and statuses.count("running") != 1:
            raise ValueError("running suite must have exactly one running experiment")
        if self.status == "partial" and "running" in statuses:
            raise ValueError("partial suite cannot contain a running experiment")
        if self.status == "partial" and (
            all(status == "pending" for status in statuses)
            or all(status == "complete" for status in statuses)
        ):
            raise ValueError("partial suite must contain mixed or failed state")
        return self


@dataclass(frozen=True)
class SuitePlanSummary:
    suite_run_id: str
    status: SuiteStatus
    manifest_path: str
    experiment_count: int
    comparison_count: int
    question_count: int
    planned_retrieval_queries: int
    reused: bool
    manifest: ExperimentSuiteManifest


@dataclass(frozen=True)
class SuiteRetrieveSummary:
    suite_run_id: str
    status: SuiteStatus
    manifest_path: str
    completed_experiments: int
    pending_experiments: int
    reused_completed_runs: int
    manifest: ExperimentSuiteManifest


@dataclass(frozen=True)
class SuiteIndexExperimentSummary:
    experiment_id: str
    system_id: str
    system_config_hash: str
    report: JsonDict


@dataclass(frozen=True)
class SuiteIndexSummary:
    suite_run_id: str
    manifest_path: str
    experiments: tuple[SuiteIndexExperimentSummary, ...]


@dataclass(frozen=True)
class SuiteRetrieveProgress:
    experiment_id: str
    experiment_index: int
    experiment_total: int
    retrieval: RetrievalRunProgress


SuiteProgressCallback = Callable[[SuiteRetrieveProgress], None]


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: Path, payload: str) -> None:
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: ExperimentSuiteManifest) -> None:
    _atomic_write_text(path, canonical_json(manifest.to_dict()) + "\n")


def _read_manifest(path: Path) -> ExperimentSuiteManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("manifest must contain a JSON object")
        return ExperimentSuiteManifest.from_dict(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ExperimentSuiteError(f"invalid suite manifest {path}: {exc}") from exc


def _suite_dir(resolution: SuiteResolution) -> Path:
    first_config = resolution.experiments[0].resolved_system_config_path
    path = resolution.repo_root / first_config
    return path.parent.parent


def _manifest_path(resolution: SuiteResolution) -> Path:
    return _suite_dir(resolution) / "manifest.json"


def _suite_lock_root(resolution: SuiteResolution) -> Path:
    return _suite_dir(resolution).parent


@contextmanager
def exclusive_suite_lock(
    resolution: SuiteResolution,
    *,
    operation: str,
) -> Iterator[None]:
    """Block every other coordinated operation while suite state may change."""

    path = _suite_lock_root(resolution) / f".{resolution.suite_run_id}.lock"
    with file_lock(
        path,
        mode="exclusive",
        operation=operation,
        conflict_message=(
            "another operation is already active for suite "
            f"{resolution.suite_run_id}"
        ),
        error_type=ExperimentSuiteError,
    ):
        yield


@contextmanager
def suite_read_operation_lock(
    resolution: SuiteResolution,
    *,
    operation: str,
    channel: Literal["human-qrel-primary", "judge"],
    channel_mode: Literal["shared", "exclusive"] = "exclusive",
) -> Iterator[None]:
    """Read one immutable pool while serializing writers within a channel.

    Human-QREL work and LLM judging use different channel files, so they may
    overlap.  Their shared suite lock still prevents plan, retrieval, or pool
    mutation until both readers have finished.
    """

    root = _suite_lock_root(resolution)
    suite_path = root / f".{resolution.suite_run_id}.lock"
    channel_path = root / f".{resolution.suite_run_id}.{channel}.lock"
    with file_lock(
        suite_path,
        mode="shared",
        operation=operation,
        conflict_message=(
            "another operation is already active for suite "
            f"{resolution.suite_run_id}"
        ),
        error_type=ExperimentSuiteError,
    ):
        with file_lock(
            channel_path,
            mode=channel_mode,
            operation=operation,
            conflict_message=(
                f"another {channel} operation is already active for suite "
                f"{resolution.suite_run_id}"
            ),
            error_type=ExperimentSuiteError,
        ):
            yield


def _new_manifest(resolution: SuiteResolution) -> ExperimentSuiteManifest:
    now = _now_utc()
    return ExperimentSuiteManifest(
        schema_version=(
            COHORT_SUITE_MANIFEST_SCHEMA_VERSION
            if any(item.evaluation_query_limit is not None for item in resolution.experiments)
            else SUITE_MANIFEST_SCHEMA_VERSION
        ),
        suite_run_id=resolution.suite_run_id,
        suite_id=resolution.config.suite_id,
        status="planned",
        resolution_hash=resolution.resolution_hash,
        suite_config_path=repo_relative_path(
            resolution.config_path,
            resolution.repo_root,
        ),
        suite_config_hash=resolution.suite_config_hash,
        question_set=resolution.question_input,
        corpus=resolution.corpus_input,
        experiments=list(resolution.experiments),
        comparisons=list(resolution.comparisons),
        experiment_runs=[
            ExperimentRunState(experiment_id=experiment.experiment_id, status="pending")
            for experiment in resolution.experiments
        ],
        created_at_utc=now,
        updated_at_utc=now,
    )


def _verify_plan_artifacts(
    resolution: SuiteResolution,
    manifest: ExperimentSuiteManifest,
) -> None:
    if (
        manifest.suite_run_id != resolution.suite_run_id
        or manifest.resolution_hash != resolution.resolution_hash
        or manifest.suite_config_hash != resolution.suite_config_hash
        or manifest.suite_config_path
        != repo_relative_path(resolution.config_path, resolution.repo_root)
        or manifest.question_set != resolution.question_input
        or manifest.corpus != resolution.corpus_input
        or manifest.experiments != list(resolution.experiments)
        or manifest.comparisons != list(resolution.comparisons)
    ):
        raise ExperimentSuiteError(
            "persisted suite plan differs from the currently resolved suite",
        )
    for experiment in manifest.experiments:
        path = resolution.repo_root / experiment.resolved_system_config_path
        if not path.is_file():
            raise ExperimentSuiteError(
                f"resolved config is missing for {experiment.experiment_id}",
            )
        if hash_file(path) != experiment.resolved_system_config_file_hash:
            raise ExperimentSuiteError(
                f"resolved config changed for {experiment.experiment_id}",
            )


class ExperimentSuitePlanJob:
    """Resolve a suite and durably materialize its immutable experiment inputs."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
        reuse_from: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root
        self.reuse_from = Path(reuse_from) if reuse_from is not None else None

    def run(self) -> SuitePlanSummary:
        resolution = resolve_experiment_suite(
            self.config_path,
            repo_root=self.repo_root,
            suite_output_root=self.suite_output_root,
        )
        with exclusive_suite_lock(resolution, operation="plan"):
            return self._run_resolved(resolution)

    def _run_resolved(self, resolution: SuiteResolution) -> SuitePlanSummary:
        manifest_path = _manifest_path(resolution)
        reused = False
        if manifest_path.exists():
            manifest = _read_manifest(manifest_path)
            _verify_plan_artifacts(resolution, manifest)
            reused = True
        else:
            manifest = _new_manifest(resolution)
            if self.reuse_from is not None:
                manifest = _reuse_completed_runs(
                    manifest, resolve_repo_path(self.reuse_from, resolution.repo_root),
                    resolution.repo_root,
                )
            suite_dir = _suite_dir(resolution)
            unexpected = [
                path
                for path in suite_dir.iterdir()
                if path.name != "resolved_system_configs"
            ] if suite_dir.exists() else []
            if unexpected:
                raise ExperimentSuiteError(
                    f"suite output exists without a valid manifest: {suite_dir}",
                )
            for experiment in resolution.experiments:
                path = resolution.repo_root / experiment.resolved_system_config_path
                if path.exists():
                    if (
                        not path.is_file()
                        or hash_file(path)
                        != experiment.resolved_system_config_file_hash
                    ):
                        raise ExperimentSuiteError(
                            "resolved config from an interrupted plan differs from "
                            f"the current resolution: {path}",
                        )
                else:
                    _atomic_write_text(
                        path,
                        resolution.resolved_config_payloads[experiment.experiment_id],
                    )
                if hash_file(path) != experiment.resolved_system_config_file_hash:
                    raise ExperimentSuiteError(
                        f"resolved config persistence failed for {experiment.experiment_id}",
                    )
            _write_manifest(manifest_path, manifest)
        return SuitePlanSummary(
            suite_run_id=manifest.suite_run_id,
            status=manifest.status,
            manifest_path=repo_relative_path(manifest_path, resolution.repo_root),
            experiment_count=len(manifest.experiments),
            comparison_count=len(manifest.comparisons),
            question_count=manifest.question_set.question_count,
            planned_retrieval_queries=sum(
                experiment.selected_query_count for experiment in manifest.experiments
            ),
            reused=reused,
            manifest=manifest,
        )


class ExperimentSuiteIndexJob:
    """Build or verify every planned experiment's indexes sequentially."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(self) -> SuiteIndexSummary:
        resolution = resolve_experiment_suite(
            self.config_path,
            repo_root=self.repo_root,
            suite_output_root=self.suite_output_root,
        )
        with exclusive_suite_lock(resolution, operation="index"):
            return self._run_resolved(resolution)

    def _run_resolved(self, resolution: SuiteResolution) -> SuiteIndexSummary:
        manifest_path = _manifest_path(resolution)
        if not manifest_path.is_file():
            raise ExperimentSuiteError(
                "suite has not been planned; run `thesis-rag eval suite plan` first",
            )
        manifest = _read_manifest(manifest_path)
        _verify_plan_artifacts(resolution, manifest)

        summaries: list[SuiteIndexExperimentSummary] = []
        for experiment in manifest.experiments:
            config_path = resolution.repo_root / experiment.resolved_system_config_path
            resolved = load_system_config(config_path)
            if resolved.config_hash != experiment.resolved_system_config_hash:
                raise ExperimentSuiteError(
                    f"resolved config hash mismatch for {experiment.experiment_id}",
                )
            system = instantiate_system(resolved, repo_root=resolution.repo_root)
            try:
                build_index = getattr(system, "build_index", None)
                if not callable(build_index):
                    raise ExperimentSuiteError(
                        f"system {experiment.system_id!r} does not provide an "
                        "index builder",
                    )
                report = build_index()
            except ExperimentSuiteError:
                raise
            except Exception as exc:
                raise ExperimentSuiteError(
                    f"index build failed for {experiment.experiment_id}: {exc}",
                ) from exc
            finally:
                system.close()
            _validate_index_report(
                report,
                experiment_id=experiment.experiment_id,
                expected_corpus_hash=manifest.corpus.corpus_hash,
            )
            summaries.append(
                SuiteIndexExperimentSummary(
                    experiment_id=experiment.experiment_id,
                    system_id=experiment.system_id,
                    system_config_hash=experiment.resolved_system_config_hash,
                    report=report,
                ),
            )
        return SuiteIndexSummary(
            suite_run_id=manifest.suite_run_id,
            manifest_path=repo_relative_path(manifest_path, resolution.repo_root),
            experiments=tuple(summaries),
        )


def _validate_index_report(
    report: object,
    *,
    experiment_id: str,
    expected_corpus_hash: str,
) -> None:
    if not isinstance(report, dict):
        raise ExperimentSuiteError(
            f"index builder for {experiment_id} returned no report object",
        )
    required_strings = ("corpus_hash", "index_hash")
    if any(
        not isinstance(report.get(key), str) or not report[key]
        for key in required_strings
    ):
        raise ExperimentSuiteError(
            f"index report for {experiment_id} has invalid identity fields",
        )
    if report["corpus_hash"] != expected_corpus_hash:
        raise ExperimentSuiteError(
            f"index report corpus mismatch for {experiment_id}",
        )
    for key in ("indexed_document_count", "unique_item_count"):
        value = report.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ExperimentSuiteError(
                f"index report for {experiment_id} has invalid {key}",
            )


def _updated_manifest(
    manifest: ExperimentSuiteManifest,
    *,
    runs: list[ExperimentRunState],
    status: SuiteStatus,
    last_error: str | None,
) -> ExperimentSuiteManifest:
    return ExperimentSuiteManifest.from_dict(
        {
            **manifest.to_dict(),
            "status": status,
            "experiment_runs": [run.to_dict() for run in runs],
            "updated_at_utc": _now_utc(),
            "last_error": last_error,
        },
    )


def _run_state_from_summary(
    *,
    experiment_id: str,
    previous: ExperimentRunState,
    summary: RetrievalRunSummary,
    root: Path,
) -> ExperimentRunState:
    manifest_path = root / summary.manifest_path
    return ExperimentRunState(
        experiment_id=experiment_id,
        status=summary.status if summary.status in {"complete", "incomplete"} else "failed",
        run_id=summary.run_id,
        run_manifest_path=summary.manifest_path,
        run_manifest_hash=hash_file(manifest_path),
        selected_queries=summary.selected_queries,
        successful_queries=summary.successful_queries,
        failed_queries=summary.failed_queries,
        reused_queries_last_invocation=summary.reused_queries,
        invocation_count=previous.invocation_count + 1,
        last_error=None,
    )


def _validate_retrieval_manifest(
    *,
    suite: ExperimentSuiteManifest,
    experiment: ResolvedExperiment,
    path: Path,
    run_id: str,
    status: str,
) -> RetrievalRunManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("run manifest must contain an object")
        manifest = RetrievalRunManifest.from_dict(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ExperimentSuiteError(
            f"invalid retrieval manifest for {experiment.experiment_id}: {exc}",
        ) from exc
    identity = manifest.identity
    expected = {
        "run_id": run_id,
        "status": status,
        "system_config_hash": experiment.resolved_system_config_hash,
        "corpus_hash": suite.corpus.corpus_hash,
        "snapshot_manifest_hash": suite.corpus.snapshot_manifest_hash,
        "questions_hash": suite.question_set.questions_hash,
        "selected_query_ids_hash": experiment.selected_query_ids_hash,
        "selected_queries_hash": experiment.selected_queries_hash,
        "selected_splits": experiment.splits,
        "selection_seed": experiment.selection_seed,
        "query_limit": experiment.query_limit,
        "top_k": experiment.top_k,
        "stage_evaluation_depth": experiment.stage_evaluation_depth,
    }
    actual: JsonDict = {
        "run_id": manifest.run_id,
        "status": manifest.status,
        "system_config_hash": identity.system_config_hash,
        "corpus_hash": identity.corpus_hash,
        "snapshot_manifest_hash": identity.snapshot_manifest_hash,
        "questions_hash": identity.questions_hash,
        "selected_query_ids_hash": identity.selected_query_ids_hash,
        "selected_queries_hash": identity.selected_queries_hash,
        "selected_splits": identity.selected_splits,
        "selection_seed": identity.selection_seed,
        "query_limit": identity.query_limit,
        "top_k": identity.top_k,
        "stage_evaluation_depth": identity.stage_evaluation_depth,
    }
    if actual != expected:
        raise ExperimentSuiteError(
            f"retrieval manifest identity mismatch for {experiment.experiment_id}",
        )
    return manifest


def _load_pinned_retrieval(suite: ExperimentSuiteManifest, experiment: ResolvedExperiment,
                           state: ExperimentRunState, root: Path):
    if (state.status != "complete" or state.run_id is None
            or state.run_manifest_path is None or state.run_manifest_hash is None):
        raise ExperimentSuiteError(
            f"experiment {state.experiment_id} has no complete pinned run"
        )
    path = resolve_repo_path(state.run_manifest_path, root)
    ArtifactReference("retrieval manifest", path, state.run_manifest_hash).verify(
        error_type=ExperimentSuiteError
    )
    manifest = _validate_retrieval_manifest(
        suite=suite, experiment=experiment, path=path, run_id=state.run_id, status="complete",
    )
    run_dir = resolve_repo_path(manifest.output_path, root)
    for name, filename in (("retrieval_results", "retrieval_results.jsonl"),
                           ("latency", "latency.csv"), ("config_resolved", "config_resolved.yaml")):
        digest = manifest.output_hashes.get(name)
        if digest is None:
            raise ExperimentSuiteError(f"retrieval run has no committed {name}")
        ArtifactReference(name, run_dir / filename, digest).verify(
            error_type=ExperimentSuiteError,
        )
    if (state.selected_queries != manifest.query_counts["selected"]
            or state.successful_queries != manifest.query_counts["successful"]):
        raise ExperimentSuiteError("suite query counts differ from the pinned retrieval run")
    return experiment, manifest, path, run_dir / "retrieval_results.jsonl"


def _reuse_completed_runs(target: ExperimentSuiteManifest, source_path: Path,
                         root: Path) -> ExperimentSuiteManifest:
    """Explicitly attach verified historical runs; never relabel their runtime provenance."""
    if not source_path.is_file():
        raise ExperimentSuiteError(f"reuse source suite does not exist: {source_path}")
    source_hash = hash_file(source_path)
    source = _read_manifest(source_path)
    if source.status == "running":
        raise ExperimentSuiteError("cannot reuse runs from a running suite")
    if source.corpus != target.corpus or source.question_set != target.question_set:
        raise ExperimentSuiteError("reuse source has a different corpus or question set")
    experiments = {item.experiment_id: item for item in source.experiments}
    states = {item.experiment_id: item for item in source.experiment_runs}
    runs = []
    for experiment, pending in zip(target.experiments, target.experiment_runs, strict=True):
        previous = states.get(experiment.experiment_id)
        original = experiments.get(experiment.experiment_id)
        # Explicit adoption may change evaluation, never retrieval or its provenance.
        # Stage policies/depth remain checked because they select saved candidates.
        ignored = {
            "resolved_system_config_path", "evaluation_query_limit", "final_evaluation",
        }
        if (previous is None or previous.status != "complete" or original is None
                or original.model_dump(exclude=ignored)
                != experiment.model_dump(exclude=ignored)):
            runs.append(pending)
            continue
        _load_pinned_retrieval(target, experiment, previous, root)
        runs.append(previous)
    ArtifactReference("reuse source suite", source_path, source_hash).verify(
        error_type=ExperimentSuiteError,
    )
    if not any(run.status == "complete" for run in runs):
        raise ExperimentSuiteError("reuse source contains no matching completed experiments")
    return _updated_manifest(target, runs=runs, last_error=None,
                             status="complete" if all(r.status == "complete" for r in runs) else "partial")


def load_pinned_retrievals(suite: ExperimentSuiteManifest, root: Path):
    """Verify complete suite-bound runs and their materialized retrieval files."""
    experiments = {item.experiment_id: item for item in suite.experiments}
    runs = [_load_pinned_retrieval(suite, experiments[state.experiment_id], state, root)
            for state in suite.experiment_runs]
    if len({run.run_id for _, run, _, _ in runs}) != len(runs):
        raise ExperimentSuiteError("one retrieval run cannot back two experiments")
    return runs


class ExperimentSuiteRetrieveJob:
    """Execute resolved experiments sequentially with identity-safe auto-resume."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        suite_output_root: str | Path = "outputs/thesis/suites",
        retrieval_output_root: str | Path = "outputs/thesis/runs",
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.suite_output_root = Path(suite_output_root)
        self.retrieval_output_root = Path(retrieval_output_root)
        self.repo_root = repo_root

    def run(
        self,
        *,
        progress_callback: SuiteProgressCallback | None = None,
        experiment_ids: list[str] | None = None,
    ) -> SuiteRetrieveSummary:
        resolution = resolve_experiment_suite(
            self.config_path,
            repo_root=self.repo_root,
            suite_output_root=self.suite_output_root,
        )
        selected = set(experiment_ids) if experiment_ids else None
        if any(item.evaluation_query_limit is not None for item in resolution.experiments):
            raise ExperimentSuiteError(
                "evaluation subcohorts reuse completed retrievals; "
                "run suite plan --reuse-from with the original suite manifest"
            )
        known = {experiment.experiment_id for experiment in resolution.experiments}
        if selected is not None and selected - known:
            raise ExperimentSuiteError(f"unknown experiments: {sorted(selected - known)}")
        with exclusive_suite_lock(resolution, operation="retrieve"):
            return self._run_resolved(
                resolution,
                progress_callback=progress_callback,
                selected=selected,
            )

    def _run_resolved(
        self,
        resolution: SuiteResolution,
        *,
        progress_callback: SuiteProgressCallback | None,
        selected: set[str] | None = None,
    ) -> SuiteRetrieveSummary:
        manifest_path = _manifest_path(resolution)
        if not manifest_path.is_file():
            raise ExperimentSuiteError(
                "suite has not been planned; run `thesis-rag eval suite plan` first",
            )
        manifest = _read_manifest(manifest_path)
        _verify_plan_artifacts(resolution, manifest)
        runs = list(manifest.experiment_runs)
        reused_completed_runs = 0

        for index, experiment in enumerate(manifest.experiments):
            if selected is not None and experiment.experiment_id not in selected:
                continue
            previous = runs[index]
            runs[index] = ExperimentRunState(
                experiment_id=experiment.experiment_id,
                status="running",
                run_id=previous.run_id,
                run_manifest_path=previous.run_manifest_path,
                run_manifest_hash=previous.run_manifest_hash,
                selected_queries=previous.selected_queries,
                successful_queries=previous.successful_queries,
                failed_queries=previous.failed_queries,
                reused_queries_last_invocation=0,
                invocation_count=previous.invocation_count,
                last_error=None,
            )
            manifest = _updated_manifest(
                manifest,
                runs=runs,
                status="running",
                last_error=None,
            )
            _write_manifest(manifest_path, manifest)

            def on_progress(update: RetrievalRunProgress) -> None:
                if progress_callback is not None:
                    progress_callback(
                        SuiteRetrieveProgress(
                            experiment_id=experiment.experiment_id,
                            experiment_index=index + 1,
                            experiment_total=len(manifest.experiments),
                            retrieval=update,
                        ),
                    )

            job = RetrievalRunJob(
                system_config_path=experiment.resolved_system_config_path,
                questions_path=manifest.question_set.questions_path,
                snapshot_manifest_path=manifest.corpus.snapshot_manifest_path,
                splits=experiment.splits,
                selection_seed=experiment.selection_seed,
                query_limit=experiment.query_limit,
                top_k=experiment.top_k,
                stage_evaluation_depth=experiment.stage_evaluation_depth,
                output_root=self.retrieval_output_root,
                repo_root=resolution.repo_root,
            )
            try:
                summary = job.run(
                    resume="auto",
                    expected_run_id=previous.run_id,
                    progress_callback=on_progress,
                )
                _validate_retrieval_manifest(
                    suite=manifest,
                    experiment=experiment,
                    path=resolution.repo_root / summary.manifest_path,
                    run_id=summary.run_id,
                    status=summary.status,
                )
            except RetrievalRunInterrupted as exc:
                run_manifest_path = (
                    resolve_repo_path(self.retrieval_output_root, resolution.repo_root)
                    / exc.run_id
                    / "manifest.json"
                )
                run_manifest = RetrievalRunManifest.from_dict(
                    json.loads(run_manifest_path.read_text(encoding="utf-8")),
                )
                runs[index] = ExperimentRunState(
                    experiment_id=experiment.experiment_id,
                    status="interrupted",
                    run_id=exc.run_id,
                    run_manifest_path=repo_relative_path(
                        run_manifest_path,
                        resolution.repo_root,
                    ),
                    run_manifest_hash=hash_file(run_manifest_path),
                    selected_queries=run_manifest.query_counts["selected"],
                    successful_queries=run_manifest.query_counts["successful"],
                    failed_queries=run_manifest.query_counts["failed"],
                    invocation_count=previous.invocation_count + 1,
                    last_error="interrupted by user",
                )
                manifest = _updated_manifest(
                    manifest,
                    runs=runs,
                    status="partial",
                    last_error=f"experiment {experiment.experiment_id} interrupted",
                )
                _write_manifest(manifest_path, manifest)
                raise ExperimentSuiteInterrupted(
                    manifest.suite_run_id,
                    exc.run_id,
                ) from exc
            except Exception as exc:
                runs[index] = ExperimentRunState(
                    experiment_id=experiment.experiment_id,
                    status="failed",
                    run_id=previous.run_id,
                    run_manifest_path=previous.run_manifest_path,
                    run_manifest_hash=previous.run_manifest_hash,
                    selected_queries=previous.selected_queries,
                    successful_queries=previous.successful_queries,
                    failed_queries=previous.failed_queries,
                    invocation_count=previous.invocation_count + 1,
                    last_error=str(exc)[:4000],
                )
                manifest = _updated_manifest(
                    manifest,
                    runs=runs,
                    status="partial",
                    last_error=(
                        f"experiment {experiment.experiment_id} failed: {exc}"
                    )[:4000],
                )
                _write_manifest(manifest_path, manifest)
                raise ExperimentSuiteError(str(exc)) from exc

            state = _run_state_from_summary(
                experiment_id=experiment.experiment_id,
                previous=previous,
                summary=summary,
                root=resolution.repo_root,
            )
            runs[index] = state
            if (
                state.status == "complete"
                and state.reused_queries_last_invocation == state.selected_queries
            ):
                reused_completed_runs += 1
            if state.status != "complete":
                manifest = _updated_manifest(
                    manifest,
                    runs=runs,
                    status="partial",
                    last_error=(
                        f"experiment {experiment.experiment_id} is {state.status}"
                    ),
                )
                _write_manifest(manifest_path, manifest)
                break
            next_status: SuiteStatus = (
                "complete"
                if all(run.status == "complete" for run in runs)
                else "partial"
            )
            manifest = _updated_manifest(
                manifest,
                runs=runs,
                status=next_status,
                last_error=None,
            )
            _write_manifest(manifest_path, manifest)

        completed = sum(run.status == "complete" for run in manifest.experiment_runs)
        return SuiteRetrieveSummary(
            suite_run_id=manifest.suite_run_id,
            status=manifest.status,
            manifest_path=repo_relative_path(manifest_path, resolution.repo_root),
            completed_experiments=completed,
            pending_experiments=len(manifest.experiment_runs) - completed,
            reused_completed_runs=reused_completed_runs,
            manifest=manifest,
        )
