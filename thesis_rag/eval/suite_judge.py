"""Suite-owned orchestration for resumable async relevance judging."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import (
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    hash_jsonl_records,
    resolve_repo_path,
    write_json_object_atomic,
    write_jsonl_atomic,
)
from thesis_rag.eval.relevance_judgments import (
    build_judge_identity,
    judgment_cache_read_roots,
    judgment_cache_root,
)
from thesis_rag.eval.relevance_types import (
    RelevanceError,
    RelevanceProgressCallback,
)
from thesis_rag.eval.relevance_judge import (
    AssuranceMode,
    AsyncProviderFactory,
    AsyncRelevanceJudgeEngine,
    JudgmentEngineResult,
    ViewAssurance,
    default_async_provider_factory,
    read_judgment_view,
)
from thesis_rag.eval.relevance_config import JudgeEndpointProfile
from thesis_rag.eval.suite import suite_read_operation_lock
from thesis_rag.eval.suite_config import (
    ExperimentSuiteError,
    resolve_experiment_suite,
)
from thesis_rag.eval.suite_pool import (
    ExperimentSuitePoolJob,
    SuitePoolManifest,
    require_materialized_suite_pool,
)
from thesis_rag.schema import (
    JUDGE_EXECUTION_SCHEMA_VERSION,
    JUDGMENT_VIEW_MANIFEST_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    hash_text,
    stable_id,
)


class JudgmentViewManifest(ArtifactModel):
    """Deterministic materialization of one assurance-specific judgment view."""

    judgment_view_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    pool_manifest_hash: Sha256
    semantic_judge_id: NonEmptyStr
    requested_assurance: AssuranceMode
    view_assurance: ViewAssurance
    expected_digest: Sha256 | None
    status: Literal["complete", "incomplete"]
    judgments_path: NonEmptyStr
    judgments_hash: Sha256
    candidate_count: int
    judgment_count: int
    pending_count: int
    selected_evidence_ids_hash: Sha256
    schema_version: Literal["judgment_view_manifest_v1"] = (
        JUDGMENT_VIEW_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if (
            self.candidate_count < 0
            or self.judgment_count < 0
            or self.pending_count < 0
        ):
            raise ValueError("judgment view counts must be nonnegative")
        if self.judgment_count + self.pending_count != self.candidate_count:
            raise ValueError("judgment view counts do not cover the candidate pool")
        expected = stable_id(
            "judgment_view",
            self.suite_run_id,
            self.pool_manifest_id,
            self.semantic_judge_id,
            self.requested_assurance,
            (
                self.expected_digest
                if self.requested_assurance == "strict_digest"
                else None
            ),
            self.view_assurance,
            self.judgments_hash,
            self.selected_evidence_ids_hash,
        )
        if self.judgment_view_id != expected:
            raise ValueError("judgment_view_id does not match immutable content")
        return self


class JudgeExecutionRecord(ArtifactModel):
    """Append-only, redacted execution provenance for one judge invocation."""

    execution_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    semantic_judge_id: NonEmptyStr
    requested_assurance: AssuranceMode
    status: Literal["complete", "incomplete", "interrupted", "failed"]
    started_at_utc: NonEmptyStr
    completed_at_utc: NonEmptyStr
    duration_seconds: float
    slot_count: int
    slots: list[JsonDict]
    counts: JsonDict
    priority_selection: JsonDict
    judgment_view_id: str | None
    judgment_view_hash: Sha256 | None
    errors: list[NonEmptyStr]
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    schema_version: Literal["judge_execution_v1"] = JUDGE_EXECUTION_SCHEMA_VERSION

    @model_validator(mode="after")
    def execution_is_consistent(self) -> Self:
        if self.duration_seconds < 0:
            raise ValueError("judge execution duration must be nonnegative")
        if self.slot_count != len(self.slots) or self.slot_count <= 0:
            raise ValueError("judge execution slot count is inconsistent")
        if (self.judgment_view_id is None) != (self.judgment_view_hash is None):
            raise ValueError("judgment view identity and hash must be atomic")
        return self


@dataclass(frozen=True)
class SuiteJudgeSummary:
    suite_run_id: str
    judge_id: str
    semantic_judge_id: str
    status: Literal["complete", "incomplete"]
    requested_assurance: AssuranceMode
    view_assurance: ViewAssurance
    judgment_count: int
    new_judgments: int
    reused_judgments: int
    failed_judgments: int
    pending_judgments: int
    view_manifest_path: str
    view_manifest_hash: str
    execution_log_path: str


class ExperimentSuiteJudgeInterrupted(RuntimeError):
    """Raised only after accepted evidence and interruption provenance are safe."""

    def __init__(self, suite_run_id: str) -> None:
        super().__init__(f"retrieval suite judging interrupted: {suite_run_id}")
        self.suite_run_id = suite_run_id


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _append_execution(path: Path, record: JudgeExecutionRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record.to_dict()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_view_manifest(path: Path) -> JudgmentViewManifest | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest must be an object")
        return JudgmentViewManifest.from_dict(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RelevanceError(f"invalid judgment view manifest {path}: {exc}") from exc


def _redacted_error(
    error: BaseException | str,
    *,
    profiles: dict[str, object],
) -> str:
    value = str(error).strip() or type(error).__name__
    for profile in profiles.values():
        endpoint = getattr(profile, "endpoint", None)
        if isinstance(endpoint, str):
            value = value.replace(endpoint, "<redacted-endpoint>")
        env_name = getattr(profile, "api_key_env", None)
        if isinstance(env_name, str):
            secret = os.environ.get(env_name)
            if secret:
                value = value.replace(secret, "<redacted-secret>")
    return re.sub(r"https?://[^\s,'\"]+", "<redacted-endpoint>", value)


class ExperimentSuiteJudgeJob:
    """Judge an already materialized suite pool with bounded async slots."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        relevance_config_path: str | Path = "configs/thesis/relevance.yaml",
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
        provider_factory: AsyncProviderFactory = default_async_provider_factory,
    ) -> None:
        self.config_path = Path(config_path)
        self.relevance_config_path = Path(relevance_config_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root
        self.provider_factory = provider_factory

    def run(
        self,
        *,
        judge_id: str | None = None,
        assurance: AssuranceMode | None = None,
        judge_slots: list[str] | None = None,
        judgment_limit: int | None = None,
        verify: bool = True,
        progress_callback: RelevanceProgressCallback | None = None,
    ) -> SuiteJudgeSummary:
        try:
            return asyncio.run(
                self.arun(
                    judge_id=judge_id,
                    assurance=assurance,
                    judge_slots=judge_slots,
                    judgment_limit=judgment_limit,
                    verify=verify,
                    progress_callback=progress_callback,
                ),
            )
        except ExperimentSuiteJudgeInterrupted:
            raise
        except KeyboardInterrupt as exc:
            raise ExperimentSuiteJudgeInterrupted("unknown") from exc

    async def arun(
        self,
        *,
        judge_id: str | None = None,
        assurance: AssuranceMode | None = None,
        judge_slots: list[str] | None = None,
        judgment_limit: int | None = None,
        verify: bool = True,
        progress_callback: RelevanceProgressCallback | None = None,
    ) -> SuiteJudgeSummary:
        resolution = resolve_experiment_suite(
            self.config_path,
            repo_root=self.repo_root,
            suite_output_root=self.suite_output_root,
        )
        with suite_read_operation_lock(
            resolution,
            operation="judge",
            channel="judge",
        ):
            pool_job = ExperimentSuitePoolJob(
                self.config_path,
                relevance_config_path=self.relevance_config_path,
                suite_output_root=self.suite_output_root,
                repo_root=self.repo_root,
            )
            resolved = pool_job.prepare_resolved(
                resolution,
                verify=verify,
                progress_callback=progress_callback,
            )
            configured = resolved.prepared.state.config
            selected_judge_id = judge_id or configured.judging.default_judge_id
            definition = configured.judging.judges.get(selected_judge_id)
            if definition is None:
                raise RelevanceError(f"unknown relevance judge: {selected_judge_id!r}")
            if not definition.enabled:
                raise RelevanceError(
                    f"relevance judge {selected_judge_id!r} is disabled",
                )
            requested_assurance: AssuranceMode = (
                assurance or definition.default_assurance
            )
            active_state = replace(
                resolved.prepared.state,
                config=configured.with_judge(selected_judge_id),
            )
            resolved = replace(
                resolved,
                prepared=replace(resolved.prepared, state=active_state),
            )
            pool_manifest, pool_manifest_path = require_materialized_suite_pool(
                resolved,
            )
            state = resolved.prepared.state
            profiles = state.config.judge.endpoint_profiles
            slots = list(judge_slots or state.config.judge.default_slots)
            if not slots:
                raise RelevanceError("at least one judge slot must be selected")

            prompt_path = resolve_repo_path(
                state.config.judge.prompt.path,
                state.repo_root,
            )
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            identity = build_judge_identity(state, prompt)
            view_key = stable_id(
                "judgment_view_path",
                identity["judge_id"],
                requested_assurance,
                (
                    state.config.judge.model.expected_digest
                    if requested_assurance == "strict_digest"
                    else None
                ),
            ).removeprefix("sha256:")
            suite_dir = resolved.manifest_path.parent
            judge_root = suite_dir / "judge"
            checkpoint_dir = judge_root / "checkpoints" / view_key
            checkpoint_judgments_path = checkpoint_dir / "judgments.jsonl"
            checkpoint_manifest_path = checkpoint_dir / "manifest.json"
            previous_manifest = _read_view_manifest(checkpoint_manifest_path)
            previous_view = read_judgment_view(checkpoint_judgments_path)
            if previous_manifest is not None:
                evidence_ids_hash = hash_jsonable(
                    [
                        str(record.provenance["deployment_evidence_id"])
                        for record in previous_view
                    ],
                )
                view_assurance: ViewAssurance = (
                    "declared_compatible_mixed"
                    if any(
                        record.provenance.get("assurance_class")
                        != "strict_digest"
                        for record in previous_view
                    )
                    else "strict_digest"
                )
                artifact_references(
                    {"judgments": previous_manifest.judgments_path},
                    {"judgments": previous_manifest.judgments_hash},
                    repo_root=state.repo_root,
                    label="judge checkpoint",
                    error_type=RelevanceError,
                )
                require_compatible(
                    (
                        previous_manifest.suite_run_id,
                        previous_manifest.pool_manifest_id,
                        previous_manifest.semantic_judge_id,
                        previous_manifest.requested_assurance,
                        previous_manifest.pool_manifest_hash,
                        previous_manifest.judgments_path,
                        previous_manifest.candidate_count,
                        previous_manifest.judgment_count,
                        previous_manifest.pending_count,
                        previous_manifest.selected_evidence_ids_hash,
                        previous_manifest.view_assurance,
                        previous_manifest.expected_digest,
                    ),
                    (
                        resolved.manifest.suite_run_id,
                        pool_manifest.pool_manifest_id,
                        identity["judge_id"],
                        requested_assurance,
                        hash_file(pool_manifest_path),
                        repo_relative_path(
                            checkpoint_judgments_path,
                            state.repo_root,
                        ),
                        len(resolved.prepared.candidates),
                        len(previous_view),
                        len(resolved.prepared.candidates) - len(previous_view),
                        evidence_ids_hash,
                        view_assurance,
                        (
                            state.config.judge.model.expected_digest
                            if requested_assurance == "strict_digest"
                            else None
                        ),
                    ),
                    label="judge checkpoint active inputs",
                    error_type=RelevanceError,
                )
            elif checkpoint_judgments_path.exists():
                raise RelevanceError("judge checkpoint exists without its manifest")

            cache_dir = judgment_cache_root(state)
            evidence_dir = cache_dir / "evidence" / identity["judge_id"]
            failure_dir = cache_dir / "technical_failures" / identity["judge_id"]
            engine = AsyncRelevanceJudgeEngine(
                resolved.prepared,
                assurance=requested_assurance,
                slot_names=slots,
                evidence_dir=evidence_dir,
                failure_dir=failure_dir,
                evidence_read_dirs=[
                    path / "evidence" / identity["judge_id"]
                    for path in judgment_cache_read_roots(state)
                ],
                previous_view=previous_view,
                provider_factory=self.provider_factory,
                judgment_limit=judgment_limit,
                progress_callback=progress_callback,
            )
            started = _utc_now()
            execution_id = stable_id(
                "judge_execution",
                resolved.manifest.suite_run_id,
                started,
                uuid.uuid4().hex,
            )
            execution_log = judge_root / "executions.jsonl"
            try:
                result = await engine.run()
            except asyncio.CancelledError as exc:
                result = engine.snapshot()
                manifest, materialized_manifest_path = self._materialize_view(
                    result,
                    assurance=requested_assurance,
                    semantic_judge_id=identity["judge_id"],
                    pool_manifest=pool_manifest,
                    pool_manifest_path=pool_manifest_path,
                    checkpoint_judgments_path=checkpoint_judgments_path,
                    checkpoint_manifest_path=checkpoint_manifest_path,
                    views_root=judge_root / "views",
                    expected_digest=state.config.judge.model.expected_digest,
                    repo_root=state.repo_root,
                )
                self._record_execution(
                    execution_log,
                    execution_id=execution_id,
                    started=started,
                    status="interrupted",
                    result=result,
                    slots=slots,
                    profiles=profiles,
                    pool_manifest=pool_manifest,
                    semantic_judge_id=identity["judge_id"],
                    assurance=requested_assurance,
                    view_manifest=manifest,
                    view_manifest_path=materialized_manifest_path,
                    errors=["interrupted"],
                    repo_root=state.repo_root,
                )
                raise ExperimentSuiteJudgeInterrupted(
                    resolved.manifest.suite_run_id,
                ) from exc
            except Exception as exc:
                result = engine.snapshot()
                manifest, materialized_manifest_path = self._materialize_view(
                    result,
                    assurance=requested_assurance,
                    semantic_judge_id=identity["judge_id"],
                    pool_manifest=pool_manifest,
                    pool_manifest_path=pool_manifest_path,
                    checkpoint_judgments_path=checkpoint_judgments_path,
                    checkpoint_manifest_path=checkpoint_manifest_path,
                    views_root=judge_root / "views",
                    expected_digest=state.config.judge.model.expected_digest,
                    repo_root=state.repo_root,
                )
                error = _redacted_error(exc, profiles=profiles)
                self._record_execution(
                    execution_log,
                    execution_id=execution_id,
                    started=started,
                    status="failed",
                    result=result,
                    slots=slots,
                    profiles=profiles,
                    pool_manifest=pool_manifest,
                    semantic_judge_id=identity["judge_id"],
                    assurance=requested_assurance,
                    view_manifest=manifest,
                    view_manifest_path=materialized_manifest_path,
                    errors=[error],
                    repo_root=state.repo_root,
                )
                raise

            manifest, materialized_manifest_path = self._materialize_view(
                result,
                assurance=requested_assurance,
                semantic_judge_id=identity["judge_id"],
                pool_manifest=pool_manifest,
                pool_manifest_path=pool_manifest_path,
                checkpoint_judgments_path=checkpoint_judgments_path,
                checkpoint_manifest_path=checkpoint_manifest_path,
                views_root=judge_root / "views",
                expected_digest=state.config.judge.model.expected_digest,
                repo_root=state.repo_root,
            )
            status: Literal["complete", "incomplete"] = (
                "complete"
                if result.pending_judgments == 0 and result.failed_judgments == 0
                else "incomplete"
            )
            self._record_execution(
                execution_log,
                execution_id=execution_id,
                started=started,
                status=status,
                result=result,
                slots=slots,
                profiles=profiles,
                pool_manifest=pool_manifest,
                semantic_judge_id=identity["judge_id"],
                assurance=requested_assurance,
                view_manifest=manifest,
                view_manifest_path=materialized_manifest_path,
                errors=list(result.failures.values()),
                repo_root=state.repo_root,
            )
            return SuiteJudgeSummary(
                suite_run_id=resolved.manifest.suite_run_id,
                judge_id=selected_judge_id,
                semantic_judge_id=identity["judge_id"],
                status=status,
                requested_assurance=requested_assurance,
                view_assurance=result.view_assurance,
                judgment_count=len(result.judgments),
                new_judgments=result.new_judgments,
                reused_judgments=result.reused_judgments,
                failed_judgments=result.failed_judgments,
                pending_judgments=result.pending_judgments,
                view_manifest_path=repo_relative_path(
                    materialized_manifest_path,
                    state.repo_root,
                ),
                view_manifest_hash=hash_file(materialized_manifest_path),
                execution_log_path=repo_relative_path(
                    execution_log,
                    state.repo_root,
                ),
            )

    def _materialize_view(
        self,
        result: JudgmentEngineResult,
        *,
        assurance: AssuranceMode,
        semantic_judge_id: str,
        pool_manifest: SuitePoolManifest,
        pool_manifest_path: Path,
        checkpoint_judgments_path: Path,
        checkpoint_manifest_path: Path,
        views_root: Path,
        expected_digest: str | None,
        repo_root: Path,
    ) -> tuple[JudgmentViewManifest, Path]:
        judgments_hash = hash_jsonl_records(
            record.to_dict() for record in result.judgments
        )
        evidence_ids = [
            str(record.provenance["deployment_evidence_id"])
            for record in result.judgments
        ]
        evidence_hash = hash_jsonable(evidence_ids)
        status: Literal["complete", "incomplete"] = (
            "complete"
            if result.pending_judgments == 0 and result.failed_judgments == 0
            else "incomplete"
        )
        digest = expected_digest if assurance == "strict_digest" else None
        view_id = stable_id(
            "judgment_view",
            pool_manifest.suite_run_id,
            pool_manifest.pool_manifest_id,
            semantic_judge_id,
            assurance,
            digest,
            result.view_assurance,
            judgments_hash,
            evidence_hash,
        )

        def manifest_for(judgments_path: Path) -> JudgmentViewManifest:
            return JudgmentViewManifest(
                judgment_view_id=view_id,
                suite_run_id=pool_manifest.suite_run_id,
                pool_manifest_id=pool_manifest.pool_manifest_id,
                pool_manifest_hash=hash_file(pool_manifest_path),
                semantic_judge_id=semantic_judge_id,
                requested_assurance=assurance,
                view_assurance=result.view_assurance,
                expected_digest=digest,
                status=status,
                judgments_path=repo_relative_path(judgments_path, repo_root),
                judgments_hash=judgments_hash,
                candidate_count=len(result.judgments) + result.pending_judgments,
                judgment_count=len(result.judgments),
                pending_count=result.pending_judgments,
                selected_evidence_ids_hash=evidence_hash,
            )

        # Checkpoints are mutable and resumable by design.  They are never used as
        # immutable scientific views while incomplete.
        write_jsonl_atomic(checkpoint_judgments_path, result.judgments)
        require_compatible(
            hash_file(checkpoint_judgments_path),
            judgments_hash,
            label="judge checkpoint canonical serialization",
            error_type=RelevanceError,
        )
        checkpoint_manifest = manifest_for(checkpoint_judgments_path)
        write_json_object_atomic(
            checkpoint_manifest_path,
            checkpoint_manifest.to_dict(),
            validate=True,
        )
        if status != "complete":
            return checkpoint_manifest, checkpoint_manifest_path

        immutable_dir = views_root / view_id
        immutable_judgments_path = immutable_dir / "judgments.jsonl"
        immutable_manifest_path = immutable_dir / "manifest.json"
        immutable_manifest = manifest_for(immutable_judgments_path)
        existing = _read_view_manifest(immutable_manifest_path)
        if existing is not None:
            artifact_references(
                {"judgments": existing.judgments_path},
                {"judgments": existing.judgments_hash},
                repo_root=repo_root,
                label="immutable judgment view",
                error_type=RelevanceError,
            )
            require_compatible(
                existing.to_dict(),
                immutable_manifest.to_dict(),
                label="immutable judgment view content",
                error_type=RelevanceError,
            )
            return existing, immutable_manifest_path

        bundle = ImmutableArtifactBundle(
            manifest_path=immutable_manifest_path,
            output_paths={"judgments": immutable_judgments_path},
            label="immutable judgment view",
            error_type=RelevanceError,
        )
        bundle.write_jsonl("judgments", result.judgments)
        require_compatible(
            bundle.output_hashes["judgments"],
            judgments_hash,
            label="immutable judgment view canonical serialization",
            error_type=RelevanceError,
        )
        bundle.commit(immutable_manifest)
        return immutable_manifest, immutable_manifest_path

    def _record_execution(
        self,
        path: Path,
        *,
        execution_id: str,
        started: str,
        status: Literal["complete", "incomplete", "interrupted", "failed"],
        result: JudgmentEngineResult,
        slots: list[str],
        profiles: dict[str, JudgeEndpointProfile],
        pool_manifest: SuitePoolManifest,
        semantic_judge_id: str,
        assurance: AssuranceMode,
        view_manifest: JudgmentViewManifest,
        view_manifest_path: Path,
        errors: list[str],
        repo_root: Path,
    ) -> None:
        completed = _utc_now()
        git_commit, git_dirty, git_state_hash = capture_git_state(repo_root)
        preflight_by_slot = {item.slot_id: item for item in result.preflights}
        occurrences: dict[str, int] = {}
        slot_records: list[JsonDict] = []
        for profile_name in slots:
            occurrence = occurrences.get(profile_name, 0) + 1
            occurrences[profile_name] = occurrence
            slot_id = f"{profile_name}#{occurrence}"
            profile = profiles[profile_name]
            preflight = preflight_by_slot.get(slot_id)
            slot_records.append(
                {
                    "slot_id": slot_id,
                    "profile_name": profile_name,
                    "service": profile.service,
                    "request_model_id": profile.request_model_id,
                    "endpoint_hash": hash_text(profile.endpoint),
                    "api_key_env": profile.api_key_env,
                    "provider_routing": (
                        profile.provider_routing.to_dict()
                        if profile.provider_routing is not None
                        else None
                    ),
                    "preflight": preflight.to_dict() if preflight else None,
                    "counts": result.slot_counts.get(slot_id, {}),
                },
            )
        record = JudgeExecutionRecord(
            execution_id=execution_id,
            suite_run_id=pool_manifest.suite_run_id,
            pool_manifest_id=pool_manifest.pool_manifest_id,
            semantic_judge_id=semantic_judge_id,
            requested_assurance=assurance,
            status=status,
            started_at_utc=started,
            completed_at_utc=completed,
            duration_seconds=max(
                0.0,
                (_parse_utc(completed) - _parse_utc(started)).total_seconds(),
            ),
            slot_count=len(slots),
            slots=slot_records,
            counts={
                "judgments": len(result.judgments),
                "new": result.new_judgments,
                "reused": result.reused_judgments,
                "failed": result.failed_judgments,
                "pending": result.pending_judgments,
                "requests": sum(
                    int(counts.get("requests", 0))
                    for counts in result.slot_counts.values()
                ),
            },
            priority_selection=result.selection,
            judgment_view_id=view_manifest.judgment_view_id,
            judgment_view_hash=hash_file(view_manifest_path),
            errors=sorted(
                {
                    _redacted_error(error, profiles=profiles)
                    for error in errors
                    if error
                },
            ),
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_state_hash=git_state_hash,
        )
        _append_execution(path, record)
