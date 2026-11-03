"""Suite-owned relevance pooling and exact local-judge cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import (
    ArtifactProvenance,
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    hash_jsonl_records,
    read_json_object,
    resolve_repo_path,
)
from thesis_rag.eval.relevance import (
    prepare_relevance_pool,
)
from thesis_rag.eval.relevance_config import load_relevance_config
from thesis_rag.eval.relevance_types import (
    RelevancePoolPreparation,
    RelevanceProgressCallback,
    RunEvaluationSelection,
)
from thesis_rag.eval.suite import ExperimentSuiteManifest, exclusive_suite_lock, load_pinned_retrievals
from thesis_rag.eval.suite_config import (
    ExperimentSuiteError,
    SuiteResolution,
    resolve_experiment_suite,
)
from thesis_rag.schema import (
    SUITE_POOL_MANIFEST_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    stable_id,
)


class SuitePoolManifest(ArtifactModel):
    """Immutable commit record for one suite-owned relevance pool."""

    pool_manifest_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    suite_manifest_path: NonEmptyStr
    suite_manifest_hash: Sha256
    suite_resolution_hash: Sha256
    relevance_config_path: NonEmptyStr
    pool_policy_hash: Sha256
    corpus_hash: Sha256
    question_artifact_hash: Sha256
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    output_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    output_artifact_hashes: dict[NonEmptyStr, Sha256]
    counts: JsonDict
    created_at_utc: NonEmptyStr
    status: Literal["complete"] = "complete"
    schema_version: Literal["retrieval_suite_pool_manifest_v3"] = (
        SUITE_POOL_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        required = {"anchor_projections", "candidate_pool", "pool_report"}
        if set(self.output_artifact_paths) != required:
            raise ValueError("pool manifest requires exactly three output artifacts")
        if set(self.output_artifact_hashes) != required:
            raise ValueError("pool manifest output paths and hashes must match")
        expected = stable_id(
            "suite_pool_manifest",
            self.suite_run_id,
            self.suite_manifest_hash,
            self.pool_policy_hash,
            self.output_artifact_hashes,
            self.counts,
        )
        if self.pool_manifest_id != expected:
            raise ValueError("pool_manifest_id does not match immutable lineage")
        return self


@dataclass(frozen=True)
class SuitePoolSummary:
    suite_run_id: str
    dry_run: bool
    candidate_count: int
    projection_count: int
    cost_report: dict[str, object]
    artifacts: dict[str, object]
    reused: bool = False


@dataclass(frozen=True)
class PreparedSuitePool:
    """Verified suite inputs plus the exact in-memory relevance pool."""

    manifest: ExperimentSuiteManifest
    manifest_path: Path
    relevance_path: Path
    prepared: RelevancePoolPreparation


def require_materialized_suite_pool(
    resolved: PreparedSuitePool,
) -> tuple[SuitePoolManifest, Path]:
    """Verify and return the committed pool backing a resolved suite."""

    pool_manifest_path = resolved.manifest_path.parent / "pool" / "manifest.json"
    if not pool_manifest_path.is_file():
        raise ExperimentSuiteError(
            "this operation requires a materialized pool; run `eval suite pool` "
            "after its dry-run gate",
        )
    pool_manifest = SuitePoolManifest.from_dict(
        read_json_object(pool_manifest_path, error_type=ExperimentSuiteError),
    )
    require_compatible(
        (
            pool_manifest.suite_run_id,
            pool_manifest.suite_manifest_hash,
            pool_manifest.pool_policy_hash,
        ),
        (
            resolved.manifest.suite_run_id,
            hash_file(resolved.manifest_path),
            hash_jsonable(resolved.prepared.state.config.pool_policy_payload()),
        ),
        label="materialized suite pool lineage",
        error_type=ExperimentSuiteError,
    )
    artifact_references(
        pool_manifest.output_artifact_paths,
        pool_manifest.output_artifact_hashes,
        repo_root=resolved.prepared.state.repo_root,
        label="materialized suite pool",
        error_type=ExperimentSuiteError,
    )
    expected_hashes = {
        "anchor_projections": hash_jsonl_records(
            item.to_dict() for item in resolved.prepared.projections
        ),
        "candidate_pool": hash_jsonl_records(
            item.to_dict() for item in resolved.prepared.candidates
        ),
    }
    for name, expected_hash in expected_hashes.items():
        require_compatible(
            pool_manifest.output_artifact_hashes[name],
            expected_hash,
            label=f"materialized suite pool {name}",
            error_type=ExperimentSuiteError,
        )
    return pool_manifest, pool_manifest_path


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _manifest_artifacts(
    manifest: SuitePoolManifest,
    manifest_path: Path,
    root: Path,
) -> dict[str, object]:
    return {
        **{
            f"{name}_path": path
            for name, path in manifest.output_artifact_paths.items()
        },
        **{
            f"{name}_hash": value
            for name, value in manifest.output_artifact_hashes.items()
        },
        "anchor_projection_count": manifest.counts["anchor_projections"],
        "candidate_count": manifest.counts["candidate_pool"],
        "pool_manifest_id": manifest.pool_manifest_id,
        "pool_manifest_path": repo_relative_path(manifest_path, root),
        "pool_manifest_hash": hash_file(manifest_path),
    }


def _selected_target_levels(manifest: ExperimentSuiteManifest) -> set[str]:
    levels: set[str] = set()
    for experiment in manifest.experiments:
        final = experiment.final_evaluation
        if final.include_in_pool or final.compute_metrics:
            levels.update(final.target_levels)
        for stage in experiment.stages:
            if stage.include_in_pool or stage.compute_metrics:
                levels.update(stage.target_levels)
    return levels


def _materialize_suite_pool(
    *,
    prepared: RelevancePoolPreparation,
    suite_manifest: ExperimentSuiteManifest,
    suite_manifest_path: Path,
    relevance_path: Path,
    pool_dir: Path,
    root: Path,
) -> tuple[dict[str, object], bool]:
    pool_manifest_path = pool_dir / "manifest.json"
    projection_path = pool_dir / "anchor_projections.jsonl"
    candidate_path = pool_dir / "candidate_pool.jsonl"
    pool_report_path = pool_dir / "pool_report.json"
    expected_projection_hash = hash_jsonl_records(
        record.to_dict() for record in prepared.projections
    )
    expected_candidate_hash = hash_jsonl_records(
        record.to_dict() for record in prepared.candidates
    )
    pool_policy_hash = hash_jsonable(
        prepared.state.config.pool_policy_payload(),
    )
    suite_manifest_hash = hash_file(suite_manifest_path)
    counts: JsonDict = {
        "questions": len(prepared.state.questions),
        "anchor_projections": len(prepared.projections),
        "candidate_pool": len(prepared.candidates),
    }
    expected_output_paths = {
        "anchor_projections": repo_relative_path(projection_path, root),
        "candidate_pool": repo_relative_path(candidate_path, root),
        "pool_report": repo_relative_path(pool_report_path, root),
    }
    expected_suite_manifest_path = repo_relative_path(suite_manifest_path, root)
    expected_relevance_path = repo_relative_path(relevance_path, root)

    if pool_manifest_path.is_file():
        existing = SuitePoolManifest.from_dict(
            read_json_object(pool_manifest_path, error_type=ExperimentSuiteError),
        )
        require_compatible(
            (
                existing.suite_run_id,
                existing.suite_manifest_path,
                existing.suite_manifest_hash,
                existing.suite_resolution_hash,
                existing.relevance_config_path,
                existing.pool_policy_hash,
                existing.corpus_hash,
                existing.question_artifact_hash,
                existing.output_artifact_paths,
                existing.counts,
            ),
            (
                suite_manifest.suite_run_id,
                expected_suite_manifest_path,
                suite_manifest_hash,
                suite_manifest.resolution_hash,
                expected_relevance_path,
                pool_policy_hash,
                suite_manifest.corpus.corpus_hash,
                suite_manifest.question_set.questions_hash,
                expected_output_paths,
                counts,
            ),
            label="existing suite pool immutable inputs",
            error_type=ExperimentSuiteError,
        )
        expected_hashes = {
            "anchor_projections": expected_projection_hash,
            "candidate_pool": expected_candidate_hash,
        }
        for name, expected_hash in expected_hashes.items():
            require_compatible(
                existing.output_artifact_hashes[name],
                expected_hash,
                label=f"existing suite pool {name}",
                error_type=ExperimentSuiteError,
            )
        artifact_references(
            existing.output_artifact_paths,
            existing.output_artifact_hashes,
            repo_root=root,
            label="existing suite pool",
            error_type=ExperimentSuiteError,
        )
        return _manifest_artifacts(existing, pool_manifest_path, root), True

    bundle = ImmutableArtifactBundle(
        manifest_path=pool_manifest_path,
        output_paths={
            "anchor_projections": projection_path,
            "candidate_pool": candidate_path,
            "pool_report": pool_report_path,
        },
        label="suite pool",
        error_type=ExperimentSuiteError,
    )
    bundle.write_jsonl("anchor_projections", prepared.projections)
    bundle.write_jsonl("candidate_pool", prepared.candidates)
    if (
        bundle.output_paths["anchor_projections"].is_file()
        and (
            hash_file(projection_path) != expected_projection_hash
            or hash_file(candidate_path) != expected_candidate_hash
        )
    ):
        raise AssertionError("materialized pool hashes differ from preparation")
    pool_report = {
        key: value
        for key, value in prepared.cost_report.items()
        if key not in {"judgment_status", "eta"}
    }
    pool_report["schema_version"] = "pool_composition_report_v2"
    bundle.write_json("pool_report", pool_report)
    output_paths = expected_output_paths
    output_hashes = bundle.output_hashes
    provenance = ArtifactProvenance.capture(root, code_paths={})
    pool_manifest_id = stable_id(
        "suite_pool_manifest",
        suite_manifest.suite_run_id,
        suite_manifest_hash,
        pool_policy_hash,
        output_hashes,
        counts,
    )
    manifest = SuitePoolManifest(
        pool_manifest_id=pool_manifest_id,
        suite_run_id=suite_manifest.suite_run_id,
        suite_manifest_path=expected_suite_manifest_path,
        suite_manifest_hash=suite_manifest_hash,
        suite_resolution_hash=suite_manifest.resolution_hash,
        relevance_config_path=expected_relevance_path,
        pool_policy_hash=pool_policy_hash,
        corpus_hash=suite_manifest.corpus.corpus_hash,
        question_artifact_hash=suite_manifest.question_set.questions_hash,
        git_commit=provenance.git_commit,
        git_dirty=provenance.git_dirty,
        git_state_hash=provenance.git_state_hash,
        output_artifact_paths=output_paths,
        output_artifact_hashes=output_hashes,
        counts=counts,
        created_at_utc=_now_utc(),
    )
    bundle.commit(manifest)
    return _manifest_artifacts(manifest, pool_manifest_path, root), False


class ExperimentSuitePoolJob:
    """Pool only the exact completed retrieval views selected by one suite."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        relevance_config_path: str | Path = "configs/thesis/relevance.yaml",
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.relevance_config_path = Path(relevance_config_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(
        self,
        *,
        dry_run: bool = False,
        verify: bool = True,
        progress_callback: RelevanceProgressCallback | None = None,
    ) -> SuitePoolSummary:
        resolution = resolve_experiment_suite(
            self.config_path,
            repo_root=self.repo_root,
            suite_output_root=self.suite_output_root,
        )
        with exclusive_suite_lock(resolution, operation="pool"):
            return self._run_resolved(
                resolution,
                dry_run=dry_run,
                verify=verify,
                progress_callback=progress_callback,
            )

    def _run_resolved(
        self,
        resolution: SuiteResolution,
        *,
        dry_run: bool,
        verify: bool,
        progress_callback: RelevanceProgressCallback | None,
    ) -> SuitePoolSummary:
        resolved = self.prepare_resolved(
            resolution,
            verify=verify,
            progress_callback=progress_callback,
        )
        root = resolution.repo_root
        manifest = resolved.manifest
        manifest_path = resolved.manifest_path
        relevance_path = resolved.relevance_path
        prepared = resolved.prepared
        artifacts: dict[str, object] = {}
        reused = False
        if not dry_run:
            artifacts, reused = _materialize_suite_pool(
                prepared=prepared,
                suite_manifest=manifest,
                suite_manifest_path=manifest_path,
                relevance_path=relevance_path,
                pool_dir=manifest_path.parent / "pool",
                root=root,
            )
            if artifacts["candidate_count"] != prepared.cost_report[
                "unique_candidates_after_dedup"
            ]:
                raise AssertionError(
                    "pool estimate and materialized candidate count diverged",
                )
        return SuitePoolSummary(
            suite_run_id=manifest.suite_run_id,
            dry_run=dry_run,
            candidate_count=len(prepared.candidates),
            projection_count=len(prepared.projections),
            cost_report=prepared.cost_report,
            artifacts=artifacts,
            reused=reused,
        )

    def prepare_resolved(
        self,
        resolution: SuiteResolution,
        *,
        verify: bool,
        progress_callback: RelevanceProgressCallback | None,
    ) -> PreparedSuitePool:
        """Verify all immutable suite inputs and reconstruct its exact pool."""

        if resolution.question_input.seal_kind == "manual_benchmark":
            raise ExperimentSuiteError(
                "manual Page navigation uses known targets, not QREL pooling or judging; "
                "run suite metrics and report after retrieval"
            )
        root = resolution.repo_root
        manifest_path = (
            resolve_repo_path(self.suite_output_root, root)
            / resolution.suite_run_id
            / "manifest.json"
        )
        if not manifest_path.is_file():
            raise ExperimentSuiteError(
                "suite has not been planned; run the suite plan command first",
            )
        manifest = ExperimentSuiteManifest.from_dict(
            read_json_object(manifest_path, error_type=ExperimentSuiteError),
        )
        if manifest.resolution_hash != resolution.resolution_hash:
            raise ExperimentSuiteError(
                "persisted suite plan differs from the current suite resolution",
            )
        if manifest.status != "complete":
            raise ExperimentSuiteError(
                "suite pooling requires every selected retrieval experiment to be "
                "complete",
            )

        relevance_path = resolve_repo_path(self.relevance_config_path, root)
        relevance = load_relevance_config(relevance_path)
        configured_snapshot = resolve_repo_path(
            relevance.inputs.snapshot_manifest_path,
            root,
        )
        if (
            configured_snapshot.resolve()
            != resolve_repo_path(manifest.corpus.snapshot_manifest_path, root).resolve()
            or hash_file(configured_snapshot) != manifest.corpus.snapshot_manifest_hash
        ):
            raise ExperimentSuiteError(
                "relevance config and suite do not pin the same corpus snapshot",
            )

        selected_levels = _selected_target_levels(manifest)
        disabled_levels = {
            level
            for level, enabled in (
                ("page", relevance.target_spaces.include_pages),
                ("section", relevance.target_spaces.include_sections),
                (
                    "context_group",
                    relevance.target_spaces.include_context_groups,
                ),
            )
            if level in selected_levels and not enabled
        }
        if disabled_levels:
            raise ExperimentSuiteError(
                "suite selects target levels disabled by the relevance config: "
                f"{sorted(disabled_levels)}",
            )

        retrieval_paths: list[Path] = []
        selections: dict[str, RunEvaluationSelection] = {}
        for experiment, run_manifest, _, retrieval_path in load_pinned_retrievals(manifest, root):
            retrieval_paths.append(retrieval_path)
            selections[run_manifest.run_id] = RunEvaluationSelection(experiment)

        cached_scope_pages = None
        if any(e.final_evaluation.page_scopes is not None
               and e.final_evaluation.page_scopes.judgment_mode == "cached_only" for e in manifest.experiments):
            frozen_pool_path = manifest_path.parent / "pool/manifest.json"
            if frozen_pool_path.exists():
                from thesis_rag.eval.evidence_sources import jsonl_records

                frozen = SuitePoolManifest.from_dict(read_json_object(frozen_pool_path, error_type=ExperimentSuiteError))
                if frozen.suite_run_id != manifest.suite_run_id or frozen.suite_manifest_hash != hash_file(manifest_path):
                    raise ExperimentSuiteError("cached Page scope pool belongs to another suite")
                artifact_references(frozen.output_artifact_paths, frozen.output_artifact_hashes,
                                    repo_root=root, label="cached Page scope pool", error_type=ExperimentSuiteError)
                cached_scope_pages = {}
                for record in jsonl_records(resolve_repo_path(frozen.output_artifact_paths["candidate_pool"], root)):
                    if record["target"]["target_space_id"] == "page":
                        cached_scope_pages.setdefault(record["query_id"], set()).add(record["target"]["page_id"])

        prepared = prepare_relevance_pool(
            relevance_path,
            retrieval_result_paths=retrieval_paths,
            suite_run_id=manifest.suite_run_id,
            selection_by_run_id=selections,
            question_artifact_path=manifest.question_set.questions_path,
            expected_question_artifact_hash=(
                manifest.question_set.questions_hash
            ),
            repo_root=root,
            verify=verify,
            progress_callback=progress_callback,
            cached_scope_pages=cached_scope_pages,
        )
        return PreparedSuitePool(
            manifest=manifest,
            manifest_path=manifest_path,
            relevance_path=relevance_path,
            prepared=prepared,
        )
