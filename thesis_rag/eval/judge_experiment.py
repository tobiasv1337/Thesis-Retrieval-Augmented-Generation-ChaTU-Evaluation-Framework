"""Configuration, orchestration, and comparison for multi-judge experiments."""

from __future__ import annotations

from dataclasses import dataclass
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
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import artifact_references, require_compatible
from thesis_rag.eval.artifact_io import (
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
    write_json_object_atomic,
)
from thesis_rag.eval.relevance_judgments import RELEVANCE_PIPELINE_VERSION
from thesis_rag.eval.relevance_agreement import score_agreement
from thesis_rag.eval.relevance_config import RelevanceConfig, load_relevance_config
from thesis_rag.eval.relevance_judgment_core import RelevanceJudgeResponse
from thesis_rag.eval.relevance_judge import (
    AssuranceMode,
    AsyncProviderFactory,
    default_async_provider_factory,
    read_judgment_view,
)
from thesis_rag.eval.relevance_schemas import (
    RelevanceCandidateRecord,
    RelevanceJudgmentRecord,
)
from thesis_rag.eval.relevance_types import RelevanceError, RelevanceProgressCallback
from thesis_rag.eval.suite_config import ExperimentSuiteError, resolve_experiment_suite
from thesis_rag.eval.suite_judge import (
    ExperimentSuiteJudgeJob,
    JudgmentViewManifest,
    SuiteJudgeSummary,
)
from thesis_rag.eval.suite_pool import SuitePoolManifest
from thesis_rag.eval.suite_qrel_review import ExperimentSuiteQrelReviewJob
from thesis_rag.schema import (
    CANDIDATE_POOL_SCHEMA_VERSION,
    JUDGE_COMPARISON_REPORT_SCHEMA_VERSION,
    JUDGE_EXPERIMENT_CONFIG_SCHEMA_VERSION,
    JUDGE_EXPERIMENT_RUN_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


class JudgeExperimentConfig(ArtifactModel):
    """A named set of semantic judges executed over one existing suite pool."""

    judge_experiment_id: NonEmptyStr
    relevance_config_path: NonEmptyStr
    judge_ids: list[NonEmptyStr]
    schema_version: Literal["relevance_judge_experiment_v1"] = (
        JUDGE_EXPERIMENT_CONFIG_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def judge_ids_are_unique(self) -> Self:
        if not self.judge_ids:
            raise ValueError("judge experiment must select at least one judge")
        if len(self.judge_ids) != len(set(self.judge_ids)):
            raise ValueError("judge experiment judge_ids must be unique")
        return self


class JudgeExperimentResult(ArtifactModel):
    judge_id: NonEmptyStr
    semantic_judge_id: NonEmptyStr
    requested_assurance: AssuranceMode
    status: Literal["complete", "incomplete"]
    judgment_count: NonNegativeInt
    failed_judgments: NonNegativeInt
    pending_judgments: NonNegativeInt
    view_manifest_path: NonEmptyStr
    view_manifest_hash: Sha256
    execution_log_path: NonEmptyStr


class JudgeExperimentRunManifest(ArtifactModel):
    """Immutable index of independently materialized views from one judge set."""

    judge_experiment_run_id: NonEmptyStr
    judge_experiment_id: NonEmptyStr
    judge_experiment_config_path: NonEmptyStr
    judge_experiment_config_hash: Sha256
    relevance_config_path: NonEmptyStr
    judge_experiment_policy_hash: Sha256
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    pool_manifest_hash: Sha256
    judge_results: list[JudgeExperimentResult]
    status: Literal["complete", "incomplete"]
    created_at_utc: NonEmptyStr
    schema_version: Literal["judge_experiment_run_v1"] = (
        JUDGE_EXPERIMENT_RUN_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        ids = [item.judge_id for item in self.judge_results]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("judge experiment run judges must be nonempty and unique")
        expected_status = (
            "complete"
            if all(item.status == "complete" for item in self.judge_results)
            else "incomplete"
        )
        if self.status != expected_status:
            raise ValueError("judge experiment run status disagrees with its results")
        expected = stable_id(
            "judge_experiment_run",
            self.judge_experiment_id,
            self.judge_experiment_config_hash,
            self.judge_experiment_policy_hash,
            self.suite_run_id,
            self.pool_manifest_id,
            [item.to_dict() for item in self.judge_results],
            self.status,
        )
        if self.judge_experiment_run_id != expected:
            raise ValueError("judge experiment run identity mismatch")
        return self


class JudgeComparisonReport(ArtifactModel):
    """Agreement metrics over immutable views of one multi-judge run."""

    judge_comparison_report_id: NonEmptyStr
    judge_experiment_run_id: NonEmptyStr
    judge_experiment_run_manifest_path: NonEmptyStr
    judge_experiment_run_manifest_hash: Sha256
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    pool_manifest_hash: Sha256
    judge_ids: list[NonEmptyStr]
    manual_judgments_path: NonEmptyStr | None
    manual_judgments_hash: Sha256 | None
    manual_reference_status: Literal["missing", "partial", "complete"]
    manual_reference_count: NonNegativeInt
    human_comparisons: dict[NonEmptyStr, JsonDict]
    pairwise_comparisons: dict[NonEmptyStr, JsonDict]
    status: Literal["complete", "incomplete"]
    created_at_utc: NonEmptyStr
    schema_version: Literal["judge_comparison_report_v1"] = (
        JUDGE_COMPARISON_REPORT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if self.judge_ids != list(dict.fromkeys(self.judge_ids)):
            raise ValueError("judge comparison judge_ids must be unique and ordered")
        if (self.manual_judgments_path is None) != (
            self.manual_judgments_hash is None
        ):
            raise ValueError("manual judgment path and hash must be atomic")
        if set(self.human_comparisons) != set(self.judge_ids):
            raise ValueError("human comparisons must cover every judge")
        expected_pairs = {
            f"{left}::{right}"
            for index, left in enumerate(self.judge_ids)
            for right in self.judge_ids[index + 1 :]
        }
        if set(self.pairwise_comparisons) != expected_pairs:
            raise ValueError("pairwise comparisons do not cover the judge cohort")
        expected = stable_id(
            "judge_comparison_report",
            self.judge_experiment_run_id,
            self.judge_experiment_run_manifest_hash,
            self.pool_manifest_id,
            self.pool_manifest_hash,
            self.judge_ids,
            self.manual_judgments_hash,
            self.manual_reference_status,
            self.manual_reference_count,
            self.human_comparisons,
            self.pairwise_comparisons,
            self.status,
        )
        if self.judge_comparison_report_id != expected:
            raise ValueError("judge comparison report identity mismatch")
        return self


@dataclass(frozen=True)
class JudgeExperimentSummary:
    suite_run_id: str
    judge_experiment_id: str
    status: Literal["complete", "incomplete"]
    results: tuple[SuiteJudgeSummary, ...]
    manifest_path: str
    manifest_hash: str


@dataclass(frozen=True)
class JudgeComparisonSummary:
    suite_run_id: str
    judge_experiment_id: str
    status: Literal["complete", "incomplete"]
    manual_reference_status: Literal["missing", "partial", "complete"]
    report_path: str
    report_hash: str


@dataclass(frozen=True)
class VerifiedJudgeView:
    """One recursively verified immutable view from a judge experiment."""

    result: JudgeExperimentResult
    manifest: JudgmentViewManifest
    manifest_path: Path
    judgments_path: Path
    judgments: tuple[RelevanceJudgmentRecord, ...]


@dataclass(frozen=True)
class VerifiedJudgeExperimentViews:
    """Verified shared lineage and independent views for one judge cohort."""

    repo_root: Path
    experiment: JudgeExperimentConfig
    relevance: RelevanceConfig
    run_manifest: JudgeExperimentRunManifest
    run_manifest_path: Path
    pool_manifest: SuitePoolManifest
    pool_manifest_path: Path
    expected_keys: frozenset[tuple[str, str, str, str, str]]
    views: tuple[VerifiedJudgeView, ...]


def load_judge_experiment_config(path: str | Path) -> JudgeExperimentConfig:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not read judge experiment {config_path}: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Judge experiment must be a YAML mapping: {config_path}")
    return JudgeExperimentConfig.from_dict(loaded)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def immutable_judge_comparison_report_path(
    experiment_dir: Path,
    comparison_report_id: str,
) -> Path:
    """Return the content-addressed path for one immutable comparison report."""

    return experiment_dir / "comparisons" / comparison_report_id / "report.json"


def load_verified_judge_experiment_views(
    suite_config_path: str | Path,
    judge_experiment_path: str | Path,
    *,
    suite_output_root: str | Path = "outputs/thesis/suites",
    repo_root: str | Path | None = None,
    require_complete: bool = False,
) -> VerifiedJudgeExperimentViews:
    """Load a judge experiment once and verify every referenced artifact."""

    suite_path = Path(suite_config_path)
    root = resolve_repo_root(suite_path, repo_root)
    experiment_path = resolve_repo_path(judge_experiment_path, root)
    experiment = load_judge_experiment_config(experiment_path)
    relevance_path = resolve_repo_path(experiment.relevance_config_path, root)
    relevance = load_relevance_config(relevance_path)
    resolution = resolve_experiment_suite(
        suite_path,
        repo_root=root,
        suite_output_root=suite_output_root,
    )
    experiment_dir = (
        resolve_repo_path(suite_output_root, root)
        / resolution.suite_run_id
        / "judge"
        / "experiments"
        / experiment.judge_experiment_id
    )
    run_manifest_path = experiment_dir / "manifest.json"
    if not run_manifest_path.is_file():
        raise ExperimentSuiteError(
            "judge views require a completed or checkpointed judge experiment run",
        )
    run_manifest = JudgeExperimentRunManifest.from_dict(
        read_json_object(run_manifest_path, error_type=ExperimentSuiteError),
    )
    require_compatible(
        (
            run_manifest.judge_experiment_id,
            run_manifest.judge_experiment_config_hash,
            run_manifest.judge_experiment_policy_hash,
            run_manifest.suite_run_id,
            [item.judge_id for item in run_manifest.judge_results],
        ),
        (
            experiment.judge_experiment_id,
            config_hash(experiment.to_dict()),
            _judge_experiment_policy_hash(relevance, experiment.judge_ids, root),
            resolution.suite_run_id,
            experiment.judge_ids,
        ),
        label="judge experiment manifest and active configs",
        error_type=ExperimentSuiteError,
    )
    if require_complete and run_manifest.status != "complete":
        raise ExperimentSuiteError(
            "this operation requires a complete judge experiment",
        )

    pool_manifest_path = experiment_dir.parents[2] / "pool" / "manifest.json"
    artifact_references(
        {"pool_manifest": pool_manifest_path},
        {"pool_manifest": run_manifest.pool_manifest_hash},
        repo_root=root,
        label="judge experiment pool dependency",
        error_type=ExperimentSuiteError,
    )
    pool_manifest = SuitePoolManifest.from_dict(
        read_json_object(pool_manifest_path, error_type=ExperimentSuiteError),
    )
    require_compatible(
        pool_manifest.pool_manifest_id,
        run_manifest.pool_manifest_id,
        label="judge experiment pool identity",
        error_type=ExperimentSuiteError,
    )
    artifact_references(
        pool_manifest.output_artifact_paths,
        pool_manifest.output_artifact_hashes,
        repo_root=root,
        label="judge experiment pool outputs",
        error_type=ExperimentSuiteError,
    )
    candidate_path = resolve_repo_path(
        pool_manifest.output_artifact_paths["candidate_pool"],
        root,
    )
    expected_keys = frozenset(
        (
            *RelevanceCandidateRecord.from_dict(payload).key,
            relevance.relevance_guideline_version,
        )
        for payload in read_jsonl(
            candidate_path,
            expected_schema_version=CANDIDATE_POOL_SCHEMA_VERSION,
        )
    )

    views: list[VerifiedJudgeView] = []
    for result in run_manifest.judge_results:
        view_manifest_path = resolve_repo_path(result.view_manifest_path, root)
        artifact_references(
            {"view_manifest": view_manifest_path},
            {"view_manifest": result.view_manifest_hash},
            repo_root=root,
            label=f"judgment view manifest for {result.judge_id}",
            error_type=ExperimentSuiteError,
        )
        view_manifest = JudgmentViewManifest.from_dict(
            read_json_object(view_manifest_path, error_type=ExperimentSuiteError),
        )
        require_compatible(
            (
                view_manifest.semantic_judge_id,
                view_manifest.pool_manifest_id,
                view_manifest.pool_manifest_hash,
                view_manifest.status,
                view_manifest.judgment_count,
                view_manifest.pending_count,
            ),
            (
                result.semantic_judge_id,
                run_manifest.pool_manifest_id,
                run_manifest.pool_manifest_hash,
                result.status,
                result.judgment_count,
                result.pending_judgments,
            ),
            label=f"judgment view lineage for {result.judge_id}",
            error_type=ExperimentSuiteError,
        )
        judgments_path = resolve_repo_path(view_manifest.judgments_path, root)
        artifact_references(
            {"judgments": judgments_path},
            {"judgments": view_manifest.judgments_hash},
            repo_root=root,
            label=f"judgment view for {result.judge_id}",
            error_type=ExperimentSuiteError,
        )
        judgments = tuple(read_judgment_view(judgments_path))
        if any(
            record.provenance.get("semantic_judge_id") != result.semantic_judge_id
            for record in judgments
        ):
            raise ExperimentSuiteError(
                f"judgment view contains a foreign judge for {result.judge_id}",
            )
        scores = {record.key: record.score for record in judgments}
        if len(scores) != len(judgments) or not set(scores) <= expected_keys:
            raise ExperimentSuiteError(
                f"judgment view contains duplicate or foreign candidates for "
                f"{result.judge_id}",
            )
        if require_complete and set(scores) != expected_keys:
            raise ExperimentSuiteError(
                f"complete metrics require full pool coverage from {result.judge_id}",
            )
        views.append(
            VerifiedJudgeView(
                result=result,
                manifest=view_manifest,
                manifest_path=view_manifest_path,
                judgments_path=judgments_path,
                judgments=judgments,
            ),
        )
    if run_manifest.status == "complete" and any(
        {record.key for record in view.judgments} != expected_keys for view in views
    ):
        raise ExperimentSuiteError(
            "complete judge views do not cover the complete candidate pool",
        )
    return VerifiedJudgeExperimentViews(
        repo_root=root,
        experiment=experiment,
        relevance=relevance,
        run_manifest=run_manifest,
        run_manifest_path=run_manifest_path,
        pool_manifest=pool_manifest,
        pool_manifest_path=pool_manifest_path,
        expected_keys=expected_keys,
        views=tuple(views),
    )


def _judge_experiment_policy_hash(
    relevance: RelevanceConfig,
    judge_ids: list[str],
    root: Path,
) -> str:
    """Bind selected configuration, prompt bytes, and response contract."""

    prompt_path = resolve_repo_path(relevance.judging.prompt.path, root)
    return config_hash(
        {
            **relevance.judge_experiment_policy_payload(judge_ids),
            "prompt_hash": hash_file(prompt_path),
            "output_schema_hash": hash_jsonable(
                RelevanceJudgeResponse.model_json_schema(),
            ),
            "pipeline_version": RELEVANCE_PIPELINE_VERSION,
        },
    )


class ExperimentSuiteMultiJudgeJob:
    """Execute a configured judge set sequentially over one immutable pool."""

    def __init__(
        self,
        suite_config_path: str | Path,
        judge_experiment_path: str | Path,
        *,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
        provider_factory: AsyncProviderFactory = default_async_provider_factory,
    ) -> None:
        self.suite_config_path = Path(suite_config_path)
        self.judge_experiment_path = Path(judge_experiment_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root
        self.provider_factory = provider_factory

    def run(
        self,
        *,
        assurance: AssuranceMode | None = None,
        judgment_limit: int | None = None,
        verify: bool = True,
        progress_callback: RelevanceProgressCallback | None = None,
    ) -> JudgeExperimentSummary:
        root = resolve_repo_root(self.suite_config_path, self.repo_root)
        experiment_path = resolve_repo_path(self.judge_experiment_path, root)
        experiment = load_judge_experiment_config(experiment_path)
        relevance_path = resolve_repo_path(experiment.relevance_config_path, root)
        relevance = load_relevance_config(relevance_path)
        unknown = sorted(set(experiment.judge_ids) - set(relevance.judging.judges))
        if unknown:
            raise RelevanceError(f"judge experiment references unknown judges: {unknown}")
        disabled = [
            judge_id
            for judge_id in experiment.judge_ids
            if not relevance.judging.judges[judge_id].enabled
        ]
        if disabled:
            raise RelevanceError(f"judge experiment references disabled judges: {disabled}")

        resolution = resolve_experiment_suite(
            self.suite_config_path,
            repo_root=root,
            suite_output_root=self.suite_output_root,
        )
        suite_dir = (
            resolve_repo_path(self.suite_output_root, root)
            / resolution.suite_run_id
        )
        pool_manifest_path = suite_dir / "pool" / "manifest.json"
        if not pool_manifest_path.is_file():
            raise ExperimentSuiteError(
                "multi-judge execution requires a materialized suite pool",
            )
        pool_manifest = SuitePoolManifest.from_dict(
            read_json_object(pool_manifest_path, error_type=ExperimentSuiteError),
        )

        summaries: list[SuiteJudgeSummary] = []
        for judge_id in experiment.judge_ids:
            summary = ExperimentSuiteJudgeJob(
                self.suite_config_path,
                relevance_config_path=relevance_path,
                suite_output_root=self.suite_output_root,
                repo_root=root,
                provider_factory=self.provider_factory,
            ).run(
                judge_id=judge_id,
                assurance=assurance,
                judgment_limit=judgment_limit,
                verify=verify,
                progress_callback=progress_callback,
            )
            summaries.append(summary)

        results = [
            JudgeExperimentResult(
                judge_id=item.judge_id,
                semantic_judge_id=item.semantic_judge_id,
                requested_assurance=item.requested_assurance,
                status=item.status,
                judgment_count=item.judgment_count,
                failed_judgments=item.failed_judgments,
                pending_judgments=item.pending_judgments,
                view_manifest_path=item.view_manifest_path,
                view_manifest_hash=item.view_manifest_hash,
                execution_log_path=item.execution_log_path,
            )
            for item in summaries
        ]
        status: Literal["complete", "incomplete"] = (
            "complete"
            if all(item.status == "complete" for item in results)
            else "incomplete"
        )
        experiment_hash = config_hash(experiment.to_dict())
        judge_policy_hash = _judge_experiment_policy_hash(
            relevance,
            experiment.judge_ids,
            root,
        )
        run_id = stable_id(
            "judge_experiment_run",
            experiment.judge_experiment_id,
            experiment_hash,
            judge_policy_hash,
            resolution.suite_run_id,
            pool_manifest.pool_manifest_id,
            [item.to_dict() for item in results],
            status,
        )
        manifest = JudgeExperimentRunManifest(
            judge_experiment_run_id=run_id,
            judge_experiment_id=experiment.judge_experiment_id,
            judge_experiment_config_path=repo_relative_path(
                experiment_path,
                root,
            ),
            judge_experiment_config_hash=experiment_hash,
            relevance_config_path=repo_relative_path(relevance_path, root),
            judge_experiment_policy_hash=judge_policy_hash,
            suite_run_id=resolution.suite_run_id,
            pool_manifest_id=pool_manifest.pool_manifest_id,
            pool_manifest_hash=hash_file(pool_manifest_path),
            judge_results=results,
            status=status,
            created_at_utc=_now_utc(),
        )
        manifest_path = (
            suite_dir
            / "judge"
            / "experiments"
            / experiment.judge_experiment_id
            / "manifest.json"
        )
        if manifest_path.is_file():
            existing = JudgeExperimentRunManifest.from_dict(
                read_json_object(manifest_path, error_type=RelevanceError),
            )
            if existing.judge_experiment_run_id == run_id:
                manifest = existing
            else:
                write_json_object_atomic(
                    manifest_path,
                    manifest.to_dict(),
                    validate=True,
                )
        else:
            write_json_object_atomic(
                manifest_path,
                manifest.to_dict(),
                validate=True,
            )
        return JudgeExperimentSummary(
            suite_run_id=resolution.suite_run_id,
            judge_experiment_id=experiment.judge_experiment_id,
            status=status,
            results=tuple(summaries),
            manifest_path=repo_relative_path(manifest_path, root),
            manifest_hash=hash_file(manifest_path),
        )


class ExperimentSuiteJudgeComparisonJob:
    """Compare immutable judge views with each other and manual judgments."""

    def __init__(
        self,
        suite_config_path: str | Path,
        judge_experiment_path: str | Path,
        *,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.suite_config_path = Path(suite_config_path)
        self.judge_experiment_path = Path(judge_experiment_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(self) -> JudgeComparisonSummary:
        verified = load_verified_judge_experiment_views(
            self.suite_config_path,
            self.judge_experiment_path,
            suite_output_root=self.suite_output_root,
            repo_root=self.repo_root,
        )
        root = verified.repo_root
        experiment = verified.experiment
        relevance = verified.relevance
        run_manifest = verified.run_manifest
        run_manifest_path = verified.run_manifest_path
        pool_manifest = verified.pool_manifest
        pool_manifest_path = verified.pool_manifest_path
        expected_keys = set(verified.expected_keys)
        score_maps = {
            view.result.judge_id: {
                record.key: record.score for record in view.judgments
            }
            for view in verified.views
        }

        manual_path: Path | None = None
        manual_hash: str | None = None
        manual_scores: dict[tuple[str, str, str, str, str], int] = {}
        suite_manual = ExperimentSuiteQrelReviewJob(
            self.suite_config_path,
            relevance_config_path=experiment.relevance_config_path,
            suite_output_root=self.suite_output_root,
            repo_root=root,
        ).current_materialized_view()
        suite_events_path = (
            pool_manifest_path.parent.parent
            / "human_qrels"
            / "primary"
            / "events.jsonl"
        )
        if suite_manual is not None:
            manual_path = resolve_repo_path(
                suite_manual.manifest.judgments_path,
                root,
            )
            manual_hash = suite_manual.manifest.judgments_hash
            manual_records = suite_manual.judgments
        else:
            manual_records = ()
            manual_path_value = relevance.inputs.manual_judgments_path
            if not suite_events_path.is_file() and manual_path_value is not None:
                configured_manual_path = resolve_repo_path(manual_path_value, root)
                if configured_manual_path.is_file():
                    manual_path = configured_manual_path
                    manual_hash = hash_file(configured_manual_path)
                    manual_records = tuple(
                        RelevanceJudgmentRecord.from_dict(payload)
                        for payload in read_jsonl(
                            configured_manual_path,
                            expected_schema_version=JUDGMENT_SCHEMA_VERSION,
                        )
                    )
        for record in manual_records:
            if record.provenance.get("kind") != "manual":
                raise RelevanceError(
                    "manual judgment file may contain only manual records",
                )
            if record.key in manual_scores:
                raise RelevanceError(
                    f"duplicate manual relevance judgment: {record.key}",
                )
            if record.key in expected_keys:
                manual_scores[record.key] = record.score

        if not expected_keys or not manual_scores:
            manual_status: Literal["missing", "partial", "complete"] = "missing"
        elif set(manual_scores) == expected_keys:
            manual_status = "complete"
        else:
            manual_status = "partial"
        human_comparisons = {
            judge_id: score_agreement(
                manual_scores,
                score_maps[judge_id],
                reference_label="human",
                candidate_label=judge_id,
            )
            for judge_id in experiment.judge_ids
        }
        pairwise_comparisons = {
            f"{left}::{right}": score_agreement(
                score_maps[left],
                score_maps[right],
                reference_label=left,
                candidate_label=right,
            )
            for index, left in enumerate(experiment.judge_ids)
            for right in experiment.judge_ids[index + 1 :]
        }
        run_manifest_hash = hash_file(run_manifest_path)
        report_status: Literal["complete", "incomplete"] = run_manifest.status
        report_id = stable_id(
            "judge_comparison_report",
            run_manifest.judge_experiment_run_id,
            run_manifest_hash,
            run_manifest.pool_manifest_id,
            run_manifest.pool_manifest_hash,
            experiment.judge_ids,
            manual_hash,
            manual_status,
            len(manual_scores),
            human_comparisons,
            pairwise_comparisons,
            report_status,
        )
        report = JudgeComparisonReport(
            judge_comparison_report_id=report_id,
            judge_experiment_run_id=run_manifest.judge_experiment_run_id,
            judge_experiment_run_manifest_path=repo_relative_path(
                run_manifest_path,
                root,
            ),
            judge_experiment_run_manifest_hash=run_manifest_hash,
            suite_run_id=run_manifest.suite_run_id,
            pool_manifest_id=run_manifest.pool_manifest_id,
            pool_manifest_hash=run_manifest.pool_manifest_hash,
            judge_ids=list(experiment.judge_ids),
            manual_judgments_path=(
                repo_relative_path(manual_path, root)
                if manual_path is not None
                else None
            ),
            manual_judgments_hash=manual_hash,
            manual_reference_status=manual_status,
            manual_reference_count=len(manual_scores),
            human_comparisons=human_comparisons,
            pairwise_comparisons=pairwise_comparisons,
            status=report_status,
            created_at_utc=_now_utc(),
        )
        report_path = immutable_judge_comparison_report_path(
            run_manifest_path.parent,
            report_id,
        )
        if report_path.is_file():
            existing = JudgeComparisonReport.from_dict(
                read_json_object(report_path, error_type=RelevanceError),
            )
            require_compatible(
                existing.judge_comparison_report_id,
                report_id,
                label="content-addressed judge comparison report",
                error_type=RelevanceError,
            )
            report = existing
        else:
            write_json_object_atomic(report_path, report.to_dict(), validate=True)
        return JudgeComparisonSummary(
            suite_run_id=run_manifest.suite_run_id,
            judge_experiment_id=experiment.judge_experiment_id,
            status=report.status,
            manual_reference_status=report.manual_reference_status,
            report_path=repo_relative_path(report_path, root),
            report_hash=hash_file(report_path),
        )
