"""Suite-owned, lineage-complete retrieval metric materialization."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Literal, Self

from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.counterpart_families import (
    CounterpartFamilyProjection,
    CounterpartFamilyProjectionRef,
    load_counterpart_family_projection,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import (
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    hash_jsonl_records,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.judge_experiment import (
    VerifiedJudgeExperimentViews,
    load_verified_judge_experiment_views,
)
from thesis_rag.eval.relevance import build_metrics_for_qrel_view
from thesis_rag.eval.relevance_schemas import (
    QrelResolutionRecord,
    QrelView,
    RelevanceJudgmentRecord,
)
from thesis_rag.eval.relevance_types import RelevanceProgressCallback
from thesis_rag.eval.runner import RetrievalRunManifest
from thesis_rag.eval.suite import ExperimentSuiteManifest, exclusive_suite_lock
from thesis_rag.eval.suite_config import (
    SuiteResolution,
    resolve_experiment_suite,
)
from thesis_rag.eval.suite_pool import (
    ExperimentSuitePoolJob,
    PreparedSuitePool,
    SuitePoolManifest,
    require_materialized_suite_pool,
)
from thesis_rag.eval.suite_qrel_review import (
    ExperimentSuiteQrelReviewJob,
    MaterializedManualQrelView,
)
from thesis_rag.schema import (
    LEGACY_SUITE_METRICS_MANIFEST_SCHEMA_VERSION,
    SUITE_LATENCY_RECORD_SCHEMA_VERSION,
    SUITE_METRIC_GROUP_SCHEMA_VERSION,
    SUITE_METRICS_MANIFEST_SCHEMA_VERSION,
    SUITE_QUERY_METRIC_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    stable_id,
)


SUITE_METRICS_VERSION = "suite_metrics_v3"
QUERY_SLICE_VERSION = "question_metadata_slices_v2"
LATENCY_SUMMARY_VERSION = "linear_quantiles_v1"
MANUAL_VIEW_LABEL = "manual_primary"

_QUERY_METRIC_NAMES = (
    "ndcg",
    "recall_at_grade_2",
    "hit_at_grade_2",
    "strict_hit_at_grade_3",
    "mrr_at_grade_2",
    "strict_mrr_at_grade_3",
    "reference_anchor_hit",
    "reference_anchor_set_hit",
    "reference_anchor_fact_coverage",
    "counterpart_family_ndcg",
    "counterpart_family_recall_at_grade_2",
    "counterpart_family_hit_at_grade_2",
    "counterpart_family_mrr_at_grade_2",
    "counterpart_duplicate_query_rate",
    "counterpart_redundant_slots",
    "counterpart_redundant_slot_rate",
    "relevant_counterpart_redundant_slots",
    "relevant_counterpart_redundant_slot_rate",
    "unique_relevant_page_families",
    "covered_reference_anchor_count",
    "covered_reference_fact_count",
)


class SuiteMetricsError(ValueError):
    """Raised when exact suite metrics cannot be safely materialized."""


class SuiteJudgmentViewRef(ArtifactModel):
    """Complete immutable QREL evidence view selected for one metric bundle."""

    view_label: NonEmptyStr
    view_kind: Literal[
        "anchor",
        "manual",
        "llm",
        "llm_judged",
        "llm_hybrid",
    ]
    view_id: NonEmptyStr
    owner_id: NonEmptyStr
    manifest_path: NonEmptyStr
    manifest_hash: Sha256
    judgments_path: NonEmptyStr
    judgments_hash: Sha256
    candidate_count: PositiveInt


class SuiteMetricRunRef(ArtifactModel):
    """One exact retrieval run consumed by suite metrics."""

    experiment_id: NonEmptyStr
    run_id: NonEmptyStr
    system_id: NonEmptyStr
    run_manifest_path: NonEmptyStr
    run_manifest_hash: Sha256
    retrieval_results_hash: Sha256
    latency_path: NonEmptyStr
    latency_hash: Sha256
    selected_query_ids_hash: Sha256


class SuiteMetricGroupRecord(ArtifactModel):
    """Aggregated metrics for one experiment/ranking/target/judgment view."""

    metric_record_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    question_artifact_hash: Sha256
    experiment_id: NonEmptyStr
    run_id: NonEmptyStr
    run_manifest_hash: Sha256
    system_id: NonEmptyStr
    split: Literal["development", "validation", "test"]
    stream: NonEmptyStr
    target_space_id: NonEmptyStr
    judgment_view_label: NonEmptyStr
    judgment_view_id: NonEmptyStr
    judgment_manifest_hash: Sha256
    metric_policy_hash: Sha256
    cutoffs: list[PositiveInt]
    query_count: PositiveInt
    query_ids_hash: Sha256
    metrics: JsonDict
    corpus_hash: Sha256
    schema_version: Literal["suite_metric_group_v1"] = (
        SUITE_METRIC_GROUP_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if self.cutoffs != sorted(set(self.cutoffs)):
            raise ValueError("suite metric cutoffs must be sorted and unique")
        expected = stable_id(
            "suite_metric_group",
            self.suite_run_id,
            self.pool_manifest_id,
            self.question_artifact_hash,
            self.experiment_id,
            self.run_id,
            self.run_manifest_hash,
            self.split,
            self.stream,
            self.target_space_id,
            self.judgment_view_label,
            self.judgment_view_id,
            self.judgment_manifest_hash,
            self.metric_policy_hash,
            self.query_ids_hash,
            self.metrics,
        )
        if self.metric_record_id != expected:
            raise ValueError("suite metric record identity mismatch")
        return self


class SuiteQueryMetricRecord(ArtifactModel):
    """Per-query values retained for paired comparisons and declared slices."""

    query_metric_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    experiment_id: NonEmptyStr
    run_id: NonEmptyStr
    split: Literal["development", "validation", "test"]
    stream: NonEmptyStr
    target_space_id: NonEmptyStr
    judgment_view_label: NonEmptyStr
    judgment_view_id: NonEmptyStr
    metric_policy_hash: Sha256
    query_id: NonEmptyStr
    query_slices: list[NonEmptyStr]
    metrics: JsonDict
    schema_version: Literal["suite_query_metric_v1"] = (
        SUITE_QUERY_METRIC_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if self.query_slices != sorted(set(self.query_slices)):
            raise ValueError("query slices must be sorted and unique")
        expected = stable_id(
            "suite_query_metric",
            self.suite_run_id,
            self.pool_manifest_id,
            self.experiment_id,
            self.run_id,
            self.split,
            self.stream,
            self.target_space_id,
            self.judgment_view_label,
            self.judgment_view_id,
            self.metric_policy_hash,
            self.query_id,
            self.query_slices,
            self.metrics,
        )
        if self.query_metric_id != expected:
            raise ValueError("suite query metric identity mismatch")
        return self


class SuiteLatencyRecord(ArtifactModel):
    """Pinned aggregate of one online timing series from a retrieval run."""

    latency_record_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    run_id: NonEmptyStr
    run_manifest_hash: Sha256
    latency_artifact_hash: Sha256
    stage_index: NonNegativeInt | None
    stage: NonEmptyStr
    timing_name: NonEmptyStr
    observation_count: PositiveInt
    query_count: PositiveInt
    duration_ms: JsonDict
    schema_version: Literal["suite_latency_record_v1"] = (
        SUITE_LATENCY_RECORD_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        expected = stable_id(
            "suite_latency_record",
            self.suite_run_id,
            self.experiment_id,
            self.run_id,
            self.run_manifest_hash,
            self.latency_artifact_hash,
            self.stage_index,
            self.stage,
            self.timing_name,
            self.observation_count,
            self.query_count,
            self.duration_ms,
        )
        if self.latency_record_id != expected:
            raise ValueError("suite latency record identity mismatch")
        return self


class SuiteMetricsManifest(ArtifactModel):
    """Commit-last manifest for one content-addressed metric bundle."""

    metrics_bundle_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    suite_manifest_path: NonEmptyStr
    suite_manifest_hash: Sha256
    suite_resolution_hash: Sha256
    question_set_id: NonEmptyStr
    question_manifest_hash: Sha256
    question_artifact_hash: Sha256
    corpus_hash: Sha256
    pool_manifest_id: NonEmptyStr
    pool_manifest_path: NonEmptyStr
    pool_manifest_hash: Sha256
    relevance_config_path: NonEmptyStr
    relevance_config_hash: Sha256
    metric_policy_hash: Sha256
    metrics_implementation_hash: Sha256
    counterpart_family_projection: CounterpartFamilyProjectionRef | None = None
    judgment_views: list[SuiteJudgmentViewRef]
    retrieval_runs: list[SuiteMetricRunRef]
    output_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    output_artifact_hashes: dict[NonEmptyStr, Sha256]
    counts: JsonDict
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    created_at_utc: NonEmptyStr
    status: Literal["complete"] = "complete"
    schema_version: Literal[
        "suite_metrics_manifest_v1",
        "suite_metrics_manifest_v2",
    ] = (
        SUITE_METRICS_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if (
            self.schema_version == SUITE_METRICS_MANIFEST_SCHEMA_VERSION
            and self.counterpart_family_projection is None
        ):
            raise ValueError(
                "suite metrics v2 requires a counterpart-family projection",
            )
        if (
            self.schema_version == LEGACY_SUITE_METRICS_MANIFEST_SCHEMA_VERSION
            and self.counterpart_family_projection is not None
        ):
            raise ValueError(
                "legacy suite metrics must not declare counterpart-family inputs",
            )
        required = {"metric_groups", "query_metrics", "latency"}
        if set(self.output_artifact_paths) != required:
            raise ValueError("suite metrics manifest output set is incomplete")
        if set(self.output_artifact_hashes) != required:
            raise ValueError("suite metrics manifest output hashes are incomplete")
        view_labels = [view.view_label for view in self.judgment_views]
        if not view_labels or len(view_labels) != len(set(view_labels)):
            raise ValueError("suite metric judgment views must be nonempty and unique")
        experiment_ids = [run.experiment_id for run in self.retrieval_runs]
        if not experiment_ids or len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("suite metric retrieval runs must be nonempty and unique")
        expected = _metrics_bundle_id(
            suite_run_id=self.suite_run_id,
            suite_manifest_hash=self.suite_manifest_hash,
            suite_resolution_hash=self.suite_resolution_hash,
            question_set_id=self.question_set_id,
            question_manifest_hash=self.question_manifest_hash,
            question_artifact_hash=self.question_artifact_hash,
            corpus_hash=self.corpus_hash,
            pool_manifest_id=self.pool_manifest_id,
            pool_manifest_hash=self.pool_manifest_hash,
            relevance_config_hash=self.relevance_config_hash,
            metric_policy_hash=self.metric_policy_hash,
            metrics_implementation_hash=self.metrics_implementation_hash,
            judgment_views=self.judgment_views,
            retrieval_runs=self.retrieval_runs,
            counterpart_family_projection=self.counterpart_family_projection,
        )
        if self.metrics_bundle_id != expected:
            raise ValueError("suite metrics bundle identity mismatch")
        return self


@dataclass(frozen=True)
class SuiteMetricsSummary:
    suite_run_id: str
    metrics_bundle_id: str
    judgment_view_count: int
    metric_group_count: int
    query_metric_count: int
    latency_record_count: int
    manifest_path: str
    reused: bool


@dataclass(frozen=True)
class MaterializedSuiteMetrics:
    manifest: SuiteMetricsManifest
    manifest_path: Path
    metric_groups: tuple[SuiteMetricGroupRecord, ...]
    query_metrics: tuple[SuiteQueryMetricRecord, ...]
    latency_records: tuple[SuiteLatencyRecord, ...]


@dataclass(frozen=True)
class _JudgmentSource:
    ref: SuiteJudgmentViewRef
    qrel_view: QrelView
    judgments: tuple[RelevanceJudgmentRecord, ...]


@dataclass(frozen=True)
class _PreparedMetrics:
    resolution: SuiteResolution
    suite_manifest: ExperimentSuiteManifest
    suite_manifest_path: Path
    pool_manifest: SuitePoolManifest
    pool_manifest_path: Path
    relevance_path: Path
    metric_policy_hash: str
    implementation_hash: str
    git_commit: str | None
    git_dirty: bool
    git_state_hash: str
    counterpart_families: CounterpartFamilyProjection
    judgment_views: tuple[SuiteJudgmentViewRef, ...]
    run_refs: tuple[SuiteMetricRunRef, ...]
    metric_groups: tuple[SuiteMetricGroupRecord, ...]
    query_metrics: tuple[SuiteQueryMetricRecord, ...]
    latency_records: tuple[SuiteLatencyRecord, ...]
    metrics_bundle_id: str


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _metrics_implementation_hash() -> str:
    from thesis_rag.corpus import counterpart_families as counterpart_families_module
    from thesis_rag.eval import counterpart_metrics as counterpart_metrics_module
    from thesis_rag.eval import context_metrics as context_metrics_module
    from thesis_rag.eval import relevance as relevance_module

    return hash_jsonable(
        {
            "suite_metrics": hash_file(Path(__file__)),
            "relevance_metrics": hash_file(Path(relevance_module.__file__)),
            "context_metrics": hash_file(Path(context_metrics_module.__file__)),
            "counterpart_families": hash_file(
                Path(counterpart_families_module.__file__),
            ),
            "counterpart_metrics": hash_file(
                Path(counterpart_metrics_module.__file__),
            ),
            "version": SUITE_METRICS_VERSION,
        },
    )


def _metrics_bundle_id(
    *,
    suite_run_id: str,
    suite_manifest_hash: str,
    suite_resolution_hash: str,
    question_set_id: str,
    question_manifest_hash: str,
    question_artifact_hash: str,
    corpus_hash: str,
    pool_manifest_id: str,
    pool_manifest_hash: str,
    relevance_config_hash: str,
    metric_policy_hash: str,
    metrics_implementation_hash: str,
    judgment_views: Sequence[SuiteJudgmentViewRef],
    retrieval_runs: Sequence[SuiteMetricRunRef],
    counterpart_family_projection: CounterpartFamilyProjectionRef | None = None,
) -> str:
    values: list[object] = [
        suite_run_id,
        suite_manifest_hash,
        suite_resolution_hash,
        question_set_id,
        question_manifest_hash,
        question_artifact_hash,
        corpus_hash,
        pool_manifest_id,
        pool_manifest_hash,
        relevance_config_hash,
        metric_policy_hash,
        metrics_implementation_hash,
        [view.to_dict() for view in judgment_views],
        [run.to_dict() for run in retrieval_runs],
    ]
    if counterpart_family_projection is not None:
        values.append(counterpart_family_projection.to_dict())
    return stable_id("suite_metrics_bundle", *values)


def _query_slices(question: object) -> list[str]:
    metadata = getattr(question, "metadata", {})
    values = {f"language:{getattr(question, 'language')}"}
    for key in (
        "primary_type",
        "surface_form",
        "query_framing",
        "derivation_kind",
        "source_language",
        "language_direction",
        "source_language_coverage",
        "translation_quality",
        "query_origin",
        "priority_cohort",
        "priority_strict",
        "nlu_training_overlap",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            values.add(f"{key}:{value}")
    traits = metadata.get("traits", [])
    if isinstance(traits, list):
        values.update(
            f"trait:{value}" for value in traits if isinstance(value, str) and value
        )
    return sorted(values)


def _slice_aggregates(records: Sequence[SuiteQueryMetricRecord]) -> JsonDict:
    by_slice: dict[str, list[SuiteQueryMetricRecord]] = defaultdict(list)
    for record in records:
        for slice_id in record.query_slices:
            by_slice[slice_id].append(record)
    result: JsonDict = {}
    for slice_id, slice_records in sorted(by_slice.items()):
        cutoffs = sorted(
            {
                cutoff
                for record in slice_records
                for cutoff in record.metrics.get("at_k", {})
            },
            key=int,
        )
        at_k: JsonDict = {}
        for cutoff in cutoffs:
            values: JsonDict = {}
            for metric_name in _QUERY_METRIC_NAMES:
                observed = [
                    item.metrics["at_k"][cutoff].get(metric_name)
                    for item in slice_records
                    if cutoff in item.metrics.get("at_k", {})
                    and isinstance(
                        item.metrics["at_k"][cutoff].get(metric_name),
                        (int, float),
                    )
                    and not isinstance(
                        item.metrics["at_k"][cutoff].get(metric_name),
                        bool,
                    )
                ]
                values[metric_name] = {
                    "count": len(observed),
                    "mean": mean(observed) if observed else None,
                }
            at_k[cutoff] = values
        query_ids = sorted(item.query_id for item in slice_records)
        result[slice_id] = {
            "query_count": len(query_ids),
            "query_ids_hash": hash_jsonable(query_ids),
            "at_k": at_k,
        }
    return result


def _qrel_resolution_diagnostics(
    prepared: PreparedSuitePool,
    resolutions: Sequence[QrelResolutionRecord],
    *,
    target_space_id: str,
    qrel_view: QrelView,
) -> JsonDict:
    """Summarize how reviewed anchors and judgments interact in one space."""

    projections = {
        item.projection_id: item for item in prepared.prepared.projections
    }
    counts: dict[str, int] = defaultdict(int)
    affected_queries: set[str] = set()
    for resolution in resolutions:
        if resolution.target.target_space_id != target_space_id:
            continue
        candidate_projections = [
            projections[projection_id]
            for projection_id in resolution.anchor_projection_ids
        ]
        if resolution.anchor_score_floor is not None:
            counts["anchor_backed_candidates"] += 1
        if any(
            item.projection_kind == "partial_overlap"
            for item in candidate_projections
        ):
            counts["partial_overlap_candidates"] += 1
        if any("disagreement" in flag for flag in resolution.conflict_flags):
            counts["anchor_judge_disagreements"] += 1
            affected_queries.add(resolution.query_id)
        if resolution.resolution == "anchor_floor_applied":
            counts["anchor_floors_applied"] += 1
            counts["anchor_floor_total_delta"] += resolution.floor_violation_delta
        elif resolution.resolution == "llm_upgrade":
            counts["judge_upgrades"] += 1
    return {
        **{
            name: counts[name]
            for name in (
                "anchor_backed_candidates",
                "partial_overlap_candidates",
                "anchor_judge_disagreements",
                "anchor_floors_applied",
                "anchor_floor_total_delta",
                "judge_upgrades",
            )
        },
        "affected_query_count": len(affected_queries),
        "effective_view": qrel_view,
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_latency_records(
    *,
    suite_run_id: str,
    run_ref: SuiteMetricRunRef,
    path: Path,
    query_ids: frozenset[str] | None = None,
) -> tuple[SuiteLatencyRecord, ...]:
    expected_columns = {
        "run_id",
        "query_id",
        "system_id",
        "attempt",
        "stage_index",
        "stage",
        "timing_name",
        "duration_ms",
    }
    grouped: dict[
        tuple[int | None, str, str], list[tuple[str, float]]
    ] = defaultdict(list)
    seen: set[tuple[str, int | None, str, str]] = set()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or ()) != expected_columns:
                raise SuiteMetricsError(
                    f"latency artifact has unexpected columns: {path}",
                )
            for row in reader:
                if (
                    row["run_id"] != run_ref.run_id
                    or row["system_id"] != run_ref.system_id
                ):
                    raise SuiteMetricsError(
                        "latency artifact contains foreign run rows",
                    )
                try:
                    stage_index = (
                        int(row["stage_index"]) if row["stage_index"] else None
                    )
                except ValueError as exc:
                    raise SuiteMetricsError("latency stage index is invalid") from exc
                if stage_index is not None and stage_index < 0:
                    raise SuiteMetricsError("latency stage index must be nonnegative")
                key = (
                    row["query_id"],
                    stage_index,
                    row["stage"],
                    row["timing_name"],
                )
                if key in seen:
                    raise SuiteMetricsError(f"duplicate latency row: {key}")
                seen.add(key)
                try:
                    duration = float(row["duration_ms"])
                except ValueError as exc:
                    raise SuiteMetricsError("latency duration is not numeric") from exc
                if not math.isfinite(duration) or duration < 0:
                    raise SuiteMetricsError(
                        "latency duration must be finite and nonnegative",
                    )
                if not row["query_id"] or not row["stage"] or not row["timing_name"]:
                    raise SuiteMetricsError("latency identity fields must not be empty")
                if query_ids is not None and row["query_id"] not in query_ids:
                    continue
                grouped[(stage_index, row["stage"], row["timing_name"])].append(
                    (row["query_id"], duration),
                )
    except OSError as exc:
        raise SuiteMetricsError(
            f"could not read latency artifact {path}: {exc}",
        ) from exc

    records: list[SuiteLatencyRecord] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -1 if item[0][0] is None else item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    for (stage_index, stage, timing_name), observations in ordered_groups:
        values = [value for _query_id, value in observations]
        duration_ms: JsonDict = {
            "summary_version": LATENCY_SUMMARY_VERSION,
            "minimum": min(values),
            "mean": mean(values),
            "median": median(values),
            "p95": _linear_quantile(values, 0.95),
            "p99": _linear_quantile(values, 0.99),
            "maximum": max(values),
            "total": sum(values),
        }
        latency_id = stable_id(
            "suite_latency_record",
            suite_run_id,
            run_ref.experiment_id,
            run_ref.run_id,
            run_ref.run_manifest_hash,
            run_ref.latency_hash,
            stage_index,
            stage,
            timing_name,
            len(observations),
            len({query_id for query_id, _value in observations}),
            duration_ms,
        )
        records.append(
            SuiteLatencyRecord(
                latency_record_id=latency_id,
                suite_run_id=suite_run_id,
                experiment_id=run_ref.experiment_id,
                run_id=run_ref.run_id,
                run_manifest_hash=run_ref.run_manifest_hash,
                latency_artifact_hash=run_ref.latency_hash,
                stage_index=stage_index,
                stage=stage,
                timing_name=timing_name,
                observation_count=len(observations),
                query_count=len({query_id for query_id, _value in observations}),
                duration_ms=duration_ms,
            ),
        )
    return tuple(records)


class ExperimentSuiteMetricsJob:
    """Compute complete immutable metrics for selected manual and judge views."""

    def __init__(
        self,
        suite_config_path: str | Path,
        *,
        relevance_config_path: str | Path | None = None,
        judge_experiment_path: str | Path | None = None,
        include_manual: bool = True,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.suite_config_path = Path(suite_config_path)
        self.relevance_config_path = (
            Path(relevance_config_path) if relevance_config_path is not None else None
        )
        self.judge_experiment_path = (
            Path(judge_experiment_path) if judge_experiment_path is not None else None
        )
        self.include_manual = include_manual
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(
        self,
        *,
        verify: bool = True,
        progress_callback: RelevanceProgressCallback | None = None,
    ) -> SuiteMetricsSummary:
        prepared = self._prepare(
            verify=verify,
            progress_callback=progress_callback,
        )
        with exclusive_suite_lock(prepared.resolution, operation="metrics"):
            materialized, reused = self._materialize(prepared)
            if any(e.final_evaluation.page_scopes is not None for e in prepared.suite_manifest.experiments):
                from thesis_rag.eval.routing_scopes import write_scope_metrics
                write_scope_metrics(materialized, repo_root=prepared.resolution.repo_root)
        return SuiteMetricsSummary(
            suite_run_id=materialized.manifest.suite_run_id,
            metrics_bundle_id=materialized.manifest.metrics_bundle_id,
            judgment_view_count=len(materialized.manifest.judgment_views),
            metric_group_count=len(materialized.metric_groups),
            query_metric_count=len(materialized.query_metrics),
            latency_record_count=len(materialized.latency_records),
            manifest_path=repo_relative_path(
                materialized.manifest_path,
                prepared.resolution.repo_root,
            ),
            reused=reused,
        )

    def require_materialized(
        self,
        *,
        verify: bool = True,
    ) -> MaterializedSuiteMetrics:
        """Recompute expected bytes and recursively verify the current bundle."""

        prepared = self._prepare(verify=verify, progress_callback=None)
        manifest_path = self._manifest_path(prepared)
        if not manifest_path.is_file():
            raise SuiteMetricsError(
                "suite report requires materialized metrics; run `eval suite metrics`",
            )
        return self._verify_materialized(prepared, manifest_path)

    def _prepare(
        self,
        *,
        verify: bool,
        progress_callback: RelevanceProgressCallback | None,
    ) -> _PreparedMetrics:
        root = resolve_repo_root(self.suite_config_path, self.repo_root)
        verified_judges: VerifiedJudgeExperimentViews | None = None
        if self.judge_experiment_path is not None:
            verified_judges = load_verified_judge_experiment_views(
                self.suite_config_path,
                self.judge_experiment_path,
                suite_output_root=self.suite_output_root,
                repo_root=root,
                require_complete=True,
            )
            relevance_path = resolve_repo_path(
                verified_judges.experiment.relevance_config_path,
                root,
            )
            if self.relevance_config_path is not None and (
                resolve_repo_path(self.relevance_config_path, root).resolve()
                != relevance_path.resolve()
            ):
                raise SuiteMetricsError(
                    "explicit relevance config differs from the judge experiment",
                )
        else:
            relevance_path = resolve_repo_path(
                self.relevance_config_path or "configs/thesis/relevance.yaml",
                root,
            )
        resolution = resolve_experiment_suite(
            self.suite_config_path,
            repo_root=root,
            suite_output_root=self.suite_output_root,
        )
        pool_job = ExperimentSuitePoolJob(
            self.suite_config_path,
            relevance_config_path=relevance_path,
            suite_output_root=self.suite_output_root,
            repo_root=root,
        )
        resolved_pool = pool_job.prepare_resolved(
            resolution,
            verify=verify,
            progress_callback=progress_callback,
        )
        pool_manifest, pool_manifest_path = require_materialized_suite_pool(
            resolved_pool,
        )
        if verified_judges is not None and (
            verified_judges.pool_manifest.pool_manifest_id
            != pool_manifest.pool_manifest_id
            or hash_file(verified_judges.pool_manifest_path)
            != hash_file(pool_manifest_path)
        ):
            raise SuiteMetricsError("judge experiment and metrics use different pools")

        sources: list[_JudgmentSource] = [
            self._anchor_source(pool_manifest, pool_manifest_path, root),
        ]
        if self.include_manual:
            manual = ExperimentSuiteQrelReviewJob(
                self.suite_config_path,
                relevance_config_path=relevance_path,
                suite_output_root=self.suite_output_root,
                repo_root=root,
            ).current_materialized_view()
            if manual is None:
                raise SuiteMetricsError(
                    "manual metrics require a complete current suite Human-QREL view; "
                    "run qrel-materialize or select --no-manual",
                )
            sources.append(self._manual_source(manual, pool_manifest, root))
        if verified_judges is not None:
            sources.extend(self._judge_sources(verified_judges, root))
        labels = [source.ref.view_label for source in sources]
        if len(labels) != len(set(labels)):
            raise SuiteMetricsError("judgment view labels collide")
        for source in sources:
            self._validate_judgment_source(resolved_pool, source)

        counterpart_families = load_counterpart_family_projection(
            resolved_pool.prepared.state.snapshot_manifest,
            repo_root=root,
        )

        metric_policy_hash = hash_jsonable(
            {
                "suite_metrics_version": SUITE_METRICS_VERSION,
                "query_slice_version": QUERY_SLICE_VERSION,
                "latency_summary_version": LATENCY_SUMMARY_VERSION,
                "metrics": resolved_pool.prepared.state.config.metrics.to_dict(),
                "relevance_threshold": 2,
                "strict_relevance_threshold": 3,
                "judgment_policy": "anchor_judged_hybrid_views_v2",
                "counterpart_family_projection": (
                    counterpart_families.ref.to_dict()
                ),
            },
        )
        implementation_hash = _metrics_implementation_hash()
        run_refs = self._run_refs(resolved_pool.manifest, root)
        run_ref_by_id = {run.run_id: run for run in run_refs}
        experiment_order = {
            experiment.experiment_id: index
            for index, experiment in enumerate(resolved_pool.manifest.experiments)
        }
        questions = resolved_pool.prepared.state.questions_by_id
        completeness_resolutions = {
            (result.run_id, result.query_id): result.debug.get("verified_legacy_stage_completeness", {})
            for result in resolved_pool.prepared.state.retrieval_results
        }
        metric_groups: list[SuiteMetricGroupRecord] = []
        query_metrics: list[SuiteQueryMetricRecord] = []
        for source in sources:
            low_groups, low_queries, resolutions = build_metrics_for_qrel_view(
                resolved_pool.prepared,
                source.judgments,
                qrel_view=source.qrel_view,
                qrel_view_id=source.ref.view_id,
                progress_callback=progress_callback,
                counterpart_page_families=(
                    counterpart_families.family_by_page_id
                ),
            )
            low_query_models: list[SuiteQueryMetricRecord] = []
            for raw in low_queries:
                run_ref = run_ref_by_id[str(raw["run_id"])]
                query_id = str(raw["query_id"])
                metrics = {
                    key: value
                    for key, value in raw.items()
                    if key
                    not in {
                        "run_id",
                        "system_id",
                        "split",
                        "stream",
                        "target_space_id",
                        "qrel_view",
                        "query_id",
                    }
                }
                if completeness_resolution := completeness_resolutions.get((run_ref.run_id, query_id), {}).get(raw["stream"]):
                    metrics["stage_completeness_resolution"] = completeness_resolution
                slices = _query_slices(questions[query_id])
                query_metric_id = stable_id(
                    "suite_query_metric",
                    resolved_pool.manifest.suite_run_id,
                    pool_manifest.pool_manifest_id,
                    run_ref.experiment_id,
                    run_ref.run_id,
                    raw["split"],
                    raw["stream"],
                    raw["target_space_id"],
                    source.ref.view_label,
                    source.ref.view_id,
                    metric_policy_hash,
                    query_id,
                    slices,
                    metrics,
                )
                low_query_models.append(
                    SuiteQueryMetricRecord(
                        query_metric_id=query_metric_id,
                        suite_run_id=resolved_pool.manifest.suite_run_id,
                        pool_manifest_id=pool_manifest.pool_manifest_id,
                        experiment_id=run_ref.experiment_id,
                        run_id=run_ref.run_id,
                        split=raw["split"],
                        stream=str(raw["stream"]),
                        target_space_id=str(raw["target_space_id"]),
                        judgment_view_label=source.ref.view_label,
                        judgment_view_id=source.ref.view_id,
                        metric_policy_hash=metric_policy_hash,
                        query_id=query_id,
                        query_slices=slices,
                        metrics=metrics,
                    ),
                )
            query_metrics.extend(low_query_models)
            query_by_signature: dict[
                tuple[str, str, str, str], list[SuiteQueryMetricRecord]
            ] = defaultdict(list)
            for item in low_query_models:
                query_by_signature[
                    (item.run_id, item.split, item.stream, item.target_space_id)
                ].append(item)
            for low in low_groups:
                run_ref = run_ref_by_id[low.run_id]
                signature = (low.run_id, low.split, low.stream, low.target_space_id)
                metrics = {
                    **low.metrics,
                    "qrel_resolution": _qrel_resolution_diagnostics(
                        resolved_pool,
                        resolutions,
                        target_space_id=low.target_space_id,
                        qrel_view=source.qrel_view,
                    ),
                    "query_slices": _slice_aggregates(query_by_signature[signature]),
                }
                metric_id = stable_id(
                    "suite_metric_group",
                    resolved_pool.manifest.suite_run_id,
                    pool_manifest.pool_manifest_id,
                    resolved_pool.manifest.question_set.questions_hash,
                    run_ref.experiment_id,
                    run_ref.run_id,
                    run_ref.run_manifest_hash,
                    low.split,
                    low.stream,
                    low.target_space_id,
                    source.ref.view_label,
                    source.ref.view_id,
                    source.ref.manifest_hash,
                    metric_policy_hash,
                    low.query_ids_hash,
                    metrics,
                )
                metric_groups.append(
                    SuiteMetricGroupRecord(
                        metric_record_id=metric_id,
                        suite_run_id=resolved_pool.manifest.suite_run_id,
                        pool_manifest_id=pool_manifest.pool_manifest_id,
                        question_artifact_hash=(
                            resolved_pool.manifest.question_set.questions_hash
                        ),
                        experiment_id=run_ref.experiment_id,
                        run_id=run_ref.run_id,
                        run_manifest_hash=run_ref.run_manifest_hash,
                        system_id=low.system_id,
                        split=low.split,
                        stream=low.stream,
                        target_space_id=low.target_space_id,
                        judgment_view_label=source.ref.view_label,
                        judgment_view_id=source.ref.view_id,
                        judgment_manifest_hash=source.ref.manifest_hash,
                        metric_policy_hash=metric_policy_hash,
                        cutoffs=low.cutoffs,
                        query_count=low.query_count,
                        query_ids_hash=low.query_ids_hash,
                        metrics=metrics,
                        corpus_hash=low.corpus_hash,
                    ),
                )
        metric_groups.sort(
            key=lambda item: (
                item.judgment_view_label,
                experiment_order[item.experiment_id],
                item.split,
                item.stream,
                item.target_space_id,
            ),
        )
        query_metrics.sort(
            key=lambda item: (
                item.judgment_view_label,
                experiment_order[item.experiment_id],
                item.split,
                item.stream,
                item.target_space_id,
                item.query_id,
            ),
        )
        latency_records = tuple(
            record
            for run_ref in run_refs
            for record in _read_latency_records(
                suite_run_id=resolved_pool.manifest.suite_run_id,
                run_ref=run_ref,
                path=resolve_repo_path(run_ref.latency_path, root),
                query_ids=frozenset(
                    result.query_id
                    for result in resolved_pool.prepared.state.retrieval_results
                    if result.run_id == run_ref.run_id
                ),
            )
        )
        git_commit, git_dirty, git_state_hash = capture_git_state(root)
        view_refs = tuple(source.ref for source in sources)
        suite_manifest_hash = hash_file(resolved_pool.manifest_path)
        bundle_id = _metrics_bundle_id(
            suite_run_id=resolved_pool.manifest.suite_run_id,
            suite_manifest_hash=suite_manifest_hash,
            suite_resolution_hash=resolved_pool.manifest.resolution_hash,
            question_set_id=resolved_pool.manifest.question_set.question_set_id,
            question_manifest_hash=resolved_pool.manifest.question_set.manifest_hash,
            question_artifact_hash=(
                resolved_pool.manifest.question_set.questions_hash
            ),
            corpus_hash=resolved_pool.manifest.corpus.corpus_hash,
            pool_manifest_id=pool_manifest.pool_manifest_id,
            pool_manifest_hash=hash_file(pool_manifest_path),
            relevance_config_hash=hash_file(relevance_path),
            metric_policy_hash=metric_policy_hash,
            metrics_implementation_hash=implementation_hash,
            judgment_views=view_refs,
            retrieval_runs=run_refs,
            counterpart_family_projection=counterpart_families.ref,
        )
        return _PreparedMetrics(
            resolution=resolution,
            suite_manifest=resolved_pool.manifest,
            suite_manifest_path=resolved_pool.manifest_path,
            pool_manifest=pool_manifest,
            pool_manifest_path=pool_manifest_path,
            relevance_path=relevance_path,
            metric_policy_hash=metric_policy_hash,
            implementation_hash=implementation_hash,
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_state_hash=git_state_hash,
            counterpart_families=counterpart_families,
            judgment_views=view_refs,
            run_refs=run_refs,
            metric_groups=tuple(metric_groups),
            query_metrics=tuple(query_metrics),
            latency_records=latency_records,
            metrics_bundle_id=bundle_id,
        )

    @staticmethod
    def _manual_source(
        manual: MaterializedManualQrelView,
        pool_manifest: SuitePoolManifest,
        root: Path,
    ) -> _JudgmentSource:
        if (
            manual.manifest.pool_manifest_id != pool_manifest.pool_manifest_id
            or manual.manifest.pool_manifest_hash
            != hash_file(resolve_repo_path(manual.manifest.pool_manifest_path, root))
        ):
            raise SuiteMetricsError("manual judgment view belongs to another pool")
        return _JudgmentSource(
            ref=SuiteJudgmentViewRef(
                view_label=MANUAL_VIEW_LABEL,
                view_kind="manual",
                view_id=manual.manifest.manual_qrel_view_id,
                owner_id=manual.manifest.reviewer_id,
                manifest_path=repo_relative_path(manual.manifest_path, root),
                manifest_hash=hash_file(manual.manifest_path),
                judgments_path=manual.manifest.judgments_path,
                judgments_hash=manual.manifest.judgments_hash,
                candidate_count=manual.manifest.candidate_count,
            ),
            qrel_view="judged_only",
            judgments=manual.judgments,
        )

    @staticmethod
    def _anchor_source(
        pool_manifest: SuitePoolManifest,
        pool_manifest_path: Path,
        root: Path,
    ) -> _JudgmentSource:
        projections_path = pool_manifest.output_artifact_paths[
            "anchor_projections"
        ]
        projections_hash = pool_manifest.output_artifact_hashes[
            "anchor_projections"
        ]
        view_id = stable_id(
            "suite_anchor_qrel_view",
            pool_manifest.pool_manifest_id,
            projections_hash,
        )
        return _JudgmentSource(
            ref=SuiteJudgmentViewRef(
                view_label="anchor_only",
                view_kind="anchor",
                view_id=view_id,
                owner_id="reviewed_source_anchors",
                manifest_path=repo_relative_path(pool_manifest_path, root),
                manifest_hash=hash_file(pool_manifest_path),
                judgments_path=projections_path,
                judgments_hash=projections_hash,
                candidate_count=int(pool_manifest.counts["candidate_pool"]),
            ),
            qrel_view="anchor_only",
            judgments=(),
        )

    @staticmethod
    def _judge_sources(
        verified: VerifiedJudgeExperimentViews,
        root: Path,
    ) -> list[_JudgmentSource]:
        sources: list[_JudgmentSource] = []
        for view in verified.views:
            common = {
                "owner_id": view.result.semantic_judge_id,
                "manifest_path": repo_relative_path(view.manifest_path, root),
                "manifest_hash": hash_file(view.manifest_path),
                "judgments_path": repo_relative_path(view.judgments_path, root),
                "judgments_hash": view.manifest.judgments_hash,
                "candidate_count": view.manifest.candidate_count,
            }
            for qrel_view, kind in (
                ("judged_only", "llm_judged"),
                ("hybrid", "llm_hybrid"),
            ):
                sources.append(
                    _JudgmentSource(
                        ref=SuiteJudgmentViewRef(
                            view_label=f"{view.result.judge_id}:{qrel_view}",
                            view_kind=kind,
                            view_id=stable_id(
                                "suite_llm_qrel_view",
                                view.manifest.judgment_view_id,
                                qrel_view,
                            ),
                            **common,
                        ),
                        qrel_view=qrel_view,
                        judgments=view.judgments,
                    ),
                )
        return sources

    @staticmethod
    def _validate_judgment_source(
        resolved_pool: PreparedSuitePool,
        source: _JudgmentSource,
    ) -> None:
        state = resolved_pool.prepared.state
        candidates = {
            candidate.key: candidate
            for candidate in resolved_pool.prepared.candidates
        }
        expected_keys = {
            (*key, state.config.relevance_guideline_version) for key in candidates
        }
        if source.qrel_view == "anchor_only":
            if source.judgments:
                raise SuiteMetricsError("anchor-only view contains judgments")
            if source.ref.candidate_count != len(candidates):
                raise SuiteMetricsError("anchor-only candidate count differs from pool")
            return
        records = {record.key: record for record in source.judgments}
        if len(records) != len(source.judgments) or set(records) != expected_keys:
            raise SuiteMetricsError(
                f"judgment view {source.ref.view_label!r} is not pool-complete",
            )
        for key, record in records.items():
            candidate = candidates[key[:4]]
            if (
                record.query_text_hash != candidate.query_text_hash
                or record.target != candidate.target
                or record.corpus_hash != candidate.corpus_hash
                or record.guideline_version
                != state.config.relevance_guideline_version
            ):
                raise SuiteMetricsError(
                    f"judgment view {source.ref.view_label!r} changed active inputs",
                )
        if source.ref.candidate_count != len(candidates):
            raise SuiteMetricsError("judgment view candidate count differs from pool")

    @staticmethod
    def _run_refs(
        manifest: ExperimentSuiteManifest,
        root: Path,
    ) -> tuple[SuiteMetricRunRef, ...]:
        experiments = {item.experiment_id: item for item in manifest.experiments}
        refs: list[SuiteMetricRunRef] = []
        for state in manifest.experiment_runs:
            if (
                state.status != "complete"
                or state.run_id is None
                or state.run_manifest_path is None
                or state.run_manifest_hash is None
            ):
                raise SuiteMetricsError("suite metrics require complete retrieval runs")
            path = resolve_repo_path(state.run_manifest_path, root)
            if not path.is_file() or hash_file(path) != state.run_manifest_hash:
                raise SuiteMetricsError("retrieval run manifest changed")
            run = RetrievalRunManifest.from_dict(
                read_json_object(path, error_type=SuiteMetricsError),
            )
            latency_path = resolve_repo_path(run.output_path, root) / "latency.csv"
            latency_hash = run.output_hashes.get("latency")
            retrieval_hash = run.output_hashes.get("retrieval_results")
            experiment = experiments[state.experiment_id]
            if (
                run.run_id != state.run_id
                or run.identity.selected_query_ids_hash
                != experiment.selected_query_ids_hash
                or latency_hash is None
                or retrieval_hash is None
                or not latency_path.is_file()
                or hash_file(latency_path) != latency_hash
            ):
                raise SuiteMetricsError(
                    f"retrieval artifacts changed for {state.experiment_id}",
                )
            refs.append(
                SuiteMetricRunRef(
                    experiment_id=state.experiment_id,
                    run_id=run.run_id,
                    system_id=run.identity.system_id,
                    run_manifest_path=repo_relative_path(path, root),
                    run_manifest_hash=state.run_manifest_hash,
                    retrieval_results_hash=retrieval_hash,
                    latency_path=repo_relative_path(latency_path, root),
                    latency_hash=latency_hash,
                    selected_query_ids_hash=run.identity.selected_query_ids_hash,
                ),
            )
        return tuple(refs)

    def _manifest_path(self, prepared: _PreparedMetrics) -> Path:
        return (
            prepared.suite_manifest_path.parent
            / "metrics"
            / prepared.metrics_bundle_id
            / "manifest.json"
        )

    def _materialize(
        self,
        prepared: _PreparedMetrics,
    ) -> tuple[MaterializedSuiteMetrics, bool]:
        manifest_path = self._manifest_path(prepared)
        if manifest_path.is_file():
            return self._verify_materialized(prepared, manifest_path), True
        output_dir = manifest_path.parent
        metric_groups_path = output_dir / "metric_groups.jsonl"
        query_metrics_path = output_dir / "query_metrics.jsonl"
        latency_path = output_dir / "latency.jsonl"
        bundle = ImmutableArtifactBundle(
            manifest_path=manifest_path,
            output_paths={
                "metric_groups": metric_groups_path,
                "query_metrics": query_metrics_path,
                "latency": latency_path,
            },
            label="suite metrics",
            error_type=SuiteMetricsError,
        )
        bundle.write_jsonl("metric_groups", prepared.metric_groups)
        bundle.write_jsonl("query_metrics", prepared.query_metrics)
        bundle.write_jsonl("latency", prepared.latency_records)
        root = prepared.resolution.repo_root
        output_paths = {
            "metric_groups": repo_relative_path(metric_groups_path, root),
            "query_metrics": repo_relative_path(query_metrics_path, root),
            "latency": repo_relative_path(latency_path, root),
        }
        output_hashes = bundle.output_hashes
        manifest = SuiteMetricsManifest(
            metrics_bundle_id=prepared.metrics_bundle_id,
            suite_run_id=prepared.suite_manifest.suite_run_id,
            suite_manifest_path=repo_relative_path(
                prepared.suite_manifest_path,
                root,
            ),
            suite_manifest_hash=hash_file(prepared.suite_manifest_path),
            suite_resolution_hash=prepared.suite_manifest.resolution_hash,
            question_set_id=(
                prepared.suite_manifest.question_set.question_set_id
            ),
            question_manifest_hash=(
                prepared.suite_manifest.question_set.manifest_hash
            ),
            question_artifact_hash=(
                prepared.suite_manifest.question_set.questions_hash
            ),
            corpus_hash=prepared.suite_manifest.corpus.corpus_hash,
            pool_manifest_id=prepared.pool_manifest.pool_manifest_id,
            pool_manifest_path=repo_relative_path(prepared.pool_manifest_path, root),
            pool_manifest_hash=hash_file(prepared.pool_manifest_path),
            relevance_config_path=repo_relative_path(prepared.relevance_path, root),
            relevance_config_hash=hash_file(prepared.relevance_path),
            metric_policy_hash=prepared.metric_policy_hash,
            metrics_implementation_hash=prepared.implementation_hash,
            counterpart_family_projection=prepared.counterpart_families.ref,
            judgment_views=list(prepared.judgment_views),
            retrieval_runs=list(prepared.run_refs),
            output_artifact_paths=output_paths,
            output_artifact_hashes=output_hashes,
            counts={
                "judgment_views": len(prepared.judgment_views),
                "retrieval_runs": len(prepared.run_refs),
                "metric_groups": len(prepared.metric_groups),
                "query_metrics": len(prepared.query_metrics),
                "latency_records": len(prepared.latency_records),
                "counterpart_relationships": (
                    prepared.counterpart_families.ref.counterpart_relationship_count
                ),
                "counterpart_families": (
                    prepared.counterpart_families.ref.counterpart_family_count
                ),
            },
            git_commit=prepared.git_commit,
            git_dirty=prepared.git_dirty,
            git_state_hash=prepared.git_state_hash,
            created_at_utc=_now_utc(),
        )
        bundle.commit(manifest)
        return self._verify_materialized(prepared, manifest_path), False

    @staticmethod
    def _verify_materialized(
        prepared: _PreparedMetrics,
        manifest_path: Path,
    ) -> MaterializedSuiteMetrics:
        try:
            manifest = SuiteMetricsManifest.from_dict(
                read_json_object(manifest_path, error_type=SuiteMetricsError),
            )
        except ValueError as exc:
            raise SuiteMetricsError(f"invalid suite metrics manifest: {exc}") from exc
        root = prepared.resolution.repo_root
        expected = {
            "metrics_bundle_id": prepared.metrics_bundle_id,
            "suite_run_id": prepared.suite_manifest.suite_run_id,
            "suite_manifest_path": repo_relative_path(
                prepared.suite_manifest_path,
                root,
            ),
            "suite_manifest_hash": hash_file(prepared.suite_manifest_path),
            "suite_resolution_hash": prepared.suite_manifest.resolution_hash,
            "question_set_id": (
                prepared.suite_manifest.question_set.question_set_id
            ),
            "question_manifest_hash": (
                prepared.suite_manifest.question_set.manifest_hash
            ),
            "question_artifact_hash": (
                prepared.suite_manifest.question_set.questions_hash
            ),
            "corpus_hash": prepared.suite_manifest.corpus.corpus_hash,
            "pool_manifest_id": prepared.pool_manifest.pool_manifest_id,
            "pool_manifest_path": repo_relative_path(
                prepared.pool_manifest_path,
                root,
            ),
            "pool_manifest_hash": hash_file(prepared.pool_manifest_path),
            "relevance_config_path": repo_relative_path(prepared.relevance_path, root),
            "relevance_config_hash": hash_file(prepared.relevance_path),
            "metric_policy_hash": prepared.metric_policy_hash,
            "metrics_implementation_hash": prepared.implementation_hash,
            "counterpart_family_projection": prepared.counterpart_families.ref,
            "judgment_views": list(prepared.judgment_views),
            "retrieval_runs": list(prepared.run_refs),
            "counts": {
                "judgment_views": len(prepared.judgment_views),
                "retrieval_runs": len(prepared.run_refs),
                "metric_groups": len(prepared.metric_groups),
                "query_metrics": len(prepared.query_metrics),
                "latency_records": len(prepared.latency_records),
                "counterpart_relationships": (
                    prepared.counterpart_families.ref.counterpart_relationship_count
                ),
                "counterpart_families": (
                    prepared.counterpart_families.ref.counterpart_family_count
                ),
            },
        }
        require_compatible(
            {key: getattr(manifest, key) for key in expected},
            expected,
            label="suite metrics manifest inputs",
            error_type=SuiteMetricsError,
        )
        artifact_references(
            manifest.output_artifact_paths,
            manifest.output_artifact_hashes,
            repo_root=root,
            label="suite metrics outputs",
            error_type=SuiteMetricsError,
        )
        expected_records = {
            "metric_groups": prepared.metric_groups,
            "query_metrics": prepared.query_metrics,
            "latency": prepared.latency_records,
        }
        # Actual file hashes were checked above. Match them against the canonical
        # computation without loading and copying a second complete metric set.
        for name, records in expected_records.items():
            expected_hash = hash_jsonl_records(record.to_dict() for record in records)
            require_compatible(
                manifest.output_artifact_hashes[name],
                expected_hash,
                label=f"suite metrics {name} canonical computation",
                error_type=SuiteMetricsError,
            )
        return MaterializedSuiteMetrics(
            manifest=manifest,
            manifest_path=manifest_path,
            metric_groups=prepared.metric_groups,
            query_metrics=prepared.query_metrics,
            latency_records=prepared.latency_records,
        )
