"""Deterministic comparison reports and explicit suite selection records."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    PositiveInt,
    Sha256,
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
from thesis_rag.eval.suite import exclusive_suite_lock
from thesis_rag.eval.suite_config import (
    ComparisonDefinition,
    SuiteResolution,
    resolve_experiment_suite,
)
from thesis_rag.eval.suite_metrics import (
    ExperimentSuiteMetricsJob,
    SuiteMetricGroupRecord,
    SuiteQueryMetricRecord,
)
from thesis_rag.eval.suite_metric_reader import (
    QueryMetricIndex,
    SuiteMetrics as MaterializedSuiteMetrics,
    load_suite_metrics,
)
from thesis_rag.schema import (
    SUITE_COMPARISON_RECORD_SCHEMA_VERSION,
    SUITE_REPORT_CONFIG_SCHEMA_VERSION,
    SUITE_REPORT_MANIFEST_SCHEMA_VERSION,
    SUITE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


SUITE_REPORT_VERSION = "suite_comparison_report_v1"
BOOTSTRAP_VERSION = "paired_query_bootstrap_mean_v1"

_PAIRED_METRICS = (
    "ndcg",
    "recall_at_grade_2",
    "hit_at_grade_2",
    "strict_hit_at_grade_3",
    "mrr_at_grade_2",
    "strict_mrr_at_grade_3",
    "counterpart_family_ndcg",
    "counterpart_family_recall_at_grade_2",
    "counterpart_family_hit_at_grade_2",
    "counterpart_family_mrr_at_grade_2",
    "counterpart_duplicate_query_rate",
    "counterpart_redundant_slot_rate",
    "relevant_counterpart_redundant_slot_rate",
)

_CONTEXT_CHARACTER_STREAM_PREFIX = "context_items@chars="


def _full_context_report_scope(stream: str) -> tuple[str, str]:
    """Return an honest label and metric suffix for one context stream."""

    if stream.startswith(_CONTEXT_CHARACTER_STREAM_PREFIX):
        budget = int(stream.removeprefix(_CONTEXT_CHARACTER_STREAM_PREFIX))
        return (
            "Complete budget-visible evidence prefix "
            f"(`up to {budget:,} characters`; diagnostic, not a selection metric):",
            "visible",
        )
    return (
        "Full materialized evidence context (`ALL`; diagnostic, not a "
        "selection metric):",
        "ALL",
    )


def _comparison_signature_sort_key(
    signature: tuple[str, str, str, str],
) -> tuple[str, str, int, int | str, str]:
    """Keep context budgets numeric and place unbounded context last."""

    view, split, stream, target_space = signature
    if stream.startswith(_CONTEXT_CHARACTER_STREAM_PREFIX):
        stream_group = 1
        stream_order: int | str = int(
            stream.removeprefix(_CONTEXT_CHARACTER_STREAM_PREFIX),
        )
    elif stream == "context_items":
        stream_group = 2
        stream_order = 0
    else:
        stream_group = 0
        stream_order = stream
    return view, split, stream_group, stream_order, target_space


class SuiteReportError(ValueError):
    """Raised when a suite report cannot be verified or materialized."""


class SelectionPolicy(ArtifactModel):
    """Metric evidence frozen before a later human selection decision."""

    policy_id: NonEmptyStr
    comparison_id: NonEmptyStr
    judgment_view_label: NonEmptyStr
    split: Literal["development", "validation", "test"]
    stream: NonEmptyStr
    target_space_id: NonEmptyStr
    metric_name: Literal[
        "ndcg",
        "recall_at_grade_2",
        "hit_at_grade_2",
        "mrr_at_grade_2",
        "counterpart_family_ndcg",
        "counterpart_family_recall_at_grade_2",
        "counterpart_family_hit_at_grade_2",
        "counterpart_family_mrr_at_grade_2",
    ] = "ndcg"
    cutoff: PositiveInt = 20
    direction: Literal["maximize", "minimize"] = "maximize"

    @model_validator(mode="after")
    def cutoff_is_supported(self) -> Self:
        if self.cutoff > 20:
            raise ValueError("selection metric cutoff must not exceed 20")
        return self


class PriorityPageReportConfig(ArtifactModel):
    manifest_path: NonEmptyStr
    manifest_hash: Sha256
    overview_experiment_ids: list[NonEmptyStr] = Field(default_factory=list)
    descendant_depth: int = Field(default=0, ge=0, le=5, exclude_if=lambda value: not value)


class SuiteReportConfig(ArtifactModel):
    """Prespecified comparison/report policy independent of retrieval identity."""

    report_config_id: NonEmptyStr
    comparison_ids: list[NonEmptyStr] = Field(default_factory=list)
    primary_judgment_view: NonEmptyStr | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    bootstrap_samples: int = Field(
        default=1000, ge=0, description="Zero disables confidence intervals.",
    )
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_seed: int = 14731
    selection_policies: list[SelectionPolicy] = Field(default_factory=list)
    priority_pages: PriorityPageReportConfig | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    schema_version: Literal["suite_report_config_v1"] = (
        SUITE_REPORT_CONFIG_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def values_are_unique(self) -> Self:
        if self.comparison_ids != list(dict.fromkeys(self.comparison_ids)):
            raise ValueError("report comparison IDs must be unique and ordered")
        policy_ids = [policy.policy_id for policy in self.selection_policies]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("selection policy IDs must be unique")
        if self.bootstrap_samples > 10_000:
            raise ValueError("bootstrap_samples must not exceed 10000")
        return self


class SelectionDecisionInput(ArtifactModel):
    policy_id: NonEmptyStr
    selected_experiment_ids: list[NonEmptyStr]
    rationale: NonEmptyStr
    decided_by: NonEmptyStr
    decided_at_utc: NonEmptyStr

    @model_validator(mode="after")
    def selection_is_nonempty(self) -> Self:
        if not self.selected_experiment_ids:
            raise ValueError("selection decision requires at least one experiment")
        if self.selected_experiment_ids != list(
            dict.fromkeys(self.selected_experiment_ids),
        ):
            raise ValueError("selected experiment IDs must be unique and ordered")
        _parse_utc(self.decided_at_utc)
        return self


class SuiteSelectionDecisionConfig(ArtifactModel):
    """Explicit post-evidence decision input; never inferred by the report."""

    metrics_bundle_id: NonEmptyStr
    report_config_hash: Sha256
    decisions: list[SelectionDecisionInput]
    schema_version: Literal["suite_selection_decision_config_v1"] = (
        SUITE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def decisions_are_unique(self) -> Self:
        policy_ids = [decision.policy_id for decision in self.decisions]
        if not policy_ids or len(policy_ids) != len(set(policy_ids)):
            raise ValueError("selection decisions must be nonempty and unique")
        return self


class SuiteComparisonRecord(ArtifactModel):
    """One declared comparison in one compatible metric-view signature."""

    comparison_record_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    metrics_bundle_id: NonEmptyStr
    comparison_id: NonEmptyStr
    comparison_kind: Literal["group", "baseline_candidates"]
    member_experiment_ids: list[NonEmptyStr]
    split: Literal["development", "validation", "test"]
    stream: NonEmptyStr
    target_space_id: NonEmptyStr
    judgment_view_label: NonEmptyStr
    judgment_view_id: NonEmptyStr
    metric_record_ids: dict[NonEmptyStr, NonEmptyStr]
    query_count: PositiveInt
    query_ids_hash: Sha256
    cutoffs: list[PositiveInt]
    aggregate_metrics: JsonDict
    paired_deltas: JsonDict
    schema_version: Literal["suite_comparison_record_v1"] = (
        SUITE_COMPARISON_RECORD_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if self.member_experiment_ids != list(
            dict.fromkeys(self.member_experiment_ids),
        ):
            raise ValueError("comparison members must be unique and ordered")
        if set(self.metric_record_ids) != set(self.member_experiment_ids):
            raise ValueError("comparison metric records must cover every member")
        if self.cutoffs != sorted(set(self.cutoffs)):
            raise ValueError("comparison cutoffs must be sorted and unique")
        expected = stable_id(
            "suite_comparison_record",
            self.suite_run_id,
            self.metrics_bundle_id,
            self.comparison_id,
            self.comparison_kind,
            self.member_experiment_ids,
            self.split,
            self.stream,
            self.target_space_id,
            self.judgment_view_label,
            self.judgment_view_id,
            self.metric_record_ids,
            self.query_ids_hash,
            self.cutoffs,
            self.aggregate_metrics,
            self.paired_deltas,
        )
        if self.comparison_record_id != expected:
            raise ValueError("suite comparison record identity mismatch")
        return self


class SelectionDecisionRecord(ArtifactModel):
    selection_record_id: NonEmptyStr
    policy: SelectionPolicy
    selected_experiment_ids: list[NonEmptyStr]
    evidence_metric_record_ids: dict[NonEmptyStr, NonEmptyStr]
    evidence_values: dict[NonEmptyStr, float | None]
    rationale: NonEmptyStr
    decided_by: NonEmptyStr
    decided_at_utc: NonEmptyStr

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if not set(self.selected_experiment_ids) <= set(
            self.evidence_metric_record_ids,
        ):
            raise ValueError("selected experiments lack metric evidence")
        if set(self.evidence_values) != set(self.evidence_metric_record_ids):
            raise ValueError("selection evidence values and records differ")
        _parse_utc(self.decided_at_utc)
        expected = stable_id(
            "suite_selection_record",
            self.policy.to_dict(),
            self.selected_experiment_ids,
            self.evidence_metric_record_ids,
            self.evidence_values,
            self.rationale,
            self.decided_by,
            self.decided_at_utc,
        )
        if self.selection_record_id != expected:
            raise ValueError("suite selection record identity mismatch")
        return self


class SuiteReportManifest(ArtifactModel):
    """Commit-last index of one immutable comparison/selection report."""

    report_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    suite_manifest_path: NonEmptyStr
    suite_manifest_hash: Sha256
    metrics_bundle_id: NonEmptyStr
    metrics_manifest_path: NonEmptyStr
    metrics_manifest_hash: Sha256
    report_config_id: NonEmptyStr
    report_config_path: NonEmptyStr | None
    report_config_hash: Sha256
    decision_config_path: NonEmptyStr | None
    decision_config_hash: Sha256 | None
    report_implementation_hash: Sha256
    comparison_record_ids: list[NonEmptyStr]
    selection_records: list[SelectionDecisionRecord]
    output_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    output_artifact_hashes: dict[NonEmptyStr, Sha256]
    counts: JsonDict
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    created_at_utc: NonEmptyStr
    status: Literal["comparison_only", "selection_recorded"]
    schema_version: Literal["suite_report_manifest_v1"] = (
        SUITE_REPORT_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if (self.decision_config_path is None) != (
            self.decision_config_hash is None
        ):
            raise ValueError("decision config path and hash must be atomic")
        if set(self.output_artifact_paths) != {"comparisons", "markdown"}:
            raise ValueError("suite report requires comparison and Markdown outputs")
        if set(self.output_artifact_hashes) != {"comparisons", "markdown"}:
            raise ValueError("suite report output hashes are incomplete")
        if self.comparison_record_ids != sorted(set(self.comparison_record_ids)):
            raise ValueError("comparison record IDs must be sorted and unique")
        expected_status = (
            "selection_recorded" if self.selection_records else "comparison_only"
        )
        if self.status != expected_status:
            raise ValueError("suite report status disagrees with decisions")
        expected = _report_id(
            suite_run_id=self.suite_run_id,
            suite_manifest_hash=self.suite_manifest_hash,
            metrics_bundle_id=self.metrics_bundle_id,
            metrics_manifest_hash=self.metrics_manifest_hash,
            report_config_hash=self.report_config_hash,
            decision_config_hash=self.decision_config_hash,
            implementation_hash=self.report_implementation_hash,
            comparison_record_ids=self.comparison_record_ids,
            selection_records=self.selection_records,
        )
        if self.report_id != expected:
            raise ValueError("suite report identity mismatch")
        return self


@dataclass(frozen=True)
class SuiteReportSummary:
    suite_run_id: str
    report_id: str
    status: Literal["comparison_only", "selection_recorded"]
    comparison_record_count: int
    selection_record_count: int
    report_path: str
    markdown_path: str
    reused: bool
    page_scope_report_path: str | None = None


@dataclass(frozen=True)
class MaterializedSuiteReport:
    manifest: SuiteReportManifest
    manifest_path: Path
    comparisons: tuple[SuiteComparisonRecord, ...]
    metrics: MaterializedSuiteMetrics
    report_config: SuiteReportConfig


@dataclass(frozen=True)
class _PreparedReport:
    resolution: SuiteResolution
    metrics: MaterializedSuiteMetrics
    report_config: SuiteReportConfig
    report_config_path: Path | None
    report_config_hash: str
    decision_config: SuiteSelectionDecisionConfig | None
    decision_config_path: Path | None
    decision_config_hash: str | None
    implementation_hash: str
    comparison_records: tuple[SuiteComparisonRecord, ...]
    selection_records: tuple[SelectionDecisionRecord, ...]
    markdown: str
    git_commit: str | None
    git_dirty: bool
    git_state_hash: str
    report_id: str


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _report_implementation_hash() -> str:
    from thesis_rag.eval import suite_metric_reader

    return hash_jsonable(
        {
            "suite_report": hash_file(Path(__file__)),
            "metric_reader": hash_file(Path(suite_metric_reader.__file__)),
            "version": SUITE_REPORT_VERSION,
            "bootstrap_version": BOOTSTRAP_VERSION,
        },
    )


def _report_id(
    *,
    suite_run_id: str,
    suite_manifest_hash: str,
    metrics_bundle_id: str,
    metrics_manifest_hash: str,
    report_config_hash: str,
    decision_config_hash: str | None,
    implementation_hash: str,
    comparison_record_ids: Sequence[str],
    selection_records: Sequence[SelectionDecisionRecord],
) -> str:
    return stable_id(
        "suite_comparison_report",
        suite_run_id,
        suite_manifest_hash,
        metrics_bundle_id,
        metrics_manifest_hash,
        report_config_hash,
        decision_config_hash,
        implementation_hash,
        list(comparison_record_ids),
        [record.to_dict() for record in selection_records],
    )


def load_suite_report_config(path: str | Path) -> SuiteReportConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteReportError(f"could not read report config {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SuiteReportError("suite report config must be a YAML mapping")
    try:
        return SuiteReportConfig.from_dict(payload)
    except ValueError as exc:
        raise SuiteReportError(f"invalid suite report config: {exc}") from exc


def load_selection_decision_config(
    path: str | Path,
) -> SuiteSelectionDecisionConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteReportError(
            f"could not read selection decision config {source}: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise SuiteReportError("selection decision config must be a YAML mapping")
    try:
        return SuiteSelectionDecisionConfig.from_dict(payload)
    except ValueError as exc:
        raise SuiteReportError(f"invalid selection decision config: {exc}") from exc


def _default_report_config(
    comparisons: Sequence[ComparisonDefinition],
) -> SuiteReportConfig:
    return SuiteReportConfig(
        report_config_id="suite_default_comparison_report",
        comparison_ids=[comparison.comparison_id for comparison in comparisons],
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> JsonDict | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    size = len(values)
    means = [
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "version": BOOTSTRAP_VERSION,
        "samples": samples,
        "confidence_level": confidence,
        "lower": _quantile(means, alpha),
        "upper": _quantile(means, 1.0 - alpha),
        "seed": seed,
        "resampling_unit": "query",
    }


def _comparison_pairs(
    comparison: ComparisonDefinition,
) -> list[tuple[str, str]]:
    if comparison.baseline_experiment_id is not None:
        return [
            (comparison.baseline_experiment_id, candidate)
            for candidate in comparison.candidate_experiment_ids
        ]
    return list(combinations(comparison.experiment_ids, 2))


def _matching_policies(
    config: SuiteReportConfig,
    *,
    comparison_id: str,
    signature: tuple[str, str, str, str],
) -> set[tuple[str, int]]:
    view_label, split, stream, target_space = signature
    selected = {
        (policy.metric_name, policy.cutoff)
        for policy in config.selection_policies
        if policy.comparison_id == comparison_id
        and policy.judgment_view_label == view_label
        and policy.split == split
        and policy.stream == stream
        and policy.target_space_id == target_space
    }
    return selected


def _primary_items_cutoff(
    config: SuiteReportConfig,
    *,
    judgment_view_label: str,
    split: str,
) -> int | None:
    """Return the one prespecified primary items cutoff, when unambiguous."""

    cutoffs = {
        policy.cutoff
        for policy in config.selection_policies
        if policy.judgment_view_label == judgment_view_label
        and policy.split == split
        and policy.stream == "items"
    }
    return next(iter(cutoffs)) if len(cutoffs) == 1 else None


def _comparison_display_cutoff(
    comparison: SuiteComparisonRecord,
    config: SuiteReportConfig,
) -> int:
    """Use the primary retrieval cutoff for items and full depth elsewhere."""

    if comparison.stream == "items":
        primary = _primary_items_cutoff(
            config,
            judgment_view_label=comparison.judgment_view_label,
            split=comparison.split,
        )
        if primary in comparison.cutoffs:
            return primary
    return max(comparison.cutoffs)


def _paired_delta(
    left: Mapping[str, SuiteQueryMetricRecord],
    right: Mapping[str, SuiteQueryMetricRecord],
    *,
    cutoff: int | None,
    metric_name: str,
    bootstrap: bool,
    config: SuiteReportConfig,
    seed_payload: JsonDict,
) -> JsonDict:
    values: list[tuple[str, float, float]] = []
    for query_id in sorted(set(left) & set(right)):
        left_metrics = left[query_id].metrics
        right_metrics = right[query_id].metrics
        left_scope = (
            left_metrics["full_context"]
            if cutoff is None
            else left_metrics["at_k"][str(cutoff)]
        )
        right_scope = (
            right_metrics["full_context"]
            if cutoff is None
            else right_metrics["at_k"][str(cutoff)]
        )
        left_value = left_scope.get(metric_name)
        right_value = right_scope.get(metric_name)
        if (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
        ):
            values.append((query_id, float(left_value), float(right_value)))
    deltas = [right_value - left_value for _query, left_value, right_value in values]
    seed = int(hash_jsonable(seed_payload).removeprefix("sha256:")[:16], 16)
    return {
        "delta_direction": "right_minus_left",
        "paired_query_count": len(values),
        "paired_query_ids_hash": hash_jsonable(
            [query for query, _left, _right in values],
        ),
        "left_mean": (
            mean(value for _query, value, _right in values) if values else None
        ),
        "right_mean": (
            mean(value for _query, _left, value in values) if values else None
        ),
        "mean_delta": mean(deltas) if deltas else None,
        "right_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "right_losses": sum(delta < 0 for delta in deltas),
        "bootstrap_interval": (
            _bootstrap_interval(
                deltas,
                samples=config.bootstrap_samples,
                confidence=config.confidence_level,
                seed=seed,
            )
            if bootstrap
            else None
        ),
    }


def _build_comparison_records(
    metrics: MaterializedSuiteMetrics,
    comparisons: Sequence[ComparisonDefinition],
    config: SuiteReportConfig,
) -> tuple[SuiteComparisonRecord, ...]:
    groups_by_experiment: dict[
        str, dict[tuple[str, str, str, str], SuiteMetricGroupRecord]
    ] = defaultdict(dict)
    queries = QueryMetricIndex(metrics, _comparison_target_space_id)
    for group in metrics.metric_groups:
        signature = (
            group.judgment_view_label,
            group.split,
            group.stream,
            _comparison_target_space_id(group.target_space_id),
        )
        if signature in groups_by_experiment[group.experiment_id]:
            raise SuiteReportError("duplicate suite metric group signature")
        groups_by_experiment[group.experiment_id][signature] = group
    records: list[SuiteComparisonRecord] = []
    for comparison in comparisons:
        members = comparison.member_ids()
        signatures = set(groups_by_experiment[members[0]])
        for member in members[1:]:
            signatures &= set(groups_by_experiment[member])
        if not signatures:
            raise SuiteReportError(
                f"declared comparison {comparison.comparison_id!r} has no "
                "compatible metric view",
            )
        for signature in sorted(signatures, key=_comparison_signature_sort_key):
            view_label, split, stream, target_space = signature
            if view_label == "anchor_only":
                # Anchors are deliberately incomplete provenance evidence, not a
                # complete QREL view for system-quality comparisons.
                continue
            if target_space == "context_reference_anchors":
                # Source-anchor recovery is benchmark provenance diagnostics,
                # not a system-neutral relevance or context-sufficiency view.
                continue
            groups = {
                member: groups_by_experiment[member][signature]
                for member in members
            }
            query_hashes = {group.query_ids_hash for group in groups.values()}
            query_counts = {group.query_count for group in groups.values()}
            view_ids = {group.judgment_view_id for group in groups.values()}
            cutoffs = {tuple(group.cutoffs) for group in groups.values()}
            if (
                len(query_hashes) != 1
                or len(query_counts) != 1
                or len(view_ids) != 1
                or len(cutoffs) != 1
            ):
                raise SuiteReportError(
                    f"comparison {comparison.comparison_id!r} mixes incompatible "
                    "cohorts, views, or cutoffs",
                )
            configured_cutoffs = list(next(iter(cutoffs)))
            active_cutoffs = [
                cutoff
                for cutoff in configured_cutoffs
                if all(
                    group.metrics["at_k"][str(cutoff)].get(
                        "qrel_metrics_computable",
                    )
                    is True
                    for group in groups.values()
                )
            ]
            if not active_cutoffs:
                raise SuiteReportError(
                    f"comparison {comparison.comparison_id!r} has no commonly "
                    f"evaluable cutoff for {signature}",
                )
            default_metric = {
                "context_native_mixed": "mrr_at_grade_2",
            }.get(target_space, "ndcg")
            bootstrap_metrics = _matching_policies(
                config,
                comparison_id=comparison.comparison_id,
                signature=signature,
            )
            if not bootstrap_metrics:
                primary_items_cutoff = (
                    _primary_items_cutoff(
                        config,
                        judgment_view_label=view_label,
                        split=split,
                    )
                    if stream == "items"
                    else None
                )
                bootstrap_metrics = {
                    (
                        default_metric,
                        primary_items_cutoff
                        if primary_items_cutoff in active_cutoffs
                        else max(active_cutoffs),
                    ),
                }
            paired: JsonDict = {}
            has_full_context = target_space == "context_native_mixed" and all(
                isinstance(group.metrics.get("full_context"), dict)
                and group.metrics["full_context"].get("qrel_metrics_computable")
                is True
                for group in groups.values()
            )
            for left, right in _comparison_pairs(comparison):
                pair_key = f"{left}::{right}"
                left_queries = queries[(left, *signature)]
                right_queries = queries[(right, *signature)]
                if set(left_queries) != set(right_queries):
                    raise SuiteReportError(
                        f"paired comparison {pair_key} has different query IDs",
                    )
                at_k: JsonDict = {}
                for cutoff in active_cutoffs:
                    at_k[str(cutoff)] = {
                        metric_name: _paired_delta(
                            left_queries,
                            right_queries,
                            cutoff=cutoff,
                            metric_name=metric_name,
                            bootstrap=config.bootstrap_samples > 0
                            and (metric_name, cutoff) in bootstrap_metrics,
                            config=config,
                            seed_payload={
                                "base_seed": config.bootstrap_seed,
                                "comparison": comparison.comparison_id,
                                "signature": signature,
                                "pair": pair_key,
                                "cutoff": cutoff,
                                "metric": metric_name,
                            },
                        )
                        for metric_name in _PAIRED_METRICS
                    }
                pair_metrics: JsonDict = {"at_k": at_k}
                if has_full_context:
                    pair_metrics["full_context"] = {
                        metric_name: _paired_delta(
                            left_queries,
                            right_queries,
                            cutoff=None,
                            metric_name=metric_name,
                            bootstrap=config.bootstrap_samples > 0
                            and metric_name == "mrr_at_grade_2",
                            config=config,
                            seed_payload={
                                "base_seed": config.bootstrap_seed,
                                "comparison": comparison.comparison_id,
                                "signature": signature,
                                "pair": pair_key,
                                "scope": "full_context",
                                "metric": metric_name,
                            },
                        )
                        for metric_name in _PAIRED_METRICS
                    }
                paired[pair_key] = pair_metrics
            aggregate = {
                member: {
                    "source_target_space_id": groups[member].target_space_id,
                    "at_k": groups[member].metrics["at_k"],
                    **(
                        {"full_context": groups[member].metrics["full_context"]}
                        if has_full_context
                        else {}
                    ),
                    "query_slices": groups[member].metrics["query_slices"],
                    "relevant_target_count_at_grade_2": groups[
                        member
                    ].metrics.get("relevant_target_count_at_grade_2"),
                    "relevant_target_count_at_grade_3": groups[
                        member
                    ].metrics.get("relevant_target_count_at_grade_3"),
                    "judgment_score_counts": groups[member].metrics.get(
                        "judgment_score_counts",
                        {},
                    ),
                    "context_diagnostics": {
                        name: groups[member].metrics.get(name)
                        for name in (
                            "mean_visible_character_count",
                            "partial_item_query_rate",
                            "mean_context_item_count",
                            "direct_intent_query_rate",
                            "parent_expansion_query_rate",
                            "routing_hint_query_rate",
                            "mean_routing_hint_character_count",
                        )
                        if name in groups[member].metrics
                    },
                }
                for member in members
            }
            kind: Literal["group", "baseline_candidates"] = (
                "baseline_candidates"
                if comparison.baseline_experiment_id is not None
                else "group"
            )
            values = {
                "suite_run_id": metrics.manifest.suite_run_id,
                "metrics_bundle_id": metrics.manifest.metrics_bundle_id,
                "comparison_id": comparison.comparison_id,
                "comparison_kind": kind,
                "member_experiment_ids": members,
                "split": split,
                "stream": stream,
                "target_space_id": target_space,
                "judgment_view_label": view_label,
                "judgment_view_id": next(iter(view_ids)),
                "metric_record_ids": {
                    member: groups[member].metric_record_id for member in members
                },
                "query_count": next(iter(query_counts)),
                "query_ids_hash": next(iter(query_hashes)),
                "cutoffs": active_cutoffs,
                "aggregate_metrics": aggregate,
                "paired_deltas": paired,
            }
            comparison_id = stable_id(
                "suite_comparison_record",
                values["suite_run_id"],
                values["metrics_bundle_id"],
                values["comparison_id"],
                values["comparison_kind"],
                values["member_experiment_ids"],
                values["split"],
                values["stream"],
                values["target_space_id"],
                values["judgment_view_label"],
                values["judgment_view_id"],
                values["metric_record_ids"],
                values["query_ids_hash"],
                values["cutoffs"],
                values["aggregate_metrics"],
                values["paired_deltas"],
            )
            records.append(
                SuiteComparisonRecord(
                    comparison_record_id=comparison_id,
                    **values,
                ),
            )
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.comparison_id,
                item.judgment_view_label,
                item.split,
                item.stream,
                item.target_space_id,
            ),
        ),
    )


def _comparison_target_space_id(target_space_id: str) -> str:
    """Return the semantic comparison level while preserving source records.

    Chunk identities deliberately include the chunker because chunks from two
    representations are not interchangeable QREL targets.  Retrieval-system
    selection nevertheless compares their per-query metric values at the shared
    semantic level ``chunk``.  The exact source target space remains bound by
    ``metric_record_ids`` and ``source_target_space_id`` in the aggregate.
    """

    return "chunk" if target_space_id.startswith("chunk:") else target_space_id


def _build_selection_records(
    *,
    metrics: MaterializedSuiteMetrics,
    comparisons: Sequence[SuiteComparisonRecord],
    report_config: SuiteReportConfig,
    report_config_hash: str,
    decision_config: SuiteSelectionDecisionConfig | None,
) -> tuple[SelectionDecisionRecord, ...]:
    if decision_config is None:
        return ()
    if (
        decision_config.metrics_bundle_id != metrics.manifest.metrics_bundle_id
        or decision_config.report_config_hash != report_config_hash
    ):
        raise SuiteReportError(
            "selection decision does not pin the current metrics/report policy",
        )
    policies = {policy.policy_id: policy for policy in report_config.selection_policies}
    records: list[SelectionDecisionRecord] = []
    for decision in decision_config.decisions:
        policy = policies.get(decision.policy_id)
        if policy is None:
            raise SuiteReportError(
                f"selection decision references unknown policy {decision.policy_id!r}",
            )
        matches = [
            comparison
            for comparison in comparisons
            if comparison.comparison_id == policy.comparison_id
            and comparison.judgment_view_label == policy.judgment_view_label
            and comparison.split == policy.split
            and comparison.stream == policy.stream
            and comparison.target_space_id == policy.target_space_id
        ]
        if len(matches) != 1:
            raise SuiteReportError(
                f"selection policy {policy.policy_id!r} has no unique evidence view",
            )
        comparison = matches[0]
        if not set(decision.selected_experiment_ids) <= set(
            comparison.member_experiment_ids,
        ):
            raise SuiteReportError(
                "selection contains an experiment outside comparison",
            )
        cutoff = str(policy.cutoff)
        try:
            evidence_values = {
                experiment_id: comparison.aggregate_metrics[experiment_id]["at_k"]
                [cutoff][policy.metric_name]
                for experiment_id in comparison.member_experiment_ids
            }
        except KeyError as exc:
            raise SuiteReportError(
                f"selection policy {policy.policy_id!r} requests unavailable "
                "metric evidence",
            ) from exc
        evidence_records = dict(comparison.metric_record_ids)
        selection_id = stable_id(
            "suite_selection_record",
            policy.to_dict(),
            decision.selected_experiment_ids,
            evidence_records,
            evidence_values,
            decision.rationale,
            decision.decided_by,
            decision.decided_at_utc,
        )
        records.append(
            SelectionDecisionRecord(
                selection_record_id=selection_id,
                policy=policy,
                selected_experiment_ids=decision.selected_experiment_ids,
                evidence_metric_record_ids=evidence_records,
                evidence_values=evidence_values,
                rationale=decision.rationale,
                decided_by=decision.decided_by,
                decided_at_utc=decision.decided_at_utc,
            ),
        )
    return tuple(sorted(records, key=lambda item: item.policy.policy_id))


def _validate_selection_policies(
    policies: Sequence[SelectionPolicy],
    comparisons: Sequence[SuiteComparisonRecord],
) -> None:
    """Fail closed when a frozen policy has no exact report evidence."""

    for policy in policies:
        matches = [
            comparison
            for comparison in comparisons
            if comparison.comparison_id == policy.comparison_id
            and comparison.judgment_view_label == policy.judgment_view_label
            and comparison.split == policy.split
            and comparison.stream == policy.stream
            and comparison.target_space_id == policy.target_space_id
        ]
        if len(matches) != 1:
            raise SuiteReportError(
                f"selection policy {policy.policy_id!r} has no unique metric view",
            )
        cutoff = str(policy.cutoff)
        if any(
            cutoff not in matches[0].aggregate_metrics[experiment_id]["at_k"]
            or policy.metric_name
            not in matches[0].aggregate_metrics[experiment_id]["at_k"][cutoff]
            for experiment_id in matches[0].member_experiment_ids
        ):
            raise SuiteReportError(
                f"selection policy {policy.policy_id!r} requests unavailable "
                "metric evidence",
            )


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "—" if value is None else str(value)


def _render_markdown(
    *,
    metrics: MaterializedSuiteMetrics,
    comparisons: Sequence[SuiteComparisonRecord],
    selections: Sequence[SelectionDecisionRecord],
    report_config: SuiteReportConfig,
) -> str:
    lines = [
        "# Retrieval Suite Comparison Report",
        "",
        f"- Suite: `{metrics.manifest.suite_run_id}`",
        f"- Metrics bundle: `{metrics.manifest.metrics_bundle_id}`",
        "- Judgment views: "
        + ", ".join(view.view_label for view in metrics.manifest.judgment_views),
        "- Pool-relative metrics use relevance score >= 2; strict diagnostics "
        "use score 3.",
        "",
    ]
    if not comparisons:
        lines.extend(
            [
                "No retrieval comparison groups are declared in this suite.",
                "",
            ],
        )
    active_comparison_id: str | None = None
    for comparison in comparisons:
        cutoff = str(_comparison_display_cutoff(comparison, report_config))
        first_values = comparison.aggregate_metrics[
            comparison.member_experiment_ids[0]
        ]["at_k"][cutoff]
        counterpart_aware = (
            comparison.target_space_id == "page"
            and "counterpart_family_ndcg" in first_values
        )
        mixed_context = comparison.target_space_id == "context_native_mixed"
        if mixed_context:
            table_header = (
                f"| Experiment | MRR@{cutoff} | Hit@{cutoff} | "
                "Strict hit | Mean chars | Partial rate | Routing-hint rate | "
                "Mean routing-hint chars |"
            )
            table_divider = (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            primary_metric = "mrr_at_grade_2"
        elif counterpart_aware:
            table_header = (
                f"| Experiment | Raw nDCG@{cutoff} | "
                f"Counterpart-aware nDCG@{cutoff} | "
                f"Counterpart-aware recall@{cutoff} | Duplicate queries | "
                "Redundant slots |"
            )
            table_divider = "| --- | ---: | ---: | ---: | ---: | ---: |"
            primary_metric = "ndcg"
        else:
            table_header = (
                f"| Experiment | nDCG@{cutoff} | MRR@{cutoff} | "
                f"Hit@{cutoff} | Pool recall@{cutoff} |"
            )
            table_divider = "| --- | ---: | ---: | ---: | ---: |"
            primary_metric = "ndcg"
        if comparison.comparison_id != active_comparison_id:
            lines.extend([f"## {comparison.comparison_id}", ""])
            active_comparison_id = comparison.comparison_id
        lines.extend(
            [
                f"### View `{comparison.judgment_view_label}` · target "
                f"`{comparison.target_space_id}`",
                "",
                f"Split `{comparison.split}` · stream `{comparison.stream}` · "
                f"{comparison.query_count} queries.",
                "",
                table_header,
                table_divider,
            ],
        )
        for experiment_id in comparison.member_experiment_ids:
            aggregate = comparison.aggregate_metrics[experiment_id]
            values = aggregate["at_k"][cutoff]
            diagnostics = aggregate.get("context_diagnostics", {})
            if mixed_context:
                displayed = (
                    experiment_id,
                    _format_metric(values["mrr_at_grade_2"]),
                    _format_metric(values["hit_at_grade_2"]),
                    _format_metric(values["strict_hit_at_grade_3"]),
                    _format_metric(values.get("prefix_character_count")),
                    _format_metric(diagnostics.get("partial_item_query_rate")),
                    _format_metric(diagnostics.get("routing_hint_query_rate")),
                    _format_metric(
                        diagnostics.get("mean_routing_hint_character_count"),
                    ),
                )
            elif counterpart_aware:
                displayed = (
                    experiment_id,
                    _format_metric(values["ndcg"]),
                    _format_metric(values["counterpart_family_ndcg"]),
                    _format_metric(
                        values["counterpart_family_recall_at_grade_2"],
                    ),
                    _format_metric(values["counterpart_duplicate_query_rate"]),
                    _format_metric(values["counterpart_redundant_slots"]),
                )
            else:
                displayed = (
                    experiment_id,
                    _format_metric(values["ndcg"]),
                    _format_metric(values["mrr_at_grade_2"]),
                    _format_metric(values["hit_at_grade_2"]),
                    _format_metric(values["recall_at_grade_2"]),
                )
            lines.append(
                "| "
                + " | ".join(displayed)
                + " |",
            )
        lines.append("")
        has_full_context = mixed_context and all(
            "full_context" in comparison.aggregate_metrics[experiment_id]
            for experiment_id in comparison.member_experiment_ids
        )
        if has_full_context:
            full_scope_description, full_scope_suffix = _full_context_report_scope(
                comparison.stream,
            )
            lines.extend(
                [
                    full_scope_description,
                    "",
                    f"| Experiment | MRR@{full_scope_suffix} | "
                    f"Hit@{full_scope_suffix} | Strict hit@{full_scope_suffix} | "
                    f"Mean chars@{full_scope_suffix} |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ],
            )
            for experiment_id in comparison.member_experiment_ids:
                full = comparison.aggregate_metrics[experiment_id]["full_context"]
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            experiment_id,
                            _format_metric(full.get("mrr_at_grade_2")),
                            _format_metric(full.get("hit_at_grade_2")),
                            _format_metric(full.get("strict_hit_at_grade_3")),
                            _format_metric(full.get("prefix_character_count")),
                        ),
                    )
                    + " |",
                )
            lines.append("")
        for pair_id, pair in comparison.paired_deltas.items():
            delta = pair["at_k"][cutoff][primary_metric]
            interval = delta["bootstrap_interval"]
            interval_text = (
                f" [{interval['lower']:.4f}, {interval['upper']:.4f}]"
                if interval is not None
                else ""
            )
            lines.append(
                f"- `{pair_id}`: paired {primary_metric}@{cutoff} delta "
                f"{_format_metric(delta['mean_delta'])}{interval_text} "
                f"({delta['paired_query_count']} paired queries).",
            )
            if counterpart_aware:
                family_delta = pair["at_k"][cutoff][
                    "counterpart_family_ndcg"
                ]
                family_interval = family_delta["bootstrap_interval"]
                family_interval_text = (
                    f" [{family_interval['lower']:.4f}, "
                    f"{family_interval['upper']:.4f}]"
                    if family_interval is not None
                    else ""
                )
                lines.append(
                    f"- `{pair_id}`: paired counterpart-aware nDCG@{cutoff} "
                    "delta "
                    f"{_format_metric(family_delta['mean_delta'])}"
                    f"{family_interval_text} "
                    f"({family_delta['paired_query_count']} paired queries).",
                )
            if has_full_context and "full_context" in pair:
                full_delta = pair["full_context"]["mrr_at_grade_2"]
                full_interval = full_delta["bootstrap_interval"]
                full_interval_text = (
                    f" [{full_interval['lower']:.4f}, "
                    f"{full_interval['upper']:.4f}]"
                    if full_interval is not None
                    else ""
                )
                lines.append(
                    "- "
                    f"`{pair_id}`: paired diagnostic mrr_at_grade_2@"
                    f"{full_scope_suffix} "
                    f"delta {_format_metric(full_delta['mean_delta'])}"
                    f"{full_interval_text} "
                    f"({full_delta['paired_query_count']} paired queries).",
                )
        lines.extend(
            [
                "",
                "Per-query values and query-metadata slices remain available "
                "in the bound JSONL artifacts.",
                "",
            ],
        )

    online_latency = [
        item
        for item in metrics.latency_records
        if item.stage == "pipeline"
        and item.timing_name == "online_retrieval_total"
    ]
    online_experiments = {item.experiment_id for item in online_latency}
    pipeline_latency = online_latency + [
        item
        for item in metrics.latency_records
        if item.stage == "pipeline"
        and item.timing_name == "total"
        and item.experiment_id not in online_experiments
    ]
    if pipeline_latency:
        lines.extend(
            [
                "## Online Retrieval Latency",
                "",
                "The comparison uses the system-owned online retrieval boundary; "
                "evaluation-wall timing is diagnostic only.",
                "",
                "| Experiment | Median ms | p95 ms | p99 ms | Mean ms |",
                "| --- | ---: | ---: | ---: | ---: |",
            ],
        )
        for item in pipeline_latency:
            lines.append(
                f"| {item.experiment_id} | {item.duration_ms['median']:.3f} | "
                f"{item.duration_ms['p95']:.3f} | {item.duration_ms['p99']:.3f} | "
                f"{item.duration_ms['mean']:.3f} |",
            )
        lines.append("")

    lines.extend(["## Selection Record", ""])
    if not selections:
        lines.extend(
            [
                "No empirical selection decision is recorded in this report.",
                "",
            ],
        )
    else:
        for selection in selections:
            lines.extend(
                [
                    f"- Policy `{selection.policy.policy_id}` selected "
                    + ", ".join(
                        f"`{item}`"
                        for item in selection.selected_experiment_ids
                    )
                    + ".",
                    f"  Rationale: {selection.rationale}",
                    f"  Decided by `{selection.decided_by}` at "
                    f"`{selection.decided_at_utc}`.",
                ],
            )
        lines.append("")
    return "\n".join(lines)


class ExperimentSuiteReportJob:
    """Consume one exact suite metric bundle into declared comparisons."""

    def __init__(
        self,
        suite_config_path: str | Path,
        *,
        relevance_config_path: str | Path | None = None,
        judge_experiment_path: str | Path | None = None,
        include_manual: bool = True,
        report_config_path: str | Path | None = None,
        decision_config_path: str | Path | None = None,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.suite_config_path = Path(suite_config_path)
        self.relevance_config_path = relevance_config_path
        self.judge_experiment_path = judge_experiment_path
        self.include_manual = include_manual
        self.report_config_path = (
            Path(report_config_path) if report_config_path is not None else None
        )
        self.decision_config_path = (
            Path(decision_config_path) if decision_config_path is not None else None
        )
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(self, *, verify: bool = True, analysis: bool = False) -> SuiteReportSummary:
        prepared = self._prepare(verify=verify)
        with exclusive_suite_lock(prepared.resolution, operation="report"):
            manifest, manifest_path, reused = self._materialize(prepared)
            from thesis_rag.eval.routing_scopes import write_scope_metrics
            from thesis_rag.eval.routing_scope_report import scope_report_id, write_scope_report

            scope_manifest = write_scope_metrics(prepared.metrics, repo_root=prepared.resolution.repo_root,
                                                 require_existing=True)
            scope_report = None
            if scope_manifest is not None:
                priority_config = prepared.report_config.priority_pages
                report_identity = scope_report_id(scope_manifest, priority_config=priority_config, plots=analysis)
                scope_report = write_scope_report(scope_manifest, manifest_path.parent / report_identity,
                                                  repo_root=prepared.resolution.repo_root, plots=analysis,
                                                  priority_config=priority_config,
                                                  metrics_manifest_path=prepared.metrics.manifest_path)
        return SuiteReportSummary(
            suite_run_id=manifest.suite_run_id,
            report_id=manifest.report_id,
            status=manifest.status,
            comparison_record_count=len(manifest.comparison_record_ids),
            selection_record_count=len(manifest.selection_records),
            report_path=repo_relative_path(
                manifest_path,
                prepared.resolution.repo_root,
            ),
            markdown_path=manifest.output_artifact_paths["markdown"],
            reused=reused,
            page_scope_report_path=str(scope_report) if scope_report else None,
        )

    def require_materialized(
        self,
        *,
        verify: bool = True,
    ) -> MaterializedSuiteReport:
        """Recompute and recursively verify the active immutable report."""

        prepared = self._prepare(verify=verify)
        manifest_path = self._manifest_path(prepared)
        if not manifest_path.is_file():
            raise SuiteReportError(
                "suite analysis requires a materialized report; run `eval suite "
                "report` first",
            )
        manifest = self._verify(prepared, manifest_path)
        return MaterializedSuiteReport(
            manifest=manifest,
            manifest_path=manifest_path,
            comparisons=prepared.comparison_records,
            metrics=prepared.metrics,
            report_config=prepared.report_config,
        )

    def _prepare(self, *, verify: bool) -> _PreparedReport:
        root = resolve_repo_root(self.suite_config_path, self.repo_root)
        resolution = resolve_experiment_suite(
            self.suite_config_path,
            repo_root=root,
            suite_output_root=self.suite_output_root,
        )
        metrics = load_suite_metrics(ExperimentSuiteMetricsJob(
            self.suite_config_path,
            relevance_config_path=self.relevance_config_path,
            judge_experiment_path=self.judge_experiment_path,
            include_manual=self.include_manual,
            suite_output_root=self.suite_output_root,
            repo_root=root,
        ), verify=verify)
        if self.report_config_path is None:
            report_config = _default_report_config(resolution.comparisons)
            report_path = None
        else:
            report_path = resolve_repo_path(self.report_config_path, root)
            report_config = load_suite_report_config(report_path)
        report_hash = config_hash(report_config.to_dict())
        selected_ids = report_config.comparison_ids or [
            comparison.comparison_id for comparison in resolution.comparisons
        ]
        known = {
            comparison.comparison_id: comparison
            for comparison in resolution.comparisons
        }
        unknown = sorted(set(selected_ids) - set(known))
        if unknown:
            raise SuiteReportError(
                f"report config references unknown comparisons: {unknown}",
            )
        comparisons = [known[comparison_id] for comparison_id in selected_ids]
        policy_unknown = sorted(
            {
                policy.comparison_id
                for policy in report_config.selection_policies
                if policy.comparison_id not in selected_ids
            },
        )
        if policy_unknown:
            raise SuiteReportError(
                "selection policies reference unreported comparisons: "
                f"{policy_unknown}",
            )
        comparison_records = _build_comparison_records(
            metrics,
            comparisons,
            report_config,
        )
        _validate_selection_policies(
            report_config.selection_policies,
            comparison_records,
        )
        if self.decision_config_path is None:
            decision_path = None
            decision_config = None
            decision_hash = None
        else:
            decision_path = resolve_repo_path(self.decision_config_path, root)
            decision_config = load_selection_decision_config(decision_path)
            decision_hash = config_hash(decision_config.to_dict())
        selection_records = _build_selection_records(
            metrics=metrics,
            comparisons=comparison_records,
            report_config=report_config,
            report_config_hash=report_hash,
            decision_config=decision_config,
        )
        markdown = _render_markdown(
            metrics=metrics,
            comparisons=comparison_records,
            selections=selection_records,
            report_config=report_config,
        )
        implementation_hash = _report_implementation_hash()
        git_commit, git_dirty, git_state_hash = capture_git_state(root)
        comparison_ids = sorted(
            record.comparison_record_id for record in comparison_records
        )
        report_id = _report_id(
            suite_run_id=resolution.suite_run_id,
            suite_manifest_hash=metrics.manifest.suite_manifest_hash,
            metrics_bundle_id=metrics.manifest.metrics_bundle_id,
            metrics_manifest_hash=hash_file(metrics.manifest_path),
            report_config_hash=report_hash,
            decision_config_hash=decision_hash,
            implementation_hash=implementation_hash,
            comparison_record_ids=comparison_ids,
            selection_records=selection_records,
        )
        return _PreparedReport(
            resolution=resolution,
            metrics=metrics,
            report_config=report_config,
            report_config_path=report_path,
            report_config_hash=report_hash,
            decision_config=decision_config,
            decision_config_path=decision_path,
            decision_config_hash=decision_hash,
            implementation_hash=implementation_hash,
            comparison_records=comparison_records,
            selection_records=selection_records,
            markdown=markdown,
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_state_hash=git_state_hash,
            report_id=report_id,
        )

    def _materialize(
        self,
        prepared: _PreparedReport,
    ) -> tuple[SuiteReportManifest, Path, bool]:
        root = prepared.resolution.repo_root
        manifest_path = self._manifest_path(prepared)
        report_dir = manifest_path.parent
        comparisons_path = report_dir / "comparisons.jsonl"
        markdown_path = report_dir / "report.md"
        if manifest_path.is_file():
            manifest = self._verify(prepared, manifest_path)
            return manifest, manifest_path, True
        bundle = ImmutableArtifactBundle(
            manifest_path=manifest_path,
            output_paths={
                "comparisons": comparisons_path,
                "markdown": markdown_path,
            },
            label="suite report",
            error_type=SuiteReportError,
        )
        bundle.write_jsonl("comparisons", prepared.comparison_records)
        bundle.write_text("markdown", prepared.markdown)
        output_paths = {
            "comparisons": repo_relative_path(comparisons_path, root),
            "markdown": repo_relative_path(markdown_path, root),
        }
        output_hashes = bundle.output_hashes
        manifest = SuiteReportManifest(
            report_id=prepared.report_id,
            suite_run_id=prepared.resolution.suite_run_id,
            suite_manifest_path=prepared.metrics.manifest.suite_manifest_path,
            suite_manifest_hash=prepared.metrics.manifest.suite_manifest_hash,
            metrics_bundle_id=prepared.metrics.manifest.metrics_bundle_id,
            metrics_manifest_path=repo_relative_path(
                prepared.metrics.manifest_path,
                root,
            ),
            metrics_manifest_hash=hash_file(prepared.metrics.manifest_path),
            report_config_id=prepared.report_config.report_config_id,
            report_config_path=(
                repo_relative_path(prepared.report_config_path, root)
                if prepared.report_config_path is not None
                else None
            ),
            report_config_hash=prepared.report_config_hash,
            decision_config_path=(
                repo_relative_path(prepared.decision_config_path, root)
                if prepared.decision_config_path is not None
                else None
            ),
            decision_config_hash=prepared.decision_config_hash,
            report_implementation_hash=prepared.implementation_hash,
            comparison_record_ids=sorted(
                record.comparison_record_id
                for record in prepared.comparison_records
            ),
            selection_records=list(prepared.selection_records),
            output_artifact_paths=output_paths,
            output_artifact_hashes=output_hashes,
            counts={
                "comparison_records": len(prepared.comparison_records),
                "selection_records": len(prepared.selection_records),
                "judgment_views": len(prepared.metrics.manifest.judgment_views),
                "retrieval_runs": len(prepared.metrics.manifest.retrieval_runs),
            },
            git_commit=prepared.git_commit,
            git_dirty=prepared.git_dirty,
            git_state_hash=prepared.git_state_hash,
            created_at_utc=_now_utc(),
            status=(
                "selection_recorded"
                if prepared.selection_records
                else "comparison_only"
            ),
        )
        bundle.commit(manifest)
        return self._verify(prepared, manifest_path), manifest_path, False

    @staticmethod
    def _manifest_path(prepared: _PreparedReport) -> Path:
        return (
            prepared.metrics.manifest_path.parents[2]
            / "reports"
            / prepared.report_id
            / "manifest.json"
        )

    @staticmethod
    def _verify(
        prepared: _PreparedReport,
        manifest_path: Path,
    ) -> SuiteReportManifest:
        try:
            manifest = SuiteReportManifest.from_dict(
                read_json_object(manifest_path, error_type=SuiteReportError),
            )
        except ValueError as exc:
            raise SuiteReportError(f"invalid suite report manifest: {exc}") from exc
        root = prepared.resolution.repo_root
        expected = {
            "report_id": prepared.report_id,
            "suite_run_id": prepared.resolution.suite_run_id,
            "suite_manifest_path": prepared.metrics.manifest.suite_manifest_path,
            "suite_manifest_hash": prepared.metrics.manifest.suite_manifest_hash,
            "metrics_bundle_id": prepared.metrics.manifest.metrics_bundle_id,
            "metrics_manifest_path": repo_relative_path(
                prepared.metrics.manifest_path,
                root,
            ),
            "metrics_manifest_hash": hash_file(prepared.metrics.manifest_path),
            "report_config_id": prepared.report_config.report_config_id,
            "report_config_path": (
                repo_relative_path(prepared.report_config_path, root)
                if prepared.report_config_path is not None
                else None
            ),
            "report_config_hash": prepared.report_config_hash,
            "decision_config_path": (
                repo_relative_path(prepared.decision_config_path, root)
                if prepared.decision_config_path is not None
                else None
            ),
            "decision_config_hash": prepared.decision_config_hash,
            "report_implementation_hash": prepared.implementation_hash,
            "comparison_record_ids": sorted(
                record.comparison_record_id
                for record in prepared.comparison_records
            ),
            "selection_records": list(prepared.selection_records),
            "counts": {
                "comparison_records": len(prepared.comparison_records),
                "selection_records": len(prepared.selection_records),
                "judgment_views": len(prepared.metrics.manifest.judgment_views),
                "retrieval_runs": len(prepared.metrics.manifest.retrieval_runs),
            },
        }
        require_compatible(
            {key: getattr(manifest, key) for key in expected},
            expected,
            label="suite report manifest inputs",
            error_type=SuiteReportError,
        )
        artifact_references(
            manifest.output_artifact_paths,
            manifest.output_artifact_hashes,
            repo_root=root,
            label="suite report outputs",
            error_type=SuiteReportError,
        )
        expected_outputs = {
            "comparisons": hash_jsonl_records(
                record.to_dict() for record in prepared.comparison_records
            ),
        }
        markdown_path = resolve_repo_path(
            manifest.output_artifact_paths["markdown"],
            root,
        )
        require_compatible(
            markdown_path.read_text(encoding="utf-8"),
            prepared.markdown,
            label="suite report Markdown content",
            error_type=SuiteReportError,
        )
        expected_outputs["markdown"] = hash_file(markdown_path)
        for name, expected_hash in expected_outputs.items():
            require_compatible(
                manifest.output_artifact_hashes[name],
                expected_hash,
                label=f"suite report {name} canonical computation",
                error_type=SuiteReportError,
            )
        loaded = tuple(
            SuiteComparisonRecord.from_dict(payload)
            for payload in read_jsonl(
                resolve_repo_path(
                    manifest.output_artifact_paths["comparisons"],
                    root,
                ),
                expected_schema_version=SUITE_COMPARISON_RECORD_SCHEMA_VERSION,
            )
        )
        require_compatible(
            loaded,
            prepared.comparison_records,
            label="suite comparison records",
            error_type=SuiteReportError,
        )
        return manifest
