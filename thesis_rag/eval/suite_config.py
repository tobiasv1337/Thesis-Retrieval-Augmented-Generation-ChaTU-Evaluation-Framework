"""Strict configuration and deterministic resolution for retrieval suites."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, Self

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
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_io import (
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.calibration_cohort import (
    DevelopmentCalibrationCohortError,
    verify_development_calibration_cohort,
)
from thesis_rag.eval.question_language_swap import (
    LANGUAGE_SWAP_ARTIFACTS,
    LanguageSwapQuestionSetManifest,
    QuestionLanguageSwapError,
    verify_language_swap_question_set,
)
from thesis_rag.eval.query_selection import (
    SUPPORTED_BENCHMARK_SPLITS,
    select_query_cohort,
)
from thesis_rag.eval.manual_benchmark import (
    ManualBenchmarkManifest,
    verify_manual_benchmark,
)
from thesis_rag.eval.priority_questions import (
    PriorityQuestionManifest,
    verify_priority_questions,
)
from thesis_rag.eval.user_logs import (
    UserLogQuestionSetManifest,
    verify_user_log_question_set,
)
from thesis_rag.eval.schemas import (
    DevelopmentCalibrationCohortManifestRecord,
    QuestionRecord,
    QuestionSetManifestRecord,
)
from thesis_rag.registry import (
    load_system_config,
    resolve_system_config,
)
from thesis_rag.schema import (
    COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION,
    DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION,
    EXPERIMENT_SUITE_SCHEMA_VERSION,
    LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION,
    QUESTION_SCHEMA_VERSION,
    PRIORITY_QUESTION_SET_SCHEMA_VERSION,
    USER_LOG_QUESTION_SET_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    stable_id,
)


DEFAULT_QUESTION_ARTIFACT = "questions_llm_reviewed"
SELECTABLE_QUESTION_ARTIFACTS = {
    "base": frozenset(
        {
            "questions_llm_reviewed",
            "questions_dev",
            "questions_validation",
            "questions_test",
        },
    ),
    "human": frozenset(
        {
            "questions_human_reviewed",
            "questions_test_human_reviewed",
            "questions_double_reviewed",
            "questions_test_double_accepted",
            "questions_smoke",
        },
    ),
}
CANONICAL_RETRIEVAL_DEPTH = 20

TargetLevel = Literal["page", "section", "context_group", "chunk"]
EvaluationStream = Literal["items", "context_items"]

# Experiment overrides deliberately cover only the bounded leaf ablations
# approved in the retrieval plan. New semantic knobs require an explicit code
# review here rather than becoming an accidental general patch language.
EXPERIMENT_OVERRIDE_TYPES: dict[str, type[object]] = {
    "settings.routing_manifest_path": str,
    "settings.include_summary": bool,
    "settings.parent_level": str,
    "settings.pipeline.page_routing.enabled": bool,
    "settings.pipeline.page_routing.mode": str,
    "settings.pipeline.page_routing.top_groups": int,
    "settings.pipeline.page_routing.include_global": bool,
    "settings.pipeline.page_routing.fusion_weights": dict,
    "settings.pipeline.page_routing.candidate_union": dict,
    "settings.pipeline.page_routing.descendant_depth": int,
    "settings.pipeline.page_routing.mapping_source": str,
    "settings.pipeline.protected_priority_slots": int,
    "settings.pipeline.main_nlu_priority_slots": int,
    "settings.pipeline.priority_ordering": str,
    "settings.pipeline.trace_all_candidates": bool,
    "settings.pipeline.main_nlu_routing.fusion_weights": dict,
    "settings.pipeline.main_nlu_routing.candidate_union": dict,
    "settings.pipeline.main_nlu_routing.fallback_to_page_routing": bool,
    "settings.pipeline.page_routing.include_summary": bool,
    "settings.pipeline.page_routing.manifest_path": str,
    "settings.pipeline.page_routing.page_subset_manifest_path": str,
    "settings.scope": str,
    "settings.model_manifest_path": str,
    "settings.query_language_signal": str,
    "settings.pipeline.candidate_k": int,
    "settings.pipeline.config_version": str,
    "settings.pipeline.counterpart_deduplication": str,
    "settings.pipeline.content_rrf.component_id": str,
    "settings.pipeline.content_rrf.enabled": bool,
    "settings.pipeline.content_rrf.rank_constant": int,
    "settings.pipeline.representation.chunker_id": str,
    "settings.pipeline.representation.chunk_artifact_set_manifest_path": str,
    "settings.pipeline.representation.page_subset_manifest_path": str,
    "settings.pipeline.language_policy": str,
    "settings.pipeline.dense_global.enabled": bool,
    "settings.pipeline.sparse_global.enabled": bool,
    "settings.pipeline.main_nlu_routing.enabled": bool,
    "settings.pipeline.main_nlu_routing.mode": str,
    "settings.pipeline.reranker.enabled": bool,
    "settings.pipeline.reranker.component_id": str,
    "settings.pipeline.reranker.profile_id": str,
    "settings.pipeline.reranker.expected_profile_hash": str,
    "settings.pipeline.reranker.endpoint": str,
    "settings.pipeline.reranker.input_format": str,
    "settings.pipeline.page_routing.top_p": int,
    "settings.pipeline.context.policy_id": str,
    "settings.pipeline.context.source_top_k": int,
}


class ExperimentSuiteError(ValueError):
    """Raised when a suite cannot be loaded or resolved safely."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ExperimentSuiteError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _unique_non_empty(values: list[str], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class QuestionSuiteInput(ArtifactModel):
    manifest_path: NonEmptyStr
    artifact_name: NonEmptyStr = DEFAULT_QUESTION_ARTIFACT


class CorpusSuiteInput(ArtifactModel):
    snapshot_manifest_path: NonEmptyStr = (
        "data/thesis/corpus/snapshot_manifest.json"
    )


class ContextEvaluationPolicy(ArtifactModel):
    """Bounded views over the materialized downstream evidence context."""

    rank_cutoffs: list[PositiveInt] = Field(
        default_factory=lambda: [1, 3, 5, 10, 15],
    )
    character_budgets: list[PositiveInt] = Field(default_factory=list)
    include_full_context: bool = True
    compute_mixed_metrics: bool = True
    compute_reference_anchor_diagnostics: bool = False

    @model_validator(mode="after")
    def _validate_context_policy(self) -> Self:
        _unique_non_empty(self.rank_cutoffs, "context rank cutoffs")
        if self.rank_cutoffs != sorted(self.rank_cutoffs):
            raise ValueError("context rank cutoffs must be sorted")
        if self.rank_cutoffs[-1] > CANONICAL_RETRIEVAL_DEPTH:
            raise ValueError("context rank cutoffs must not exceed 20")
        if self.character_budgets != sorted(set(self.character_budgets)):
            raise ValueError("context character budgets must be sorted and unique")
        if not self.include_full_context and not self.character_budgets:
            raise ValueError("context evaluation requires a full or budgeted view")
        if not (
            self.compute_mixed_metrics
            or self.compute_reference_anchor_diagnostics
        ):
            raise ValueError("context evaluation must enable at least one view")
        return self


class PageScopeEvaluationPolicy(ArtifactModel):
    """Evaluate the union below K saved Page predictions, not a flat Page ranking."""

    depths: list[int] = Field(default_factory=lambda: list(range(6)))
    root_cutoffs: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    mapping_source: NonEmptyStr = "data/az/mapping.csv"
    chunker_id: NonEmptyStr = "section_preferred_800_100"
    judgment_mode: Literal["complete", "cached_only"] = Field(
        default="complete", exclude_if=lambda value: value == "complete",
    )

    @model_validator(mode="after")
    def _validate_grid(self) -> Self:
        for values, low, high in ((self.depths, 0, 5), (self.root_cutoffs, 1, 20)):
            if not values or values != sorted(set(values)) or any(not low <= v <= high for v in values):
                raise ValueError("Page scope depths/cutoffs must be unique, ascending and in range")
        return self


class FinalEvaluationPolicy(ArtifactModel):
    streams: list[EvaluationStream] = Field(default_factory=lambda: ["items"])
    target_levels: list[TargetLevel] = Field(
        default_factory=lambda: ["page", "section"],
    )
    include_in_pool: bool = True
    compute_metrics: bool = True
    context: ContextEvaluationPolicy | None = None
    page_scopes: PageScopeEvaluationPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def _validate_unique_values(self) -> Self:
        _unique_non_empty(self.streams, "final evaluation streams")
        _unique_non_empty(self.target_levels, "final evaluation target levels")
        evaluates_context = "context_items" in self.streams
        if self.page_scopes is not None and (
            self.streams != ["items"] or self.target_levels != ["page"]
            or not self.include_in_pool or not self.compute_metrics
        ):
            raise ValueError("Page scopes require pooled, evaluated Page-only items")
        if evaluates_context != (self.context is not None):
            raise ValueError(
                "context policy must be present exactly when context_items is evaluated",
            )
        return self


class StageEvaluationPolicy(ArtifactModel):
    stage_id: NonEmptyStr
    variant_id: NonEmptyStr | None = None
    target_levels: list[TargetLevel]
    include_in_pool: bool = False
    compute_metrics: bool = True
    depth: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_target_levels(self) -> Self:
        _unique_non_empty(self.target_levels, "stage target levels")
        if self.depth is not None and self.depth > CANONICAL_RETRIEVAL_DEPTH:
            raise ValueError("stage evaluation depth must not exceed 20")
        return self


class ExperimentEvaluationPolicy(ArtifactModel):
    final: FinalEvaluationPolicy | None = None
    stages: list[StageEvaluationPolicy] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_stages(self) -> Self:
        identities = [(stage.stage_id, stage.variant_id) for stage in self.stages]
        if len(identities) != len(set(identities)):
            raise ValueError("evaluated stage IDs and variants must be unique")
        return self


class SuiteDefaults(ArtifactModel):
    top_k: PositiveInt = CANONICAL_RETRIEVAL_DEPTH
    stage_evaluation_depth: PositiveInt = CANONICAL_RETRIEVAL_DEPTH
    selection_seed: int = 14731
    query_limit: PositiveInt | None = None
    evaluation_query_limit: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    final_evaluation: FinalEvaluationPolicy = Field(
        default_factory=FinalEvaluationPolicy,
    )

    @model_validator(mode="after")
    def _validate_depths(self) -> Self:
        if self.top_k > CANONICAL_RETRIEVAL_DEPTH:
            raise ValueError("suite top_k must not exceed the canonical depth 20")
        if self.stage_evaluation_depth > CANONICAL_RETRIEVAL_DEPTH:
            raise ValueError("stage_evaluation_depth must not exceed 20")
        return self


class ExperimentDefinition(ArtifactModel):
    experiment_id: NonEmptyStr
    system_config: NonEmptyStr
    splits: list[Literal["development", "validation", "test", "diagnostic"]]
    overrides: JsonDict = Field(default_factory=dict)
    query_limit: PositiveInt | None = None
    evaluation: ExperimentEvaluationPolicy = Field(
        default_factory=ExperimentEvaluationPolicy,
    )

    @model_validator(mode="after")
    def _validate_experiment(self) -> Self:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.experiment_id) is None:
            raise ValueError("experiment_id contains unsafe path characters")
        _unique_non_empty(self.splits, "experiment splits")
        ordered = [
            split for split in SUPPORTED_BENCHMARK_SPLITS if split in self.splits
        ]
        if self.splits != ordered:
            raise ValueError(
                "experiment splits must follow development, validation, test order",
            )
        return self


class ComparisonDefinition(ArtifactModel):
    comparison_id: NonEmptyStr
    experiment_ids: list[NonEmptyStr] = Field(default_factory=list)
    baseline_experiment_id: NonEmptyStr | None = None
    candidate_experiment_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        grouped = bool(self.experiment_ids)
        baseline = self.baseline_experiment_id is not None
        if grouped == baseline:
            raise ValueError(
                "comparison requires either experiment_ids or baseline/candidates",
            )
        if grouped:
            _unique_non_empty(self.experiment_ids, "comparison experiment IDs")
            if len(self.experiment_ids) < 2:
                raise ValueError("comparison requires at least two experiments")
        else:
            _unique_non_empty(
                self.candidate_experiment_ids,
                "comparison candidate experiment IDs",
            )
            if self.baseline_experiment_id in self.candidate_experiment_ids:
                raise ValueError("comparison baseline cannot also be a candidate")
        return self

    def member_ids(self) -> list[str]:
        if self.experiment_ids:
            return list(self.experiment_ids)
        assert self.baseline_experiment_id is not None
        return [self.baseline_experiment_id, *self.candidate_experiment_ids]


class ExperimentSuiteConfig(ArtifactModel):
    suite_id: NonEmptyStr
    questions: QuestionSuiteInput
    corpus: CorpusSuiteInput = Field(default_factory=CorpusSuiteInput)
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    experiments: list[ExperimentDefinition]
    comparisons: list[ComparisonDefinition] = Field(default_factory=list)
    schema_version: Literal[
        "retrieval_experiment_suite_v1",
        "retrieval_experiment_suite_v2",
        COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION,
    ] = (
        EXPERIMENT_SUITE_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_suite(self) -> Self:
        if (
            self.defaults.evaluation_query_limit is not None
            and self.schema_version != COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION
        ):
            raise ValueError("evaluation_query_limit requires a v3 suite")
        if self.schema_version == "retrieval_experiment_suite_v1":
            policies = [
                self.defaults.final_evaluation,
                *[
                    experiment.evaluation.final
                    for experiment in self.experiments
                    if experiment.evaluation.final is not None
                ],
            ]
            if any(policy.context is not None for policy in policies):
                raise ValueError("v1 suites cannot configure context evaluation")
        experiment_ids = [item.experiment_id for item in self.experiments]
        _unique_non_empty(experiment_ids, "suite experiment IDs")
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("suite comparison IDs must be unique")
        known = set(experiment_ids)
        by_id = {item.experiment_id: item for item in self.experiments}
        for comparison in self.comparisons:
            unknown = sorted(set(comparison.member_ids()) - known)
            if unknown:
                raise ValueError(
                    f"comparison {comparison.comparison_id!r} references unknown "
                    f"experiments: {unknown}",
                )
            cohort_signatures = {
                (
                    tuple(by_id[experiment_id].splits),
                    by_id[experiment_id].query_limit or self.defaults.query_limit,
                )
                for experiment_id in comparison.member_ids()
            }
            if len(cohort_signatures) != 1:
                raise ValueError(
                    f"comparison {comparison.comparison_id!r} mixes query cohorts",
                )
        return self

    def to_dict(self) -> JsonDict:
        payload = super().to_dict()
        if self.schema_version == "retrieval_experiment_suite_v1":
            payload["defaults"]["final_evaluation"].pop("context", None)
            for experiment in payload["experiments"]:
                final = experiment["evaluation"].get("final")
                if isinstance(final, dict):
                    final.pop("context", None)
        return payload


class ResolvedOverride(ArtifactModel):
    path: NonEmptyStr
    base_value: Any
    resolved_value: Any


class ResolvedQuestionInput(ArtifactModel):
    question_set_id: NonEmptyStr
    seal_kind: Literal[
        "base", "human", "calibration", "language_swap", "manual_benchmark", "user_logs", "priority"
    ]
    base_question_set_id: NonEmptyStr
    manifest_path: NonEmptyStr
    manifest_hash: Sha256
    artifact_name: NonEmptyStr
    questions_path: NonEmptyStr
    questions_hash: Sha256
    question_count: PositiveInt


class ResolvedCorpusInput(ArtifactModel):
    snapshot_manifest_path: NonEmptyStr
    snapshot_manifest_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256


class ResolvedExperiment(ArtifactModel):
    experiment_id: NonEmptyStr
    system_id: NonEmptyStr
    base_system_config_path: NonEmptyStr
    base_system_config_hash: Sha256
    resolved_system_config_path: NonEmptyStr
    resolved_system_config_hash: Sha256
    resolved_system_config_file_hash: Sha256
    overrides: list[ResolvedOverride]
    splits: list[NonEmptyStr]
    selection_seed: int
    query_limit: PositiveInt | None
    selected_query_count: PositiveInt
    selected_query_ids_hash: Sha256
    selected_queries_hash: Sha256
    evaluation_query_limit: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    top_k: PositiveInt
    stage_evaluation_depth: PositiveInt | None
    final_evaluation: FinalEvaluationPolicy
    stages: list[StageEvaluationPolicy]

    @model_validator(mode="after")
    def _validate_evaluation_cohort(self) -> Self:
        if (
            self.evaluation_query_limit is not None
            and self.evaluation_query_limit > self.selected_query_count
        ):
            raise ValueError("evaluation cohort exceeds the retrieval cohort")
        return self

    @property
    def evaluated_query_count(self) -> int:
        return self.evaluation_query_limit or self.selected_query_count


@dataclass(frozen=True)
class SuiteResolution:
    repo_root: Path
    config_path: Path
    config: ExperimentSuiteConfig
    suite_config_hash: str
    question_input: ResolvedQuestionInput
    corpus_input: ResolvedCorpusInput
    experiments: tuple[ResolvedExperiment, ...]
    comparisons: tuple[ComparisonDefinition, ...]
    resolved_config_payloads: dict[str, str]
    resolution_hash: str
    suite_run_id: str


def load_experiment_suite_config(path: str | Path) -> ExperimentSuiteConfig:
    config_path = Path(path)
    try:
        loaded = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
        if not isinstance(loaded, dict):
            raise ValueError("suite configuration must be a YAML mapping")
        return ExperimentSuiteConfig.from_dict(loaded)
    except (OSError, TypeError, yaml.YAMLError, ValueError) as exc:
        if isinstance(exc, ExperimentSuiteError):
            raise
        raise ExperimentSuiteError(
            f"invalid experiment suite config {config_path}: {exc}",
        ) from exc


def _lookup_leaf(config: JsonDict, path: str) -> tuple[JsonDict, str, Any]:
    parts = path.split(".")
    current: Any = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ExperimentSuiteError(f"override path does not exist: {path}")
        current = current[part]
    leaf = parts[-1]
    if not isinstance(current, dict):
        raise ExperimentSuiteError(f"override path does not exist: {path}")
    return current, leaf, current.get(leaf)


def _apply_overrides(
    base: JsonDict,
    overrides: JsonDict,
) -> tuple[JsonDict, list[ResolvedOverride]]:
    resolved = deepcopy(base)
    records: list[ResolvedOverride] = []
    for path, value in sorted(overrides.items()):
        expected_type = EXPERIMENT_OVERRIDE_TYPES.get(path)
        if expected_type is None:
            raise ExperimentSuiteError(f"override path is not allowlisted: {path}")
        parent, leaf, base_value = _lookup_leaf(resolved, path)
        if type(value) is not expected_type or (
            base_value is not None and type(base_value) is not expected_type
        ):
            raise ExperimentSuiteError(
                f"override {path!r} must preserve type {expected_type.__name__}",
            )
        if expected_type is str and not str(value).strip():
            raise ExperimentSuiteError(f"override {path!r} must not be empty")
        if path.endswith((".top_p", ".top_groups", ".rank_constant", ".source_top_k")) and int(
            value,
        ) <= 0:
            raise ExperimentSuiteError(f"override {path!r} must be positive")
        if path.endswith(".candidate_k") and int(value) < 20:
            raise ExperimentSuiteError(
                f"override {path!r} must be at least the canonical Top 20",
            )
        parent[leaf] = value
        records.append(
            ResolvedOverride(
                path=path,
                base_value=base_value,
                resolved_value=value,
            ),
        )
    return resolved, records


def _load_questions(
    config: ExperimentSuiteConfig,
    root: Path,
) -> tuple[
    ResolvedQuestionInput,
    tuple[QuestionRecord, ...],
    QuestionSetManifestRecord
    | DevelopmentCalibrationCohortManifestRecord
    | LanguageSwapQuestionSetManifest
    | ManualBenchmarkManifest
    | PriorityQuestionManifest
    | UserLogQuestionSetManifest,
]:
    manifest_path = resolve_repo_path(config.questions.manifest_path, root)
    try:
        raw_manifest = read_json_object(
            manifest_path,
            error_type=ExperimentSuiteError,
        )
    except ValueError as exc:
        raise ExperimentSuiteError(f"invalid question-set manifest: {exc}") from exc
    if raw_manifest.get("schema_version") == PRIORITY_QUESTION_SET_SCHEMA_VERSION:
        if any(
            experiment.splits != ["development"] for experiment in config.experiments
        ):
            raise ExperimentSuiteError("Priority cohorts require the Development split")
        try:
            selection, questions = verify_priority_questions(
                manifest_path, repo_root=root, artifact_name=config.questions.artifact_name
            )
        except (OSError, ValueError) as exc:
            raise ExperimentSuiteError(f"Invalid priority question selection: {exc}") from exc
        if not questions:
            raise ExperimentSuiteError("Selected priority question artifact is empty")
        name = config.questions.artifact_name
        return (
            ResolvedQuestionInput(
                question_set_id=selection.question_set_id,
                seal_kind="priority",
                base_question_set_id=selection.question_set_id,
                manifest_path=repo_relative_path(manifest_path, root),
                manifest_hash=hash_file(manifest_path),
                artifact_name=name,
                questions_path=selection.artifact_paths[name],
                questions_hash=selection.artifact_hashes[name],
                question_count=len(questions),
            ),
            questions,
            selection,
        )
    if (
        raw_manifest.get("schema_version")
        == DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION
    ):
        if config.questions.artifact_name != "questions_smoke":
            raise ExperimentSuiteError(
                "Development calibration manifests expose only 'questions_smoke'",
            )
        if any(
            split != "development"
            for experiment in config.experiments
            for split in experiment.splits
        ):
            raise ExperimentSuiteError(
                "Development calibration manifests may be used only by "
                "Development experiments",
            )
        try:
            calibration, questions = verify_development_calibration_cohort(
                manifest_path,
                repo_root=root,
            )
        except DevelopmentCalibrationCohortError as exc:
            raise ExperimentSuiteError(
                f"invalid Development calibration cohort: {exc}",
            ) from exc
        return (
            ResolvedQuestionInput(
                question_set_id=calibration.calibration_cohort_id,
                seal_kind="calibration",
                base_question_set_id=calibration.base_question_set_id,
                manifest_path=repo_relative_path(manifest_path, root),
                manifest_hash=hash_file(manifest_path),
                artifact_name="questions_smoke",
                questions_path=calibration.questions_path,
                questions_hash=calibration.questions_hash,
                question_count=calibration.question_count,
            ),
            questions,
            calibration,
        )
    if raw_manifest.get("schema_version") == USER_LOG_QUESTION_SET_SCHEMA_VERSION:
        if config.questions.artifact_name != "questions":
            raise ExperimentSuiteError("user-log manifests expose only 'questions'")
        try:
            logs, questions = verify_user_log_question_set(manifest_path)
        except (OSError, ValueError) as exc:
            raise ExperimentSuiteError(f"invalid user-log question set: {exc}") from exc
        if not questions:
            raise ExperimentSuiteError("user-log question set is empty")
        return (
            ResolvedQuestionInput(
                question_set_id=logs.question_set_id,
                seal_kind="user_logs",
                base_question_set_id=logs.question_set_id,
                manifest_path=repo_relative_path(manifest_path, root),
                manifest_hash=hash_file(manifest_path),
                artifact_name="questions",
                questions_path=repo_relative_path(
                    manifest_path.parent / "questions.jsonl", root,
                ),
                questions_hash=logs.artifact_hashes["questions"],
                question_count=len(questions),
            ),
            questions,
            logs,
        )
    if raw_manifest.get("schema_version") == "manual_page_benchmark_manifest_v1":
        if config.questions.artifact_name != "questions":
            raise ExperimentSuiteError("manual benchmarks expose only 'questions'")
        if any(
            experiment.splits != ["diagnostic"] for experiment in config.experiments
        ):
            raise ExperimentSuiteError("manual benchmarks require the diagnostic split")
        try:
            manual, questions = verify_manual_benchmark(manifest_path, repo_root=root)
        except ValueError as exc:
            raise ExperimentSuiteError(f"invalid manual benchmark: {exc}") from exc
        return (
            ResolvedQuestionInput(
                question_set_id=manual.question_set_id,
                seal_kind="manual_benchmark",
                base_question_set_id=manual.question_set_id,
                manifest_path=repo_relative_path(manifest_path, root),
                manifest_hash=hash_file(manifest_path),
                artifact_name="questions",
                questions_path=manual.artifact_paths["questions"],
                questions_hash=manual.artifact_hashes["questions"],
                question_count=len(questions),
            ),
            questions,
            manual,
        )
    if raw_manifest.get("schema_version") == LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION:
        name = config.questions.artifact_name
        if name not in LANGUAGE_SWAP_ARTIFACTS:
            raise ExperimentSuiteError(
                f"language-swap manifests do not expose artifact {name!r}",
            )
        try:
            language_swap, views = verify_language_swap_question_set(
                manifest_path,
                repo_root=root,
            )
        except QuestionLanguageSwapError as exc:
            raise ExperimentSuiteError(
                f"invalid language-swap question set: {exc}",
            ) from exc
        questions = views[name]
        if not questions:
            raise ExperimentSuiteError(
                f"language-swap artifact {name!r} contains no questions",
            )
        return (
            ResolvedQuestionInput(
                question_set_id=language_swap.question_set_id,
                seal_kind="language_swap",
                base_question_set_id=language_swap.base_question_set_id,
                manifest_path=repo_relative_path(manifest_path, root),
                manifest_hash=hash_file(manifest_path),
                artifact_name=name,
                questions_path=language_swap.artifact_paths[name],
                questions_hash=language_swap.artifact_hashes[name],
                question_count=len(questions),
            ),
            questions,
            language_swap,
        )
    try:
        manifest = QuestionSetManifestRecord.from_dict(raw_manifest)
    except ValueError as exc:
        raise ExperimentSuiteError(f"invalid question-set manifest: {exc}") from exc
    if manifest.status != "sealed":
        raise ExperimentSuiteError("question-set manifest must be sealed")
    archive_path = resolve_repo_path(manifest.archive_path, root)
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != manifest.archive_byte_size
        or hash_file(archive_path) != manifest.archive_hash
    ):
        raise ExperimentSuiteError("sealed question-set archive is missing or changed")
    name = config.questions.artifact_name
    if name not in SELECTABLE_QUESTION_ARTIFACTS[manifest.seal_kind]:
        raise ExperimentSuiteError(
            f"artifact {name!r} is not selectable from a {manifest.seal_kind} seal",
        )
    if name not in manifest.artifact_paths:
        raise ExperimentSuiteError(
            f"question-set manifest has no artifact named {name!r}",
        )
    questions_path = resolve_repo_path(manifest.artifact_paths[name], root)
    if not questions_path.is_file():
        raise ExperimentSuiteError(f"sealed question artifact is missing: {questions_path}")
    if questions_path.stat().st_size != manifest.artifact_byte_sizes[name]:
        raise ExperimentSuiteError("sealed question artifact byte size mismatch")
    questions_hash = hash_file(questions_path)
    if questions_hash != manifest.artifact_hashes[name]:
        raise ExperimentSuiteError("sealed question artifact hash mismatch")
    try:
        questions = tuple(
            QuestionRecord.from_dict(record)
            for record in read_jsonl(
                questions_path,
                expected_schema_version=QUESTION_SCHEMA_VERSION,
            )
        )
    except (OSError, ValueError) as exc:
        raise ExperimentSuiteError(f"invalid sealed question artifact: {exc}") from exc
    if not questions:
        raise ExperimentSuiteError("sealed question artifact must not be empty")
    if len(questions) != manifest.artifact_counts[name]:
        raise ExperimentSuiteError("sealed question artifact count mismatch")
    query_ids = [question.query_id for question in questions]
    if len(query_ids) != len(set(query_ids)):
        raise ExperimentSuiteError("sealed question artifact has duplicate query IDs")
    return (
        ResolvedQuestionInput(
            question_set_id=manifest.question_set_id,
            seal_kind=manifest.seal_kind,
            base_question_set_id=manifest.base_question_set_id,
            manifest_path=repo_relative_path(manifest_path, root),
            manifest_hash=hash_file(manifest_path),
            artifact_name=name,
            questions_path=repo_relative_path(questions_path, root),
            questions_hash=questions_hash,
            question_count=len(questions),
        ),
        questions,
        manifest,
    )


def _load_corpus(
    config: ExperimentSuiteConfig,
    root: Path,
    question_manifest: (
        QuestionSetManifestRecord
        | DevelopmentCalibrationCohortManifestRecord
        | LanguageSwapQuestionSetManifest
        | ManualBenchmarkManifest
        | PriorityQuestionManifest
        | UserLogQuestionSetManifest
    ),
) -> ResolvedCorpusInput:
    snapshot_path = resolve_repo_path(config.corpus.snapshot_manifest_path, root)
    try:
        snapshot = SnapshotManifest.from_dict(
            read_json_object(snapshot_path, error_type=ExperimentSuiteError),
        )
    except ValueError as exc:
        raise ExperimentSuiteError(f"invalid snapshot manifest: {exc}") from exc
    snapshot_hash = hash_file(snapshot_path)
    # Logs are observed requests, not generated from a source snapshot. The
    # Suite still pins its evaluation corpus; the log set does not invent one.
    source_bound = not isinstance(question_manifest, UserLogQuestionSetManifest)
    if source_bound and snapshot_hash != question_manifest.snapshot_manifest_hash:
        raise ExperimentSuiteError(
            "suite snapshot differs from the sealed question-set snapshot",
        )
    if source_bound and snapshot.corpus_hash != question_manifest.corpus_hash:
        raise ExperimentSuiteError(
            "suite corpus differs from the sealed question-set corpus",
        )
    if (
        source_bound
        and snapshot.source_snapshot_hash != question_manifest.source_snapshot_hash
    ):
        raise ExperimentSuiteError(
            "suite source snapshot differs from the sealed question set",
        )
    return ResolvedCorpusInput(
        snapshot_manifest_path=repo_relative_path(snapshot_path, root),
        snapshot_manifest_hash=snapshot_hash,
        corpus_hash=snapshot.corpus_hash,
        source_snapshot_hash=snapshot.source_snapshot_hash,
    )


def _resolution_payload(
    *,
    suite_id: str,
    suite_config_hash: str,
    question_input: ResolvedQuestionInput,
    corpus_input: ResolvedCorpusInput,
    experiments: list[ResolvedExperiment],
    comparisons: tuple[ComparisonDefinition, ...],
) -> JsonDict:
    def experiment_payload(experiment: ResolvedExperiment) -> JsonDict:
        payload = experiment.to_dict()
        final = payload.get("final_evaluation")
        if isinstance(final, dict) and final.get("context") is None:
            final.pop("context", None)
        return {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "base_system_config_path",
                "resolved_system_config_path",
            }
        }

    return {
        "suite_id": suite_id,
        "suite_config_hash": suite_config_hash,
        "question_set": {
            "question_set_id": question_input.question_set_id,
            "seal_kind": question_input.seal_kind,
            "base_question_set_id": question_input.base_question_set_id,
            "manifest_hash": question_input.manifest_hash,
            "artifact_name": question_input.artifact_name,
            "questions_hash": question_input.questions_hash,
            "question_count": question_input.question_count,
        },
        "corpus": {
            "snapshot_manifest_hash": corpus_input.snapshot_manifest_hash,
            "corpus_hash": corpus_input.corpus_hash,
            "source_snapshot_hash": corpus_input.source_snapshot_hash,
        },
        "experiments": [experiment_payload(experiment) for experiment in experiments],
        "comparisons": [comparison.to_dict() for comparison in comparisons],
    }


def resolve_experiment_suite(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    suite_output_root: str | Path = "outputs/thesis/suites",
) -> SuiteResolution:
    path = Path(config_path)
    root = resolve_repo_root(path, repo_root)
    resolved_path = resolve_repo_path(path, root)
    config = load_experiment_suite_config(resolved_path)
    if config.suite_id.startswith("template_"):
        raise ExperimentSuiteError(
            "template suite_id must be replaced by a reviewed experiment identity",
        )
    suite_config_hash = config_hash(config.to_dict())
    question_input, questions, question_manifest = _load_questions(config, root)
    if (
        config.defaults.evaluation_query_limit is not None
        and question_input.seal_kind == "manual_benchmark"
    ):
        raise ExperimentSuiteError(
            "evaluation subcohorts currently require a QREL-based suite"
        )
    corpus_input = _load_corpus(config, root, question_manifest)

    preliminary: list[tuple[ResolvedExperiment, str]] = []
    for experiment in config.experiments:
        base_path = resolve_repo_path(experiment.system_config, root)
        try:
            base = load_system_config(base_path)
            raw_resolved, override_records = _apply_overrides(
                base.raw_config,
                experiment.overrides,
            )
            if override_records:
                # The base file's optional self-check has already been verified
                # by load_system_config. A derived config receives its own hash
                # from the suite and must not retain the base's expected hash.
                raw_resolved["expected_config_hash"] = None
            resolved = resolve_system_config(raw_resolved, path=base_path)
        except ValueError as exc:
            raise ExperimentSuiteError(
                f"experiment {experiment.experiment_id!r} cannot resolve: {exc}",
            ) from exc
        query_limit = (
            experiment.query_limit
            if experiment.query_limit is not None
            else config.defaults.query_limit
        )
        try:
            selected, diagnostics = select_query_cohort(
                questions,
                splits=experiment.splits,
                limit=query_limit,
                seed=config.defaults.selection_seed,
            )
        except ValueError as exc:
            raise ExperimentSuiteError(
                f"experiment {experiment.experiment_id!r} query selection failed: {exc}",
            ) from exc
        stages = experiment.evaluation.stages
        stage_depth = config.defaults.stage_evaluation_depth if stages else None
        normalized_stages = [
            stage.model_copy(
                update={
                    "depth": stage.depth or config.defaults.stage_evaluation_depth,
                },
            )
            for stage in stages
        ]
        final_policy = experiment.evaluation.final or config.defaults.final_evaluation
        payload = yaml.safe_dump(
            resolved.raw_config,
            allow_unicode=True,
            sort_keys=True,
        )
        preliminary.append(
            (
                ResolvedExperiment(
                    experiment_id=experiment.experiment_id,
                    system_id=resolved.system_id,
                    base_system_config_path=repo_relative_path(base_path, root),
                    base_system_config_hash=base.config_hash,
                    resolved_system_config_path="pending",
                    resolved_system_config_hash=resolved.config_hash,
                    resolved_system_config_file_hash=hash_text(payload),
                    overrides=override_records,
                    splits=list(experiment.splits),
                    selection_seed=config.defaults.selection_seed,
                    query_limit=query_limit,
                    selected_query_count=len(selected),
                    selected_query_ids_hash=str(diagnostics["selected_query_ids_hash"]),
                    selected_queries_hash=hash_jsonable(
                        [question.to_dict() for question in selected],
                    ),
                    evaluation_query_limit=config.defaults.evaluation_query_limit,
                    top_k=config.defaults.top_k,
                    stage_evaluation_depth=stage_depth,
                    final_evaluation=final_policy,
                    stages=normalized_stages,
                ),
                payload,
            ),
        )

    provisional = [item[0] for item in preliminary]
    resolution_hash = hash_jsonable(
        _resolution_payload(
            suite_id=config.suite_id,
            suite_config_hash=suite_config_hash,
            question_input=question_input,
            corpus_input=corpus_input,
            experiments=provisional,
            comparisons=tuple(config.comparisons),
        ),
    )
    suite_run_id = stable_id("retrieval_suite", resolution_hash)
    output_root = resolve_repo_path(suite_output_root, root)
    suite_dir = output_root / suite_run_id
    experiments: list[ResolvedExperiment] = []
    payloads: dict[str, str] = {}
    for provisional_experiment, payload in preliminary:
        experiment_id = provisional_experiment.experiment_id
        config_output = suite_dir / "resolved_system_configs" / f"{experiment_id}.yaml"
        experiment = provisional_experiment.model_copy(
            update={
                "resolved_system_config_path": repo_relative_path(config_output, root),
            },
        )
        experiments.append(experiment)
        payloads[experiment_id] = payload

    # The output path is intentionally excluded from the semantic resolution;
    # replacing the provisional path therefore must not alter this hash.
    final_hash = hash_jsonable(
        _resolution_payload(
            suite_id=config.suite_id,
            suite_config_hash=suite_config_hash,
            question_input=question_input,
            corpus_input=corpus_input,
            experiments=experiments,
            comparisons=tuple(config.comparisons),
        ),
    )
    if final_hash != resolution_hash:
        raise ExperimentSuiteError("suite resolution changed while assigning paths")
    return SuiteResolution(
        repo_root=root,
        config_path=resolved_path,
        config=config,
        suite_config_hash=suite_config_hash,
        question_input=question_input,
        corpus_input=corpus_input,
        experiments=tuple(experiments),
        comparisons=tuple(config.comparisons),
        resolved_config_payloads=payloads,
        resolution_hash=resolution_hash,
        suite_run_id=suite_run_id,
    )


def resolution_payload_from_manifest(manifest: Any) -> JsonDict:
    """Return the semantic projection used by a suite manifest validator."""

    return _resolution_payload(
        suite_id=manifest.suite_id,
        suite_config_hash=manifest.suite_config_hash,
        question_input=manifest.question_set,
        corpus_input=manifest.corpus,
        experiments=list(manifest.experiments),
        comparisons=tuple(manifest.comparisons),
    )
