"""Strict question-generation configuration contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self, get_args

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.schema import GUIDELINE_VERSION, QUESTION_CONFIG_SCHEMA_VERSION


Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Language = Literal["de", "en"]
StructuredOutputMode = Literal["json_schema", "json_object"]
ReasoningEffort = Literal["none", "low", "medium", "high"]
TemporalClassification = Literal[
    "stable_or_recurring",
    "snapshot_qualified",
    "highly_volatile",
    "expired_event_or_news",
    "not_temporal",
]

def _require_sum_one(values: list[float], label: str) -> None:
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"{label} values must sum to 1.0")


class QuestionSourceArtifacts(ArtifactModel):
    snapshot_manifest_path: NonEmptyStr
    pages_path: NonEmptyStr
    sections_path: NonEmptyStr
    section_context_groups_path: NonEmptyStr
    page_quality_path: NonEmptyStr
    source_relationships_path: NonEmptyStr
    page_table_path: NonEmptyStr
    section_table_path: NonEmptyStr
    tables_path: NonEmptyStr


class QuestionArtifactPaths(ArtifactModel):
    source_families: NonEmptyStr
    generation_units: NonEmptyStr
    generation_unit_planning_report: NonEmptyStr
    generation_batch_manifests: NonEmptyStr
    generation_attempt_results: NonEmptyStr
    question_candidates_raw: NonEmptyStr
    question_candidates_filtered: NonEmptyStr
    question_review_events: NonEmptyStr
    human_question_review_events: NonEmptyStr
    question_dedup_clusters: NonEmptyStr
    question_dedup_report: NonEmptyStr
    question_dedup_manual_decisions: NonEmptyStr
    questions_llm_reviewed: NonEmptyStr
    questions_human_reviewed: NonEmptyStr
    questions_test_human_reviewed: NonEmptyStr
    questions_double_reviewed: NonEmptyStr
    questions_test_double_accepted: NonEmptyStr
    questions_dev: NonEmptyStr
    questions_validation: NonEmptyStr
    questions_test: NonEmptyStr
    questions_smoke: NonEmptyStr
    questions_manual_utterance_diagnostic: NonEmptyStr
    human_review_plan: NonEmptyStr
    human_review_sampling_report: NonEmptyStr
    human_review_agreement: NonEmptyStr
    base_question_set_manifest: NonEmptyStr
    base_question_set_archive: NonEmptyStr
    human_question_set_manifest: NonEmptyStr
    human_question_set_archive: NonEmptyStr
    split_manifest: NonEmptyStr

    @model_validator(mode="after")
    def paths_are_unique(self) -> Self:
        values = list(self.to_dict().values())
        if len(values) != len(set(values)):
            raise ValueError("Question config artifact paths must be unique")
        return self


class QuestionIdentity(ArtifactModel):
    source_family_rule_version: NonEmptyStr
    generation_unit_rule_version: NonEmptyStr
    question_candidate_rule_version: NonEmptyStr
    dedup_rule_version: NonEmptyStr
    question_review_guideline_version: NonEmptyStr
    relevance_guideline_version: NonEmptyStr

    @model_validator(mode="after")
    def relevance_version_is_current(self) -> Self:
        if self.relevance_guideline_version != GUIDELINE_VERSION:
            raise ValueError(
                "identity.relevance_guideline_version must be "
                f"{GUIDELINE_VERSION}",
            )
        return self


class TargetCounts(ArtifactModel):
    accepted_total: NonNegativeInt
    final_test: NonNegativeInt
    final_test_human_reviewed: NonNegativeInt
    human_reviewed_total: NonNegativeInt
    double_reviewed_final_test: NonNegativeInt
    double_accepted_final_test: NonNegativeInt


class ThreeWayCounts(ArtifactModel):
    development: NonNegativeInt
    validation: NonNegativeInt
    test: NonNegativeInt

    @property
    def split_total(self) -> int:
        return self.development + self.validation + self.test


class SmokeCounts(ArtifactModel):
    development_human_reviewed_subset: NonNegativeInt


class SplitConfig(ArtifactModel):
    random_seed: NonNegativeInt
    target_counts: TargetCounts
    large_benchmark: ThreeWayCounts
    human_core: ThreeWayCounts
    smoke: SmokeCounts

    @model_validator(mode="after")
    def counts_reconcile(self) -> Self:
        targets = self.target_counts
        if self.large_benchmark.split_total != targets.accepted_total:
            raise ValueError(
                "splits.large_benchmark totals must equal target_counts.accepted_total",
            )
        if self.large_benchmark.test != targets.final_test:
            raise ValueError(
                "splits.large_benchmark.test must equal target_counts.final_test",
            )
        if self.human_core.split_total != targets.human_reviewed_total:
            raise ValueError(
                "splits.human_core totals must equal "
                "target_counts.human_reviewed_total",
            )
        if self.human_core.test != targets.final_test_human_reviewed:
            raise ValueError(
                "splits.human_core.test must equal "
                "target_counts.final_test_human_reviewed",
            )
        if targets.double_reviewed_final_test > targets.final_test_human_reviewed:
            raise ValueError(
                "double-reviewed final-test count cannot exceed the human-reviewed "
                "final-test count",
            )
        if targets.double_accepted_final_test > targets.final_test_human_reviewed:
            raise ValueError(
                "double-accepted final-test count cannot exceed the human-reviewed "
                "final-test count",
            )
        if (
            self.smoke.development_human_reviewed_subset
            > self.human_core.development
        ):
            raise ValueError(
                "the smoke subset cannot exceed human-reviewed development questions",
            )
        return self


class SourceSelectionConfig(ArtifactModel):
    minimum_context_group_focus_chars: PositiveInt
    low_priority_below_page_chars: PositiveInt
    low_priority_url_substrings: list[NonEmptyStr] = Field(default_factory=list)
    max_candidates_per_source_family: PositiveInt
    max_candidates_per_page_record: PositiveInt
    suppress_contained_fragments_by_default: bool

    @model_validator(mode="after")
    def limits_are_ordered(self) -> Self:
        if (
            self.minimum_context_group_focus_chars
            >= self.low_priority_below_page_chars
        ):
            raise ValueError(
                "minimum context-group chars must be below the low-priority page size",
            )
        if (
            self.max_candidates_per_page_record
            > self.max_candidates_per_source_family
        ):
            raise ValueError(
                "max_candidates_per_page_record cannot exceed "
                "max_candidates_per_source_family",
            )
        normalized_substrings = [
            substring.casefold() for substring in self.low_priority_url_substrings
        ]
        if len(normalized_substrings) != len(set(normalized_substrings)):
            raise ValueError("low_priority_url_substrings must be unique")
        return self


class BasicGenerationUnitConfig(ArtifactModel):
    enabled: bool
    max_input_chars: PositiveInt
    max_question_budget: PositiveInt


class ContextGroupGenerationUnitConfig(BasicGenerationUnitConfig):
    window_overlap_chars: NonNegativeInt

    @model_validator(mode="after")
    def overlap_is_bounded(self) -> "ContextGroupGenerationUnitConfig":
        if self.window_overlap_chars >= self.max_input_chars:
            raise ValueError("context-group overlap must be below max_input_chars")
        return self


class GroupWindowGenerationUnitConfig(BasicGenerationUnitConfig):
    max_context_groups: PositiveInt


class GenerationUnitsConfig(ArtifactModel):
    context_group: ContextGroupGenerationUnitConfig
    table_context: BasicGenerationUnitConfig
    page_synthesis: GroupWindowGenerationUnitConfig

    def unit_configs(self) -> tuple[BasicGenerationUnitConfig, ...]:
        return (
            self.context_group,
            self.table_context,
            self.page_synthesis,
        )


class SurfaceFormTargets(ArtifactModel):
    complete_question: Fraction
    conversational_short: Fraction
    telegraphic_keywords: Fraction

    @model_validator(mode="after")
    def targets_sum_to_one(self) -> Self:
        _require_sum_one(
            [
                self.complete_question,
                self.conversational_short,
                self.telegraphic_keywords,
            ],
            "surface_form_targets",
        )
        return self


class EvidenceScopeTargets(ArtifactModel):
    localized_share: Fraction
    same_page_multi_section_share: Fraction

    @model_validator(mode="after")
    def targets_sum_to_one(self) -> Self:
        _require_sum_one(
            [self.localized_share, self.same_page_multi_section_share],
            "evidence_scope_targets",
        )
        return self


class StudentRoleTargets(ArtifactModel):
    """Minimum student coverage without prescribing every accepted role mix."""

    inclusive_minimum_share: Fraction
    student_only_minimum_share_within_student: Fraction


class QueryVariationTargets(ArtifactModel):
    """Independent minimum shares for natural paraphrases and situations."""

    natural_paraphrase_minimum_share: Fraction
    situational_request_minimum_share: Fraction


class BackfillPlanningConfig(ArtifactModel):
    """Deterministic call headroom for accepted-question deficits."""

    expected_accepted_yield: Annotated[float, Field(gt=0.0, le=1.0)]
    minimum_attempts_per_cell: PositiveInt


class LocalModelConfig(ArtifactModel):
    required_for_canonical_results: bool
    model_id: NonEmptyStr | None
    model_revision: NonEmptyStr | None
    model_path: NonEmptyStr | None
    model_digest: Sha256 | None
    endpoint: NonEmptyStr | None
    structured_output_mode: StructuredOutputMode
    temperature: Annotated[float, Field(ge=0.0)]
    max_tokens: PositiveInt
    timeout_seconds: PositiveInt
    max_retries: NonNegativeInt


class ReviewLocalModelConfig(LocalModelConfig):
    reasoning_effort: ReasoningEffort


class PromptConfig(ArtifactModel):
    prompt_id: NonEmptyStr
    path: NonEmptyStr


class GenerationPrompts(ArtifactModel):
    de: PromptConfig
    en: PromptConfig


class TemporalPolicy(ArtifactModel):
    allowed_temporal_classes: list[TemporalClassification]
    rejected_temporal_classes: list[TemporalClassification]

    @model_validator(mode="after")
    def classes_partition_policy(self) -> Self:
        allowed = set(self.allowed_temporal_classes)
        rejected = set(self.rejected_temporal_classes)
        if len(allowed) != len(self.allowed_temporal_classes):
            raise ValueError("allowed_temporal_classes must be unique")
        if len(rejected) != len(self.rejected_temporal_classes):
            raise ValueError("rejected_temporal_classes must be unique")
        overlap = allowed & rejected
        if overlap:
            raise ValueError(
                f"allowed and rejected temporal classes overlap: {sorted(overlap)}",
            )
        missing = set(get_args(TemporalClassification)) - allowed - rejected
        if missing:
            raise ValueError(
                "temporal policy must classify every supported value; missing "
                f"{sorted(missing)}",
            )
        return self


class GenerationConfig(ArtifactModel):
    base_seed: NonNegativeInt
    response_cache_dir: NonEmptyStr
    languages: list[Language]
    source_selection: SourceSelectionConfig
    generation_units: GenerationUnitsConfig
    surface_form_targets: SurfaceFormTargets
    evidence_scope_targets: EvidenceScopeTargets
    student_role_targets: StudentRoleTargets
    query_variation_targets: QueryVariationTargets
    backfill: BackfillPlanningConfig
    local_model: LocalModelConfig
    prompts: GenerationPrompts
    temporal_policy: TemporalPolicy

    @model_validator(mode="after")
    def languages_are_canonical(self) -> Self:
        if self.languages != ["de", "en"]:
            raise ValueError("generation.languages must be exactly ['de', 'en']")
        return self


class LexicalOverlapFilter(ArtifactModel):
    low_max: Fraction
    medium_max: Fraction

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> Self:
        if not self.low_max < self.medium_max:
            raise ValueError(
                "lexical-overlap thresholds must satisfy low_max < medium_max",
            )
        return self


class DeterministicFilters(ArtifactModel):
    maximum_question_chars: PositiveInt
    maximum_evidence_quote_chars: PositiveInt
    telegraphic_keywords_soft_max_tokens: PositiveInt
    lexical_overlap: LexicalOverlapFilter


class ReviewConfig(ArtifactModel):
    question_guideline_version: NonEmptyStr
    relevance_guideline_version: NonEmptyStr
    prompt: PromptConfig
    local_llm: ReviewLocalModelConfig
    anchor_grades: dict[Literal["0", "1", "2", "3"], NonEmptyStr]

    @model_validator(mode="after")
    def review_policy_is_complete(self) -> Self:
        if self.relevance_guideline_version != GUIDELINE_VERSION:
            raise ValueError(
                "review.relevance_guideline_version must be "
                f"{GUIDELINE_VERSION}",
            )
        if set(self.anchor_grades) != {"0", "1", "2", "3"}:
            raise ValueError("review.anchor_grades must define grades 0-3")
        return self


class NormalizedExactDedup(ArtifactModel):
    enabled: bool


class LexicalDedup(ArtifactModel):
    token_jaccard_review_at: Fraction
    char_ngram_review_at: Fraction
    auto_duplicate_at: Fraction
    char_ngram_size: PositiveInt = 3

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> Self:
        if self.token_jaccard_review_at >= self.auto_duplicate_at:
            raise ValueError(
                "token Jaccard review threshold must be below auto_duplicate_at",
            )
        if self.char_ngram_review_at >= self.auto_duplicate_at:
            raise ValueError(
                "character n-gram review threshold must be below auto_duplicate_at",
            )
        return self


class SemanticDedup(ArtifactModel):
    enabled: bool = False
    model_id: NonEmptyStr | None
    model_revision: NonEmptyStr | None = None
    model_digest: Sha256 | None = None
    endpoint: NonEmptyStr | None = None
    cache_dir: NonEmptyStr = "outputs/thesis/cache/question_dedup_embeddings"
    batch_size: PositiveInt = 32
    review_at: Fraction
    answer_review_at: Fraction
    question_only_review_at: Fraction
    answer_rescue_question_review_at: Fraction
    answer_rescue_answer_review_at: Fraction
    cross_language_question_review_at: Fraction
    cross_language_answer_review_at: Fraction

    @model_validator(mode="after")
    def proposal_thresholds_are_ordered(self) -> Self:
        if self.question_only_review_at < self.review_at:
            raise ValueError(
                "question-only semantic threshold must not be below review_at",
            )
        if self.answer_rescue_question_review_at > self.review_at:
            raise ValueError(
                "answer-rescue question threshold must not exceed review_at",
            )
        if self.answer_rescue_answer_review_at < self.answer_review_at:
            raise ValueError(
                "answer-rescue answer threshold must not be below answer_review_at",
            )
        if self.cross_language_question_review_at > self.review_at:
            raise ValueError(
                "cross-language question threshold must not exceed review_at",
            )
        if self.cross_language_answer_review_at < self.answer_review_at:
            raise ValueError(
                "cross-language answer threshold must not be below answer_review_at",
            )
        return self

    @model_validator(mode="after")
    def model_is_identified(self) -> Self:
        if self.enabled and (self.model_id is None or self.endpoint is None):
            raise ValueError(
                "enabled semantic deduplication requires model_id and endpoint",
            )
        if self.enabled and self.model_revision is None and self.model_digest is None:
            raise ValueError(
                "enabled semantic deduplication requires model_revision or model_digest",
            )
        return self


class SplitLeakageDedup(ArtifactModel):
    block_unresolved_cross_split: bool


class TranslationEquivalentDedup(ArtifactModel):
    keep_useful_pairs: bool


class DedupAdjudication(ArtifactModel):
    enabled: bool = True
    prompt: PromptConfig
    cache_dir: NonEmptyStr = "outputs/thesis/cache/question_dedup_adjudication"
    batch_size: PositiveInt = 10
    maximum_adjudication_pairs: PositiveInt
    local_llm: ReviewLocalModelConfig


class DeduplicationConfig(ArtifactModel):
    normalized_exact: NormalizedExactDedup
    lexical: LexicalDedup
    semantic: SemanticDedup
    split_leakage: SplitLeakageDedup
    translation_equivalents: TranslationEquivalentDedup
    adjudication: DedupAdjudication


class ManualUtteranceDiagnostic(ArtifactModel):
    enabled: bool
    training_seen_by_legacy_nlu: bool
    exclude_from_primary_validation_and_test: bool


class QuestionConfig(ArtifactModel):
    """Complete question-set configuration with semantic invariants."""

    question_config_id: NonEmptyStr
    output_dir: NonEmptyStr
    source_artifacts: QuestionSourceArtifacts
    artifact_paths: QuestionArtifactPaths
    identity: QuestionIdentity
    splits: SplitConfig
    generation: GenerationConfig
    deterministic_filters: DeterministicFilters
    review: ReviewConfig
    deduplication: DeduplicationConfig
    manual_utterance_diagnostic: ManualUtteranceDiagnostic
    schema_version: Literal[QUESTION_CONFIG_SCHEMA_VERSION] = (
        QUESTION_CONFIG_SCHEMA_VERSION
    )
