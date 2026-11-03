"""Strict, versioned contracts for evaluation question-set artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import Field, field_validator, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.eval.question_config import QuestionConfig, TemporalClassification
from thesis_rag.interfaces import EvidenceTarget, JsonDict, RagQuery, TargetLevel
from thesis_rag.schema import (
    DEDUP_CLUSTER_SCHEMA_VERSION,
    DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION,
    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    QUESTION_DEDUP_REPORT_SCHEMA_VERSION,
    GENERATION_UNIT_SCHEMA_VERSION,
    GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
    GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
    HUMAN_REVIEW_AGREEMENT_SCHEMA_VERSION,
    HUMAN_REVIEW_PLAN_SCHEMA_VERSION,
    HUMAN_REVIEW_SAMPLING_REPORT_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    QUESTION_CONFIG_SCHEMA_VERSION,
    QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
    QUESTION_SCHEMA_VERSION,
    QUESTION_SET_MANIFEST_SCHEMA_VERSION,
    SOURCE_FAMILY_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    config_hash,
    hash_jsonable,
    stable_id,
    validate_schema_version,
)


QUESTION_REVIEW_GUIDELINE_VERSION = "question_review_v26"
SOURCE_FAMILY_RULE_VERSION = "source_family_rules_v2"
GENERATION_UNIT_RULE_VERSION = "generation_unit_rules_v7"
QUESTION_CANDIDATE_RULE_VERSION = "question_candidate_rules_v7"
DEDUP_RULE_VERSION = "question_dedup_rules_v11"
SPLIT_RULE_VERSION = "source_family_split_rules_v2"


class QuestionRecord(RagQuery):
    """Persisted evaluation query, including manual diagnostic questions."""

    schema_version: Literal[QUESTION_SCHEMA_VERSION] = QUESTION_SCHEMA_VERSION

Language = Literal["de", "en"]
LanguageCoverage = Literal[
    "de_only",
    "en_only",
    "de_en_counterpart",
    "de_en_unpaired",
]
RelationshipKind = Literal[
    "exact_duplicate",
    "same_webpage",
    "same_base_webpage",
    "fragment_subrecord",
    "language_counterpart",
]
GenerationUnitKind = Literal[
    "context_group",
    "table_context",
    "page_synthesis",
]
RequestedEvidenceScope = Literal[
    "localized",
    "same_page_multi_section",
    "flexible",
]
Split = Literal["development", "validation", "test", "unassigned"]
SourceMethod = Literal[
    "local_llm_generated",
    "human_authored",
    "human_rewritten",
    "manual_repair",
    "legacy_manual_utterance",
]
PrimaryQuestionType = Literal[
    "factual",
    "procedural",
    "requirements_or_eligibility",
    "navigational_or_contact",
    "comparison_or_selection",
    "other",
]
UserRole = Literal[
    "public",
    "researcher",
    "staff",
    "student",
    "visitor",
]
QuestionTrait = Literal[
    "ambiguous_intentional",
    "temporal",
    "deadline",
    "table_grounded",
    "bilingual_counterpart",
    "same_page_multi_section",
    "short_or_underspecified",
    "form_or_portal",
    "study_program_specific",
]
SurfaceForm = Literal[
    "complete_question",
    "conversational_short",
    "telegraphic_keywords",
    "other",
]
QueryFraming = Literal[
    "direct_lookup",
    "situational_request",
]
GenerationBatchPurpose = Literal["initial", "post_filter_review", "post_dedup"]
ReviewReasonCode = Literal[
    "unsupported_by_evidence",
    "missing_required_facts",
    "reference_answer_incomplete",
    "reference_answer_overinclusive",
    "unnatural_query",
    "not_user_realistic",
    "too_broad_for_anchor",
    "accidental_ambiguity",
    "answer_leakage",
    "wrong_language",
    "missing_context_reference",
    "temporal_policy_reject",
    "multi_section_not_necessary",
    "source_shaped_query",
    "review_output_inconsistent",
    "review_technical_failure",
]
FilterStatus = Literal["raw", "passed", "flagged", "rejected"]
GenerationAttemptStatus = Literal[
    "succeeded",
    "not_applicable",
    "model_request_failed",
    "malformed_response",
    "invalid_candidate",
    "evidence_alignment_failed",
]
ReviewerKind = Literal["local_llm", "human", "system"]
ReviewDecision = Literal[
    "accept",
    "edit_required",
    "reject",
    "needs_human_review",
    "supersede",
]
ReviewEndpointType = Literal[
    "injected_review_provider",
    "openai_compatible_endpoint",
]
ReviewIndependenceMode = Literal[
    "same_exact_model_strict_prompt",
    "different_exact_model_identity",
]
AnchorStatus = Literal["accepted", "rejected", "needs_repair"]
SufficiencyRule = Literal["all_of", "any_of", "page_contains_all"]
DedupClusterKind = Literal[
    "exact_duplicate",
    "same_language_paraphrase",
    "translation_equivalent",
    "partial_overlap",
    "similar_but_distinct",
    "uncertain",
]
DedupDecision = Literal[
    "keep_all",
    "canonicalize",
    "retain_variants",
    "manual_review",
]
DedupQueryRelation = Literal[
    "equivalent",
    "partial_overlap",
    "distinct",
    "uncertain",
]
DedupAnswerRelation = Literal[
    "equivalent",
    "complementary",
    "conflicting",
    "unrelated",
    "uncertain",
]
ManifestStatus = Literal["partial", "reviewed", "sealed"]
FactRef = Annotated[str, Field(pattern=r"^fact:[1-9][0-9]*$")]


def source_family_id(
    *,
    corpus_hash: str,
    member_page_ids: list[str],
    construction_config_hash: str,
    rule_version: str = SOURCE_FAMILY_RULE_VERSION,
) -> str:
    """Build a deterministic source-family ID from stable source members."""

    return stable_id(
        "source_family",
        rule_version,
        corpus_hash,
        sorted(member_page_ids),
        construction_config_hash,
    )


def generation_unit_id(
    *,
    source_family_id_value: str,
    unit_kind: str,
    input_refs: list[JsonDict],
    selection_config_hash: str,
    rule_version: str = GENERATION_UNIT_RULE_VERSION,
) -> str:
    """Build a deterministic generation-unit ID from its exact input refs."""

    return stable_id(
        "generation_unit",
        rule_version,
        source_family_id_value,
        unit_kind,
        input_refs,
        selection_config_hash,
    )


def generation_unit_priority_key(
    *,
    random_seed: int,
    split: str,
    query_language: str,
    source_family_id_value: str,
    page_id: str,
    unit_kind: str,
    input_refs: list[JsonDict],
) -> str:
    """Build a deterministic ordering key for adaptive generation backfill."""

    return stable_id(
        "generation_unit_priority",
        random_seed,
        split,
        query_language,
        source_family_id_value,
        page_id,
        unit_kind,
        input_refs,
    )


def split_manifest_id(
    *,
    corpus_hash: str,
    source_snapshot_hash: str,
    source_families_hash: str,
    split_config_hash: str,
    split_rule_version: str = SPLIT_RULE_VERSION,
) -> str:
    """Build a deterministic ID for one source-family split assignment."""

    return stable_id(
        "split_manifest",
        split_rule_version,
        corpus_hash,
        source_snapshot_hash,
        source_families_hash,
        split_config_hash,
    )


def question_candidate_id(
    *,
    source_method: str,
    source_family_id_value: str,
    generation_unit_id_value: str | None,
    question_text: str,
    query_language: str,
    user_roles: list[str],
    reference_answer: str,
    required_facts: list[str],
    generation_attempt_id: str | None = None,
    parent_candidate_id: str | None = None,
) -> str:
    """Build a deterministic ID for one immutable candidate revision."""

    return stable_id(
        "question_candidate",
        QUESTION_CANDIDATE_RULE_VERSION,
        source_method,
        source_family_id_value,
        generation_unit_id_value,
        question_text,
        query_language,
        user_roles,
        reference_answer,
        required_facts,
        generation_attempt_id,
        parent_candidate_id,
    )


def review_event_id(event: JsonDict) -> str:
    """Build a deterministic review-event ID from immutable event content."""

    return stable_id(
        "question_review_event",
        {
            key: value
            for key, value in event.items()
            if key != "review_event_id"
        },
    )


def human_review_plan_id(
    *,
    base_question_set_id_value: str,
    base_manifest_hash: str,
    selection_config_hash: str,
    source_questions_hash: str,
    source_dedup_report_hash: str,
    strata: list[JsonDict],
    split_fallback_orders: JsonDict,
    double_review_query_ids: list[str],
    double_accepted_test_target: int,
) -> str:
    """Build the immutable identity of one frozen human-review selection."""

    return stable_id(
        "human_review_plan",
        base_question_set_id_value,
        base_manifest_hash,
        selection_config_hash,
        source_questions_hash,
        source_dedup_report_hash,
        strata,
        split_fallback_orders,
        double_review_query_ids,
        double_accepted_test_target,
    )


def development_calibration_cohort_id(
    *,
    base_question_set_id_value: str,
    base_manifest_hash: str,
    corpus_hash: str,
    source_snapshot_hash: str,
    source_artifact_name: str,
    source_questions_hash: str,
    human_review_plan_id_value: str,
    human_review_plan_hash: str,
    selection_version: str,
    selection_config_hash: str,
    question_config_id: str,
    question_config_hash: str,
    development_review_events_hash: str,
    active_acceptances: list[JsonDict],
    ordered_query_ids: list[str],
    questions_hash: str,
    question_count: int,
    questions_byte_size: int,
    materialization_version: str,
    software_provenance: JsonDict,
) -> str:
    """Build the immutable identity of one Development calibration cohort."""

    return stable_id(
        "development_calibration_cohort",
        base_question_set_id_value,
        base_manifest_hash,
        corpus_hash,
        source_snapshot_hash,
        source_artifact_name,
        source_questions_hash,
        human_review_plan_id_value,
        human_review_plan_hash,
        selection_version,
        selection_config_hash,
        question_config_id,
        question_config_hash,
        development_review_events_hash,
        active_acceptances,
        ordered_query_ids,
        questions_hash,
        question_count,
        questions_byte_size,
        materialization_version,
        software_provenance,
    )


def question_set_id(
    *,
    seal_kind: Literal["base", "human"],
    corpus_hash: str,
    source_snapshot_hash: str,
    base_question_set_id_value: str | None,
    human_review_plan_id_value: str | None,
    artifact_hashes: JsonDict,
    config_hashes: JsonDict,
) -> str:
    """Build the immutable identity of one sealed evaluation question set."""

    if seal_kind == "base":
        return stable_id(
            "base_question_set",
            corpus_hash,
            source_snapshot_hash,
            artifact_hashes,
            config_hashes,
        )
    if base_question_set_id_value is None or human_review_plan_id_value is None:
        raise ValueError("human question-set identity requires its base and review plan")
    return stable_id(
        "human_question_set",
        base_question_set_id_value,
        corpus_hash,
        source_snapshot_hash,
        human_review_plan_id_value,
        artifact_hashes,
        config_hashes,
    )


def question_dedup_cluster_id(
    *,
    cluster_kind: str,
    candidate_ids: list[str],
    retained_candidate_ids: list[str],
    decision: str,
    canonical_candidate_id: str | None,
    construction_config_hash: str,
    corpus_hash: str,
) -> str:
    """Build a deterministic ID for a candidate-level deduplication cluster."""

    return stable_id(
        "question_dedup_cluster",
        DEDUP_RULE_VERSION,
        cluster_kind,
        sorted(candidate_ids),
        sorted(retained_candidate_ids),
        decision,
        canonical_candidate_id,
        construction_config_hash,
        corpus_hash,
    )


def question_dedup_manual_decision_id(decision: JsonDict) -> str:
    """Build a deterministic ID from one immutable human dedup decision."""

    return stable_id(
        "question_dedup_manual_decision",
        {
            key: value
            for key, value in decision.items()
            if key != "manual_decision_id"
        },
    )


def generation_batch_manifest_id(
    *,
    batch_index: int,
    previous_batch_manifest_id: str | None,
    generation_units_hash: str,
    question_config_hash: str,
    selector_policy_hash: str,
    model_config_hash: str,
    structured_output_schema_hash: str,
    generation_software_version: str,
    prompt_hashes: JsonDict,
    batch_purpose: GenerationBatchPurpose,
    trigger_artifact_hashes: JsonDict,
    deficit_snapshot: list[JsonDict],
    selection_identity: list[JsonDict],
) -> str:
    """Build the content-derived ID for one generation reservation batch."""

    return stable_id(
        "generation_batch_manifest",
        batch_index,
        previous_batch_manifest_id,
        generation_units_hash,
        question_config_hash,
        selector_policy_hash,
        model_config_hash,
        structured_output_schema_hash,
        generation_software_version,
        prompt_hashes,
        batch_purpose,
        trigger_artifact_hashes,
        deficit_snapshot,
        selection_identity,
    )


def _require_sorted_unique(
    values: list[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_unique(values: list[str], label: str, *, allow_empty: bool = True) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _validate_count_tree(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if isinstance(item, dict):
            _validate_count_tree(item, f"{label}.{key}")
        elif not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")


def _normalise_counts(
    value: JsonDict,
    *,
    outer_keys: set[str],
    inner_keys: set[str],
    label: str,
) -> dict[str, dict[str, int]]:
    if set(value) != outer_keys:
        raise ValueError(f"{label} must define exactly {sorted(outer_keys)}")
    normalised: dict[str, dict[str, int]] = {}
    for outer_key in sorted(outer_keys):
        inner = value[outer_key]
        if not isinstance(inner, dict):
            raise ValueError(f"{label}.{outer_key} must be an object")
        unsupported = set(inner) - inner_keys
        if unsupported:
            raise ValueError(
                f"{label}.{outer_key} has unsupported counts: {sorted(unsupported)}",
            )
        normalised[outer_key] = {
            key: int(inner.get(key, 0))
            for key in sorted(inner_keys)
        }
    return normalised


class SourceFamilyMember(ArtifactModel):
    """One page record participating in a deterministic source family."""

    page_id: NonEmptyStr
    page_key: NonEmptyStr
    language: Language
    eligible_question_generation: bool
    eligible_standard_index: bool
    canonical_page_id: NonEmptyStr
    exact_duplicate: bool = False
    url: NonEmptyStr | None = None
    normalized_url: NonEmptyStr | None = None
    normalized_base_url: NonEmptyStr | None = None
    fragment: NonEmptyStr | None = None
    language_counterpart_page_ids: list[NonEmptyStr] = Field(default_factory=list)
    content_flags: list[NonEmptyStr] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def flags_are_unique(self) -> Self:
        _require_sorted_unique(
            self.language_counterpart_page_ids,
            "SourceFamilyMember.language_counterpart_page_ids",
            allow_empty=True,
        )
        _require_sorted_unique(
            self.content_flags,
            "SourceFamilyMember.content_flags",
            allow_empty=True,
        )
        return self


class SourceFamilyRelationshipEdge(ArtifactModel):
    """Compact relationship edge embedded in a source-family artifact."""

    source_relationship_id: NonEmptyStr
    relationship_kind: RelationshipKind
    member_page_ids: list[NonEmptyStr]

    @model_validator(mode="after")
    def members_are_stable(self) -> Self:
        _require_sorted_unique(
            self.member_page_ids,
            "SourceFamilyRelationshipEdge.member_page_ids",
        )
        if len(self.member_page_ids) < 2:
            raise ValueError("source-family relationship edges need at least two members")
        return self


class SourceFamilyRecord(ArtifactModel):
    """Deterministic transitive closure over source relationship records."""

    source_family_id: NonEmptyStr
    corpus_hash: Sha256
    members: list[SourceFamilyMember]
    relationship_ids: list[NonEmptyStr]
    relationship_kinds: list[RelationshipKind]
    relationship_edges: list[SourceFamilyRelationshipEdge]
    language_coverage: LanguageCoverage
    generation_eligible_page_ids: list[NonEmptyStr]
    family_metadata: JsonDict
    source_family_rule_version: NonEmptyStr
    construction_config_hash: Sha256
    schema_version: Literal[SOURCE_FAMILY_SCHEMA_VERSION] = SOURCE_FAMILY_SCHEMA_VERSION

    @model_validator(mode="after")
    def family_is_internally_consistent(self) -> Self:
        member_ids = [member.page_id for member in self.members]
        _require_sorted_unique(member_ids, "SourceFamilyRecord.members.page_id")
        _require_sorted_unique(
            self.relationship_ids,
            "SourceFamilyRecord.relationship_ids",
            allow_empty=True,
        )
        _require_sorted_unique(
            list(self.relationship_kinds),
            "SourceFamilyRecord.relationship_kinds",
            allow_empty=True,
        )
        edge_ids = [edge.source_relationship_id for edge in self.relationship_edges]
        _require_sorted_unique(
            edge_ids,
            "SourceFamilyRecord.relationship_edges.source_relationship_id",
            allow_empty=True,
        )
        if edge_ids != self.relationship_ids:
            raise ValueError("relationship_ids must exactly match relationship_edges")
        edge_kinds = sorted({edge.relationship_kind for edge in self.relationship_edges})
        if edge_kinds != self.relationship_kinds:
            raise ValueError("relationship_kinds must exactly match relationship_edges")
        member_id_set = set(member_ids)
        for edge in self.relationship_edges:
            if not set(edge.member_page_ids) <= member_id_set:
                raise ValueError("relationship_edges may only reference family members")

        expected_counterparts: dict[str, set[str]] = {
            page_id: set() for page_id in member_ids
        }
        for edge in self.relationship_edges:
            if edge.relationship_kind != "language_counterpart":
                continue
            for page_id in edge.member_page_ids:
                expected_counterparts[page_id].update(
                    other_id
                    for other_id in edge.member_page_ids
                    if other_id != page_id
                )
        members_by_id = {member.page_id: member for member in self.members}
        for member in self.members:
            expected = sorted(expected_counterparts[member.page_id])
            if member.language_counterpart_page_ids != expected:
                raise ValueError(
                    "member language counterpart IDs must exactly match "
                    "language_counterpart edges",
                )
            if any(
                members_by_id[counterpart_id].language == member.language
                for counterpart_id in expected
            ):
                raise ValueError("language counterparts must use the other language")

        expected_eligible = sorted(
            member.page_id
            for member in self.members
            if member.eligible_question_generation
        )
        if self.generation_eligible_page_ids != expected_eligible:
            raise ValueError(
                "generation_eligible_page_ids must exactly match eligible members",
            )
        _require_sorted_unique(
            self.generation_eligible_page_ids,
            "SourceFamilyRecord.generation_eligible_page_ids",
            allow_empty=True,
        )

        eligible_languages = {
            member.language
            for member in self.members
            if member.page_id in self.generation_eligible_page_ids
        } or {member.language for member in self.members}
        has_usable_counterpart = any(
            member.eligible_standard_index
            and any(
                members_by_id[counterpart_id].eligible_standard_index
                for counterpart_id in member.language_counterpart_page_ids
            )
            for member in self.members
        )
        expected_coverage = (
            "de_en_counterpart"
            if has_usable_counterpart
            else "de_en_unpaired"
            if eligible_languages == {"de", "en"}
            else "de_only"
            if eligible_languages == {"de"}
            else "en_only"
        )
        if self.language_coverage != expected_coverage:
            raise ValueError(
                "language_coverage conflicts with eligible members and direct "
                "usable counterpart edges",
            )

        expected_id = source_family_id(
            corpus_hash=self.corpus_hash,
            member_page_ids=member_ids,
            construction_config_hash=self.construction_config_hash,
            rule_version=self.source_family_rule_version,
        )
        if self.source_family_id != expected_id:
            raise ValueError("source_family_id does not match immutable family content")
        return self


class GenerationUnitInput(ArtifactModel):
    """A page, section, chunk, table, or span used as model-visible context."""

    target_level: TargetLevel
    page_id: NonEmptyStr
    role: NonEmptyStr
    target_id: NonEmptyStr | None = None
    section_id: NonEmptyStr | None = None
    chunk_id: NonEmptyStr | None = None
    chunker_id: NonEmptyStr | None = None
    table_id: NonEmptyStr | None = None
    char_start: NonNegativeInt | None = None
    char_end: NonNegativeInt | None = None
    text_hash: Sha256 | None = None
    source_path: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def target_is_consistent(self) -> Self:
        EvidenceTarget(
            target_level=self.target_level,
            target_id=self.target_id,
            page_id=self.page_id,
            section_id=self.section_id,
            chunk_id=self.chunk_id,
            chunker_id=self.chunker_id,
            char_start=self.char_start,
            char_end=self.char_end,
            text_hash=self.text_hash,
            source_path=self.source_path,
            metadata=self.metadata,
        )
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be supplied together")
        if self.target_level == "page" and self.target_id != self.page_id:
            raise ValueError("page input target_id must equal page_id")
        if self.target_level == "section" and self.target_id != self.section_id:
            raise ValueError("section input target_id must equal section_id")
        if self.target_level == "chunk" and self.target_id != self.chunk_id:
            raise ValueError("chunk input target_id must equal chunk_id")
        if self.table_id is not None and self.section_id is None:
            raise ValueError("table inputs require a containing section_id")
        return self


class GenerationUnitRecord(ArtifactModel):
    """Exact model-visible source context selected for question generation."""

    generation_unit_id: NonEmptyStr
    source_family_id: NonEmptyStr
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    unit_kind: GenerationUnitKind
    query_language: Language
    evidence_language: Language
    page_id: NonEmptyStr
    input_refs: list[GenerationUnitInput]
    context_group_ids: list[NonEmptyStr]
    section_ids: list[NonEmptyStr]
    table_ids: list[NonEmptyStr]
    generation_unit_text_hash: Sha256
    prompt_context_hash: Sha256
    selection_config_hash: Sha256
    max_question_budget: PositiveInt
    priority_key: NonEmptyStr
    split: Split
    selection_reason_codes: list[NonEmptyStr]
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal[GENERATION_UNIT_SCHEMA_VERSION] = (
        GENERATION_UNIT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def unit_is_internally_consistent(self) -> Self:
        if self.query_language != self.evidence_language:
            raise ValueError("query_language must match evidence_language")
        if not self.input_refs:
            raise ValueError("GenerationUnitRecord.input_refs must not be empty")
        if any(input_ref.page_id != self.page_id for input_ref in self.input_refs):
            raise ValueError("every input_ref.page_id must match the unit page_id")

        input_sections = sorted(
            {ref.section_id for ref in self.input_refs if ref.section_id is not None},
        )
        input_tables = sorted(
            {ref.table_id for ref in self.input_refs if ref.table_id is not None},
        )
        if not set(input_sections) <= set(self.section_ids):
            raise ValueError("input_ref section IDs must be declared by section_ids")
        if not set(input_tables) <= set(self.table_ids):
            raise ValueError("input_ref table IDs must be declared by table_ids")
        _require_sorted_unique(
            self.context_group_ids,
            "GenerationUnitRecord.context_group_ids",
            allow_empty=True,
        )
        _require_sorted_unique(
            self.section_ids,
            "GenerationUnitRecord.section_ids",
            allow_empty=True,
        )
        _require_sorted_unique(
            self.table_ids,
            "GenerationUnitRecord.table_ids",
            allow_empty=True,
        )
        _require_sorted_unique(
            self.selection_reason_codes,
            "GenerationUnitRecord.selection_reason_codes",
            allow_empty=True,
        )

        if self.unit_kind == "context_group":
            if len(self.context_group_ids) != 1 or not self.section_ids:
                raise ValueError(
                    "context_group units require one group and Section context",
                )
        elif self.unit_kind == "table_context":
            if not self.table_ids or input_tables != self.table_ids:
                raise ValueError(
                    "table_context table_ids must exactly match its table inputs",
                )
        elif self.unit_kind == "page_synthesis":
            if len(self.context_group_ids) < 2 or len(self.section_ids) < 2:
                raise ValueError(
                    "page_synthesis units require at least two context groups",
                )

        input_dicts = [input_ref.to_dict() for input_ref in self.input_refs]
        expected_id = generation_unit_id(
            source_family_id_value=self.source_family_id,
            unit_kind=self.unit_kind,
            input_refs=input_dicts,
            selection_config_hash=self.selection_config_hash,
        )
        if self.generation_unit_id != expected_id:
            raise ValueError("generation_unit_id does not match immutable unit inputs")
        return self


class EvidenceQuotation(ArtifactModel):
    """Verbatim generator quotation aligned into supplied source text."""

    quotation: NonEmptyStr
    required_fact_refs: list[FactRef]
    source_ref: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def fact_refs_are_unique(self) -> Self:
        _require_unique(
            self.required_fact_refs,
            "EvidenceQuotation.required_fact_refs",
            allow_empty=False,
        )
        return self


class EvidenceGroup(ArtifactModel):
    """Group semantics for targets that are jointly needed."""

    evidence_group_id: NonEmptyStr
    target_ids: list[NonEmptyStr]
    sufficiency_rule: SufficiencyRule
    required_fact_refs: list[FactRef]

    @model_validator(mode="after")
    def references_are_stable(self) -> Self:
        _require_sorted_unique(self.target_ids, "EvidenceGroup.target_ids")
        _require_unique(
            self.required_fact_refs,
            "EvidenceGroup.required_fact_refs",
            allow_empty=False,
        )
        return self


class GenerationAttemptPlan(ArtifactModel):
    """One deterministic local-model call reserved by a generation batch."""

    generation_attempt_id: NonEmptyStr
    local_attempt_index: NonNegativeInt
    generation_seed: NonNegativeInt
    resume_key: Sha256
    input_hash: Sha256
    preferred_role_profile: Literal["student_inclusive", "student_only"] | None
    preferred_query_framing: QueryFraming | None
    prefer_natural_paraphrase: bool
    requested_surface_form: SurfaceForm
    requested_evidence_scope: RequestedEvidenceScope
    requested_traits: list[QuestionTrait]

    @model_validator(mode="after")
    def controls_are_canonical(self) -> Self:
        _require_sorted_unique(
            self.requested_traits,
            "GenerationAttemptPlan.requested_traits",
            allow_empty=True,
        )
        return self


class GenerationBatchUnit(ArtifactModel):
    """Budget consumption and calls reserved for one generation unit."""

    generation_unit_id: NonEmptyStr
    source_family_id: NonEmptyStr
    page_id: NonEmptyStr
    split: Split
    query_language: Language
    deficit_cell_id: NonEmptyStr | None
    prior_consumed_question_budget: NonNegativeInt
    consumed_question_budget: PositiveInt
    cumulative_consumed_question_budget: PositiveInt
    remaining_question_budget: NonNegativeInt
    max_question_budget: PositiveInt
    attempts: list[GenerationAttemptPlan]

    @model_validator(mode="after")
    def budget_reconciles(self) -> Self:
        if self.cumulative_consumed_question_budget != (
            self.prior_consumed_question_budget + self.consumed_question_budget
        ):
            raise ValueError("generation batch unit cumulative budget is inconsistent")
        if self.remaining_question_budget != (
            self.max_question_budget - self.cumulative_consumed_question_budget
        ):
            raise ValueError("generation batch unit remaining budget is inconsistent")
        if len(self.attempts) != self.consumed_question_budget:
            raise ValueError("one attempt must be reserved per consumed question budget")
        indexes = [attempt.local_attempt_index for attempt in self.attempts]
        expected = list(
            range(
                self.prior_consumed_question_budget,
                self.cumulative_consumed_question_budget,
            ),
        )
        if indexes != expected:
            raise ValueError("generation attempt indexes must continue unit consumption")
        _require_unique(
            [attempt.generation_attempt_id for attempt in self.attempts],
            "GenerationBatchUnit generation_attempt_id values",
        )
        _require_unique(
            [attempt.resume_key for attempt in self.attempts],
            "GenerationBatchUnit resume keys",
        )
        return self


class GenerationBatchManifestRecord(ArtifactModel):
    """Append-only reservation of generation opportunities and model calls."""

    generation_batch_manifest_id: NonEmptyStr
    batch_index: NonNegativeInt
    previous_batch_manifest_id: NonEmptyStr | None
    batch_purpose: GenerationBatchPurpose
    trigger_artifact_hashes: dict[NonEmptyStr, Sha256]
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    generation_units_hash: Sha256
    question_config_hash: Sha256
    selector_policy_hash: Sha256
    model_config_hash: Sha256
    structured_output_schema_hash: Sha256
    generation_software_version: NonEmptyStr
    prompt_hashes: dict[Language, Sha256]
    deficit_snapshot: list[JsonDict]
    selected_units: list[GenerationBatchUnit]
    consumed_question_budget: PositiveInt
    base_seed: NonNegativeInt
    batch_seed: NonNegativeInt
    created_at_utc: NonEmptyStr
    schema_version: Literal[GENERATION_BATCH_MANIFEST_SCHEMA_VERSION] = (
        GENERATION_BATCH_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def manifest_reconciles(self) -> Self:
        if self.batch_purpose == "initial" and self.trigger_artifact_hashes:
            raise ValueError("initial generation batches cannot have trigger artifacts")
        if self.batch_purpose == "post_filter_review" and set(
            self.trigger_artifact_hashes
        ) != {"question_candidates_filtered", "question_review_events"}:
            raise ValueError(
                "post-filter/review batches require filtered-candidate and review-event hashes",
            )
        if self.batch_purpose == "post_dedup" and "question_dedup_clusters" not in (
            self.trigger_artifact_hashes
        ):
            raise ValueError("post-dedup batches require a dedup-cluster hash")
        if set(self.prompt_hashes) != {"de", "en"}:
            raise ValueError("generation batch prompt_hashes must contain de and en")
        if not self.selected_units:
            raise ValueError("generation batch must select at least one unit")
        if self.consumed_question_budget != sum(
            unit.consumed_question_budget for unit in self.selected_units
        ):
            raise ValueError("generation batch consumed budget is inconsistent")
        _require_unique(
            [unit.generation_unit_id for unit in self.selected_units],
            "GenerationBatchManifestRecord generation unit IDs",
        )
        attempts = [
            attempt
            for unit in self.selected_units
            for attempt in unit.attempts
        ]
        _require_unique(
            [attempt.generation_attempt_id for attempt in attempts],
            "GenerationBatchManifestRecord attempt IDs",
        )
        _require_unique(
            [attempt.resume_key for attempt in attempts],
            "GenerationBatchManifestRecord resume keys",
        )
        expected_id = generation_batch_manifest_id(
            batch_index=self.batch_index,
            previous_batch_manifest_id=self.previous_batch_manifest_id,
            generation_units_hash=self.generation_units_hash,
            question_config_hash=self.question_config_hash,
            selector_policy_hash=self.selector_policy_hash,
            model_config_hash=self.model_config_hash,
            structured_output_schema_hash=self.structured_output_schema_hash,
            generation_software_version=self.generation_software_version,
            prompt_hashes=self.prompt_hashes,
            batch_purpose=self.batch_purpose,
            trigger_artifact_hashes=self.trigger_artifact_hashes,
            deficit_snapshot=self.deficit_snapshot,
            selection_identity=[
                {
                    "generation_unit_id": unit.generation_unit_id,
                    "consumed_question_budget": unit.consumed_question_budget,
                    "deficit_cell_id": unit.deficit_cell_id,
                    "prior_consumed_question_budget": (
                        unit.prior_consumed_question_budget
                    ),
                }
                for unit in self.selected_units
            ],
        )
        if self.generation_batch_manifest_id != expected_id:
            raise ValueError(
                "generation_batch_manifest_id does not match immutable batch content",
            )
        return self


class GenerationAttemptResultRecord(ArtifactModel):
    """Sanitized terminal result for one reserved generation attempt."""

    generation_attempt_id: NonEmptyStr
    generation_batch_manifest_id: NonEmptyStr
    generation_unit_id: NonEmptyStr
    source_family_id: NonEmptyStr
    page_id: NonEmptyStr
    split: Split
    query_language: Language
    local_attempt_index: NonNegativeInt
    resume_key: Sha256
    input_hash: Sha256
    generation_seed: NonNegativeInt
    status: GenerationAttemptStatus
    retry_count: NonNegativeInt
    cache_hit: bool
    candidate_id: NonEmptyStr | None
    response_hash: Sha256 | None
    error_code: NonEmptyStr | None
    alignment_summary: JsonDict
    created_at_utc: NonEmptyStr
    schema_version: Literal[GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION] = (
        GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def result_is_terminal(self) -> Self:
        succeeded = self.status == "succeeded"
        if succeeded != (self.candidate_id is not None):
            raise ValueError("only succeeded attempts have candidate_id")
        if succeeded and self.error_code is not None:
            raise ValueError("succeeded attempts cannot have an error_code")
        if not succeeded and self.error_code is None:
            raise ValueError("non-success attempts require a stable reason code")
        return self


def _evidence_target(value: Any) -> EvidenceTarget:
    if isinstance(value, EvidenceTarget):
        return value
    if isinstance(value, dict):
        return EvidenceTarget.from_dict(value)
    raise ValueError("evidence targets must be EvidenceTarget objects or mappings")


class QuestionCandidateRecord(ArtifactModel):
    """Immutable generated, imported, or human-authored question proposal."""

    candidate_id: NonEmptyStr
    source_method: SourceMethod
    source_family_id: NonEmptyStr
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    split: Split = "unassigned"
    question_text: NonEmptyStr
    query_language: Language
    evidence_language: Language
    user_roles: list[UserRole]
    primary_type: PrimaryQuestionType
    traits: list[QuestionTrait]
    surface_form: SurfaceForm
    reference_answer: NonEmptyStr
    required_facts: list[NonEmptyStr]
    evidence_quotations: list[EvidenceQuotation]
    temporal_classification: TemporalClassification
    deterministic_filter_status: FilterStatus
    deterministic_filter_reasons: list[NonEmptyStr]
    candidate_revision: PositiveInt = 1
    parent_candidate_id: NonEmptyStr | None = None
    generation_unit_id: NonEmptyStr | None = None
    generation_attempt_id: NonEmptyStr | None = None
    generator_provenance: JsonDict = Field(default_factory=dict)
    verified_evidence_targets: list[EvidenceTarget] = Field(default_factory=list)
    evidence_groups: list[EvidenceGroup] = Field(default_factory=list)
    query_evidence_lexical_overlap: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal[QUESTION_CANDIDATE_SCHEMA_VERSION] = (
        QUESTION_CANDIDATE_SCHEMA_VERSION
    )

    @field_validator("verified_evidence_targets", mode="before")
    @classmethod
    def parse_evidence_targets(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [_evidence_target(item) for item in value]
        return value

    @field_validator("question_text")
    @classmethod
    def question_has_meaningful_content(cls, value: str) -> str:
        if not value.strip() or not any(character.isalnum() for character in value):
            raise ValueError("question_text must contain a Unicode letter or digit")
        return value

    @field_validator("reference_answer")
    @classmethod
    def answer_has_meaningful_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reference_answer must contain non-whitespace content")
        return value

    @field_validator("required_facts")
    @classmethod
    def facts_have_meaningful_content(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("required_facts must contain non-whitespace content")
        return values

    @model_validator(mode="after")
    def candidate_is_internally_consistent(self) -> Self:
        if self.query_language != self.evidence_language:
            raise ValueError("query_language must match evidence_language")
        _require_sorted_unique(
            list(self.traits),
            "QuestionCandidateRecord.traits",
            allow_empty=True,
        )
        _require_sorted_unique(
            list(self.user_roles),
            "QuestionCandidateRecord.user_roles",
        )
        _require_unique(
            self.required_facts,
            "QuestionCandidateRecord.required_facts",
            allow_empty=False,
        )
        if not self.evidence_quotations:
            raise ValueError("evidence_quotations must not be empty")
        _require_unique(
            self.deterministic_filter_reasons,
            "QuestionCandidateRecord.deterministic_filter_reasons",
        )
        if self.candidate_revision > 1 and self.parent_candidate_id is None:
            raise ValueError("edited revisions require parent_candidate_id")
        if self.source_method == "local_llm_generated":
            if self.generation_unit_id is None:
                raise ValueError("local_llm_generated requires generation_unit_id")
            if self.generation_attempt_id is None:
                raise ValueError("local_llm_generated requires generation_attempt_id")
            if self.split == "unassigned":
                raise ValueError("local_llm_generated requires an assigned split")
            provenance_kind = self.generator_provenance.get("kind")
            if not isinstance(provenance_kind, str) or not provenance_kind:
                raise ValueError("local generation requires generator_provenance.kind")

        valid_fact_refs = {
            f"fact:{index}"
            for index in range(1, len(self.required_facts) + 1)
        }
        quotation_refs = {
            fact_ref
            for quotation in self.evidence_quotations
            for fact_ref in quotation.required_fact_refs
        }
        if quotation_refs - valid_fact_refs:
            raise ValueError("evidence quotations reference unknown required facts")
        if quotation_refs != valid_fact_refs:
            raise ValueError("every required fact must have an evidence quotation")

        target_ids = {
            target.target_id
            for target in self.verified_evidence_targets
            if target.target_id is not None
        }
        for group in self.evidence_groups:
            if set(group.required_fact_refs) - valid_fact_refs:
                raise ValueError("evidence groups reference unknown required facts")
            if set(group.target_ids) - target_ids:
                raise ValueError("evidence groups reference unknown evidence targets")

        expected_id = question_candidate_id(
            source_method=self.source_method,
            source_family_id_value=self.source_family_id,
            generation_unit_id_value=self.generation_unit_id,
            question_text=self.question_text,
            query_language=self.query_language,
            user_roles=list(self.user_roles),
            reference_answer=self.reference_answer,
            required_facts=self.required_facts,
            generation_attempt_id=self.generation_attempt_id,
            parent_candidate_id=self.parent_candidate_id,
        )
        if self.candidate_id != expected_id:
            raise ValueError("candidate_id does not match immutable candidate content")
        return self


class AnchorAssessment(ArtifactModel):
    """Reviewer's grade for one known source anchor."""

    target: EvidenceTarget
    grade: Literal[0, 1, 2, 3]
    status: AnchorStatus
    label: NonEmptyStr
    explanation: NonEmptyStr
    supported_fact_refs: list[FactRef]
    propagation_rule: NonEmptyStr | None = None

    @model_validator(mode="after")
    def fact_refs_are_unique(self) -> Self:
        _require_unique(
            self.supported_fact_refs,
            "AnchorAssessment.supported_fact_refs",
            allow_empty=True,
        )
        return self

    @field_validator("target", mode="before")
    @classmethod
    def parse_target(cls, value: Any) -> EvidenceTarget:
        return _evidence_target(value)


class SufficientEvidenceSet(ArtifactModel):
    """One set of targets whose union is sufficient to answer the question."""

    target_ids: list[NonEmptyStr] = Field(min_length=1)

    @model_validator(mode="after")
    def targets_are_unique(self) -> Self:
        _require_unique(self.target_ids, "SufficientEvidenceSet.target_ids")
        return self


class LocalLlmQuestionReviewProvenance(ArtifactModel):
    """Immutable model, prompt, input, and execution identity for LLM review."""

    kind: Literal["local_llm_question_review"]
    model_id: NonEmptyStr
    model_revision: NonEmptyStr | None
    model_path: NonEmptyStr | None
    model_digest: Sha256 | None
    prompt_id: NonEmptyStr
    prompt_hash: Sha256
    review_input_hash: Sha256
    structured_output_schema_hash: Sha256
    review_policy_hash: Sha256
    review_pipeline_version: NonEmptyStr
    review_seed_base: NonNegativeInt
    relevance_guideline_version: NonEmptyStr
    endpoint_type: ReviewEndpointType
    response_hash: Sha256
    generation_seed: NonNegativeInt
    temperature: Annotated[float, Field(ge=0.0)]
    max_tokens: PositiveInt
    structured_output_mode: Literal["json_schema", "json_object"]
    reasoning_effort: Literal["none", "low", "medium", "high"]
    reported_decision: (
        Literal[
            "accept",
            "edit_required",
            "reject",
            "needs_human_review",
        ]
        | None
    )
    output_consistency: Literal[
        "valid",
        "normalized_to_human_review",
        "technical_failure",
    ]
    retry_count: NonNegativeInt
    reviewed_at_utc: NonEmptyStr
    same_model_id_as_generator: bool
    same_model_revision_as_generator: bool
    same_model_digest_as_generator: bool
    independence_mode: ReviewIndependenceMode

    @model_validator(mode="after")
    def model_identity_is_immutable(self) -> Self:
        if self.model_revision is None and self.model_digest is None:
            raise ValueError("local review provenance needs a model revision or digest")
        if self.output_consistency == "technical_failure":
            if self.reported_decision is not None:
                raise ValueError(
                    "technical review failures cannot claim a reported decision",
                )
        elif self.reported_decision is None:
            raise ValueError(
                "usable local review output requires a reported decision",
            )
        return self


class QuestionReviewEventRecord(ArtifactModel):
    """Append-only validity-review event for a question candidate."""

    review_event_id: NonEmptyStr
    candidate_id: NonEmptyStr
    reviewer_kind: ReviewerKind
    reviewer_id: NonEmptyStr
    guideline_version: NonEmptyStr
    decision: ReviewDecision
    reason_codes: list[ReviewReasonCode]
    inferred_user_roles: list[UserRole] | None
    inferred_query_framing: QueryFraming | None
    natural_paraphrase_present: bool | None
    query_form_fit: bool | None
    source_shaping_present: bool | None
    answer_scope_fit: bool | None
    realistic_user_need: bool | None
    plausible_user_goal: NonEmptyStr | None
    independent_review: bool
    review_summary: NonEmptyStr
    anchor_assessments: list[AnchorAssessment]
    sufficient_evidence_sets: list[SufficientEvidenceSet]
    provenance: JsonDict
    created_at_utc: NonEmptyStr
    query_id: NonEmptyStr | None = None
    new_candidate_id: NonEmptyStr | None = None
    schema_version: Literal[QUESTION_REVIEW_EVENT_SCHEMA_VERSION] = (
        QUESTION_REVIEW_EVENT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def event_is_internally_consistent(self) -> Self:
        _require_unique(
            self.reason_codes,
            "QuestionReviewEventRecord.reason_codes",
        )
        if self.decision == "accept" and self.reason_codes:
            raise ValueError("accepted review events cannot have negative reason codes")
        technical_failure = False
        normalized_output = False
        if self.reviewer_kind == "local_llm":
            llm_provenance = LocalLlmQuestionReviewProvenance.from_dict(
                self.provenance,
            )
            normalized_output = (
                llm_provenance.output_consistency == "normalized_to_human_review"
            )
            technical_failure = (
                llm_provenance.output_consistency == "technical_failure"
            )
        if technical_failure:
            if (
                self.decision != "needs_human_review"
                or self.reason_codes != ["review_technical_failure"]
                or self.inferred_user_roles is not None
                or self.inferred_query_framing is not None
                or self.natural_paraphrase_present is not None
                or self.query_form_fit is not None
                or self.source_shaping_present is not None
                or self.answer_scope_fit is not None
                or self.realistic_user_need is not None
                or self.plausible_user_goal is not None
                or self.anchor_assessments
                or self.sufficient_evidence_sets
            ):
                raise ValueError(
                    "technical review failures must be unassessed human-review events",
                )
        else:
            if (
                self.inferred_user_roles is None
                or self.inferred_query_framing is None
                or self.natural_paraphrase_present is None
                or self.query_form_fit is None
                or self.source_shaping_present is None
                or self.answer_scope_fit is None
                or self.realistic_user_need is None
            ):
                raise ValueError(
                    "ordinary review events require roles, framing, paraphrase, "
                    "form, source-shaping, scope, and user-need judgments",
                )
            _require_sorted_unique(
                list(self.inferred_user_roles),
                "QuestionReviewEventRecord.inferred_user_roles",
            )
            if self.realistic_user_need != (self.plausible_user_goal is not None):
                raise ValueError(
                    "plausible_user_goal is required exactly when "
                    "realistic_user_need is true",
                )
        if self.decision == "accept" and (
            not self.query_form_fit
            or self.source_shaping_present
            or not self.answer_scope_fit
            or not self.realistic_user_need
        ):
            raise ValueError(
                "accepted review events require query-form fit, source-independent "
                "wording, answer-scope fit, and a realistic user need",
            )
        if self.decision in {"edit_required", "reject", "needs_human_review"} and not (
            self.reason_codes
        ):
            raise ValueError("non-accept review events require at least one reason code")
        if self.decision == "accept" and not self.sufficient_evidence_sets:
            raise ValueError("accepted review events require sufficient evidence")
        if self.decision != "accept" and self.sufficient_evidence_sets:
            raise ValueError("non-accept review events cannot claim sufficient evidence")
        evidence_sets = [
            tuple(sorted(evidence_set.target_ids))
            for evidence_set in self.sufficient_evidence_sets
        ]
        if len(evidence_sets) != len(set(evidence_sets)):
            raise ValueError("sufficient evidence sets must be unique")
        provenance_kind = self.provenance.get("kind")
        if not isinstance(provenance_kind, str) or not provenance_kind:
            raise ValueError("QuestionReviewEventRecord.provenance.kind is required")
        if normalized_output:
            if (
                self.decision != "needs_human_review"
                or self.reason_codes != ["review_output_inconsistent"]
                or self.anchor_assessments
            ):
                raise ValueError(
                    "normalized inconsistent reviews must be unassessed human-review events",
                )
        elif not technical_failure and not self.anchor_assessments:
            raise ValueError(
                "ordinary review events require at least one anchor assessment",
            )
        if not self.review_summary.strip():
            raise ValueError("review_summary must contain non-whitespace content")
        expected_id = review_event_id(self.to_dict())
        if self.review_event_id != expected_id:
            raise ValueError("review_event_id does not match immutable review content")
        return self


class QuestionDedupClusterRecord(ArtifactModel):
    """Deterministic duplicate or translation-equivalence cluster."""

    dedup_cluster_id: NonEmptyStr
    cluster_kind: DedupClusterKind
    candidate_ids: list[NonEmptyStr]
    retained_candidate_ids: list[NonEmptyStr]
    canonical_candidate_id: NonEmptyStr | None
    source_family_ids: list[NonEmptyStr]
    splits: list[Split]
    retained_splits: list[Split]
    query_relations: list[DedupQueryRelation]
    answer_relations: list[DedupAnswerRelation]
    decision: DedupDecision
    similarity: JsonDict
    evidence: JsonDict
    dedup_rule_version: NonEmptyStr
    construction_config_hash: Sha256
    corpus_hash: Sha256
    schema_version: Literal[DEDUP_CLUSTER_SCHEMA_VERSION] = (
        DEDUP_CLUSTER_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def cluster_is_internally_consistent(self) -> Self:
        _require_sorted_unique(
            self.candidate_ids,
            "QuestionDedupClusterRecord.candidate_ids",
        )
        if len(self.candidate_ids) < 2:
            raise ValueError("dedup clusters require at least two candidates")
        _require_sorted_unique(
            self.retained_candidate_ids,
            "QuestionDedupClusterRecord.retained_candidate_ids",
        )
        if set(self.retained_candidate_ids) - set(self.candidate_ids):
            raise ValueError("retained candidates must be cluster members")
        if self.decision == "canonicalize":
            if self.retained_candidate_ids != [self.canonical_candidate_id]:
                raise ValueError(
                    "canonicalized clusters must retain only their canonical candidate",
                )
        elif self.decision == "retain_variants":
            if (
                len(self.retained_candidate_ids) < 2
                or self.canonical_candidate_id not in self.retained_candidate_ids
            ):
                raise ValueError(
                    "variant clusters require a retained canonical representative",
                )
        elif self.canonical_candidate_id is not None:
            raise ValueError(
                "keep-all and manual-review clusters cannot select a representative",
            )
        if self.decision in {"keep_all", "manual_review"} and (
            self.retained_candidate_ids != self.candidate_ids
        ):
            raise ValueError("unresolved clusters must retain every candidate")
        _require_sorted_unique(
            self.source_family_ids,
            "QuestionDedupClusterRecord.source_family_ids",
            allow_empty=True,
        )
        _require_sorted_unique(
            self.splits,
            "QuestionDedupClusterRecord.splits",
        )
        _require_sorted_unique(
            self.retained_splits,
            "QuestionDedupClusterRecord.retained_splits",
        )
        if set(self.retained_splits) - set(self.splits):
            raise ValueError("retained splits must be original cluster splits")
        if "unassigned" in self.splits:
            raise ValueError("accepted candidate clusters cannot be unassigned")
        _require_sorted_unique(
            self.query_relations,
            "QuestionDedupClusterRecord.query_relations",
        )
        _require_sorted_unique(
            self.answer_relations,
            "QuestionDedupClusterRecord.answer_relations",
        )
        if (
            "equivalent" in self.query_relations
            and self.decision in {"canonicalize", "retain_variants"}
            and len(self.retained_splits) != 1
        ):
            raise ValueError(
                "resolved equivalent-query clusters must retain one split",
            )
        expected_id = question_dedup_cluster_id(
            cluster_kind=self.cluster_kind,
            candidate_ids=list(self.candidate_ids),
            retained_candidate_ids=list(self.retained_candidate_ids),
            decision=self.decision,
            canonical_candidate_id=self.canonical_candidate_id,
            construction_config_hash=self.construction_config_hash,
            corpus_hash=self.corpus_hash,
        )
        if self.dedup_cluster_id != expected_id:
            raise ValueError("dedup_cluster_id does not match immutable cluster content")
        return self


class QuestionDedupManualDecisionRecord(ArtifactModel):
    """A reviewed resolution for one candidate pair and dedup lineage."""

    manual_decision_id: NonEmptyStr
    pair_id: NonEmptyStr
    candidate_ids: list[NonEmptyStr]
    construction_config_hash: Sha256
    query_relation: DedupQueryRelation
    answer_relation: DedupAnswerRelation
    resolution: Literal["keep_all", "canonicalize", "retain_variants"]
    canonical_candidate_id: NonEmptyStr | None
    reviewer_kind: Literal["human"] = "human"
    reviewer_id: NonEmptyStr
    rationale: NonEmptyStr
    reviewed_at_utc: NonEmptyStr
    schema_version: Literal[QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION] = (
        QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def decision_is_internally_consistent(self) -> Self:
        _require_sorted_unique(
            self.candidate_ids,
            "QuestionDedupManualDecisionRecord.candidate_ids",
        )
        if len(self.candidate_ids) != 2:
            raise ValueError("manual dedup decisions require exactly two candidates")
        expected_pair_id = stable_id("question_dedup_pair", *self.candidate_ids)
        if self.pair_id != expected_pair_id:
            raise ValueError("pair_id does not match candidate_ids")
        if self.resolution == "keep_all":
            if self.canonical_candidate_id is not None:
                raise ValueError("keep-all decisions cannot select a canonical candidate")
        else:
            if self.query_relation != "equivalent" or self.answer_relation not in {
                "equivalent",
                "complementary",
                "conflicting",
            }:
                raise ValueError(
                    "canonicalized or retained variants require equivalent queries "
                    "and equivalent, complementary, or conflicting answers",
                )
            if (
                self.canonical_candidate_id is not None
                and self.canonical_candidate_id not in self.candidate_ids
            ):
                raise ValueError(
                    "a pair-level canonical candidate must be a member of the pair",
                )
        expected_id = question_dedup_manual_decision_id(self.to_dict())
        if self.manual_decision_id != expected_id:
            raise ValueError(
                "manual_decision_id does not match immutable decision content",
            )
        return self


class QuestionDedupReportRecord(ArtifactModel):
    """Reproducible summary of one complete current deduplication view."""

    status: Literal[
        "target_reached",
        "deficits",
        "manual_review_required",
        "selection_infeasible",
    ]
    pipeline_version: NonEmptyStr
    dedup_rule_version: NonEmptyStr
    construction_config_hash: Sha256
    corpus_hash: Sha256
    input_hashes: dict[NonEmptyStr, Sha256]
    output_hashes: dict[NonEmptyStr, Sha256]
    semantic_model: JsonDict
    selection_policy: JsonDict
    selection_policy_hash: Sha256
    selection_method: Literal["greedy_exact", "milp_exact", "infeasible"]
    exact_selection_feasible: bool
    counts: JsonDict
    targets_by_split: JsonDict
    selected_by_split: JsonDict
    deficits_by_split: JsonDict
    distribution_deficits_by_split: JsonDict
    selection_excess_by_split: JsonDict
    selected_distributions: JsonDict
    removed_candidate_ids: list[NonEmptyStr]
    not_selected_candidate_ids: list[NonEmptyStr]
    unresolved_cross_split_cluster_ids: list[NonEmptyStr]
    schema_version: Literal[QUESTION_DEDUP_REPORT_SCHEMA_VERSION] = (
        QUESTION_DEDUP_REPORT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def report_is_canonical(self) -> Self:
        if self.selection_policy_hash != hash_jsonable(self.selection_policy):
            raise ValueError("selection_policy_hash does not match selection_policy")
        for values, label in (
            (self.removed_candidate_ids, "removed_candidate_ids"),
            (self.not_selected_candidate_ids, "not_selected_candidate_ids"),
            (
                self.unresolved_cross_split_cluster_ids,
                "unresolved_cross_split_cluster_ids",
            ),
        ):
            _require_sorted_unique(
                values,
                f"QuestionDedupReportRecord.{label}",
                allow_empty=True,
            )
        return self


class SplitFamilyAssignment(ArtifactModel):
    """Split assignment for one source family."""

    source_family_id: NonEmptyStr
    split: Split
    language_coverage: LanguageCoverage
    member_count: PositiveInt
    generation_eligible_page_count: NonNegativeInt
    generation_eligible_page_ids: list[NonEmptyStr]
    estimated_question_capacity: NonNegativeInt
    assignment_reason: NonEmptyStr

    @model_validator(mode="after")
    def assignment_is_internally_consistent(self) -> Self:
        _require_sorted_unique(
            self.generation_eligible_page_ids,
            "SplitFamilyAssignment.generation_eligible_page_ids",
            allow_empty=True,
        )
        if self.generation_eligible_page_count != len(
            self.generation_eligible_page_ids,
        ):
            raise ValueError(
                "generation_eligible_page_count must match page IDs",
            )
        if self.generation_eligible_page_count > self.member_count:
            raise ValueError("eligible page count cannot exceed member_count")
        if self.split == "unassigned" and self.generation_eligible_page_count:
            raise ValueError("eligible families must receive a query split")
        if self.split != "unassigned" and not self.generation_eligible_page_count:
            raise ValueError("zero-eligible families must be unassigned")
        if bool(self.generation_eligible_page_count) != bool(
            self.estimated_question_capacity
        ):
            raise ValueError(
                "estimated question capacity must be positive exactly for "
                "generation-eligible families",
            )
        return self


class SplitManifestRecord(ArtifactModel):
    """Reproducible source-family split manifest."""

    split_manifest_id: NonEmptyStr
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    snapshot_manifest_hash: Sha256
    source_family_rule_version: NonEmptyStr
    split_rule_version: NonEmptyStr
    random_seed: NonNegativeInt
    source_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    source_artifact_hashes: dict[NonEmptyStr, Sha256]
    output_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    output_artifact_hashes: dict[NonEmptyStr, Sha256]
    config_hashes: dict[NonEmptyStr, Sha256]
    target_ratios: dict[
        Literal["development", "validation", "test"],
        Annotated[float, Field(ge=0.0, le=1.0)],
    ]
    assignments: list[SplitFamilyAssignment]
    counts_by_split: JsonDict
    counts_by_language_coverage: JsonDict
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal[SPLIT_MANIFEST_SCHEMA_VERSION] = (
        SPLIT_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def manifest_is_internally_consistent(self) -> Self:
        if set(self.target_ratios) != {"development", "validation", "test"}:
            raise ValueError("target_ratios must define development, validation, test")
        if abs(sum(self.target_ratios.values()) - 1.0) > 1e-9:
            raise ValueError("target_ratios values must sum to 1.0")
        if not self.assignments:
            raise ValueError("SplitManifestRecord.assignments must not be empty")
        assignment_ids = [item.source_family_id for item in self.assignments]
        _require_sorted_unique(
            assignment_ids,
            "SplitManifestRecord.assignments.source_family_id",
        )
        _validate_count_tree(self.counts_by_split, "counts_by_split")
        _validate_count_tree(
            self.counts_by_language_coverage,
            "counts_by_language_coverage",
        )

        split_names = {"development", "validation", "test", "unassigned"}
        split_count_names = {
            "families",
            "members",
            "generation_eligible_pages",
            "estimated_question_capacity",
        }
        expected_split_counts = {
            split: {name: 0 for name in split_count_names}
            for split in split_names
        }
        for assignment in self.assignments:
            counts = expected_split_counts[assignment.split]
            counts["families"] += 1
            counts["members"] += assignment.member_count
            counts["generation_eligible_pages"] += (
                assignment.generation_eligible_page_count
            )
            counts["estimated_question_capacity"] += (
                assignment.estimated_question_capacity
            )
        actual_split_counts = _normalise_counts(
            self.counts_by_split,
            outer_keys=split_names,
            inner_keys=split_count_names,
            label="counts_by_split",
        )
        if actual_split_counts != {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(expected_split_counts.items())
        }:
            raise ValueError("counts_by_split disagrees with assignments")

        coverage_count_names = split_names | {"total"}
        expected_coverage_counts = {
            coverage: {name: 0 for name in coverage_count_names}
            for coverage in get_args(LanguageCoverage)
        }
        for assignment in self.assignments:
            counts = expected_coverage_counts[assignment.language_coverage]
            counts["total"] += 1
            counts[assignment.split] += 1
        actual_coverage_counts = _normalise_counts(
            self.counts_by_language_coverage,
            outer_keys=set(get_args(LanguageCoverage)),
            inner_keys=coverage_count_names,
            label="counts_by_language_coverage",
        )
        if actual_coverage_counts != {
            coverage: dict(sorted(counts.items()))
            for coverage, counts in sorted(expected_coverage_counts.items())
        }:
            raise ValueError("counts_by_language_coverage disagrees with assignments")

        source_families_hash = self.output_artifact_hashes.get("source_families")
        split_config_hash = self.config_hashes.get("splits")
        if source_families_hash is None or split_config_hash is None:
            raise ValueError(
                "split manifest requires source_families and splits provenance hashes",
            )
        expected_id = split_manifest_id(
            corpus_hash=self.corpus_hash,
            source_snapshot_hash=self.source_snapshot_hash,
            source_families_hash=source_families_hash,
            split_config_hash=split_config_hash,
            split_rule_version=self.split_rule_version,
        )
        if self.split_manifest_id != expected_id:
            raise ValueError("split_manifest_id does not match immutable provenance")
        return self


class QuestionSetManifestRecord(ArtifactModel):
    """Detached verification manifest for one sealed question-set archive."""

    question_set_id: NonEmptyStr
    seal_kind: Literal["base", "human"]
    base_question_set_id: NonEmptyStr
    base_manifest_path: NonEmptyStr | None = None
    base_manifest_hash: Sha256 | None = None
    status: ManifestStatus
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    snapshot_manifest_hash: Sha256
    corpus_archive_hash: Sha256
    source_relationships_hash: Sha256
    source_families_hash: Sha256
    generation_units_hash: Sha256
    split_manifest_id: NonEmptyStr
    split_manifest_hash: Sha256
    human_review_plan_id: NonEmptyStr | None = None
    artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    artifact_hashes: dict[NonEmptyStr, Sha256]
    artifact_counts: JsonDict
    artifact_byte_sizes: dict[NonEmptyStr, NonNegativeInt]
    archive_members: dict[NonEmptyStr, NonEmptyStr]
    archive_path: NonEmptyStr
    archive_hash: Sha256
    archive_byte_size: PositiveInt
    archive_compression: Literal["gzip"]
    config_hashes: dict[NonEmptyStr, Sha256]
    split_counts: JsonDict
    tier_counts: JsonDict
    review_counts: JsonDict
    subset_relationships: JsonDict
    random_seeds: JsonDict
    rule_versions: JsonDict
    model_provenance: JsonDict
    prompt_provenance: JsonDict
    software_provenance: JsonDict
    review_guideline_version: NonEmptyStr
    limitations: list[str]
    created_at_utc: NonEmptyStr
    updated_at_utc: NonEmptyStr
    schema_version: Literal[QUESTION_SET_MANIFEST_SCHEMA_VERSION] = (
        QUESTION_SET_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def manifest_values_are_consistent(self) -> Self:
        _validate_count_tree(self.artifact_counts, "artifact_counts")
        _validate_count_tree(self.artifact_byte_sizes, "artifact_byte_sizes")
        _validate_count_tree(self.split_counts, "split_counts")
        _validate_count_tree(self.tier_counts, "tier_counts")
        _validate_count_tree(self.review_counts, "review_counts")
        _validate_count_tree(self.random_seeds, "random_seeds")
        artifact_names = set(self.artifact_paths)
        for values, label in (
            (self.artifact_hashes, "artifact_hashes"),
            (self.artifact_counts, "artifact_counts"),
            (self.artifact_byte_sizes, "artifact_byte_sizes"),
            (self.archive_members, "archive_members"),
        ):
            if set(values) != artifact_names:
                raise ValueError(
                    f"{label} must define exactly the artifact_paths keys",
                )
        members = list(self.archive_members.values())
        if len(members) != len(set(members)):
            raise ValueError("archive member paths must be unique")
        expected_id = question_set_id(
            seal_kind=self.seal_kind,
            corpus_hash=self.corpus_hash,
            source_snapshot_hash=self.source_snapshot_hash,
            base_question_set_id_value=(
                None if self.seal_kind == "base" else self.base_question_set_id
            ),
            human_review_plan_id_value=self.human_review_plan_id,
            artifact_hashes=self.artifact_hashes,
            config_hashes=self.config_hashes,
        )
        if self.question_set_id != expected_id:
            raise ValueError("question_set_id does not match immutable inputs")
        if self.seal_kind == "base":
            if self.base_question_set_id != self.question_set_id:
                raise ValueError("base seal must identify itself as the base question set")
            if any(
                value is not None
                for value in (
                    self.base_manifest_path,
                    self.base_manifest_hash,
                    self.human_review_plan_id,
                )
            ):
                raise ValueError("base seal cannot reference human or parent artifacts")
        elif (
            self.base_question_set_id == self.question_set_id
            or self.base_manifest_path is None
            or self.base_manifest_hash is None
            or self.human_review_plan_id is None
        ):
            raise ValueError(
                "human seal requires a distinct base ID, base manifest, and review plan",
            )
        return self


class HumanReviewStratumRecord(ArtifactModel):
    """One proportional human-review stratum with a frozen reserve order."""

    stratum_id: NonEmptyStr
    split: Literal["development", "validation", "test"]
    language: Language
    surface_form: SurfaceForm
    evidence_scope: Literal["localized", "same_page_multi_section"]
    primary_target: NonNegativeInt
    double_review_target: NonNegativeInt
    double_accepted_target: NonNegativeInt
    ordered_query_ids: list[NonEmptyStr]

    @model_validator(mode="after")
    def stratum_is_consistent(self) -> Self:
        _require_unique(
            self.ordered_query_ids,
            "HumanReviewStratumRecord.ordered_query_ids",
        )
        if self.primary_target > len(self.ordered_query_ids):
            raise ValueError("human-review primary target exceeds stratum capacity")
        if self.double_review_target > self.primary_target:
            raise ValueError("double-review target exceeds the primary target")
        if self.double_accepted_target > self.primary_target:
            raise ValueError("double-accepted target exceeds the primary target")
        expected_id = stable_id(
            "human_review_stratum",
            self.split,
            self.language,
            self.surface_form,
            self.evidence_scope,
        )
        if self.stratum_id != expected_id:
            raise ValueError("stratum_id does not match immutable stratum values")
        return self


class HumanReviewPlanRecord(ArtifactModel):
    """Frozen proportional selection and same-stratum replacement order."""

    human_review_plan_id: NonEmptyStr
    base_question_set_id: NonEmptyStr
    base_manifest_path: NonEmptyStr
    base_manifest_hash: Sha256
    selection_version: NonEmptyStr
    selection_config_hash: Sha256
    source_questions_hash: Sha256
    source_dedup_report_hash: Sha256
    targets_by_split: dict[
        Literal["development", "validation", "test"],
        NonNegativeInt,
    ]
    double_review_test_target: NonNegativeInt
    double_review_query_ids: list[NonEmptyStr]
    double_accepted_test_target: NonNegativeInt
    strata: list[HumanReviewStratumRecord]
    split_fallback_orders: dict[
        Literal["development", "validation", "test"],
        list[NonEmptyStr],
    ]
    created_at_utc: NonEmptyStr
    schema_version: Literal[HUMAN_REVIEW_PLAN_SCHEMA_VERSION] = (
        HUMAN_REVIEW_PLAN_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def plan_is_consistent(self) -> Self:
        if set(self.targets_by_split) != {"development", "validation", "test"}:
            raise ValueError("human-review targets must define all benchmark splits")
        stratum_ids = [item.stratum_id for item in self.strata]
        _require_sorted_unique(stratum_ids, "HumanReviewPlanRecord.strata")
        query_ids = [
            query_id for item in self.strata for query_id in item.ordered_query_ids
        ]
        _require_unique(query_ids, "HumanReviewPlanRecord query IDs")
        if set(self.split_fallback_orders) != {
            "development",
            "validation",
            "test",
        }:
            raise ValueError("split fallback orders must define every benchmark split")
        initial_ids = {
            query_id
            for item in self.strata
            for query_id in item.ordered_query_ids[: item.primary_target]
        }
        fallback_ids: list[str] = []
        for split, ordered_ids in self.split_fallback_orders.items():
            _require_unique(
                ordered_ids,
                f"HumanReviewPlanRecord.split_fallback_orders.{split}",
            )
            native_split_ids = {
                query_id
                for item in self.strata
                if item.split == split
                for query_id in item.ordered_query_ids
            }
            if not set(ordered_ids) <= native_split_ids:
                raise ValueError(
                    f"split fallback order contains a query outside {split}",
                )
            fallback_ids.extend(ordered_ids)
        _require_unique(fallback_ids, "HumanReviewPlanRecord split fallback query IDs")
        if set(fallback_ids) != set(query_ids) - initial_ids:
            raise ValueError(
                "split fallback orders must exactly cover non-initial questions",
            )
        for split, target in self.targets_by_split.items():
            actual = sum(
                item.primary_target for item in self.strata if item.split == split
            )
            if actual != target:
                raise ValueError(f"human-review target does not reconcile for {split}")
        double_target = sum(item.double_review_target for item in self.strata)
        if double_target != self.double_review_test_target:
            raise ValueError("double-review targets do not reconcile")
        if any(
            item.double_review_target and item.split != "test" for item in self.strata
        ):
            raise ValueError("double-review targets are restricted to the test split")
        double_accepted_target = sum(
            item.double_accepted_target for item in self.strata
        )
        if double_accepted_target != self.double_accepted_test_target:
            raise ValueError("double-accepted targets do not reconcile")
        if any(
            item.double_accepted_target and item.split != "test"
            for item in self.strata
        ):
            raise ValueError("double-accepted targets are restricted to the test split")
        _require_unique(
            self.double_review_query_ids,
            "HumanReviewPlanRecord.double_review_query_ids",
        )
        if len(self.double_review_query_ids) != self.double_review_test_target:
            raise ValueError("fixed double-review IDs do not reconcile with the target")
        if self.double_accepted_test_target > self.targets_by_split["test"]:
            raise ValueError(
                "double-accepted target exceeds the accepted human test target",
            )
        double_ids = set(self.double_review_query_ids)
        initial_test_ids = {
            query_id
            for item in self.strata
            if item.split == "test"
            for query_id in item.ordered_query_ids[: item.primary_target]
        }
        if not double_ids <= initial_test_ids:
            raise ValueError(
                "fixed double-review IDs must come from the initial primary test cohort",
            )
        for item in self.strata:
            selected_in_stratum = double_ids & set(
                item.ordered_query_ids[: item.primary_target],
            )
            if len(selected_in_stratum) != item.double_review_target:
                raise ValueError(
                    "fixed double-review IDs do not reconcile within a stratum",
                )
        expected_id = human_review_plan_id(
            base_question_set_id_value=self.base_question_set_id,
            base_manifest_hash=self.base_manifest_hash,
            selection_config_hash=self.selection_config_hash,
            source_questions_hash=self.source_questions_hash,
            source_dedup_report_hash=self.source_dedup_report_hash,
            strata=[item.to_dict() for item in self.strata],
            split_fallback_orders={
                split: list(query_ids)
                for split, query_ids in self.split_fallback_orders.items()
            },
            double_review_query_ids=self.double_review_query_ids,
            double_accepted_test_target=self.double_accepted_test_target,
        )
        if self.human_review_plan_id != expected_id:
            raise ValueError("human_review_plan_id does not match immutable selection")
        return self


class HumanReviewSamplingReportRecord(ArtifactModel):
    """Distribution audit for the frozen primary and double-review cohorts."""

    human_review_plan_id: NonEmptyStr
    base_question_set_id: NonEmptyStr
    source_questions_hash: Sha256
    selection_version: NonEmptyStr
    population_count: PositiveInt
    primary_initial_count: PositiveInt
    double_review_count: PositiveInt
    targets_by_split: dict[
        Literal["development", "validation", "test"],
        NonNegativeInt,
    ]
    cohorts: JsonDict
    warnings: list[NonEmptyStr]
    created_at_utc: NonEmptyStr
    schema_version: Literal[HUMAN_REVIEW_SAMPLING_REPORT_SCHEMA_VERSION] = (
        HUMAN_REVIEW_SAMPLING_REPORT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def sampling_counts_are_consistent(self) -> Self:
        if set(self.targets_by_split) != {"development", "validation", "test"}:
            raise ValueError("sampling targets must define all benchmark splits")
        if sum(self.targets_by_split.values()) != self.primary_initial_count:
            raise ValueError("primary initial sampling count does not reconcile")
        _require_unique(self.warnings, "HumanReviewSamplingReportRecord.warnings")
        if set(self.cohorts) != {"primary_initial", "double_review"}:
            raise ValueError("sampling report must define both review cohorts")
        return self


class HumanReviewAgreementReportRecord(ArtifactModel):
    """Reproducible agreement and completion summary for the frozen plan."""

    human_review_plan_id: NonEmptyStr
    base_question_set_id: NonEmptyStr
    source_questions_hash: Sha256
    review_events_hash: Sha256
    status: Literal["primary_incomplete", "double_incomplete", "complete"]
    primary_counts: JsonDict
    primary_validation: JsonDict
    double_review_counts: JsonDict
    double_accepted_counts: JsonDict
    agreement: JsonDict
    split_fallback_assignments: list[JsonDict]
    final_accepted_distribution: JsonDict
    double_accepted_distribution: JsonDict
    created_at_utc: NonEmptyStr
    schema_version: Literal[HUMAN_REVIEW_AGREEMENT_SCHEMA_VERSION] = (
        HUMAN_REVIEW_AGREEMENT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def counts_are_non_negative(self) -> Self:
        _validate_count_tree(self.primary_counts, "primary_counts")
        _validate_count_tree(self.double_review_counts, "double_review_counts")
        _validate_count_tree(self.double_accepted_counts, "double_accepted_counts")
        _validate_count_tree(
            self.final_accepted_distribution,
            "final_accepted_distribution",
        )
        _validate_count_tree(
            self.double_accepted_distribution,
            "double_accepted_distribution",
        )
        fallback_query_ids: list[str] = []
        for index, assignment in enumerate(self.split_fallback_assignments):
            if not isinstance(assignment, dict) or set(assignment) != {
                "query_id",
                "assigned_stratum_id",
                "native_stratum_id",
                "split",
                "decision",
            }:
                raise ValueError(
                    "split_fallback_assignments entries must contain the exact "
                    f"assignment fields (entry {index})",
                )
            if assignment["split"] not in {"development", "validation", "test"}:
                raise ValueError("split fallback assignment has an invalid split")
            if assignment["decision"] not in {
                "accept",
                "edit_required",
                "reject",
            }:
                raise ValueError("split fallback assignment has an invalid decision")
            for key in ("query_id", "assigned_stratum_id", "native_stratum_id"):
                if not isinstance(assignment[key], str) or not assignment[key]:
                    raise ValueError(
                        f"split fallback assignment {key} must be a non-empty string",
                    )
            fallback_query_ids.append(str(assignment["query_id"]))
        _require_unique(
            fallback_query_ids,
            "HumanReviewAgreementReportRecord split fallback query IDs",
        )
        if set(self.primary_validation) != {
            "development",
            "validation",
            "test",
            "total",
        }:
            raise ValueError(
                "primary_validation must define every split and the total",
            )
        for label, values in self.primary_validation.items():
            if not isinstance(values, dict):
                raise ValueError(f"primary_validation.{label} must be an object")
            reviewed = values.get("reviewed_initial")
            if not isinstance(reviewed, int) or isinstance(reviewed, bool) or reviewed < 0:
                raise ValueError(
                    f"primary_validation.{label}.reviewed_initial must be "
                    "a non-negative integer",
                )
            for rate_name in (
                "accept_confirmation_rate",
                "edit_required_rate",
                "reject_rate",
                "binary_non_accept_rate",
            ):
                rate = values.get(rate_name)
                if rate is not None and (
                    not isinstance(rate, (int, float))
                    or isinstance(rate, bool)
                    or not 0.0 <= float(rate) <= 1.0
                ):
                    raise ValueError(
                        f"primary_validation.{label}.{rate_name} must be null "
                        "or a probability",
                    )
        return self


class DevelopmentCalibrationAcceptanceRecord(ArtifactModel):
    """One active primary acceptance bound into the calibration cohort."""

    query_id: NonEmptyStr
    review_event_id: NonEmptyStr
    reviewer_id: NonEmptyStr
    review_event_hash: Sha256


class DevelopmentCalibrationCohortManifestRecord(ArtifactModel):
    """Commit-last manifest for the accepted Development judge cohort."""

    calibration_cohort_id: NonEmptyStr
    status: Literal["ready"] = "ready"
    purpose: Literal["development_judge_calibration"] = (
        "development_judge_calibration"
    )
    allowed_splits: list[Literal["development"]]
    base_question_set_id: NonEmptyStr
    base_manifest_path: NonEmptyStr
    base_manifest_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    snapshot_manifest_hash: Sha256
    source_artifact_name: Literal["questions_llm_reviewed"]
    source_questions_path: NonEmptyStr
    source_questions_hash: Sha256
    human_review_plan_id: NonEmptyStr
    human_review_plan_path: NonEmptyStr
    human_review_plan_hash: Sha256
    selection_version: NonEmptyStr
    selection_config_hash: Sha256
    human_review_events_path: NonEmptyStr
    development_review_events_hash: Sha256
    active_acceptances: list[DevelopmentCalibrationAcceptanceRecord]
    question_config_id: NonEmptyStr
    question_config_path: NonEmptyStr
    question_config_hash: Sha256
    questions_path: NonEmptyStr
    questions_hash: Sha256
    question_count: PositiveInt
    questions_byte_size: PositiveInt
    ordered_query_ids: list[NonEmptyStr]
    materialization_version: NonEmptyStr
    software_provenance: JsonDict
    created_at_utc: NonEmptyStr
    schema_version: Literal[DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION] = (
        DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def cohort_is_consistent(self) -> Self:
        if self.allowed_splits != ["development"]:
            raise ValueError("calibration cohort is restricted to Development")
        _require_unique(
            self.ordered_query_ids,
            "DevelopmentCalibrationCohortManifestRecord.ordered_query_ids",
        )
        if len(self.ordered_query_ids) != self.question_count:
            raise ValueError("calibration cohort question count does not reconcile")
        acceptance_ids = [item.query_id for item in self.active_acceptances]
        if acceptance_ids != self.ordered_query_ids:
            raise ValueError(
                "calibration acceptances must follow the exact question order",
            )
        _require_unique(
            [item.review_event_id for item in self.active_acceptances],
            "DevelopmentCalibrationCohortManifestRecord review event IDs",
        )
        expected_id = development_calibration_cohort_id(
            base_question_set_id_value=self.base_question_set_id,
            base_manifest_hash=self.base_manifest_hash,
            corpus_hash=self.corpus_hash,
            source_snapshot_hash=self.source_snapshot_hash,
            source_artifact_name=self.source_artifact_name,
            source_questions_hash=self.source_questions_hash,
            human_review_plan_id_value=self.human_review_plan_id,
            human_review_plan_hash=self.human_review_plan_hash,
            selection_version=self.selection_version,
            selection_config_hash=self.selection_config_hash,
            question_config_id=self.question_config_id,
            question_config_hash=self.question_config_hash,
            development_review_events_hash=self.development_review_events_hash,
            active_acceptances=[item.to_dict() for item in self.active_acceptances],
            ordered_query_ids=self.ordered_query_ids,
            questions_hash=self.questions_hash,
            question_count=self.question_count,
            questions_byte_size=self.questions_byte_size,
            materialization_version=self.materialization_version,
            software_provenance=self.software_provenance,
        )
        if self.calibration_cohort_id != expected_id:
            raise ValueError(
                "calibration_cohort_id does not match immutable inputs",
            )
        return self


def validate_eval_contract_record(record: JsonDict) -> str:
    """Validate one evaluation question contract or config record."""

    version = validate_schema_version(record)
    models: dict[str, type[ArtifactModel]] = {
        QUESTION_SCHEMA_VERSION: QuestionRecord,
        QUESTION_CONFIG_SCHEMA_VERSION: QuestionConfig,
        SOURCE_FAMILY_SCHEMA_VERSION: SourceFamilyRecord,
        GENERATION_UNIT_SCHEMA_VERSION: GenerationUnitRecord,
        GENERATION_BATCH_MANIFEST_SCHEMA_VERSION: GenerationBatchManifestRecord,
        GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION: GenerationAttemptResultRecord,
        QUESTION_CANDIDATE_SCHEMA_VERSION: QuestionCandidateRecord,
        QUESTION_REVIEW_EVENT_SCHEMA_VERSION: QuestionReviewEventRecord,
        DEDUP_CLUSTER_SCHEMA_VERSION: QuestionDedupClusterRecord,
        QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION: (
            QuestionDedupManualDecisionRecord
        ),
        QUESTION_DEDUP_REPORT_SCHEMA_VERSION: QuestionDedupReportRecord,
        SPLIT_MANIFEST_SCHEMA_VERSION: SplitManifestRecord,
        QUESTION_SET_MANIFEST_SCHEMA_VERSION: QuestionSetManifestRecord,
        HUMAN_REVIEW_PLAN_SCHEMA_VERSION: HumanReviewPlanRecord,
        HUMAN_REVIEW_SAMPLING_REPORT_SCHEMA_VERSION: (
            HumanReviewSamplingReportRecord
        ),
        HUMAN_REVIEW_AGREEMENT_SCHEMA_VERSION: HumanReviewAgreementReportRecord,
        DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION: (
            DevelopmentCalibrationCohortManifestRecord
        ),
    }
    model = models.get(version)
    if model is None:
        raise ValueError(f"Unsupported evaluation contract version: {version}")
    model.from_dict(record)
    return version


def validate_question_config_record(
    record: QuestionConfig | JsonDict,
) -> QuestionConfig:
    """Return the validated complete question-set configuration."""

    return record if isinstance(record, QuestionConfig) else QuestionConfig.from_dict(record)


def normalized_question_config_hash(record: QuestionConfig | JsonDict) -> str:
    """Hash the validated, behavior-affecting question configuration."""

    return config_hash(validate_question_config_record(record).to_dict())
