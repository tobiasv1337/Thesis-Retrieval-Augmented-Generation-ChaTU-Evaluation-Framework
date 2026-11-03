from __future__ import annotations

import hashlib

from thesis_rag.eval.schemas import (
    DEDUP_RULE_VERSION,
    GENERATION_UNIT_RULE_VERSION,
    QUESTION_REVIEW_GUIDELINE_VERSION,
    SOURCE_FAMILY_RULE_VERSION,
    SPLIT_RULE_VERSION,
    AnchorAssessment,
    EvidenceQuotation,
    GenerationUnitInput,
    GenerationUnitRecord,
    QuestionCandidateRecord,
    QuestionDedupClusterRecord,
    QuestionReviewEventRecord,
    QuestionSetManifestRecord,
    SourceFamilyMember,
    SourceFamilyRecord,
    SplitFamilyAssignment,
    SplitManifestRecord,
    generation_unit_id,
    generation_unit_priority_key,
    question_candidate_id,
    question_dedup_cluster_id,
    question_set_id,
    review_event_id,
    source_family_id,
    split_manifest_id,
)
from thesis_rag.interfaces import EvidenceTarget
from thesis_rag.schema import QUESTION_REVIEW_EVENT_SCHEMA_VERSION


def mock_sha256(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


CORPUS_HASH = mock_sha256("corpus")
SNAPSHOT_HASH = mock_sha256("snapshot")
RELATIONSHIPS_HASH = mock_sha256("relationships")
CONFIG_HASH = mock_sha256("config")
SOURCE_FAMILIES_HASH = mock_sha256("source-families")
GENERATION_UNITS_HASH = mock_sha256("generation-units")


def source_family_record() -> SourceFamilyRecord:
    members = [
        SourceFamilyMember(
            page_id="page:de",
            page_key="page_key:de",
            language="de",
            eligible_question_generation=True,
            eligible_standard_index=True,
            canonical_page_id="page:de",
            url="https://example.test/de/topic",
            normalized_url="https://example.test/de/topic",
            normalized_base_url="https://example.test/de/topic",
            language_counterpart_page_ids=["page:en"],
            content_flags=["prose"],
        ),
        SourceFamilyMember(
            page_id="page:en",
            page_key="page_key:en",
            language="en",
            eligible_question_generation=True,
            eligible_standard_index=True,
            canonical_page_id="page:en",
            url="https://example.test/en/topic",
            normalized_url="https://example.test/en/topic",
            normalized_base_url="https://example.test/en/topic",
            language_counterpart_page_ids=["page:de"],
            content_flags=["prose"],
        ),
    ]
    member_ids = [member.page_id for member in members]
    return SourceFamilyRecord(
        source_family_id=source_family_id(
            corpus_hash=CORPUS_HASH,
            member_page_ids=member_ids,
            construction_config_hash=CONFIG_HASH,
        ),
        corpus_hash=CORPUS_HASH,
        members=members,
        relationship_ids=["source_relationship:de-en"],
        relationship_kinds=["language_counterpart"],
        relationship_edges=[
            {
                "source_relationship_id": "source_relationship:de-en",
                "relationship_kind": "language_counterpart",
                "member_page_ids": member_ids,
            },
        ],
        language_coverage="de_en_counterpart",
        generation_eligible_page_ids=member_ids,
        family_metadata={"intent": "topic"},
        source_family_rule_version=SOURCE_FAMILY_RULE_VERSION,
        construction_config_hash=CONFIG_HASH,
    )


def generation_unit_record() -> GenerationUnitRecord:
    family = source_family_record()
    input_ref = GenerationUnitInput(
        target_level="section",
        target_id="section:de:intro",
        page_id="page:de",
        section_id="section:de:intro",
        role="group_focus",
        text_hash=mock_sha256("generation-input"),
        source_path="data/az/scraper/topics/topic/content.html",
    )
    input_refs = [input_ref.to_dict()]
    return GenerationUnitRecord(
        generation_unit_id=generation_unit_id(
            source_family_id_value=family.source_family_id,
            unit_kind="context_group",
            input_refs=input_refs,
            selection_config_hash=CONFIG_HASH,
        ),
        source_family_id=family.source_family_id,
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        unit_kind="context_group",
        query_language="de",
        evidence_language="de",
        page_id="page:de",
        input_refs=[input_ref],
        context_group_ids=["context_group:fixture"],
        section_ids=["section:de:intro"],
        table_ids=[],
        generation_unit_text_hash=mock_sha256("generation-unit-text"),
        prompt_context_hash=mock_sha256("prompt-context"),
        selection_config_hash=CONFIG_HASH,
        max_question_budget=1,
        priority_key=generation_unit_priority_key(
            random_seed=311037,
            split="development",
            query_language="de",
            source_family_id_value=family.source_family_id,
            page_id="page:de",
            unit_kind="context_group",
            input_refs=input_refs,
        ),
        split="development",
        selection_reason_codes=["eligible_section"],
    )


def evidence_target() -> EvidenceTarget:
    return EvidenceTarget(
        target_level="span",
        target_id="span:deadline",
        page_id="page:de",
        section_id="section:de:intro",
        char_start=12,
        char_end=38,
        text_hash=mock_sha256("evidence-target"),
        source_path="data/az/scraper/topics/topic/content.html",
        metadata={"quote": "Die Frist endet am 1. April."},
    )


def question_candidate_record() -> QuestionCandidateRecord:
    family = source_family_record()
    unit = generation_unit_record()
    required_facts = ["Die Frist endet am 1. April."]
    generation_attempt_id = "generation_attempt:fixture"
    candidate_id = question_candidate_id(
        source_method="local_llm_generated",
        source_family_id_value=family.source_family_id,
        generation_unit_id_value=unit.generation_unit_id,
        question_text="Wann endet die Frist?",
        query_language="de",
        user_roles=["student"],
        reference_answer="Die Frist endet am 1. April.",
        required_facts=required_facts,
        generation_attempt_id=generation_attempt_id,
    )
    return QuestionCandidateRecord(
        candidate_id=candidate_id,
        source_method="local_llm_generated",
        source_family_id=family.source_family_id,
        generation_unit_id=unit.generation_unit_id,
        generation_attempt_id=generation_attempt_id,
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        split="development",
        question_text="Wann endet die Frist?",
        query_language="de",
        evidence_language="de",
        user_roles=["student"],
        primary_type="procedural",
        traits=["deadline", "temporal"],
        surface_form="complete_question",
        reference_answer="Die Frist endet am 1. April.",
        required_facts=required_facts,
        evidence_quotations=[
            EvidenceQuotation(
                quotation="Die Frist endet am 1. April.",
                required_fact_refs=["fact:1"],
                source_ref={"section_id": "section:de:intro"},
            ),
        ],
        temporal_classification="snapshot_qualified",
        deterministic_filter_status="passed",
        deterministic_filter_reasons=[],
        generator_provenance={
            "kind": "local_llm",
            "model_id": "fixture-model",
            "input_hash": mock_sha256("generator-input"),
        },
        verified_evidence_targets=[evidence_target()],
        query_evidence_lexical_overlap={"score": 0.33, "bucket": "medium"},
    )


def review_event_record() -> QuestionReviewEventRecord:
    candidate = question_candidate_record()
    assessment = AnchorAssessment(
        target=evidence_target(),
        grade=3,
        status="accepted",
        label="sufficient",
        explanation="The span directly answers the deadline question.",
        supported_fact_refs=["fact:1"],
        propagation_rule="primary_anchor",
    )
    base_event = {
        "schema_version": QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "reviewer_kind": "human",
        "reviewer_id": "reviewer:fixture",
        "guideline_version": QUESTION_REVIEW_GUIDELINE_VERSION,
        "decision": "accept",
        "reason_codes": [],
        "inferred_user_roles": ["student"],
        "inferred_query_framing": "direct_lookup",
        "natural_paraphrase_present": False,
        "query_form_fit": True,
        "source_shaping_present": False,
        "answer_scope_fit": True,
        "realistic_user_need": True,
        "plausible_user_goal": "A student needs the deadline to submit on time.",
        "independent_review": False,
        "review_summary": "The question is fully supported by the cited span.",
        "anchor_assessments": [assessment.to_dict()],
        "sufficient_evidence_sets": [
            {"target_ids": [str(assessment.target.target_id)]},
        ],
        "provenance": {"kind": "manual", "interface": "fixture"},
        "created_at_utc": "2026-01-01T00:00:00Z",
        "query_id": None,
        "new_candidate_id": None,
    }
    event_id = review_event_id(base_event)
    event_data = dict(base_event)
    event_data["anchor_assessments"] = [assessment]
    return QuestionReviewEventRecord(
        review_event_id=event_id,
        **event_data,
    )


def dedup_cluster_record() -> QuestionDedupClusterRecord:
    candidate = question_candidate_record()
    paraphrase_id = f"{candidate.candidate_id}:paraphrase"
    candidate_ids = [candidate.candidate_id, paraphrase_id]
    cluster_id = question_dedup_cluster_id(
        cluster_kind="same_language_paraphrase",
        candidate_ids=candidate_ids,
        retained_candidate_ids=candidate_ids,
        decision="manual_review",
        canonical_candidate_id=None,
        construction_config_hash=CONFIG_HASH,
        corpus_hash=CORPUS_HASH,
    )
    return QuestionDedupClusterRecord(
        dedup_cluster_id=cluster_id,
        cluster_kind="same_language_paraphrase",
        candidate_ids=candidate_ids,
        retained_candidate_ids=candidate_ids,
        canonical_candidate_id=None,
        source_family_ids=[candidate.source_family_id],
        splits=["development"],
        retained_splits=["development"],
        query_relations=["uncertain"],
        answer_relations=["uncertain"],
        decision="manual_review",
        similarity={
            "token_jaccard": 0.82,
            "semantic_score": 0.88,
        },
        evidence={"method": "fixture"},
        dedup_rule_version=DEDUP_RULE_VERSION,
        construction_config_hash=CONFIG_HASH,
        corpus_hash=CORPUS_HASH,
    )


def question_set_manifest_record() -> QuestionSetManifestRecord:
    artifact_paths = {
        "source_families": "data/thesis/eval/source_families.jsonl",
        "generation_units": "data/thesis/eval/generation_units.jsonl",
    }
    artifact_hashes = {
        "source_families": SOURCE_FAMILIES_HASH,
        "generation_units": GENERATION_UNITS_HASH,
    }
    config_hashes = {
        "questions": CONFIG_HASH,
        "source_families": CONFIG_HASH,
    }
    set_id = question_set_id(
        seal_kind="base",
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        base_question_set_id_value=None,
        human_review_plan_id_value=None,
        artifact_hashes=artifact_hashes,
        config_hashes=config_hashes,
    )
    return QuestionSetManifestRecord(
        question_set_id=set_id,
        seal_kind="base",
        base_question_set_id=set_id,
        status="sealed",
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        snapshot_manifest_hash=mock_sha256("snapshot-manifest"),
        corpus_archive_hash=mock_sha256("corpus-archive"),
        source_relationships_hash=RELATIONSHIPS_HASH,
        source_families_hash=SOURCE_FAMILIES_HASH,
        generation_units_hash=GENERATION_UNITS_HASH,
        split_manifest_id="split_manifest:fixture",
        split_manifest_hash=mock_sha256("split-manifest"),
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_counts={
            "source_families": 1,
            "generation_units": 1,
        },
        artifact_byte_sizes={
            "source_families": 100,
            "generation_units": 200,
        },
        archive_members=artifact_paths,
        archive_path="data/thesis/eval/base_question_set_archive.tar.gz",
        archive_hash=mock_sha256("question-set-archive"),
        archive_byte_size=300,
        archive_compression="gzip",
        config_hashes=config_hashes,
        split_counts={
            "development": {"accepted": 1},
            "validation": {"accepted": 0},
            "test": {"accepted": 0},
        },
        tier_counts={
            "llm_reviewed": 1,
            "human_reviewed": 0,
            "double_reviewed": 0,
        },
        review_counts={},
        subset_relationships={},
        random_seeds={"split": 311037},
        rule_versions={"sealing": "fixture"},
        model_provenance={"generation": {"model_id": "fixture"}},
        prompt_provenance={"generation": {"hash": CONFIG_HASH}},
        software_provenance={"git_revision": "fixture"},
        review_guideline_version=QUESTION_REVIEW_GUIDELINE_VERSION,
        limitations=["fixture"],
        created_at_utc="2026-01-01T00:00:00Z",
        updated_at_utc="2026-01-01T00:00:00Z",
    )


def split_manifest_record() -> SplitManifestRecord:
    family = source_family_record()
    assignment = SplitFamilyAssignment(
        source_family_id=family.source_family_id,
        split="development",
        language_coverage=family.language_coverage,
        member_count=len(family.members),
        generation_eligible_page_count=len(family.generation_eligible_page_ids),
        generation_eligible_page_ids=family.generation_eligible_page_ids,
        estimated_question_capacity=6,
        assignment_reason="language_stratified_capacity_balanced_assignment",
    )
    return SplitManifestRecord(
        split_manifest_id=split_manifest_id(
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
            source_families_hash=SOURCE_FAMILIES_HASH,
            split_config_hash=CONFIG_HASH,
        ),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        snapshot_manifest_hash=mock_sha256("snapshot-manifest"),
        source_family_rule_version=SOURCE_FAMILY_RULE_VERSION,
        split_rule_version=SPLIT_RULE_VERSION,
        random_seed=311037,
        source_artifact_paths={
            "snapshot_manifest": "data/thesis/corpus/snapshot_manifest.json",
            "corpus_archive": "data/thesis/corpus/raw_source_archive.tar.zst",
            "pages": "data/thesis/corpus/pages.jsonl",
            "page_quality": "data/thesis/corpus/page_quality.jsonl",
            "source_relationships": "data/thesis/corpus/source_relationships.jsonl",
        },
        source_artifact_hashes={
            "snapshot_manifest": mock_sha256("snapshot-manifest"),
            "corpus_archive": mock_sha256("corpus-archive"),
            "pages": mock_sha256("pages"),
            "page_quality": mock_sha256("page-quality"),
            "source_relationships": RELATIONSHIPS_HASH,
        },
        output_artifact_paths={
            "source_families": "data/thesis/eval/source_families.jsonl",
            "split_manifest": "data/thesis/eval/split_manifest.json",
        },
        output_artifact_hashes={
            "source_families": SOURCE_FAMILIES_HASH,
        },
        config_hashes={
            "questions": CONFIG_HASH,
            "source_families": CONFIG_HASH,
            "splits": CONFIG_HASH,
        },
        target_ratios={
            "development": 0.5,
            "validation": 0.25,
            "test": 0.25,
        },
        assignments=[assignment],
        counts_by_split={
            "development": {
                "families": 1,
                "generation_eligible_pages": 2,
                "members": 2,
                "estimated_question_capacity": 6,
            },
            "validation": {
                "families": 0,
                "generation_eligible_pages": 0,
                "members": 0,
                "estimated_question_capacity": 0,
            },
            "test": {
                "families": 0,
                "generation_eligible_pages": 0,
                "members": 0,
                "estimated_question_capacity": 0,
            },
            "unassigned": {
                "families": 0,
                "generation_eligible_pages": 0,
                "members": 0,
                "estimated_question_capacity": 0,
            },
        },
        counts_by_language_coverage={
            "de_en_counterpart": {
                "development": 1,
                "total": 1,
            },
            "de_only": {
                "total": 0,
            },
            "de_en_unpaired": {
                "total": 0,
            },
            "en_only": {
                "total": 0,
            },
        },
    )
