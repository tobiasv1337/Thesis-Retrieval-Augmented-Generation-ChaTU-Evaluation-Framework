from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.thesis_rag.fixtures.eval_question_contracts import (
    dedup_cluster_record,
    generation_unit_record,
    mock_sha256,
    question_candidate_record,
    question_set_manifest_record,
    review_event_record,
    source_family_record,
    split_manifest_record,
)
from thesis_rag.eval.schemas import (
    GenerationUnitInput,
    SourceFamilyMember,
    SourceFamilyRecord,
    generation_unit_id,
    normalized_question_config_hash,
    question_dedup_manual_decision_id,
    review_event_id,
)
from thesis_rag.schema import (
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    QUESTION_CONFIG_SCHEMA_VERSION,
    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    JsonlValidationError,
    read_jsonl,
    stable_id,
    validate_record_shape,
    write_jsonl,
)


def test_eval_contract_records_validate_through_generic_schema_registry(tmp_path):
    records = [
        source_family_record().to_dict(),
        generation_unit_record().to_dict(),
        question_candidate_record().to_dict(),
        review_event_record().to_dict(),
        dedup_cluster_record().to_dict(),
        split_manifest_record().to_dict(),
        question_set_manifest_record().to_dict(),
    ]

    for record in records:
        validate_record_shape(record)

    path = tmp_path / "eval_contracts.jsonl"
    write_jsonl(path, records)

    assert read_jsonl(path) == records


def test_eval_contract_ids_are_deterministic():
    assert source_family_record().source_family_id == source_family_record().source_family_id
    assert generation_unit_record().generation_unit_id == (
        generation_unit_record().generation_unit_id
    )
    assert question_candidate_record().candidate_id == (
        question_candidate_record().candidate_id
    )
    assert review_event_record().review_event_id == review_event_record().review_event_id
    assert split_manifest_record().split_manifest_id == (
        split_manifest_record().split_manifest_id
    )


def test_manual_dedup_decision_validates_through_generic_schema_registry():
    candidate_ids = ["question_candidate:a", "question_candidate:b"]
    record = {
        "pair_id": stable_id("question_dedup_pair", *candidate_ids),
        "candidate_ids": candidate_ids,
        "construction_config_hash": mock_sha256("dedup-config"),
        "query_relation": "distinct",
        "answer_relation": "unrelated",
        "resolution": "keep_all",
        "canonical_candidate_id": None,
        "reviewer_kind": "human",
        "reviewer_id": "thesis_author",
        "rationale": "These are different information needs.",
        "reviewed_at_utc": "2026-08-02T00:00:00Z",
        "schema_version": QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    }
    record["manual_decision_id"] = question_dedup_manual_decision_id(record)

    assert validate_record_shape(record) == (
        QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION
    )


def test_generation_unit_contract_emits_v3_only():
    record = generation_unit_record().to_dict()

    assert record["schema_version"] == "generation_unit_v3"
    assert "max_question_budget" in record
    assert "priority_key" in record
    assert "question_budget" not in record

    legacy = dict(record)
    legacy["schema_version"] = "generation_unit_v1"

    with pytest.raises(ValueError, match="Unsupported"):
        validate_record_shape(legacy)


def test_question_candidate_requires_non_empty_required_facts():
    candidate = question_candidate_record()

    broken_record = candidate.to_dict()
    broken_record["required_facts"] = []

    with pytest.raises(JsonlValidationError, match="required_facts"):
        validate_record_shape(broken_record)


def test_question_candidate_rejects_cross_lingual_generation_contract():
    record = question_candidate_record().to_dict()
    record["evidence_language"] = "en"

    with pytest.raises(JsonlValidationError, match="query_language"):
        validate_record_shape(record)


def test_generation_unit_input_allows_section_refs_without_text_hash():
    section_ref = GenerationUnitInput(
        target_level="section",
        target_id="section:without-hash",
        page_id="page:de",
        section_id="section:without-hash",
        role="outline_context",
    )
    record = generation_unit_record()
    changed = record.to_dict()
    changed["input_refs"] = [section_ref.to_dict()]
    changed["section_ids"] = ["section:without-hash"]
    changed["generation_unit_id"] = generation_unit_id(
        source_family_id_value=record.source_family_id,
        unit_kind=record.unit_kind,
        input_refs=changed["input_refs"],
        selection_config_hash=record.selection_config_hash,
    )

    validate_record_shape(changed)


def test_review_event_id_ignores_assigned_id_but_tracks_content():
    event = review_event_record()
    record = event.to_dict()
    changed_id = dict(record)
    changed_id["review_event_id"] = "question_review_event:other"

    assert review_event_id(changed_id) == event.review_event_id

    changed_content = dict(record)
    changed_content["decision"] = "reject"

    assert review_event_id(changed_content) != event.review_event_id


def test_ordinary_review_event_cannot_omit_anchor_assessments():
    record = review_event_record().to_dict()
    record.update(
        {
            "decision": "needs_human_review",
            "reason_codes": ["missing_context_reference"],
            "anchor_assessments": [],
            "sufficient_evidence_sets": [],
        },
    )
    record["review_event_id"] = review_event_id(record)

    with pytest.raises(JsonlValidationError, match="ordinary review events"):
        validate_record_shape(record)


def test_technical_review_failure_cannot_claim_semantic_judgments():
    record = review_event_record().to_dict()
    record.update(
        {
            "reviewer_kind": "local_llm",
            "reviewer_id": "question_reviewer:technical-fixture",
            "decision": "needs_human_review",
            "reason_codes": ["review_technical_failure"],
            "inferred_user_roles": ["student"],
            "inferred_query_framing": "direct_lookup",
            "natural_paraphrase_present": True,
            "query_form_fit": True,
            "source_shaping_present": False,
            "answer_scope_fit": True,
            "realistic_user_need": None,
            "plausible_user_goal": None,
            "anchor_assessments": [],
            "sufficient_evidence_sets": [],
            "provenance": {
                "kind": "local_llm_question_review",
                "model_id": "fixture-model",
                "model_revision": "fixture-revision",
                "model_path": None,
                "model_digest": None,
                "prompt_id": "fixture-prompt",
                "prompt_hash": mock_sha256("prompt"),
                "review_input_hash": mock_sha256("input"),
                "structured_output_schema_hash": mock_sha256("schema"),
                "review_policy_hash": mock_sha256("policy"),
                "review_pipeline_version": "fixture-pipeline",
                "review_seed_base": 1,
                "relevance_guideline_version": "fixture-relevance",
                "endpoint_type": "injected_review_provider",
                "response_hash": mock_sha256("response"),
                "generation_seed": 1,
                "temperature": 0.0,
                "max_tokens": 100,
                "structured_output_mode": "json_schema",
                "reasoning_effort": "none",
                "reported_decision": None,
                "output_consistency": "technical_failure",
                "retry_count": 2,
                "reviewed_at_utc": record["created_at_utc"],
                "same_model_id_as_generator": True,
                "same_model_revision_as_generator": False,
                "same_model_digest_as_generator": False,
                "independence_mode": "different_exact_model_identity",
            },
        },
    )
    record["review_event_id"] = review_event_id(record)

    with pytest.raises(JsonlValidationError, match="technical review failures"):
        validate_record_shape(record)


def test_checked_in_question_config_validates_and_hashes_deterministically():
    config_path = Path("configs/thesis/questions.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert validate_record_shape(config) == QUESTION_CONFIG_SCHEMA_VERSION
    assert normalized_question_config_hash(config) == normalized_question_config_hash(
        config,
    )
    assert (
        config["artifact_paths"]["generation_batch_manifests"]
        == "data/thesis/eval/generation_batch_manifests.jsonl"
    )
    assert "generation_backfill_events" not in config["artifact_paths"]
    assert (
        config["manual_utterance_diagnostic"][
            "exclude_from_primary_validation_and_test"
        ]
        is True
    )
    assert "minimum_required_facts" not in config["deterministic_filters"]
    assert config["schema_version"] == "question_config_v26"
    assert config["question_config_id"] == "thesis_questions_v34"
    assert config["generation"]["backfill"] == {
        "expected_accepted_yield": 0.5,
        "minimum_attempts_per_cell": 10,
    }
    assert (
        config["artifact_paths"]["question_dedup_manual_decisions"]
        == "data/thesis/eval/question_dedup_manual_decisions.jsonl"
    )
    assert config["artifact_paths"]["questions_test_human_reviewed"] == (
        "data/thesis/eval/questions_test_human_reviewed.jsonl"
    )
    assert config["artifact_paths"]["questions_test_double_accepted"] == (
        "data/thesis/eval/questions_test_double_accepted.jsonl"
    )
    assert config["artifact_paths"]["human_review_plan"] == (
        "data/thesis/eval/question_human_review_plan.json"
    )
    assert config["artifact_paths"]["human_review_sampling_report"] == (
        "data/thesis/eval/question_human_review_sampling_report.json"
    )
    assert config["artifact_paths"]["human_review_agreement"] == (
        "data/thesis/eval/question_human_review_agreement.json"
    )
    assert config["artifact_paths"]["base_question_set_archive"] == (
        "data/thesis/eval/base_question_set_archive.tar.gz"
    )
    assert config["artifact_paths"]["human_question_set_archive"] == (
        "data/thesis/eval/human_question_set_archive.tar.gz"
    )
    assert config["splits"]["target_counts"] == {
        "accepted_total": 2500,
        "final_test": 1000,
        "final_test_human_reviewed": 250,
        "human_reviewed_total": 270,
        "double_reviewed_final_test": 100,
        "double_accepted_final_test": 100,
    }
    assert config["splits"]["large_benchmark"] == {
        "development": 1000,
        "validation": 500,
        "test": 1000,
    }
    assert config["splits"]["human_core"] == {
        "development": 20,
        "validation": 0,
        "test": 250,
    }
    assert config["splits"]["smoke"] == {
        "development_human_reviewed_subset": 20,
    }
    assert "question_budgets" not in config["generation"]
    assert config["generation"]["base_seed"] == 732451
    assert config["identity"]["dedup_rule_version"] == "question_dedup_rules_v11"
    assert config["generation"]["surface_form_targets"] == {
        "complete_question": 0.30,
        "conversational_short": 0.55,
        "telegraphic_keywords": 0.15,
    }
    assert config["generation"]["evidence_scope_targets"] == {
        "localized_share": 0.85,
        "same_page_multi_section_share": 0.15,
    }
    assert config["generation"]["student_role_targets"] == {
        "inclusive_minimum_share": 0.70,
        "student_only_minimum_share_within_student": 0.30,
    }
    assert config["generation"]["query_variation_targets"] == {
        "natural_paraphrase_minimum_share": 0.20,
        "situational_request_minimum_share": 0.15,
    }
    assert config["generation"]["source_selection"][
        "low_priority_url_substrings"
    ] == [
        "/aktuelles/",
        "/berichte/",
        "/interviews/",
        "/magazin",
        "/news",
    ]
    assert (
        config["generation"]["generation_units"]["context_group"]
        ["window_overlap_chars"]
        == 200
    )
    assert "max_context_sections" not in (
        config["generation"]["generation_units"]["table_context"]
    )
    assert (
        config["generation"]["local_model"]["structured_output_mode"]
        == "json_schema"
    )
    assert "reason_codes" not in config["review"]
    assert (
        config["review"]["local_llm"]["model_id"]
        == "gemma4:31b-review-32k"
    )
    assert config["review"]["local_llm"]["model_revision"] is None
    assert config["review"]["local_llm"]["model_digest"] == (
        "sha256:36f5ca871cd0cecd37bc056f276caf9c"
        "984c5fb6827135dbe164a9103c25935c"
    )
    assert (
        config["review"]["local_llm"]["endpoint"]
        == "http://localhost:11434/v1"
    )
    assert config["review"]["local_llm"]["reasoning_effort"] == "medium"
    assert config["review"]["local_llm"]["max_tokens"] == 8000
    review_prompt_path = Path(config["review"]["prompt"]["path"])
    assert review_prompt_path.exists()
    assert review_prompt_path.name == "question_review.md"
    assert config["review"]["prompt"]["prompt_id"] == "question_review_v26"
    assert "human_core" not in config["review"]
    assert config["deduplication"]["semantic"] == {
        "enabled": True,
        "model_id": "bge-m3:latest",
        "model_revision": None,
        "model_digest": (
            "sha256:7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
        ),
        "endpoint": "http://localhost:11434/v1",
        "cache_dir": "outputs/thesis/cache/question_dedup_embeddings",
        "batch_size": 32,
        "review_at": 0.68,
        "answer_review_at": 0.68,
        "question_only_review_at": 0.78,
        "answer_rescue_question_review_at": 0.50,
        "answer_rescue_answer_review_at": 0.84,
        "cross_language_question_review_at": 0.65,
        "cross_language_answer_review_at": 0.75,
    }
    adjudication = config["deduplication"]["adjudication"]
    assert adjudication["batch_size"] == 5
    assert adjudication["maximum_adjudication_pairs"] == 1600
    assert adjudication["local_llm"]["reasoning_effort"] == "none"
    assert adjudication["local_llm"]["timeout_seconds"] == 600
    assert adjudication["local_llm"]["model_digest"] == (
        config["review"]["local_llm"]["model_digest"]
    )


def test_jsonl_expected_schema_version_filters_eval_records(tmp_path):
    path = tmp_path / "candidates.jsonl"
    candidate = question_candidate_record().to_dict()
    write_jsonl(path, [candidate])

    records = read_jsonl(
        path,
        expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
    )

    assert records == [candidate]


def test_eval_models_are_strict_frozen_and_forbid_unknown_fields():
    member = source_family_record().members[0]
    invalid_type = member.to_dict()
    invalid_type["page_id"] = 7

    with pytest.raises(ValidationError, match="page_id"):
        SourceFamilyMember.from_dict(invalid_type)

    extra_field = source_family_record().to_dict()
    extra_field["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        SourceFamilyRecord.from_dict(extra_field)

    with pytest.raises(ValidationError, match="frozen"):
        member.page_id = "page:changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("splits", "target_counts", "accepted_total"),
            3999,
            "accepted_total",
        ),
        (
            ("splits", "smoke", "development_human_reviewed_subset"),
            51,
            "smoke subset",
        ),
        (
            ("splits", "target_counts", "double_accepted_final_test"),
            251,
            "double-accepted final-test count",
        ),
        (
            ("generation", "surface_form_targets", "complete_question"),
            0.50,
            "surface_form_targets",
        ),
        (
            ("generation", "student_role_targets", "inclusive_minimum_share"),
            1.10,
            "less than or equal",
        ),
        (
            (
                "generation",
                "query_variation_targets",
                "natural_paraphrase_minimum_share",
            ),
            1.10,
            "less than or equal",
        ),
        (
            ("generation", "evidence_scope_targets", "localized_share"),
            0.80,
            "sum to 1.0",
        ),
        (
            (
                "generation",
                "source_selection",
                "max_candidates_per_page_record",
            ),
            9,
            "per_source_family",
        ),
        (
            (
                "deterministic_filters",
                "lexical_overlap",
                "medium_max",
            ),
            0.20,
            "lexical-overlap thresholds",
        ),
    ],
)
def test_question_config_rejects_semantically_inconsistent_values(
    path,
    value,
    message,
):
    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(JsonlValidationError, match=message):
        validate_record_shape(config)


def test_question_config_temporal_classes_form_a_complete_partition():
    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config["generation"]["temporal_policy"]["rejected_temporal_classes"].append(
        "not_temporal",
    )

    with pytest.raises(JsonlValidationError, match="overlap"):
        validate_record_shape(config)


def test_question_config_v26_rejects_legacy_or_unknown_configuration_fields():
    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    legacy = dict(config)
    legacy["schema_version"] = "question_config_v18"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_record_shape(legacy)

    config["generation"]["question_budgets"] = {"unused": 1}
    with pytest.raises(JsonlValidationError, match="question_budgets"):
        validate_record_shape(config)

    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config["generation"]["controlled_challenge_share"] = 0.10
    with pytest.raises(JsonlValidationError, match="controlled_challenge_share"):
        validate_record_shape(config)

    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config["review"]["human_core"] = {"development": 50}
    with pytest.raises(JsonlValidationError, match="human_core"):
        validate_record_shape(config)


def test_question_config_human_review_counts_reconcile_with_targets():
    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config["splits"]["human_core"]["development"] = 1

    with pytest.raises(JsonlValidationError, match="human_reviewed_total"):
        validate_record_shape(config)


def test_source_family_edges_must_reference_members_and_match_summaries():
    record = source_family_record().to_dict()
    record["relationship_edges"][0]["member_page_ids"][1] = "page:unknown"

    with pytest.raises(JsonlValidationError, match="family members"):
        validate_record_shape(record)

    record = source_family_record().to_dict()
    record["relationship_kinds"] = []
    with pytest.raises(JsonlValidationError, match="relationship_kinds"):
        validate_record_shape(record)


def test_generation_unit_references_must_match_its_page_sections_and_tables():
    record = generation_unit_record().to_dict()
    record["input_refs"][0]["page_id"] = "page:other"

    with pytest.raises(JsonlValidationError, match="page_id"):
        validate_record_shape(record)

    record = generation_unit_record().to_dict()
    record["section_ids"] = []
    with pytest.raises(JsonlValidationError, match="section_ids"):
        validate_record_shape(record)

    record = generation_unit_record().to_dict()
    record["input_refs"][0]["table_id"] = "table:undeclared"
    with pytest.raises(JsonlValidationError, match="table IDs"):
        validate_record_shape(record)


def test_split_manifest_counts_must_reconcile_with_assignments():
    record = split_manifest_record().to_dict()
    record["counts_by_split"]["development"]["families"] = 2

    with pytest.raises(JsonlValidationError, match="counts_by_split"):
        validate_record_shape(record)

    record = split_manifest_record().to_dict()
    record["counts_by_language_coverage"]["de_en_counterpart"]["total"] = 2
    with pytest.raises(JsonlValidationError, match="counts_by_language_coverage"):
        validate_record_shape(record)


def test_candidate_evidence_may_only_reference_existing_required_facts():
    record = question_candidate_record().to_dict()
    record["evidence_quotations"][0]["required_fact_refs"] = ["fact:999"]

    with pytest.raises(JsonlValidationError, match="unknown required facts"):
        validate_record_shape(record)
