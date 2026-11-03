from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.thesis_rag.fixtures.eval_question_contracts import (
    generation_unit_record,
    question_candidate_record,
)
from tests.thesis_rag.test_question_generation import (
    _valid_provider,
    _write_job_fixture,
)
from thesis_rag.cli import app
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.question_filtering import (
    QuestionFilteringError,
    QuestionFilteringJob,
    QuestionFilteringProgress,
    VerifiedGenerationState,
    _filter_candidate,
    filter_generation_state,
)
from thesis_rag.eval.generation_selection import GenerationDeficitCell
from thesis_rag.eval.question_generation import QuestionGenerationJob
from thesis_rag.eval.question_review import (
    QUERY_FORM_REVIEW_STANDARDS,
    QuestionReviewError,
    QuestionReviewJob,
    QuestionReviewResponse,
    _active_review_input,
    _review_model_output_schema,
    _review_response_payload,
    _validate_review_response,
)
from thesis_rag.eval.question_workflow import (
    FilterReviewJob,
    FilterReviewProgress,
    _accepted_deficit_plan,
    _apply_backfill_headroom,
)
from thesis_rag.eval.schemas import (
    QuestionCandidateRecord,
    question_candidate_id,
    review_event_id,
)
from thesis_rag.schema import canonical_json, hash_text, read_jsonl, write_jsonl


def _config() -> QuestionConfig:
    return QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )


def test_backfill_headroom_uses_configured_yield_and_per_cell_minimum() -> None:
    cells = [
        GenerationDeficitCell(
            deficit_cell_id="larger",
            count=7,
            split="development",
        ),
        GenerationDeficitCell(
            deficit_cell_id="tail",
            count=1,
            split="test",
        ),
    ]

    planned = _apply_backfill_headroom(cells, config=_config())

    assert [cell.count for cell in planned] == [14, 10]


def _raw_candidate(
    question: str,
    *,
    evidence: str = "Die Frist endet am ersten April.",
    traits: list[str] | None = None,
    surface_form: str = "complete_question",
    temporal_classification: str = "snapshot_qualified",
    quotation: str | None = None,
    user_roles: list[str] | None = None,
) -> QuestionCandidateRecord:
    record = question_candidate_record().to_dict()
    record.update(
        {
            "question_text": question,
            "traits": sorted(traits or []),
            "surface_form": surface_form,
            "temporal_classification": temporal_classification,
            "deterministic_filter_status": "raw",
            "deterministic_filter_reasons": [],
            "query_evidence_lexical_overlap": {},
            "metadata": {},
            "user_roles": user_roles or ["student"],
            "verified_evidence_targets": [
                {
                    "target_level": "span",
                    "target_id": "span:filter-test",
                    "page_id": "page:de",
                    "section_id": "section:de:intro",
                    "char_start": 0,
                    "char_end": len(evidence),
                    "text_hash": hash_text(evidence),
                    "source_path": "data/source.html",
                    "metadata": {"quotation_index": 1},
                },
            ],
        },
    )
    if quotation is not None:
        record["evidence_quotations"][0]["quotation"] = quotation
    record["candidate_id"] = question_candidate_id(
        source_method=record["source_method"],
        source_family_id_value=record["source_family_id"],
        generation_unit_id_value=record["generation_unit_id"],
        question_text=record["question_text"],
        query_language=record["query_language"],
        user_roles=record["user_roles"],
        reference_answer=record["reference_answer"],
        required_facts=record["required_facts"],
        generation_attempt_id=record["generation_attempt_id"],
        parent_candidate_id=record["parent_candidate_id"],
    )
    return QuestionCandidateRecord.from_dict(record)


def _filter(candidate: QuestionCandidateRecord, evidence: str):
    return _filter_candidate(
        candidate,
        page={"text": evidence},
        config=_config(),
        policy_hash=hash_text("filter-policy"),
    )


@pytest.mark.parametrize("question", ["BAföG?", "Moses Frist", "WS 26/27", "Wo?"])
def test_meaningful_short_and_telegraphic_questions_are_not_rejected(question: str):
    filtered = _filter(
        _raw_candidate(question, surface_form="telegraphic_keywords"),
        "Die Frist endet am ersten April.",
    )

    assert filtered.deterministic_filter_status != "rejected"
    assert "question_too_short" not in filtered.deterministic_filter_reasons


def test_question_candidate_requires_meaningful_text_but_no_minimum_length():
    assert _raw_candidate("X").question_text == "X"
    with pytest.raises(ValidationError, match="Unicode letter or digit"):
        _raw_candidate("?!")


def test_filter_hard_rejections_are_limited_to_frozen_objective_rules():
    config = _config()
    too_long = _raw_candidate("x" * (config.deterministic_filters.maximum_question_chars + 1))
    long_quote = "x" * (
        config.deterministic_filters.maximum_evidence_quote_chars + 1
    )

    assert _filter(too_long, "evidence").deterministic_filter_reasons == [
        "question_too_long",
    ]
    assert _filter(
        _raw_candidate("Frist?", quotation=long_quote),
        "evidence",
    ).deterministic_filter_reasons == ["evidence_quote_too_long"]
    assert _filter(
        _raw_candidate(
            "Frist?",
            temporal_classification="expired_event_or_news",
        ),
        "evidence",
    ).deterministic_filter_reasons == ["temporal_policy_reject"]


@pytest.mark.parametrize(
    "surface_form",
    [
        "complete_question",
        "conversational_short",
        "telegraphic_keywords",
    ],
)
def test_high_overlap_is_persisted_but_does_not_filter(surface_form: str):
    evidence = "Die Frist endet am ersten April."
    filtered = _filter(
        _raw_candidate(
            "Endet die Frist am ersten April?",
            evidence=evidence,
            surface_form=surface_form,
        ),
        evidence,
    )

    assert filtered.query_evidence_lexical_overlap["score"] == 1.0
    assert filtered.query_evidence_lexical_overlap["bucket"] == "high"
    assert filtered.query_evidence_lexical_overlap["overlap_terms"]
    assert filtered.deterministic_filter_reasons == []
    assert filtered.deterministic_filter_status == "passed"


def test_verbose_telegraphic_query_is_soft_flagged_for_review():
    verbose = _filter(
        _raw_candidate(
            "exam preparation revision duration simulation speaking writing breathing nerves",
            surface_form="telegraphic_keywords",
        ),
        "exam preparation advice",
    )
    concise = _filter(
        _raw_candidate(
            "exam preparation revision duration simulation speaking writing breathing",
            surface_form="telegraphic_keywords",
        ),
        "exam preparation advice",
    )

    assert verbose.deterministic_filter_status == "flagged"
    assert verbose.deterministic_filter_reasons == [
        "telegraphic_keywords_too_verbose",
    ]
    assert concise.deterministic_filter_status == "passed"


def test_copied_sentence_is_exact_and_not_a_fuzzy_leakage_heuristic():
    evidence = "Die Frist endet am ersten April."
    copied = _filter(
        _raw_candidate("Die Frist endet am ersten April?", evidence=evidence),
        evidence,
    )
    changed = _filter(
        _raw_candidate("Wann endet die Frist am ersten April?", evidence=evidence),
        evidence,
    )

    assert "copied_source_sentence" in copied.deterministic_filter_reasons
    assert "copied_source_sentence" not in changed.deterministic_filter_reasons
    source_question = "Wann endet die Frist in diesem Semester?"
    not_declarative = _filter(
        _raw_candidate(source_question, evidence=source_question),
        source_question,
    )
    assert "copied_source_sentence" not in not_declarative.deterministic_filter_reasons


def test_filtering_does_not_remove_duplicate_candidates():
    evidence = "Die Frist endet am ersten April."
    candidate = _raw_candidate("Frist im April?", evidence=evidence)
    state = VerifiedGenerationState(
        config=_config(),
        repo_root=Path.cwd(),
        generation_units=(generation_unit_record(),),
        manifests=(),
        attempts=(),
        raw_candidates=(candidate, candidate),
        sources={"pages": [{"page_id": "page:de", "text": evidence}]},
        generation_units_hash=hash_text("units"),
    )

    updates: list[QuestionFilteringProgress] = []
    filtered = filter_generation_state(state, progress_callback=updates.append)

    assert len(filtered) == 2
    assert [(update.processed, update.total) for update in updates] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert updates[-1].passed == 2
    assert updates[-1].flagged == 0
    assert updates[-1].rejected == 0


def _multi_section_candidate() -> QuestionCandidateRecord:
    record = _raw_candidate("Welche Frist gilt und wo liegt das Formular?").to_dict()
    record["reference_answer"] = (
        "Die Frist endet am ersten April, und das Formular liegt im Portal."
    )
    record["required_facts"] = [
        "Die Frist endet am ersten April.",
        "Das Formular liegt im Portal.",
    ]
    record["evidence_quotations"] = [
        {
            "quotation": "Die Frist endet am ersten April.",
            "required_fact_refs": ["fact:1"],
            "source_ref": {"input_index": 1},
        },
        {
            "quotation": "Das Formular liegt im Portal.",
            "required_fact_refs": ["fact:2"],
            "source_ref": {"input_index": 2},
        },
    ]
    first = record["verified_evidence_targets"][0]
    second = {
        **first,
        "target_id": "span:filter-test-2",
        "char_start": 40,
        "char_end": 60,
        "text_hash": hash_text("Das Formular liegt dort."),
        "metadata": {"quotation_index": 2},
    }
    record["traits"] = ["same_page_multi_section"]
    record["verified_evidence_targets"] = [first, second]
    record["evidence_groups"] = [
        {
            "evidence_group_id": "evidence_group:test",
            "target_ids": ["span:filter-test", "span:filter-test-2"],
            "sufficiency_rule": "all_of",
            "required_fact_refs": ["fact:1", "fact:2"],
        },
    ]
    record["candidate_id"] = question_candidate_id(
        source_method=record["source_method"],
        source_family_id_value=record["source_family_id"],
        generation_unit_id_value=record["generation_unit_id"],
        question_text=record["question_text"],
        query_language=record["query_language"],
        user_roles=record["user_roles"],
        reference_answer=record["reference_answer"],
        required_facts=record["required_facts"],
        generation_attempt_id=record["generation_attempt_id"],
        parent_candidate_id=record["parent_candidate_id"],
    )
    return QuestionCandidateRecord.from_dict(record)


def _response(
    candidate: QuestionCandidateRecord,
    *,
    decision: str = "accept",
    grades: list[int] | None = None,
    reason_codes: list[str] | None = None,
    sufficient_sets: list[list[str]] | None = None,
) -> QuestionReviewResponse:
    selected_grades = grades or [3] * len(candidate.verified_evidence_targets)
    all_fact_refs = [
        f"fact:{index}" for index in range(1, len(candidate.required_facts) + 1)
    ]
    if sufficient_sets is None:
        if decision != "accept":
            sufficient_sets = []
        elif candidate.evidence_groups:
            sufficient_sets = [candidate.evidence_groups[0].target_ids]
        else:
            grade_three = [
                str(target.target_id)
                for target, grade in zip(
                    candidate.verified_evidence_targets,
                    selected_grades,
                    strict=True,
                )
                if grade == 3
            ]
            sufficient_sets = [[grade_three[0]]] if grade_three else []
    return QuestionReviewResponse.from_dict(
        {
            "decision": decision,
            "reason_codes": reason_codes or [],
            "inferred_user_roles": candidate.user_roles,
            "inferred_query_framing": "direct_lookup",
            "natural_paraphrase_present": False,
            "query_form_fit": True,
            "source_shaping_present": False,
            "answer_scope_fit": True,
            "realistic_user_need": True,
            "plausible_user_goal": (
                "A university user needs this information for a concrete task."
            ),
            "anchor_assessments": [
                {
                    "target_id": target.target_id,
                    "grade": grade,
                    "supported_fact_refs": (
                        all_fact_refs
                        if grade == 3
                        else candidate.evidence_quotations[
                            int(target.metadata["quotation_index"]) - 1
                        ].required_fact_refs
                        if grade == 2
                        else []
                    ),
                    "explanation": "The supplied span has this degree of support.",
                }
                for target, grade in zip(
                    candidate.verified_evidence_targets,
                    selected_grades,
                    strict=True,
                )
            ],
            "sufficient_evidence_sets": [
                {"target_ids": target_ids} for target_ids in sufficient_sets
            ],
            "review_summary": "The candidate was checked against the supplied evidence.",
        },
    )


def test_review_response_payload_accepts_narrow_local_model_syntax_variants():
    candidate = _raw_candidate("Wann endet die Frist?")
    payload = _response(candidate).to_dict()
    payload["sufficient_evidence_sets"] = [
        item["target_ids"] for item in payload["sufficient_evidence_sets"]
    ]
    raw = "```json\n" + json.dumps(payload) + "\n```"

    decoded = _review_response_payload(raw)
    parsed = QuestionReviewResponse.from_dict(decoded)

    assert parsed.decision == "accept"
    assert parsed.sufficient_evidence_sets[0].target_ids == [
        str(candidate.verified_evidence_targets[0].target_id),
    ]


def test_review_response_payload_rejects_arbitrary_surrounding_text():
    with pytest.raises(ValueError, match="response prefix"):
        _review_response_payload('Here is the result: {"decision":"accept"}')


def test_blind_review_input_excludes_generator_reasoning_and_prior_reviews():
    candidate = _raw_candidate("Wann endet die Frist?")
    candidate = candidate.model_copy(
        update={
            "metadata": {
                "deterministic_filter": {
                    "diagnostics": ["high_lexical_overlap"],
                },
            },
        },
    )
    payload = _active_review_input(
        candidate,
        "Verified source context",
        source_page_text="Die Frist endet am ersten April.",
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )
    serialized = canonical_json(payload)

    assert "generator_provenance" not in serialized
    assert "reasoning_summary" not in serialized
    assert "review_event" not in serialized
    assert "deterministic_filter" not in serialized
    assert "high_lexical_overlap" not in serialized
    assert "fixture-model" not in serialized
    assert "primary_type" not in payload
    assert "traits" not in payload
    assert "user_roles" not in payload
    assert payload["query_form"] == {
        "name": candidate.surface_form,
        "review_standard": QUERY_FORM_REVIEW_STANDARDS[candidate.surface_form],
    }
    assert payload["snapshot_created_at_utc"] == "2026-07-12T09:43:46+00:00"
    assert "Verified source context" in serialized


@pytest.mark.parametrize(
    "surface_form",
    ["complete_question", "conversational_short", "telegraphic_keywords"],
)
def test_review_input_supplies_only_the_selected_query_form_standard(
    surface_form: str,
):
    candidate = _raw_candidate("Moses Frist", surface_form=surface_form)

    payload = _active_review_input(
        candidate,
        "Source context",
        source_page_text="Die Frist endet am ersten April.",
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )
    query_form = payload["query_form"]

    assert query_form["name"] == surface_form
    assert query_form["review_standard"] == QUERY_FORM_REVIEW_STANDARDS[surface_form]
    assert all(
        guidance not in canonical_json(query_form)
        for name, guidance in QUERY_FORM_REVIEW_STANDARDS.items()
        if name != surface_form
    )
    if surface_form == "telegraphic_keywords":
        assert "Full-sentence grammar" in query_form["review_standard"]
        assert "not required" in query_form["review_standard"]
    if surface_form == "conversational_short":
        assert "naturally type into a chatbot" in query_form["review_standard"]
        assert "compact noun phrase" in query_form["review_standard"]
        assert "First person" in query_form["review_standard"]
        assert "Mild natural underspecification is valid" in (
            query_form["review_standard"]
        )
        assert "global uniqueness" in query_form["review_standard"]
        assert "source wording" not in query_form["review_standard"]


def test_review_input_includes_groups_only_for_joint_multi_section_scope():
    localized = _active_review_input(
        _raw_candidate("Wann endet die Frist?"),
        "Source context",
        source_page_text="Die Frist endet am ersten April.",
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )
    multi = _active_review_input(
        _multi_section_candidate(),
        "Source context",
        source_page_text="x" * 100,
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )

    assert localized["evidence_scope"] == {"mode": "localized"}
    assert multi["evidence_scope"] == {
        "mode": "same_page_multi_section",
        "groups": [
            {
                "target_ids": ["evidence:1", "evidence:2"],
                "sufficiency_rule": "all_of",
                "required_fact_refs": ["fact:1", "fact:2"],
            },
        ],
    }
    assert localized["required_facts"] == [
        {"fact_id": "fact:1", "text": localized_fact}
        for localized_fact in _raw_candidate(
            "Wann endet die Frist?",
        ).required_facts
    ]
    assert "claimed_fact_refs" not in localized["evidence_targets"][0]
    assert localized["evidence_targets"][0]["target_id"] == "evidence:1"


def test_review_input_includes_small_adjacent_source_margins():
    quotation = 'b"-Symbol, um die Seite zu löschen.'
    page_text = 'Klicken Sie auf das "Papierkorb"-Symbol, um die Seite zu löschen.'
    candidate = _raw_candidate(
        "seite löschen symbol",
        evidence=quotation,
        quotation=quotation,
    )
    start = page_text.index(quotation)
    target = candidate.verified_evidence_targets[0].model_copy(
        update={"char_start": start, "char_end": start + len(quotation)},
    )
    candidate = candidate.model_copy(update={"verified_evidence_targets": [target]})

    payload = _active_review_input(
        candidate,
        quotation,
        source_page_text=page_text,
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )

    evidence = payload["evidence_targets"][0]
    assert evidence["verified_quotation"] == quotation
    assert evidence["adjacent_source_before"].endswith("Papierkor")
    assert evidence["adjacent_source_after"] == ""


def test_review_prompt_defines_semantic_review_without_internal_jargon():
    prompt = Path("configs/thesis/prompts/question_review.md").read_text(
        encoding="utf-8",
    )
    normalized_prompt = " ".join(prompt.split())

    assert len(prompt.split()) < 1750
    assert "Review Input JSON" in prompt
    assert "Five review dimensions" in prompt
    assert "`realistic_user_need`" in prompt
    assert "`inferred_user_roles`" in prompt
    assert "`inferred_query_framing`" in prompt
    assert "`natural_paraphrase_present`" in prompt
    assert "`query_form_fit`" in prompt
    assert "`source_shaping_present`" in prompt
    assert "`answer_scope_fit`" in prompt
    for role in ("student", "staff", "researcher", "visitor", "public"):
        assert f"`{role}`" in prompt
    assert "sorted, unique list" in normalized_prompt
    assert "same information need naturally applies" in normalized_prompt
    assert "`public` is not a catch-all" in normalized_prompt
    assert "physically visiting a campus facility" in normalized_prompt
    assert "Internal account or service needs exclude" in normalized_prompt
    assert "This gate is separate from query-form fit" in prompt
    assert "source-derived count" in prompt
    assert "source_shaped_query" in prompt
    assert "source-shaped" in prompt
    assert "hard scoring boundary" in normalized_prompt
    assert "practical user goal" in normalized_prompt
    assert "actionable contact channel" in normalized_prompt
    assert "a pointer saying that details are on the page is incomplete" in (
        normalized_prompt
    )
    assert "do not expose the linked content" in normalized_prompt
    assert "the verified quotation itself identifies a usable destination" in (
        normalized_prompt
    )
    assert "Do not reject an answer merely for being terse" in prompt
    assert "never for unsupported content or incomplete coverage" in normalized_prompt
    assert "`reference_answer_overinclusive`" in prompt
    assert "supported_fact_refs" in prompt
    assert "minimal sufficient target combination" in normalized_prompt
    assert "never supply missing support" in normalized_prompt
    assert "can enter the benchmark unchanged" in normalized_prompt
    assert "substantial rewriting, different evidence" in normalized_prompt
    assert "`snapshot_created_at_utc`" in prompt
    assert "acceptance or rejection rate" not in prompt.lower()
    assert "rejection quota" not in prompt.lower()
    assert "`plausible_user_goal`" in prompt
    assert "Natural mild underspecification is valid" in prompt
    assert "do not demand global uniqueness" in prompt
    for framing in (
        "direct_lookup",
        "situational_request",
    ):
        assert f"`{framing}`" in prompt
    assert "A direct lookup can therefore also contain a natural paraphrase" in (
        normalized_prompt
    )
    assert "functional description" in prompt
    assert "silently supply a subject missing from the query" in normalized_prompt
    assert "Career Service" not in prompt
    assert "workshop" not in prompt.lower()
    assert "Which three" not in prompt
    assert "plausible minor typo" in normalized_prompt
    assert "difference only in letter casing is never" in normalized_prompt


def test_review_response_schema_enumerates_supported_reason_codes():
    reason_schema = _review_model_output_schema()["properties"][
        "reason_codes"
    ]["items"]

    assert set(reason_schema["enum"]) == {
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
    }
    assert "evidence_quote_unaligned" not in reason_schema["enum"]
    assert "role_mismatch" not in reason_schema["enum"]
    assert "review_output_inconsistent" not in reason_schema["enum"]
    assert "review_technical_failure" not in reason_schema["enum"]


def test_review_consistency_requires_explicit_sufficient_evidence_sets():
    localized = _raw_candidate("Wann endet die Frist?")
    multi = _multi_section_candidate()
    localized_joint_record = multi.to_dict()
    localized_joint_record["traits"] = []
    localized_joint_record["evidence_groups"] = []
    localized_joint = QuestionCandidateRecord.from_dict(localized_joint_record)

    with pytest.raises(QuestionReviewError, match="sufficient evidence"):
        _validate_review_response(
            _response(localized, grades=[3], sufficient_sets=[]),
            localized,
        )
    _validate_review_response(
        _response(
            localized_joint,
            grades=[2, 2],
            sufficient_sets=[["span:filter-test", "span:filter-test-2"]],
        ),
        localized_joint,
    )
    overlapping = _response(
        localized_joint,
        grades=[2, 2],
        sufficient_sets=[["span:filter-test", "span:filter-test-2"]],
    ).to_dict()
    overlapping["anchor_assessments"][1]["supported_fact_refs"] = ["fact:1"]
    with pytest.raises(QuestionReviewError, match="every required fact"):
        _validate_review_response(
            QuestionReviewResponse.from_dict(overlapping),
            localized_joint,
        )
    _validate_review_response(_response(multi, grades=[2, 2]), multi)
    with pytest.raises(QuestionReviewError, match="grade-2 anchors"):
        _validate_review_response(_response(multi, grades=[2, 1]), multi)
    wrong_target = _response(localized).to_dict()
    wrong_target["anchor_assessments"][0]["target_id"] = "span:other"
    with pytest.raises(QuestionReviewError, match="exactly"):
        _validate_review_response(
            QuestionReviewResponse.from_dict(wrong_target),
            localized,
        )


def test_nonaccept_review_is_advisory_and_supports_missing_context_reason():
    candidate = _raw_candidate("Was zeigt das Bild oben?")
    response = _response(
        candidate,
        decision="edit_required",
        grades=[0],
        reason_codes=["missing_context_reference"],
    )

    _validate_review_response(response, candidate)
    assert "field_edits" not in QuestionReviewResponse.model_json_schema()["properties"]
    assert response.decision == "edit_required"


@pytest.mark.parametrize(
    "field",
    ["query_form_fit", "answer_scope_fit", "realistic_user_need"],
)
def test_accept_requires_explicit_fit_and_realistic_user_need(field: str):
    candidate = _raw_candidate("Wann endet die Frist?")
    payload = _response(candidate).to_dict()
    payload[field] = False
    if field == "realistic_user_need":
        payload["plausible_user_goal"] = None

    with pytest.raises(QuestionReviewError, match="accepted review requires"):
        _validate_review_response(QuestionReviewResponse.from_dict(payload), candidate)


def test_overinclusive_answer_reason_requires_failed_answer_scope():
    candidate = _raw_candidate("Wann endet die Frist?")
    payload = _response(
        candidate,
        decision="edit_required",
        reason_codes=["reference_answer_overinclusive"],
    ).to_dict()

    with pytest.raises(QuestionReviewError, match="answer_scope_fit=false"):
        _validate_review_response(QuestionReviewResponse.from_dict(payload), candidate)

    payload["answer_scope_fit"] = False
    _validate_review_response(QuestionReviewResponse.from_dict(payload), candidate)


def test_accept_allows_independently_inferred_roles_to_differ_from_generator():
    candidate = _raw_candidate("Wann endet die Frist?")
    payload = _response(candidate).to_dict()
    payload["inferred_user_roles"] = ["public"]

    response = QuestionReviewResponse.from_dict(payload)

    _validate_review_response(response, candidate)
    assert response.inferred_user_roles == ["public"]


def test_review_accepts_exact_sorted_multi_role_classification():
    candidate = _raw_candidate(
        "Wo befindet sich der Seminarraum?",
        user_roles=["staff", "student", "visitor"],
    )
    response = _response(candidate)

    _validate_review_response(response, candidate)
    assert response.inferred_user_roles == ["staff", "student", "visitor"]


def test_review_canonicalizes_unsorted_role_output():
    candidate = _raw_candidate(
        "Wo befindet sich der Seminarraum?",
        user_roles=["staff", "student", "visitor"],
    )
    payload = _response(candidate).to_dict()
    payload["inferred_user_roles"] = ["visitor", "student", "staff"]

    response = QuestionReviewResponse.from_dict(payload)

    _validate_review_response(response, candidate)
    assert response.inferred_user_roles == ["staff", "student", "visitor"]


def test_review_rejects_duplicate_role_output():
    candidate = _raw_candidate("Wann endet die Frist?")
    payload = _response(candidate).to_dict()
    payload["inferred_user_roles"] = ["student", "student"]

    with pytest.raises(ValidationError, match="sorted and unique"):
        QuestionReviewResponse.from_dict(payload)


def test_accept_rejects_source_shaped_wording_independently_of_query_form():
    candidate = _raw_candidate("Welche drei Voraussetzungen gelten?")
    payload = _response(candidate).to_dict()
    payload["source_shaping_present"] = True

    with pytest.raises(QuestionReviewError, match="source-shaped"):
        _validate_review_response(QuestionReviewResponse.from_dict(payload), candidate)


def test_review_requires_goal_exactly_for_realistic_user_need():
    candidate = _raw_candidate("Wann endet die Frist?")
    missing_goal = _response(candidate).to_dict()
    missing_goal["plausible_user_goal"] = None

    with pytest.raises(ValidationError, match="plausible_user_goal"):
        QuestionReviewResponse.from_dict(missing_goal)

    implausible_with_goal = _response(candidate).to_dict()
    implausible_with_goal["realistic_user_need"] = False

    with pytest.raises(ValidationError, match="plausible_user_goal"):
        QuestionReviewResponse.from_dict(implausible_with_goal)


def _set_small_targets(config: dict, development: int) -> None:
    config["generation"]["surface_form_targets"] = {
        "complete_question": 1.0,
        "conversational_short": 0.0,
        "telegraphic_keywords": 0.0,
    }
    config["generation"]["evidence_scope_targets"] = {
        "localized_share": 1.0,
        "same_page_multi_section_share": 0.0,
    }
    config["generation"]["student_role_targets"] = {
        "inclusive_minimum_share": 1.0,
        "student_only_minimum_share_within_student": 1.0,
    }
    config["generation"]["query_variation_targets"] = {
        "natural_paraphrase_minimum_share": 0.0,
        "situational_request_minimum_share": 0.0,
    }
    config["splits"]["large_benchmark"] = {
        "development": development,
        "validation": 0,
        "test": 0,
    }
    config["splits"]["target_counts"].update(
        {
            "accepted_total": development,
            "final_test": 0,
            "final_test_human_reviewed": 0,
            "human_reviewed_total": 0,
            "double_reviewed_final_test": 0,
            "double_accepted_final_test": 0,
        },
    )
    human_core = {"development": 0, "validation": 0, "test": 0}
    config["splits"]["human_core"] = human_core
    config["splits"]["smoke"]["development_human_reviewed_subset"] = 0


def _prepare_pipeline(
    tmp_path: Path,
    *,
    initial_candidates: int = 1,
    target: int | None = None,
    suffix: str = "",
):
    config_path, config = _write_job_fixture(tmp_path, suffix=suffix)
    config["review"]["local_llm"].update(
        {
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
            "model_digest": None,
            "endpoint": None,
        },
    )
    config["review"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_review.md").resolve(),
    )
    if target is not None:
        _set_small_targets(config, target)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    if initial_candidates:
        QuestionGenerationJob(
            config_path,
            repo_root=tmp_path,
            provider=_valid_provider,
        ).run(
            deficits=[
                {
                    "deficit_cell_id": "initial:development",
                    "count": initial_candidates,
                    "split": "development",
                },
            ],
            limit=initial_candidates,
            verify=False,
        )
    else:
        for artifact_name in (
            "generation_batch_manifests",
            "generation_attempt_results",
            "question_candidates_raw",
        ):
            artifact_path = Path(config["artifact_paths"][artifact_name])
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.touch()
    QuestionFilteringJob(config_path, repo_root=tmp_path).run(verify=False)
    return config_path, config


def _accepting_review_provider(request):
    active = json.loads(request.user_prompt.split("\n", 1)[1])
    return {
        "decision": "accept",
        "reason_codes": [],
        "inferred_user_roles": ["student"],
        "inferred_query_framing": "direct_lookup",
        "natural_paraphrase_present": False,
        "query_form_fit": True,
        "source_shaping_present": False,
        "answer_scope_fit": True,
        "realistic_user_need": True,
        "plausible_user_goal": (
            "A student needs the deadline to complete the process on time."
        ),
        "anchor_assessments": [
            {
                "target_id": target["target_id"],
                "grade": 3,
                "supported_fact_refs": [
                    fact["fact_id"] for fact in active["required_facts"]
                ],
                "explanation": "The target directly and completely supports the answer.",
            }
            for target in active["evidence_targets"]
        ],
        "sufficient_evidence_sets": [
            {"target_ids": [active["evidence_targets"][0]["target_id"]]},
        ],
        "review_summary": "The question is realistic and fully supported.",
    }


def _configure_cli_review_endpoint(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["review"]["local_llm"]["endpoint"] = "http://review.invalid/v1"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_review_counts_independently_inferred_roles_instead_of_generator_roles(
    tmp_path: Path,
):
    config_path, _ = _prepare_pipeline(
        tmp_path,
        suffix="-independent-reviewed-roles",
    )

    def provider(request):
        response = _accepting_review_provider(request)
        response["inferred_user_roles"] = ["public"]
        return response

    result = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    assert result.accepted_by_split_and_role_profile["development"] == {
        "student_inclusive": 0,
        "student_only": 0,
    }
    assert result.events[0].inferred_user_roles == ["public"]


def test_filter_cli_materializes_temp_artifacts(tmp_path: Path):
    config_path, config = _prepare_pipeline(tmp_path, suffix="-filter-cli")
    filtered_path = Path(config["artifact_paths"]["question_candidates_filtered"])
    filtered_path.unlink()

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "filter",
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--no-verify",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Filtered candidates materialized" in result.output
    assert "question_candidates_filtered" in result.output
    assert len(read_jsonl(filtered_path)) == 1


def test_review_cli_honors_limit_with_injected_local_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=2,
        suffix="-review-cli",
    )
    _configure_cli_review_endpoint(config_path)
    requests = []

    def provider(request):
        requests.append(request)
        return _accepting_review_provider(request)

    monkeypatch.setattr(
        "thesis_rag.eval.question_review.LocalQuestionReviewProvider",
        lambda state: provider,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "review",
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--limit",
            "1",
            "--no-verify",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Question review complete" in result.output
    assert "1 appended" in result.output
    assert len(requests) == 1
    events = read_jsonl(Path(config["artifact_paths"]["question_review_events"]))
    assert len(events) == 1
    assert events[0]["decision"] == "accept"
    assert events[0]["inferred_user_roles"] == ["student"]
    assert events[0]["inferred_query_framing"] == "direct_lookup"
    assert events[0]["natural_paraphrase_present"] is False
    assert events[0]["query_form_fit"] is True
    assert events[0]["source_shaping_present"] is False
    assert events[0]["answer_scope_fit"] is True
    assert events[0]["realistic_user_need"] is True
    assert events[0]["plausible_user_goal"] == (
        "A student needs the deadline to complete the process on time."
    )
    persisted_target_id = events[0]["anchor_assessments"][0]["target"]["target_id"]
    assert persisted_target_id.startswith("span:")
    assert events[0]["sufficient_evidence_sets"] == [
        {"target_ids": [persisted_target_id]},
    ]


def test_filter_review_cli_backfills_with_injected_local_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=1,
        target=2,
        suffix="-filter-review-cli",
    )
    _configure_cli_review_endpoint(config_path)
    generation_requests = []
    review_requests = []

    def generation_provider(request):
        generation_requests.append(request)
        return _valid_provider(request)

    def review_provider(request):
        review_requests.append(request)
        return _accepting_review_provider(request)

    monkeypatch.setattr(
        "thesis_rag.eval.question_generation.LocalQuestionGenerationProvider",
        lambda config: generation_provider,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.question_review.LocalQuestionReviewProvider",
        lambda state: review_provider,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "filter-review",
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--generation-batch-limit",
            "1",
            "--no-verify",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Workflow status: target_reached" in result.output
    assert "round 0" in result.output
    assert len(generation_requests) >= 1
    assert len(review_requests) == 1 + len(generation_requests)
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert manifests[0]["batch_purpose"] == "initial"
    assert all(
        manifest["batch_purpose"] == "post_filter_review"
        for manifest in manifests[1:]
    )


def test_review_job_is_blind_append_only_resumable_and_discloses_same_model(tmp_path: Path):
    config_path, config = _prepare_pipeline(tmp_path)
    requests = []

    def provider(request):
        requests.append(request)
        return _accepting_review_provider(request)

    job = QuestionReviewJob(config_path, repo_root=tmp_path, provider=provider)
    first = job.run(verify=False)
    second = job.run(verify=False)

    assert first.appended_events == 1
    assert second.appended_events == 0
    assert second.reused_events == 1
    assert len(requests) == 1
    assert requests[0].user_prompt.startswith("Review Input JSON:\n")
    assert "generator_provenance" not in requests[0].user_prompt
    assert requests[0].reasoning_effort == "medium"
    event = first.events[0]
    assert event.decision == "accept"
    assert event.provenance["same_model_id_as_generator"] is True
    assert event.provenance["same_model_revision_as_generator"] is True
    assert event.provenance["independence_mode"] == "same_exact_model_strict_prompt"
    assert event.provenance["reasoning_effort"] == "medium"
    records = read_jsonl(Path(config["artifact_paths"]["question_review_events"]))
    assert len(records) == 1


def test_review_resume_fails_closed_on_relinked_stale_anchor(tmp_path: Path):
    config_path, config = _prepare_pipeline(tmp_path, suffix="-stale-review")
    job = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=_accepting_review_provider,
    )
    job.run(verify=False)
    path = Path(config["artifact_paths"]["question_review_events"])
    records = read_jsonl(path)
    records[0]["anchor_assessments"][0]["target"]["char_end"] += 1
    records[0]["review_event_id"] = review_event_id(records[0])
    write_jsonl(path, records)

    with pytest.raises(QuestionReviewError, match="differs from verified"):
        job.run(verify=False)


def test_review_resume_fails_closed_on_relinked_false_provenance(tmp_path: Path):
    config_path, config = _prepare_pipeline(tmp_path, suffix="-false-provenance")
    job = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=_accepting_review_provider,
    )
    job.run(verify=False)
    path = Path(config["artifact_paths"]["question_review_events"])
    records = read_jsonl(path)
    records[0]["provenance"]["model_id"] = "different-model"
    records[0]["review_event_id"] = review_event_id(records[0])
    write_jsonl(path, records)

    with pytest.raises(QuestionReviewError, match="reviewer identity"):
        job.run(verify=False)


def test_review_technical_failure_is_reported_after_retries(tmp_path: Path):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=2,
        suffix="-retry",
    )
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _accepting_review_provider(request)
        raise RuntimeError("review endpoint unavailable")

    result = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    assert calls == 4  # one success, then the configured three attempts
    assert result.appended_events == 1
    assert len(result.technical_failures) == 1
    assert "after retries" in next(iter(result.technical_failures.values()))
    records = read_jsonl(Path(config["artifact_paths"]["question_review_events"]))
    assert len(records) == 1
    assert records[0]["decision"] == "accept"


def test_persistent_invalid_model_output_requires_human_review_and_can_backfill(
    tmp_path: Path,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=1,
        target=1,
        suffix="-invalid-output-human-review",
    )
    review_calls = 0
    review_prompts = []

    def provider(request):
        nonlocal review_calls
        review_calls += 1
        review_prompts.append(request.system_prompt)
        if review_calls <= 3:
            return "not valid json"
        return _accepting_review_provider(request)

    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=_valid_provider,
        review_provider=provider,
    ).run(verify=False)

    assert review_calls >= 4
    assert "Required output correction" not in review_prompts[0]
    assert all(
        "Required output correction" in prompt for prompt in review_prompts[1:3]
    )
    assert result.status == "target_reached"
    assert result.technical_review_failures == {}
    assert result.needs_human_review_by_cause == {"technical_model_output": 1}
    events = read_jsonl(Path(config["artifact_paths"]["question_review_events"]))
    technical_event = events[0]
    assert technical_event["decision"] == "needs_human_review"
    assert technical_event["reason_codes"] == ["review_technical_failure"]
    assert technical_event["inferred_user_roles"] is None
    assert technical_event["inferred_query_framing"] is None
    assert technical_event["natural_paraphrase_present"] is None
    assert technical_event["query_form_fit"] is None
    assert technical_event["source_shaping_present"] is None
    assert technical_event["answer_scope_fit"] is None
    assert technical_event["realistic_user_need"] is None
    assert technical_event["plausible_user_goal"] is None
    assert technical_event["anchor_assessments"] == []
    assert technical_event["provenance"]["reported_decision"] is None
    assert technical_event["provenance"]["output_consistency"] == "technical_failure"
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert manifests[0]["batch_purpose"] == "initial"
    assert all(
        manifest["batch_purpose"] == "post_filter_review"
        for manifest in manifests[1:]
    )


def test_review_continues_after_one_candidate_fails_technically(tmp_path: Path):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=3,
        suffix="-continue-after-failure",
    )
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        if calls in {2, 3, 4}:
            raise RuntimeError("review endpoint unavailable")
        return _accepting_review_provider(request)

    result = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    assert calls == 5
    assert result.appended_events == 2
    assert len(result.technical_failures) == 1
    records = read_jsonl(Path(config["artifact_paths"]["question_review_events"]))
    assert len(records) == 2


def test_filter_review_stops_cleanly_before_backfill_when_review_is_incomplete(
    tmp_path: Path,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=2,
        target=2,
        suffix="-review-incomplete",
    )
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("review endpoint unavailable")
        return _accepting_review_provider(request)

    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=_valid_provider,
        review_provider=provider,
    ).run(verify=False)

    assert calls == 4
    assert result.status == "review_incomplete"
    assert len(result.technical_review_failures) == 1
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert [item["batch_purpose"] for item in manifests] == ["initial"]


def test_review_retries_structurally_valid_but_inconsistent_anchor_output(tmp_path: Path):
    config_path, _ = _prepare_pipeline(tmp_path, suffix="-anchor-retry")
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        response = _accepting_review_provider(request)
        if calls == 1:
            response["anchor_assessments"][0]["target_id"] = "span:unknown"
        return response

    result = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    assert calls == 2
    assert result.appended_events == 1
    assert result.events[0].provenance["retry_count"] == 1


@pytest.mark.parametrize(
    "coverage_failure",
    ["unknown_target", "unknown_sufficient_target", "empty_assessments"],
)
def test_review_normalizes_persistent_target_coverage_failure(
    tmp_path: Path,
    coverage_failure: str,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        suffix=f"-anchor-human-review-{coverage_failure}",
    )
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        response = _accepting_review_provider(request)
        if coverage_failure == "unknown_target":
            response["anchor_assessments"][0]["target_id"] = "span:unknown"
        elif coverage_failure == "unknown_sufficient_target":
            response["sufficient_evidence_sets"][0]["target_ids"] = [
                "span:unknown",
            ]
        else:
            response["anchor_assessments"] = []
        return response

    result = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    event = result.events[0]
    max_retries = config["review"]["local_llm"]["max_retries"]
    assert calls == max_retries + 1
    assert event.decision == "needs_human_review"
    assert event.reason_codes == ["review_output_inconsistent"]
    assert event.anchor_assessments == []
    assert event.sufficient_evidence_sets == []
    assert event.provenance["retry_count"] == max_retries
    assert event.provenance["reported_decision"] == "accept"
    assert event.provenance["output_consistency"] == "normalized_to_human_review"
    assert result.needs_human_review_by_cause == {
        "technical_output_inconsistent": 1,
    }

    resumed = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=lambda request: pytest.fail("stored review should be reused"),
    ).run(verify=False)

    assert resumed.appended_events == 0
    assert resumed.reused_events == 1


def test_review_normalizes_evidence_consistency_failure_to_human_review(tmp_path: Path):
    config_path, config = _prepare_pipeline(tmp_path, suffix="-evidence-consistency")
    calls = 0
    requests = []

    def provider(request):
        nonlocal calls
        calls += 1
        requests.append(request)
        response = _accepting_review_provider(request)
        response["anchor_assessments"][0]["grade"] = 2
        response["sufficient_evidence_sets"] = []
        return response

    result = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    event = result.events[0]
    max_retries = config["review"]["local_llm"]["max_retries"]
    assert calls == max_retries + 1
    assert "Required correction" not in requests[0].system_prompt
    assert "grade-2 anchor must support a non-empty strict subset" in (
        requests[1].system_prompt
    )
    assert "Do not raise evidence grades" in requests[1].system_prompt
    assert "prefer `reject` over inflating grades" in requests[1].system_prompt
    assert "only the very small repair defined above" in requests[1].system_prompt
    assert event.decision == "needs_human_review"
    assert event.reason_codes == ["review_output_inconsistent"]
    assert event.sufficient_evidence_sets == []
    assert event.provenance["reported_decision"] == "accept"
    assert event.provenance["output_consistency"] == "normalized_to_human_review"
    assert event.provenance["retry_count"] == max_retries
    assert event.anchor_assessments == []

    resumed = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=lambda request: pytest.fail("stored review should be reused"),
    ).run(verify=False)

    assert resumed.appended_events == 0
    assert resumed.reused_events == 1


def test_filter_fails_closed_without_overwriting_previous_view(tmp_path: Path):
    config_path, config = _prepare_pipeline(tmp_path, suffix="-corrupt")
    filtered_path = Path(config["artifact_paths"]["question_candidates_filtered"])
    previous = filtered_path.read_bytes()
    attempt_path = Path(config["artifact_paths"]["generation_attempt_results"])
    attempts = read_jsonl(attempt_path)
    attempts[0]["generation_seed"] += 1
    write_jsonl(attempt_path, attempts)

    with pytest.raises(QuestionFilteringError, match="reservation"):
        QuestionFilteringJob(config_path, repo_root=tmp_path).run(verify=False)

    assert filtered_path.read_bytes() == previous


@pytest.mark.parametrize(
    "artifact_name",
    [
        "generation_batch_manifests",
        "generation_attempt_results",
        "question_candidates_raw",
    ],
)
def test_filter_fails_closed_on_missing_generation_history(
    tmp_path: Path,
    artifact_name: str,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        suffix=f"-missing-{artifact_name}",
    )
    filtered_path = Path(config["artifact_paths"]["question_candidates_filtered"])
    previous = filtered_path.read_bytes()
    Path(config["artifact_paths"][artifact_name]).unlink()

    with pytest.raises(QuestionFilteringError, match="history is missing"):
        QuestionFilteringJob(config_path, repo_root=tmp_path).run(verify=False)

    assert filtered_path.read_bytes() == previous


@pytest.mark.parametrize("corruption", ["source", "evidence", "schema"])
def test_filter_fails_closed_on_source_evidence_and_schema_corruption(
    tmp_path: Path,
    corruption: str,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        suffix=f"-corrupt-{corruption}",
    )
    filtered_path = Path(config["artifact_paths"]["question_candidates_filtered"])
    previous = filtered_path.read_bytes()
    if corruption == "source":
        path = Path(config["source_artifacts"]["pages_path"])
        pages = read_jsonl(path)
        pages[0]["text"] = "X" + pages[0]["text"]
        pages[0]["text_hash"] = hash_text(pages[0]["text"])
        pages[0]["content_source_hash"] = hash_text(pages[0]["text"])
        write_jsonl(path, pages)
    else:
        path = Path(config["artifact_paths"]["question_candidates_raw"])
        candidates = read_jsonl(path)
        if corruption == "evidence":
            candidates[0]["verified_evidence_targets"][0]["text_hash"] = hash_text(
                "different evidence",
            )
            write_jsonl(path, candidates)
        else:
            candidates[0]["schema_version"] = "question_candidate_v2"
            path.write_text(canonical_json(candidates[0]) + "\n", encoding="utf-8")

    with pytest.raises(QuestionFilteringError):
        QuestionFilteringJob(config_path, repo_root=tmp_path).run(verify=False)

    assert filtered_path.read_bytes() == previous


def test_filter_review_backfills_with_manifest_history_until_target(tmp_path: Path):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=1,
        target=2,
        suffix="-workflow",
    )
    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=_valid_provider,
        review_provider=_accepting_review_provider,
    ).run(generation_batch_limit=1, verify=False)

    assert result.status == "target_reached"
    assert result.accepted_by_split["development"] >= 2
    assert result.accepted_by_split["validation"] == 0
    assert result.accepted_by_split["test"] == 0
    assert not any(
        result.surface_form_deficits_by_split["development"].values(),
    )
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert manifests[0]["batch_purpose"] == "initial"
    assert all(
        manifest["batch_purpose"] == "post_filter_review"
        for manifest in manifests[1:]
    )
    assert manifests[1]["previous_batch_manifest_id"] == (
        manifests[0]["generation_batch_manifest_id"]
    )
    assert set(manifests[1]["trigger_artifact_hashes"]) == {
        "question_candidates_filtered",
        "question_review_events",
    }
    assert not Path(tmp_path / "generation_backfill_events.jsonl").exists()


def test_filter_review_backfills_reviewer_inferred_natural_paraphrase(
    tmp_path: Path,
):
    config_path, config = _write_job_fixture(
        tmp_path,
        suffix="-framing-workflow",
    )
    config["review"]["local_llm"].update(
        {
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
            "model_digest": None,
            "endpoint": None,
        },
    )
    config["review"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_review.md").resolve(),
    )
    _set_small_targets(config, 1)
    config["generation"]["surface_form_targets"] = {
        "complete_question": 1.0,
        "conversational_short": 0.0,
        "telegraphic_keywords": 0.0,
    }
    config["generation"]["query_variation_targets"] = {
        "natural_paraphrase_minimum_share": 1.0,
        "situational_request_minimum_share": 0.0,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    review_calls = 0

    def reviewer(request):
        nonlocal review_calls
        review_calls += 1
        response = _accepting_review_provider(request)
        response["natural_paraphrase_present"] = review_calls > 1
        return response

    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=_valid_provider,
        review_provider=reviewer,
    ).run(generation_batch_limit=1, verify=False)

    assert result.status == "target_reached"
    assert result.accepted_by_split["development"] == 2
    assert result.accepted_by_split_and_query_framing["development"] == {
        "direct_lookup": 2,
        "situational_request": 0,
    }
    assert result.accepted_natural_paraphrases_by_split["development"] == 1
    assert result.query_variation_deficits_by_split["development"] == {
        "natural_paraphrase": 0,
        "situational_request": 0,
    }
    manifests = read_jsonl(
        Path(config["artifact_paths"]["generation_batch_manifests"]),
    )
    assert len(manifests) == 2
    assert (
        manifests[1]["selected_units"][0]["attempts"][0][
            "prefer_natural_paraphrase"
        ]
        is True
    )


def test_accepted_scope_deficit_backfills_even_when_split_total_is_reached():
    config = _config()

    plan, scope_deficits, role_deficits, _, _ = _accepted_deficit_plan(
        config,
        targets={"development": 40, "validation": 20, "test": 20},
        accepted_by_split_and_scope={
            "development": {
                "localized": 34,
                "same_page_multi_section": 6,
            },
            "validation": {
                "localized": 18,
                "same_page_multi_section": 2,
            },
            "test": {
                "localized": 17,
                "same_page_multi_section": 3,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 28,
                "student_only": 20,
            },
            "validation": {
                "student_inclusive": 14,
                "student_only": 10,
            },
            "test": {
                "student_inclusive": 14,
                "student_only": 10,
            },
        },
    )

    assert [cell.to_dict() for cell in plan.cells] == [
        {
            "deficit_cell_id": (
                "post_filter_review:validation:same_page_multi_section:"
                "variation:situational_request"
            ),
            "count": 10,
            "split": "validation",
            "query_language": None,
            "unit_kind": None,
            "source_family_id": None,
            "preferred_role_profile": None,
            "preferred_query_variation": "situational_request",
            "requested_surface_form": None,
            "requested_evidence_scope": "same_page_multi_section",
        },
    ]
    assert scope_deficits == {
        "development": {"localized": 0, "same_page_multi_section": 0},
        "validation": {"localized": 0, "same_page_multi_section": 1},
        "test": {"localized": 0, "same_page_multi_section": 0},
    }
    assert all(
        not any(split_deficits.values())
        for split_deficits in role_deficits.values()
    )


def test_accepted_backfill_cells_do_not_double_count_total_and_scope_deficits():
    plan, _, _, _, _ = _accepted_deficit_plan(
        _config(),
        targets={"development": 40},
        accepted_by_split_and_scope={
            "development": {
                "localized": 30,
                "same_page_multi_section": 2,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 28,
                "student_only": 20,
            },
        },
    )

    assert {
        cell.requested_evidence_scope: cell.count for cell in plan.cells
    } == {
        "same_page_multi_section": 10,
        "localized": 10,
    }
    assert sum(cell.count for cell in plan.cells) == 20


def test_student_role_deficit_is_targeted_before_scope_backfill():
    plan, scope_deficits, role_deficits, _, _ = _accepted_deficit_plan(
        _config(),
        targets={"development": 10},
        accepted_by_split_and_scope={
            "development": {
                "localized": 7,
                "same_page_multi_section": 1,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 5,
                "student_only": 1,
            },
        },
    )

    assert scope_deficits == {
        "development": {"localized": 2, "same_page_multi_section": 0},
    }
    assert role_deficits == {
        "development": {
            "student_inclusive": 2,
            "student_only": 2,
        },
    }
    assert [cell.to_dict() for cell in plan.cells] == [
        {
            "deficit_cell_id": (
                "post_filter_review:development:role:student_only:"
                "variation:situational_request"
            ),
            "count": 10,
            "split": "development",
            "query_language": None,
            "unit_kind": None,
            "source_family_id": None,
            "preferred_role_profile": "student_only",
            "preferred_query_variation": "situational_request",
            "requested_surface_form": None,
            "requested_evidence_scope": None,
        },
    ]


def test_role_backfill_uses_student_only_then_non_overlapping_inclusive_remainder():
    plan, _, role_deficits, _, _ = _accepted_deficit_plan(
        _config(),
        targets={"development": 10},
        accepted_by_split_and_scope={
            "development": {
                "localized": 9,
                "same_page_multi_section": 1,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 3,
                "student_only": 1,
            },
        },
    )

    assert role_deficits == {
        "development": {
            "student_inclusive": 4,
            "student_only": 2,
        },
    }
    assert [
        (
            cell.preferred_role_profile,
            cell.preferred_query_variation,
            cell.count,
        )
        for cell in plan.cells
    ] == [
        ("student_only", "situational_request", 10),
        ("student_inclusive", "situational_request", 10),
    ]


def test_primary_backfill_overlays_largest_relative_variation_deficit():
    plan, _, _, _, variation_deficits = _accepted_deficit_plan(
        _config(),
        targets={"development": 20},
        accepted_by_split_and_scope={
            "development": {
                "localized": 10,
                "same_page_multi_section": 1,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 8,
                "student_only": 3,
            },
        },
        accepted_by_split_and_query_framing={
            "development": {
                "direct_lookup": 11,
                "situational_request": 0,
            },
        },
        accepted_natural_paraphrases_by_split={"development": 3},
    )

    assert variation_deficits["development"] == {
        "natural_paraphrase": 1,
        "situational_request": 3,
    }
    assert plan.cells
    assert {
        cell.preferred_query_variation for cell in plan.cells
    } == {"situational_request"}


def test_primary_backfill_switches_to_natural_overlay_after_situational_target():
    plan, _, _, _, variation_deficits = _accepted_deficit_plan(
        _config(),
        targets={"development": 10},
        accepted_by_split_and_scope={
            "development": {
                "localized": 7,
                "same_page_multi_section": 1,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 5,
                "student_only": 1,
            },
        },
        accepted_by_split_and_query_framing={
            "development": {
                "direct_lookup": 6,
                "situational_request": 2,
            },
        },
        accepted_natural_paraphrases_by_split={"development": 0},
    )

    assert variation_deficits["development"] == {
        "natural_paraphrase": 2,
        "situational_request": 0,
    }
    assert plan.cells
    assert {
        cell.preferred_query_variation for cell in plan.cells
    } == {"natural_paraphrase"}


def test_primary_backfill_has_no_overlay_when_variation_targets_are_satisfied():
    plan, _, _, _, variation_deficits = _accepted_deficit_plan(
        _config(),
        targets={"development": 10},
        accepted_by_split_and_scope={
            "development": {
                "localized": 7,
                "same_page_multi_section": 1,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 5,
                "student_only": 1,
            },
        },
        accepted_by_split_and_query_framing={
            "development": {
                "direct_lookup": 6,
                "situational_request": 2,
            },
        },
        accepted_natural_paraphrases_by_split={"development": 2},
    )

    assert not any(variation_deficits["development"].values())
    assert plan.cells
    assert all(
        cell.preferred_query_variation is None for cell in plan.cells
    )


def test_surface_form_deficit_is_backfilled_after_other_targets_are_satisfied():
    plan, scope_deficits, role_deficits, surface_deficits, _ = (
        _accepted_deficit_plan(
            _config(),
            targets={"development": 10},
            accepted_by_split_and_scope={
                "development": {
                    "localized": 9,
                    "same_page_multi_section": 1,
                },
            },
            accepted_by_split_and_role_profile={
                "development": {
                    "student_inclusive": 7,
                    "student_only": 5,
                },
            },
            accepted_by_split_and_surface_form={
                "development": {
                    "complete_question": 3,
                    "conversational_short": 5,
                    "telegraphic_keywords": 1,
                },
            },
        )
    )

    assert not any(scope_deficits["development"].values())
    assert not any(role_deficits["development"].values())
    assert surface_deficits == {
        "development": {
            "complete_question": 0,
            "conversational_short": 0,
            "telegraphic_keywords": 1,
        },
    }
    assert len(plan.cells) == 1
    assert plan.cells[0].requested_surface_form == "telegraphic_keywords"
    assert plan.cells[0].preferred_role_profile is None


def test_query_variation_deficits_use_independent_reviewer_judgments():
    (
        plan,
        scope_deficits,
        role_deficits,
        surface_deficits,
        variation_deficits,
    ) = _accepted_deficit_plan(
        _config(),
        targets={"development": 10},
        accepted_by_split_and_scope={
            "development": {
                "localized": 9,
                "same_page_multi_section": 1,
            },
        },
        accepted_by_split_and_role_profile={
            "development": {
                "student_inclusive": 7,
                "student_only": 5,
            },
        },
        accepted_by_split_and_surface_form={
            "development": {
                "complete_question": 3,
                "conversational_short": 5,
                "telegraphic_keywords": 2,
            },
        },
        accepted_by_split_and_query_framing={
            "development": {
                "direct_lookup": 9,
                "situational_request": 1,
            },
        },
        accepted_natural_paraphrases_by_split={"development": 1},
    )

    assert not any(scope_deficits["development"].values())
    assert not any(role_deficits["development"].values())
    assert not any(surface_deficits["development"].values())
    assert variation_deficits == {
        "development": {
            "natural_paraphrase": 1,
            "situational_request": 1,
        },
    }
    assert [
        (cell.preferred_query_variation, cell.count) for cell in plan.cells
    ] == [
        ("natural_paraphrase", 10),
        ("situational_request", 10),
    ]


def test_filter_review_initializes_fresh_history_and_reports_phase_progress(
    tmp_path: Path,
):
    config_path, config = _write_job_fixture(tmp_path, suffix="-fresh-workflow")
    config["review"]["local_llm"].update(
        {
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
            "model_digest": None,
            "endpoint": None,
        },
    )
    config["review"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_review.md").resolve(),
    )
    _set_small_targets(config, 2)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    updates: list[FilterReviewProgress] = []

    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=_valid_provider,
        review_provider=_accepting_review_provider,
    ).run(
        generation_batch_limit=1,
        verify=False,
        progress_callback=updates.append,
    )

    assert result.status == "target_reached"
    for artifact_name in (
        "generation_batch_manifests",
        "generation_attempt_results",
        "question_candidates_raw",
        "question_candidates_filtered",
        "question_review_events",
    ):
        assert Path(config["artifact_paths"][artifact_name]).is_file()
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert manifests[0]["batch_purpose"] == "initial"
    assert all(
        manifest["batch_purpose"] == "post_filter_review"
        for manifest in manifests[1:]
    )
    phases = {
        (update.phase, update.generation_stage)
        for update in updates
    }
    assert phases == {
        ("generation", "initial"),
        ("filtering", None),
        ("review", None),
        ("generation", "backfill"),
    }


def test_increased_target_reuses_history_and_appends_backfill(tmp_path: Path):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=1,
        target=1,
        suffix="-increased-target",
    )
    first_review = QuestionReviewJob(
        config_path,
        repo_root=tmp_path,
        provider=_accepting_review_provider,
    ).run(verify=False)
    first_event_id = first_review.events[0].review_event_id

    increased = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _set_small_targets(increased, 2)
    config_path.write_text(yaml.safe_dump(increased), encoding="utf-8")

    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=_valid_provider,
        review_provider=_accepting_review_provider,
    ).run(generation_batch_limit=1, verify=False)

    assert result.status == "target_reached"
    assert result.accepted_by_split["development"] >= 2
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert manifests[0]["batch_purpose"] == "initial"
    assert all(
        manifest["batch_purpose"] == "post_filter_review"
        for manifest in manifests[1:]
    )
    events = read_jsonl(Path(config["artifact_paths"]["question_review_events"]))
    assert len(events) >= 2
    assert events[0]["review_event_id"] == first_event_id


def test_filter_review_reports_capacity_exhaustion_without_refunding_failures(
    tmp_path: Path,
):
    config_path, config = _prepare_pipeline(
        tmp_path,
        initial_candidates=0,
        target=1,
        suffix="-capacity",
    )
    result = FilterReviewJob(
        config_path,
        repo_root=tmp_path,
        generation_provider=lambda request: "not json",
        review_provider=_accepting_review_provider,
    ).run(generation_batch_limit=1, verify=False)

    assert result.status == "capacity_exhausted"
    assert result.deficits_by_split["development"] == 1
    attempts = read_jsonl(Path(config["artifact_paths"]["generation_attempt_results"]))
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    unit_cap = manifests[0]["selected_units"][0]["max_question_budget"]
    assert len(attempts) == unit_cap
    assert {attempt["status"] for attempt in attempts} == {"malformed_response"}
    assert sum(
        manifest["consumed_question_budget"] for manifest in manifests
    ) == unit_cap
    assert manifests[0]["batch_purpose"] == "initial"
    assert {
        manifest["batch_purpose"] for manifest in manifests[1:]
    } == {"post_filter_review"}
