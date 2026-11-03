from __future__ import annotations

from collections import Counter
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.thesis_rag.fixtures.eval_question_contracts import (
    mock_sha256,
    question_candidate_record,
    review_event_record,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.artifact_io import write_jsonl_atomic
from thesis_rag.eval.question_dedup import (
    QUESTION_DEDUP_EXCESS_SELECTION_VERSION,
    DedupStatistics,
    QuestionDedupAdjudicationLabel,
    QuestionDedupAdjudicationResponse,
    QuestionDedupError,
    QuestionDedupJob,
    _Pair,
    _adjudication_input,
    _apply_manual_decisions,
    _apply_adjudication,
    _classify_pairs,
    _dedup_construction_config,
    _materialize_clusters,
    _milp_target_candidates,
    _question_records,
    _query_relation_from_adjudication,
    _raise_for_technical_adjudication_failures,
    _removal_rank,
    _removed_candidate_ids,
    _semantic_pair_scores,
    _select_target_candidates,
    _target_selection,
    char_ngram_jaccard,
    normalize_question_text,
    token_jaccard,
)
from thesis_rag.eval.question_dedup_plan import QuestionDedupPlanJob
from thesis_rag.eval.question_review import CurrentQuestionReviews
from thesis_rag.eval.question_workflow import QuestionSetBuildJob
from thesis_rag.eval.schemas import (
    QuestionCandidateRecord,
    QuestionDedupManualDecisionRecord,
    QuestionReviewEventRecord,
    question_candidate_id,
    question_dedup_manual_decision_id,
    review_event_id,
)
from thesis_rag.interfaces import EvidenceTarget
from thesis_rag.schema import (
    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    stable_id,
)


def _config():
    raw = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    return QuestionConfig.from_dict(raw).deduplication


def _candidate(
    question: str,
    *,
    answer: str,
    language: str = "de",
    split: str = "development",
    suffix: str,
) -> QuestionCandidateRecord:
    base = question_candidate_record().to_dict()
    fact = answer
    target = EvidenceTarget(
        target_level="span",
        target_id=f"span:{suffix}",
        page_id=f"page:{suffix}",
        section_id=f"section:{suffix}",
        char_start=0,
        char_end=len(answer),
        text_hash=mock_sha256(f"target:{suffix}"),
        source_path=f"data/{suffix}.html",
        metadata={},
    )
    attempt_id = f"generation_attempt:{suffix}"
    source_family_id = f"source_family:{suffix}"
    base.update(
        {
            "source_family_id": source_family_id,
            "generation_attempt_id": attempt_id,
            "split": split,
            "question_text": question,
            "query_language": language,
            "evidence_language": language,
            "reference_answer": answer,
            "required_facts": [fact],
            "evidence_quotations": [
                {
                    "quotation": answer,
                    "required_fact_refs": ["fact:1"],
                    "source_ref": {"section_id": f"section:{suffix}"},
                },
            ],
            "verified_evidence_targets": [target.to_dict()],
            "corpus_hash": mock_sha256("corpus"),
        },
    )
    base["candidate_id"] = question_candidate_id(
        source_method=base["source_method"],
        source_family_id_value=source_family_id,
        generation_unit_id_value=base["generation_unit_id"],
        question_text=question,
        query_language=language,
        user_roles=base["user_roles"],
        reference_answer=answer,
        required_facts=[fact],
        generation_attempt_id=attempt_id,
    )
    return QuestionCandidateRecord.from_dict(base)


def _event(candidate: QuestionCandidateRecord) -> QuestionReviewEventRecord:
    base = review_event_record().to_dict()
    target = candidate.verified_evidence_targets[0]
    base.update(
        {
            "candidate_id": candidate.candidate_id,
            "anchor_assessments": [
                {
                    "target": target.to_dict(),
                    "grade": 3,
                    "status": "accepted",
                    "label": "sufficient",
                    "explanation": "The target supports the fact.",
                    "supported_fact_refs": ["fact:1"],
                    "propagation_rule": None,
                },
            ],
            "sufficient_evidence_sets": [{"target_ids": [target.target_id]}],
        },
    )
    base.pop("review_event_id", None)
    return QuestionReviewEventRecord.from_dict(
        {"review_event_id": review_event_id(base), **base},
    )


def _manual_decision(
    left: QuestionCandidateRecord,
    right: QuestionCandidateRecord,
    *,
    construction_hash: str,
    canonical_candidate_id: str | None,
    query_relation: str = "equivalent",
    answer_relation: str = "complementary",
    resolution: str = "canonicalize",
) -> QuestionDedupManualDecisionRecord:
    candidate_ids = sorted([left.candidate_id, right.candidate_id])
    raw = {
        "pair_id": stable_id("question_dedup_pair", *candidate_ids),
        "candidate_ids": candidate_ids,
        "construction_config_hash": construction_hash,
        "query_relation": query_relation,
        "answer_relation": answer_relation,
        "resolution": resolution,
        "canonical_candidate_id": canonical_candidate_id,
        "reviewer_kind": "human",
        "reviewer_id": "thesis_author",
        "rationale": "Reviewed pair decision.",
        "reviewed_at_utc": "2026-08-02T00:00:00Z",
        "schema_version": QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    }
    raw["manual_decision_id"] = question_dedup_manual_decision_id(raw)
    return QuestionDedupManualDecisionRecord.from_dict(raw)


def _with_surface(
    candidate: QuestionCandidateRecord,
    surface_form: str,
) -> QuestionCandidateRecord:
    raw = candidate.to_dict()
    raw["surface_form"] = surface_form
    return QuestionCandidateRecord.from_dict(raw)


def _with_filter_status(
    candidate: QuestionCandidateRecord,
    status: str,
) -> QuestionCandidateRecord:
    raw = candidate.to_dict()
    raw["deterministic_filter_status"] = status
    raw["deterministic_filter_reasons"] = (
        ["test_quality_flag"] if status == "flagged" else []
    )
    return QuestionCandidateRecord.from_dict(raw)


def _with_variation(
    event: QuestionReviewEventRecord,
    *,
    framing: str,
    paraphrase: bool,
) -> QuestionReviewEventRecord:
    raw = event.to_dict()
    raw.update(
        {
            "inferred_query_framing": framing,
            "natural_paraphrase_present": paraphrase,
        },
    )
    raw.pop("review_event_id")
    return QuestionReviewEventRecord.from_dict(
        {"review_event_id": review_event_id(raw), **raw},
    )


def test_normalization_and_lexical_similarity_ignore_harmless_formatting():
    left = "  Wann endet die Frist? "
    right = "wann endet die frist"

    assert normalize_question_text(left) == normalize_question_text(right)
    assert token_jaccard(left, right) == 1.0
    assert char_ngram_jaccard(left, right) == 1.0


def test_pair_identity_is_canonical_independent_of_candidate_order():
    pair = _Pair(
        left_id="question_candidate:z",
        right_id="question_candidate:a",
        kind="exact_duplicate",
        decision="canonicalize",
        query_relation="equivalent",
        answer_relation="equivalent",
        question_similarity={},
        answer_similarity={},
        safeguards={},
    )

    assert pair.ids == ("question_candidate:a", "question_candidate:z")


def test_manual_decision_overrides_pair_and_selects_reviewed_canonical():
    left = _candidate(
        "contact details for advising",
        answer="Email address.",
        split="development",
        suffix="manual-left",
    )
    right = _candidate(
        "advising contact details",
        answer="Email, address, and office hours.",
        split="validation",
        suffix="manual-right",
    )
    ids = sorted([left.candidate_id, right.candidate_id])
    pair = _Pair(
        left_id=ids[0],
        right_id=ids[1],
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.9},
        answer_similarity={"semantic_cosine": 0.7},
        safeguards={"number_conflict": False},
    )
    construction_hash = mock_sha256("manual-dedup")
    decision = _manual_decision(
        left,
        right,
        construction_hash=construction_hash,
        canonical_candidate_id=right.candidate_id,
    )

    resolved = _apply_manual_decisions(
        (pair,),
        candidates=(left, right),
        decisions=(decision,),
    )
    clusters = _materialize_clusters(
        resolved,
        candidates=(left, right),
        construction_hash=construction_hash,
        corpus_hash=mock_sha256("corpus"),
    )

    assert resolved[0].decision == "canonicalize"
    assert resolved[0].safeguards["manual_resolution"]["reviewer_id"] == (
        "thesis_author"
    )
    assert clusters[0].decision == "canonicalize"
    assert clusters[0].canonical_candidate_id == right.candidate_id
    assert clusters[0].retained_candidate_ids == [right.candidate_id]


def test_manual_decision_can_add_pair_below_automatic_proposal_thresholds():
    left = _candidate(
        "academic advising hours",
        answer="Tuesday through Thursday.",
        suffix="manual-added-left",
    )
    right = _candidate(
        "when can I reach general student advising",
        answer="Tuesday through Thursday.",
        suffix="manual-added-right",
    )
    construction_hash = mock_sha256("manual-added-pair")
    decision = _manual_decision(
        left,
        right,
        construction_hash=construction_hash,
        canonical_candidate_id=left.candidate_id,
        answer_relation="equivalent",
    )

    resolved = _apply_manual_decisions(
        (),
        candidates=(left, right),
        decisions=(decision,),
    )
    clusters = _materialize_clusters(
        resolved,
        candidates=(left, right),
        construction_hash=construction_hash,
        corpus_hash=mock_sha256("corpus"),
    )

    assert len(resolved) == 1
    assert resolved[0].safeguards["human_added_pair"] is True
    assert resolved[0].safeguards["manual_resolution"]["reviewer_id"] == (
        "thesis_author"
    )
    assert clusters[0].decision == "canonicalize"
    assert clusters[0].canonical_candidate_id == left.candidate_id


def test_manual_keep_all_resolves_technical_pair_without_canonicalization():
    left = _candidate(
        "semester fees",
        answer="A fee is charged.",
        suffix="manual-fees",
    )
    right = _candidate(
        "student job vacancies",
        answer="Vacancies are listed online.",
        suffix="manual-jobs",
    )
    ids = sorted([left.candidate_id, right.candidate_id])
    pair = _Pair(
        left_id=ids[0],
        right_id=ids[1],
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.7},
        answer_similarity={"semantic_cosine": 0.2},
        safeguards={"number_conflict": False},
    )
    construction_hash = mock_sha256("manual-distinct")
    decision = _manual_decision(
        left,
        right,
        construction_hash=construction_hash,
        canonical_candidate_id=None,
        query_relation="distinct",
        answer_relation="unrelated",
        resolution="keep_all",
    )

    resolved = _apply_manual_decisions(
        (pair,),
        candidates=(left, right),
        decisions=(decision,),
    )

    assert resolved[0].kind == "similar_but_distinct"
    assert resolved[0].decision == "keep_all"
    assert resolved[0].preferred_canonical_candidate_id is None


def test_reviewed_canonical_is_retained_first_in_multi_candidate_component():
    canonical = _candidate(
        "Wann endet die Frist?",
        answer="Am 1. April.",
        suffix="manual-component-canonical",
    )
    translation = _candidate(
        "When is the deadline?",
        answer="On April 1.",
        language="en",
        suffix="manual-component-translation",
    )
    duplicate = _candidate(
        "Bis wann läuft die Frist?",
        answer="Bis zum 1. April.",
        suffix="manual-component-duplicate",
    )

    def pair(left, right, *, decision, preferred=None):
        return _Pair(
            left_id=left.candidate_id,
            right_id=right.candidate_id,
            kind="same_language_paraphrase",
            decision=decision,
            query_relation="equivalent",
            answer_relation="equivalent",
            question_similarity={},
            answer_similarity={},
            safeguards={},
            preferred_canonical_candidate_id=preferred,
        )

    clusters = _materialize_clusters(
        (
            pair(
                canonical,
                translation,
                decision="retain_variants",
                preferred=canonical.candidate_id,
            ),
            pair(canonical, duplicate, decision="canonicalize"),
            pair(translation, duplicate, decision="retain_variants"),
        ),
        candidates=(canonical, translation, duplicate),
        construction_hash=mock_sha256("manual-component"),
        corpus_hash=mock_sha256("corpus"),
    )

    assert len(clusters) == 1
    assert clusters[0].canonical_candidate_id == canonical.candidate_id
    assert clusters[0].retained_candidate_ids == sorted(
        [canonical.candidate_id, translation.candidate_id],
    )


def test_semantic_proposals_include_every_pair_above_threshold():
    candidates = tuple(
        _candidate(
            f"Ähnliche Frage {index}",
            answer=f"Antwort {index}",
            suffix=f"all-pairs-{index}",
        )
        for index in range(14)
    )
    embeddings = {
        candidate.candidate_id: ((1.0, 0.0), (1.0, 0.0))
        for candidate in candidates
    }

    scores = _semantic_pair_scores(
        candidates,
        embeddings=embeddings,
        minimum_similarity=0.65,
    )

    assert len(scores) == 14 * 13 // 2
    assert set(scores.values()) == {1.0}


def test_semantic_answer_rescue_reaches_production_pair_classification():
    left = _candidate(
        "Wann findet die Lange Nacht statt?",
        answer="Sie findet am 19. Juni statt.",
        suffix="answer-rescue-left",
    )
    right = _candidate(
        "Nächstes Datum der Wissenschaftsveranstaltung",
        answer="Der Termin ist der 19. Juni.",
        suffix="answer-rescue-right",
    )
    question_score = 0.65
    answer_score = 0.95
    embeddings = {
        left.candidate_id: ((1.0, 0.0), (1.0, 0.0)),
        right.candidate_id: (
            (question_score, sqrt(1.0 - question_score**2)),
            (answer_score, sqrt(1.0 - answer_score**2)),
        ),
    }

    pairs = _classify_pairs(
        (left, right),
        config=_config(),
        embeddings=embeddings,
    )

    assert len(pairs) == 1
    assert pairs[0].decision == "manual_review"
    assert pairs[0].question_similarity["semantic_cosine"] == question_score
    assert pairs[0].answer_similarity["semantic_cosine"] == answer_score


def test_cross_language_route_uses_relaxed_joint_thresholds():
    left = _candidate(
        "Rentenversicherungsnachweis beantragen",
        answer="Der Nachweis kann online beantragt werden.",
        language="de",
        suffix="cross-language-left",
    )
    right = _candidate(
        "How do I request proof of statutory pension insurance?",
        answer="Request the certificate online.",
        language="en",
        suffix="cross-language-right",
    )
    question_score = 0.66
    answer_score = 0.80
    embeddings = {
        left.candidate_id: ((1.0, 0.0), (1.0, 0.0)),
        right.candidate_id: (
            (question_score, sqrt(1.0 - question_score**2)),
            (answer_score, sqrt(1.0 - answer_score**2)),
        ),
    }

    pairs = _classify_pairs(
        (left, right),
        config=_config(),
        embeddings=embeddings,
    )

    assert len(pairs) == 1
    assert pairs[0].safeguards["answers_near_identical"] is False
    assert pairs[0].question_similarity["semantic_cosine"] == question_score
    assert pairs[0].answer_similarity["semantic_cosine"] == answer_score

    same_language = _candidate(
        "Wie beantrage ich den Nachweis der gesetzlichen Rentenversicherung?",
        answer="Der Nachweis wird online beantragt.",
        language="de",
        suffix="cross-language-negative",
    )
    same_language_embeddings = {
        left.candidate_id: embeddings[left.candidate_id],
        same_language.candidate_id: embeddings[right.candidate_id],
    }

    assert _classify_pairs(
        (left, same_language),
        config=_config(),
        embeddings=same_language_embeddings,
    ) == ()


def test_adjudication_input_contains_verified_evidence_quotations():
    left = _candidate("Frist?", answer="Die Frist ist am 1. April.", suffix="quote-a")
    right = _candidate(
        "Wann endet die Frist?",
        answer="Die Frist endet am 1. April.",
        suffix="quote-b",
    )
    pair = _Pair(
        left_id=left.candidate_id,
        right_id=right.candidate_id,
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.8},
        answer_similarity={"semantic_cosine": 0.9},
        safeguards={},
    )

    payload = _adjudication_input(pair, left=left, right=right)

    assert payload["candidate_a"]["evidence_quotations"] == [
        {
            "quotation": "Die Frist ist am 1. April.",
            "required_fact_refs": ["fact:1"],
        },
    ]
    assert payload["candidate_b"]["evidence_quotations"] == [
        {
            "quotation": "Die Frist endet am 1. April.",
            "required_fact_refs": ["fact:1"],
        },
    ]


@pytest.mark.parametrize(
    ("subject_relation", "scope_relation", "expected"),
    [
        ("same", "same", "equivalent"),
        ("same", "overlapping", "partial_overlap"),
        ("same", "different", "distinct"),
        ("different", "same", "distinct"),
        ("uncertain", "same", "uncertain"),
        ("same", "uncertain", "uncertain"),
    ],
)
def test_query_relation_is_derived_from_subject_and_scope(
    subject_relation,
    scope_relation,
    expected,
):
    label = QuestionDedupAdjudicationLabel(
        pair_id="question_dedup_pair:test",
        subject_relation=subject_relation,
        requested_scope_relation=scope_relation,
        answer_relation="equivalent",
        explanation="Compared the concrete subject and requested scope.",
    )

    assert _query_relation_from_adjudication(label) == expected


def test_exact_question_with_compatible_answer_is_canonicalized():
    left = _candidate(
        "Wann endet die Frist?",
        answer="Die Frist endet am 1. April.",
        suffix="left",
    )
    right = _candidate(
        "wann endet die frist",
        answer="Die Frist endet am 1. April.",
        suffix="right",
    )

    pairs = _classify_pairs((left, right), config=_config(), embeddings={})

    assert len(pairs) == 1
    assert pairs[0].kind == "exact_duplicate"
    assert pairs[0].decision == "canonicalize"


def test_similar_question_with_conflicting_numbers_requires_adjudication():
    left = _candidate(
        "Wann endet die Frist?",
        answer="Die Frist endet am 1. April.",
        suffix="left",
    )
    right = _candidate(
        "wann endet die frist",
        answer="Die Frist endet am 2. April.",
        suffix="right",
    )

    pair = _classify_pairs((left, right), config=_config(), embeddings={})[0]

    assert pair.kind == "uncertain"
    assert pair.decision == "manual_review"
    assert pair.query_relation == "equivalent"
    assert pair.answer_relation == "conflicting"
    assert pair.safeguards["number_conflict"] is True


def test_lexically_similar_question_with_different_answer_requires_adjudication():
    left = _candidate(
        "Wo beantrage ich den Studierendenausweis?",
        answer="Der Ausweis wird beim Studierendensekretariat beantragt.",
        suffix="left",
    )
    right = _candidate(
        "Wo beantrage ich den Studierendenausweis?",
        answer="Der Ausweis wird im Campus Center abgeholt.",
        suffix="right",
    )

    pair = _classify_pairs((left, right), config=_config(), embeddings={})[0]

    assert pair.kind == "uncertain"
    assert pair.decision == "manual_review"


def test_shared_evidence_target_does_not_make_different_answers_equivalent():
    left = _candidate(
        "Was bietet der Career Service an?",
        answer="Der Career Service berät zur Karriereplanung.",
        suffix="left",
    )
    right = _candidate(
        "Was bietet der Career Service an?",
        answer="Der Career Service vermittelt Praktika.",
        suffix="right",
    )
    right_payload = right.to_dict()
    right_payload["verified_evidence_targets"] = [
        target.to_dict() for target in left.verified_evidence_targets
    ]
    right = QuestionCandidateRecord.from_dict(right_payload)

    pair = _classify_pairs((left, right), config=_config(), embeddings={})[0]

    assert pair.safeguards["shared_evidence_target"] is True
    assert pair.safeguards["answers_near_identical"] is False
    assert pair.decision == "manual_review"


def test_semantic_translation_requires_review_then_retains_same_split_variants():
    left = _candidate(
        "Welche Räume bietet EINS?",
        answer="EINS bietet Arbeitsplätze und Seminarräume.",
        suffix="de",
    )
    right = _candidate(
        "What facilities does EINS offer?",
        answer="EINS offers workspaces and seminar rooms.",
        language="en",
        suffix="en",
    )
    embeddings = {
        left.candidate_id: ((1.0, 0.0), (0.0, 1.0)),
        right.candidate_id: ((1.0, 0.0), (0.0, 1.0)),
    }

    pair = _classify_pairs((left, right), config=_config(), embeddings=embeddings)[0]

    assert pair.kind == "uncertain"
    assert pair.decision == "manual_review"

    reviewed = _apply_adjudication(
        pair,
        response=QuestionDedupAdjudicationResponse(
            query_relation="equivalent",
            answer_relation="equivalent",
            explanation="The two questions and answers are translations.",
        ),
        left=left,
        right=right,
        keep_translation_pairs=True,
        provenance={"status": "test"},
    )

    assert reviewed.kind == "translation_equivalent"
    assert reviewed.decision == "retain_variants"


def test_semantic_cross_split_translation_requires_review():
    left = _candidate(
        "Welche Räume bietet EINS?",
        answer="EINS bietet Arbeitsplätze und Seminarräume.",
        split="validation",
        suffix="de",
    )
    right = _candidate(
        "What facilities does EINS offer?",
        answer="EINS offers workspaces and seminar rooms.",
        language="en",
        split="test",
        suffix="en",
    )
    embeddings = {
        left.candidate_id: ((1.0, 0.0), (0.0, 1.0)),
        right.candidate_id: ((1.0, 0.0), (0.0, 1.0)),
    }

    pair = _classify_pairs((left, right), config=_config(), embeddings=embeddings)[0]

    assert pair.kind == "uncertain"
    assert pair.decision == "manual_review"


def test_equivalent_query_with_complementary_answers_is_canonicalized():
    left = _candidate(
        "Kontakt Studienberatung",
        answer="Die Studienberatung ist per E-Mail erreichbar.",
        suffix="contact-email-review",
    )
    right = _candidate(
        "Wie erreiche ich die Studienberatung?",
        answer="Die Studienberatung bietet außerdem offene Zoom-Sprechstunden an.",
        suffix="contact-zoom-review",
    )
    pair = _Pair(
        left_id=min(left.candidate_id, right.candidate_id),
        right_id=max(left.candidate_id, right.candidate_id),
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.9},
        answer_similarity={"semantic_cosine": 0.6},
        safeguards={"number_conflict": False},
    )

    reviewed = _apply_adjudication(
        pair,
        response=QuestionDedupAdjudicationResponse(
            query_relation="equivalent",
            answer_relation="complementary",
            explanation="The need may be shared, but the answers add different facts.",
        ),
        left=left,
        right=right,
        keep_translation_pairs=True,
        provenance={"status": "test"},
    )

    assert reviewed.kind == "same_language_paraphrase"
    assert reviewed.decision == "canonicalize"
    assert reviewed.query_relation == "equivalent"
    assert reviewed.answer_relation == "complementary"


def test_exact_query_overrides_a_less_similar_llm_query_label():
    left = _candidate(
        "Womit beschäftigt sich das Fachgebiet Schienenfahrzeuge?",
        answer="Es erforscht Schienenfahrzeuge im Personen- und Güterverkehr.",
        suffix="rail-broad",
    )
    right = _candidate(
        "womit beschäftigt sich das fachgebiet schienenfahrzeuge",
        answer="Es verbindet Forschung zu Schienenfahrzeugen mit der Lehre.",
        suffix="rail-narrow",
    )
    pair = _Pair(
        left_id=min(left.candidate_id, right.candidate_id),
        right_id=max(left.candidate_id, right.candidate_id),
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"normalized_exact": 1.0},
        answer_similarity={"semantic_cosine": 0.7},
        safeguards={"number_conflict": False},
    )

    reviewed = _apply_adjudication(
        pair,
        response=QuestionDedupAdjudicationResponse(
            query_relation="partial_overlap",
            answer_relation="complementary",
            explanation="The second answer adds teaching information.",
        ),
        left=left,
        right=right,
        keep_translation_pairs=True,
        provenance={"status": "test"},
    )

    assert reviewed.kind == "exact_duplicate"
    assert reviewed.decision == "canonicalize"
    assert reviewed.query_relation == "equivalent"
    assert reviewed.answer_relation == "complementary"
    assert reviewed.safeguards["adjudication"] == {
        "status": "test",
        "reported_query_relation": "partial_overlap",
        "normalized_exact_query_override": True,
    }


def test_equivalent_query_with_conflicting_answers_is_canonicalized():
    left = _candidate(
        "Wann findet das Unikino statt?",
        answer="Das Unikino findet dienstags statt.",
        split="development",
        suffix="cinema-tuesday",
    )
    right = _candidate(
        "Wann ist das Unikino?",
        answer="Das Unikino findet donnerstags statt.",
        split="test",
        suffix="cinema-thursday",
    )
    pair = _Pair(
        left_id=min(left.candidate_id, right.candidate_id),
        right_id=max(left.candidate_id, right.candidate_id),
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.9},
        answer_similarity={"semantic_cosine": 0.7},
        safeguards={"number_conflict": False},
    )

    reviewed = _apply_adjudication(
        pair,
        response=QuestionDedupAdjudicationResponse(
            query_relation="equivalent",
            answer_relation="conflicting",
            explanation="The same event is assigned different weekdays.",
        ),
        left=left,
        right=right,
        keep_translation_pairs=True,
        provenance={"status": "test"},
    )

    assert reviewed.kind == "same_language_paraphrase"
    assert reviewed.decision == "canonicalize"
    assert reviewed.query_relation == "equivalent"
    assert reviewed.answer_relation == "conflicting"

    cluster = _materialize_clusters(
        (reviewed,),
        candidates=(left, right),
        construction_hash=mock_sha256("dedup-config"),
        corpus_hash=mock_sha256("corpus"),
    )[0]

    assert cluster.decision == "canonicalize"
    assert cluster.answer_relations == ["conflicting"]
    assert cluster.candidate_ids == sorted([left.candidate_id, right.candidate_id])
    assert cluster.source_family_ids == sorted(
        [left.source_family_id, right.source_family_id],
    )


def test_equivalent_query_with_complementary_answers_retains_both_variants():
    left = _candidate(
        "Kontakt Studienberatung",
        answer="Die Studienberatung ist per E-Mail erreichbar.",
        suffix="contact-email",
    )
    right = _candidate(
        "Wie erreiche ich die Studienberatung?",
        answer="Die Studienberatung bietet außerdem offene Zoom-Sprechstunden an.",
        suffix="contact-zoom",
    )
    pair = _Pair(
        left_id=min(left.candidate_id, right.candidate_id),
        right_id=max(left.candidate_id, right.candidate_id),
        kind="same_language_paraphrase",
        decision="retain_variants",
        query_relation="equivalent",
        answer_relation="complementary",
        question_similarity={"semantic_cosine": 0.9},
        answer_similarity={"semantic_cosine": 0.6},
        safeguards={"number_conflict": False},
    )

    clusters = _materialize_clusters(
        (pair,),
        candidates=(left, right),
        construction_hash=mock_sha256("dedup-config"),
        corpus_hash=mock_sha256("corpus"),
    )

    assert len(clusters) == 1
    assert clusters[0].decision == "retain_variants"
    assert clusters[0].retained_candidate_ids == sorted(
        [left.candidate_id, right.candidate_id],
    )
    assert clusters[0].answer_relations == ["complementary"]
    assert _removed_candidate_ids(clusters) == set()


def test_cross_split_equivalent_query_keeps_one_query_split_but_all_source_links():
    left = _candidate(
        "Kontakt Studienberatung",
        answer="Die Studienberatung ist per E-Mail erreichbar.",
        split="development",
        suffix="contact-dev",
    )
    right = _candidate(
        "Wie erreiche ich die Studienberatung?",
        answer="Die Studienberatung bietet außerdem offene Zoom-Sprechstunden an.",
        split="test",
        suffix="contact-test",
    )
    pair = _Pair(
        left_id=min(left.candidate_id, right.candidate_id),
        right_id=max(left.candidate_id, right.candidate_id),
        kind="same_language_paraphrase",
        decision="canonicalize",
        query_relation="equivalent",
        answer_relation="complementary",
        question_similarity={"semantic_cosine": 0.9},
        answer_similarity={"semantic_cosine": 0.6},
        safeguards={"number_conflict": False},
    )

    cluster = _materialize_clusters(
        (pair,),
        candidates=(left, right),
        construction_hash=mock_sha256("dedup-config"),
        corpus_hash=mock_sha256("corpus"),
    )[0]

    assert cluster.decision == "canonicalize"
    assert cluster.retained_splits == ["test"]
    assert cluster.retained_candidate_ids == [right.candidate_id]
    assert cluster.candidate_ids == sorted([left.candidate_id, right.candidate_id])
    assert cluster.source_family_ids == sorted(
        [left.source_family_id, right.source_family_id],
    )

    right_with_context = right.to_dict()
    right_with_context["metadata"] = {
        "unit_kind": "context_group",
        "generation_context_scope": "context_group",
        "context_group_ids": ["section_context_group:contact-test"],
        "section_ids": ["section:contact-test"],
    }
    right = QuestionCandidateRecord.from_dict(right_with_context)
    question = _question_records((right,), {right.candidate_id: _event(right)}, (cluster,))[0]
    assert question.metadata["information_need_cluster_ids"] == [
        cluster.dedup_cluster_id,
    ]
    assert question.metadata["judgment_pool_candidate_ids"] == cluster.candidate_ids
    assert question.metadata["related_candidate_ids"] == [left.candidate_id]
    assert question.metadata["judgment_pool_source_family_ids"] == (
        cluster.source_family_ids
    )
    assert question.metadata["unit_kind"] == "context_group"
    assert question.metadata["generation_context_scope"] == "context_group"
    assert question.metadata["context_group_ids"] == [
        "section_context_group:contact-test",
    ]
    assert question.metadata["section_ids"] == ["section:contact-test"]
    assert question.acceptable_targets == [right.verified_evidence_targets[0]]


def test_job_writes_clusters_report_and_materialized_questions(tmp_path, monkeypatch):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_dedup_adjudication.md").resolve(),
    )
    config = QuestionConfig.from_dict(raw_config)
    left = _candidate(
        "Wann endet die Frist?",
        answer="Die Frist endet am 1. April.",
        suffix="left",
    )
    right = _candidate(
        "wann endet die frist",
        answer="Die Frist endet am 1. April.",
        suffix="right",
    )
    events = (_event(left), _event(right))
    reviews = CurrentQuestionReviews(
        candidates=(left, right),
        events=events,
        events_by_candidate_id={event.candidate_id: event for event in events},
        missing_candidate_ids=(),
    )
    state = SimpleNamespace(config=config, repo_root=tmp_path)
    filtered_path = tmp_path / config.artifact_paths.question_candidates_filtered
    review_path = tmp_path / config.artifact_paths.question_review_events
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.write_text("filtered\n", encoding="utf-8")
    review_path.write_text("reviews\n", encoding="utf-8")
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup.load_verified_generation_state",
        lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup.load_current_question_reviews",
        lambda _state: reviews,
    )
    progress = []

    result = QuestionDedupJob(
        repo_root=tmp_path,
        embedding_provider=lambda texts: [(1.0, 0.0) for _ in texts],
    ).run(verify=False, progress_callback=progress.append)

    assert len(result.clusters) == 1
    assert result.clusters[0].decision == "canonicalize"
    assert len(result.removed_candidate_ids) == 1
    assert len(result.selected_candidates) == 1
    assert (tmp_path / result.output_path).is_file()
    assert (tmp_path / result.report_path).is_file()
    assert (tmp_path / result.questions_path).is_file()
    assert [update.phase for update in progress] == [
        "loading",
        "loading",
        "embedding",
        "embedding",
        "matching",
        "matching",
        "adjudication",
    ]
    assert progress[-1].processed == progress[-1].total == 0


def test_dedup_plan_uses_only_accepts_and_writes_no_canonical_artifacts(
    tmp_path,
    monkeypatch,
):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config = QuestionConfig.from_dict(raw_config)
    left = _candidate(
        "Wann endet die Frist?",
        answer="Die Frist endet am 1. April.",
        suffix="plan-left",
    )
    right = _candidate(
        "Bis wann läuft die Frist?",
        answer="Die Frist endet am 1. April.",
        suffix="plan-right",
    )
    rejected = _candidate(
        "Unklare Seite?",
        answer="Nicht relevant.",
        suffix="plan-rejected",
    )
    left_event = _event(left)
    right_event = _event(right)
    rejected_raw = _event(rejected).to_dict()
    rejected_raw.update(
        {
            "decision": "reject",
            "reason_codes": ["unnatural_query"],
            "query_form_fit": False,
            "sufficient_evidence_sets": [],
        },
    )
    rejected_raw["review_event_id"] = review_event_id(rejected_raw)
    rejected_event = QuestionReviewEventRecord.from_dict(rejected_raw)
    events = (left_event, right_event, rejected_event)
    reviews = CurrentQuestionReviews(
        candidates=(left, right, rejected),
        events=events,
        events_by_candidate_id={event.candidate_id: event for event in events},
        missing_candidate_ids=(),
    )
    state = SimpleNamespace(
        config=config,
        repo_root=tmp_path,
        raw_candidates=(left, right, rejected),
    )
    filtered_path = tmp_path / config.artifact_paths.question_candidates_filtered
    review_path = tmp_path / config.artifact_paths.question_review_events
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(filtered_path, (left, right, rejected))
    review_path.write_text("reviews\n", encoding="utf-8")
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup_plan.load_verified_generation_state",
        lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup_plan._reviewed_snapshot_state",
        lambda active_state: active_state,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup_plan.load_current_question_reviews",
        lambda _state: reviews,
    )
    embedded_texts = []

    def embed(texts):
        embedded_texts.extend(texts)
        return [(1.0, 0.0) for _ in texts]

    result = QuestionDedupPlanJob(
        repo_root=tmp_path,
        embedding_provider=embed,
    ).run(verify=False)

    assert result.accepted_count == 2
    assert result.total_pair_count == 1
    assert len(embedded_texts) == 4
    assert result.report["counts"]["excluded_non_accepted_candidates"] == 1
    assert result.report["configuration"]["question_config_id"] == (
        config.question_config_id
    )
    assert result.report["selected_profile"]["answer_rescue_question_floor"] == 0.50
    assert result.report["selected_profile"]["answer_rescue_answer_floor"] == 0.84
    assert result.report["selected_profile"]["cross_language_question_floor"] == 0.65
    assert result.report["selected_profile"]["cross_language_answer_floor"] == 0.75
    assert result.report["selected_profile"]["configured_adjudication"] == {
        "maximum_adjudication_pairs": 1600,
        "batch_size": 5,
        "reasoning_effort": "none",
        "base_request_count_upper_bound": 1,
        "request_attempt_upper_bound_with_retries": 3,
    }
    assert result.report["side_effects"]["llm_calls"] == 0
    assert (tmp_path / result.report_path).is_file()
    assert not (
        tmp_path / config.artifact_paths.question_dedup_clusters
    ).exists()
    assert not (
        tmp_path / config.artifact_paths.question_dedup_report
    ).exists()
    assert not (
        tmp_path / config.artifact_paths.questions_llm_reviewed
    ).exists()


def test_conflicting_edge_prevents_transitive_cluster_collapse():
    candidates = tuple(
        _candidate(
            f"Frage {suffix}",
            answer="Dieselbe Antwort.",
            suffix=suffix,
        )
        for suffix in ("a", "b", "c")
    )
    a, b, c = (candidate.candidate_id for candidate in candidates)

    def pair(left, right, decision):
        return _Pair(
            left_id=min(left, right),
            right_id=max(left, right),
            kind=(
                "same_language_paraphrase"
                if decision == "canonicalize"
                else "similar_but_distinct"
            ),
            decision=decision,
            query_relation=(
                "equivalent" if decision == "canonicalize" else "distinct"
            ),
            answer_relation=(
                "equivalent" if decision == "canonicalize" else "conflicting"
            ),
            question_similarity={"token_jaccard": 0.99},
            answer_similarity={"token_jaccard": 1.0},
            safeguards={"number_conflict": decision == "keep_all"},
        )

    clusters = _materialize_clusters(
        tuple(sorted((pair(a, b, "canonicalize"), pair(b, c, "canonicalize"), pair(a, c, "keep_all")), key=lambda item: item.ids)),
        candidates=candidates,
        construction_hash=mock_sha256("dedup-config"),
        corpus_hash=mock_sha256("corpus"),
    )

    assert len(clusters) == 1
    assert clusters[0].decision == "manual_review"
    assert clusters[0].canonical_candidate_id is None


def test_missing_transitive_edge_requires_manual_cluster_review():
    candidates = tuple(
        _candidate(
            f"Frage {suffix}",
            answer="Dieselbe Antwort.",
            suffix=f"bridge-{suffix}",
        )
        for suffix in ("a", "b", "c")
    )
    a, b, c = (candidate.candidate_id for candidate in candidates)

    def equivalent_pair(left, right):
        return _Pair(
            left_id=min(left, right),
            right_id=max(left, right),
            kind="same_language_paraphrase",
            decision="canonicalize",
            query_relation="equivalent",
            answer_relation="equivalent",
            question_similarity={"semantic_cosine": 0.8},
            answer_similarity={"semantic_cosine": 1.0},
            safeguards={"number_conflict": False},
        )

    clusters = _materialize_clusters(
        tuple(
            sorted(
                (equivalent_pair(a, b), equivalent_pair(b, c)),
                key=lambda item: item.ids,
            ),
        ),
        candidates=candidates,
        construction_hash=mock_sha256("dedup-config"),
        corpus_hash=mock_sha256("corpus"),
    )

    assert len(clusters) == 1
    assert clusters[0].decision == "manual_review"
    assert clusters[0].canonical_candidate_id is None
    assert clusters[0].evidence["cluster_safeguards"] == {
        "complete_pairwise_support": False,
        "missing_pair_count": 1,
    }


def test_target_selection_keeps_required_surface_and_variation_coverage():
    complete_a = _candidate(
        "Wann endet die erste Frist?",
        answer="Die Frist endet im April.",
        suffix="complete-a",
    )
    complete_b = _candidate(
        "Wann endet die zweite Frist?",
        answer="Die Frist endet im Mai.",
        suffix="complete-b",
    )
    conversational = _with_surface(
        _candidate(
            "hilfe mit frist",
            answer="Die Frist endet im Juni.",
            suffix="conversational",
        ),
        "conversational_short",
    )
    direct_a = _event(complete_a)
    direct_b = _event(complete_b)
    situation = _with_variation(
        _event(conversational),
        framing="situational_request",
        paraphrase=True,
    )
    events = {
        event.candidate_id: event
        for event in (direct_a, direct_b, situation)
    }
    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )

    selected = _select_target_candidates(
        (complete_a, complete_b, conversational),
        events_by_id=events,
        config=config,
        targets={"development": 2, "validation": 0, "test": 0},
    )

    assert len(selected) == 2
    assert conversational.candidate_id in {
        candidate.candidate_id for candidate in selected
    }


def test_exact_selector_finds_a_constraint_complete_target():
    candidates = (
        _with_surface(
            _candidate(
                "Wann endet die Frist?",
                answer="Im April.",
                suffix="milp-complete",
            ),
            "complete_question",
        ),
        *(
            _with_surface(
                _candidate(
                    f"hilfe mit frist {index}",
                    answer=f"Im Monat {index}.",
                    suffix=f"milp-conversational-{index}",
                ),
                "conversational_short",
            )
            for index in range(3)
        ),
        _with_surface(
            _candidate(
                "frist termin",
                answer="Im Mai.",
                suffix="milp-telegraphic",
            ),
            "telegraphic_keywords",
        ),
    )
    events = {candidate.candidate_id: _event(candidate) for candidate in candidates}
    first = candidates[0]
    events[first.candidate_id] = _with_variation(
        events[first.candidate_id],
        framing="situational_request",
        paraphrase=True,
    )
    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )

    selected = _milp_target_candidates(
        candidates,
        events_by_id=events,
        config=config,
        targets={"development": 4, "validation": 0, "test": 0},
        clusters=(),
    )

    assert selected is not None
    assert len(selected) == 4
    assert Counter(candidate.surface_form for candidate in selected) == {
        "complete_question": 1,
        "conversational_short": 2,
        "telegraphic_keywords": 1,
    }


def test_selector_reports_infeasible_when_a_surface_cell_is_exhausted():
    candidates = tuple(
        _with_surface(
            _candidate(
                f"hilfe mit frist {index}",
                answer=f"Im Monat {index}.",
                suffix=f"milp-infeasible-{index}",
            ),
            "conversational_short" if index < 3 else "telegraphic_keywords",
        )
        for index in range(4)
    )
    events = {candidate.candidate_id: _event(candidate) for candidate in candidates}
    first = candidates[0]
    events[first.candidate_id] = _with_variation(
        events[first.candidate_id],
        framing="situational_request",
        paraphrase=True,
    )
    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )

    selection = _target_selection(
        candidates,
        events_by_id=events,
        config=config,
        targets={"development": 4, "validation": 0, "test": 0},
    )

    assert selection.method == "infeasible"
    assert selection.exact_feasible is False
    assert selection.selected == tuple(
        sorted(candidates, key=lambda item: (item.split, item.candidate_id)),
    )


def test_excess_removal_rank_prioritizes_flags_and_binds_the_seed():
    passed = _candidate(
        "Wie beantrage ich den Ausweis?",
        answer="Der Antrag wird im Portal gestellt.",
        suffix="seeded-passed",
    )
    flagged = _with_filter_status(
        _candidate(
            "Ausweis im Portal beantragen",
            answer="Der Antrag wird im Portal gestellt.",
            suffix="seeded-flagged",
        ),
        "flagged",
    )

    passed_rank = _removal_rank(passed, seed=311037)
    flagged_rank = _removal_rank(flagged, seed=311037)

    assert flagged_rank[0] < passed_rank[0]
    assert passed_rank == _removal_rank(passed, seed=311037)
    assert passed_rank != _removal_rank(passed, seed=311038)

    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )
    construction = _dedup_construction_config(
        SimpleNamespace(config=config),
        adjudication_prompt="Deduplicate the supplied pairs.",
    )
    assert construction["excess_selection"] == {
        "selection_version": QUESTION_DEDUP_EXCESS_SELECTION_VERSION,
        "policy": "flagged_then_seeded_quota_preserving",
        "random_seed": 311037,
    }


def test_target_selection_removes_flagged_before_passed_with_quotas_intact():
    complete = _candidate(
        "Wann endet die Bewerbungsfrist?",
        answer="Die Frist endet im April.",
        suffix="flagged-selection-complete",
    )
    passed_conversational = _with_surface(
        _candidate(
            "ich brauche hilfe mit der bewerbungsfrist",
            answer="Die Frist endet im April.",
            suffix="flagged-selection-passed",
        ),
        "conversational_short",
    )
    flagged_conversational = _with_filter_status(
        _with_surface(
            _candidate(
                "hilfe mit bewerbungsfrist und weiteren informationen",
                answer="Die Frist endet im April.",
                suffix="flagged-selection-flagged",
            ),
            "conversational_short",
        ),
        "flagged",
    )
    complete_event = _event(complete)
    passed_event = _with_variation(
        _event(passed_conversational),
        framing="situational_request",
        paraphrase=True,
    )
    flagged_event = _with_variation(
        _event(flagged_conversational),
        framing="situational_request",
        paraphrase=True,
    )
    events = {
        event.candidate_id: event
        for event in (complete_event, passed_event, flagged_event)
    }
    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )

    selected = _select_target_candidates(
        (complete, passed_conversational, flagged_conversational),
        events_by_id=events,
        config=config,
        targets={"development": 2, "validation": 0, "test": 0},
    )

    assert {candidate.candidate_id for candidate in selected} == {
        complete.candidate_id,
        passed_conversational.candidate_id,
    }
    assert {candidate.surface_form for candidate in selected} == {
        "complete_question",
        "conversational_short",
    }


def test_target_selection_keeps_retained_variants_together():
    complete = _candidate(
        "Wo ist die barrierefreie Toilette?",
        answer="Sie befindet sich im Erdgeschoss.",
        suffix="variant-complete",
    )
    conversational = _with_surface(
        _candidate(
            "accessible toilet location",
            answer="It is on the ground floor.",
            language="en",
            suffix="variant-conversational",
        ),
        "conversational_short",
    )
    disposable = _with_surface(
        _candidate(
            "weitere info",
            answer="Weitere Information.",
            suffix="variant-disposable",
        ),
        "telegraphic_keywords",
    )
    direct = _event(complete)
    situation = _with_variation(
        _event(conversational),
        framing="situational_request",
        paraphrase=True,
    )
    events = {
        event.candidate_id: event
        for event in (direct, situation, _event(disposable))
    }
    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )
    cluster = SimpleNamespace(
        decision="retain_variants",
        retained_candidate_ids=sorted(
            [complete.candidate_id, conversational.candidate_id],
        ),
    )

    selected = _select_target_candidates(
        (complete, conversational, disposable),
        events_by_id=events,
        config=config,
        targets={"development": 2, "validation": 0, "test": 0},
        clusters=(cluster,),
    )

    assert {candidate.candidate_id for candidate in selected} == {
        complete.candidate_id,
        conversational.candidate_id,
    }


def test_build_workflow_uses_post_dedup_backfill_and_rechecks(monkeypatch):
    config = QuestionConfig.from_dict(
        yaml.safe_load(
            Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
        ),
    )
    filter_result = SimpleNamespace(
        status="target_reached",
        target_counts={"development": 100, "validation": 50, "test": 50},
    )
    stats_with_deficit = DedupStatistics(
        accepted_by_split={"development": 99, "validation": 50, "test": 50},
        accepted_by_split_and_scope={
            "development": {"localized": 84, "same_page_multi_section": 15},
            "validation": {"localized": 43, "same_page_multi_section": 7},
            "test": {"localized": 43, "same_page_multi_section": 7},
        },
        accepted_by_split_and_role_profile={
            split: {"student_inclusive": target, "student_only": target}
            for split, target in (("development", 100), ("validation", 50), ("test", 50))
        },
        accepted_by_split_and_surface_form={
            "development": {
                "complete_question": 30,
                "conversational_short": 55,
                "telegraphic_keywords": 14,
            },
            "validation": {
                "complete_question": 15,
                "conversational_short": 28,
                "telegraphic_keywords": 7,
            },
            "test": {
                "complete_question": 15,
                "conversational_short": 28,
                "telegraphic_keywords": 7,
            },
        },
        accepted_by_split_and_query_framing={
            split: {"direct_lookup": target, "situational_request": target}
            for split, target in (("development", 100), ("validation", 50), ("test", 50))
        },
        accepted_natural_paraphrases_by_split={
            "development": 100,
            "validation": 50,
            "test": 50,
        },
    )
    stats_complete = DedupStatistics(
        **{
            **stats_with_deficit.__dict__,
            "accepted_by_split": {
                "development": 100,
                "validation": 50,
                "test": 50,
            },
        },
    )
    dedup_results = iter(
        (
            SimpleNamespace(
                status="deficits",
                statistics=stats_with_deficit,
                deficits_by_split={"development": 1, "validation": 0, "test": 0},
                removed_candidate_ids=("candidate:duplicate",),
                output_hash=mock_sha256("clusters-1"),
                report_hash=mock_sha256("report-1"),
                questions_hash=mock_sha256("questions-1"),
            ),
            SimpleNamespace(
                status="target_reached",
                statistics=stats_complete,
                deficits_by_split={"development": 0, "validation": 0, "test": 0},
                removed_candidate_ids=("candidate:duplicate",),
                questions_path="data/thesis/eval/questions_llm_reviewed.jsonl",
            ),
        ),
    )
    generation_calls = []

    class FakeFilterReviewJob:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            return filter_result

    class FakeDedupJob:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            return next(dedup_results)

    class FakeGenerationJob:
        def __init__(self, *args, **kwargs):
            self.config = config

        def run(self, **kwargs):
            generation_calls.append(kwargs)
            return SimpleNamespace(
                batch_manifest=SimpleNamespace(
                    generation_batch_manifest_id="generation_batch_manifest:post-dedup",
                    consumed_question_budget=1,
                ),
            )

    monkeypatch.setattr("thesis_rag.eval.question_workflow.FilterReviewJob", FakeFilterReviewJob)
    monkeypatch.setattr("thesis_rag.eval.question_workflow.QuestionDedupJob", FakeDedupJob)
    monkeypatch.setattr("thesis_rag.eval.question_workflow.QuestionGenerationJob", FakeGenerationJob)

    result = QuestionSetBuildJob().run(verify=False)

    assert result.status == "target_reached"
    assert len(result.rounds) == 2
    assert generation_calls[0]["batch_purpose"] == "post_dedup"
    assert set(generation_calls[0]["trigger_artifact_hashes"]) == {
        "question_dedup_clusters",
        "question_dedup_report",
        "questions_llm_reviewed",
    }


def test_uncertain_cross_language_pair_is_adjudicated_and_cached(tmp_path, monkeypatch):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_dedup_adjudication.md").resolve(),
    )
    config = QuestionConfig.from_dict(raw_config)
    left = _candidate(
        "Welche Einrichtungen bietet EINS?",
        answer="EINS bietet Arbeitsplätze.",
        split="validation",
        suffix="eins-de",
    )
    right = _candidate(
        "What facilities does EINS offer?",
        answer="EINS offers workspaces.",
        language="en",
        split="test",
        suffix="eins-en",
    )
    events = (_event(left), _event(right))
    reviews = CurrentQuestionReviews(
        candidates=(left, right),
        events=events,
        events_by_candidate_id={event.candidate_id: event for event in events},
        missing_candidate_ids=(),
    )
    state = SimpleNamespace(config=config, repo_root=tmp_path)
    filtered_path = tmp_path / config.artifact_paths.question_candidates_filtered
    review_path = tmp_path / config.artifact_paths.question_review_events
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.write_text("filtered\n", encoding="utf-8")
    review_path.write_text("reviews\n", encoding="utf-8")
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup.load_verified_generation_state",
        lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup.load_current_question_reviews",
        lambda _state: reviews,
    )
    embedding_calls = []
    adjudication_calls = []

    def embeddings(texts):
        embedding_calls.append(texts)
        assert len(texts) == 4
        return ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0))

    def adjudicate(request):
        adjudication_calls.append(request)
        pair_id = yaml.safe_load(request.user_prompt)["pairs"][0]["pair_id"]
        return {
            "labels": [
                {
                    "pair_id": pair_id,
                    "subject_relation": "same",
                    "requested_scope_relation": "same",
                    "answer_relation": "equivalent",
                    "explanation": "Both ask for the facilities offered by EINS.",
                },
            ],
        }

    first = QuestionDedupJob(
        repo_root=tmp_path,
        embedding_provider=embeddings,
        adjudication_provider=adjudicate,
    ).run(verify=False)

    assert len(embedding_calls) == 1
    assert len(adjudication_calls) == 1
    assert first.clusters[0].cluster_kind == "translation_equivalent"
    assert first.clusters[0].decision == "canonicalize"

    def must_not_run(*args, **kwargs):
        raise AssertionError("cached provider must not run")

    second = QuestionDedupJob(
        repo_root=tmp_path,
        embedding_provider=must_not_run,
        adjudication_provider=must_not_run,
    ).run(verify=False)

    assert second.output_hash == first.output_hash
    assert second.report_hash == first.report_hash
    assert second.questions_hash == first.questions_hash


def test_residual_adjudication_failure_blocks_canonical_outputs(tmp_path, monkeypatch):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_dedup_adjudication.md").resolve(),
    )
    raw_config["deduplication"]["adjudication"]["local_llm"][
        "max_retries"
    ] = 0
    config = QuestionConfig.from_dict(raw_config)
    left = _candidate(
        "Welche Einrichtungen bietet EINS?",
        answer="EINS bietet Arbeitsplätze.",
        split="validation",
        suffix="failure-output-de",
    )
    right = _candidate(
        "What facilities does EINS offer?",
        answer="EINS offers workspaces.",
        language="en",
        split="test",
        suffix="failure-output-en",
    )
    events = (_event(left), _event(right))
    reviews = CurrentQuestionReviews(
        candidates=(left, right),
        events=events,
        events_by_candidate_id={event.candidate_id: event for event in events},
        missing_candidate_ids=(),
    )
    state = SimpleNamespace(config=config, repo_root=tmp_path)
    filtered_path = tmp_path / config.artifact_paths.question_candidates_filtered
    review_path = tmp_path / config.artifact_paths.question_review_events
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.write_text("filtered\n", encoding="utf-8")
    review_path.write_text("reviews\n", encoding="utf-8")
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup.load_verified_generation_state",
        lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.question_dedup.load_current_question_reviews",
        lambda _state: reviews,
    )

    def embeddings(_texts):
        return ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0))

    def failing_adjudicator(_request):
        raise RuntimeError("diagnostic endpoint failure")

    with pytest.raises(
        QuestionDedupError,
        match="1 failed pair.*diagnostic endpoint failure",
    ):
        QuestionDedupJob(
            repo_root=tmp_path,
            embedding_provider=embeddings,
            adjudication_provider=failing_adjudicator,
        ).run(verify=False)

    assert not (
        tmp_path / config.artifact_paths.question_dedup_clusters
    ).exists()
    assert not (
        tmp_path / config.artifact_paths.question_dedup_report
    ).exists()
    assert not (
        tmp_path / config.artifact_paths.questions_llm_reviewed
    ).exists()


def test_uncertain_pairs_are_adjudicated_in_configured_batches(tmp_path):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["batch_size"] = 2
    config = QuestionConfig.from_dict(raw_config)
    candidates = tuple(
        _candidate(
            f"Informationsfrage {suffix}",
            answer=f"Unterschiedliche Antwort {suffix}.",
            suffix=f"batch-{suffix}",
        )
        for suffix in ("a", "b", "c")
    )

    def uncertain_pair(left, right):
        ids = tuple(sorted((left.candidate_id, right.candidate_id)))
        return _Pair(
            left_id=ids[0],
            right_id=ids[1],
            kind="uncertain",
            decision="manual_review",
            query_relation="uncertain",
            answer_relation="uncertain",
            question_similarity={"semantic_cosine": 0.8},
            answer_similarity={"semantic_cosine": 0.5},
            safeguards={"number_conflict": False},
        )

    pairs = tuple(
        uncertain_pair(candidates[left], candidates[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    batch_sizes = []

    def provider(request):
        requested = yaml.safe_load(request.user_prompt)["pairs"]
        batch_sizes.append(len(requested))
        return {
            "labels": [
                {
                    "pair_id": item["pair_id"],
                    "subject_relation": "different",
                    "requested_scope_relation": "same",
                    "answer_relation": "unrelated",
                    "explanation": "The information needs differ.",
                }
                for item in requested
            ],
        }

    result = QuestionDedupJob(
        repo_root=tmp_path,
        adjudication_provider=provider,
    )._adjudicate_pairs(
        pairs,
        candidates=candidates,
        state=SimpleNamespace(config=config, repo_root=tmp_path),
        system_prompt="Compare each pair and return structured labels.",
        dry_run=True,
    )

    assert batch_sizes == [2, 1]
    assert {pair.decision for pair in result} == {"keep_all"}


def test_failed_adjudication_batches_fall_back_to_single_pairs(tmp_path):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["batch_size"] = 4
    raw_config["deduplication"]["adjudication"]["local_llm"]["max_retries"] = 0
    config = QuestionConfig.from_dict(raw_config)
    candidates = tuple(
        _candidate(
            f"Informationsfrage {suffix}",
            answer=f"Unterschiedliche Antwort {suffix}.",
            suffix=f"fallback-{suffix}",
        )
        for suffix in ("a", "b", "c", "d", "e")
    )

    def uncertain_pair(left, right):
        ids = tuple(sorted((left.candidate_id, right.candidate_id)))
        return _Pair(
            left_id=ids[0],
            right_id=ids[1],
            kind="uncertain",
            decision="manual_review",
            query_relation="uncertain",
            answer_relation="uncertain",
            question_similarity={"semantic_cosine": 0.8},
            answer_similarity={"semantic_cosine": 0.5},
            safeguards={"number_conflict": False},
        )

    pairs = tuple(
        uncertain_pair(candidates[index], candidates[index + 1])
        for index in range(4)
    )
    batch_sizes = []

    def provider(request):
        requested = yaml.safe_load(request.user_prompt)["pairs"]
        batch_sizes.append(len(requested))
        if len(requested) > 1:
            return {"labels": []}
        return {
            "labels": [
                {
                    "pair_id": requested[0]["pair_id"],
                    "subject_relation": "different",
                    "requested_scope_relation": "same",
                    "answer_relation": "unrelated",
                    "explanation": "The subjects differ despite similar scope.",
                },
            ],
        }

    result = QuestionDedupJob(
        repo_root=tmp_path,
        adjudication_provider=provider,
    )._adjudicate_pairs(
        pairs,
        candidates=candidates,
        state=SimpleNamespace(config=config, repo_root=tmp_path),
        system_prompt="Compare each pair independently.",
        dry_run=True,
    )

    assert batch_sizes == [4, 2, 1, 1, 2, 1, 1]
    assert {pair.decision for pair in result} == {"keep_all"}
    _raise_for_technical_adjudication_failures(result)


def test_residual_single_pair_failure_blocks_canonical_dedup(tmp_path):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["local_llm"]["max_retries"] = 0
    config = QuestionConfig.from_dict(raw_config)
    left = _candidate("Frage A", answer="Antwort A.", suffix="failure-a")
    right = _candidate("Frage B", answer="Antwort B.", suffix="failure-b")
    ids = tuple(sorted((left.candidate_id, right.candidate_id)))
    pair = _Pair(
        left_id=ids[0],
        right_id=ids[1],
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.8},
        answer_similarity={"semantic_cosine": 0.5},
        safeguards={"number_conflict": False},
    )

    def provider(_request):
        raise RuntimeError("diagnostic endpoint failure")

    result = QuestionDedupJob(
        repo_root=tmp_path,
        adjudication_provider=provider,
    )._adjudicate_pairs(
        (pair,),
        candidates=(left, right),
        state=SimpleNamespace(config=config, repo_root=tmp_path),
        system_prompt="Compare each pair independently.",
        dry_run=True,
    )

    with pytest.raises(
        QuestionDedupError,
        match="1 failed pair.*diagnostic endpoint failure",
    ):
        _raise_for_technical_adjudication_failures(result)


def test_adjudication_cache_reuses_pairs_when_backfill_changes_batches(tmp_path):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"]["batch_size"] = 2
    config = QuestionConfig.from_dict(raw_config)
    candidates = tuple(
        _candidate(
            f"Informationsfrage {suffix}",
            answer=f"Unterschiedliche Antwort {suffix}.",
            suffix=f"backfill-cache-{suffix}",
        )
        for suffix in ("a", "b", "c", "d", "e")
    )

    def uncertain_pair(left, right):
        ids = tuple(sorted((left.candidate_id, right.candidate_id)))
        return _Pair(
            left_id=ids[0],
            right_id=ids[1],
            kind="uncertain",
            decision="manual_review",
            query_relation="uncertain",
            answer_relation="uncertain",
            question_similarity={"semantic_cosine": 0.8},
            answer_similarity={"semantic_cosine": 0.5},
            safeguards={"number_conflict": False},
        )

    initial_pairs = (
        uncertain_pair(candidates[0], candidates[1]),
        uncertain_pair(candidates[1], candidates[2]),
        uncertain_pair(candidates[2], candidates[3]),
    )
    added_pair = uncertain_pair(candidates[3], candidates[4])
    batch_sizes = []

    def provider(request):
        requested = yaml.safe_load(request.user_prompt)["pairs"]
        batch_sizes.append(len(requested))
        return {
            "labels": [
                {
                    "pair_id": item["pair_id"],
                    "subject_relation": "different",
                    "requested_scope_relation": "same",
                    "answer_relation": "unrelated",
                    "explanation": "The information needs differ.",
                }
                for item in requested
            ],
        }

    job = QuestionDedupJob(
        repo_root=tmp_path,
        adjudication_provider=provider,
    )
    state = SimpleNamespace(config=config, repo_root=tmp_path)
    first = job._adjudicate_pairs(
        initial_pairs,
        candidates=candidates,
        state=state,
        system_prompt="Compare each pair independently.",
        dry_run=False,
    )
    second = job._adjudicate_pairs(
        (added_pair, *initial_pairs),
        candidates=candidates,
        state=state,
        system_prompt="Compare each pair independently.",
        dry_run=False,
    )

    assert batch_sizes == [2, 1, 1]
    assert {pair.decision for pair in first} == {"keep_all"}
    assert {pair.decision for pair in second} == {"keep_all"}


def test_adjudication_pair_guard_fails_before_model_requests(tmp_path):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["deduplication"]["adjudication"][
        "maximum_adjudication_pairs"
    ] = 1
    config = QuestionConfig.from_dict(raw_config)
    candidates = tuple(
        _candidate(
            f"Informationsfrage {suffix}",
            answer=f"Antwort {suffix}.",
            suffix=f"guard-{suffix}",
        )
        for suffix in ("a", "b", "c")
    )

    def pair(left, right):
        ids = tuple(sorted((left.candidate_id, right.candidate_id)))
        return _Pair(
            left_id=ids[0],
            right_id=ids[1],
            kind="uncertain",
            decision="manual_review",
            query_relation="uncertain",
            answer_relation="uncertain",
            question_similarity={"semantic_cosine": 0.8},
            answer_similarity={"semantic_cosine": 0.5},
            safeguards={"number_conflict": False},
        )

    def must_not_run(_request):
        raise AssertionError("pair guard must run before the model provider")

    with pytest.raises(QuestionDedupError, match="proposal guard exceeded"):
        QuestionDedupJob(
            repo_root=tmp_path,
            adjudication_provider=must_not_run,
        )._adjudicate_pairs(
            (pair(candidates[0], candidates[1]), pair(candidates[1], candidates[2])),
            candidates=candidates,
            state=SimpleNamespace(config=config, repo_root=tmp_path),
            system_prompt="Compare each pair.",
            dry_run=True,
        )


def test_empty_adjudication_response_retries_with_format_correction(tmp_path):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config = QuestionConfig.from_dict(raw_config)
    left = _candidate(
        "Kontakt Studienberatung",
        answer="Die Studienberatung ist per E-Mail erreichbar.",
        suffix="retry-left",
    )
    right = _candidate(
        "Studienberatung Kontaktdaten",
        answer="Die Studienberatung ist per E-Mail erreichbar.",
        suffix="retry-right",
    )
    ids = tuple(sorted((left.candidate_id, right.candidate_id)))
    pair = _Pair(
        left_id=ids[0],
        right_id=ids[1],
        kind="uncertain",
        decision="manual_review",
        query_relation="uncertain",
        answer_relation="uncertain",
        question_similarity={"semantic_cosine": 0.8},
        answer_similarity={"semantic_cosine": 0.9},
        safeguards={"number_conflict": False},
    )
    requests = []

    def provider(request):
        requests.append(request)
        if len(requests) == 1:
            return ""
        pair_id = yaml.safe_load(request.user_prompt)["pairs"][0]["pair_id"]
        return {
            "labels": [
                {
                    "pair_id": pair_id,
                    "subject_relation": "same",
                    "requested_scope_relation": "same",
                    "answer_relation": "equivalent",
                    "explanation": "Both ask for the same contact details.",
                },
            ],
        }

    result = QuestionDedupJob(
        repo_root=tmp_path,
        adjudication_provider=provider,
    )._adjudicate_pairs(
        (pair,),
        candidates=(left, right),
        state=SimpleNamespace(config=config, repo_root=tmp_path),
        system_prompt="Compare each pair and return structured labels.",
        dry_run=True,
    )

    assert len(requests) == 2
    assert requests[0].reasoning_effort == "none"
    assert requests[0].system_prompt == (
        "Compare each pair and return structured labels."
    )
    assert "previous response was missing or invalid" in (
        requests[1].system_prompt
    )
    assert result[0].decision == "canonicalize"
