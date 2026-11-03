from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from thesis_rag.context import (
    ContextSource,
    context_prefix,
    context_relevance_identity,
    materialize_context,
)
from thesis_rag.eval.relevance import (
    _RawRanking,
    _build_candidate_pool,
    _build_metrics,
    _candidate_pool_breakdown,
    _configured_raw_rankings,
    _load_retrieval_results,
    _metric_group_records,
    _project_question,
    _ranking_views,
    _retrieval_projection_diagnostics,
    _validate_retrieval_query_coverage,
    _validate_retrieval_stream,
    _without_retrieval_traces,
    resolve_qrel_records,
)
from thesis_rag.eval.relevance_judgment_core import RelevanceJudgeResponse
from thesis_rag.eval.relevance_judgments import (
    build_active_judgment_input,
    build_judge_identity,
    select_pending_judgments,
)
from thesis_rag.eval.relevance_types import (
    PreparedJudgment,
    RelevanceError,
    RelevancePoolPreparation,
    RelevanceTarget,
)
from thesis_rag.eval.relevance_config import load_relevance_config
from thesis_rag.eval.query_selection import select_query_cohort
from thesis_rag.eval.relevance_schemas import (
    AnchorProjectionRecord,
    QrelRecord,
    RelevanceCandidateRecord,
    RelevanceJudgmentRecord,
    RetrievalTargetRef,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.eval.suite_config import (
    ContextEvaluationPolicy,
    FinalEvaluationPolicy,
    StageEvaluationPolicy,
)
from thesis_rag.interfaces import (
    EvidenceTarget,
    RetrievalResult,
    RetrievedItem,
    StageCandidate,
    StageResult,
)
from thesis_rag.schema import (
    hash_text,
    stable_id,
    validate_record_shape,
    write_jsonl,
)


CORPUS_HASH = hash_text("fixture corpus")
ARTIFACT_HASH = hash_text("fixture artifact")
TARGET_SPACE_HASH = hash_text("fixture target space")
INPUT_HASH = hash_text("fixture judgment input")


def test_canonical_relevance_judge_disables_extended_reasoning() -> None:
    config = load_relevance_config("configs/thesis/relevance.yaml")

    assert config.judge.prompt.prompt_id == "chatu_rag_relevance_v1"
    assert config.judge.generation.reasoning_effort == "none"
    assert config.judge.generation.temperature == 0.0
    assert config.judge.generation.max_tokens == 2000
    assert config.target_spaces.include_pages is True
    assert config.target_spaces.include_sections is True
    assert config.target_spaces.include_context_groups is False
    assert config.target_spaces.chunker_ids == []


def test_relevance_prompt_is_single_call_rag_specific_without_lineage_noise(
) -> None:
    prompt = Path("configs/thesis/prompts/relevance_judge.md").read_text()

    assert prompt.startswith("You are assessing retrieval results for ChaTU")
    assert "UMBRELA/TREC-oriented grades" in prompt
    assert "central relevance boundary is answer support" in prompt
    assert "Query scope:" in prompt
    assert "Topicality and context:" in prompt
    assert "Direct answer support:" in prompt
    assert "Answer coverage and sufficiency:" in prompt
    assert "Source authority and freshness are outside" in prompt
    assert "Measure how trustworthy" not in prompt
    assert '"query_scope_summary"' in prompt
    assert '"answer_support_summary"' in prompt
    assert '"missing_requested_information"' in prompt
    assert '"score": 0' in prompt
    assert "open-source UMBRELA" not in prompt
    assert not prompt.startswith("#")


def test_judge_experiment_policy_ignores_unselected_configuration() -> None:
    config = load_relevance_config("configs/thesis/relevance.yaml")
    selected = ["gemma4_31b", "llama3_1_8b"]
    baseline = config.judge_experiment_policy_payload(selected)
    unrelated = config.model_copy(
        update={
            "judging": config.judging.model_copy(
                update={
                    "judges": {
                        **config.judging.judges,
                        "llama3_3_70b": config.judging.judges[
                            "llama3_3_70b"
                        ].model_copy(update={"enabled": False}),
                    },
                },
            ),
        },
    )

    assert unrelated.judge_experiment_policy_payload(selected) == baseline
    changed_selected = config.model_copy(
        update={
            "judging": config.judging.model_copy(
                update={
                    "judges": {
                        **config.judging.judges,
                        "llama3_1_8b": config.judging.judges[
                            "llama3_1_8b"
                        ].model_copy(update={"enabled": False}),
                    },
                },
            ),
        },
    )
    assert changed_selected.judge_experiment_policy_payload(selected) != baseline


def test_relevance_config_rejects_a_disabled_default_judge(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/thesis/relevance.yaml").read_text())
    raw["judging"]["judges"][raw["judging"]["default_judge_id"]][
        "enabled"
    ] = False
    path = tmp_path / "disabled-default.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError, match="default_judge_id must be enabled"):
        load_relevance_config(path)


@pytest.mark.parametrize(
    "version",
    [
        "relevance_config_v3",
        "relevance_config_v4",
        "relevance_config_v5",
        "relevance_config_v6",
        "relevance_config_v7",
    ],
)
def test_previous_relevance_configs_are_rejected_with_migration_guidance(
    tmp_path: Path,
    version: str,
) -> None:
    config = yaml.safe_load(Path("configs/thesis/relevance.yaml").read_text())
    config["schema_version"] = version
    config["pooling"]["streams"] = ["items", "context_items"]
    path = tmp_path / "old-relevance.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="Version 8 removes"):
        load_relevance_config(path)


def _target(
    *,
    target_id: str,
    level: str,
    page_id: str,
    text: str,
    char_start: int,
    char_end: int,
    section_id: str | None = None,
    chunker_id: str | None = None,
) -> RetrievalTargetRef:
    target_space = level if level != "chunk" else f"chunk:{chunker_id}"
    return RetrievalTargetRef(
        target_id=target_id,
        target_level=level,
        target_space_id=target_space,
        page_id=page_id,
        section_id=section_id,
        chunk_id=target_id if level == "chunk" else None,
        chunker_id=chunker_id,
        text_hash=hash_text(text),
        source_path="fixture.html",
        title="Fixture",
        heading_path=["Fixture"] if level != "page" else [],
        char_start=char_start,
        char_end=char_end,
        target_artifact_hash=ARTIFACT_HASH,
    )


def test_native_chunk_target_can_remain_unaligned_to_canonical_sections() -> None:
    target = RetrievalTargetRef(
        target_id="legacy_chunk:one",
        target_level="chunk",
        target_space_id="chunk:legacy",
        page_id="page:one",
        chunk_id="legacy_chunk:one",
        chunker_id="legacy",
        text_hash=hash_text("Legacy evidence"),
        source_path="legacy.html",
        target_artifact_hash=ARTIFACT_HASH,
    )

    assert target.section_id is None
    assert target.char_start is None
    assert target.char_end is None


def test_rendered_page_target_does_not_claim_canonical_page_offsets() -> None:
    target = RetrievalTargetRef(
        target_id="rendered-page:one",
        target_level="page",
        target_space_id="context_rendered:page",
        page_id="page:one",
        text_hash=hash_text("Rendered legacy page"),
        source_path="legacy.html",
        target_artifact_hash=ARTIFACT_HASH,
    )

    assert target.char_start is None
    assert target.char_end is None

    with pytest.raises(ValueError, match="page-global offsets"):
        RetrievalTargetRef(
            target_id="page:one",
            target_level="page",
            target_space_id="page",
            page_id="page:one",
            text_hash=hash_text("Canonical page"),
            source_path="canonical.html",
            target_artifact_hash=ARTIFACT_HASH,
        )


def test_retrieval_target_rejects_a_partial_offset_pair() -> None:
    with pytest.raises(ValueError, match="both present or absent"):
        RetrievalTargetRef(
            target_id="legacy_chunk:one",
            target_level="chunk",
            target_space_id="chunk:legacy",
            page_id="page:one",
            chunk_id="legacy_chunk:one",
            chunker_id="legacy",
            text_hash=hash_text("Legacy evidence"),
            source_path="legacy.html",
            char_start=0,
            target_artifact_hash=ARTIFACT_HASH,
        )


def _two_anchor_question() -> tuple[QuestionRecord, str]:
    page_text = "Intro. First useful fact. Bridge. Second useful fact. End."
    first_start = page_text.index("First")
    first_end = first_start + len("First useful fact.")
    second_start = page_text.index("Second")
    second_end = second_start + len("Second useful fact.")
    spans = [
        EvidenceTarget(
            target_level="span",
            target_id="span:first",
            page_id="page:one",
            section_id="section:first",
            char_start=first_start,
            char_end=first_end,
            text_hash=hash_text(page_text[first_start:first_end]),
            source_path="fixture.html",
        ),
        EvidenceTarget(
            target_level="span",
            target_id="span:second",
            page_id="page:one",
            section_id="section:second",
            char_start=second_start,
            char_end=second_end,
            text_hash=hash_text(page_text[second_start:second_end]),
            source_path="fixture.html",
        ),
    ]
    assessments = [
        {
            "target": span.to_dict(),
            "grade": 2,
            "status": "accepted",
            "label": "partially_useful",
            "explanation": "Fixture.",
            "supported_fact_refs": [f"fact:{index}"],
            "propagation_rule": None,
        }
        for index, span in enumerate(spans, start=1)
    ]
    return (
        QuestionRecord(
            query_id="question:fixture",
            text="What are both useful facts?",
            language="en",
            source_targets=spans,
            acceptable_targets=spans,
            metadata={
                "candidate_id": "question_candidate:fixture",
                "review_event_id": "question_review_event:fixture",
                "split": "development",
                "surface_form": "complete_question",
                "reference_answer": "Both facts.",
                "required_facts": ["First useful fact.", "Second useful fact."],
                "temporal_classification": "snapshot_qualified",
                "anchor_assessments": assessments,
                "sufficient_evidence_sets": [
                    {"target_ids": ["span:first", "span:second"]},
                ],
            },
        ),
        page_text,
    )


def test_projection_uses_containment_and_complete_sufficient_sets() -> None:
    question, page_text = _two_anchor_question()
    first = question.source_targets[0]
    second = question.source_targets[1]
    page = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    section = _target(
        target_id="section:all",
        level="section",
        page_id="page:one",
        section_id="section:all",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    first_chunk_text = page_text[first.char_start : first.char_end]
    first_chunk = _target(
        target_id="chunk:first",
        level="chunk",
        page_id="page:one",
        section_id="section:first",
        chunker_id="normalized",
        text=first_chunk_text,
        char_start=first.char_start,
        char_end=first.char_end,
    )
    cross_section_text = page_text[first.char_start : second.char_end]
    cross_section_chunk = _target(
        target_id="chunk:cross-section",
        level="chunk",
        page_id="page:one",
        section_id="section:first",
        chunker_id="normalized",
        text=cross_section_text,
        char_start=first.char_start,
        char_end=second.char_end,
    )
    targets = [page, section, first_chunk, cross_section_chunk]
    state = SimpleNamespace(
        page_texts={"page:one": page_text},
        config=SimpleNamespace(
            projection=SimpleNamespace(rule_version="projection_fixture_v1"),
        ),
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
        target_space_manifest_hashes={
            target.target_space_id: TARGET_SPACE_HASH for target in targets
        },
    )

    projections = _project_question(
        question,
        state=state,
        targets_by_page={
            "page:one": tuple(RelevanceTarget(target, "") for target in targets),
        },
    )
    by_target = {projection.target.target_id: projection for projection in projections}

    assert by_target["chunk:first"].anchor_score_floor == 2
    assert by_target["chunk:first"].fully_contained_sufficient_set_ids == []
    assert by_target["section:all"].anchor_score_floor == 3
    assert by_target["page:one"].anchor_score_floor == 3
    assert by_target["chunk:cross-section"].anchor_score_floor == 3
    assert by_target["chunk:cross-section"].contained_anchor_ids == [
        "span:first",
        "span:second",
    ]
    assert "contains_complete_sufficient_evidence_set" in (
        by_target["chunk:cross-section"].derivation_rules
    )


def test_partial_anchor_overlap_enters_pool_without_relevance_floor() -> None:
    question, page_text = _two_anchor_question()
    first = question.source_targets[0]
    overlap_start = int(first.char_start) + 3
    overlap_end = int(first.char_end) - 2
    target = _target(
        target_id="chunk:partial",
        level="chunk",
        page_id="page:one",
        section_id="section:first",
        chunker_id="normalized",
        text=page_text[overlap_start:overlap_end],
        char_start=overlap_start,
        char_end=overlap_end,
    )
    state = SimpleNamespace(
        page_texts={"page:one": page_text},
        config=SimpleNamespace(
            projection=SimpleNamespace(rule_version="projection_fixture_v3"),
        ),
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
        target_space_manifest_hashes={target.target_space_id: TARGET_SPACE_HASH},
    )

    projections = _project_question(
        question,
        state=state,
        targets_by_page={
            "page:one": (RelevanceTarget(target, target.text_hash),),
        },
    )

    assert len(projections) == 1
    assert projections[0].projection_kind == "partial_overlap"
    assert projections[0].anchor_score_floor is None
    assert projections[0].contained_anchor_ids == []
    assert projections[0].overlapping_anchor_ids == ["span:first"]


def test_projection_rejects_ambiguous_cross_page_anchor_input() -> None:
    question, page_text = _two_anchor_question()
    assessments = list(question.metadata["anchor_assessments"])
    second = dict(assessments[1])
    second_target = dict(second["target"])
    second_target["page_id"] = "page:two"
    second["target"] = second_target
    assessments[1] = second
    broken = QuestionRecord.from_dict(
        {
            **question.to_dict(),
            "metadata": {**question.metadata, "anchor_assessments": assessments},
        },
    )
    state = SimpleNamespace(
        page_texts={"page:one": page_text, "page:two": page_text},
    )

    with pytest.raises(ValueError, match="multiple source pages"):
        _project_question(broken, state=state, targets_by_page={})


def test_context_group_anchor_floor_uses_member_containment_not_fake_span() -> None:
    question, page_text = _two_anchor_question()
    first, second = question.source_targets
    group_text = (
        page_text[first.char_start : first.char_end]
        + "\n\n"
        + page_text[second.char_start : second.char_end]
    )
    group = RetrievalTargetRef(
        target_id="group:both",
        target_level="context_group",
        target_space_id="context_group",
        page_id="page:one",
        context_group_id="group:both",
        text_hash=hash_text(group_text),
        source_path="fixture.html",
        member_section_ids=["section:first", "section:second"],
        focus_section_ids=["section:first", "section:second"],
        target_artifact_hash=ARTIFACT_HASH,
    )
    state = SimpleNamespace(
        page_texts={"page:one": page_text},
        config=SimpleNamespace(
            projection=SimpleNamespace(rule_version="projection_fixture_v1"),
        ),
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
        target_space_manifest_hashes={"context_group": TARGET_SPACE_HASH},
    )

    projections = _project_question(
        question,
        state=state,
        targets_by_page={
            "page:one": (
                RelevanceTarget(
                    group,
                    group_text,
                    containment_spans=(
                        (int(first.char_start), int(first.char_end)),
                        (int(second.char_start), int(second.char_end)),
                    ),
                ),
            ),
        },
    )

    assert len(projections) == 1
    assert projections[0].target.char_start is None
    assert projections[0].target.char_end is None
    assert projections[0].anchor_score_floor == 3
    assert projections[0].contained_anchor_ids == ["span:first", "span:second"]


def _candidate(
    question: QuestionRecord,
    target: RetrievalTargetRef,
) -> RelevanceCandidateRecord:
    query_hash = hash_text(question.text)
    return RelevanceCandidateRecord(
        pool_record_id=stable_id(
            "relevance_pool",
            question.query_id,
            target.target_space_id,
            target.target_id,
            CORPUS_HASH,
        ),
        query_id=question.query_id,
        query_text_hash=query_hash,
        target=target,
        origins=["anchor_projection"],
        anchor_projection_ids=["anchor_projection:fixture"],
        information_need_cluster_ids=[],
        source_family_ids=[],
        related_candidate_ids=[],
        seen_in=[],
        judgment_priority="source_anchor_backfill",
        corpus_hash=CORPUS_HASH,
    )


def _projection(
    question: QuestionRecord,
    target: RetrievalTargetRef,
    floor: int,
) -> AnchorProjectionRecord:
    query_hash = hash_text(question.text)
    projection_id = AnchorProjectionRecord.canonical_v3_id(
        query_id=question.query_id,
        query_text_hash=query_hash,
        target=target,
        candidate_id="question_candidate:fixture",
        review_event_id="question_review_event:fixture",
        projection_kind="full_containment",
        anchor_score_floor=floor,
        contained_anchor_ids=["span:first"],
        overlapping_anchor_ids=[],
        fully_contained_sufficient_set_ids=[],
        supported_fact_refs=["fact:1"],
        derivation_rules=["contains_grade_2_anchor"],
        corpus_hash=CORPUS_HASH,
        target_space_manifest_hash=TARGET_SPACE_HASH,
        projection_rule_version="projection_fixture_v1",
    )
    return AnchorProjectionRecord(
        projection_id=projection_id,
        query_id=question.query_id,
        query_text_hash=query_hash,
        candidate_id="question_candidate:fixture",
        review_event_id="question_review_event:fixture",
        target=target,
        anchor_score_floor=floor,
        contained_anchor_ids=["span:first"],
        fully_contained_sufficient_set_ids=[],
        supported_fact_refs=["fact:1"],
        derivation_rules=["contains_grade_2_anchor"],
        corpus_hash=CORPUS_HASH,
        target_space_manifest_hash=TARGET_SPACE_HASH,
        projection_rule_version="projection_fixture_v1",
    )


def test_legacy_anchor_projection_records_remain_verifiable() -> None:
    question, page_text = _two_anchor_question()
    target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    record = _projection(question, target, 2).to_dict()
    record.update(
        {
            "projection_id": stable_id(
                "anchor_projection",
                question.query_id,
                hash_text(question.text),
                target.target_space_id,
                target.target_id,
                target.text_hash,
                ["span:first"],
                2,
                "projection_fixture_v1",
            ),
            "schema_version": "anchor_projection_v2",
        },
    )

    validate_record_shape(record, "anchor_projection_v2")


def test_v3_anchor_projection_identity_binds_overlap_evidence() -> None:
    question, page_text = _two_anchor_question()
    target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    payload = _projection(question, target, 2).to_dict()
    payload["overlapping_anchor_ids"] = ["span:second"]

    with pytest.raises(ValueError, match="projection_id does not match"):
        AnchorProjectionRecord.from_dict(payload)


def test_candidate_identity_is_query_target_space_target_and_corpus() -> None:
    question, page_text = _two_anchor_question()
    target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    candidate = _candidate(question, target)

    assert candidate.pool_record_id == stable_id(
        "relevance_pool",
        question.query_id,
        "page",
        "page:one",
        CORPUS_HASH,
    )


def _judgment(
    question: QuestionRecord,
    target: RetrievalTargetRef,
    *,
    score: int,
    kind: str,
) -> RelevanceJudgmentRecord:
    query_hash = hash_text(question.text)
    if kind == "llm_judge":
        evidence_id = stable_id("fixture_deployment_evidence", INPUT_HASH)
        provenance = {
            "kind": kind,
            "semantic_judge_id": "judge:fixture",
            "deployment_evidence_id": evidence_id,
            "assurance_class": "strict_digest",
            "reported_digest": hash_text("fixture-model"),
            "response_hash": hash_text(f"fixture-response:{score}"),
        }
        identity_part = evidence_id
    else:
        provenance = {"kind": kind}
        identity_part = "manual"
    judgment_id = stable_id(
        "relevance_judgment",
        question.query_id,
        query_hash,
        target.target_space_id,
        target.target_id,
        target.text_hash,
        "relevance_v1",
        INPUT_HASH,
        identity_part,
    )
    return RelevanceJudgmentRecord(
        judgment_id=judgment_id,
        query_id=question.query_id,
        query_text_hash=query_hash,
        target=target,
        query_scope_summary="Fixture information need.",
        answer_support_summary="Fixture target support.",
        missing_requested_information=(
            [] if score == 3 else ["Fixture missing information."]
        ),
        score=score,
        label={
            0: "irrelevant",
            1: "topical_only",
            2: "partially_useful",
            3: "sufficient",
        }[score],
        guideline_version="relevance_v1",
        judgment_input_hash=INPUT_HASH,
        provenance=provenance,
        corpus_hash=CORPUS_HASH,
    )


def test_qrel_resolution_applies_anchor_floor_and_keeps_unknown_unjudged() -> None:
    question, page_text = _two_anchor_question()
    anchored_target = _target(
        target_id="page:anchored",
        level="page",
        page_id="page:anchored",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    projection = _projection(question, anchored_target, 3)
    anchored = RelevanceCandidateRecord.from_dict(
        {
            **_candidate(question, anchored_target).to_dict(),
            "anchor_projection_ids": [projection.projection_id],
        },
    )
    unknown_target = _target(
        target_id="page:unknown",
        level="page",
        page_id="page:unknown",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    unknown = RelevanceCandidateRecord.from_dict(
        {
            **_candidate(question, unknown_target).to_dict(),
            "origins": ["information_need_cluster"],
            "anchor_projection_ids": [],
            "information_need_cluster_ids": ["cluster:unknown"],
            "judgment_priority": "other_non_retrieval",
        },
    )
    prepared = RelevancePoolPreparation(
        state=SimpleNamespace(
            config=SimpleNamespace(relevance_guideline_version="relevance_v1"),
        ),
        projections=(projection,),
        candidates=(anchored, unknown),
        cost_report={},
    )

    resolutions = resolve_qrel_records(
        prepared,
        (_judgment(question, anchored_target, score=1, kind="llm_judge"),),
    )
    by_target = {item.target.target_id: item for item in resolutions}

    assert by_target["page:anchored"].anchor_score_floor == 3
    assert by_target["page:anchored"].llm_score == 1
    assert by_target["page:anchored"].effective_score == 3
    assert by_target["page:anchored"].resolution == "anchor_floor_applied"
    assert by_target["page:anchored"].floor_violation_delta == 2
    assert by_target["page:unknown"].effective_score is None
    assert by_target["page:unknown"].resolution == "unjudged"
def test_candidate_breakdown_keeps_overlapping_origins_explicit() -> None:
    question, page_text = _two_anchor_question()
    target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    projection = _projection(question, target, 3)
    candidate = _candidate(question, target)
    candidate = RelevanceCandidateRecord.from_dict(
        {
            **candidate.to_dict(),
            "origins": ["anchor_projection", "retrieval_hit"],
            "anchor_projection_ids": [projection.projection_id],
            "seen_in": [
                {
                    "run_id": "run:fixture",
                    "system_id": "system:fixture",
                    "config_hash": hash_text("retriever"),
                    "stream": "items",
                    "stream_kind": "final",
                    "stage_id": None,
                    "variant_id": None,
                    "rank": 1,
                    "source_rank": 1,
                    "score": 1.0,
                    "score_name": "fixture_score",
                },
            ],
            "judgment_priority": "final",
        },
    )

    breakdown = _candidate_pool_breakdown([candidate], [projection])

    assert breakdown["by_origin"] == {
        "anchor_projection": 1,
        "retrieval_hit": 1,
    }
    assert breakdown["by_origin_combination"] == {
        "anchor_projection+retrieval_hit": 1,
    }
    assert breakdown["by_anchor_floor"] == {"3": 1}
    assert breakdown["retrieval_observations"] == 1


def test_limited_judgment_selection_is_deterministic_and_stratified(
    tmp_path: Path,
) -> None:
    base_question, page_text = _two_anchor_question()
    prepared: list[PreparedJudgment] = []
    projections: list[AnchorProjectionRecord] = []
    spaces = ("page", "section", "chunk")
    for floor in (None, 2, 3):
        for space_index, space in enumerate(spaces):
            for language in ("de", "en"):
                suffix = f"{floor}-{space}-{language}"
                question = QuestionRecord.from_dict(
                    {
                        **base_question.to_dict(),
                        "query_id": f"question:{suffix}",
                        "text": f"Fixture query {suffix}",
                        "language": language,
                    },
                )
                target = _target(
                    target_id=f"{space}:{suffix}",
                    level=space,
                    page_id=f"page:{suffix}",
                    section_id=(
                        f"section:{suffix}" if space in {"section", "chunk"} else None
                    ),
                    chunker_id="fixture" if space == "chunk" else None,
                    text=page_text,
                    char_start=0,
                    char_end=len(page_text),
                )
                candidate = _candidate(question, target)
                if floor is None:
                    candidate = RelevanceCandidateRecord.from_dict(
                        {
                            **candidate.to_dict(),
                            "origins": ["information_need_cluster"],
                            "anchor_projection_ids": [],
                            "information_need_cluster_ids": [
                                f"information_need_cluster:{suffix}",
                            ],
                            "judgment_priority": "other_non_retrieval",
                        },
                    )
                else:
                    projection = _projection(question, target, floor)
                    projections.append(projection)
                    candidate = RelevanceCandidateRecord.from_dict(
                        {
                            **candidate.to_dict(),
                            "anchor_projection_ids": [projection.projection_id],
                        },
                    )
                prepared.append(
                    PreparedJudgment(
                        candidate=candidate,
                        question=question,
                        target=RelevanceTarget(target, page_text),
                        active_input={},
                        input_hash=hash_text(suffix),
                        cache_path=tmp_path / f"{space_index}-{suffix}.json",
                        reused=None,
                    ),
                )

    selected, diagnostics = select_pending_judgments(
        prepared,
        projections,
        limit=6,
        seed=14731,
    )
    repeated, repeated_diagnostics = select_pending_judgments(
        list(reversed(prepared)),
        projections,
        limit=6,
        seed=14731,
    )

    assert [item.candidate.pool_record_id for item in selected] == [
        item.candidate.pool_record_id for item in repeated
    ]
    assert diagnostics == repeated_diagnostics
    assert diagnostics["strategy"] == "strict_phase_stratified_v2"
    assert diagnostics["selected_by_anchor_floor"] == {
        "2": 3,
        "3": 3,
    }
    assert diagnostics["selected_by_target_space"] == {
        "chunk:fixture": 2,
        "page": 2,
        "section": 2,
    }
    assert diagnostics["selected_by_language"] == {"de": 3, "en": 3}


def test_judgment_selection_exhausts_final_candidates_before_intermediate() -> None:
    question, page_text = _two_anchor_question()
    prepared: list[PreparedJudgment] = []
    for index, priority in enumerate(
        ("intermediate_only", "final", "final"),
        start=1,
    ):
        target = _target(
            target_id=f"page:{index}",
            level="page",
            page_id=f"page:{index}",
            text=page_text,
            char_start=0,
            char_end=len(page_text),
        )
        stream_kind = "final" if priority == "final" else "intermediate"
        stream = "items" if priority == "final" else "stage:vector_search"
        candidate = RelevanceCandidateRecord.from_dict(
            {
                **_candidate(question, target).to_dict(),
                "origins": ["retrieval_hit"],
                "anchor_projection_ids": [],
                "seen_in": [
                    {
                        "run_id": "run:fixture",
                        "system_id": "system:fixture",
                        "config_hash": hash_text("fixture config"),
                        "stream": stream,
                        "stream_kind": stream_kind,
                        "stage_id": (
                            "vector_search" if stream_kind == "intermediate" else None
                        ),
                        "variant_id": None,
                        "rank": index,
                        "source_rank": index,
                        "score": 1.0,
                        "score_name": "fixture_score",
                    },
                ],
                "judgment_priority": priority,
            },
        )
        prepared.append(
            PreparedJudgment(
                candidate=candidate,
                question=question,
                target=RelevanceTarget(target, page_text),
                active_input={},
                input_hash=hash_text(f"input:{index}"),
                cache_path=Path(f"/tmp/fixture-{index}.json"),
                reused=None,
            ),
        )

    selected, diagnostics = select_pending_judgments(
        prepared,
        (),
        limit=2,
        seed=14731,
    )
    all_selected, all_diagnostics = select_pending_judgments(
        prepared,
        (),
        limit=None,
        seed=14731,
    )

    assert [item.candidate.judgment_priority for item in selected] == [
        "final",
        "final",
    ]
    assert diagnostics["selected_by_judgment_priority"] == {"final": 2}
    assert [item.candidate.judgment_priority for item in all_selected] == [
        "final",
        "final",
    ]
    assert all_diagnostics["strategy"] == "priority_phase_complete_v1"


def test_judgment_selection_uses_strict_four_phase_barriers() -> None:
    question, page_text = _two_anchor_question()
    items: list[PreparedJudgment] = []
    projections: list[AnchorProjectionRecord] = []
    for index, priority in enumerate(
        (
            "other_non_retrieval",
            "intermediate_only",
            "source_anchor_backfill",
            "final",
        ),
    ):
        target = _target(
            target_id=f"page:phase-{index}",
            level="page",
            page_id=f"page:phase-{index}",
            text=page_text,
            char_start=0,
            char_end=len(page_text),
        )
        base = _candidate(question, target).to_dict()
        if priority == "final":
            base.update(
                {
                    "origins": ["retrieval_hit"],
                    "anchor_projection_ids": [],
                    "seen_in": [
                        {
                            "run_id": "run:fixture",
                            "system_id": "system:fixture",
                            "config_hash": hash_text("config"),
                            "stream": "items",
                            "stream_kind": "final",
                            "stage_id": None,
                            "variant_id": None,
                            "rank": 1,
                            "source_rank": 1,
                            "score": 1.0,
                            "score_name": "fixture",
                        },
                    ],
                },
            )
        elif priority == "intermediate_only":
            base.update(
                {
                    "origins": ["retrieval_hit"],
                    "anchor_projection_ids": [],
                    "seen_in": [
                        {
                            "run_id": "run:fixture",
                            "system_id": "system:fixture",
                            "config_hash": hash_text("config"),
                            "stream": "stage:dense",
                            "stream_kind": "intermediate",
                            "stage_id": "dense",
                            "variant_id": None,
                            "rank": 1,
                            "source_rank": 1,
                            "score": 1.0,
                            "score_name": "fixture",
                        },
                    ],
                },
            )
        elif priority == "other_non_retrieval":
            base.update(
                {
                    "origins": ["information_need_cluster"],
                    "anchor_projection_ids": [],
                    "information_need_cluster_ids": ["cluster:fixture"],
                },
            )
        elif priority == "source_anchor_backfill":
            projection = _projection(question, target, 3)
            projections.append(projection)
            base["anchor_projection_ids"] = [projection.projection_id]
        base["judgment_priority"] = priority
        candidate = RelevanceCandidateRecord.from_dict(base)
        items.append(
            PreparedJudgment(
                candidate=candidate,
                question=question,
                target=RelevanceTarget(target, page_text),
                active_input={},
                input_hash=hash_text(f"phase:{index}"),
                cache_path=Path(f"/tmp/phase-{index}.json"),
                reused=None,
            ),
        )

    pending = items
    expected = [
        "final",
        "source_anchor_backfill",
        "intermediate_only",
        "other_non_retrieval",
    ]
    for phase in expected:
        selected, diagnostics = select_pending_judgments(
            pending,
            projections,
            limit=99,
            seed=17,
        )
        assert {item.candidate.judgment_priority for item in selected} == {phase}
        assert diagnostics["active_priority_phase"] == phase
        selected_ids = {item.candidate.pool_record_id for item in selected}
        pending = [
            item
            for item in pending
            if item.candidate.pool_record_id not in selected_ids
        ]


def test_query_limit_is_split_scoped_deterministic_and_prefix_stable() -> None:
    base, _page_text = _two_anchor_question()
    questions: list[QuestionRecord] = []
    for index in range(36):
        split = ("development", "validation", "test")[index % 3]
        language = "de" if index % 4 else "en"
        surface_form = (
            "conversational_short" if index % 5 else "telegraphic_keywords"
        )
        questions.append(
            QuestionRecord.from_dict(
                {
                    **base.to_dict(),
                    "query_id": f"question:{index:02d}",
                    "text": f"Fixture query {index}",
                    "language": language,
                    "metadata": {
                        **base.metadata,
                        "split": split,
                        "surface_form": surface_form,
                    },
                },
            ),
        )

    first, diagnostics = select_query_cohort(
        questions,
        splits=["development"],
        limit=5,
        seed=19,
    )
    larger, larger_diagnostics = select_query_cohort(
        list(reversed(questions)),
        splits=["development"],
        limit=10,
        seed=19,
    )

    assert [question.query_id for question in first] == [
        question.query_id for question in larger[:5]
    ]
    assert all(question.metadata["split"] == "development" for question in larger)
    assert diagnostics["strategy"] == "stratified_query_prefix_v1"
    assert diagnostics["eligible_queries"] == 12
    assert diagnostics["selected_queries"] == 5
    assert larger_diagnostics["selected_queries"] == 10
    with pytest.raises(ValueError, match="exceeds the 12 eligible queries"):
        select_query_cohort(
            questions,
            splits=["development"],
            limit=13,
            seed=19,
        )


def test_metric_cutoffs_cannot_exceed_the_judged_pool_depth() -> None:
    config = load_relevance_config("configs/thesis/relevance.yaml")
    raw = config.to_dict()
    raw["metrics"]["cutoffs"] = [1, 21]

    with pytest.raises(ValueError, match="must not exceed"):
        type(config).from_dict(raw)


def test_retrieval_runs_must_cover_the_same_selected_query_cohort() -> None:
    partial = [
        SimpleNamespace(
            run_id="run:one",
            system_id="system:one",
            query_id="question:one",
        ),
    ]

    with pytest.raises(RelevanceError, match="fixed selected query cohort"):
        _validate_retrieval_query_coverage(
            partial,
            ["question:one", "question:two"],
        )

    diagnostics = _validate_retrieval_query_coverage(
        [
            *partial,
            SimpleNamespace(
                run_id="run:one",
                system_id="system:one",
                query_id="question:two",
            ),
        ],
        ["question:one", "question:two"],
    )
    assert diagnostics["coverage_complete"] is True
    assert diagnostics["common_query_count"] == 2
    assert diagnostics["intended_query_ids_hash"] == diagnostics[
        "common_query_ids_hash"
    ]


def test_retrieval_run_identity_cannot_mix_configurations(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    records = [
        RetrievalResult(
            query_id=f"question:{index}",
            system_id="system:one",
            run_id="run:one",
            items=[],
            context_items=[],
            direct_context_items=[],
            enabled_stages={},
            stage_traces=[],
            debug={},
            timings_ms={},
            config_hash=hash_text(f"config:{index}"),
            corpus_hash=CORPUS_HASH,
        ).to_dict()
        for index in range(2)
    ]
    write_jsonl(path, records)

    with pytest.raises(RelevanceError, match="mix configuration hashes"):
        _load_retrieval_results([path])

    records[0]["stage_traces"] = ["invalid trace"]
    path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage_traces"):
        _load_retrieval_results([path])

    records[0]["stage_traces"] = []
    del records[0]["schema_version"]
    path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        _load_retrieval_results([path])


def test_retrieval_stream_requires_unique_targets_and_contiguous_ranks() -> None:
    item = SimpleNamespace(rank=1, target_id="page:one")
    duplicate = SimpleNamespace(rank=2, target_id="page:one")
    gap = SimpleNamespace(rank=3, target_id="page:two")

    with pytest.raises(RelevanceError, match="duplicate target IDs"):
        _validate_retrieval_stream([item, duplicate], "fixture")
    with pytest.raises(RelevanceError, match="contiguous from 1"):
        _validate_retrieval_stream([item, gap], "fixture")


def test_stage_ranking_projects_to_separate_native_section_and_page_views() -> None:
    config = load_relevance_config("configs/thesis/relevance.yaml")
    config = config.model_copy(
        update={
            "pooling": config.pooling.model_copy(
                update={"retrieval_top_k": 5},
            ),
        },
    )
    page_one = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text="Page one",
        char_start=0,
        char_end=8,
    )
    page_two = _target(
        target_id="page:two",
        level="page",
        page_id="page:two",
        text="Page two",
        char_start=0,
        char_end=8,
    )
    section_one = _target(
        target_id="section:one",
        level="section",
        page_id="page:one",
        section_id="section:one",
        text="Section one",
        char_start=0,
        char_end=11,
    )
    section_two = _target(
        target_id="section:two",
        level="section",
        page_id="page:two",
        section_id="section:two",
        text="Section two",
        char_start=0,
        char_end=11,
    )
    chunks = [
        _target(
            target_id=f"chunk:{index}",
            level="chunk",
            page_id="page:one" if index < 3 else "page:two",
            section_id="section:one" if index < 3 else "section:two",
            chunker_id="fixture",
            text=f"Chunk {index}",
            char_start=index,
            char_end=index + 7,
        )
        for index in range(1, 4)
    ]
    stage_items = tuple(
        StageCandidate(
            rank=index,
            target_id=chunk.target_id,
            target_level="chunk",
            chunk_id=chunk.target_id,
            chunker_id="fixture",
            section_id=chunk.section_id,
            page_id=chunk.page_id,
            score=float(index),
            score_name="fixture_score",
            metadata={"overlapping_section_ids": [chunk.section_id]},
        )
        for index, chunk in enumerate(chunks, start=1)
    )
    ranking = _RawRanking(
        stream="stage:dense",
        stream_kind="intermediate",
        stage_id="dense",
        variant_id=None,
        items=stage_items,
        evaluable_up_to_k=3,
        ranking_complete=False,
        target_levels=("chunk", "page", "section"),
    )
    targets = {
        (target.target_space_id, target.target_id): RelevanceTarget(target, "")
        for target in [page_one, page_two, section_one, section_two, *chunks]
    }

    views = _ranking_views(
        ranking,
        config=config,
        selected_chunkers={"fixture": {}},
        targets=targets,
    )
    by_space = {view.target_space_id: view for view in views}

    assert list(by_space) == ["chunk:fixture", "page", "section"]
    assert [item.target_id for item in by_space["chunk:fixture"].items] == [
        "chunk:1",
        "chunk:2",
        "chunk:3",
    ]
    assert [item.target_id for item in by_space["page"].items] == [
        "page:one",
        "page:two",
    ]
    assert [item.source_rank for item in by_space["page"].items] == [1, 3]
    assert [item.target_id for item in by_space["section"].items] == [
        "section:one",
        "section:two",
    ]
    assert by_space["chunk:fixture"].evaluable_up_to_k == 3
    assert by_space["page"].evaluable_up_to_k == 2
    assert by_space["section"].evaluable_up_to_k == 2

    ambiguous_ranking = _RawRanking(
        stream=ranking.stream,
        stream_kind=ranking.stream_kind,
        stage_id=ranking.stage_id,
        variant_id=ranking.variant_id,
        items=tuple(
            item.model_copy(
                update={
                    "metadata": {
                        "overlapping_section_ids": [
                            str(item.section_id),
                            "section:other",
                        ],
                    },
                },
            )
            for item in stage_items
        ),
        evaluable_up_to_k=3,
        ranking_complete=False,
        target_levels=ranking.target_levels,
    )
    ambiguous_views = _ranking_views(
        ambiguous_ranking,
        config=config,
        selected_chunkers={"fixture": {}},
        targets=targets,
    )
    assert [view.target_space_id for view in ambiguous_views] == [
        "chunk:fixture",
        "page",
    ]

    legacy_compatible_views = _ranking_views(
        ranking,
        config=config,
        selected_chunkers={},
        targets={
            key: value
            for key, value in targets.items()
            if key[0] in {"page", "section"}
        },
    )
    assert [view.target_space_id for view in legacy_compatible_views] == [
        "page",
        "section",
    ]


def test_context_group_is_explicit_and_projects_only_upward_to_page() -> None:
    group_text = "First section\n\nSecond section"
    group = RetrievalTargetRef(
        target_id="group:one",
        target_level="context_group",
        target_space_id="context_group",
        page_id="page:one",
        context_group_id="group:one",
        text_hash=hash_text(group_text),
        source_path="fixture.html",
        member_section_ids=["section:one", "section:two"],
        focus_section_ids=["section:two"],
        target_artifact_hash=ARTIFACT_HASH,
    )
    page = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=group_text,
        char_start=0,
        char_end=len(group_text),
    )
    ranking = _RawRanking(
        stream="context_items",
        stream_kind="final",
        stage_id=None,
        variant_id=None,
        items=(
            RetrievedItem(
                rank=1,
                target_id=group.target_id,
                target_level="context_group",
                context_group_id=group.context_group_id,
                chunk_id=None,
                chunker_id=None,
                section_id=None,
                overlapping_section_ids=[],
                page_id=group.page_id,
                source_path=group.source_path,
                url=None,
                title=None,
                heading_path=[],
                text=group_text,
                text_hash=group.text_hash,
                score=1.0,
                score_name="fixture",
            ),
        ),
        evaluable_up_to_k=20,
        ranking_complete=True,
        target_levels=("context_group", "section", "page"),
    )
    config = load_relevance_config("configs/thesis/relevance.yaml")
    config = config.model_copy(
        update={
            "target_spaces": config.target_spaces.model_copy(
                update={"include_context_groups": True},
            ),
        },
    )
    views = _ranking_views(
        ranking,
        config=config,
        selected_chunkers={},
        targets={
            ("context_group", group.target_id): RelevanceTarget(group, group_text),
            ("page", page.target_id): RelevanceTarget(page, group_text),
        },
    )

    assert [view.target_space_id for view in views] == ["context_group", "page"]
    assert all(view.evaluable_up_to_k == 20 for view in views)


@pytest.mark.parametrize("compact", [False, True])
def test_suite_selection_is_exact_for_stream_stage_variant_and_purpose(compact) -> None:
    item = RetrievedItem(
        rank=1,
        target_id="page:one",
        target_level="page",
        chunk_id=None,
        chunker_id=None,
        section_id=None,
        overlapping_section_ids=[],
        page_id="page:one",
        source_path="fixture.html",
        url=None,
        title=None,
        heading_path=[],
        text="Evidence",
        text_hash=hash_text("Evidence"),
        score=1.0,
        score_name="fixture",
        metadata={"candidate_provenance": {"scores": [0.5]}, "mapping_status": "mapped"},
    )

    def stage(variant: str, index: int) -> StageResult:
        return StageResult(
            stage_id="dense",
            stage_label=f"Dense {variant}",
            stage_index=index,
            variant_id=variant,
            input_count=1,
            output_count=1,
            requested_depth=20,
            evaluable_up_to_k=20,
            ranking_complete=True,
            items=[
                StageCandidate(
                    rank=1,
                    target_id="page:one",
                    target_level="page",
                    chunk_id=None,
                    chunker_id=None,
                    section_id=None,
                    page_id="page:one",
                    score=1.0,
                    score_name="fixture",
                ),
            ],
        )

    context_items, context_serialization = materialize_context(
        [
            ContextSource(
                item=item,
                origin="retrieval",
                source_item_ids=(item.target_id,),
                source_item_ranks=(item.rank,),
            ),
        ],
        status="materialized",
    )
    result = RetrievalResult(
        query_id="question:one",
        system_id="system:one",
        run_id="run:one",
        items=[item],
        context_items=context_items,
        direct_context_items=[],
        context_serialization=context_serialization,
        enabled_stages={},
        stage_traces=[],
        stage_results=[stage("pool", 0), stage("metrics", 1), stage("ignored", 2)],
        debug={},
        timings_ms={},
        config_hash=hash_text("config"),
        corpus_hash=CORPUS_HASH,
    )
    experiment = SimpleNamespace(
        experiment_id="fixture_experiment",
        top_k=20,
        final_evaluation=FinalEvaluationPolicy(
            streams=["items"],
            target_levels=["page"],
            include_in_pool=True,
            compute_metrics=True,
        ),
        stages=[
            StageEvaluationPolicy(
                stage_id="dense",
                variant_id="pool",
                target_levels=["page"],
                include_in_pool=True,
                compute_metrics=False,
                depth=20,
            ),
            StageEvaluationPolicy(
                stage_id="dense",
                variant_id="metrics",
                target_levels=["page"],
                include_in_pool=False,
                compute_metrics=True,
                depth=20,
            ),
        ],
    )
    selection = SimpleNamespace(experiment=experiment)
    config = load_relevance_config("configs/thesis/relevance.yaml")
    if compact:
        original = result.to_dict()
        reduced = _without_retrieval_traces(result)
        assert result.to_dict() == original
        assert reduced.items[0].metadata == {"mapping_status": "mapped"}
        assert reduced.context_items[0].metadata == {"mapping_status": "mapped"}
        assert reduced.context_serialization == result.context_serialization
        assert reduced.stage_results == result.stage_results
        validate_record_shape(reduced.to_dict())
        result = reduced

    pooled = _configured_raw_rankings(
        result,
        config,
        selection=selection,
        purpose="pool",
    )
    metrics = _configured_raw_rankings(
        result,
        config,
        selection=selection,
        purpose="metrics",
    )

    assert [ranking.stream for ranking in pooled] == ["items", "stage:dense@pool"]
    assert [ranking.stream for ranking in metrics] == [
        "items",
        "stage:dense@metrics",
    ]


def test_context_rankings_expose_unbounded_and_character_budget_views() -> None:
    first = RetrievedItem(
        rank=1,
        target_id="page:one",
        target_level="page",
        chunk_id=None,
        chunker_id=None,
        section_id=None,
        overlapping_section_ids=[],
        page_id="page:one",
        source_path="one.html",
        url=None,
        title=None,
        heading_path=[],
        text="AAAA",
        text_hash=hash_text("AAAA"),
        score=None,
        score_name="direct",
    )
    second = first.model_copy(
        update={
            "rank": 2,
            "target_id": "page:two",
            "page_id": "page:two",
            "source_path": "two.html",
            "text": "BBBB",
            "text_hash": hash_text("BBBB"),
        },
    )
    context_items, serialization = materialize_context(
        [
            ContextSource(item=first, origin="direct_intent"),
            ContextSource(
                item=second,
                origin="retrieval",
                source_item_ids=(second.target_id,),
                source_item_ranks=(2,),
            ),
        ],
        status="materialized",
    )
    result = RetrievalResult(
        query_id="question:one",
        system_id="system:one",
        run_id="run:one",
        items=[first, second],
        context_items=context_items,
        direct_context_items=[first],
        context_serialization=serialization,
        enabled_stages={},
        stage_traces=[],
        debug={},
        timings_ms={},
        config_hash=hash_text("config"),
        corpus_hash=CORPUS_HASH,
    )
    experiment = SimpleNamespace(
        top_k=2,
        final_evaluation=FinalEvaluationPolicy(
            streams=["context_items"],
            target_levels=["page"],
            context=ContextEvaluationPolicy(
                rank_cutoffs=[1, 2],
                character_budgets=[5, 8, 10],
            ),
        ),
        stages=[],
    )

    rankings = _configured_raw_rankings(
        result,
        load_relevance_config("configs/thesis/relevance.yaml"),
        selection=SimpleNamespace(experiment=experiment),
        purpose="metrics",
    )

    assert [ranking.stream for ranking in rankings] == [
        "context_items",
        "context_items@chars=5",
        "context_items@chars=8",
        "context_items@chars=10",
    ]
    assert [len(ranking.items) for ranking in rankings] == [2, 1, 2, 2]
    partial = rankings[2].items[-1]
    assert partial.text == "BB"
    assert partial.character_count == 2
    assert partial.serialized_char_start == 6
    assert partial.serialized_char_end == 8
    assert partial.metadata["context_fragment"]["source_context_item_id"] == (
        context_items[1].context_item_id
    )
    assert context_relevance_identity(partial) != (
        "page",
        context_items[1].target_id,
    )

    budget_only = experiment.final_evaluation.model_copy(
        update={
            "context": ContextEvaluationPolicy(
                rank_cutoffs=[1, 2],
                character_budgets=[5, 10],
                include_full_context=False,
            ),
        },
    )
    budget_rankings = _configured_raw_rankings(
        result,
        load_relevance_config("configs/thesis/relevance.yaml"),
        selection=SimpleNamespace(
            experiment=SimpleNamespace(
                top_k=2,
                final_evaluation=budget_only,
                stages=[],
            ),
        ),
        purpose="metrics",
    )
    assert [ranking.stream for ranking in budget_rankings] == [
        "context_items@chars=5",
        "context_items@chars=10",
    ]

    empty_context, empty_serialization = materialize_context(
        [],
        status="not_constructed",
    )
    missing = result.model_copy(
        update={
            "context_items": empty_context,
            "direct_context_items": [],
            "context_serialization": empty_serialization,
        },
    )
    with pytest.raises(RelevanceError, match="requires a materialized context"):
        _configured_raw_rankings(
            missing,
            load_relevance_config("configs/thesis/relevance.yaml"),
            selection=SimpleNamespace(experiment=experiment),
            purpose="metrics",
        )


def test_context_pool_and_metrics_include_rendered_and_projected_pages() -> None:
    question, page_text = _two_anchor_question()
    page_target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    page_item = RetrievedItem(
        rank=1,
        target_id="page:one",
        target_level="page",
        chunk_id=None,
        chunker_id=None,
        section_id=None,
        overlapping_section_ids=[],
        page_id="page:one",
        source_path="fixture.html",
        url=None,
        title="Fixture",
        heading_path=[],
        text=page_text,
        text_hash=hash_text(page_text),
        score=None,
        score_name="direct_intent",
    )
    context_items, serialization = materialize_context(
        [ContextSource(item=page_item, origin="direct_intent")],
        status="materialized",
    )
    result = RetrievalResult(
        query_id=question.query_id,
        system_id="system:fixture",
        run_id="run:fixture",
        items=[],
        context_items=context_items,
        direct_context_items=[page_item],
        context_serialization=serialization,
        enabled_stages={},
        stage_traces=[],
        debug={},
        timings_ms={},
        config_hash=hash_text("fixture config"),
        corpus_hash=CORPUS_HASH,
    )
    experiment = SimpleNamespace(
        experiment_id="fixture_experiment",
        top_k=5,
        final_evaluation=FinalEvaluationPolicy(
            streams=["context_items"],
            target_levels=["page"],
            context=ContextEvaluationPolicy(
                rank_cutoffs=[1, 2],
                character_budgets=[5],
            ),
        ),
        stages=[],
    )
    state = SimpleNamespace(
        config=load_relevance_config("configs/thesis/relevance.yaml"),
        questions=(question,),
        questions_by_id={question.query_id: question},
        retrieval_results=(result,),
        selection_by_run_id={
            result.run_id: SimpleNamespace(experiment=experiment),
        },
        selected_chunkers={},
        targets={
            ("page", "page:one"): RelevanceTarget(page_target, page_text),
        },
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
        cluster_page_families_by_query={},
        suite_run_id="suite:fixture",
    )
    rendered_items = [
        context_items[0],
        context_prefix(context_items, character_budget=5)[0],
    ]
    qrels = []
    for rendered_item in rendered_items:
        target_space, target_id = context_relevance_identity(rendered_item)
        rendered_ref = RetrievalTargetRef(
            target_id=target_id,
            target_level="page",
            target_space_id=target_space,
            page_id="page:one",
            text_hash=rendered_item.text_hash,
            source_path="fixture.html",
            title="Fixture",
            heading_path=[],
            char_start=0,
            char_end=len(rendered_item.text),
            target_artifact_hash=hash_text(target_id),
        )
        state.targets[(target_space, target_id)] = RelevanceTarget(
            rendered_ref,
            rendered_item.text,
        )
        qrels.append(
            QrelRecord(
                query_id=question.query_id,
                target_id=target_id,
                target_level="page",
                target_space_id=target_space,
                score=3,
                corpus_hash=CORPUS_HASH,
                view="hybrid",
            ),
        )

    records = _build_metrics(
        state,
        views={"hybrid": tuple(qrels)},
        projections=[],
        qrel_manifest_id="qrel_manifest:fixture",
        progress_callback=None,
    )
    signatures = {(record.stream, record.target_space_id) for record in records}

    assert signatures == {
        ("context_items", "page"),
        ("context_items", "context_rendered:page"),
        ("context_items", "context_native_mixed"),
        ("context_items@chars=5", "page"),
        ("context_items@chars=5", "context_rendered:page"),
        ("context_items@chars=5", "context_native_mixed"),
    }
    assert all(record.cutoffs == [1, 2] for record in records)
    budget_native = next(
        record
        for record in records
        if record.stream == "context_items@chars=5"
        and record.target_space_id == "context_native_mixed"
    )
    assert budget_native.metrics["partial_item_query_rate"] == 1.0

    candidates = _build_candidate_pool(state, (), progress_callback=None)
    diagnostics = _retrieval_projection_diagnostics(state)
    # Canonical Page, exact direct-intent rendering, and the budget fragment.
    assert len(candidates) == 3
    assert sum(len(candidate.seen_in) for candidate in candidates) == 3
    assert diagnostics["projected_retrieval_observations"] == 3
    assert diagnostics["target_levels"]["page"][
        "target_observations_after_parent_dedup"
    ] == 3


def test_candidate_pool_deduplicates_final_and_stage_hits_and_sets_priority() -> None:
    question, page_text = _two_anchor_question()
    page_one = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    page_two_text = "Second page."
    page_two = _target(
        target_id="page:two",
        level="page",
        page_id="page:two",
        text=page_two_text,
        char_start=0,
        char_end=len(page_two_text),
    )
    final_item = RetrievedItem(
        rank=1,
        target_id="page:one",
        target_level="page",
        chunk_id=None,
        chunker_id=None,
        section_id=None,
        overlapping_section_ids=[],
        page_id="page:one",
        source_path="fixture.html",
        url=None,
        title="Fixture",
        heading_path=[],
        text=page_text,
        text_hash=hash_text(page_text),
        score=1.0,
        score_name="final_score",
    )
    stage = StageResult(
        stage_id="vector_search",
        stage_label="Vector search",
        stage_index=0,
        variant_id=None,
        input_count=2,
        output_count=2,
        requested_depth=2,
        evaluable_up_to_k=2,
        ranking_complete=True,
        items=[
            StageCandidate(
                rank=1,
                target_id="legacy:one",
                target_level="chunk",
                chunk_id="legacy:one",
                chunker_id="legacy",
                section_id=None,
                page_id="page:one",
                score=0.1,
                score_name="distance",
            ),
            StageCandidate(
                rank=2,
                target_id="legacy:two",
                target_level="chunk",
                chunk_id="legacy:two",
                chunker_id="legacy",
                section_id=None,
                page_id="page:two",
                score=0.2,
                score_name="distance",
            ),
        ],
    )
    result = RetrievalResult(
        query_id=question.query_id,
        system_id="system:fixture",
        run_id="run:fixture",
        items=[final_item],
        context_items=[],
        direct_context_items=[],
        enabled_stages={},
        stage_traces=[],
        stage_results=[stage],
        debug={},
        timings_ms={},
        config_hash=hash_text("fixture config"),
        corpus_hash=CORPUS_HASH,
    )
    config = load_relevance_config("configs/thesis/relevance.yaml")
    experiment = SimpleNamespace(
        experiment_id="fixture_experiment",
        top_k=20,
        final_evaluation=FinalEvaluationPolicy(
            streams=["items"],
            target_levels=["page"],
            include_in_pool=True,
            compute_metrics=True,
        ),
        stages=[
            StageEvaluationPolicy(
                stage_id="vector_search",
                target_levels=["page"],
                include_in_pool=True,
                compute_metrics=True,
                depth=20,
            ),
        ],
    )
    state = SimpleNamespace(
        questions=(question,),
        questions_by_id={question.query_id: question},
        cluster_page_families_by_query={},
        retrieval_results=(result,),
        config=config,
        selected_chunkers={},
        targets={
            ("page", "page:one"): RelevanceTarget(page_one, page_text),
            ("page", "page:two"): RelevanceTarget(page_two, page_two_text),
        },
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
        suite_run_id="suite:fixture",
        selection_by_run_id={
            "run:fixture": SimpleNamespace(experiment=experiment),
        },
    )

    candidates = _build_candidate_pool(state, (), progress_callback=None)
    by_target = {candidate.target.target_id: candidate for candidate in candidates}

    assert set(by_target) == {"page:one", "page:two"}
    assert by_target["page:one"].judgment_priority == "final"
    assert [item.stream for item in by_target["page:one"].seen_in] == [
        "items",
        "stage:vector_search",
    ]
    assert by_target["page:two"].judgment_priority == "intermediate_only"

    qrels = tuple(
        QrelRecord(
            query_id=question.query_id,
            target_id=target_id,
            target_level="page",
            target_space_id="page",
            score=3 if target_id == "page:one" else 0,
            corpus_hash=CORPUS_HASH,
            view="hybrid",
        )
        for target_id in ("page:one", "page:two")
    )
    metric_records = _build_metrics(
        state,
        views={"hybrid": qrels},
        projections=(),
        candidates=candidates,
        qrel_manifest_id="qrel_manifest:fixture",
        progress_callback=None,
    )
    by_stream = {record.stream: record for record in metric_records}

    assert set(by_stream) == {"items", "stage:vector_search"}
    assert by_stream["stage:vector_search"].target_space_id == "page"
    assert by_stream["stage:vector_search"].metrics["at_k"]["1"][
        "qrel_metrics_computable"
    ] is True


def test_judge_input_is_independent_and_uses_breadcrumb_context() -> None:
    question, page_text = _two_anchor_question()
    target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )

    active_input = build_active_judgment_input(
        question,
        RelevanceTarget(
            target,
            page_text,
            ("TU Berlin", "Studium", "Pruefungen"),
        ),
    )

    assert active_input["target"]["breadcrumbs"] == [
        "TU Berlin",
        "Studium",
        "Pruefungen",
    ]
    assert "reference_answer" not in str(active_input)
    assert "required_facts" not in str(active_input)
    assert "temporal_classification" not in str(active_input)
    assert "anchor_score_floor" not in str(active_input)


def test_judge_response_schema_is_auditable_without_grade_two_shortcut() -> None:
    properties = RelevanceJudgeResponse.model_json_schema()["properties"]

    assert list(properties) == [
        "query_scope_summary",
        "answer_support_summary",
        "missing_requested_information",
        "score",
    ]
    response = RelevanceJudgeResponse.from_dict(
        {
            "query_scope_summary": "The query asks for a deadline.",
            "answer_support_summary": "The deadline is stated indirectly.",
            "missing_requested_information": [],
            "score": 2,
        },
    )
    assert response.score == 2
    assert response.missing_requested_information == []
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RelevanceJudgeResponse.from_dict(
            {
                **response.to_dict(),
                "free_form_chain_of_thought": "not part of the contract",
            },
        )


def test_metrics_keep_anchor_diagnostics_separate_from_qrel_view() -> None:
    question, page_text = _two_anchor_question()
    target = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    projection = _projection(question, target, 3)
    projection_id = AnchorProjectionRecord.canonical_v3_id(
        query_id=question.query_id,
        query_text_hash=hash_text(question.text),
        target=target,
        candidate_id=projection.candidate_id,
        review_event_id=projection.review_event_id,
        projection_kind="full_containment",
        anchor_score_floor=3,
        contained_anchor_ids=["span:first", "span:second"],
        overlapping_anchor_ids=[],
        fully_contained_sufficient_set_ids=["sufficient_set:fixture"],
        supported_fact_refs=["fact:1", "fact:2"],
        derivation_rules=["contains_complete_sufficient_evidence_set"],
        corpus_hash=CORPUS_HASH,
        target_space_manifest_hash=TARGET_SPACE_HASH,
        projection_rule_version="projection_fixture_v1",
    )
    projection = AnchorProjectionRecord.from_dict(
        {
            **projection.to_dict(),
            "projection_id": projection_id,
            "contained_anchor_ids": ["span:first", "span:second"],
            "fully_contained_sufficient_set_ids": ["sufficient_set:fixture"],
            "supported_fact_refs": ["fact:1", "fact:2"],
            "derivation_rules": ["contains_complete_sufficient_evidence_set"],
        },
    )
    qrel = QrelRecord(
        query_id=question.query_id,
        target_id=target.target_id,
        target_level="page",
        target_space_id="page",
        score=3,
        corpus_hash=CORPUS_HASH,
        view="hybrid",
    )
    retrieved = RetrievedItem(
        rank=1,
        target_id=target.target_id,
        target_level="page",
        chunk_id=None,
        chunker_id=None,
        section_id=None,
        overlapping_section_ids=[],
        page_id=target.page_id,
        source_path=target.source_path,
        url=None,
        title=target.title,
        heading_path=[],
        text=page_text,
        text_hash=target.text_hash,
        score=1.0,
        score_name="fixture_score",
    )
    state = SimpleNamespace(
        config=SimpleNamespace(
            metrics=SimpleNamespace(cutoffs=[1]),
            pooling=SimpleNamespace(retrieval_top_k=20),
        ),
        questions=(question,),
        questions_by_id={question.query_id: question},
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
    )

    record = _metric_group_records(
        state=state,
        run_id="run:fixture",
        system_id="system:fixture",
        split="development",
        stream="items",
        target_space="page",
        results_by_query={question.query_id: (retrieved,)},
        view="hybrid",
        qrels=[qrel],
        projections=[projection],
        qrel_manifest_id="qrel_manifest:fixture",
    )

    at_one = record.metrics["at_k"]["1"]
    assert at_one["ndcg"] == 1.0
    assert "judged" not in at_one
    assert at_one["reference_anchor_hit"] == 1.0
    assert at_one["reference_anchor_set_hit"] == 1.0
    assert at_one["reference_anchor_fact_coverage"] == 1.0
    assert record.metrics["judgment_scope"] == "pooled_candidates"
    assert record.metrics["reference_anchor_fact_coverage_eligible_query_count"] == 1

    grade_two_only = _metric_group_records(
        state=state,
        run_id="run:fixture",
        system_id="system:fixture",
        split="development",
        stream="items",
        target_space="page",
        results_by_query={question.query_id: (retrieved,)},
        view="hybrid",
        qrels=[qrel.model_copy(update={"score": 2})],
        projections=[projection],
        qrel_manifest_id="qrel_manifest:fixture",
    )

    assert grade_two_only.metrics["qrel_eligible_query_count"] == 1
    assert grade_two_only.metrics["strict_qrel_eligible_query_count"] == 0
    assert grade_two_only.metrics["at_k"]["1"]["hit_at_grade_2"] == 1.0
    assert grade_two_only.metrics["at_k"]["1"]["strict_hit_at_grade_3"] is None


def test_stage_metrics_require_evaluable_depth_and_complete_pool_coverage() -> None:
    question, page_text = _two_anchor_question()
    first = _target(
        target_id="page:one",
        level="page",
        page_id="page:one",
        text=page_text,
        char_start=0,
        char_end=len(page_text),
    )
    second_text = "A pooled but not retrieved page."
    second = _target(
        target_id="page:two",
        level="page",
        page_id="page:two",
        text=second_text,
        char_start=0,
        char_end=len(second_text),
    )
    retrieved = RetrievedItem(
        rank=1,
        target_id=first.target_id,
        target_level="page",
        chunk_id=None,
        chunker_id=None,
        section_id=None,
        overlapping_section_ids=[],
        page_id=first.page_id,
        source_path=first.source_path,
        url=None,
        title=first.title,
        heading_path=[],
        text=page_text,
        text_hash=first.text_hash,
        score=1.0,
        score_name="fixture_score",
    )
    state = SimpleNamespace(
        config=SimpleNamespace(
            metrics=SimpleNamespace(
                cutoffs=[1, 3],
                minimum_judgment_coverage=1.0,
            ),
            pooling=SimpleNamespace(retrieval_top_k=3),
        ),
        questions=(question,),
        questions_by_id={question.query_id: question},
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
    )
    first_qrel = QrelRecord(
        query_id=question.query_id,
        target_id=first.target_id,
        target_level="page",
        target_space_id="page",
        score=3,
        corpus_hash=CORPUS_HASH,
        view="hybrid",
    )
    second_qrel = QrelRecord(
        query_id=question.query_id,
        target_id=second.target_id,
        target_level="page",
        target_space_id="page",
        score=0,
        corpus_hash=CORPUS_HASH,
        view="hybrid",
    )
    kwargs = {
        "state": state,
        "run_id": "run:fixture",
        "system_id": "system:fixture",
        "split": "development",
        "stream": "stage:vector_search",
        "target_space": "page",
        "results_by_query": {question.query_id: (retrieved,)},
        "evaluable_by_query": {question.query_id: 1},
        "view": "hybrid",
        "projections": [],
        "candidates": [_candidate(question, first), _candidate(question, second)],
        "qrel_manifest_id": "qrel_manifest:fixture",
    }

    incomplete = _metric_group_records(qrels=[first_qrel], **kwargs)
    complete = _metric_group_records(qrels=[first_qrel, second_qrel], **kwargs)

    assert incomplete.metrics["at_k"]["1"]["qrel_metrics_computable"] is False
    assert incomplete.metrics["at_k"]["1"]["ndcg"] is None
    assert incomplete.metrics["at_k"]["1"]["blocked_reasons"] == [
        "insufficient_judgment_coverage",
    ]
    assert complete.metrics["at_k"]["1"]["qrel_metrics_computable"] is True
    assert complete.metrics["at_k"]["1"]["ndcg"] == 1.0
    assert complete.metrics["at_k"]["3"]["qrel_metrics_computable"] is False
    assert complete.metrics["at_k"]["3"]["blocked_reasons"] == [
        "ranking_not_evaluable_at_cutoff",
    ]


def test_metrics_materialize_declared_target_spaces_for_an_empty_stream() -> None:
    question, _page_text = _two_anchor_question()
    result = RetrievalResult(
        query_id=question.query_id,
        system_id="system:fixture",
        run_id="run:fixture",
        items=[],
        context_items=[],
        direct_context_items=[],
        enabled_stages={},
        stage_traces=[],
        debug={},
        timings_ms={},
        config_hash=hash_text("fixture config"),
        corpus_hash=CORPUS_HASH,
    )
    page_qrel = QrelRecord(
        query_id=question.query_id,
        target_id="page:one",
        target_level="page",
        target_space_id="page",
        score=3,
        corpus_hash=CORPUS_HASH,
        view="hybrid",
    )
    section_qrel = QrelRecord(
        query_id=question.query_id,
        target_id="section:one",
        target_level="section",
        target_space_id="section",
        score=3,
        corpus_hash=CORPUS_HASH,
        view="hybrid",
    )
    state = SimpleNamespace(
        config=load_relevance_config("configs/thesis/relevance.yaml"),
        questions=(question,),
        questions_by_id={question.query_id: question},
        retrieval_results=(result,),
        selection_by_run_id={},
        selected_chunkers={},
        targets={},
        snapshot_manifest=SimpleNamespace(corpus_hash=CORPUS_HASH),
    )

    records = _build_metrics(
        state,
        views={"hybrid": (page_qrel, section_qrel)},
        projections=[],
        qrel_manifest_id="qrel_manifest:fixture",
        progress_callback=None,
    )

    assert {record.target_space_id for record in records} == {"page", "section"}
    assert all(record.metrics["at_k"]["20"]["ndcg"] == 0.0 for record in records)
