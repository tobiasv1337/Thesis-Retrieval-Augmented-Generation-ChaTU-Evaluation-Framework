import pytest

from thesis_rag.context import ContextSource, materialize_context
from thesis_rag.interfaces import (
    AnswerResult,
    EvidenceTarget,
    RagQuery,
    RagSystemInfo,
    RetrievedItem,
    RetrievalResult,
    StageCandidate,
    StageResult,
    StageTrace,
)


def _retrieved_item() -> RetrievedItem:
    return RetrievedItem(
        rank=1,
        target_id="chunk:1",
        target_level="chunk",
        chunk_id="chunk:1",
        chunker_id="baseline",
        section_id="section:1",
        overlapping_section_ids=["section:1"],
        page_id="page:1",
        source_path="data/az/scraper/topics/example/content.html",
        url="https://example.test",
        title="Example",
        heading_path=["Root", "Heading"],
        text="Some evidence text.",
        text_hash="sha256:982d9e3eb996f559e633f4d194def3761d909f5a3b647d1a851fead67c32c9d1",
        score=0.5,
        score_name="fixture_score",
        metadata={"topic": "example"},
    )


def test_rag_query_roundtrip_and_derived_targets() -> None:
    source = EvidenceTarget(
        target_level="span",
        target_id=None,
        page_id="page:1",
        section_id="section:1",
        chunk_id="chunk:1",
        chunker_id="baseline",
        char_start=0,
        char_end=12,
        text_hash="sha256:37a0025aa1f2e29f9f88f59f961e44a320849dfc79a85ef3c02654d50d2e7480",
    )
    acceptable = EvidenceTarget(
        target_level="section",
        target_id="section:1",
        page_id="page:1",
        section_id="section:1",
        metadata={"table_id": "table:1"},
    )

    query = RagQuery(
        query_id="q1",
        text="Where is the evidence?",
        language="en",
        source_targets=[source],
        acceptable_targets=[acceptable],
    )
    restored = RagQuery.from_dict(query.to_dict())

    assert restored == query
    assert restored.source_page_ids == ["page:1"]
    assert restored.source_section_ids == ["section:1"]
    assert restored.source_chunk_ids == ["chunk:1"]
    assert restored.acceptable_page_ids == ["page:1"]
    assert restored.acceptable_section_ids == ["section:1"]
    assert restored.acceptable_chunk_ids == []
    assert restored.acceptable_targets[0].metadata == {"table_id": "table:1"}


def test_evidence_target_enforces_level_specific_ids() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        EvidenceTarget(
            target_level="chunk",
            target_id="chunk:1",
            page_id="page:1",
            section_id="section:1",
            chunker_id="baseline",
        )

    with pytest.raises(ValueError, match="char_start and char_end"):
        EvidenceTarget(
            target_level="span",
            target_id=None,
            page_id="page:1",
            section_id="section:1",
            text_hash="sha256:37a0025aa1f2e29f9f88f59f961e44a320849dfc79a85ef3c02654d50d2e7480",
        )

    with pytest.raises(ValueError, match="metadata"):
        EvidenceTarget(
            target_level="page",
            target_id="page:1",
            page_id="page:1",
            metadata=[],
        )


def test_retrieval_result_roundtrip() -> None:
    item = _retrieved_item()
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
    trace = StageTrace(
        stage="dense",
        enabled=True,
        mode="faiss",
        input_count=None,
        output_count=1,
        output_ids_sample=["chunk:1"],
        timings_ms={"dense": 1.2},
    )
    stage_result = StageResult(
        stage_id="dense",
        stage_label="Dense retrieval",
        stage_index=0,
        variant_id=None,
        input_count=None,
        output_count=1,
        requested_depth=10,
        evaluable_up_to_k=10,
        ranking_complete=True,
        items=[
            StageCandidate(
                rank=1,
                target_id=item.target_id,
                target_level=item.target_level,
                chunk_id=item.chunk_id,
                chunker_id=item.chunker_id,
                section_id=item.section_id,
                page_id=item.page_id,
                score=item.score,
                score_name=item.score_name,
            ),
        ],
    )
    result = RetrievalResult(
        query_id="q1",
        system_id="fixture",
        run_id="run1",
        items=[item],
        context_items=context_items,
        direct_context_items=[],
        context_serialization=context_serialization,
        enabled_stages={"dense": {"enabled": True}},
        stage_traces=[trace],
        stage_results=[stage_result],
        debug={"note": "ok"},
        timings_ms={"total": 1.3},
        config_hash="sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
    )

    assert RetrievalResult.from_dict(result.to_dict()) == result
    assert context_items[0].serialized_char_start == 0
    assert context_items[0].serialized_char_end == len(item.text)


def test_context_lineage_deduplicates_source_id_rank_pairs_atomically() -> None:
    item = _retrieved_item()

    context_items, _serialization = materialize_context(
        [
            ContextSource(
                item=item,
                origin="parent_expansion",
                source_item_ids=("chunk:a", "chunk:b", "chunk:a", "chunk:a"),
                source_item_ranks=(1, 1, 2, 1),
            ),
        ],
        status="materialized",
    )

    assert list(
        zip(
            context_items[0].source_item_ids,
            context_items[0].source_item_ranks,
            strict=True,
        ),
    ) == [("chunk:a", 1), ("chunk:b", 1), ("chunk:a", 2)]


def test_context_item_identity_is_validated_when_loading() -> None:
    item = _retrieved_item()
    context_items, _serialization = materialize_context(
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
    raw = context_items[0].to_dict()
    raw["context_item_id"] = "context_item:tampered"

    with pytest.raises(ValueError, match="context_item_id"):
        type(context_items[0]).from_dict(raw)


def test_stage_result_rejects_evaluable_depth_beyond_stored_ranking() -> None:
    item = _retrieved_item()
    with pytest.raises(ValueError, match="exceeds its stored prefix"):
        StageResult(
            stage_id="dense",
            stage_label="Dense retrieval",
            stage_index=0,
            variant_id=None,
            input_count=None,
            output_count=100,
            requested_depth=10,
            evaluable_up_to_k=2,
            ranking_complete=False,
            items=[
                StageCandidate(
                    rank=1,
                    target_id=item.target_id,
                    target_level=item.target_level,
                    chunk_id=item.chunk_id,
                    chunker_id=item.chunker_id,
                    section_id=item.section_id,
                    page_id=item.page_id,
                    score=item.score,
                    score_name=item.score_name,
                ),
            ],
        )


def test_retrieved_item_enforces_level_specific_ids() -> None:
    with pytest.raises(ValueError, match="chunker_id"):
        RetrievedItem(
            rank=1,
            target_id="chunk:1",
            target_level="chunk",
            chunk_id="chunk:1",
            chunker_id=None,
            section_id="section:1",
            overlapping_section_ids=["section:1"],
            page_id="page:1",
            source_path="data/az/scraper/topics/example/content.html",
            url=None,
            title=None,
            heading_path=[],
            text="Evidence.",
            text_hash="sha256:982d9e3eb996f559e633f4d194def3761d909f5a3b647d1a851fead67c32c9d1",
            score=None,
            score_name="fixture",
        )


def test_legacy_chunk_can_preserve_page_only_alignment() -> None:
    item = RetrievedItem(
        rank=1,
        target_id="legacy_chunk:1",
        target_level="chunk",
        chunk_id="legacy_chunk:1",
        chunker_id="legacy_recursive_character_800_100",
        section_id=None,
        overlapping_section_ids=[],
        page_id="page:1",
        source_path="data/az/scraper/topics/example/content.html",
        url=None,
        title=None,
        heading_path=[],
        text="Legacy evidence.",
        text_hash="sha256:ab32bf1d73cbc97bd35a9670cd14f32cbe3ad71e420f092921200643c5f8b491",
        score=0.4,
        score_name="faiss_l2_distance",
    )

    assert item.section_id is None
    assert item.overlapping_section_ids == []


def test_retrieved_item_requires_primary_section_in_overlap_list() -> None:
    data = _retrieved_item().to_dict()
    data["overlapping_section_ids"] = ["section:other"]

    with pytest.raises(ValueError, match="section_id must occur"):
        RetrievedItem.from_dict(data)


def test_answer_and_system_info_roundtrip() -> None:
    answer = AnswerResult(
        query_id="q1",
        system_id="fixture",
        answer="A short answer.",
        context_item_ids=["chunk:1"],
        prompt_hash="sha256:cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
        model_id="local-model",
        model_revision=None,
        timings_ms={"answer": 5.0},
        debug={"temperature": 0.0},
    )
    info = RagSystemInfo(
        system_id="fixture",
        name="Fixture",
        class_path="tests.thesis_rag.fixtures.registry_system.FixtureRagSystem",
        capabilities=["retrieval"],
        config_hash="sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910",
    )

    assert AnswerResult.from_dict(answer.to_dict()) == answer
    assert RagSystemInfo.from_dict(info.to_dict()) == info


def test_persisted_interface_records_include_schema_versions() -> None:
    item = _retrieved_item()
    trace = StageTrace(
        stage="dense",
        enabled=True,
        mode="faiss",
        input_count=0,
        output_count=1,
    )
    context_items, context_serialization = materialize_context(
        [],
        status="not_constructed",
    )
    retrieval = RetrievalResult(
        query_id="q1",
        system_id="fixture",
        run_id="run1",
        items=[item],
        context_items=context_items,
        direct_context_items=[],
        context_serialization=context_serialization,
        enabled_stages={"dense": {"enabled": True}},
        stage_traces=[trace],
        debug={},
        timings_ms={},
        config_hash="sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910",
        corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
    )
    answer = AnswerResult(
        query_id="q1",
        system_id="fixture",
        answer="",
        context_item_ids=[],
        prompt_hash="sha256:cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
        model_id="fixture",
        model_revision=None,
        timings_ms={},
    )

    assert trace.to_dict()["schema_version"] == "stage_trace_v1"
    assert retrieval.to_dict()["schema_version"] == "retrieval_result_v5"
    assert answer.to_dict()["schema_version"] == "answer_result_v2"


def test_interface_hash_fields_require_sha256_prefix() -> None:
    with pytest.raises(ValueError, match="text_hash"):
        EvidenceTarget(
            target_level="span",
            target_id=None,
            page_id="page:1",
            section_id="section:1",
            char_start=0,
            char_end=5,
            text_hash="not-a-hash",
        )

    item_data = _retrieved_item().to_dict()
    item_data["text_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="text_hash"):
        RetrievedItem.from_dict(item_data)

    with pytest.raises(ValueError, match="config_hash"):
        context_items, context_serialization = materialize_context(
            [],
            status="not_constructed",
        )
        RetrievalResult(
            query_id="q1",
            system_id="fixture",
            run_id="run1",
            items=[],
            context_items=context_items,
            direct_context_items=[],
            context_serialization=context_serialization,
            enabled_stages={},
            stage_traces=[],
            debug={},
            timings_ms={},
            config_hash="not-a-hash",
            corpus_hash="sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe",
        )

    with pytest.raises(ValueError, match="prompt_hash"):
        AnswerResult(
            query_id="q1",
            system_id="fixture",
            answer="",
            context_item_ids=[],
            prompt_hash="not-a-hash",
            model_id="fixture",
            model_revision=None,
            timings_ms={},
        )
