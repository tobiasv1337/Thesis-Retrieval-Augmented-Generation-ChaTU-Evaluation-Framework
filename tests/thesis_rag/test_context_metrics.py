from __future__ import annotations

from thesis_rag.context import (
    ContextSource,
    context_prefix,
    context_relevance_identity,
    materialize_context,
)
from thesis_rag.eval.context_metrics import (
    ContextMetricInput,
    _overlap_deduplicated,
    build_context_metric_payloads,
)
from thesis_rag.eval.relevance_schemas import (
    AnchorProjectionRecord,
    QrelRecord,
    RetrievalTargetRef,
)
from thesis_rag.interfaces import RetrievedItem
from thesis_rag.schema import hash_text


CORPUS_HASH = hash_text("context-metric-corpus")


def _item(
    *,
    rank: int,
    target_id: str,
    target_level: str,
    page_id: str,
    text: str,
    section_id: str | None = None,
    overlapping_section_ids: list[str] | None = None,
) -> RetrievedItem:
    chunker_id = "fixture_chunker" if target_level == "chunk" else None
    return RetrievedItem(
        rank=rank,
        target_id=target_id,
        target_level=target_level,
        chunk_id=target_id if target_level == "chunk" else None,
        chunker_id=chunker_id,
        section_id=section_id,
        overlapping_section_ids=(
            overlapping_section_ids
            if overlapping_section_ids is not None
            else [section_id] if section_id else []
        ),
        page_id=page_id,
        source_path=f"{page_id}.html",
        url=None,
        title=None,
        heading_path=[],
        text=text,
        text_hash=hash_text(text),
        score=None,
        score_name="fixture",
    )


def _metric_input(
    query_id: str,
    item: RetrievedItem,
    *,
    evaluable_up_to_k: int = 1,
) -> ContextMetricInput:
    context, serialization = materialize_context(
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
    return ContextMetricInput(
        query_id=query_id,
        items=tuple(context),
        evaluable_up_to_k=evaluable_up_to_k,
        visible_character_count=serialization.character_count,
        partial_item_visible=False,
    )


def _projection(
    query_id: str,
    item: RetrievedItem,
    *,
    anchors: list[str],
    facts: list[str],
) -> AnchorProjectionRecord:
    target = RetrievalTargetRef(
        target_id=item.target_id,
        target_level=item.target_level,
        target_space_id="page",
        page_id=item.page_id,
        text_hash=item.text_hash,
        source_path=item.source_path,
        char_start=0,
        char_end=len(item.text),
        target_artifact_hash=hash_text(f"artifact:{item.target_id}"),
    )
    fields = {
        "query_id": query_id,
        "query_text_hash": hash_text("Question"),
        "target": target,
        "candidate_id": f"candidate:{query_id}",
        "review_event_id": f"review:{query_id}",
        "projection_kind": "full_containment",
        "anchor_score_floor": 2,
        "contained_anchor_ids": anchors,
        "overlapping_anchor_ids": [],
        "fully_contained_sufficient_set_ids": [],
        "supported_fact_refs": facts,
        "derivation_rules": ["fixture_containment"],
        "corpus_hash": CORPUS_HASH,
        "target_space_manifest_hash": hash_text("page-target-space"),
        "projection_rule_version": "fixture_v1",
    }
    return AnchorProjectionRecord(
        projection_id=AnchorProjectionRecord.canonical_v3_id(**fields),
        **fields,
    )


def test_mixed_context_metrics_do_not_count_a_page_and_its_child_twice() -> None:
    page = _item(
        rank=1,
        target_id="page:one",
        target_level="page",
        page_id="page:one",
        text="Whole page",
    )
    chunk = _item(
        rank=1,
        target_id="chunk:one",
        target_level="chunk",
        page_id="page:one",
        section_id="section:one",
        text="Answering chunk",
    )
    context, _serialization = materialize_context(
        [
            ContextSource(item=page, origin="direct_intent"),
            ContextSource(
                item=chunk,
                origin="retrieval",
                source_item_ids=(chunk.target_id,),
                source_item_ranks=(chunk.rank,),
            ),
        ],
        status="materialized",
    )
    rendered_page_space, rendered_page_id = context_relevance_identity(context[0])
    qrels = [
        QrelRecord(
            query_id="question:one",
            target_id=rendered_page_id,
            target_level="page",
            target_space_id=rendered_page_space,
            score=3,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
        QrelRecord(
            query_id="question:one",
            target_id="chunk:one",
            target_level="chunk",
            target_space_id="chunk:fixture_chunker",
            score=3,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
    ]

    native, reference, native_queries, _reference_queries = (
        build_context_metric_payloads(
            [
                ContextMetricInput(
                    query_id="question:one",
                    items=tuple(context),
                    evaluable_up_to_k=20,
                    visible_character_count=100,
                    partial_item_visible=False,
                    routing_hint_character_count=42,
                ),
            ],
            qrels=qrels,
            projections=[],
            question_metadata={},
            cutoffs=[1, 2],
        )
    )

    query = native_queries["question:one"]
    assert query["raw_context_item_count"] == 2
    assert query["deduplicated_context_item_count"] == 1
    assert native["relevant_target_count_at_grade_3"] == 1
    assert native["routing_hint_query_rate"] == 1.0
    assert native["mean_routing_hint_character_count"] == 42.0
    assert native["at_k"]["2"]["qrel_metrics_computable"] is True
    assert reference["at_k"]["2"]["reference_anchor_set_hit"] is None


def test_context_cutoff_precedes_deduplication_and_preserves_original_ranks() -> None:
    page = _item(
        rank=1,
        target_id="page:one",
        target_level="page",
        page_id="page:one",
        text="Whole first page",
    )
    child = _item(
        rank=2,
        target_id="chunk:one",
        target_level="chunk",
        page_id="page:one",
        section_id="section:one",
        text="Child of first page",
    )
    other = _item(
        rank=3,
        target_id="page:two",
        target_level="page",
        page_id="page:two",
        text="Relevant second page",
    )
    context, _serialization = materialize_context(
        [
            ContextSource(item=page, origin="direct_intent"),
            ContextSource(
                item=child,
                origin="retrieval",
                source_item_ids=(child.target_id,),
                source_item_ranks=(child.rank,),
            ),
            ContextSource(
                item=other,
                origin="retrieval",
                source_item_ids=(other.target_id,),
                source_item_ranks=(other.rank,),
            ),
        ],
        status="materialized",
    )
    rendered_space, rendered_id = context_relevance_identity(context[0])
    qrels = [
        QrelRecord(
            query_id="question:one",
            target_id=rendered_id,
            target_level="page",
            target_space_id=rendered_space,
            score=0,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
        QrelRecord(
            query_id="question:one",
            target_id="page:two",
            target_level="page",
            target_space_id="page",
            score=3,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
    ]

    native, _reference, native_queries, _reference_queries = (
        build_context_metric_payloads(
            [
                ContextMetricInput(
                    query_id="question:one",
                    items=tuple(context),
                    evaluable_up_to_k=3,
                    visible_character_count=100,
                    partial_item_visible=False,
                ),
            ],
            qrels=qrels,
            projections=[],
            question_metadata={},
            cutoffs=[2, 3],
        )
    )

    query = native_queries["question:one"]["at_k"]
    assert query["2"]["hit_at_grade_2"] == 0.0
    assert query["3"]["hit_at_grade_2"] == 1.0
    assert query["3"]["mrr_at_grade_2"] == 1.0 / 3.0
    assert native["at_k"]["3"]["mrr_at_grade_2"] == 1.0 / 3.0


def test_later_parent_remains_without_double_counting_first_hit() -> None:
    chunk = _item(
        rank=1,
        target_id="chunk:one",
        target_level="chunk",
        page_id="page:one",
        section_id="section:one",
        text="Answering chunk",
    )
    page = _item(
        rank=2,
        target_id="page:one",
        target_level="page",
        page_id="page:one",
        text="Answering chunk with additional page evidence",
    )
    context, serialization = materialize_context(
        [
            ContextSource(
                item=chunk,
                origin="retrieval",
                source_item_ids=(chunk.target_id,),
                source_item_ranks=(chunk.rank,),
            ),
            ContextSource(
                item=page,
                origin="parent_expansion",
                source_item_ids=(chunk.target_id,),
                source_item_ranks=(chunk.rank,),
            ),
        ],
        status="materialized",
    )
    qrels = [
        QrelRecord(
            query_id="question:one",
            target_id="chunk:one",
            target_level="chunk",
            target_space_id="chunk:fixture_chunker",
            score=2,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
        QrelRecord(
            query_id="question:one",
            target_id="page:one",
            target_level="page",
            target_space_id="page",
            score=3,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
    ]

    native, _reference, queries, _reference_queries = build_context_metric_payloads(
        [
            ContextMetricInput(
                query_id="question:one",
                items=tuple(context),
                evaluable_up_to_k=2,
                visible_character_count=serialization.character_count,
                partial_item_visible=False,
            ),
        ],
        qrels=qrels,
        projections=[],
        question_metadata={},
        cutoffs=[1, 2],
    )

    query = queries["question:one"]
    assert query["deduplicated_context_item_count"] == 2
    assert query["at_k"]["1"]["mrr_at_grade_2"] == 1.0
    assert query["at_k"]["2"]["mrr_at_grade_2"] == 1.0
    assert query["at_k"]["2"]["strict_mrr_at_grade_3"] == 0.5
    assert query["at_k"]["1"]["prefix_character_count"] == len(chunk.text)
    assert query["at_k"]["2"]["prefix_character_count"] == (
        serialization.character_count
    )
    assert native["at_k"]["2"]["prefix_character_count"] == (
        serialization.character_count
    )


def test_cross_section_chunk_is_removed_only_after_all_sections_are_covered() -> None:
    section_a = _item(
        rank=1,
        target_id="section:a",
        target_level="section",
        page_id="page:one",
        section_id="section:a",
        text="Section A",
    )
    section_b = _item(
        rank=2,
        target_id="section:b",
        target_level="section",
        page_id="page:one",
        section_id="section:b",
        text="Section B",
    )
    cross_section_chunk = _item(
        rank=3,
        target_id="chunk:ab",
        target_level="chunk",
        page_id="page:one",
        section_id="section:a",
        overlapping_section_ids=["section:a", "section:b"],
        text="Evidence crossing A and B",
    )

    def deduplicated_target_ids(items: list[RetrievedItem]) -> list[str]:
        context, _serialization = materialize_context(
            [
                ContextSource(
                    item=item,
                    origin="parent_expansion",
                    source_item_ids=(item.target_id,),
                    source_item_ranks=(item.rank,),
                )
                for item in items
            ],
            status="materialized",
        )
        return [item.target_id for item in _overlap_deduplicated(context)]

    assert deduplicated_target_ids([section_a, cross_section_chunk]) == [
        "section:a",
        "chunk:ab",
    ]
    assert deduplicated_target_ids(
        [section_a, section_b, cross_section_chunk],
    ) == ["section:a", "section:b"]


def test_separator_only_budget_is_included_in_prefix_character_count() -> None:
    first = _item(
        rank=1,
        target_id="page:one",
        target_level="page",
        page_id="page:one",
        text="AAAA",
    )
    second = _item(
        rank=2,
        target_id="page:two",
        target_level="page",
        page_id="page:two",
        text="BBBB",
    )
    context, _serialization = materialize_context(
        [
            ContextSource(
                item=item,
                origin="retrieval",
                source_item_ids=(item.target_id,),
                source_item_ranks=(item.rank,),
            )
            for item in (first, second)
        ],
        status="materialized",
    )
    budget = context[0].serialized_char_end + 1
    visible = context_prefix(context, character_budget=budget)

    native, _reference, queries, _reference_queries = build_context_metric_payloads(
        [
            ContextMetricInput(
                query_id="question:one",
                items=visible,
                evaluable_up_to_k=2,
                visible_character_count=budget,
                partial_item_visible=False,
            ),
        ],
        qrels=[
            QrelRecord(
                query_id="question:one",
                target_id="page:one",
                target_level="page",
                target_space_id="page",
                score=0,
                corpus_hash=CORPUS_HASH,
                view="judged_only",
            ),
        ],
        projections=[],
        question_metadata={},
        cutoffs=[1],
    )

    assert len(visible) == 1
    assert queries["question:one"]["at_k"]["1"]["prefix_character_count"] == budget
    assert native["full_context"]["prefix_character_count"] == budget


def test_full_context_diagnostic_includes_blocks_beyond_the_largest_cutoff() -> None:
    sources = [
        _item(
            rank=rank,
            target_id=f"page:{rank}",
            target_level="page",
            page_id=f"page:{rank}",
            text=f"Page {rank}",
        )
        for rank in range(1, 17)
    ]
    context, _serialization = materialize_context(
        [
            ContextSource(
                item=item,
                origin="retrieval",
                source_item_ids=(item.target_id,),
                source_item_ranks=(item.rank,),
            )
            for item in sources
        ],
        status="materialized",
    )
    qrels = [
        QrelRecord(
            query_id="question:one",
            target_id=item.target_id,
            target_level="page",
            target_space_id="page",
            score=3 if item.rank == 16 else 0,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        )
        for item in sources
    ]

    native, _reference, native_queries, _reference_queries = (
        build_context_metric_payloads(
            [
                ContextMetricInput(
                    query_id="question:one",
                    items=tuple(context),
                    evaluable_up_to_k=15,
                    visible_character_count=1000,
                    partial_item_visible=False,
                ),
            ],
            qrels=qrels,
            projections=[],
            question_metadata={},
            cutoffs=[15],
        )
    )

    query = native_queries["question:one"]
    assert query["at_k"]["15"]["hit_at_grade_2"] == 0.0
    assert query["full_context"]["hit_at_grade_2"] == 1.0
    assert query["full_context"]["mrr_at_grade_2"] == 1.0 / 16.0
    assert native["full_context"]["hit_at_grade_2"] == 1.0


def test_context_metrics_mark_an_unjudged_visible_target_incomplete() -> None:
    page = _item(
        rank=1,
        target_id="page:one",
        target_level="page",
        page_id="page:one",
        text="Whole page",
    )
    context, _serialization = materialize_context(
        [ContextSource(item=page, origin="direct_intent")],
        status="materialized",
    )

    native, _bundle, _native_queries, _bundle_queries = (
        build_context_metric_payloads(
            [
                ContextMetricInput(
                    query_id="question:one",
                    items=tuple(context),
                    evaluable_up_to_k=20,
                    visible_character_count=5,
                    partial_item_visible=True,
                ),
            ],
            qrels=[],
            projections=[],
            question_metadata={},
            cutoffs=[1],
        )
    )

    assert native["partial_item_query_rate"] == 1.0
    assert native["at_k"]["1"]["qrel_metrics_computable"] is False
    assert native["at_k"]["1"]["hit_at_grade_2"] is None


def test_mixed_context_metrics_use_only_eligible_evaluable_queries() -> None:
    inputs = []
    qrels = []
    for query_id, returned_score, evaluable_up_to_k in (
        ("question:grade-two", 2, 1),
        ("question:strict-miss", 0, 1),
        ("question:no-relevant-target", 0, 1),
        ("question:truncated", 2, 0),
    ):
        item = _item(
            rank=1,
            target_id=f"page:{query_id}",
            target_level="page",
            page_id=f"page:{query_id}",
            text=query_id,
        )
        inputs.append(
            _metric_input(
                query_id,
                item,
                evaluable_up_to_k=evaluable_up_to_k,
            ),
        )
        qrels.append(
            QrelRecord(
                query_id=query_id,
                target_id=item.target_id,
                target_level="page",
                target_space_id="page",
                score=returned_score,
                corpus_hash=CORPUS_HASH,
                view="judged_only",
            ),
        )
    qrels.append(
        QrelRecord(
            query_id="question:strict-miss",
            target_id="page:hidden-strict-target",
            target_level="page",
            target_space_id="page",
            score=3,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        ),
    )

    native, _reference, queries, _reference_queries = (
        build_context_metric_payloads(
            inputs,
            qrels=qrels,
            projections=[],
            question_metadata={},
            cutoffs=[1],
        )
    )

    aggregate = native["at_k"]["1"]
    assert native["qrel_eligible_query_count"] == 3
    assert native["strict_qrel_eligible_query_count"] == 1
    assert aggregate["ranking_depth_complete"] is False
    assert aggregate["hit_at_grade_2"] == 0.5
    assert aggregate["strict_hit_at_grade_3"] == 0.0
    assert aggregate["metric_denominators"]["hit_at_grade_2"][
        "query_count"
    ] == 2
    assert aggregate["metric_denominators"]["strict_hit_at_grade_3"][
        "query_count"
    ] == 1
    assert queries["question:grade-two"]["at_k"]["1"][
        "strict_hit_at_grade_3"
    ] is None
    assert queries["question:no-relevant-target"]["at_k"]["1"][
        "hit_at_grade_2"
    ] is None
    assert queries["question:truncated"]["at_k"]["1"][
        "hit_at_grade_2"
    ] is None


def test_reference_anchor_metrics_combine_source_evidence_across_items() -> None:
    sources = [
        _item(
            rank=rank,
            target_id=f"page:{rank}",
            target_level="page",
            page_id=f"page:{rank}",
            text=f"Fact {rank}",
        )
        for rank in (1, 2)
    ]
    context, _serialization = materialize_context(
        [
            ContextSource(
                item=item,
                origin="retrieval",
                source_item_ids=(item.target_id,),
                source_item_ranks=(item.rank,),
            )
            for item in sources
        ],
        status="materialized",
    )
    projections = [
        _projection(
            "question:one",
            item,
            anchors=[f"span:{rank}"],
            facts=[f"fact:{rank}"],
        )
        for rank, item in enumerate(sources, start=1)
    ]
    partial_item = _item(
        rank=1,
        target_id="page:partial-projection",
        target_level="page",
        page_id="page:partial-projection",
        text="Only the first reference fact",
    )
    projections.append(
        _projection(
            "question:partial-projection",
            partial_item,
            anchors=["span:1"],
            facts=["fact:1"],
        ),
    )
    qrels = [
        QrelRecord(
            query_id="question:one",
            target_id=item.target_id,
            target_level="page",
            target_space_id="page",
            score=2,
            corpus_hash=CORPUS_HASH,
            view="judged_only",
        )
        for item in sources
    ]

    _native, reference, _native_queries, reference_queries = (
        build_context_metric_payloads(
            [
                ContextMetricInput(
                    query_id="question:one",
                    items=tuple(context),
                    evaluable_up_to_k=20,
                    visible_character_count=100,
                    partial_item_visible=False,
                ),
                _metric_input("question:partial-projection", partial_item),
            ],
            qrels=qrels,
            projections=projections,
            question_metadata={
                "question:one": {
                    "required_facts": ["Fact 1", "Fact 2"],
                    "sufficient_evidence_sets": [
                        {"target_ids": ["span:1", "span:2"]},
                    ],
                },
                "question:partial-projection": {
                    "required_facts": ["Fact 1", "Fact 2"],
                    "sufficient_evidence_sets": [
                        {"target_ids": ["span:1", "span:2"]},
                    ],
                },
            },
            cutoffs=[1, 2],
        )
    )

    query = reference_queries["question:one"]["at_k"]
    assert query["1"]["reference_anchor_set_hit"] == 0.0
    assert query["1"]["reference_anchor_fact_coverage"] == 0.5
    assert query["2"]["reference_anchor_set_hit"] == 1.0
    assert query["2"]["reference_anchor_fact_coverage"] == 1.0
    assert reference["at_k"]["2"]["reference_anchor_set_hit"] == 1.0
    partial = reference_queries["question:partial-projection"]["at_k"]["1"]
    assert partial["reference_anchor_set_hit"] is None
    assert partial["reference_anchor_fact_coverage"] is None
    assert reference["reference_anchor_set_eligible_query_count"] == 1
    assert reference["reference_anchor_fact_coverage_eligible_query_count"] == 1
