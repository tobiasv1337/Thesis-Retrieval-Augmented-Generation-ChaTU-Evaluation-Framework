"""Complete short lists remain evaluable after deduplication, without guessing."""

from types import SimpleNamespace as NS

import pytest

from thesis_rag.adapters.legacy_baseline import LegacyBaselineSystem
from thesis_rag.eval.relevance import (
    _configured_raw_rankings,
    _verified_legacy_stage_completeness,
    _without_retrieval_traces,
)
from thesis_rag.eval.relevance_config import load_relevance_config
from thesis_rag.eval.suite_config import FinalEvaluationPolicy, StageEvaluationPolicy
from thesis_rag.interfaces import RetrievedItem, RetrievalResult, StageCandidate, StageResult, StageTrace
from thesis_rag.schema import hash_text, validate_record_shape


def _candidate(index, rank):
    return StageCandidate(
        rank=rank, target_id=f"legacy_chunk:{index}", target_level="chunk",
        chunk_id=f"legacy_chunk:{index}", chunker_id="legacy_recursive_character_800_100_min200",
        section_id=None, page_id="page:one", score=float(index), score_name="faiss_l2_distance",
    )


@pytest.mark.parametrize("count,duplicates,total,unmapped,complete,evaluable", [
    (9, 1, 10, False, True, 20),
    (13, 1, 14, False, True, 20),
    (18, 2, 20, False, True, 20),
    (9, 1, 30, False, False, 9),  # Only a prefix was retained.
    (20, 1, 21, False, False, 20),  # The adapter's depth limit cuts the raw list.
    (9, 0, 10, True, False, None),  # Missing identity is not a duplicate.
    (0, 0, 0, False, True, 20),
])
def test_adapter_tracks_exhaustion_before_deduplication(
    count, duplicates, total, unmapped, complete, evaluable,
):
    system = object.__new__(LegacyBaselineSystem)
    system.settings = NS(mode="legacy_refreshed_nlu")
    system.original_oracle = None
    system._stage_candidate = lambda candidate, rank: (
        _candidate(candidate["id"], rank) if candidate["id"] is not None else None
    )
    candidates = [{"id": i} for i in range(count)] + [{"id": 0}] * duplicates
    if unmapped:
        candidates.append({"id": None})
    stage = system._stage_result({
        "stage": "variant_result", "input_count": total, "output_count": total,
        "candidates": candidates, "metadata": {"variant_id": "current"},
    }, stage_index=5, requested_depth=20)
    assert stage.ranking_complete is complete
    assert stage.evaluable_up_to_k == evaluable
    assert stage.output_count == (count if complete else total)
    assert stage.metadata["raw_output_count"] == total
    assert stage.metadata["unmapped_candidates_removed"] == int(unmapped)
    assert stage.metadata["duplicate_candidates_removed"] == (duplicates if len(candidates) <= 20 else 0)
    validate_record_shape(stage.to_dict())


def _historical_result(count=9, duplicates=1):
    stage_items = [_candidate(i, i + 1) for i in range(count)]
    items = [RetrievedItem(
        **item.model_dump(exclude={"metadata"}), overlapping_section_ids=[],
        source_path="fixture.html", url=None, title=None, heading_path=[],
        text="Evidence", text_hash=hash_text("Evidence"),
    ) for item in stage_items]
    stage = StageResult(
        stage_id="post_filters", stage_label="After filters", stage_index=5,
        variant_id="current", input_count=30, output_count=count + duplicates,
        requested_depth=20, evaluable_up_to_k=count, ranking_complete=False,
        items=stage_items, metadata={
            "raw_legacy_stage": "variant_result", "trace_origin": "reconstructed_replay",
            "duplicate_candidates_removed": duplicates,
        },
    )
    traces = [StageTrace(
        stage="variant_result", enabled=True, mode="legacy", input_count=30,
        output_count=count + duplicates, metadata={
            "variant_id": "current", "candidate_prefix_complete": True,
            "candidate_count": count + duplicates,
            "candidate_sample": [{} for _ in range(count + duplicates)],
        },
    ), StageTrace(
        stage="legacy_original_oracle_parity", enabled=True, mode="legacy_refreshed_nlu",
        input_count=count, output_count=count, metadata={
            "status": "verified", "differing_rank_count": 0,
            "compared_item_count": count, "matched_item_count": count,
            "replay_item_count": count,
        },
    )]
    return RetrievalResult(
        query_id="question:one", system_id="legacy_refreshed_nlu", run_id="run:one",
        items=items, context_items=[], direct_context_items=[], enabled_stages={},
        stage_traces=traces, stage_results=[stage], timings_ms={},
        config_hash=hash_text("config"), corpus_hash=hash_text("corpus"),
        debug={"legacy_original_oracle": {"capture": {
            "capture_id": "capture:one",
            "items": [{"legacy_chunk_id": item.target_id} for item in items],
        }}},
    )


@pytest.mark.parametrize("count,duplicates", [(9, 1), (13, 1), (18, 2)])
@pytest.mark.parametrize("compact", [False, True])
def test_historical_complete_list_is_recovered_only_for_metrics(count, duplicates, compact):
    result = _historical_result(count, duplicates)
    original = result.to_dict()
    reduced = _without_retrieval_traces(result)
    assert result.to_dict() == original
    assert reduced.stage_results == result.stage_results
    proof = reduced.debug["verified_legacy_stage_completeness"]["stage:post_filters@current"]
    assert proof["recorded_evaluable_up_to_k"] == count
    assert proof["effective_evaluable_up_to_k"] == 20
    validate_record_shape(reduced.to_dict())
    selection = NS(experiment=NS(
        top_k=20, final_evaluation=FinalEvaluationPolicy(
            streams=["items"], target_levels=["chunk"], include_in_pool=True, compute_metrics=True,
        ), stages=[StageEvaluationPolicy(
            stage_id="post_filters", variant_id="current", target_levels=["chunk"],
            depth=20, include_in_pool=True, compute_metrics=True,
        )],
    ))
    for purpose, expected in [("pool", count), ("metrics", 20)]:
        rankings = _configured_raw_rankings(
            reduced if compact else result, load_relevance_config("configs/thesis/relevance.yaml"),
            selection=selection, purpose=purpose,
        )
        checkpoint = next(r for r in rankings if r.stream == "stage:post_filters@current")
        assert checkpoint.evaluable_up_to_k == expected
        assert checkpoint.ranking_complete is (purpose == "metrics")
        assert list(checkpoint.items) == result.stage_results[0].items


@pytest.mark.parametrize("missing_proof", [
    "capture", "capture_order", "final_order", "parity", "parity_count", "raw_count",
    "prefix", "sample", "variant", "removal_count", "truncation",
])
def test_historical_incomplete_or_unverified_rankings_stay_incomplete(missing_proof):
    result = _historical_result()
    stage = result.stage_results[0]
    trace, parity = result.stage_traces
    if missing_proof == "capture":
        result.debug.clear()
    elif missing_proof == "capture_order":
        result.debug["legacy_original_oracle"]["capture"]["items"].reverse()
    elif missing_proof == "final_order":
        result.items.reverse()
    elif missing_proof == "parity":
        parity.metadata["status"] = "mismatch"
    elif missing_proof == "parity_count":
        parity.metadata["matched_item_count"] -= 1
    elif missing_proof == "raw_count":
        trace.metadata["candidate_count"] -= 1
    elif missing_proof == "prefix":
        trace.metadata["candidate_prefix_complete"] = False
    elif missing_proof == "sample":
        trace.metadata["candidate_sample"].pop()
    elif missing_proof == "variant":
        trace.metadata["variant_id"] = "keywords"
    elif missing_proof == "removal_count":
        stage.metadata["duplicate_candidates_removed"] = 0
    elif missing_proof == "truncation":
        result.stage_results[0] = stage.model_copy(update={"output_count": 30})
        stage.metadata["duplicate_candidates_removed"] = 21
    assert _verified_legacy_stage_completeness(result) == {}


@pytest.mark.parametrize("oracle", [None, {}, {"capture": None}])
def test_ablation_without_original_capture_can_enter_pool(oracle):
    result = _historical_result()
    result.debug["legacy_original_oracle"] = oracle
    reduced = _without_retrieval_traces(result)
    assert reduced.items == result.items
    assert "verified_legacy_stage_completeness" not in reduced.debug


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("depth", [2, 4])
def test_duplicate_native_slots_preserve_rank_costs(compact, depth):
    from thesis_rag.eval.relevance import _ranked_dcg

    result = _historical_result(count=3)
    ids = [item.target_id for item in result.items]
    native_ids = [ids[0], ids[0], ids[1], ids[2]]
    result.debug["legacy_original_oracle"]["capture"]["items"] = [
        {"legacy_chunk_id": target} for target in native_ids]
    result.debug["legacy_item_ranking"] = {
        "policy": "preserve_native_duplicate_slots_v1", "requested_depth": 4,
        "target_ids": native_ids,
    }
    result = result.model_copy(update={"stage_results": [StageResult(
        stage_id="final", stage_label="Final retrieval", stage_index=6,
        variant_id=None,
        input_count=3, output_count=3, requested_depth=4, evaluable_up_to_k=4,
        ranking_complete=True, items=[_candidate(i, i + 1) for i in range(3)],
    )]})
    selection = NS(experiment=NS(
        top_k=depth, final_evaluation=FinalEvaluationPolicy(
            streams=["items"], target_levels=["chunk"], include_in_pool=True, compute_metrics=True,
        ), stages=[StageEvaluationPolicy(
            stage_id="final", target_levels=["chunk"], depth=depth,
            include_in_pool=True, compute_metrics=True,
        )],
    ))
    if compact:
        result = _without_retrieval_traces(result)
    rankings = _configured_raw_rankings(
        result, load_relevance_config("configs/thesis/relevance.yaml"), selection=None,
        purpose="metrics",
    )
    assert [item.rank for item in rankings[0].items] == [1, 3, 4]
    # The first relevant chunk at native rank 3 must not become a Hit@2.
    assert ids[1] not in {item.target_id for item in rankings[0].items if item.rank <= 2}
    assert _ranked_dcg(rankings[0].items, [0, 3, 0]) == pytest.approx(7 / 2)
    for purpose in ("pool", "metrics"):
        rankings = _configured_raw_rankings(
            result, load_relevance_config("configs/thesis/relevance.yaml"),
            selection=selection, purpose=purpose,
        )
        assert {r.stream for r in rankings} == {"items", "stage:final"}
        for ranking in rankings:
            assert [item.rank for item in ranking.items] == ([1] if depth == 2 else [1, 3, 4])


@pytest.mark.parametrize("mismatch", ["order", "capture", "policy"])
def test_native_slot_metadata_must_match_stored_results(mismatch):
    from thesis_rag.eval.relevance import RelevanceError

    result = _historical_result(count=2)
    ids = [item.target_id for item in result.items]
    result.debug["legacy_item_ranking"] = {
        "policy": "preserve_native_duplicate_slots_v1", "requested_depth": 2,
        "target_ids": ids.copy(),
    }
    if mismatch == "order":
        result.debug["legacy_item_ranking"]["target_ids"].reverse()
    elif mismatch == "capture":
        result.debug["legacy_original_oracle"]["capture"]["items"].reverse()
    else:
        result.debug["legacy_item_ranking"]["policy"] = "unknown"
    with pytest.raises(RelevanceError, match="native legacy item ranking"):
        _without_retrieval_traces(result)
