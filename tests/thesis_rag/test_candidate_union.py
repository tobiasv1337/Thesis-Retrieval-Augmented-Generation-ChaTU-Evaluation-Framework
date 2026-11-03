from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.thesis_rag.test_modern_retrieval_foundation import (
    _FakeReranker, _FakeSource, _Snapshot, _config, _document, _query,
)
from thesis_rag.retrieval.candidates import CandidateRanking, RankedCandidate
from thesis_rag.retrieval.config import ModernPipelineConfig, RerankerConfig
from thesis_rag.retrieval.fusion import ReciprocalRankFusion, candidate_union
from thesis_rag.retrieval.pipeline import ModernRetrievalPipeline, PipelineComponents
from thesis_rag.retrieval.reranker import HttpCandidateReranker
from thesis_rag.reranking import digest, load_profile


def _ranking(names, *, stage="global"):
    return CandidateRanking(
        stage_id=stage, stage_label=stage, exhausted=True,
        items=[RankedCandidate(
            candidate=_document(name), rank=rank, score=1 / rank,
            score_name="fixture", component_scores=[], transitions=[],
        ) for rank, name in enumerate(names, 1)],
    )


@pytest.mark.parametrize("depth", [25, 100])
def test_union_keeps_every_global_candidate_and_exact_priority_prefix(depth):
    global_ranking = _ranking([f"global_{i}" for i in range(100)])
    priority = _ranking([f"priority_{i}" for i in range(100)], stage="priority")
    actual = candidate_union(global_ranking, priority, priority_k=depth)
    assert [m.candidate for m in actual.matches] == [
        i.candidate for i in global_ranking.items + priority.items[:depth]
    ]
    assert len(actual.matches) == 100 + depth
    assert all(m.score is None and m.score_name == "candidate_union" for m in actual.matches)
    assert actual.exhausted == (depth == 100)
    assert len(actual.removed_reasons) == 100 - depth
    assert actual == candidate_union(global_ranking, priority, priority_k=depth)


def test_union_deduplicates_ids_without_backfilling_or_removing_global_overlap():
    actual = candidate_union(
        _ranking(["a", "b", "c"]),
        _ranking(["a", "d", "b", "e"], stage="priority"), priority_k=2,
    )
    assert [m.candidate.target_id for m in actual.matches] == [
        "chunk:a", "chunk:b", "chunk:c", "chunk:d",
    ]
    assert actual.removed_reasons == {"chunk:e": "priority_candidate_limit"}


def test_union_does_not_deduplicate_distinct_chunks_or_language_counterparts():
    first = _ranking(["a"])
    second = _ranking(["b", "c"], stage="priority")
    # One more chunk on the same Page, and its different-language counterpart.
    second = second.model_copy(update={"items": [
        item.model_copy(update={"candidate": item.candidate.model_copy(update=change)})
        for item, change in zip(second.items, [{"page_id": "page:a"}, {"language": "en"}])
    ]})
    assert len(candidate_union(first, second, priority_k=2).matches) == 3


def test_union_rejects_conflicting_evidence_payloads():
    first = _ranking(["a"])
    second = _ranking(["a"], stage="priority")
    item = second.items[0]
    second = second.model_copy(update={"items": [item.model_copy(update={
        "candidate": item.candidate.model_copy(update={"title": "changed"}),
    })]})
    with pytest.raises(ValueError, match="conflicting candidate payload"):
        candidate_union(first, second, priority_k=1)


@pytest.mark.parametrize("depth", [0, -1])
def test_union_rejects_invalid_depth(depth):
    with pytest.raises(ValueError, match="positive"):
        candidate_union(_ranking([]), _ranking([], stage="priority"), priority_k=depth)


def _pipeline_config(*, mode="union", depth=25, trace=True):
    raw = _config(reranker=True)["settings"]["pipeline"]
    raw["context"]["source_top_k"] = 20
    raw["trace_all_candidates"] = trace
    if mode != "global":
        raw["page_routing"].update(
            enabled=True, mode="fixed_pages", include_global=True,
            page_subset_manifest_path="fixture/priority_manifest.json",
        )
        raw["page_routing"].update(
            {"candidate_union": {"priority_k": depth}} if mode == "union" else
            {"fusion_weights": {"global_weight": 2, "priority_weight": 1}}
        )
    return raw


class _RecordingReranker(_FakeReranker):
    def rerank(self, query, candidates, *, limit):
        self.candidates, self.limit = candidates, limit
        return super().rerank(query, candidates, limit=limit)


class _RecordingFusion(ReciprocalRankFusion):
    def __init__(self, config):
        super().__init__(config)
        self.weights = []

    def fuse(self, rankings, *, limit, weights=None):
        self.weights.append(weights)
        return super().fuse(rankings, limit=limit, weights=weights)


def _pipeline(*, mode="union", depth=25, trace=True, routed_names=None):
    config = ModernPipelineConfig.model_validate(_pipeline_config(mode=mode, depth=depth, trace=trace))
    global_docs = [_document(f"global_{i}") for i in range(100)]
    routed_docs = [_document(name) for name in (
        routed_names if routed_names is not None else [f"priority_{i}" for i in range(100)]
    )]
    fusion = _RecordingFusion(config.content_rrf)
    fusion.build_or_load(_Snapshot())
    reranker = _RecordingReranker("fake_reranker")
    components = PipelineComponents(
        dense_global=_FakeSource("fake_dense", global_docs, list(range(100, 0, -1)), score_name="dense"),
        sparse_global=_FakeSource("fake_sparse", global_docs, list(range(100, 0, -1)), score_name="bm25"),
        content_rrf=fusion,
        page_routing=_FakeSource("future_page_routing", routed_docs,
                                list(range(len(routed_docs), 0, -1)), score_name="rrf"),
        reranker=reranker,
    )
    return ModernRetrievalPipeline(config, components)


@pytest.mark.parametrize("depth", [25, 100])
def test_pipeline_reranks_the_entire_union_and_can_promote_its_last_candidate(depth):
    pipeline = _pipeline(depth=depth)
    result = pipeline.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    reranker = pipeline.components.reranker
    assert len(reranker.candidates.items) == reranker.limit == 100 + depth
    assert pipeline.components.content_rrf.weights == [None]  # inner RRF only
    assert result.items[0].target_id == f"chunk:priority_{depth - 1}"
    assert len(result.items) == len(result.context_items) == 20
    assert not result.stage_results  # Diagnostics do not add judging targets.
    traces = {t.stage: t for t in result.stage_traces}
    assert len(traces["content_rrf"].metadata["candidate_ranking"]) == 100
    assert len(traces["parent_children"].metadata["candidate_ranking"]) == 100
    assert len(traces["page_routing"].metadata["candidate_ranking"]) == 100 + depth
    assert len(traces["rerank"].metadata["candidate_ranking"]) == 100 + depth
    assert traces["rerank"].metadata["candidate_ranking"][0]["target_id"] == result.items[0].target_id


def test_union_preserves_both_branches_in_duplicate_provenance():
    pipeline = _pipeline(routed_names=["global_99"])
    pipeline.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    candidates = pipeline.components.reranker.candidates.items
    assert len(candidates) == 100
    duplicate = next(c for c in candidates if c.candidate.target_id == "chunk:global_99")
    assert duplicate.transitions[-1].input_ranks == {"content_rrf": 100, "parent_children": 1}
    assert {s.stage_id for s in duplicate.component_scores} >= {"content_rrf", "parent_children"}


def test_empty_priority_branch_keeps_all_global_candidates():
    union, global_pipeline = _pipeline(routed_names=[]), _pipeline(mode="global")
    union_result = union.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    global_result = global_pipeline.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert len(union.components.reranker.candidates.items) == 100
    assert [(i.target_id, i.text_hash) for i in union_result.items] == [
        (i.target_id, i.text_hash) for i in global_result.items
    ]


def test_rrf_control_keeps_weighted_outer_fusion_and_original_candidate_limit():
    pipeline = _pipeline(mode="rrf")
    pipeline.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert pipeline.components.content_rrf.weights == [None, (2.0, 1.0)]
    assert len(pipeline.components.reranker.candidates.items) == 100
    assert pipeline.components.reranker.limit == 100


def test_complete_candidate_tracing_does_not_change_rankings_or_stage_pool_depth():
    plain, traced = _pipeline(trace=False), _pipeline(trace=True)
    a = plain.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    b = traced.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    assert a.items == b.items and a.context_items == b.context_items
    assert a.stage_results == b.stage_results
    assert all("candidate_ranking" not in t.metadata for t in a.stage_traces)


def test_inactive_union_does_not_change_existing_serialized_configs():
    config = ModernPipelineConfig.model_validate(_pipeline_config(mode="global", trace=False))
    assert "trace_all_candidates" not in config.to_dict()
    assert "candidate_union" not in config.to_dict()["page_routing"]
    assert config.rerank_candidate_k == config.candidate_k == 100


@pytest.mark.parametrize("change", [
    {"candidate_union": {"priority_k": 101}},
    {"include_global": False},
    {"enabled": False},
    {"fusion_weights": {"global_weight": 2, "priority_weight": 1}},
])
def test_union_rejects_ambiguous_or_unavailable_branch_settings(change):
    config = _pipeline_config()
    config["page_routing"].update(change)
    with pytest.raises(ValidationError, match="candidate union"):
        ModernPipelineConfig.model_validate(config)


def test_union_requires_a_reranker():
    config = _pipeline_config()
    config["reranker"]["enabled"] = False
    with pytest.raises(ValidationError, match="candidate union requires a reranker"):
        ModernPipelineConfig.model_validate(config)


def test_explicit_page_selection_disables_union_and_restores_normal_budget():
    from thesis_rag.service.selection import selected_page_pipeline

    pipeline = _pipeline()
    selected = selected_page_pipeline(pipeline, "page:global_0")
    assert not selected.config.page_routing.enabled
    assert selected.config.rerank_candidate_k == 100
    assert pipeline.config.rerank_candidate_k == 125


def _http_reranker(profile_id="nemotron_1b"):
    root = Path(__file__).resolve().parents[2]
    adapter = HttpCandidateReranker(RerankerConfig(
        enabled=True, component_id="http_fixture", profile_id=profile_id,
        expected_profile_hash=digest(load_profile(profile_id, root)),
        input_format="title_heading_text_v1",
    ), repo_root=root)
    adapter._identity, adapter._corpus_hash = {"test": "runtime"}, _Snapshot().corpus_hash
    return adapter


@pytest.mark.parametrize("depth", [25, 100])
def test_http_reranker_scores_all_union_candidates_with_existing_service_limit(monkeypatch, depth):
    from services.reranker.serve import RerankRequest

    pipeline = _pipeline(depth=depth)
    adapter = _http_reranker()
    requests = []

    def request(path, payload):
        assert path == "/rerank"
        RerankRequest.model_validate(payload)  # Existing GPU-service contract.
        requests.append(payload)
        offset = sum(len(r["documents"]) for r in requests[:-1])
        return {"identity_hash": digest(adapter._identity), "results": [
            {"index": i, "relevance_score": float(offset + i)}
            for i in reversed(range(len(payload["documents"])))
        ]}

    monkeypatch.setattr(adapter, "_request", request)
    pipeline.components = replace(pipeline.components, reranker=adapter)
    result = pipeline.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert [len(r["documents"]) for r in requests] == [100, depth]
    assert len({doc for r in requests for doc in r["documents"]}) == 100 + depth
    assert all(r["query"] == _query().text for r in requests)
    assert result.items[0].target_id == f"chunk:priority_{depth - 1}"
    assert result.items[0].score == 99 + depth


def test_http_reranker_rejects_model_switch_between_batches(monkeypatch):
    adapter = _http_reranker()
    requests = []

    def request(path, payload):
        requests.append(payload)
        return {"identity_hash": digest(adapter._identity) if len(requests) == 1 else "changed",
                "results": [{"index": i, "relevance_score": 1.0}
                            for i in range(len(payload["documents"]))]}

    monkeypatch.setattr(adapter, "_request", request)
    with pytest.raises(ValueError, match="runtime changed"):
        adapter.rerank(_query(), _ranking([str(i) for i in range(125)]), limit=125)
    assert len(requests) == 2


def test_listwise_models_are_not_silently_split(monkeypatch):
    adapter = _http_reranker("jina_v35")
    monkeypatch.setattr(adapter, "_request", lambda *a: pytest.fail("must reject before sending"))
    with pytest.raises(ValueError, match="listwise reranking cannot split"):
        adapter.rerank(_query(), _ranking([str(i) for i in range(125)]), limit=125)
