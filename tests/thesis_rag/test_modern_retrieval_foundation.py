from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from thesis_rag.interfaces import RagQuery, RagSystem
from thesis_rag.registry import (
    SystemConfigError,
    instantiate_system,
    resolve_system_config,
)
from thesis_rag.retrieval.candidates import (
    CandidateBatch,
    CandidateDocument,
    CandidateMatch,
    CandidateRanking,
)
from thesis_rag.retrieval.config import ModernSystemSettings
from thesis_rag.retrieval.pipeline import PipelineComponents
from thesis_rag.retrieval.protocols import ComponentRuntimeIdentity
from thesis_rag.retrieval.system import (
    ModernRetrievalSystem,
    ModernRetrievalSystemError,
)
from thesis_rag.schema import hash_jsonable, hash_text


CORPUS_HASH = hash_text("modern-foundation-corpus")


@dataclass(frozen=True)
class _Snapshot:
    corpus_hash: str = CORPUS_HASH


class _StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class _FakeComponent:
    def __init__(self, component_id: str, *, identity_corpus: str = CORPUS_HASH) -> None:
        self._component_id = component_id
        self.identity_corpus = identity_corpus
        self.build_calls = 0
        self.close_calls = 0
        self.operation_calls = 0
        self.loaded_corpus: str | None = None

    @property
    def component_id(self) -> str:
        return self._component_id

    def build_or_load(self, corpus: _Snapshot) -> None:
        self.build_calls += 1
        self.loaded_corpus = corpus.corpus_hash

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self.loaded_corpus is None:
            raise RuntimeError("fake component was not built")
        return ComponentRuntimeIdentity(
            component_id=self.component_id,
            corpus_hash=self.identity_corpus,
            identity_hashes={"fixture": hash_text(f"artifact:{self.component_id}")},
        )

    def close(self) -> None:
        self.close_calls += 1


class _FakeSource(_FakeComponent):
    def __init__(
        self,
        component_id: str,
        documents: list[CandidateDocument],
        scores: list[float],
        *,
        score_name: str,
        identity_corpus: str = CORPUS_HASH,
    ) -> None:
        super().__init__(component_id, identity_corpus=identity_corpus)
        self.documents = documents
        self.scores = scores
        self.score_name = score_name

    def search(
        self,
        query: RagQuery,
        *,
        limit: int,
        scope: object | None = None,
    ) -> CandidateBatch:
        del query, scope
        self.operation_calls += 1
        pairs = list(zip(self.documents, self.scores, strict=True))
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=document,
                    score=score,
                    score_name=self.score_name,
                    metadata={"fixture_source": self.component_id},
                )
                for document, score in pairs[:limit]
            ],
            exhausted=len(pairs) <= limit,
        )


class _FakeFusion(_FakeComponent):
    def fuse(
        self,
        rankings: list[CandidateRanking],
        *,
        limit: int,
    ) -> CandidateBatch:
        self.operation_calls += 1
        by_key: dict[tuple[str, str], tuple[CandidateDocument, float]] = {}
        for ranking in rankings:
            for item in ranking.items:
                document, score = by_key.get(item.key, (item.candidate, 0.0))
                by_key[item.key] = (document, score + 1.0 / (60 + item.rank))
        ordered = sorted(
            by_key.values(),
            key=lambda item: (-item[1], item[0].target_id),
        )
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=document,
                    score=score,
                    score_name="fake_rrf",
                    transition_reason="fixture_fusion",
                )
                for document, score in ordered[:limit]
            ],
            exhausted=len(ordered) <= limit,
        )


class _FakeReranker(_FakeComponent):
    def __init__(self, component_id: str, *, reverse: bool = True) -> None:
        super().__init__(component_id)
        self.reverse = reverse

    def rerank(
        self,
        query: RagQuery,
        candidates: CandidateRanking,
        *,
        limit: int,
    ) -> CandidateBatch:
        del query
        self.operation_calls += 1
        ordered = list(candidates.items)
        if self.reverse:
            ordered.reverse()
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=item.candidate,
                    score=float(len(ordered) - rank),
                    score_name="fake_reranker",
                    transition_reason="fixture_rerank",
                )
                for rank, item in enumerate(ordered[:limit])
            ],
            exhausted=candidates.exhausted and len(ordered) <= limit,
        )


class _MutatingReranker(_FakeReranker):
    def rerank(
        self,
        query: RagQuery,
        candidates: CandidateRanking,
        *,
        limit: int,
    ) -> CandidateBatch:
        batch = super().rerank(query, candidates, limit=limit)
        first = batch.matches[0]
        changed = first.candidate.model_copy(update={"title": "Changed title"})
        return batch.model_copy(
            update={
                "matches": [
                    first.model_copy(update={"candidate": changed}),
                    *batch.matches[1:],
                ],
            },
        )


class _FakeContextExpander(_FakeComponent):
    def expand(
        self,
        candidates: CandidateRanking,
        *,
        budget: int,
    ) -> CandidateBatch:
        self.operation_calls += 1
        assert budget == 6000
        selected = list(reversed(candidates.items[:2]))
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=item.candidate,
                    score=item.score,
                    score_name="fake_context_order",
                    transition_reason="fixture_context",
                )
                for item in selected
            ],
            exhausted=True,
        )


def _document(name: str) -> CandidateDocument:
    text = f"Evidence {name}"
    return CandidateDocument(
        target_id=f"chunk:{name}",
        target_level="chunk",
        chunk_id=f"chunk:{name}",
        chunker_id="section_preferred_800_100",
        section_id=f"section:{name}",
        overlapping_section_ids=[f"section:{name}"],
        page_id=f"page:{name}",
        source_path=f"topics/{name}/content.html",
        url=f"https://example.test/{name}",
        title=f"Title {name}",
        breadcrumbs=["Study", name],
        heading_path=[name],
        language="de",
        text=text,
        text_hash=hash_text(text),
        metadata={"intent": name},
    )


def _config(
    *,
    dense: bool = True,
    sparse: bool = True,
    prior: bool = False,
    reranker: bool = False,
    context: bool = False,
    page_routing: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "system_config_v2",
        "system_id": "modern_fixture",
        "class_path": "thesis_rag.retrieval.system.ModernRetrievalSystem",
        "settings": {
            "pipeline": {
                "representation": {
                    "chunker_id": "section_preferred_800_100",
                    "fields": ["text"],
                },
                "language_policy": "cross_lingual",
                "candidate_k": 100,
                "dense_global": {
                    "enabled": dense,
                    "component_id": "fake_dense",
                    "index_root": "outputs/thesis/indexes/fixture_dense",
                    "model": {
                        "backend": "ollama",
                        "endpoint": "http://127.0.0.1:11434",
                        "model_id": "fixture-embedding",
                        "expected_model_artifact_digest": CORPUS_HASH,
                        "dimensions": 3,
                        "batch_size": 2,
                        "timeout_seconds": 1.0,
                        "truncate": True,
                    },
                    "normalize": True,
                },
                "sparse_global": {
                    "enabled": sparse,
                    "component_id": "fake_sparse",
                    "index_root": "outputs/thesis/indexes/fixture_sparse",
                    "tokenizer_id": "unicode_words_v1",
                    "k1": 1.5,
                    "b": 0.75,
                    "remove_stopwords": True,
                    "cross_shard_rrf_k": 60,
                },
                "content_rrf": {
                    "enabled": dense and sparse,
                    "component_id": "fake_rrf",
                    "rank_constant": 60,
                },
                "page_routing": {
                    "enabled": page_routing,
                    "component_id": "future_page_routing",
                    "top_p": 10,
                },
                "intent_prior": {
                    "enabled": prior,
                    "component_id": "fake_prior",
                },
                "reranker": {
                    "enabled": reranker,
                    "component_id": "fake_reranker",
                },
                "context": {
                    "enabled": context,
                    "component_id": "fake_context",
                    "policy_id": "direct_children",
                    "source_top_k": 15,
                    "token_budget": 6000,
                },
                "config_version": "modern_pipeline_config_v3",
            },
        },
    }


def _query() -> RagQuery:
    return RagQuery(query_id="query:1", text="Wie funktioniert das?", language="de")


def _components(
    documents: list[CandidateDocument],
    *,
    reranker: _FakeReranker | None = None,
    context: _FakeContextExpander | None = None,
) -> PipelineComponents:
    return PipelineComponents(
        dense_global=_FakeSource(
            "fake_dense",
            documents,
            [0.9, 0.8, 0.7],
            score_name="dense_cosine",
        ),
        sparse_global=_FakeSource(
            "fake_sparse",
            [documents[1], documents[0], documents[2]],
            [8.0, 6.0, 1.0],
            score_name="bm25",
        ),
        content_rrf=_FakeFusion("fake_rrf"),
        reranker=reranker,
        context=context,
    )


def _build_system(
    config: dict[str, Any],
    components: PipelineComponents,
) -> ModernRetrievalSystem:
    resolved = resolve_system_config(config)
    system = instantiate_system(
        resolved,
        components=components,
        clock=_StepClock(),
    )
    assert isinstance(system, ModernRetrievalSystem)
    assert isinstance(system, RagSystem)
    system.build_or_load(_Snapshot())
    return system


def test_dialog_fuses_candidates_before_exactly_one_contextualized_rerank() -> None:
    class RecordingReranker(_FakeReranker):
        seen_query = None

        def rerank(self, query, candidates, *, limit):
            self.seen_query = query
            return super().rerank(query, candidates, limit=limit)

    documents = [_document("a"), _document("b"), _document("c")]
    reranker = RecordingReranker("fake_reranker")
    components = _components(documents, reranker=reranker)
    system = _build_system(_config(reranker=True), components)
    from thesis_rag.service.history import ChatMessage, prepare_dialog

    dialog = prepare_dialog(
        "Und dafür?", "de", (ChatMessage(role="user", content="Bewerbung Master"),),
        request_id="request:one", depth=3,
    )
    output = system.pipeline.retrieve_variants(
        dialog.searches, rerank_query=dialog.rerank, top_k=2, stage_evaluation_depth=3,
    )
    assert reranker.operation_calls == 1
    assert reranker.seen_query == dialog.rerank
    assert components.dense_global.operation_calls == 2
    assert components.sparse_global.operation_calls == 2
    assert components.content_rrf.operation_calls == 3
    assert len(output.items) == 2
    for item in output.items:
        assert item.text == next(d.text for d in documents if d.target_id == item.target_id)
        scores = item.metadata["candidate_provenance"]["component_scores"]
        assert {score["stage_id"] for score in scores} >= {
            "query_0.dense_global", "query_1.dense_global", "dialog_query_fusion", "rerank",
        }
    assert len({trace.stage for trace in output.stage_traces}) == len(output.stage_traces)


def test_single_query_dialog_keeps_historical_output_and_timings_exactly() -> None:
    documents = [_document("a"), _document("b"), _document("c")]
    left = _build_system(_config(), _components(documents))
    right = _build_system(_config(), _components(documents))
    query = _query()
    assert left.pipeline.retrieve(query, top_k=2, stage_evaluation_depth=3) == (
        right.pipeline.retrieve_variants(
            (query,), rerank_query=query, top_k=2, stage_evaluation_depth=3,
        )
    )


def test_dialog_keeps_conflicting_per_query_ranks_as_separate_observations() -> None:
    class QuerySource(_FakeSource):
        def search(self, query, *, limit, scope=None):
            batch = super().search(query, limit=limit, scope=scope)
            return batch.model_copy(update={"matches": (
                list(reversed(batch.matches)) if query.text == "second" else batch.matches
            )})

    from dataclasses import replace

    documents = [_document("a"), _document("b"), _document("c")]
    components = replace(_components(documents), dense_global=QuerySource(
        "fake_dense", documents, [0.9, 0.8, 0.7], score_name="dense_cosine",
    ))
    system = _build_system(_config(), components)
    queries = tuple(_query().model_copy(update={"text": text}) for text in ("first", "second"))
    result = system.pipeline.retrieve_variants(queries, rerank_query=_query(), top_k=3)
    item = next(item for item in result.items if item.target_id == "chunk:a")
    scores = {s["stage_id"]: s for s in item.metadata["candidate_provenance"]["component_scores"]}
    assert scores["query_0.dense_global"]["rank"] == 1
    assert scores["query_1.dense_global"]["rank"] == 3


def test_dialog_rejects_empty_queries_and_mixed_ui_languages() -> None:
    from thesis_rag.retrieval.pipeline import ModernPipelineError

    system = _build_system(_config(), _components([_document(s) for s in "abc"]))
    with pytest.raises(ModernPipelineError, match="at least one"):
        system.pipeline.retrieve_variants((), rerank_query=_query(), top_k=2)
    english = _query().model_copy(update={"language": "en"})
    with pytest.raises(ModernPipelineError, match="same UI language"):
        system.pipeline.retrieve_variants((_query(),), rerank_query=english, top_k=2)


def test_modern_config_is_strict_and_enforces_fixed_global_topology() -> None:
    valid = _config()
    settings = ModernSystemSettings.from_dict(valid["settings"])
    assert settings.pipeline.candidate_k == 100

    unknown = _config()
    unknown["settings"]["pipeline"]["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        ModernSystemSettings.from_dict(unknown["settings"])
    with pytest.raises(SystemConfigError, match="unexpected"):
        resolve_system_config(unknown)

    no_fusion = _config()
    no_fusion["settings"]["pipeline"]["content_rrf"]["enabled"] = False
    with pytest.raises(ValueError, match="two global sources require content_rrf"):
        ModernSystemSettings.from_dict(no_fusion["settings"])

    no_source = _config(dense=False, sparse=False)
    with pytest.raises(ValueError, match="requires a global candidate source"):
        ModernSystemSettings.from_dict(no_source["settings"])

    excessive_context_depth = _config()
    excessive_context_depth["settings"]["pipeline"]["context"][
        "source_top_k"
    ] = 101
    with pytest.raises(ValueError, match="less than or equal to 20"):
        ModernSystemSettings.from_dict(excessive_context_depth["settings"])

    v2_with_v3_depth = _config()
    v2_with_v3_depth["settings"]["pipeline"]["config_version"] = (
        "modern_pipeline_config_v2"
    )
    with pytest.raises(ValueError, match="v2 cannot declare context.source_top_k"):
        ModernSystemSettings.from_dict(v2_with_v3_depth["settings"])

    v3_without_depth = _config()
    del v3_without_depth["settings"]["pipeline"]["context"]["source_top_k"]
    with pytest.raises(ValueError, match="v3 requires context.source_top_k"):
        ModernSystemSettings.from_dict(v3_without_depth["settings"])


def test_v2_config_keeps_historical_unconstructed_context_semantics() -> None:
    config = _config()
    pipeline = config["settings"]["pipeline"]
    pipeline["config_version"] = "modern_pipeline_config_v2"
    del pipeline["context"]["source_top_k"]
    settings = ModernSystemSettings.from_dict(config["settings"])
    system = _build_system(
        config,
        _components([_document("a"), _document("b"), _document("c")]),
    )

    result = system.retrieve(_query(), top_k=2)

    assert "source_top_k" not in settings.pipeline.to_dict()["context"]
    assert result.context_items == []
    assert result.context_serialization.status == "not_constructed"
    assert system.pipeline.version == "modern_retrieval_pipeline_v2"


def test_fake_pipeline_is_stable_and_integrates_with_rag_system_registry() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    first = _build_system(_config(), _components(documents))
    second = _build_system(_config(), _components(documents))

    first_result = first.retrieve(_query(), top_k=3, stage_evaluation_depth=2)
    second_result = second.retrieve(_query(), top_k=3, stage_evaluation_depth=2)

    assert first.runtime_identity() == second.runtime_identity()
    assert first_result == second_result
    assert first_result.timings_ms["online_retrieval_total"] == (
        first_result.timings_ms["pipeline_total"]
    )
    assert first_result.debug["pipeline_version"] == (
        "modern_retrieval_pipeline_v3"
    )
    assert [item.target_id for item in first_result.items] == [
        "chunk:a",
        "chunk:b",
        "chunk:c",
    ]
    assert [result.stage_id for result in first_result.stage_results] == [
        "dense_global",
        "sparse_global",
        "content_rrf",
    ]


def test_stage_checkpoints_preserve_scores_ranks_ids_timings_and_transitions() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    system = _build_system(_config(), _components(documents))

    result = system.retrieve(_query(), top_k=2, stage_evaluation_depth=2)

    assert [trace.stage for trace in result.stage_traces] == [
        "dense_global",
        "sparse_global",
        "content_rrf",
        "page_routing",
        "soft_intent_prior",
        "rerank",
        "final_selection",
        "context_construction",
    ]
    assert result.stage_traces[0].timings_ms == {"total": pytest.approx(1.0)}
    assert result.stage_traces[2].rank_changes["chunk:a"] == {
        "stage_id": "content_rrf",
        "input_ranks": {"dense_global": 1, "sparse_global": 2},
        "output_rank": 1,
        "action": "retained",
        "reason": "fixture_fusion",
    }
    provenance = result.items[0].metadata["candidate_provenance"]
    assert [item["stage_id"] for item in provenance["component_scores"]] == [
        "dense_global",
        "sparse_global",
        "content_rrf",
    ]
    assert [item["score_name"] for item in provenance["component_scores"]] == [
        "dense_cosine",
        "bm25",
        "fake_rrf",
    ]
    assert provenance["rank_transitions"][-1]["stage_id"] == "final_selection"
    assert result.items[0].metadata["breadcrumbs"] == ["Study", "a"]
    assert result.items[0].rank == 1
    assert result.items[0].target_id == "chunk:a"


def test_disabled_optional_components_are_not_built_or_called_or_hashed() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    unused_reranker = _FakeReranker("fake_reranker")
    with_unused = _build_system(
        _config(),
        _components(
            documents,
            reranker=unused_reranker,
        ),
    )
    without_unused = _build_system(_config(), _components(documents))

    with_result = with_unused.retrieve(_query(), top_k=3, stage_evaluation_depth=2)
    without_result = without_unused.retrieve(
        _query(),
        top_k=3,
        stage_evaluation_depth=2,
    )

    assert with_result == without_result
    assert with_unused.runtime_identity() == without_unused.runtime_identity()
    assert unused_reranker.build_calls == unused_reranker.operation_calls == 0


def test_optional_reranker_changes_only_ranks_and_keeps_prior_provenance() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    reranker = _FakeReranker("fake_reranker")
    system = _build_system(
        _config(reranker=True),
        _components(documents, reranker=reranker),
    )

    result = system.retrieve(_query(), top_k=3, stage_evaluation_depth=3)

    assert [item.target_id for item in result.items] == [
        "chunk:c",
        "chunk:b",
        "chunk:a",
    ]
    assert {item.target_id for item in result.items} == {
        document.target_id for document in documents
    }
    reranked = result.stage_results[-1]
    assert reranked.stage_id == "rerank"
    provenance = reranked.items[0].metadata["candidate_provenance"]
    assert [item["stage_id"] for item in provenance["component_scores"]] == [
        "dense_global",
        "sparse_global",
        "content_rrf",
        "rerank",
    ]
    assert provenance["rank_transitions"][-1]["action"] == "promoted"


def test_checkpoint_rejects_a_component_that_mutates_candidate_identity_payload() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    system = _build_system(
        _config(reranker=True),
        _components(documents, reranker=_MutatingReranker("fake_reranker")),
    )

    with pytest.raises(ValueError, match="changed stable candidate payload"):
        system.retrieve(_query(), top_k=3, stage_evaluation_depth=3)


def test_context_hook_does_not_change_the_final_child_ranking() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    baseline = _build_system(_config(), _components(documents))
    expanded = _build_system(
        _config(context=True),
        _components(documents, context=_FakeContextExpander("fake_context")),
    )

    baseline_result = baseline.retrieve(_query(), top_k=3)
    expanded_result = expanded.retrieve(_query(), top_k=3)

    assert [item.target_id for item in expanded_result.items] == [
        item.target_id for item in baseline_result.items
    ]
    assert [item.target_id for item in baseline_result.context_items] == [
        item.target_id for item in baseline_result.items
    ]
    assert all(item.origin == "retrieval" for item in baseline_result.context_items)
    assert [item.target_id for item in expanded_result.context_items] == [
        "chunk:b",
        "chunk:a",
    ]
    assert all(
        item.origin == "retrieval" for item in expanded_result.context_items
    )


def test_context_source_top_k_is_independent_from_published_items_top_k() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    config = _config()
    config["settings"]["pipeline"]["context"]["source_top_k"] = 3
    system = _build_system(config, _components(documents))

    result = system.retrieve(_query(), top_k=2)

    assert [item.target_id for item in result.items] == ["chunk:a", "chunk:b"]
    assert [item.target_id for item in result.context_items] == [
        "chunk:a",
        "chunk:b",
        "chunk:c",
    ]
    assert [item.source_item_ranks for item in result.context_items] == [
        [1],
        [2],
        [3],
    ]
    assert result.context_serialization.metadata["source_top_k"] == 3


def test_context_formatter_is_opt_in_and_does_not_change_retrieval() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    baseline = _build_system(_config(), _components(documents))
    config = _config()
    config["settings"]["pipeline"]["context"]["block_format"] = "metadata_chunk_v1"
    formatted = _build_system(config, _components(documents))
    original = baseline.retrieve(_query(), top_k=3)
    result = formatted.retrieve(_query(), top_k=3)

    assert result.items == original.items
    assert result.stage_results == original.stage_results
    assert result.stage_traces == original.stage_traces
    assert result.run_id != original.run_id
    assert result.config_hash != original.config_hash
    assert result.corpus_hash == original.corpus_hash
    assert result.context_serialization.metadata["expansion_enabled"] is False
    assert result.context_items[0].text == (
        "Title: Title a\nBreadcrumbs: Study > a\nSection: a\n\nContent:\nEvidence a"
    )
    assert result.context_items[0].source_item_ids == ["chunk:a"]
    assert result.context_items[0].metadata["context_rendering"]["source_text_hash"] == (
        result.items[0].text_hash
    )
    assert formatted.settings.pipeline.to_dict()["context"]["block_format"] == (
        "metadata_chunk_v1"
    )
    assert "block_format" not in baseline.settings.pipeline.to_dict()["context"]
    baseline.close()
    formatted.close()


@pytest.mark.parametrize(("version", "expected_result_hash"), [
    ("modern_pipeline_config_v2", "sha256:52e857ab4f38a3ca09311c413e1822ba286b488b336b182b32ef53994b0150dd"),
    ("modern_pipeline_config_v3", "sha256:6158bc6fe279735de376286577409409b8ac1f8cb8d461d2c5bd240d4677739f"),
])
def test_plain_context_preserves_historical_result_hashes(
    version: str, expected_result_hash: str,
) -> None:
    # Captured with the fixed clock before adding the opt-in formatter.
    config = _config()
    pipeline = config["settings"]["pipeline"]
    pipeline["config_version"] = version
    if version == "modern_pipeline_config_v2":
        del pipeline["context"]["source_top_k"]
    system = _build_system(config, _components([_document(n) for n in ("a", "b", "c")]))
    assert hash_jsonable(system.retrieve(_query(), top_k=3).to_dict()) == expected_result_hash
    system.close()


def test_metadata_formatter_rejects_unknown_format_and_unmaterialized_v2() -> None:
    config = _config()
    pipeline = config["settings"]["pipeline"]
    pipeline["context"]["block_format"] = "typo"
    with pytest.raises(ValueError, match="block_format"):
        ModernSystemSettings.from_dict(config["settings"])
    pipeline["context"]["block_format"] = "metadata_chunk_v1"
    pipeline["config_version"] = "modern_pipeline_config_v2"
    del pipeline["context"]["source_top_k"]
    with pytest.raises(ValueError, match="formatting requires modern pipeline config v3"):
        ModernSystemSettings.from_dict(config["settings"])


def test_build_rejects_component_identity_from_another_corpus() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    components = _components(documents)
    dense = components.dense_global
    assert isinstance(dense, _FakeSource)
    dense.identity_corpus = hash_text("another-corpus")
    system = ModernRetrievalSystem(_config(), components=components)

    with pytest.raises(ModernRetrievalSystemError, match="another corpus"):
        system.build_or_load(_Snapshot())


def test_modern_system_warms_both_language_paths_outside_runner_measurements() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    components = _components(documents)
    system = _build_system(_config(), components)

    system.warm_up()

    assert components.dense_global is not None
    assert components.sparse_global is not None
    assert components.content_rrf is not None
    assert components.dense_global.operation_calls == 2
    assert components.sparse_global.operation_calls == 2
    assert components.content_rrf.operation_calls == 2


def test_build_rejects_wrong_slot_protocol_before_loading_components() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    components = _components(documents)
    lifecycle_only = _FakeComponent("fake_dense")
    components = PipelineComponents(
        dense_global=lifecycle_only,  # type: ignore[arg-type]
        sparse_global=components.sparse_global,
        content_rrf=components.content_rrf,
    )
    system = ModernRetrievalSystem(_config(), components=components)

    with pytest.raises(ModernRetrievalSystemError, match="slot protocols: dense"):
        system.build_or_load(_Snapshot())

    assert lifecycle_only.build_calls == 0
    assert components.sparse_global is not None
    assert components.sparse_global.build_calls == 0


def test_build_rejects_component_id_mismatch_before_loading_components() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    components = _components(documents)
    mismatched_dense = _FakeSource(
        "another_dense",
        documents,
        [0.9, 0.8, 0.7],
        score_name="dense_cosine",
    )
    components = PipelineComponents(
        dense_global=mismatched_dense,
        sparse_global=components.sparse_global,
        content_rrf=components.content_rrf,
    )
    system = ModernRetrievalSystem(_config(), components=components)

    with pytest.raises(ModernRetrievalSystemError, match="expected 'fake_dense'"):
        system.build_or_load(_Snapshot())

    assert mismatched_dense.build_calls == 0
    assert components.sparse_global is not None
    assert components.sparse_global.build_calls == 0


@pytest.mark.parametrize("depth", [0, -1])
def test_retrieve_rejects_non_positive_stage_evaluation_depth(depth: int) -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    system = _build_system(_config(), _components(documents))

    with pytest.raises(ModernRetrievalSystemError, match="greater than zero"):
        system.retrieve(_query(), top_k=3, stage_evaluation_depth=depth)


def test_page_routing_requires_explicit_parent_manifest() -> None:
    documents = [_document(name) for name in ("a", "b", "c")]
    with pytest.raises(ModernRetrievalSystemError, match="manifest_path"):
        ModernRetrievalSystem(_config(page_routing=True), components=_components(documents))


def test_retrieval_modules_have_no_evaluation_or_qrel_dependencies() -> None:
    root = Path(__file__).parents[2] / "thesis_rag" / "retrieval"
    forbidden_imports: list[tuple[str, str]] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module or ""
            ).startswith("thesis_rag.eval"):
                forbidden_imports.append((path.name, node.module or ""))
            elif isinstance(node, ast.Import):
                forbidden_imports.extend(
                    (path.name, alias.name)
                    for alias in node.names
                    if alias.name.startswith("thesis_rag.eval")
                )

    assert forbidden_imports == []
