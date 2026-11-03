import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from tests.thesis_rag.test_service_runtime import _runtime
from tests.thesis_rag.test_service_rewrite import _runtime as _rewrite_runtime
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateMatch
from thesis_rag.service.history import ChatMessage
from thesis_rag.service.api import create_app
from thesis_rag.service.runtime import QueryRequest
from thesis_rag.service.selection import PageSelection, SelectionError


def _scopable(runtime, monkeypatch):
    calls = []
    components = runtime.retriever.pipeline.components
    docs = components.dense_global.documents
    runtime.retriever.page_titles = {doc.page_id: doc.title or doc.page_id for doc in docs}
    for source in (components.dense_global, components.sparse_global):
        def search(query, *, limit, scope=None, source=source):
            calls.append((source.component_id, query.text, scope))
            return CandidateBatch(matches=[
                CandidateMatch(candidate=doc, score=score, score_name=source.score_name)
                for doc, score in zip(source.documents, source.scores)
                if scope is None or scope.allows(doc)
            ][:limit], exhausted=True)
        monkeypatch.setattr(source, "search", search)
    return docs, calls


def _selection(runtime, page_id):
    return PageSelection(page_id=page_id, pipeline_id=runtime.pipeline_id, original_request_id="original")


def test_api_rejects_stale_page_choice_before_model_calls():
    model_calls = []
    runtime = _runtime(model_calls)
    with TestClient(create_app(Path("unused"), loader=lambda: runtime)) as client:
        result = client.post("/v1/answer", json={
            "question": "A?", "language": "de",
            "selection": {"page_id": "unknown", "pipeline_id": "old", "original_request_id": "one"},
        })
        assert result.status_code == 422 and model_calls == []
        assert result.headers["X-RAG-Error"] == "selection"


@pytest.mark.parametrize("routing", ["none", "main_nlu_hard", "main_nlu_additive", "parent"])
def test_explicit_page_search_is_scoped_before_top_k_and_bypasses_inferred_routing(routing, monkeypatch):
    async def run():
        runtime = _runtime()
        docs, calls = _scopable(runtime, monkeypatch)
        pipeline = runtime.retriever.pipeline
        def forbidden(*a, **kw):
            pytest.fail("an inferred router ran during explicit page selection")
        if routing.startswith("main_nlu"):
            from thesis_rag.retrieval.config import MainNluRoutingConfig
            pipeline.config = pipeline.config.model_copy(update={"main_nlu_routing": MainNluRoutingConfig(
                enabled=True, component_id="nlu", mode=routing.removeprefix("main_nlu_"),
            )})
            pipeline.components = replace(pipeline.components, main_nlu_routing=SimpleNamespace(route=forbidden))
        elif routing == "parent":
            pipeline.config = pipeline.config.model_copy(update={"page_routing": pipeline.config.page_routing.model_copy(
                update={"enabled": True, "include_global": False},
            )})
            pipeline.components = replace(pipeline.components, page_routing=SimpleNamespace(search=forbidden))
        original_config = pipeline.config.to_dict()
        try:
            result = await runtime.query(QueryRequest(
                question="Frist?", language="de", diagnostics=True,
                selection=_selection(runtime, docs[-1].page_id),
                history=[ChatMessage(role="user", content="Application")],
            ), answer=True)
            assert result.items and all(item.page_id == docs[-1].page_id for item in result.items)
            assert len(calls) == 4  # Dense and BM25 for both dialog variants.
            assert all(scope.page_ids == [docs[-1].page_id] for _, _, scope in calls)
            assert len([s for s in result.diagnostics["stages"] if s["stage"] == "rerank"]) == 1
            assert result.selection.original_request_id == "original"
            assert pipeline.config.to_dict() == original_config
        finally:
            await runtime.close()
    asyncio.run(run())


def test_selection_does_not_leak_into_parallel_or_next_unscoped_request(monkeypatch):
    async def run():
        runtime = _runtime()
        docs, _ = _scopable(runtime, monkeypatch)
        try:
            selected, plain = await asyncio.gather(
                runtime.query(QueryRequest(question="A?", language="de", selection=_selection(runtime, docs[0].page_id))),
                runtime.query(QueryRequest(question="B?", language="en")),
            )
            assert {item.page_id for item in selected.items} == {docs[0].page_id}
            assert {item.page_id for item in plain.items} == {doc.page_id for doc in docs}
            assert plain.selection is None
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["page", "pipeline"])
def test_invalid_or_stale_selection_makes_no_retrieval_or_model_request(invalid, monkeypatch):
    async def run():
        requests = []
        runtime = _runtime(requests)
        docs, calls = _scopable(runtime, monkeypatch)
        selection = _selection(runtime, docs[0].page_id).model_copy(update={
            "page_id" if invalid == "page" else "pipeline_id": "unavailable",
        })
        try:
            with pytest.raises(SelectionError):
                await runtime.query(QueryRequest(question="A?", language="de", selection=selection), answer=True)
            assert not calls and not requests
        finally:
            await runtime.close()
    asyncio.run(run())


def test_source_violating_selection_fails_closed(monkeypatch):
    async def run():
        runtime = _runtime()
        docs = runtime.retriever.pipeline.components.dense_global.documents
        runtime.retriever.page_titles = {doc.page_id: doc.title for doc in docs}
        # Historical fixture ignores scope deliberately: the service must reject it.
        try:
            with pytest.raises(SelectionError, match="outside"):
                await runtime.query(QueryRequest(question="A?", language="de", selection=_selection(runtime, docs[0].page_id)))
        finally:
            await runtime.close()
    asyncio.run(run())


def test_empty_selected_search_does_not_fall_back_to_other_pages(monkeypatch):
    async def run():
        runtime = _runtime()
        docs, calls = _scopable(runtime, monkeypatch)
        for source in (runtime.retriever.pipeline.components.dense_global,
                       runtime.retriever.pipeline.components.sparse_global):
            search = source.search
            def empty(query, *, limit, scope=None, search=search):
                result = search(query, limit=limit, scope=scope)
                return CandidateBatch(matches=[], exhausted=True) if scope is not None else result
            monkeypatch.setattr(source, "search", empty)
        try:
            selected = await runtime.query(QueryRequest(
                question="A?", language="de", selection=_selection(runtime, docs[0].page_id),
            ))
            assert selected.items == []
            assert len(calls) == 2 and all(scope is not None for _, _, scope in calls)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_selected_topic_informs_rewrite_and_generation_without_persistent_scope(monkeypatch):
    async def run():
        rewrites, answers = [], []
        runtime = _rewrite_runtime(rewrites, answers)
        docs, calls = _scopable(runtime, monkeypatch)
        history = [ChatMessage(role="user", content="Application", selected_page_id=docs[0].page_id)]
        try:
            result = await runtime.query(QueryRequest(question="When?", language="en", history=history), answer=True)
            assert "Previously selected page" in str(rewrites)
            assert "Previously selected page" in str(answers)
            assert history[0].content == "Application"
            assert all(scope is None for _, _, scope in calls)
            assert result.selection is None
            assert all(set(m) == {"role", "content"} for m in answers[0]["messages"])
        finally:
            await runtime.close()
    asyncio.run(run())
