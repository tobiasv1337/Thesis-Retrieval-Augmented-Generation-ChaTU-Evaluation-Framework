from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import httpx
import pytest
from typer.testing import CliRunner

from tests.thesis_rag.test_modern_retrieval_foundation import (
    _build_system, _components, _config, _document,
)
from thesis_rag.service.api import create_app
from thesis_rag.service.cli import app
from thesis_rag.service.config import GenerationConfig, RuntimeConfig, load_config
from thesis_rag.service.capacity import CapacityError
from thesis_rag.service.generation import GenerationClient, GenerationError
from thesis_rag.service.history import ChatMessage
from thesis_rag.service.prompt import PromptBudgetError, prepare_prompt
from thesis_rag.service.runtime import QueryRequest, RagRuntime
from thesis_rag.schema import hash_text
from tests.thesis_rag.token_budget_support import byte_token_counter


def _runtime(requests=None, *, delay=0, generation_depth=None, title_mapping=None):
    requests = requests if requests is not None else []

    async def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        await asyncio.sleep(delay)
        return httpx.Response(200, json={"generate": payload["messages"][-1]["content"]})

    config = RuntimeConfig(
        system_config="unused", top_k=3, context_top_k=2,
        generation=GenerationConfig(provider="legacy", history_depth=generation_depth),
    )
    system = _build_system(_config(), _components([_document(letter) for letter in "abc"]))
    return RagRuntime(config, system, GenerationClient(config.generation, transport=httpx.MockTransport(respond)),
                      title_mapping=title_mapping, token_counter=byte_token_counter())


def test_inspection_makes_no_generation_call_and_preserves_original_chunks():
    async def run():
        requests = []
        runtime = _runtime(requests)
        try:
            result = await runtime.query(QueryRequest(question="Frist?", language="de", diagnostics=True))
            assert requests == []
            assert result.answer is None
            assert result.status == "retrieved"
            assert result.diagnostics["context_status"] == "prepared"
            assert "Title: " in result.diagnostics["context"]
            assert len(result.context_target_ids) == 2
            assert all(item.text.startswith("Evidence ") for item in result.items)
            assert "[S1]" not in result.diagnostics["context"]
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("diagnostics", [False, True])
def test_answer_sends_the_once_prepared_prompt(monkeypatch, diagnostics):
    prepared = []

    def prepare(*args, **kwargs):
        result = prepare_prompt(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr("thesis_rag.service.runtime.prepare_prompt", prepare)

    async def run():
        requests = []
        runtime = _runtime(requests)
        try:
            result = await runtime.query(
                QueryRequest(question="Frist?", language="de", diagnostics=diagnostics), answer=True,
            )
            assert len(prepared) == 1
            assert len(requests) == 1
            assert requests[0]["messages"] == prepared[0].messages
            assert result.generation["prompt_hash"] == prepared[0].prompt_hash
            assert result.generation["input_token_count"] == prepared[0].input_token_count
            assert result.timings_ms["answer_context_rebuild"] == 0.0
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("returned", [20, 50])
def test_twenty_original_chunks_enter_generation_independently_of_returned_hits(returned):
    async def run():
        requests = []
        runtime = _runtime(requests)
        runtime.config = runtime.config.model_copy(update={"top_k": returned, "context_top_k": RuntimeConfig(system_config="unused").context_top_k})
        runtime.retriever.config = runtime.config
        docs = [_document(str(index)) for index in range(returned)]
        docs = [doc.model_copy(update={"text": f"Evidence <{index}>", "text_hash": hash_text(f"Evidence <{index}>")})
                for index, doc in enumerate(docs)]
        components = _components(docs)
        components.dense_global.scores = [1.0] * returned
        components.sparse_global.documents = docs
        components.sparse_global.scores = [1.0] * returned
        runtime.retriever.pipeline.components = components
        try:
            result = await runtime.query(QueryRequest(question="Evidence?", language="en"), answer=True)
            assert len(result.items) == returned
            assert len(result.context_target_ids) == 20
            messages = str(requests[0]["messages"])
            assert all(item.text in messages for item in result.items[:20])
            assert all(item.text not in messages for item in result.items[20:])
        finally:
            await runtime.close()
    asyncio.run(run())


def test_generation_can_see_more_history_than_fusion_retrieval():
    async def run():
        requests = []
        runtime = _runtime(requests, generation_depth=6)
        history = [ChatMessage(role=role, content=f"{role} {turn}")
                   for turn in range(7) for role in ("user", "assistant")]
        try:
            result = await runtime.query(QueryRequest(question="And?", language="en", history=history,
                                                     diagnostics=True), answer=True)
            assert len(result.diagnostics["query_variants"]) == 4
            assert result.diagnostics["history"]["retrieval"] == [m.to_dict() for m in history[8::2]]
            assert result.diagnostics["history"]["generation"] == [m.to_dict() for m in history[2:]]
            assert "user 3" not in result.diagnostics["rerank_query"]
            assert "user 4" in result.diagnostics["rerank_query"]
            assert "assistant 1" in str(requests[0]["messages"])
            assert "assistant 0" not in str(requests[0]["messages"])
        finally:
            await runtime.close()
    asyncio.run(run())


def test_concurrent_requests_keep_history_questions_and_answers_isolated():
    async def run():
        requests = []
        runtime = _runtime(requests, delay=0.01)
        try:
            left, right = await asyncio.gather(*[
                runtime.query(QueryRequest(
                    request_id=f"request:{topic}", question=f"Question {topic}?", language="en",
                    history=[ChatMessage(role="user", content=f"History {topic}")], diagnostics=True,
                ), answer=True) for topic in ("one", "two")
            ])
            assert left.request_id == "request:one" and right.request_id == "request:two"
            assert left.answer == "Question: Question one?"
            assert right.answer == "Question: Question two?"
            for response, own, other in ((left, "one", "two"), (right, "two", "one")):
                assert response.diagnostics["context_status"] == "sent"
                assert f"History {own}" in response.diagnostics["rerank_query"]
                assert f"History {other}" not in str(response.diagnostics)
            assert len(requests) == 2
        finally:
            await runtime.close()
    asyncio.run(run())


def test_prompt_budget_keeps_complete_chunks_and_reports_omissions():
    from thesis_rag.retrieval.candidates import RankedCandidate

    def item(letter):
        return RankedCandidate(candidate=_document(letter), rank=1, score=1.0, score_name="test",
                               component_scores=[], transitions=[]).to_retrieved_item()
    items = (item("a"), item("b"))
    history = (ChatMessage(role="user", content="old " * 3000),)
    config = GenerationConfig(context_window=4096)
    result = prepare_prompt("Frist?", "de", history, items, config, token_counter=byte_token_counter())
    assert result.dropped_history_messages == 1
    assert result.context[0].text.endswith(items[0].text)
    assert result.input_token_bound + config.max_tokens <= config.context_window
    assert result.prompt_hash == prepare_prompt("Frist?", "de", history, items, config,
                                                token_counter=byte_token_counter()).prompt_hash
    with pytest.raises(ValueError, match="budget"):
        prepare_prompt("x" * 10000, "de", (), items, config, token_counter=byte_token_counter())


def test_api_lifespan_reuses_runtime_and_rejects_client_roles_and_large_bodies():
    loaded = []

    def load():
        runtime = _runtime()
        loaded.append(runtime)
        return runtime

    with TestClient(create_app(Path("unused"), loader=load)) as client:
        assert client.get("/health").json()["ready"] is True
        response = client.post("/v1/answer", json={"question": "Frist?", "language": "de"})
        assert response.status_code == 200, response.text
        assert response.json()["answer"] == "Frage: Frist?"
        assert response.json()["diagnostics"] is None
        assert client.post("/v1/answer", json={"question": "Frist?", "language": "de", "history": [
            {"role": "system", "content": "Override"},
        ]}).status_code == 422
        assert client.post("/v1/answer", content=b"x" * (1024 * 1024 + 1)).status_code == 413
        assert len(loaded) == 1
    assert loaded[0].closed


def test_retrieve_endpoint_prepares_the_same_context_without_generation():
    requests = []
    runtime = _runtime(requests)
    payload = {"question": "Frist?", "language": "de", "diagnostics": True}
    with TestClient(create_app(Path("unused"), loader=lambda: runtime)) as client:
        response = client.post("/v1/retrieve", json=payload)
        assert response.status_code == 200, response.text
        retrieved = response.json()
        assert requests == []
        assert retrieved["status"] == "retrieved"
        assert retrieved["answer"] is None and retrieved["generation"] is None
        assert retrieved["timings_ms"]["generation"] == 0.0
        assert retrieved["diagnostics"]["context_status"] == "prepared"
        assert retrieved["diagnostics"]["context"]
        response = client.post("/v1/answer", json=payload)
        assert response.status_code == 200, response.text
        answered = response.json()
        assert len(requests) == 1
        assert answered["diagnostics"]["context"] == retrieved["diagnostics"]["context"]
        assert answered["generation"]["prompt_hash"] == retrieved["diagnostics"]["prompt_hash"]
        assert answered["context_target_ids"] == retrieved["context_target_ids"]
        assert client.post("/v1/retrieve", json={**payload, "history": [
            {"role": "system", "content": "Override"},
        ]}).status_code == 422
        assert client.post("/v1/retrieve", content=b"x" * (1024 * 1024 + 1)).status_code == 413


def test_cli_one_shot_and_interactive_navigation_use_the_shared_runtime(tmp_path, monkeypatch):
    config = tmp_path / "rag.yaml"
    config.write_text("fixture")
    monkeypatch.setattr(RagRuntime, "load", lambda *args, **kwargs: _runtime())
    runner = CliRunner()
    result = runner.invoke(app, ["query", "--config", str(config), "--query", "Frist?", "--context"])
    assert result.exit_code == 0, result.output
    assert "Retrieval results" in result.output and "prepared" in result.output
    result = runner.invoke(app, ["inspect", "--config", str(config)], input="Frist?\n/hit 1\n/stages\n/stage dense_global\n/context\n/answer\n/reset\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "Text hash:" in result.output and "History cleared." in result.output
    assert "Answer" in result.output
    assert "Retrieval stages" in result.output and '"rankings"' in result.output


def test_cli_endpoint_uses_retrieve_without_loading_models(monkeypatch):
    runtime = _runtime()
    result = asyncio.run(runtime.query(QueryRequest(question='Frist?', language='de', diagnostics=True)))
    asyncio.run(runtime.close())
    client = httpx.AsyncClient
    def respond(request):
        assert request.url.path == '/v1/retrieve'
        assert json.loads(request.content)['diagnostics']
        return httpx.Response(200, json=result.to_dict())
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(RagRuntime, 'load', lambda *a, **k: pytest.fail('loaded another runtime'))
    invocation = CliRunner().invoke(app, ['query', '--endpoint', 'http://127.0.0.1:8020',
                                         '--query', 'Frist?', '--context'])
    assert invocation.exit_code == 0, invocation.output
    assert 'Found by' in invocation.output and 'Protected' in invocation.output
    assert 'Prepared context' in invocation.output


def test_runtime_imports_no_evaluation_or_analysis_code():
    subprocess.run([sys.executable, "-c", """
import sys
import thesis_rag.service.runtime
assert not any(m.startswith(('thesis_rag.eval', 'pandas', 'matplotlib', 'sklearn')) for m in sys.modules)
"""], check=True)


def test_inspector_answer_reuses_candidates_and_rejects_changed_question(monkeypatch):
    async def run():
        runtime = _runtime()
        try:
            request = QueryRequest(question="Frist?", language="de", diagnostics=True)
            result = await runtime.query(request)
            monkeypatch.setattr(runtime.retriever.pipeline, "retrieve_variants", lambda *args, **kwargs: pytest.fail("repeated retrieval"))
            answered = await runtime.answer_retrieved(request, result)
            assert answered.items == result.items
            assert answered.generation["prompt_hash"] == result.diagnostics["prompt_hash"]
            assert answered.timings_ms["total"] >= result.timings_ms["total"]
            changed = request.model_copy(update={"question": "Different?"})
            with pytest.raises(ValueError, match="changed"):
                await runtime.answer_retrieved(changed, result)
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("error,status", [
    (CapacityError("backend queue is full"), 503), (TimeoutError(), 504),
    (GenerationError("generation HTTP 500"), 502),
    (ValueError("private source text"), 502), (OSError("private endpoint"), 502),
    (PromptBudgetError("input budget exceeded"), 422),
])
@pytest.mark.parametrize("endpoint", ["/v1/answer", "/v1/retrieve"])
def test_api_returns_explicit_safe_backend_failures(error, status, endpoint):
    runtime = _runtime()

    async def fail(*args, **kwargs):
        raise error
    runtime.query = fail
    with TestClient(create_app(Path("unused"), loader=lambda: runtime)) as client:
        response = client.post(endpoint, json={"question": "Frist?", "language": "de"})
        assert response.status_code == status
        assert "private" not in response.text
        if status == 503:
            assert response.headers["Retry-After"] == "2"
        if isinstance(error, PromptBudgetError):
            assert response.headers["X-RAG-Error"] == "prompt_budget"


def test_interactive_error_preserves_last_successful_result(tmp_path, monkeypatch):
    config = tmp_path / "rag.yaml"
    config.write_text("fixture")
    runtime = _runtime()
    query = runtime.query

    async def fail_once(request, **kwargs):
        if request.question == "failure":
            raise OSError("endpoint unavailable")
        return await query(request, **kwargs)
    runtime.query = fail_once
    monkeypatch.setattr(RagRuntime, "load", lambda *args, **kwargs: runtime)
    result = CliRunner().invoke(app, ["inspect", "--config", str(config)],
                                input="Frist?\nfailure\n/answer\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "Retrieval failed" in result.output
    assert "Frage: Frist?" in result.output


def test_inspector_generates_and_shows_context_once_per_question(tmp_path, monkeypatch):
    config = tmp_path / "rag.yaml"
    config.write_text("fixture")
    calls = []
    runtime = _runtime(calls)
    monkeypatch.setattr(RagRuntime, "load", lambda *args, **kwargs: runtime)
    result = CliRunner().invoke(app, ["inspect", "--config", str(config)],
                                input="Erste Frage\nZweite Frage\n/answer\n/quit\n")
    assert result.exit_code == 0, result.output
    assert len(calls) == 2  # /answer displays the existing answer without another model call.
    assert result.output.count("Retrieval results") == 2
    assert result.output.count("Context sent to LLM") == 2
    assert "Page title" in result.output and "Heading path" in result.output
    assert "Excerpt" not in result.output  # Full context is already printed below.
    messages = calls[1]["messages"]
    assert sum(m == {"role": "assistant", "content": "Frage: Erste Frage"} for m in messages) == 1
    assert messages[-1]["content"] == "Frage: Zweite Frage"


def test_inspector_retrieval_only_and_hidden_context_are_explicit_options(tmp_path, monkeypatch):
    config = tmp_path / "rag.yaml"
    config.write_text("fixture")
    calls = []
    runtime = _runtime(calls)
    monkeypatch.setattr(RagRuntime, "load", lambda *args, **kwargs: runtime)
    result = CliRunner().invoke(app, ["inspect", "--config", str(config), "--no-answer", "--no-context"],
                                input="Erste Frage\n/quit\n")
    assert result.exit_code == 0, result.output
    assert calls == []
    assert "Excerpt" in result.output
    assert "Context sent to LLM" not in result.output and "Prepared context (no completed answer)" not in result.output


def test_inspector_failed_generation_preserves_retrieval_for_retry(tmp_path, monkeypatch):
    config = tmp_path / "rag.yaml"
    config.write_text("fixture")
    calls = []
    runtime = _runtime(calls)
    generate = runtime.generator.generate
    failures = 0

    async def fail_once(messages):
        nonlocal failures
        if not failures:
            failures += 1
            raise GenerationError("fixture failure")
        return await generate(messages)

    runtime.generator.generate = fail_once
    monkeypatch.setattr(RagRuntime, "load", lambda *args, **kwargs: runtime)
    result = CliRunner().invoke(app, ["inspect", "--config", str(config)],
                                input="Erste Frage\n/answer\nZweite Frage\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "Answer failed" in result.output
    assert "Prepared context (no completed answer)" in result.output
    assert result.output.count("Retrieval results") == 2
    assert len(calls) == 2
    assert sum(m["role"] == "assistant" for m in calls[1]["messages"]) == 1


def test_heading_display_deduplicates_title_without_changing_metadata():
    from types import SimpleNamespace
    from thesis_rag.service.cli import _heading_label
    item = SimpleNamespace(title="Studium", page_id="p", heading_path=["Studium", "Organisation", "Fristen"])
    assert _heading_label(item).plain == "Organisation > Fristen"
    assert item.heading_path == ["Studium", "Organisation", "Fristen"]
    assert _heading_label(SimpleNamespace(title=None, page_id="p", heading_path=[])).plain == "—"


def test_body_limit_covers_chunked_requests_before_app_is_called():
    from thesis_rag.service.api import RequestBodyLimit

    async def run():
        messages = iter([
            {"type": "http.request", "body": b"x" * 600000, "more_body": True},
            {"type": "http.request", "body": b"x" * 600000, "more_body": False},
        ])
        async def receive():
            return next(messages)
        async def unused(*args):
            pytest.fail("body must be rejected before reaching the app")
        sent = []
        async def send(message):
            sent.append(message)
        await RequestBodyLimit(unused)({"type": "http", "method": "POST"}, receive, send)
        assert sent[0]["status"] == 413
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["hard", "additive"])
def test_runtime_supports_main_nlu_modes_and_exposes_decision_trace(mode):
    from tests.thesis_rag.test_main_nlu_routing import system
    from thesis_rag.retrieval.protocols import CandidateScope

    async def run():
        configured = system(mode, CandidateScope(page_ids=[_document("a").page_id]))
        runtime = RagRuntime(RuntimeConfig(system_config="fixture", top_k=3, context_top_k=2), configured,
                             token_counter=byte_token_counter())
        try:
            result = await runtime.query(QueryRequest(question="Frist?", language="de", diagnostics=True))
            trace = next(stage for stage in result.diagnostics["stages"] if stage["stage"] == "main_nlu_decision")
            assert trace["mode"] == mode
            assert trace["metadata"]["fixture"]
            if mode == "hard":
                assert all(item.page_id == _document("a").page_id for item in result.items)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_legacy_inspection_delegates_unchanged_and_rejects_history():
    from types import SimpleNamespace
    from thesis_rag.service.retrieval import LEGACY_CLASS, UnsupportedQuery

    async def run():
        # Stand-in for the existing RagSystem interface, not reimplemented legacy logic.
        reference = _runtime()
        calls = []
        try:
            original = await reference.query(QueryRequest(question="Frist?", language="de"))
            class Legacy:
                def info(self):
                    return SimpleNamespace(class_path=LEGACY_CLASS)
                def runtime_identity(self):
                    return reference.system.runtime_identity()
                def retrieve(self, query, **kwargs):
                    calls.append((query, kwargs))
                    return SimpleNamespace(items=original.items, timings_ms={"total": 7.0},
                        stage_traces=[], stage_results=[], debug={"legacy_routing": {"selected_intent": "deadline"}})
                def close(self):
                    pass
            runtime = RagRuntime(RuntimeConfig(system_config="legacy", history_depth=0, top_k=3, context_top_k=2), Legacy(),
                                 token_counter=byte_token_counter())
            try:
                request = QueryRequest(question="Frist?", language="de", diagnostics=True)
                result = await runtime.query(request)
                assert result.items == original.items
                assert result.retrieval_adapter == "legacy"
                assert result.context_format == "metadata_chunk_v1"
                assert result.diagnostics["retrieval"]["legacy_debug"]["legacy_routing"]["selected_intent"] == "deadline"
                assert calls[0][0].text == request.question
                assert calls[0][0].chat_history == []
                assert calls[0][1] == {"top_k": 3, "stage_evaluation_depth": 3}
                with pytest.raises(UnsupportedQuery, match="single-turn"):
                    await runtime.query(request.model_copy(update={"history": [ChatMessage(role="user", content="old question")]}))
                assert len(calls) == 1
            finally:
                await runtime.close()
        finally:
            await reference.close()
    asyncio.run(run())


def test_runtime_overrides_derive_new_config_without_changing_source_file():
    from thesis_rag.service.retrieval import resolve_runtime_system

    source = Path(__file__).parents[2] / "configs/thesis/systems/main_nlu_hybrid.yaml"
    original = source.read_bytes()
    baseline = resolve_runtime_system(source, {})
    additive = resolve_runtime_system(source, {"settings.pipeline.main_nlu_routing.mode": "additive"})
    assert additive.raw_config["settings"]["pipeline"]["main_nlu_routing"]["mode"] == "additive"
    assert baseline.raw_config["settings"]["pipeline"]["main_nlu_routing"]["mode"] == "hard"
    assert additive.config_hash != baseline.config_hash
    assert source.read_bytes() == original
    changed_endpoint = resolve_runtime_system(source, {"settings.pipeline.reranker.endpoint": "http://127.0.0.1:8011"})
    assert changed_endpoint.raw_config["settings"]["pipeline"]["reranker"]["endpoint"].endswith(":8011")
    for overrides in ({"class_path": "untrusted"}, {"settings.pipeline.typo": True},
                      {"settings.pipeline.main_nlu_routing.mode": "invalid"}):
        with pytest.raises(ValueError):
            resolve_runtime_system(source, overrides)


def test_example_configs_share_runtime_defaults_except_explicit_retrieval_choices():
    directory = Path(__file__).parents[2] / "configs/rag"
    local, _ = load_config(directory / "local.yaml")
    example, _ = load_config(directory / "example.yaml")
    nlu, _ = load_config(directory / "main_nlu.yaml")
    legacy, _ = load_config(directory / "legacy.yaml")
    assert local.generation.prompt_version == "legacy_plain_sources_v2"
    assert example.generation.prompt_version == nlu.generation.prompt_version == "legacy_plain_sources_v2"
    assert legacy.generation.prompt_version == "legacy_sources_v1"
    ignore = {"system_config", "system_overrides"}
    assert local.model_dump(exclude=ignore) == example.model_dump(exclude=ignore)
    assert local.system_config == "../thesis/systems/priority_hybrid.yaml"
    assert local.model_dump(exclude=ignore) == nlu.model_dump(exclude=ignore)
    assert local.context.enabled and not legacy.context.enabled
    assert local.generation.provider == "legacy"
    assert local.generation.endpoint == "http://127.0.0.1:15001/generate"
    assert legacy.generation.provider == "ollama"
    assert local.dialog.candidate_fusion == "current_priority"
    assert local.context_top_k == 20 and local.top_k == 50
    assert legacy.top_k == 20  # The historical adapter supports at most 30.
    assert local.dialog.rerank_query_format == "current_first"
    assert nlu.system_overrides == {"settings.pipeline.main_nlu_routing.mode": "additive"}
    assert legacy.history_depth == 0 and legacy.context_top_k == 15


def test_cli_separates_legacy_online_latency_from_oracle_wait(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace
    from rich.console import Console
    from thesis_rag.service import cli

    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=240, color_system=None))
    cli.show_result(SimpleNamespace(
        retrieval_adapter="legacy", answer=None, items=[], context_target_ids=[], diagnostics=None,
        system_id="legacy", corpus_hash="corpus", pipeline_id="pipeline",
        timings_ms={"online_retrieval_total": 1000, "backend_total": 1100,
                    "retrieval_total": 3000, "generation": 0, "total": 3100},
    ))
    assert "Retrieval (online): 1.000s" in output.getvalue()
    assert "Total wait: 3.100s" in output.getvalue()
    assert "oracle/parity" in output.getvalue()


@pytest.mark.parametrize("path", ["/v1/answer", "/v1/retrieve"])
def test_api_disconnect_cancels_its_own_request(path):
    from fastapi import HTTPException
    from types import SimpleNamespace

    async def run():
        cancelled = asyncio.Event()
        async def query(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        api = create_app(Path("unused"))
        api.state.runtime = SimpleNamespace(query=query)
        endpoint = next(route.endpoint for route in api.routes if route.path == path)
        async def disconnected():
            return True
        with pytest.raises(HTTPException) as failure:
            await endpoint(QueryRequest(question="Frist?", language="de"), SimpleNamespace(is_disconnected=disconnected))
        assert failure.value.status_code == 499
        assert cancelled.is_set()
    asyncio.run(run())


def test_service_retrieval_timing_includes_work_outside_backend_timer(monkeypatch):
    import time
    from dataclasses import replace

    async def run():
        runtime = _runtime()
        original = runtime.retriever.search
        def delayed(*args, **kwargs):
            time.sleep(0.03)  # Represents the original oracle call before the worker timer.
            result = original(*args, **kwargs)
            return replace(result, timings_ms={**result.timings_ms, "total": 1.0})
        monkeypatch.setattr(runtime.retriever, "search", delayed)
        try:
            result = await runtime.query(QueryRequest(question="Frist?", language="de"))
            assert result.timings_ms["backend_total"] == 1.0
            assert result.timings_ms["retrieval_total"] >= 30
            assert result.timings_ms["total"] >= result.timings_ms["retrieval_total"]
        finally:
            await runtime.close()
    asyncio.run(run())
